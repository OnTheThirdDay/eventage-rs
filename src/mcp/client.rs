use super::error::McpError;
use crate::llm::types::{FunctionDefinition, ToolDefinition};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{debug, instrument};

// ── JSON-RPC 2.0 wire types ───────────────────────────────────────────────────

#[derive(Serialize)]
struct JsonRpcRequest<'a> {
    jsonrpc: &'static str,
    id: u64,
    method: &'a str,
    params: Value,
}

#[derive(Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    jsonrpc: String,
    #[allow(dead_code)]
    id: Option<Value>,
    result: Option<Value>,
    error: Option<JsonRpcError>,
}

#[derive(Deserialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

// ── MCP wire types ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ToolsListResult {
    tools: Vec<McpToolDef>,
}

#[derive(Deserialize)]
struct McpToolDef {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
}

#[derive(Deserialize)]
struct ToolsCallResult {
    content: Vec<ContentItem>,
    #[serde(rename = "isError", default)]
    is_error: bool,
}

#[derive(Deserialize)]
struct ContentItem {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

// ── McpClient ─────────────────────────────────────────────────────────────────

/// An HTTP client for the MCP Streamable HTTP transport (JSON-RPC 2.0).
///
/// Use [`McpClient::connect_http`] to establish a session, then use [`McpClient::list_tools`]
/// and [`McpClient::call_tool`] to interact with the server.
#[derive(Clone)]
pub struct McpClient {
    http: Client,
    url: String,
    id_gen: Arc<AtomicU64>,
}

impl McpClient {
    /// Connects to an MCP server at the given HTTP `url`.
    ///
    /// This performs the mandatory `initialize` handshake before returning.
    #[instrument(skip_all, fields(url))]
    pub async fn connect_http(url: impl Into<String>) -> Result<Self, McpError> {
        let url = url.into();
        tracing::Span::current().record("url", &url);
        let http = Client::new();
        let id_gen = Arc::new(AtomicU64::new(1));

        let client = Self { http, url, id_gen };

        // initialize handshake
        let _init_result = client
            .rpc(
                "initialize",
                json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "eventage-mcp",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                }),
            )
            .await?;

        debug!("MCP initialize ok");

        // notifications/initialized is a notification (no id), send fire-and-forget
        client
            .notify("notifications/initialized", json!({}))
            .await?;

        Ok(client)
    }

    /// Lists all tools exposed by the MCP server as [`ToolDefinition`]s.
    pub async fn list_tools(&self) -> Result<Vec<ToolDefinition>, McpError> {
        let result = self.rpc("tools/list", json!({})).await?;
        let list: ToolsListResult = serde_json::from_value(result)?;

        let defs = list
            .tools
            .into_iter()
            .map(|t| ToolDefinition {
                kind: "function".to_string(),
                function: FunctionDefinition {
                    name: t.name,
                    description: t.description.unwrap_or_default(),
                    parameters: t.input_schema,
                },
            })
            .collect();

        Ok(defs)
    }

    /// Calls a tool by name with the given JSON arguments.
    ///
    /// Returns the tool's output as a JSON [`Value`]. Text items are joined,
    /// and non-text items are returned as `{"type": kind}` objects.
    pub async fn call_tool(&self, name: &str, args: Value) -> Result<Value, McpError> {
        let result = self
            .rpc(
                "tools/call",
                json!({
                    "name": name,
                    "arguments": args
                }),
            )
            .await?;

        let call: ToolsCallResult = serde_json::from_value(result)?;

        if call.is_error {
            let msg = call
                .content
                .iter()
                .filter_map(|c| c.text.as_deref())
                .collect::<Vec<_>>()
                .join("\n");
            return Err(McpError::Tool(msg));
        }

        // Collapse content items to a single value.
        let texts: Vec<&str> = call
            .content
            .iter()
            .filter(|c| c.kind == "text")
            .filter_map(|c| c.text.as_deref())
            .collect();

        if texts.is_empty() {
            if call.content.is_empty() {
                return Err(McpError::NoResult);
            }
            // Return non-text content as a JSON array.
            return Ok(json!(call
                .content
                .iter()
                .map(|c| json!({"type": c.kind}))
                .collect::<Vec<_>>()));
        }

        Ok(Value::String(texts.join("\n")))
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    async fn rpc(&self, method: &str, params: Value) -> Result<Value, McpError> {
        let id = self.id_gen.fetch_add(1, Ordering::Relaxed);
        let req = JsonRpcRequest {
            jsonrpc: "2.0",
            id,
            method,
            params,
        };

        debug!(method, id, "MCP → request");
        let resp: JsonRpcResponse = self
            .http
            .post(&self.url)
            .json(&req)
            .send()
            .await?
            .json()
            .await?;

        if let Some(err) = resp.error {
            return Err(McpError::Protocol {
                code: err.code,
                message: err.message,
            });
        }

        resp.result.ok_or(McpError::NoResult)
    }

    /// Sends a JSON-RPC notification.
    async fn notify(&self, method: &str, params: Value) -> Result<(), McpError> {
        #[derive(Serialize)]
        struct Notification<'a> {
            jsonrpc: &'static str,
            method: &'a str,
            params: Value,
        }

        self.http
            .post(&self.url)
            .json(&Notification {
                jsonrpc: "2.0",
                method,
                params,
            })
            .send()
            .await?;

        Ok(())
    }
}
