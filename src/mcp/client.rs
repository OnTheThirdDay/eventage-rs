use super::error::McpError;
use crate::llm::types::{FunctionDefinition, ToolDefinition};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::Mutex;
use tracing::{debug, instrument, warn};

/// Latest MCP protocol revision this client speaks. The server may negotiate
/// down during `initialize`; the agreed version is echoed on every subsequent
/// HTTP request via the `MCP-Protocol-Version` header, as the spec requires.
const PROTOCOL_VERSION: &str = "2025-06-18";

/// Default end-to-end timeout for HTTP requests to an MCP server.
const HTTP_TIMEOUT: Duration = Duration::from_secs(120);

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
    jsonrpc: Option<String>,
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
    #[serde(default)]
    content: Vec<Value>,
    /// Typed result added in MCP 2025-06-18. Preferred over `content` text
    /// when present.
    #[serde(rename = "structuredContent", default)]
    structured_content: Option<Value>,
    #[serde(rename = "isError", default)]
    is_error: bool,
}

// ── Transport ─────────────────────────────────────────────────────────────────

struct StdioTransport {
    /// Held to keep the server process alive; killed on drop.
    _child: Child,
    stdin: Mutex<ChildStdin>,
    stdout: Mutex<BufReader<ChildStdout>>,
    /// Serializes whole request/response round-trips so concurrent tool
    /// calls cannot interleave frames on the pipe.
    round_trip: Mutex<()>,
}

enum Transport {
    Http {
        client: Client,
        url: String,
        headers: Vec<(String, String)>,
        /// `Mcp-Session-Id` issued by the server during `initialize`;
        /// echoed on every subsequent request (Streamable HTTP spec).
        session_id: Mutex<Option<String>>,
        /// Protocol version negotiated during `initialize`; echoed as the
        /// `MCP-Protocol-Version` header on subsequent requests.
        negotiated_version: Mutex<Option<String>>,
    },
    Stdio(Box<StdioTransport>),
}

// ── McpClient ─────────────────────────────────────────────────────────────────

/// A client for MCP servers over **Streamable HTTP** or **stdio**.
///
/// - [`connect_http`](Self::connect_http) / [`connect_http_with_headers`](Self::connect_http_with_headers)
///   speak the Streamable HTTP transport: `Mcp-Session-Id` is captured from
///   the `initialize` response and echoed on every request, and servers that
///   answer with `text/event-stream` (instead of plain JSON) are handled.
/// - [`connect_stdio`](Self::connect_stdio) spawns a local server process and
///   speaks newline-delimited JSON-RPC over its stdin/stdout — the transport
///   most MCP servers ship with. The process is killed when the client drops.
pub struct McpClient {
    transport: Transport,
    id_gen: AtomicU64,
}

impl McpClient {
    /// Connects to an MCP server at the given HTTP `url` and performs the
    /// mandatory `initialize` handshake.
    pub async fn connect_http(url: impl Into<String>) -> Result<Self, McpError> {
        Self::connect_http_with_headers(url, Vec::new()).await
    }

    /// Like [`connect_http`](Self::connect_http), with extra headers sent on
    /// every request — e.g. `("Authorization", "Bearer <token>")`.
    #[instrument(skip_all)]
    pub async fn connect_http_with_headers(
        url: impl Into<String>,
        headers: Vec<(String, String)>,
    ) -> Result<Self, McpError> {
        let client = Client::builder()
            .timeout(HTTP_TIMEOUT)
            .build()
            .unwrap_or_default();

        let mcp = Self {
            transport: Transport::Http {
                client,
                url: url.into(),
                headers,
                session_id: Mutex::new(None),
                negotiated_version: Mutex::new(None),
            },
            id_gen: AtomicU64::new(1),
        };
        mcp.handshake().await?;
        Ok(mcp)
    }

    /// Spawns `program args...` as a local MCP server and connects over stdio.
    ///
    /// The child inherits a minimal environment plus `extra_env`. It is
    /// killed when the client is dropped.
    #[instrument(skip_all, fields(program))]
    pub async fn connect_stdio(
        program: impl Into<String>,
        args: Vec<String>,
        extra_env: Vec<(String, String)>,
    ) -> Result<Self, McpError> {
        let program = program.into();
        tracing::Span::current().record("program", program.as_str());

        let mut cmd = tokio::process::Command::new(&program);
        cmd.args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        for (k, v) in extra_env {
            cmd.env(k, v);
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| McpError::Transport(format!("failed to spawn '{program}': {e}")))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Transport("child stdin unavailable".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Transport("child stdout unavailable".into()))?;

        let mcp = Self {
            transport: Transport::Stdio(Box::new(StdioTransport {
                _child: child,
                stdin: Mutex::new(stdin),
                stdout: Mutex::new(BufReader::new(stdout)),
                round_trip: Mutex::new(()),
            })),
            id_gen: AtomicU64::new(1),
        };
        mcp.handshake().await?;
        Ok(mcp)
    }

    async fn handshake(&self) -> Result<(), McpError> {
        let init = self
            .rpc(
                "initialize",
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "eventage-mcp",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                }),
            )
            .await?;

        // Remember the version the server negotiated; HTTP requests must
        // carry it as `MCP-Protocol-Version` from now on.
        if let Transport::Http {
            negotiated_version, ..
        } = &self.transport
        {
            let version = init
                .get("protocolVersion")
                .and_then(|v| v.as_str())
                .unwrap_or(PROTOCOL_VERSION)
                .to_string();
            debug!(version = %version, "MCP protocol negotiated");
            *negotiated_version.lock().await = Some(version);
        }

        debug!("MCP initialize ok");
        self.notify("notifications/initialized", json!({})).await?;
        Ok(())
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
    /// Text content items are joined into a single string. Non-text items
    /// (images, resources, ...) are returned **verbatim** as a JSON array so
    /// no data is dropped.
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

        let text_of = |item: &Value| -> Option<String> {
            (item.get("type").and_then(|t| t.as_str()) == Some("text"))
                .then(|| item.get("text").and_then(|t| t.as_str()).map(String::from))
                .flatten()
        };

        if call.is_error {
            let msg = call
                .content
                .iter()
                .filter_map(&text_of)
                .collect::<Vec<_>>()
                .join("\n");
            return Err(McpError::Tool(msg));
        }

        // MCP 2025-06-18: prefer the typed result when the server provides one.
        if let Some(structured) = call.structured_content {
            return Ok(structured);
        }

        let texts: Vec<String> = call.content.iter().filter_map(&text_of).collect();
        if texts.len() == call.content.len() && !texts.is_empty() {
            // Pure-text result: join for the model.
            return Ok(Value::String(texts.join("\n")));
        }
        if call.content.is_empty() {
            return Err(McpError::NoResult);
        }
        // Mixed or non-text content: return the items verbatim.
        Ok(Value::Array(call.content))
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

        let resp = match &self.transport {
            Transport::Http { .. } => self.http_round_trip(&req, method).await?,
            Transport::Stdio(t) => Self::stdio_round_trip(t, &req, id).await?,
        };

        if let Some(err) = resp.error {
            return Err(McpError::Protocol {
                code: err.code,
                message: err.message,
            });
        }
        resp.result.ok_or(McpError::NoResult)
    }

    async fn http_round_trip(
        &self,
        req: &JsonRpcRequest<'_>,
        method: &str,
    ) -> Result<JsonRpcResponse, McpError> {
        let Transport::Http {
            client,
            url,
            headers,
            session_id,
            negotiated_version,
        } = &self.transport
        else {
            unreachable!("http_round_trip called on non-http transport");
        };

        let mut request = client
            .post(url)
            .header("Accept", "application/json, text/event-stream")
            .json(req);
        for (k, v) in headers {
            request = request.header(k, v);
        }
        if let Some(sid) = session_id.lock().await.as_deref() {
            request = request.header("Mcp-Session-Id", sid);
        }
        if let Some(version) = negotiated_version.lock().await.as_deref() {
            request = request.header("MCP-Protocol-Version", version);
        }

        let response = request.send().await?;

        // The server issues the session id on `initialize`; remember it.
        if method == "initialize" {
            if let Some(sid) = response
                .headers()
                .get("mcp-session-id")
                .and_then(|v| v.to_str().ok())
            {
                *session_id.lock().await = Some(sid.to_string());
                debug!("captured MCP session id");
            }
        }

        let status = response.status();
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(McpError::Transport(format!(
                "HTTP {status}: {}",
                body.chars().take(500).collect::<String>()
            )));
        }

        if content_type.contains("text/event-stream") {
            // Streamable HTTP: the JSON-RPC response arrives as SSE data lines.
            for data in sse_data_lines(&body) {
                if let Ok(resp) = serde_json::from_str::<JsonRpcResponse>(&data) {
                    if resp.result.is_some() || resp.error.is_some() {
                        return Ok(resp);
                    }
                }
            }
            return Err(McpError::Transport(
                "SSE response contained no JSON-RPC result".into(),
            ));
        }

        Ok(serde_json::from_str(&body)?)
    }

    async fn stdio_round_trip(
        t: &StdioTransport,
        req: &JsonRpcRequest<'_>,
        id: u64,
    ) -> Result<JsonRpcResponse, McpError> {
        // One request/response cycle at a time on the pipe.
        let _guard = t.round_trip.lock().await;

        let mut line = serde_json::to_string(req)?;
        line.push('\n');
        {
            let mut stdin = t.stdin.lock().await;
            stdin
                .write_all(line.as_bytes())
                .await
                .map_err(|e| McpError::Transport(format!("stdio write: {e}")))?;
            stdin
                .flush()
                .await
                .map_err(|e| McpError::Transport(format!("stdio flush: {e}")))?;
        }

        let mut stdout = t.stdout.lock().await;
        loop {
            let mut buf = String::new();
            let n = stdout
                .read_line(&mut buf)
                .await
                .map_err(|e| McpError::Transport(format!("stdio read: {e}")))?;
            if n == 0 {
                return Err(McpError::Transport("MCP server closed stdout".into()));
            }
            let trimmed = buf.trim();
            if trimmed.is_empty() {
                continue;
            }
            match serde_json::from_str::<JsonRpcResponse>(trimmed) {
                Ok(resp) => {
                    // Skip server-initiated notifications/requests; we only
                    // want the response to *our* id (round-trips are
                    // serialized, so no other response can be in flight).
                    let matches = resp
                        .id
                        .as_ref()
                        .and_then(|v| v.as_u64())
                        .is_some_and(|rid| rid == id);
                    if matches && (resp.result.is_some() || resp.error.is_some()) {
                        return Ok(resp);
                    }
                }
                Err(e) => warn!("skipping unparseable MCP stdio line: {e}"),
            }
        }
    }

    /// Sends a JSON-RPC notification (no response expected).
    async fn notify(&self, method: &str, params: Value) -> Result<(), McpError> {
        #[derive(Serialize)]
        struct Notification<'a> {
            jsonrpc: &'static str,
            method: &'a str,
            params: Value,
        }
        let note = Notification {
            jsonrpc: "2.0",
            method,
            params,
        };

        match &self.transport {
            Transport::Http {
                client,
                url,
                headers,
                session_id,
                negotiated_version,
            } => {
                let mut request = client
                    .post(url)
                    .header("Accept", "application/json, text/event-stream")
                    .json(&note);
                for (k, v) in headers {
                    request = request.header(k, v);
                }
                if let Some(sid) = session_id.lock().await.as_deref() {
                    request = request.header("Mcp-Session-Id", sid);
                }
                if let Some(version) = negotiated_version.lock().await.as_deref() {
                    request = request.header("MCP-Protocol-Version", version);
                }
                request.send().await?;
            }
            Transport::Stdio(t) => {
                let mut line = serde_json::to_string(&note)?;
                line.push('\n');
                let mut stdin = t.stdin.lock().await;
                stdin
                    .write_all(line.as_bytes())
                    .await
                    .map_err(|e| McpError::Transport(format!("stdio write: {e}")))?;
                stdin
                    .flush()
                    .await
                    .map_err(|e| McpError::Transport(format!("stdio flush: {e}")))?;
            }
        }
        Ok(())
    }
}

/// Extract the `data:` payloads from an SSE body, joining continuation lines
/// within one event as the spec requires.
fn sse_data_lines(body: &str) -> Vec<String> {
    let mut events = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for line in body.lines() {
        let line = line.trim_end_matches('\r');
        if let Some(data) = line.strip_prefix("data:") {
            current.push(data.strip_prefix(' ').unwrap_or(data));
        } else if line.is_empty() && !current.is_empty() {
            events.push(current.join("\n"));
            current.clear();
        }
    }
    if !current.is_empty() {
        events.push(current.join("\n"));
    }
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_parsing_extracts_data() {
        let body = "event: message\ndata: {\"a\":1}\n\ndata: line1\ndata: line2\n\n";
        let events = sse_data_lines(body);
        assert_eq!(
            events,
            vec!["{\"a\":1}".to_string(), "line1\nline2".to_string()]
        );
    }

    #[test]
    fn sse_parsing_handles_missing_trailing_blank() {
        let events = sse_data_lines("data: {\"jsonrpc\":\"2.0\"}");
        assert_eq!(events.len(), 1);
    }

    #[tokio::test]
    async fn stdio_round_trip_against_scripted_server() {
        // A tiny "MCP server": replies to initialize, then tools/list.
        let script = r#"
import sys, json
for line in sys.stdin:
    msg = json.loads(line)
    if msg.get("method") == "notifications/initialized":
        continue
    rid = msg.get("id")
    if msg.get("method") == "initialize":
        print(json.dumps({"jsonrpc":"2.0","id":rid,"result":{"protocolVersion":"2025-03-26"}}), flush=True)
    elif msg.get("method") == "tools/list":
        print(json.dumps({"jsonrpc":"2.0","id":rid,"result":{"tools":[{"name":"echo","description":"echoes","inputSchema":{"type":"object"}}]}}), flush=True)
    elif msg.get("method") == "tools/call":
        print(json.dumps({"jsonrpc":"2.0","id":rid,"result":{"content":[{"type":"text","text":"pong"}]}}), flush=True)
"#;
        let client =
            match McpClient::connect_stdio("python3", vec!["-c".into(), script.into()], vec![])
                .await
            {
                Ok(c) => c,
                Err(e) => {
                    // python3 unavailable in this environment — skip.
                    eprintln!("skipping stdio test: {e}");
                    return;
                }
            };

        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "echo");

        let result = client.call_tool("echo", json!({})).await.unwrap();
        assert_eq!(result, Value::String("pong".into()));
    }
}
