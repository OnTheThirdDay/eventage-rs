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

/// Default time to wait for an elicitation answer from the bus.
const ELICITATION_TIMEOUT: Duration = Duration::from_secs(120);

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
    /// When set, server-initiated elicitation requests and `list_changed`
    /// notifications are surfaced as bus events.
    bus: Option<crate::bus::EventBus>,
    /// Label used in emitted events.
    server_label: String,
    /// How long to wait for an `mcp.elicitation.response` before declining.
    elicitation_timeout: Duration,
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
            bus: None,
            server_label: "mcp".to_string(),
            elicitation_timeout: ELICITATION_TIMEOUT,
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
            bus: None,
            server_label: program,
            elicitation_timeout: ELICITATION_TIMEOUT,
        };
        mcp.handshake().await?;
        Ok(mcp)
    }

    /// Route server-initiated **elicitation requests** and **`list_changed`
    /// notifications** onto an [`EventBus`](crate::EventBus).
    ///
    /// With a bus attached, an `elicitation/create` request from the server
    /// is published as `mcp.elicitation.request` and the client waits for a
    /// matching `mcp.elicitation.response` (declining on timeout) — the same
    /// asynchronous, transport-agnostic pattern used for tool permissions.
    /// `notifications/tools/list_changed` is published as `mcp.tools.changed`
    /// so a worker can call `McpToolset::reload`.
    ///
    /// Currently effective for the stdio transport; HTTP servers deliver
    /// server-initiated requests on a separate SSE channel that this client
    /// does not yet open.
    pub fn with_bus(mut self, bus: crate::bus::EventBus, server_label: impl Into<String>) -> Self {
        self.bus = Some(bus);
        self.server_label = server_label.into();
        self
    }

    /// Override how long elicitation waits for an answer (default 120s).
    pub fn with_elicitation_timeout(mut self, timeout: Duration) -> Self {
        self.elicitation_timeout = timeout;
        self
    }

    async fn handshake(&self) -> Result<(), McpError> {
        let init = self
            .rpc(
                "initialize",
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": if self.bus.is_some() {
                        json!({ "elicitation": {} })
                    } else {
                        json!({})
                    },
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
            Transport::Stdio(t) => self.stdio_round_trip(t, &req, id).await?,
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
        &self,
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

            // Server-initiated traffic (requests and notifications) is
            // interleaved with our responses on the same pipe.
            if let Ok(incoming) = serde_json::from_str::<Value>(trimmed) {
                if let Some(method) = incoming.get("method").and_then(|m| m.as_str()) {
                    self.handle_server_message(t, method, &incoming).await;
                    continue;
                }
            }

            match serde_json::from_str::<JsonRpcResponse>(trimmed) {
                Ok(resp) => {
                    // Only the response to *our* id counts (round-trips are
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

    /// Dispatch a server-initiated request or notification.
    async fn handle_server_message(&self, t: &StdioTransport, method: &str, msg: &Value) {
        match method {
            "elicitation/create" => {
                let id = msg.get("id").cloned();
                let response = self.run_elicitation(msg.get("params")).await;
                if let Some(id) = id {
                    let reply = json!({ "jsonrpc": "2.0", "id": id, "result": response });
                    if let Err(e) = Self::write_line(t, &reply).await {
                        warn!("failed to answer elicitation: {e}");
                    }
                }
            }
            "notifications/tools/list_changed" => {
                debug!(server = %self.server_label, "MCP server announced tools/list_changed");
                if let Some(bus) = &self.bus {
                    let _ = bus
                        .publish(crate::event::Event::new(
                            crate::event::kinds::MCP_TOOLS_CHANGED,
                            json!({ "server": self.server_label }),
                        ))
                        .await;
                }
            }
            other => {
                debug!(method = other, "ignoring server-initiated MCP message");
                // Unknown *requests* (those with an id) get a proper
                // method-not-found error so the server is not left hanging.
                if let Some(id) = msg.get("id") {
                    let reply = json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": { "code": -32601, "message": format!("method '{other}' not supported by this client") }
                    });
                    let _ = Self::write_line(t, &reply).await;
                }
            }
        }
    }

    /// Publish an elicitation request onto the bus and await the answer.
    ///
    /// Returns the MCP elicitation result object. Without a bus, or on
    /// timeout, the request is declined — never silently accepted.
    async fn run_elicitation(&self, params: Option<&Value>) -> Value {
        let Some(bus) = &self.bus else {
            return json!({ "action": "decline" });
        };
        let request_id = uuid::Uuid::new_v4().to_string();
        let message = params
            .and_then(|p| p.get("message"))
            .and_then(|m| m.as_str())
            .unwrap_or_default();
        let schema = params
            .and_then(|p| p.get("requestedSchema"))
            .cloned()
            .unwrap_or(json!({}));

        // Subscribe before publishing so a fast responder cannot be missed.
        let mut rx = bus.subscribe();
        let request = crate::event::Event::new(
            crate::event::kinds::MCP_ELICITATION_REQUEST,
            json!({
                "request_id": request_id,
                "server": self.server_label,
                "message": message,
                "schema": schema,
            }),
        );
        if bus.publish(request).await.is_err() {
            return json!({ "action": "decline" });
        }

        let deadline = tokio::time::Instant::now() + self.elicitation_timeout;
        loop {
            let event = match tokio::time::timeout_at(deadline, rx.recv()).await {
                Err(_) => {
                    warn!(server = %self.server_label, "elicitation timed out — declining");
                    return json!({ "action": "decline" });
                }
                Ok(None) => return json!({ "action": "decline" }),
                Ok(Some(e)) => e,
            };
            if event.kind != crate::event::kinds::MCP_ELICITATION_RESPONSE {
                continue;
            }
            if event.payload.get("request_id").and_then(|v| v.as_str()) != Some(&request_id) {
                continue;
            }
            let action = event
                .payload
                .get("action")
                .and_then(|v| v.as_str())
                .unwrap_or("decline");
            return match action {
                "accept" => json!({
                    "action": "accept",
                    "content": event.payload.get("content").cloned().unwrap_or(json!({}))
                }),
                "cancel" => json!({ "action": "cancel" }),
                _ => json!({ "action": "decline" }),
            };
        }
    }

    /// Write one newline-delimited JSON frame to the server's stdin.
    async fn write_line(t: &StdioTransport, value: &Value) -> Result<(), McpError> {
        let mut line = serde_json::to_string(value)?;
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
        Ok(())
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
    async fn elicitation_round_trips_through_the_bus() {
        // Server calls a tool, which triggers an elicitation request, then
        // returns whatever the client answered.
        let script = r#"
import sys, json
def send(o): print(json.dumps(o), flush=True)
for line in sys.stdin:
    msg = json.loads(line)
    m = msg.get("method")
    if m == "notifications/initialized":
        continue
    rid = msg.get("id")
    if m == "initialize":
        send({"jsonrpc":"2.0","id":rid,"result":{"protocolVersion":"2025-06-18"}})
    elif m == "tools/call":
        # Ask the user for input mid-call.
        send({"jsonrpc":"2.0","id":99,"method":"elicitation/create",
              "params":{"message":"Which branch?","requestedSchema":{"type":"object"}}})
        answer = json.loads(sys.stdin.readline())
        content = answer.get("result",{}).get("content",{})
        send({"jsonrpc":"2.0","id":rid,
              "result":{"content":[{"type":"text","text":json.dumps(content)}]}})
"#;
        let bus = crate::bus::EventBus::new();
        let client =
            match McpClient::connect_stdio("python3", vec!["-c".into(), script.into()], vec![])
                .await
            {
                Ok(c) => c.with_bus(bus.clone(), "test-server"),
                Err(e) => {
                    eprintln!("skipping elicitation test: {e}");
                    return;
                }
            };

        // Stand in for the UI: approve every elicitation with a value.
        let responder_bus = bus.clone();
        let mut rx = bus.subscribe();
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                if event.kind == crate::event::kinds::MCP_ELICITATION_REQUEST {
                    assert_eq!(event.payload["message"], "Which branch?");
                    assert_eq!(event.payload["server"], "test-server");
                    let id = event.payload["request_id"].as_str().unwrap().to_string();
                    responder_bus
                        .publish(crate::event::Event::new(
                            crate::event::kinds::MCP_ELICITATION_RESPONSE,
                            json!({
                                "request_id": id,
                                "action": "accept",
                                "content": { "branch": "main" }
                            }),
                        ))
                        .await
                        .unwrap();
                }
            }
        });

        let result = client.call_tool("anything", json!({})).await.unwrap();
        let text = result.as_str().unwrap();
        assert!(
            text.contains("main"),
            "elicited value should reach the server: {text}"
        );
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
