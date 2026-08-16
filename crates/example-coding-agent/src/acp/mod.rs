//! The ACP server: JSON-RPC over stdio, backed by the event bus.
//!
//! The bus *is* the protocol. Everything the editor sees is derived from bus
//! events by the bridge below:
//!
//! | Bus event | ACP `session/update` |
//! |---|---|
//! | `assistant.delta` | `agent_message_chunk` / `agent_thought_chunk` |
//! | `tool.call.proposed` | `tool_call` (pending) |
//! | `tool.result` | `tool_call_update` (+ diff card) |
//! | `permission.request` | `session/request_permission` → `permission.decision` |
//!
//! Because permission requests already travel over the bus, the editor is
//! just another approver — the same path a TUI or Slack bot would use.

pub mod wire;

use crate::agent::CodingSession;
use crate::config::{ModelConfig, PermissionMode, SessionConfig};
use anyhow::Result;
use eventage::event::kinds;
use eventage::{Event, EventBus};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{oneshot, Mutex};
use tracing::{debug, info, warn};
use wire::*;

/// Writes framed JSON-RPC messages to stdout and correlates responses to
/// requests we sent the client.
pub struct Peer {
    stdout: Mutex<tokio::io::Stdout>,
    next_id: AtomicI64,
    pending: Mutex<HashMap<i64, oneshot::Sender<Value>>>,
}

impl Peer {
    pub fn new() -> Self {
        Self {
            stdout: Mutex::new(tokio::io::stdout()),
            next_id: AtomicI64::new(1),
            pending: Mutex::new(HashMap::new()),
        }
    }

    async fn write(&self, value: &Value) -> Result<()> {
        let mut line = serde_json::to_string(value)?;
        line.push('\n');
        let mut out = self.stdout.lock().await;
        out.write_all(line.as_bytes()).await?;
        out.flush().await?;
        Ok(())
    }

    /// Fire-and-forget notification to the client.
    pub async fn notify(&self, method: &'static str, params: Value) {
        let note = RpcNotification::new(method, params);
        if let Ok(value) = serde_json::to_value(note) {
            if let Err(e) = self.write(&value).await {
                warn!("failed to notify client: {e}");
            }
        }
    }

    /// Send a request to the client and await its response.
    pub async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);
        self.write(&json!({
            "jsonrpc": "2.0", "id": id, "method": method, "params": params
        }))
        .await?;
        Ok(rx.await?)
    }

    /// Route an incoming response to whoever is waiting for it.
    async fn resolve(&self, id: i64, result: Value) {
        if let Some(tx) = self.pending.lock().await.remove(&id) {
            let _ = tx.send(result);
        }
    }

    /// Push one `session/update` notification.
    pub async fn update(&self, session_id: &str, update: SessionUpdate) {
        let params = SessionNotification {
            session_id: session_id.to_string(),
            update,
        };
        if let Ok(value) = serde_json::to_value(params) {
            self.notify("session/update", value).await;
        }
    }
}

impl Default for Peer {
    fn default() -> Self {
        Self::new()
    }
}

// ── client-delegated file I/O ─────────────────────────────────────────────────

/// Handle for routing file I/O back through the editor.
///
/// This matters more than it looks: reading through the client returns the
/// **unsaved buffer**, so the agent sees the file as the user currently sees
/// it. Writing through the client keeps the editor's buffer authoritative, so
/// an agent edit cannot be silently clobbered when the user next saves.
///
/// Every method degrades to `None`/`false` when the client did not advertise
/// the capability, and callers fall back to disk.
#[derive(Clone)]
pub struct ClientFs {
    peer: Arc<Peer>,
    session_id: String,
    caps: FsCapabilities,
}

impl ClientFs {
    pub fn new(peer: Arc<Peer>, session_id: String, caps: FsCapabilities) -> Self {
        Self {
            peer,
            session_id,
            caps,
        }
    }

    /// Read via the editor; `None` if unsupported or the request failed.
    pub async fn read(&self, path: &str) -> Option<String> {
        if !self.caps.read_text_file {
            return None;
        }
        let params = serde_json::to_value(ReadTextFileParams {
            session_id: self.session_id.clone(),
            path: path.to_string(),
            line: None,
            limit: None,
        })
        .ok()?;
        let result = self.peer.request("fs/read_text_file", params).await.ok()?;
        // A result carrying an `error` is a failure however it was framed.
        // Deserialising it anyway yields `content: ""`, which the agent reads
        // as an empty file — and an empty file is a thing it will happily
        // overwrite. Older clients answer refusals this way, so the check
        // stays even now that Studio returns proper protocol errors.
        if result.get("error").is_some() {
            warn!("the editor refused a read: {}", result["error"]);
            return None;
        }
        serde_json::from_value::<ReadTextFileResult>(result)
            .ok()
            .map(|r| r.content)
    }

    /// Write via the editor; `false` if unsupported or the request failed.
    pub async fn write(&self, path: &str, content: &str) -> bool {
        if !self.caps.write_text_file {
            return false;
        }
        let Ok(params) = serde_json::to_value(WriteTextFileParams {
            session_id: self.session_id.clone(),
            path: path.to_string(),
            content: content.to_string(),
        }) else {
            return false;
        };
        // `is_ok()` alone was wrong twice over: it accepted a transport
        // success carrying an `error` property, and the caller uses this
        // boolean to decide whether to fall back to writing the file itself.
        // A refused write was therefore reported to the model as done, and
        // the edit went nowhere.
        match self.peer.request("fs/write_text_file", params).await {
            Ok(result) if result.get("error").is_none() => true,
            Ok(result) => {
                warn!("the editor refused a write: {}", result["error"]);
                false
            }
            Err(e) => {
                warn!("the editor could not be asked to write: {e}");
                false
            }
        }
    }
}

// ── bridge ────────────────────────────────────────────────────────────────────

/// Translate one bus event into ACP notifications / client requests.
///
/// Returns `true` when the event ends the turn.
async fn handle_event(peer: &Arc<Peer>, session_id: &str, event: &Event) -> bool {
    match event.kind.as_str() {
        kinds::ASSISTANT_DELTA => {
            if let Some(text) = event.payload.get("content").and_then(|v| v.as_str()) {
                peer.update(
                    session_id,
                    SessionUpdate::AgentMessageChunk {
                        content: ContentBlock::text(text),
                    },
                )
                .await;
            }
            if let Some(text) = event
                .payload
                .get("reasoning_content")
                .and_then(|v| v.as_str())
            {
                peer.update(
                    session_id,
                    SessionUpdate::AgentThoughtChunk {
                        content: ContentBlock::text(text),
                    },
                )
                .await;
            }
        }

        kinds::TOOL_CALL_PROPOSED => {
            let name = event
                .payload
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("tool");
            let id = event
                .payload
                .get("tool_call_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            peer.update(
                session_id,
                SessionUpdate::ToolCall(ToolCallUpdate {
                    tool_call_id: id.to_string(),
                    title: Some(describe_call(name, &event.payload)),
                    kind: Some(ToolKind::for_tool(name)),
                    status: Some(ToolCallStatus::InProgress),
                    raw_input: event
                        .payload
                        .get("arguments")
                        .and_then(|a| a.as_str())
                        .and_then(|s| serde_json::from_str(s).ok()),
                    ..Default::default()
                }),
            )
            .await;
        }

        kinds::TOOL_RESULT => {
            let id = event
                .payload
                .get("tool_call_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let failed = event.payload.get("error").is_some();
            let result = event.payload.get("result");

            // Tools advertise UI intent through `_diff` / `_diffs` /
            // `_locations`. A tool that touches several files at once — a
            // patch, a rename — sets `_diffs` and gets a card per file;
            // `_diff` is the single-file form, and the first entry of
            // `_diffs` duplicates it, so only one of the two is read.
            let mut content = diff_cards(result);
            if let Some(error) = event.payload.get("error").and_then(|e| e.as_str()) {
                content.push(ToolCallContent::Content {
                    content: ContentBlock::text(error),
                });
            }

            let locations = result
                .and_then(|r| r.get("_locations"))
                .and_then(|l| l.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| {
                            Some(ToolCallLocation {
                                path: item.get("path")?.as_str()?.to_string(),
                                line: item.get("line").and_then(|l| l.as_u64()).map(|l| l as u32),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();

            peer.update(
                session_id,
                SessionUpdate::ToolCallUpdate(ToolCallUpdate {
                    tool_call_id: id.to_string(),
                    status: Some(if failed {
                        ToolCallStatus::Failed
                    } else {
                        ToolCallStatus::Completed
                    }),
                    content,
                    locations,
                    ..Default::default()
                }),
            )
            .await;

            // The plan tool drives the editor's task checklist.
            if let Some(entries) = result
                .and_then(|r| r.get("_plan"))
                .and_then(|p| p.as_array())
            {
                let plan: Vec<PlanEntry> = entries
                    .iter()
                    .filter_map(|e| serde_json::from_value(e.clone()).ok())
                    .collect();
                if !plan.is_empty() {
                    peer.update(session_id, SessionUpdate::Plan { entries: plan })
                        .await;
                }
            }
        }

        kinds::PERMISSION_REQUEST => {
            forward_permission(peer, session_id, event).await;
        }

        kinds::AGENT_CYCLE_END => return true,

        _ => {}
    }
    false
}

/// Ask the editor to approve a tool call, then publish the verdict back onto
/// the bus for `PermissionPolicyHook` to consume.
async fn forward_permission(peer: &Arc<Peer>, session_id: &str, event: &Event) {
    let request_id = event
        .payload
        .get("request_id")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let tool = event
        .payload
        .get("tool")
        .and_then(|v| v.as_str())
        .unwrap_or("tool");

    let params = RequestPermissionParams {
        session_id: session_id.to_string(),
        tool_call: ToolCallUpdate {
            tool_call_id: request_id.clone(),
            title: Some(describe_call(tool, &event.payload)),
            kind: Some(ToolKind::for_tool(tool)),
            status: Some(ToolCallStatus::Pending),
            raw_input: event.payload.get("arguments").cloned(),
            ..Default::default()
        },
        options: PermissionOption::standard_set(),
    };

    let value = serde_json::to_value(params).unwrap_or(Value::Null);
    let (approve, reason) = match peer.request("session/request_permission", value).await {
        Ok(response) => match serde_json::from_value::<RequestPermissionResult>(response) {
            Ok(result) => match result.outcome {
                PermissionOutcome::Selected { option_id } => {
                    let allow = option_id.starts_with("allow");
                    (
                        allow,
                        if allow {
                            None
                        } else {
                            Some("the user rejected this action".to_string())
                        },
                    )
                }
                PermissionOutcome::Cancelled => {
                    (false, Some("the user cancelled the request".to_string()))
                }
            },
            Err(e) => (false, Some(format!("malformed permission response: {e}"))),
        },
        Err(e) => (false, Some(format!("permission request failed: {e}"))),
    };

    // NOTE: this must go back on the same bus the hook is waiting on.
    let decision = Event::new(
        kinds::PERMISSION_DECISION,
        json!({ "request_id": request_id, "approve": approve, "reason": reason }),
    );
    if let Some(bus) = CURRENT_BUS.lock().await.get(session_id).cloned() {
        let _ = bus.publish(decision).await;
    }
}

/// Human-readable one-liner for a tool call, used as the card title.
/// Reviewable diff cards for a tool result.
///
/// Tools advertise UI intent by convention. A single-file edit sets `_diff`;
/// a tool that touches several files at once — `apply_patch`, `lsp_rename` —
/// sets `_diffs` and gets one card per file. Its first entry repeats `_diff`
/// for clients that only know the singular form, so exactly one of the two is
/// read here rather than both.
fn diff_cards(result: Option<&Value>) -> Vec<ToolCallContent> {
    let diffs: Vec<&Value> = match result.and_then(|r| r.get("_diffs")) {
        Some(Value::Array(items)) => items.iter().collect(),
        _ => result.and_then(|r| r.get("_diff")).into_iter().collect(),
    };
    diffs
        .into_iter()
        .filter(|diff| diff.is_object())
        .map(|diff| ToolCallContent::Diff {
            path: diff
                .get("path")
                .and_then(|p| p.as_str())
                .unwrap_or_default()
                .to_string(),
            old_text: diff
                .get("old_text")
                .and_then(|t| t.as_str())
                .map(str::to_string),
            new_text: diff
                .get("new_text")
                .and_then(|t| t.as_str())
                .unwrap_or_default()
                .to_string(),
        })
        .collect()
}

fn describe_call(name: &str, payload: &Value) -> String {
    let args: Value = payload
        .get("arguments")
        .and_then(|a| a.as_str())
        .and_then(|s| serde_json::from_str(s).ok())
        .or_else(|| payload.get("arguments").cloned())
        .unwrap_or(Value::Null);

    let detail = args
        .get("path")
        .or_else(|| args.get("pattern"))
        .or_else(|| args.get("command"))
        .or_else(|| args.get("query"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if detail.is_empty() {
        name.to_string()
    } else {
        format!("{name}: {}", detail.chars().take(120).collect::<String>())
    }
}

/// Buses by session id, so permission decisions can be routed back.
static CURRENT_BUS: once_cell_shim::Lazy<Mutex<HashMap<String, EventBus>>> =
    once_cell_shim::Lazy::new(|| Mutex::new(HashMap::new()));

/// Minimal lazy-static shim (avoids pulling in `once_cell`).
mod once_cell_shim {
    use std::sync::OnceLock;

    pub struct Lazy<T> {
        init: fn() -> T,
        cell: OnceLock<T>,
    }

    impl<T> Lazy<T> {
        pub const fn new(init: fn() -> T) -> Self {
            Self {
                init,
                cell: OnceLock::new(),
            }
        }
    }

    impl<T> std::ops::Deref for Lazy<T> {
        type Target = T;
        fn deref(&self) -> &T {
            self.cell.get_or_init(self.init)
        }
    }
}

// ── server ────────────────────────────────────────────────────────────────────

pub struct AcpServer {
    peer: Arc<Peer>,
    sessions: Mutex<HashMap<String, Arc<CodingSession>>>,
    model: ModelConfig,
    client_caps: Mutex<ClientCapabilities>,
}

impl AcpServer {
    pub fn new(model: ModelConfig) -> Self {
        Self {
            peer: Arc::new(Peer::new()),
            sessions: Mutex::new(HashMap::new()),
            model,
            client_caps: Mutex::new(ClientCapabilities::default()),
        }
    }

    /// Read JSON-RPC messages from stdin until EOF, then drain in-flight work.
    ///
    /// Each message is handled concurrently so a long-running prompt cannot
    /// block the cancellation or permission responses queued behind it. On
    /// EOF we wait for outstanding handlers rather than dropping them, so a
    /// client that closes stdin still gets the replies it is owed.
    pub async fn run(self: Arc<Self>) -> Result<()> {
        let mut inflight = tokio::task::JoinSet::new();
        let mut lines = BufReader::new(tokio::io::stdin()).lines();

        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }
            let server = Arc::clone(&self);
            inflight.spawn(async move { server.dispatch(line).await });
            // Reap finished handlers so the set cannot grow unbounded.
            while let Some(res) = inflight.try_join_next() {
                if let Err(e) = res {
                    warn!("request handler panicked: {e}");
                }
            }
        }

        debug!("stdin closed; draining in-flight requests");
        while let Some(res) = inflight.join_next().await {
            if let Err(e) = res {
                warn!("request handler panicked: {e}");
            }
        }
        Ok(())
    }

    async fn dispatch(self: Arc<Self>, line: String) {
        let value: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                // Unparseable frame: we have no id to answer, so report it
                // with a null id as JSON-RPC prescribes rather than going
                // silent on the client.
                warn!("malformed JSON-RPC frame: {e}");
                let response = RpcResponse::err(
                    Value::Null,
                    codes::PARSE_ERROR,
                    format!("invalid JSON: {e}"),
                );
                if let Ok(value) = serde_json::to_value(response) {
                    let _ = self.peer.write(&value).await;
                }
                return;
            }
        };

        // A response to something we asked the client.
        if value.get("method").is_none() {
            if let Some(id) = value.get("id").and_then(|v| v.as_i64()) {
                let result = value.get("result").cloned().unwrap_or(Value::Null);
                self.peer.resolve(id, result).await;
            }
            return;
        }

        // Keep the id (if any) so a malformed request can still be answered.
        let id = value.get("id").cloned();
        let request: RpcRequest = match serde_json::from_value(value) {
            Ok(r) => r,
            Err(e) => {
                warn!("bad request: {e}");
                if let Some(id) = id {
                    let response = RpcResponse::err(
                        id,
                        codes::INVALID_REQUEST,
                        format!("malformed request: {e}"),
                    );
                    if let Ok(value) = serde_json::to_value(response) {
                        let _ = self.peer.write(&value).await;
                    }
                }
                return;
            }
        };
        debug!(method = %request.method, "acp request");

        let id = request.id.clone();
        let outcome = self.handle(&request.method, request.params).await;

        // Notifications get no reply.
        let Some(id) = id else { return };
        let response = match outcome {
            Ok(result) => RpcResponse::ok(id, result),
            Err(e) => {
                let message = e.to_string();
                // Distinguish "I don't implement that" from "it went wrong",
                // so clients can feature-detect optional methods.
                let code = if message.contains("not supported") {
                    codes::METHOD_NOT_FOUND
                } else if message.contains("missing field")
                    || message.contains("invalid type")
                    || message.contains("unknown mode")
                {
                    codes::INVALID_PARAMS
                } else {
                    codes::INTERNAL_ERROR
                };
                RpcResponse::err(id, code, message)
            }
        };
        if let Ok(value) = serde_json::to_value(response) {
            let _ = self.peer.write(&value).await;
        }
    }

    async fn handle(&self, method: &str, params: Value) -> Result<Value> {
        match method {
            "initialize" => {
                let req: InitializeRequest = serde_json::from_value(params).unwrap_or_default();
                let caps = req.client_capabilities.clone();
                // Which editor connected, and what it will do for us, decides
                // whether file I/O is delegated — worth having in the log when
                // diagnosing "why did it write straight to disk?".
                let client = req
                    .client_info
                    .as_ref()
                    .map(|i| format!("{} {}", i.name, i.version.as_deref().unwrap_or("?")))
                    .unwrap_or_else(|| "unknown client".into());
                info!(
                    client = %client,
                    fs_read = caps.fs.read_text_file,
                    fs_write = caps.fs.write_text_file,
                    terminal = caps.terminal,
                    "ACP session initialized"
                );
                *self.client_caps.lock().await = caps;
                Ok(serde_json::to_value(InitializeResponse {
                    protocol_version: negotiate_version(req.protocol_version),
                    agent_capabilities: AgentCapabilities {
                        load_session: true,
                        prompt_capabilities: PromptCapabilities {
                            image: true,
                            audio: false,
                            embedded_context: true,
                        },
                    },
                    auth_methods: vec![],
                    agent_info: Implementation {
                        name: "eventage-code".into(),
                        version: Some(env!("CARGO_PKG_VERSION").into()),
                    },
                })?)
            }

            "authenticate" => Ok(json!({})),

            // Extension: undo whole turns using the DAG. Editors that know
            // about it get `/rewind`; others simply never call it.
            "session/rewind" => {
                let session_id = params
                    .get("sessionId")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("missing sessionId"))?;
                let turns = params.get("turns").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
                let session = self
                    .sessions
                    .lock()
                    .await
                    .get(session_id)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("unknown session"))?;
                let remaining = session.rewind(turns).await?;
                Ok(json!({ "turnsRemaining": remaining }))
            }

            "session/new" => {
                let req: NewSessionRequest = serde_json::from_value(params)?;
                let mut config = SessionConfig::new(req.cwd.clone(), self.model.clone());
                config.mcp_servers = req.mcp_servers.iter().map(to_mcp_config).collect();
                // The id is minted here so the client handle can carry it.
                let session_id = uuid::Uuid::new_v4().to_string();
                let client = ClientFs::new(
                    Arc::clone(&self.peer),
                    session_id.clone(),
                    self.client_caps.lock().await.fs,
                );
                let session = Arc::new(
                    CodingSession::create(session_id.clone(), config, Some(client)).await?,
                );

                CURRENT_BUS
                    .lock()
                    .await
                    .insert(session_id.clone(), session.bus.clone());
                self.sessions
                    .lock()
                    .await
                    .insert(session_id.clone(), Arc::clone(&session));

                Ok(serde_json::to_value(NewSessionResponse {
                    // Read it back from the session: one source of truth.
                    session_id: session.id.clone(),
                    modes: mode_state(PermissionMode::Ask),
                })?)
            }

            "session/load" => {
                let req: LoadSessionRequest = serde_json::from_value(params)?;
                let mut config = SessionConfig::new(req.cwd.clone(), self.model.clone());
                config.mcp_servers = req.mcp_servers.iter().map(to_mcp_config).collect();
                let client = ClientFs::new(
                    Arc::clone(&self.peer),
                    req.session_id.clone(),
                    self.client_caps.lock().await.fs,
                );
                let session =
                    Arc::new(CodingSession::resume(&req.session_id, config, Some(client)).await?);
                CURRENT_BUS
                    .lock()
                    .await
                    .insert(req.session_id.clone(), session.bus.clone());
                self.sessions
                    .lock()
                    .await
                    .insert(req.session_id.clone(), session);
                Ok(json!({}))
            }

            "session/set_mode" => {
                let req: SetModeRequest = serde_json::from_value(params)?;
                let mode = PermissionMode::from_id(&req.mode_id)
                    .ok_or_else(|| anyhow::anyhow!("unknown mode '{}'", req.mode_id))?;
                if let Some(session) = self.sessions.lock().await.get(&req.session_id) {
                    session.set_mode(mode).await;
                }
                self.peer
                    .update(
                        &req.session_id,
                        SessionUpdate::CurrentModeUpdate {
                            current_mode_id: mode.id().to_string(),
                        },
                    )
                    .await;
                Ok(json!({}))
            }

            "session/prompt" => {
                let req: PromptRequest = serde_json::from_value(params)?;
                let session = self
                    .sessions
                    .lock()
                    .await
                    .get(&req.session_id)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("unknown session"))?;
                let stop = self.run_turn(session, req).await?;
                Ok(serde_json::to_value(PromptResponse { stop_reason: stop })?)
            }

            "session/cancel" => {
                let req: CancelNotification = serde_json::from_value(params)?;
                if let Some(session) = self.sessions.lock().await.get(&req.session_id) {
                    session.cancel();
                }
                Ok(json!({}))
            }

            other => Err(anyhow::anyhow!("method '{other}' not supported")),
        }
    }

    /// Run one prompt to completion, streaming updates as bus events arrive.
    async fn run_turn(
        &self,
        session: Arc<CodingSession>,
        req: PromptRequest,
    ) -> Result<StopReason> {
        let peer = Arc::clone(&self.peer);
        let session_id = req.session_id.clone();

        // Checked before the bridge is spawned, so a refusal answers at once
        // rather than after the drain timeout. `prompt_turn` re-checks under
        // the gate and is the one that decides.
        if session.is_busy() {
            anyhow::bail!("this session is already working on something; cancel it first");
        }

        // Subscribe before publishing so no event is missed.
        let mut rx = session.bus.subscribe();
        let bridge = {
            let peer = Arc::clone(&peer);
            let session_id = session_id.clone();
            tokio::spawn(async move {
                while let Some(event) = rx.recv().await {
                    if handle_event(&peer, &session_id, &event).await {
                        break;
                    }
                }
            })
        };

        // Submit and run under one gate: the two apart left a window where a
        // pipelined prompt could join the running conversation and clear its
        // cancellation flag.
        let outcome = session.prompt_turn(&req.prompt).await;

        // Let the bridge drain rather than killing it. Aborting dropped
        // whatever was still queued — the final assistant chunk, a tool
        // completion, the usage record — so a turn could look unfinished in
        // the editor purely because the forwarder was cut off. It ends on its
        // own once it sees the turn-ending event; the timeout is only there
        // so a missing one cannot hang the response.
        match tokio::time::timeout(std::time::Duration::from_secs(5), bridge).await {
            Ok(_) => {}
            Err(_) => warn!("event bridge did not finish draining; some updates may be missing"),
        }

        Ok(match outcome {
            Ok(()) if session.was_cancelled() => StopReason::Cancelled,
            Ok(()) => stop_reason_from_log(&session.bus.log().await),
            Err(e) => {
                warn!("turn failed: {e}");
                peer.update(
                    &session_id,
                    SessionUpdate::AgentMessageChunk {
                        content: ContentBlock::text(format!("\n\n**Error:** {e}")),
                    },
                )
                .await;
                StopReason::Refusal
            }
        })
    }
}

/// Work out why the turn ended, so the editor can say so.
///
/// The bus records both conditions: the strategy marks its wrap-up message
/// when the step budget runs out, and the budget hook publishes an event when
/// tokens are exhausted.
fn stop_reason_from_log(log: &[Event]) -> StopReason {
    if log.iter().any(|e| e.kind == kinds::BUDGET_EXHAUSTED) {
        return StopReason::MaxTokens;
    }
    let hit_step_limit = log
        .iter()
        .rfind(|e| e.kind == kinds::ASSISTANT_MESSAGE)
        .and_then(|e| e.payload.get("finalized_due_to"))
        .and_then(|v| v.as_str())
        == Some("max_steps");
    if hit_step_limit {
        StopReason::MaxTurnRequests
    } else {
        StopReason::EndTurn
    }
}

/// Translate an ACP MCP server spec into our session config.
fn to_mcp_config(spec: &McpServerSpec) -> crate::config::McpServerConfig {
    crate::config::McpServerConfig {
        // Fall back to the command name when the client omits a label.
        name: spec
            .name
            .clone()
            .or_else(|| spec.command.clone())
            .unwrap_or_else(|| "mcp".to_string()),
        command: spec.command.clone(),
        args: spec.args.clone(),
        env: spec
            .env
            .iter()
            .map(|e| (e.name.clone(), e.value.clone()))
            .collect(),
        url: spec.url.clone(),
    }
}

/// The mode picker the editor renders.
fn mode_state(current: PermissionMode) -> SessionModeState {
    SessionModeState {
        available_modes: PermissionMode::ALL
            .into_iter()
            .map(|m| SessionMode {
                id: m.id().to_string(),
                name: m.label().to_string(),
                description: Some(m.description().to_string()),
            })
            .collect(),
        current_mode_id: current.id().to_string(),
    }
}

/// Convert ACP prompt blocks into an eventage user message payload.
pub fn prompt_to_payload(blocks: &[ContentBlock]) -> Value {
    use eventage::llm::ContentPart;

    let parts: Vec<ContentPart> = blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(ContentPart::text(text)),
            ContentBlock::Image { data, mime_type } => {
                Some(ContentPart::image_base64(mime_type, data))
            }
            ContentBlock::Resource { resource } => resource.text.as_ref().map(|t| {
                ContentPart::text(format!("<{}>\n{t}\n</{}>", resource.uri, resource.uri))
            }),
            ContentBlock::ResourceLink { uri, .. } => {
                Some(ContentPart::text(format!("(attached: {uri})")))
            }
            ContentBlock::Audio { .. } => None,
        })
        .collect();

    json!({ "parts": serde_json::to_value(&parts).unwrap_or(Value::Null) })
}

#[cfg(test)]
mod tests {
    use super::diff_cards;

    #[test]
    fn a_multi_file_change_gets_a_card_per_file() {
        // Before this, a rename touching six files showed one diff and the
        // other five landed silently.
        let file =
            |path: &str| serde_json::json!({ "path": path, "old_text": "a", "new_text": "b" });
        let result = serde_json::json!({
            "_diff": file("/one.rs"),
            "_diffs": [file("/one.rs"), file("/two.rs"), file("/three.rs")],
        });
        // Three, not four: `_diff` repeats the first entry and is skipped.
        assert_eq!(diff_cards(Some(&result)).len(), 3);
    }

    #[test]
    fn a_single_file_change_still_gets_its_card() {
        let result = serde_json::json!({
            "_diff": { "path": "/one.rs", "old_text": "a", "new_text": "b" },
        });
        assert_eq!(diff_cards(Some(&result)).len(), 1);
        assert!(diff_cards(Some(&serde_json::json!({}))).is_empty());
        assert!(diff_cards(None).is_empty());
    }

    use super::*;

    #[test]
    fn describes_calls_for_the_card_title() {
        let payload = json!({
            "name": "read_file",
            "arguments": "{\"path\":\"src/main.rs\"}"
        });
        assert_eq!(
            describe_call("read_file", &payload),
            "read_file: src/main.rs"
        );

        let bare = json!({ "name": "plan" });
        assert_eq!(describe_call("plan", &bare), "plan");
    }

    #[test]
    fn prompt_blocks_become_multimodal_parts() {
        let blocks = vec![
            ContentBlock::text("look at this"),
            ContentBlock::Image {
                data: "QUJD".into(),
                mime_type: "image/png".into(),
            },
        ];
        let payload = prompt_to_payload(&blocks);
        let parts = payload["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "image");
    }

    #[test]
    fn mode_state_lists_every_mode() {
        let state = mode_state(PermissionMode::Plan);
        assert_eq!(state.current_mode_id, "plan");
        assert_eq!(state.available_modes.len(), 4);
    }
}
