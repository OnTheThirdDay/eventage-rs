//! The Agent Client Protocol backend: Studio as an ACP *client*.
//!
//! Here Studio drives a separate agent process over JSON-RPC on stdio, the
//! same way an editor does. That makes the app usable with any ACP-capable
//! agent, at a cost worth stating plainly: the protocol carries a rendered
//! view of a turn, not the agent's event log, so the trace panel shows
//! protocol traffic rather than hook decisions and token accounting. The UI
//! learns which it is from `full_trace` in [`AppInfo`] and says so.
//!
//! Every session gets its own child process. Sharing one process across
//! sessions would be closer to the protocol's intent, but a crash would then
//! take every open conversation with it.

use crate::backend::{Backend, Session};
use crate::feed::EventFeed;
use crate::protocol::{
    studio_kinds, AppInfo, ModeInfo, NewSessionRequest, PermissionResponse, PromptBlock,
    SessionInfo, StoredSession, StudioEvent, SummaryOverride,
};
use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use eventage::event::kinds;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin};
use tokio::sync::{oneshot, Mutex};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

/// The protocol version Studio speaks.
const PROTOCOL_VERSION: u32 = 1;

// ── Backend ───────────────────────────────────────────────────────────────────

pub struct AcpBackend {
    program: String,
    args: Vec<String>,
    default_cwd: String,
}

impl AcpBackend {
    /// `command` is the agent to launch, e.g. `["eventage-code"]`.
    pub fn new(command: Vec<String>, default_cwd: String) -> Result<Self> {
        let mut it = command.into_iter();
        let program = it.next().context("--acp needs a command to run")?;
        Ok(Self {
            program: resolve_program(&program)?,
            args: it.collect(),
            default_cwd,
        })
    }
}

/// Pin down which binary `--acp` meant, before anything changes directory.
///
/// Each session's child runs with the *workspace* as its working directory,
/// so a relative path like `./target/debug/eventage-code` would be looked up
/// in the wrong place and fail at session time with nothing useful to say.
/// Resolving it here against the directory Studio was launched from means a
/// bad path is reported at startup instead. A bare name is left alone for the
/// usual `PATH` lookup.
fn resolve_program(program: &str) -> Result<String> {
    if !program.contains(std::path::MAIN_SEPARATOR) && !program.contains('/') {
        return Ok(program.to_string());
    }
    let path = std::fs::canonicalize(program)
        .with_context(|| format!("no such ACP agent: '{program}'"))?;
    Ok(path.display().to_string())
}

#[async_trait]
impl Backend for AcpBackend {
    fn info(&self) -> AppInfo {
        AppInfo {
            backend: "acp",
            backend_detail: format!("{} {}", self.program, self.args.join(" "))
                .trim()
                .to_string(),
            model: "chosen by the agent".into(),
            provider: "ACP".into(),
            default_cwd: self.default_cwd.clone(),
            // The agent reports its own modes on session/new; these are the
            // fallback for agents that report none.
            modes: vec![ModeInfo {
                id: "default".into(),
                label: "Default".into(),
                description: "Whatever the connected agent does by default".into(),
            }],
            version: env!("CARGO_PKG_VERSION"),
            full_trace: false,
            // The connected agent owns its own credentials.
            credentials_hint: None,
        }
    }

    async fn open(&self, req: NewSessionRequest) -> Result<Arc<dyn Session>> {
        let cwd = req.cwd.clone().unwrap_or_else(|| self.default_cwd.clone());
        let cwd = std::fs::canonicalize(&cwd)
            .map_err(|e| anyhow!("cannot open workspace '{cwd}': {e}"))?
            .display()
            .to_string();
        AcpSession::spawn(&self.program, &self.args, cwd, req).await
    }

    async fn branch(&self, _source: &dyn Session, _from_seq: u64) -> Result<Arc<dyn Session>> {
        bail!("the connected agent owns its history; Studio cannot fork it")
    }

    async fn stored(&self) -> Vec<StoredSession> {
        // Persistence belongs to the agent; ACP has no way to enumerate it.
        Vec::new()
    }

    async fn forget(&self, _id: &str) -> Result<()> {
        bail!("the connected agent owns its session history; Studio cannot delete it")
    }
}

// ── Connection ────────────────────────────────────────────────────────────────

/// The writing half of the JSON-RPC connection, plus the calls in flight.
struct Peer {
    stdin: Mutex<ChildStdin>,
    next_id: AtomicI64,
    pending: Mutex<HashMap<i64, oneshot::Sender<Result<Value>>>>,
    /// Permission requests the agent is waiting on, keyed by the id Studio
    /// gave the UI.
    permissions: Mutex<HashMap<String, i64>>,
}

impl Peer {
    async fn send(&self, message: &Value) -> Result<()> {
        let mut line = serde_json::to_string(message)?;
        line.push('\n');
        let mut stdin = self.stdin.lock().await;
        stdin.write_all(line.as_bytes()).await?;
        stdin.flush().await?;
        Ok(())
    }

    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        self.send(&json!({ "jsonrpc": "2.0", "method": method, "params": params }))
            .await
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);
        self.send(&json!({
            "jsonrpc": "2.0", "id": id, "method": method, "params": params
        }))
        .await?;
        rx.await
            .map_err(|_| anyhow!("the agent exited before answering '{method}'"))?
    }

    async fn respond(&self, id: i64, result: Value) -> Result<()> {
        self.send(&json!({ "jsonrpc": "2.0", "id": id, "result": result }))
            .await
    }

    /// Fail every outstanding call. Called when the child exits, so callers
    /// get an error instead of waiting forever on a dead process.
    async fn abandon_all(&self) {
        let pending: Vec<_> = self.pending.lock().await.drain().collect();
        for (_, tx) in pending {
            let _ = tx.send(Err(anyhow!("the agent process exited")));
        }
    }
}

// ── Session ───────────────────────────────────────────────────────────────────

pub struct AcpSession {
    id: String,
    /// The id the agent knows this session by, which need not match ours.
    remote_id: String,
    peer: Arc<Peer>,
    feed: Arc<EventFeed>,
    cwd: String,
    mode: Mutex<String>,
    created_at: String,
    running: Arc<AtomicBool>,
    turn: Mutex<Option<JoinHandle<()>>>,
    child: Mutex<Child>,
    reader: JoinHandle<()>,
}

impl AcpSession {
    async fn spawn(
        program: &str,
        args: &[String],
        cwd: String,
        req: NewSessionRequest,
    ) -> Result<Arc<dyn Session>> {
        let mut child = tokio::process::Command::new(program)
            .args(args)
            .current_dir(&cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // The agent's logs belong on our stderr, not mixed into the
            // protocol stream.
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("could not start ACP agent '{program}'"))?;

        let stdin = child.stdin.take().context("agent has no stdin")?;
        let stdout = child.stdout.take().context("agent has no stdout")?;

        let peer = Arc::new(Peer {
            stdin: Mutex::new(stdin),
            next_id: AtomicI64::new(1),
            pending: Mutex::new(HashMap::new()),
            permissions: Mutex::new(HashMap::new()),
        });
        let feed = Arc::new(EventFeed::new());

        let reader = {
            let peer = Arc::clone(&peer);
            let feed = Arc::clone(&feed);
            tokio::spawn(async move {
                let mut lines = BufReader::new(stdout).lines();
                loop {
                    match lines.next_line().await {
                        Ok(Some(line)) if !line.trim().is_empty() => {
                            if let Err(e) = dispatch(&peer, &feed, &line).await {
                                warn!("could not handle agent message: {e}");
                            }
                        }
                        Ok(Some(_)) => continue,
                        // Clean EOF or a broken pipe both mean the same thing.
                        Ok(None) | Err(_) => break,
                    }
                }
                peer.abandon_all().await;
                feed.push(StudioEvent::studio(
                    studio_kinds::BACKEND_LOST,
                    json!({ "reason": "the agent process exited" }),
                ));
                feed.close();
            })
        };

        // Handshake, then open the session.
        peer.request(
            "initialize",
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "clientCapabilities": {
                    "fs": { "readTextFile": true, "writeTextFile": true },
                    "terminal": false
                },
                "clientInfo": { "name": "eventage-studio", "version": env!("CARGO_PKG_VERSION") }
            }),
        )
        .await?;

        let opened = peer
            .request("session/new", json!({ "cwd": cwd, "mcpServers": [] }))
            .await?;
        let remote_id = opened
            .get("sessionId")
            .and_then(|v| v.as_str())
            .context("the agent did not return a sessionId")?
            .to_string();

        let mode = opened
            .get("modes")
            .and_then(|m| m.get("currentModeId"))
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .to_string();

        let session = Arc::new(Self {
            id: remote_id.clone(),
            remote_id,
            peer,
            feed,
            cwd,
            mode: Mutex::new(mode),
            created_at: chrono::Utc::now().to_rfc3339(),
            running: Arc::new(AtomicBool::new(false)),
            turn: Mutex::new(None),
            child: Mutex::new(child),
            reader,
        });

        if let Some(mode) = req.mode.as_deref() {
            // Best effort: an agent that does not support modes still works.
            let _ = session.set_mode(mode).await;
        }

        Ok(session)
    }
}

/// Turn one line from the agent into feed events and protocol replies.
async fn dispatch(peer: &Arc<Peer>, feed: &Arc<EventFeed>, line: &str) -> Result<()> {
    let message: Value = serde_json::from_str(line)
        .with_context(|| format!("agent sent invalid JSON: {}", truncate(line, 200)))?;

    // A response to something we asked.
    if message.get("method").is_none() {
        let Some(id) = message.get("id").and_then(|v| v.as_i64()) else {
            bail!("agent sent a message with neither method nor id");
        };
        if let Some(tx) = peer.pending.lock().await.remove(&id) {
            let outcome = match message.get("error") {
                Some(error) => Err(anyhow!("agent error: {error}")),
                None => Ok(message.get("result").cloned().unwrap_or(json!({}))),
            };
            let _ = tx.send(outcome);
        }
        return Ok(());
    }

    let method = message["method"].as_str().unwrap_or_default();
    let params = message.get("params").cloned().unwrap_or(json!({}));

    // A request from the agent: it expects a reply.
    if let Some(id) = message.get("id").and_then(|v| v.as_i64()) {
        match method {
            "session/request_permission" => {
                // Park it: the answer comes from the UI, not from here.
                let request_id = uuid::Uuid::new_v4().to_string();
                peer.permissions.lock().await.insert(request_id.clone(), id);
                let tool_call = params.get("toolCall").cloned().unwrap_or(json!({}));
                feed.push(StudioEvent::studio(
                    kinds::PERMISSION_REQUEST,
                    json!({
                        "request_id": request_id,
                        "tool": tool_call.get("title")
                            .and_then(|v| v.as_str())
                            .unwrap_or("tool"),
                        "arguments": tool_call.get("rawInput").cloned(),
                        "options": params.get("options").cloned(),
                    }),
                ));
            }
            "fs/read_text_file" => {
                let path = params.get("path").and_then(|v| v.as_str()).unwrap_or("");
                let result = match tokio::fs::read_to_string(path).await {
                    Ok(content) => json!({ "content": content }),
                    Err(e) => json!({ "content": "", "error": e.to_string() }),
                };
                peer.respond(id, result).await?;
            }
            "fs/write_text_file" => {
                let path = params.get("path").and_then(|v| v.as_str()).unwrap_or("");
                let content = params.get("content").and_then(|v| v.as_str()).unwrap_or("");
                let result = match tokio::fs::write(path, content).await {
                    Ok(()) => json!({}),
                    Err(e) => json!({ "error": e.to_string() }),
                };
                peer.respond(id, result).await?;
            }
            other => {
                debug!("agent asked for '{other}', which Studio does not implement");
                peer.respond(id, json!({})).await?;
            }
        }
        return Ok(());
    }

    // A notification.
    if method == "session/update" {
        if let Some(update) = params.get("update") {
            for event in normalise_update(update) {
                feed.push(event);
            }
        }
    }
    Ok(())
}

/// Map one ACP `session/update` onto the kinds the UI already understands.
///
/// The point is that the UI has a single reducer: whatever it can render from
/// a locally hosted session, it renders here too, from the subset ACP carries.
fn normalise_update(update: &Value) -> Vec<StudioEvent> {
    let kind = update
        .get("sessionUpdate")
        .and_then(|v| v.as_str())
        .unwrap_or_default();

    let text_of = |field: &str| -> Option<String> {
        update
            .get(field)
            .and_then(|c| c.get("text"))
            .and_then(|v| v.as_str())
            .map(str::to_string)
    };

    match kind {
        "agent_message_chunk" => text_of("content")
            .map(|text| {
                vec![StudioEvent::studio(
                    kinds::ASSISTANT_DELTA,
                    json!({ "content": text }),
                )]
            })
            .unwrap_or_default(),

        "agent_thought_chunk" => text_of("content")
            .map(|text| {
                vec![StudioEvent::studio(
                    kinds::ASSISTANT_DELTA,
                    json!({ "reasoning_content": text }),
                )]
            })
            .unwrap_or_default(),

        "tool_call" => vec![StudioEvent::studio(
            kinds::TOOL_CALL_PROPOSED,
            json!({
                "tool_call_id": update.get("toolCallId").and_then(|v| v.as_str()).unwrap_or(""),
                "name": update.get("title").and_then(|v| v.as_str()).unwrap_or("tool"),
                "title": update.get("title").cloned(),
                "tool_kind": update.get("kind").cloned(),
                "arguments": update.get("rawInput").cloned(),
            }),
        )],

        "tool_call_update" => {
            let status = update
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("in_progress");
            let id = update
                .get("toolCallId")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            // Only a terminal status closes the card; anything else is a
            // progress tick.
            if !matches!(status, "completed" | "failed") {
                return vec![StudioEvent::studio(
                    "tool.call.progress",
                    json!({ "tool_call_id": id, "status": status }),
                )];
            }

            // ACP carries diffs as structured content; hoist them into the
            // same `_diff` shape the local tools emit so one card renders both.
            let mut result = json!({});
            if let Some(items) = update.get("content").and_then(|c| c.as_array()) {
                for item in items {
                    if item.get("type").and_then(|v| v.as_str()) == Some("diff") {
                        result["_diff"] = json!({
                            "path": item.get("path").cloned().unwrap_or(Value::Null),
                            "old_text": item.get("oldText").cloned().unwrap_or(Value::Null),
                            "new_text": item.get("newText").cloned().unwrap_or(Value::Null),
                        });
                    } else if let Some(text) = item
                        .get("content")
                        .and_then(|c| c.get("text"))
                        .and_then(|v| v.as_str())
                    {
                        result["text"] = json!(text);
                    }
                }
            }
            if let Some(locations) = update.get("locations") {
                result["_locations"] = locations.clone();
            }
            if let Some(raw) = update.get("rawOutput") {
                result["raw"] = raw.clone();
            }

            let mut payload = json!({ "tool_call_id": id, "result": result });
            if status == "failed" {
                payload["error"] = json!(result
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("the tool call failed"));
            }
            vec![StudioEvent::studio(kinds::TOOL_RESULT, payload)]
        }

        "plan" => vec![StudioEvent::studio(
            "studio.plan",
            json!({ "entries": update.get("entries").cloned().unwrap_or(json!([])) }),
        )],

        "current_mode_update" => vec![StudioEvent::studio(
            studio_kinds::MODE_CHANGED,
            json!({ "mode": update.get("currentModeId").cloned().unwrap_or(Value::Null) }),
        )],

        _ => Vec::new(),
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

#[async_trait]
impl Session for AcpSession {
    fn feed(&self) -> Arc<EventFeed> {
        Arc::clone(&self.feed)
    }

    fn info(&self) -> SessionInfo {
        SessionInfo {
            id: self.id.clone(),
            cwd: self.cwd.clone(),
            mode: self
                .mode
                .try_lock()
                .map(|m| m.clone())
                .unwrap_or_else(|_| "default".into()),
            title: self
                .feed
                .first_user_text()
                .unwrap_or_else(|| "New session".into()),
            created_at: self.created_at.clone(),
            running: self.running.load(Ordering::SeqCst),
            turns: self.feed.count_turns(),
        }
    }

    async fn prompt(&self, blocks: Vec<PromptBlock>) -> Result<()> {
        if self.running.load(Ordering::SeqCst) {
            bail!("this session is already working on something");
        }
        if self.feed.is_closed() {
            bail!("the agent process is no longer running");
        }

        // ACP has no equivalent of the agent's own cycle events, so Studio
        // brackets the turn itself. The UI's running state is then driven by
        // the same events in both backends.
        self.feed.push(StudioEvent::studio(
            kinds::USER_MESSAGE,
            json!({ "parts": blocks, "text": first_text(&blocks) }),
        ));
        self.feed
            .push(StudioEvent::studio(kinds::AGENT_CYCLE_START, json!({})));
        self.running.store(true, Ordering::SeqCst);

        let peer = Arc::clone(&self.peer);
        let feed = Arc::clone(&self.feed);
        let running = Arc::clone(&self.running);
        let session_id = self.remote_id.clone();
        *self.turn.lock().await = Some(tokio::spawn(async move {
            let outcome = peer
                .request(
                    "session/prompt",
                    json!({ "sessionId": session_id, "prompt": blocks }),
                )
                .await;
            running.store(false, Ordering::SeqCst);
            match outcome {
                Ok(result) => {
                    let reason = result
                        .get("stopReason")
                        .and_then(|v| v.as_str())
                        .unwrap_or("end_turn")
                        .to_string();
                    feed.push(StudioEvent::studio(kinds::AGENT_CYCLE_END, json!({})));
                    feed.push(StudioEvent::studio(
                        studio_kinds::TURN_ENDED,
                        json!({ "reason": reason }),
                    ));
                }
                Err(e) => {
                    feed.push(StudioEvent::studio(kinds::AGENT_CYCLE_END, json!({})));
                    feed.push(StudioEvent::studio(
                        studio_kinds::TURN_FAILED,
                        json!({ "error": e.to_string() }),
                    ));
                }
            }
        }));
        Ok(())
    }

    async fn interrupt(&self) -> Result<()> {
        if !self.running.load(Ordering::SeqCst) {
            return Ok(());
        }
        // Cancellation is a notification: the agent still answers the
        // outstanding session/prompt, which is what ends the turn.
        self.peer
            .notify("session/cancel", json!({ "sessionId": self.remote_id }))
            .await?;
        self.feed.push(StudioEvent::studio(
            studio_kinds::TURN_INTERRUPTED,
            json!({}),
        ));
        Ok(())
    }

    async fn set_mode(&self, mode: &str) -> Result<()> {
        self.peer
            .request(
                "session/set_mode",
                json!({ "sessionId": self.remote_id, "modeId": mode }),
            )
            .await?;
        *self.mode.lock().await = mode.to_string();
        Ok(())
    }

    async fn rewind(&self, turns: usize, _to: Option<&str>) -> Result<usize> {
        // An eventage agent exposes this; others will reject the method, and
        // the UI reports that rather than pretending it worked.
        let result = self
            .peer
            .request(
                "session/rewind",
                json!({ "sessionId": self.remote_id, "turns": turns }),
            )
            .await
            .map_err(|e| anyhow!("this agent does not support rewind: {e}"))?;
        let remaining = result
            .get("turnsRemaining")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        self.feed.push(StudioEvent::studio(
            studio_kinds::REWOUND,
            json!({ "turns": turns, "remaining": remaining }),
        ));
        Ok(remaining)
    }

    async fn override_summary(&self, _replacement: SummaryOverride) -> Result<()> {
        bail!("the connected agent assembles its own context; Studio cannot edit it")
    }

    async fn permission(&self, response: PermissionResponse) -> Result<()> {
        let id = self
            .peer
            .permissions
            .lock()
            .await
            .remove(&response.request_id)
            .ok_or_else(|| anyhow!("that permission request is no longer waiting"))?;

        let outcome = if response.approve {
            json!({ "outcome": { "outcome": "selected", "optionId": "allow_once" } })
        } else {
            json!({ "outcome": { "outcome": "selected", "optionId": "reject_once" } })
        };
        self.peer.respond(id, outcome).await?;

        self.feed.push(StudioEvent::studio(
            kinds::PERMISSION_DECISION,
            json!({
                "request_id": response.request_id,
                "approve": response.approve,
                "reason": response.reason,
            }),
        ));
        Ok(())
    }

    async fn shutdown(&self) {
        if let Some(handle) = self.turn.lock().await.take() {
            handle.abort();
        }
        self.reader.abort();
        self.feed.close();
        // kill_on_drop covers the crash path; this covers the orderly one.
        let _ = self.child.lock().await.kill().await;
    }
}

/// The first text block of a prompt, for the session title.
fn first_text(blocks: &[PromptBlock]) -> String {
    blocks
        .iter()
        .find_map(|b| match b {
            PromptBlock::Text { text } => Some(text.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_chunks_become_assistant_deltas() {
        let events = normalise_update(&json!({
            "sessionUpdate": "agent_message_chunk",
            "content": { "type": "text", "text": "hello" }
        }));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, kinds::ASSISTANT_DELTA);
        assert_eq!(events[0].payload["content"], "hello");
    }

    #[test]
    fn thoughts_are_kept_separate_from_the_answer() {
        let events = normalise_update(&json!({
            "sessionUpdate": "agent_thought_chunk",
            "content": { "type": "text", "text": "considering" }
        }));
        assert_eq!(events[0].payload["reasoning_content"], "considering");
        assert!(events[0].payload.get("content").is_none());
    }

    #[test]
    fn only_a_terminal_status_closes_a_tool_card() {
        let progress = normalise_update(&json!({
            "sessionUpdate": "tool_call_update",
            "toolCallId": "t1",
            "status": "in_progress"
        }));
        assert_eq!(progress[0].kind, "tool.call.progress");

        let done = normalise_update(&json!({
            "sessionUpdate": "tool_call_update",
            "toolCallId": "t1",
            "status": "completed"
        }));
        assert_eq!(done[0].kind, kinds::TOOL_RESULT);
    }

    #[test]
    fn a_protocol_diff_lands_in_the_shape_the_card_renders() {
        let events = normalise_update(&json!({
            "sessionUpdate": "tool_call_update",
            "toolCallId": "t1",
            "status": "completed",
            "content": [{
                "type": "diff",
                "path": "src/main.rs",
                "oldText": "before",
                "newText": "after"
            }]
        }));
        let diff = &events[0].payload["result"]["_diff"];
        assert_eq!(diff["path"], "src/main.rs");
        assert_eq!(diff["old_text"], "before");
        assert_eq!(diff["new_text"], "after");
    }

    #[test]
    fn a_failed_call_carries_an_error_the_ui_can_show() {
        let events = normalise_update(&json!({
            "sessionUpdate": "tool_call_update",
            "toolCallId": "t1",
            "status": "failed",
            "content": [{ "type": "content", "content": { "type": "text", "text": "no such file" } }]
        }));
        assert_eq!(events[0].payload["error"], "no such file");
    }

    #[test]
    fn unknown_updates_are_ignored_rather_than_guessed_at() {
        assert!(normalise_update(&json!({ "sessionUpdate": "something_new" })).is_empty());
        assert!(normalise_update(&json!({})).is_empty());
    }

    #[test]
    fn a_prompt_title_comes_from_its_first_text_block() {
        let blocks = vec![
            PromptBlock::Image {
                data: "…".into(),
                mime_type: "image/png".into(),
            },
            PromptBlock::text("what is this"),
        ];
        assert_eq!(first_text(&blocks), "what is this");
        assert_eq!(first_text(&[]), "");
    }
}
