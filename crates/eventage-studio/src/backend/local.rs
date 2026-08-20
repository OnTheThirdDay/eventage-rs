//! The in-process backend: Studio hosts the coding session itself.
//!
//! Because the session runs here, Studio holds its `EventBus` directly and
//! forwards every event to the UI untouched. That is the whole reason this
//! mode exists — the trace panel gets hook decisions, token accounting,
//! checkpoints and rejected branches, none of which a protocol between
//! processes would carry.

use crate::backend::{Backend, Session};
use crate::feed::EventFeed;
use crate::index::{IndexEntry, SessionIndex};
use crate::protocol::{
    studio_kinds, AppInfo, ModeInfo, NewSessionRequest, PermissionResponse, PromptBlock,
    SessionInfo, StoredSession, StudioEvent, SummaryOverride,
};
use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use eventage::agent::recovery::{reconcile_interrupted_tools, ToolRecovery};
use eventage::event::kinds;
use eventage::Event;
use eventage_code::agent::CodingSession;
use eventage_code::config::{ModelConfig, PermissionMode, Provider, SessionConfig};
use serde_json::json;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing::{info, warn};

// ── Backend ───────────────────────────────────────────────────────────────────

pub struct LocalBackend {
    /// Read at every `open`, so a change in the settings screen reaches the
    /// next session without restarting Studio — and never the running one,
    /// whose provider is already built into its agent.
    settings: Arc<crate::model_settings::ModelSettings>,
    default_cwd: String,
    index: Arc<SessionIndex>,
    state_dir: PathBuf,
}

impl LocalBackend {
    pub async fn new(
        settings: Arc<crate::model_settings::ModelSettings>,
        default_cwd: String,
    ) -> Self {
        let model = settings.get();
        let state_dir = SessionConfig::new(default_cwd.clone(), model.clone()).state_dir();
        tokio::fs::create_dir_all(&state_dir).await.ok();
        let index = Arc::new(SessionIndex::load(&state_dir).await);
        Self {
            settings,
            default_cwd,
            index,
            state_dir,
        }
    }

    fn config_for(
        &self,
        cwd: &str,
        mode: Option<&str>,
        model: ModelConfig,
    ) -> Result<SessionConfig> {
        let mut config = SessionConfig::new(cwd.to_string(), model);
        if let Some(id) = mode {
            config.mode = PermissionMode::from_id(id)
                .ok_or_else(|| anyhow!("unknown permission mode '{id}' (plan|ask|auto|yolo)"))?;
        }
        Ok(config)
    }
}

/// Warn when no credentials were configured.
///
/// `ModelConfig::from_env` falls back to a local OpenAI-compatible server
/// when it finds no key, which is right for someone running Ollama and
/// baffling for everyone else: the app looks fine until the first message
/// fails with a connection error to a host they never chose. Detecting it at
/// startup lets the UI say so before anything is typed.
fn credentials_hint(model: &ModelConfig) -> Option<String> {
    // Read off the resolved profile, never from the environment. Startup
    // moves credentials out of it, so asking `env::var("QWEN_API_KEY")` here
    // always said "unset" — and the banner told anyone with a perfectly good
    // key, pointed at a perfectly good gateway, that they had no credentials
    // at all.
    if model.credentialed {
        return None;
    }
    Some(format!(
        "No API key found, so Studio is pointed at {}. If that is not a local \
         server you are running, set ANTHROPIC_API_KEY, QWEN_API_KEY, or \
         OPENAI_API_KEY (with OPENAI_BASE_URL) and restart.",
        model.base_url()
    ))
}

fn provider_name(provider: Provider) -> &'static str {
    match provider {
        Provider::Anthropic => "Anthropic",
        Provider::OpenAiResponses => "OpenAI Responses",
        Provider::Qwen => "Qwen",
        Provider::OpenAiChat => "OpenAI-compatible",
    }
}

#[async_trait]
impl Backend for LocalBackend {
    fn model_settings(&self) -> Option<Arc<crate::model_settings::ModelSettings>> {
        Some(Arc::clone(&self.settings))
    }

    fn info(&self) -> AppInfo {
        // Read fresh, so the title bar reflects a change made in settings
        // rather than whatever was configured when Studio started.
        let model = self.settings.get();
        AppInfo {
            backend: "local",
            backend_detail: "eventage-code, hosted in this process".into(),
            model: model.model.clone(),
            provider: provider_name(model.provider).into(),
            default_cwd: self.default_cwd.clone(),
            modes: PermissionMode::ALL
                .iter()
                .map(|m| ModeInfo {
                    id: m.id().into(),
                    label: m.label().into(),
                    description: m.description().into(),
                })
                .collect(),
            version: env!("CARGO_PKG_VERSION"),
            full_trace: true,
            credentials_hint: credentials_hint(&model),
        }
    }

    async fn open(&self, req: NewSessionRequest) -> Result<Arc<dyn Session>> {
        let cwd = req.cwd.clone().unwrap_or_else(|| self.default_cwd.clone());
        let cwd = std::fs::canonicalize(&cwd)
            .map_err(|e| anyhow!("cannot open workspace '{cwd}': {e}"))?
            .display()
            .to_string();
        let config = self.config_for(&cwd, req.mode.as_deref(), self.settings.get())?;

        let (id, session) = match &req.resume {
            Some(id) => {
                info!(session = %id, "reopening session");
                let session = CodingSession::resume(id, config.clone(), None).await?;
                (id.clone(), session)
            }
            None => {
                let id = uuid::Uuid::new_v4().to_string();
                let session = CodingSession::create(id.clone(), config.clone(), None).await?;
                (id, session)
            }
        };

        self.index
            .record(
                &id,
                IndexEntry {
                    cwd: cwd.clone(),
                    title: self
                        .index
                        .get(&id)
                        .await
                        .map(|e| e.title)
                        .unwrap_or_default(),
                    updated_at: chrono::Utc::now().to_rfc3339(),
                },
            )
            .await;

        Ok(LocalSession::start(id, session, config, Arc::clone(&self.index)).await)
    }

    async fn branch(&self, source: &dyn Session, from_seq: u64) -> Result<Arc<dyn Session>> {
        let info = source.info();
        // Everything the source holds up to the fork point, minus the
        // bookkeeping Studio added — a branch inherits the conversation, not
        // the other session's turn markers.
        let kept: Vec<_> = source
            .feed()
            .since(0)
            .into_iter()
            .filter(|e| e.seq <= from_seq && !e.kind.starts_with("studio."))
            .collect();
        if kept.is_empty() {
            bail!("there is nothing before that point to branch from");
        }

        let id = uuid::Uuid::new_v4().to_string();
        let config = self.config_for(&info.cwd, Some(&info.mode), self.settings.get())?;

        // Written to the new session's own log, then opened as a resume: the
        // branch is a real session from birth, with a history it can replay,
        // rather than a live copy that would vanish on restart.
        let store =
            eventage::sqlite::SqliteEventStore::new(config.state_dir().join(format!("{id}.db")))
                .await?;
        for event in &kept {
            store.append(&to_event(event)?).await?;
        }

        info!(from = %info.id, to = %id, events = kept.len(), "branched session");
        let session = CodingSession::resume(&id, config.clone(), None).await?;
        self.index
            .record(
                &id,
                IndexEntry {
                    cwd: info.cwd.clone(),
                    title: format!("branch of {}", info.title),
                    updated_at: chrono::Utc::now().to_rfc3339(),
                },
            )
            .await;
        Ok(LocalSession::start(id, session, config, Arc::clone(&self.index)).await)
    }

    async fn stored(&self) -> Vec<StoredSession> {
        let mut out = Vec::new();
        let Ok(mut entries) = tokio::fs::read_dir(&self.state_dir).await else {
            return out;
        };
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("db") {
                continue;
            }
            let Some(id) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let meta = entry.metadata().await.ok();
            let indexed = self.index.get(id).await.unwrap_or_default();
            out.push(StoredSession {
                id: id.to_string(),
                cwd: if indexed.cwd.is_empty() {
                    self.default_cwd.clone()
                } else {
                    indexed.cwd
                },
                title: if indexed.title.is_empty() {
                    "Untitled session".into()
                } else {
                    indexed.title
                },
                updated_at: if indexed.updated_at.is_empty() {
                    meta.as_ref()
                        .and_then(|m| m.modified().ok())
                        .map(|t| chrono::DateTime::<chrono::Utc>::from(t).to_rfc3339())
                        .unwrap_or_default()
                } else {
                    indexed.updated_at
                },
                size_bytes: meta.map(|m| m.len()).unwrap_or(0),
            });
        }
        // Newest first, which is the order the sidebar wants.
        out.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        out
    }

    async fn forget(&self, id: &str) -> Result<()> {
        // Reject anything that could escape the state directory: session ids
        // reach this from an HTTP path segment.
        if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
            bail!("not a session id");
        }
        let db = self.state_dir.join(format!("{id}.db"));
        if db.exists() {
            tokio::fs::remove_file(&db).await?;
        }
        // SQLite may leave a write-ahead log and shared-memory file behind.
        for suffix in ["db-wal", "db-shm"] {
            let _ = tokio::fs::remove_file(self.state_dir.join(format!("{id}.{suffix}"))).await;
        }
        self.index.remove(id).await;
        Ok(())
    }
}

/// Turn a Studio event back into the bus event it came from.
fn to_event(studio: &StudioEvent) -> Result<Event> {
    Ok(Event {
        id: studio.id.parse().unwrap_or_else(|_| uuid::Uuid::new_v4()),
        timestamp: studio.ts.parse().unwrap_or_else(|_| chrono::Utc::now()),
        kind: studio.kind.clone(),
        payload: studio.payload.clone(),
        parent_event_id: None,
        metadata: studio.meta.clone(),
    })
}

// ── Session ───────────────────────────────────────────────────────────────────

pub struct LocalSession {
    id: String,
    session: Arc<CodingSession>,
    feed: Arc<EventFeed>,
    index: Arc<SessionIndex>,
    cwd: String,
    mode: Mutex<PermissionMode>,
    created_at: String,
    /// The task running the current turn, if any.
    turn: Mutex<Option<JoinHandle<()>>>,
    /// Tools the user chose to allow for the rest of the session.
    always_allow: Arc<Mutex<HashSet<String>>>,
    bridge: JoinHandle<()>,
}

impl LocalSession {
    async fn start(
        id: String,
        session: CodingSession,
        config: SessionConfig,
        index: Arc<SessionIndex>,
    ) -> Arc<dyn Session> {
        let session = Arc::new(session);
        let feed = Arc::new(EventFeed::new());
        let always_allow: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));

        // Subscribe before reading the log, so an event published in between
        // is seen twice rather than missed. The duplicate is then dropped by
        // id below — the cheaper mistake of the two.
        let mut rx = session.bus.subscribe();
        // Seeded from what was *recorded*, not from what went back onto the
        // bus. `restore_from` rebuilds the conversation and leaves off the
        // events that were only ever broadcast — streaming deltas, context
        // assemblies — so seeding from `bus.log()` meant a reopened session
        // came back with an empty trace and an empty context panel.
        let restored = match session.history() {
            [] => session.bus.log().await,
            history => history.to_vec(),
        };
        let mut pending_duplicates: HashSet<uuid::Uuid> = restored.iter().map(|e| e.id).collect();
        for event in &restored {
            feed.push(StudioEvent::from(event));
        }

        let bridge = {
            let feed = Arc::clone(&feed);
            let bus = session.bus.clone();
            let always_allow = Arc::clone(&always_allow);
            tokio::spawn(async move {
                while let Some(event) = rx.recv().await {
                    if !pending_duplicates.is_empty() && pending_duplicates.remove(&event.id) {
                        continue;
                    }
                    // Standing approvals are answered here rather than in the
                    // UI, so the user is not asked again for something they
                    // already said yes to. The request and the decision both
                    // stay in the trace.
                    if event.kind == kinds::PERMISSION_REQUEST {
                        let tool = event
                            .payload
                            .get("tool")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string();
                        let key = permission_key(&tool, event.payload.get("arguments"));
                        if always_allow.lock().await.contains(&key) {
                            let request_id = event
                                .payload
                                .get("request_id")
                                .and_then(|v| v.as_str())
                                .unwrap_or_default()
                                .to_string();
                            let _ = bus
                                .publish(Event::new(
                                    kinds::PERMISSION_DECISION,
                                    json!({
                                        "request_id": request_id,
                                        "approve": true,
                                        "auto": "always_allow",
                                    }),
                                ))
                                .await;
                        }
                    }
                    feed.push(StudioEvent::from(&event));
                }
            })
        };

        Arc::new(Self {
            id,
            session,
            feed,
            index,
            cwd: config.cwd.clone(),
            mode: Mutex::new(config.mode),
            created_at: chrono::Utc::now().to_rfc3339(),
            turn: Mutex::new(None),
            always_allow,
            bridge,
        })
    }

    async fn is_running(&self) -> bool {
        self.turn
            .lock()
            .await
            .as_ref()
            .is_some_and(|h| !h.is_finished())
    }
}

#[async_trait]
impl Session for LocalSession {
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
                .map(|m| m.id().to_string())
                .unwrap_or_else(|_| "ask".into()),
            title: self
                .feed
                .first_user_text()
                .unwrap_or_else(|| "New session".into()),
            created_at: self.created_at.clone(),
            running: self
                .turn
                .try_lock()
                .map(|t| t.as_ref().is_some_and(|h| !h.is_finished()))
                .unwrap_or(true),
            turns: self.feed.count_turns(),
        }
    }

    async fn prompt(&self, blocks: Vec<PromptBlock>) -> Result<()> {
        let mut turn = self.turn.lock().await;
        if turn.as_ref().is_some_and(|h| !h.is_finished()) {
            bail!("this session is already working on something");
        }

        self.session.submit_prompt(&blocks).await?;

        // Title the conversation from its opening message.
        if let Some(text) = self.feed.first_user_text() {
            self.index.title_once(&self.id, &text).await;
        }

        let session = Arc::clone(&self.session);
        *turn = Some(tokio::spawn(async move {
            let outcome = session.run_cycle().await;

            // Broadcast rather than pushed straight to the feed.
            //
            // The feed has two producers: this task, and the bridge draining
            // the bus. Pushing here raced the bridge, and the turn's own
            // `assistant.message` regularly landed *after* the notice that
            // the turn had ended — which reads to any consumer as a reply
            // arriving after the reply was over. Going through the bus puts
            // it behind the events it follows, because they leave from the
            // same place in the same order. It stays ephemeral, so it never
            // becomes part of the conversation.
            let announcement = match &outcome {
                Ok(()) => Event::new(
                    studio_kinds::TURN_ENDED,
                    json!({
                        "reason": if session.was_cancelled() { "cancelled" } else { "end_turn" }
                    }),
                ),
                Err(e) => {
                    warn!("turn failed: {e}");
                    Event::new(studio_kinds::TURN_FAILED, json!({ "error": e.to_string() }))
                }
            };
            session.bus.broadcast(announcement);
        }));
        Ok(())
    }

    async fn interrupt(&self) -> Result<()> {
        if !self.is_running().await {
            return Ok(());
        }
        self.session.cancel();
        if let Some(handle) = self.turn.lock().await.take() {
            handle.abort();
        }

        // Aborting mid-tool leaves a tool call with no result, which is not a
        // valid history to send to any provider. Close the orphans out before
        // the next turn assembles context.
        match reconcile_interrupted_tools(&self.session.bus, &ToolRecovery::new(), None).await {
            Ok(report) if !report.is_empty() => {
                info!(
                    resolved = report.total(),
                    "closed out interrupted tool calls"
                );
            }
            Ok(_) => {}
            Err(e) => warn!("could not reconcile interrupted tools: {e}"),
        }

        // Say so on the bus, so the model knows next turn that it was stopped
        // rather than that its work silently vanished.
        self.session
            .bus
            .publish(Event::new(
                kinds::SYSTEM_MESSAGE,
                json!({ "content": "[the user stopped the previous turn before it finished]" }),
            ))
            .await?;

        self.feed.push(StudioEvent::studio(
            studio_kinds::TURN_INTERRUPTED,
            json!({}),
        ));
        Ok(())
    }

    async fn set_mode(&self, mode: &str) -> Result<()> {
        let parsed = PermissionMode::from_id(mode)
            .ok_or_else(|| anyhow!("unknown permission mode '{mode}' (plan|ask|auto|yolo)"))?;
        self.session.set_mode(parsed).await;
        *self.mode.lock().await = parsed;
        self.feed.push(StudioEvent::studio(
            studio_kinds::MODE_CHANGED,
            json!({ "mode": parsed.id(), "label": parsed.label() }),
        ));
        Ok(())
    }

    async fn rewind(&self, turns: usize, to: Option<&str>) -> Result<usize> {
        if self.is_running().await {
            bail!("stop the current turn before rewinding");
        }
        let remaining = match to {
            Some(id) => {
                let anchor = id
                    .parse()
                    .map_err(|_| anyhow!("'{id}' is not a checkpoint id"))?;
                self.session.rewind_to(anchor).await?
            }
            None => self.session.rewind(turns).await?,
        };
        self.feed.push(StudioEvent::studio(
            studio_kinds::REWOUND,
            json!({ "turns": turns, "remaining": remaining }),
        ));
        Ok(remaining)
    }

    async fn override_summary(&self, replacement: SummaryOverride) -> Result<()> {
        if self.is_running().await {
            bail!("stop the current turn before changing its context");
        }
        eventage::agent::summarizing::override_summary(
            &self.session.bus,
            replacement.summary,
            replacement.summarized_count,
        )
        .await?;
        Ok(())
    }

    async fn permission(&self, response: PermissionResponse) -> Result<()> {
        if response.always && response.approve {
            if let Some((tool, args)) = request_subject(&self.feed, &response.request_id) {
                self.always_allow
                    .lock()
                    .await
                    .insert(permission_key(&tool, args.as_ref()));
            }
        }
        self.session
            .bus
            .publish(Event::new(
                kinds::PERMISSION_DECISION,
                json!({
                    "request_id": response.request_id,
                    "approve": response.approve,
                    "reason": response.reason,
                }),
            ))
            .await?;
        Ok(())
    }

    async fn shutdown(&self) {
        if let Some(handle) = self.turn.lock().await.take() {
            handle.abort();
        }
        self.bridge.abort();
        self.feed.close();

        // Wait for the log to be written before saying the session is closed.
        // Dropping the persistence task instead lost whatever it had not yet
        // flushed — the last events of the session, which are exactly the
        // ones a resume needs.
        let failures = self.session.close().await;
        if failures > 0 {
            warn!(
                failures,
                session = %self.id,
                "events failed to reach the log; this session's history is incomplete"
            );
        }
    }
}

/// Which tool a pending permission request was about.
///
/// The request id is all the UI sends back, so the tool name is recovered
/// from the request event that is already in the feed.
/// The key a standing approval is remembered under.
///
/// Tool **and arguments**, not the tool alone. Keyed by name, one "always
/// allow" on a `bash` prompt for `cargo test` silently approved every later
/// shell command in the session — the mode selector still said Auto while the
/// session behaved like Yolo for the one tool that most needs a prompt. The
/// risk in `bash`, `git` and `web_fetch` lives entirely in the arguments, so
/// that is what the grant has to name.
///
/// Exact rather than pattern-matched. A prefix language (`bash(cargo test:*)`)
/// is the richer answer and a design of its own; an exact key is the honest
/// version of what the button can promise today, and it still absorbs the
/// case it was built for — the same command re-run every turn.
///
/// `serde_json`'s maps are ordered, so equal arguments render equal here.
fn permission_key(tool: &str, args: Option<&serde_json::Value>) -> String {
    match args {
        Some(args) => format!("{tool}\u{1f}{args}"),
        None => tool.to_string(),
    }
}

/// The `(tool, arguments)` a permission request was raised for.
fn request_subject(
    feed: &EventFeed,
    request_id: &str,
) -> Option<(String, Option<serde_json::Value>)> {
    feed.since(0)
        .iter()
        .rev()
        .find(|e| {
            e.kind == kinds::PERMISSION_REQUEST
                && e.payload.get("request_id").and_then(|v| v.as_str()) == Some(request_id)
        })
        .and_then(|e| {
            let tool = e.payload.get("tool").and_then(|v| v.as_str())?.to_string();
            Some((tool, e.payload.get("arguments").cloned()))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed_with_request(tool: &str, request_id: &str) -> EventFeed {
        let feed = EventFeed::new();
        feed.push(StudioEvent::studio(
            kinds::PERMISSION_REQUEST,
            json!({ "request_id": request_id, "tool": tool }),
        ));
        feed
    }

    #[test]
    fn a_permission_answer_is_matched_back_to_its_tool() {
        let feed = feed_with_request("bash", "req-1");
        let (tool, _) = request_subject(&feed, "req-1").unwrap();
        assert_eq!(tool, "bash");
    }

    #[test]
    fn an_unknown_request_id_yields_nothing_rather_than_a_wrong_tool() {
        let feed = feed_with_request("bash", "req-1");
        assert!(request_subject(&feed, "req-2").is_none());
    }

    #[test]
    fn the_newest_request_wins_when_ids_repeat() {
        let feed = EventFeed::new();
        feed.push(StudioEvent::studio(
            kinds::PERMISSION_REQUEST,
            json!({ "request_id": "r", "tool": "bash" }),
        ));
        feed.push(StudioEvent::studio(
            kinds::PERMISSION_REQUEST,
            json!({ "request_id": "r", "tool": "write_file" }),
        ));
        assert_eq!(request_subject(&feed, "r").unwrap().0, "write_file");
    }

    #[test]
    fn a_standing_approval_does_not_cover_a_different_command() {
        // The whole point: allowing `cargo test` must not pre-approve
        // `rm -rf ~`. Keyed by tool name, it did.
        let allowed = permission_key("bash", Some(&json!({ "command": "cargo test" })));
        let other = permission_key("bash", Some(&json!({ "command": "rm -rf ~" })));
        assert_ne!(allowed, other);

        // The same call again is the case the button exists for.
        assert_eq!(
            allowed,
            permission_key("bash", Some(&json!({ "command": "cargo test" })))
        );
        // Argument order in the JSON object must not change the key.
        assert_eq!(
            permission_key("write_file", Some(&json!({ "a": 1, "b": 2 }))),
            permission_key("write_file", Some(&json!({ "b": 2, "a": 1 }))),
        );
    }
}
