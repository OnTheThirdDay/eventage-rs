//! Studio driving a cowork session.
//!
//! Cowork's own value is that a session is a graph rather than a line, and a
//! graph needs somewhere to be looked at. Studio already renders a transcript,
//! an event trace, permission prompts and a timeline you can scrub — which is
//! most of what "follow along and redirect it if you need to" means in both
//! Claude Cowork and the Codex app — so cowork gets a backend here rather than
//! a second interface of its own.
//!
//! Two things differ from the local coding backend and are worth naming:
//!
//! * **A turn is a fan-out.** Prompting does not run one agent; it snapshots
//!   the folder, splits the goal, and runs the parts against independent
//!   copies. The feed carries every part's work, tagged by workstream, so one
//!   transcript shows several things happening at once.
//! * **Nothing lands until it is adopted.** A finished turn leaves results in
//!   the shadow repository, not in the user's folder. The UI's job at that
//!   point is comparison, which is why `cowork.workstream.finished` carries
//!   the file list and Studio renders it as a choice rather than a result.
//!
//! The steering modes are cowork's — manual, auto, skip — rather than the
//! coding agent's plan/auto/yolo, because they are the vocabulary users of
//! these products already have.

use crate::backend::{AdoptConflict, Adopted, Backend, Session};
use crate::feed::EventFeed;
use crate::protocol::{
    studio_kinds, AppInfo, ModeInfo, NewSessionRequest, PermissionResponse, PromptBlock,
    SessionInfo, StoredSession, StudioEvent, SummaryOverride,
};
use anyhow::{bail, Result};
use async_trait::async_trait;
use cowork::session::{CoworkConfig, CoworkSession};
use cowork::steering::Steering;
use eventage::event::kinds;
use eventage::Event;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

pub struct CoworkBackend {
    settings: Arc<crate::model_settings::ModelSettings>,
    default_folder: String,
    max_parallel: usize,
    max_workstreams: usize,
}

impl CoworkBackend {
    pub fn new(
        settings: Arc<crate::model_settings::ModelSettings>,
        default_folder: String,
    ) -> Self {
        Self {
            settings,
            default_folder,
            max_parallel: 3,
            max_workstreams: 5,
        }
    }

    pub fn with_limits(mut self, parallel: usize, split: usize) -> Self {
        self.max_parallel = parallel;
        self.max_workstreams = split;
        self
    }
}

#[async_trait]
impl Backend for CoworkBackend {
    fn model_settings(&self) -> Option<Arc<crate::model_settings::ModelSettings>> {
        Some(Arc::clone(&self.settings))
    }

    fn info(&self) -> AppInfo {
        let model = self.settings.get();
        AppInfo {
            backend: "cowork",
            backend_detail: format!(
                "up to {} workstreams, {} at a time",
                self.max_workstreams, self.max_parallel
            ),
            model: model.model.clone(),
            provider: model.provider.label().to_string(),
            default_cwd: self.default_folder.clone(),
            modes: vec![
                ModeInfo {
                    id: Steering::Manual.id().into(),
                    label: "Manual".into(),
                    description: Steering::Manual.describe().into(),
                },
                ModeInfo {
                    id: Steering::Auto.id().into(),
                    label: "Auto".into(),
                    description: Steering::Auto.describe().into(),
                },
                ModeInfo {
                    id: Steering::Skip.id().into(),
                    label: "Skip".into(),
                    description: Steering::Skip.describe().into(),
                },
            ],
            version: env!("CARGO_PKG_VERSION"),
            full_trace: true,
            // The same profile field, for the same reason. `api_key` is not
            // the test: the keyless fallback fills in a placeholder.
            credentials_hint: match model.credentialed {
                false => Some(format!(
                    "No API key found, so cowork is pointed at {}. Set ANTHROPIC_API_KEY, \
                     QWEN_API_KEY, or OPENAI_API_KEY (with OPENAI_BASE_URL) and restart.",
                    model.base_url()
                )),
                true => None,
            },
        }
    }

    async fn open(&self, req: NewSessionRequest) -> Result<Arc<dyn Session>> {
        let folder = req
            .cwd
            .clone()
            .unwrap_or_else(|| self.default_folder.clone());
        let mut config = CoworkConfig::new(&folder);
        config.max_parallel = self.max_parallel;
        config.max_workstreams = self.max_workstreams;
        if let Some(mode) = req.mode.as_deref().and_then(Steering::from_id) {
            config.steering = mode;
        }

        let llm = eventage_code::agent::provider_for(&self.settings.get());
        let session = match req.resume.as_deref() {
            Some(id) => CoworkSession::resume(id.to_string(), llm, config).await?,
            None => CoworkSession::open(uuid::Uuid::new_v4().to_string(), llm, config).await?,
        };
        let id = session.id.clone();
        Ok(CoworkStudioSession::start(id, session, folder).await)
    }

    async fn branch(&self, _source: &dyn Session, _from_seq: u64) -> Result<Arc<dyn Session>> {
        // Cowork already branches — that is what a workstream is. Forking the
        // *session* on top would be a second, unrelated meaning of the word
        // in the same interface, which is worse than not offering it.
        bail!("cowork branches into workstreams within a session; fork one of those instead")
    }

    async fn stored(&self) -> Vec<StoredSession> {
        Vec::new()
    }

    async fn forget(&self, _id: &str) -> Result<()> {
        Ok(())
    }
}

struct CoworkStudioSession {
    id: String,
    folder: String,
    session: Arc<CoworkSession>,
    feed: Arc<EventFeed>,
    created_at: String,
    turn: Mutex<Option<JoinHandle<()>>>,
    bridge: JoinHandle<()>,
    /// Runs goals published by the channel or the scheduler.
    requests: JoinHandle<()>,
}

impl CoworkStudioSession {
    async fn start(id: String, session: CoworkSession, folder: String) -> Arc<dyn Session> {
        let session = Arc::new(session);
        let feed = Arc::new(EventFeed::new());

        // Subscribed before the log is read, so an event published in between
        // is seen twice rather than missed; the duplicate is dropped by id.
        let mut rx = session.bus.subscribe();
        let already: std::collections::HashSet<uuid::Uuid> =
            session.bus.log().await.iter().map(|e| e.id).collect();
        for event in session.bus.log().await {
            feed.push(StudioEvent::from(&event));
        }

        // Goals that arrive over cowork's own HTTP channel or from an
        // automation run through here; Studio's own `prompt` calls `run`
        // directly and publishes no request, so nothing is run twice.
        let requests = tokio::spawn(Arc::clone(&session).serve_requests());

        let bridge = {
            let feed = Arc::clone(&feed);
            let mut pending = already;
            tokio::spawn(async move {
                while let Some(event) = rx.recv().await {
                    if !pending.is_empty() && pending.remove(&event.id) {
                        continue;
                    }
                    feed.push(StudioEvent::from(&event));
                }
            })
        };

        Arc::new(Self {
            id,
            folder,
            session,
            feed,
            created_at: chrono::Utc::now().to_rfc3339(),
            turn: Mutex::new(None),
            bridge,
            requests,
        })
    }
}

#[async_trait]
impl Session for CoworkStudioSession {
    fn feed(&self) -> Arc<EventFeed> {
        Arc::clone(&self.feed)
    }

    fn info(&self) -> SessionInfo {
        SessionInfo {
            id: self.id.clone(),
            cwd: self.folder.clone(),
            mode: self.session.steering().id().to_string(),
            title: self
                .feed
                .first_user_text()
                .unwrap_or_else(|| "New session".into()),
            created_at: self.created_at.clone(),
            running: self
                .turn
                .try_lock()
                .map(|t| t.as_ref().is_some_and(|h| !h.is_finished()))
                // Locked means something is holding it, which only the turn
                // machinery does. Reporting "running" is the safe answer.
                .unwrap_or(true),
            turns: self.feed.count_turns(),
        }
    }

    async fn prompt(&self, blocks: Vec<PromptBlock>) -> Result<()> {
        let mut turn = self.turn.lock().await;
        if turn.as_ref().is_some_and(|h| !h.is_finished()) {
            bail!("this session is already working on something");
        }

        let goal: String = blocks
            .iter()
            .filter_map(|b| match b {
                PromptBlock::Text { text } => Some(text.clone()),
                // An image is a fine thing to hand a coding agent and a poor
                // thing to hand a planner: the goal has to be words.
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if goal.trim().is_empty() {
            bail!("say what to work on");
        }

        // The goal is echoed as an ordinary user message so the transcript
        // reads as a conversation, before the fan-out fills the trace.
        self.session
            .bus
            .publish(Event::new(kinds::USER_MESSAGE, json!({ "text": goal })))
            .await?;

        let session = Arc::clone(&self.session);
        *turn = Some(tokio::spawn(async move {
            let outcome = session.run(&goal).await;
            // Announced through the bus rather than pushed at the feed. The
            // feed has two producers — this task and the bridge — and pushing
            // here raced the events it is meant to follow, so a turn could
            // report itself finished before its own last result arrived.
            let payload = match &outcome {
                Ok(streams) => json!({
                    "workstreams": streams.len(),
                    "changed": streams.iter().map(|s| s.changes.len()).sum::<usize>(),
                }),
                Err(e) => json!({ "error": format!("{e:#}") }),
            };
            let kind = match outcome.is_ok() {
                true => studio_kinds::TURN_ENDED,
                false => studio_kinds::TURN_FAILED,
            };
            session.bus.broadcast(Event::new(kind, payload));
        }));
        Ok(())
    }

    async fn interrupt(&self) -> Result<()> {
        // Aborting drops the workstream futures, and with them their in-flight
        // model requests. Nothing has been written to the folder — results
        // live in the shadow repository until adopted — so a stopped turn
        // leaves the user's files exactly as they were.
        if let Some(handle) = self.turn.lock().await.take() {
            handle.abort();
        }
        self.session
            .bus
            .broadcast(Event::new(studio_kinds::TURN_INTERRUPTED, json!({})));
        Ok(())
    }

    async fn set_mode(&self, mode: &str) -> Result<()> {
        let Some(steering) = Steering::from_id(mode) else {
            bail!("unknown steering '{mode}' (expected {})", Steering::NAMES);
        };
        self.session.steer(steering).await;
        Ok(())
    }

    async fn rewind(&self, _turns: usize, _to: Option<&str>) -> Result<usize> {
        // Undoing a cowork turn is `revert`, which puts the *folder* back to
        // the session's base — a different and much stronger thing than
        // dropping conversation events, and not something to do behind a
        // control labelled for the latter.
        let restored = self.session.revert().await?;
        self.session.bus.broadcast(Event::new(
            studio_kinds::REWOUND,
            json!({
                "restored": restored.len(),
                "paths": restored.iter().map(|c| c.path.clone()).collect::<Vec<_>>(),
            }),
        ));
        Ok(0)
    }

    async fn override_summary(&self, _replacement: SummaryOverride) -> Result<()> {
        bail!("cowork workstreams are short-lived and are not summarised")
    }

    async fn permission(&self, response: PermissionResponse) -> Result<()> {
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

    async fn adopt(&self, workstream_id: &str, override_conflicts: bool) -> Result<Adopted> {
        let outcome = match override_conflicts {
            true => self.session.adopt_overriding(workstream_id).await?,
            false => self.session.adopt(workstream_id).await?,
        };
        Ok(Adopted {
            changed: outcome.applied.into_iter().map(|c| c.path).collect(),
            conflicts: outcome
                .conflicts
                .into_iter()
                .map(|c| AdoptConflict {
                    path: c.path,
                    workstream: c.workstream.as_str().to_string(),
                    live: c.live.as_str().to_string(),
                })
                .collect(),
        })
    }

    async fn seal(&self, workstream_id: &str, why: &str) -> Result<()> {
        self.session.seal(workstream_id, why).await
    }

    async fn shutdown(&self) {
        if let Some(handle) = self.turn.lock().await.take() {
            handle.abort();
        }
        self.bridge.abort();
        self.requests.abort();
        self.feed.close();
        // Waits for the log rather than dropping the task with it: the events
        // still queued are the ones a resume needs most.
        let failures = self.session.close().await;
        if failures > 0 {
            tracing::warn!(failures, "this session will reopen incomplete");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eventage::llm::MockLlmProvider;

    async fn session() -> Option<(tempfile::TempDir, Arc<dyn Session>)> {
        let folder = tempfile::tempdir().unwrap();
        std::fs::write(folder.path().join("notes.md"), "notes\n").unwrap();
        let state = tempfile::tempdir().unwrap();

        let mut config = CoworkConfig::new(folder.path());
        config.state_dir = state.path().to_path_buf();
        std::mem::forget(state);

        let inner = CoworkSession::open(
            "studio-test",
            Arc::new(MockLlmProvider::with_texts(vec!["[]", "done"])),
            config,
        )
        .await
        .ok()?;
        let studio = CoworkStudioSession::start(
            "studio-test".into(),
            inner,
            folder.path().display().to_string(),
        )
        .await;
        Some((folder, studio))
    }

    #[tokio::test]
    async fn the_steering_modes_are_coworks_own() {
        // Not the coding agent's plan/auto/yolo. These are the words users of
        // Cowork and the Codex app already have for the same control.
        let dir = tempfile::tempdir().unwrap();
        let settings = Arc::new(
            crate::model_settings::ModelSettings::load(
                eventage_code::config::ModelConfig::from_env(Some("m".into())),
                dir.path(),
            )
            .await,
        );
        let backend = CoworkBackend::new(settings, "/tmp".into());
        let ids: Vec<String> = backend.info().modes.iter().map(|m| m.id.clone()).collect();
        assert_eq!(ids, vec!["manual", "auto", "skip"]);
        assert!(backend.info().full_trace);
    }

    #[tokio::test]
    async fn setting_an_unknown_steering_is_refused_rather_than_ignored() {
        let Some((_folder, studio)) = session().await else {
            return;
        };
        assert!(studio.set_mode("yolo").await.is_err());
        assert!(studio.set_mode("skip").await.is_ok());
        assert_eq!(studio.info().mode, "skip");
    }

    #[tokio::test]
    async fn an_empty_goal_is_refused_before_anything_is_snapshotted() {
        let Some((_folder, studio)) = session().await else {
            return;
        };
        let err = studio
            .prompt(vec![PromptBlock::Text { text: "   ".into() }])
            .await;
        assert!(
            err.is_err(),
            "an empty goal started a session's worth of work"
        );
    }

    #[tokio::test]
    async fn the_goal_reaches_the_transcript_as_a_user_message() {
        // The fan-out fills the trace; the transcript still has to read as a
        // conversation, or the session looks like it started on its own.
        let Some((_folder, studio)) = session().await else {
            return;
        };
        studio
            .prompt(vec![PromptBlock::Text {
                text: "tidy the notes".into(),
            }])
            .await
            .unwrap();

        // Give the feed a moment to receive it through the bridge.
        for _ in 0..50 {
            if studio.feed().first_user_text().is_some() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        assert_eq!(
            studio.feed().first_user_text().as_deref(),
            Some("tidy the notes")
        );
        studio.shutdown().await;
    }
}
