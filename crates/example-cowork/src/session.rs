//! A working session: a goal, the workstreams it fans into, and what is kept.
//!
//! The loop is: snapshot the folder, split the goal, run the parts in parallel
//! against independent copies, then compare and keep. Each part is an ordinary
//! agent on its own bus; what makes the session navigable is that every one of
//! them branches from the *same* recorded base, so their results are
//! comparable and any of them can be thrown away without disturbing the rest.
//!
//! Abandoning is a first-class outcome rather than an error. A sealed
//! workstream's trajectory stays in the DAG as a rejected branch — the bus
//! rolls back to where it started rather than deleting — so a later attempt is
//! told what was already tried. That is the part neither Cowork nor the Codex
//! app has: both let you reject a result, and then the reasoning that produced
//! it is simply gone.
//!
//! Parallelism is bounded and the bound is the point. The recurring criticism
//! of these products is that fanning out burns the token budget, so
//! [`CoworkConfig::max_parallel`] is explicit, defaulted low, and every
//! workstream carries its own budget hook.

use crate::kinds;
use crate::shadow::{FileChange, Shadow};
use crate::steering::{SharedSteering, Steering, SteeringGate};
use anyhow::{Context, Result};
use eventage::agent::{DefaultContextAssembler, TokenBudgetHook};
use eventage::event::kinds as ev;
use eventage::llm::LlmProvider;
use eventage::{AgentBuilder, Event, EventBus, ReactStrategy};
use eventage_code::lsp::LspPool;
use eventage_code::workspace::Workspace;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};

/// How a session is set up.
#[derive(Debug, Clone)]
pub struct CoworkConfig {
    /// The folder the user granted.
    pub folder: PathBuf,
    /// Where snapshots, worktrees and the event log live.
    pub state_dir: PathBuf,
    /// How much happens without asking.
    pub steering: Steering,
    /// How many workstreams may run **at once**.
    ///
    /// Low on purpose. Fanning out is the expensive thing these products do,
    /// and the complaint that follows every one of them is the token bill —
    /// so the number is visible and chosen rather than discovered on an
    /// invoice.
    pub max_parallel: usize,
    /// How many workstreams a goal may be split **into**.
    ///
    /// Deliberately not the same number. These were one field, and the effect
    /// was that lowering concurrency silently discarded parts of the plan:
    /// asking for one at a time turned a two-part goal into a one-part goal
    /// and quietly dropped half the work. How much a goal divides into is a
    /// property of the goal; how much runs together is a property of the
    /// budget.
    pub max_workstreams: usize,
    /// Tokens one workstream may spend before it is stopped.
    pub token_budget: u64,
}

impl CoworkConfig {
    pub fn new(folder: impl Into<PathBuf>) -> Self {
        let folder = folder.into();
        Self {
            state_dir: dirs::data_dir()
                .unwrap_or_else(std::env::temp_dir)
                .join("cowork"),
            folder,
            steering: Steering::Auto,
            max_parallel: 3,
            max_workstreams: 5,
            token_budget: 200_000,
        }
    }
}

/// One line of work, on its own copy of the folder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workstream {
    pub id: String,
    pub title: String,
    /// What this stream was asked to do, in its own words to itself.
    pub brief: String,
    pub status: Status,
    /// The snapshot it produced, once it has finished.
    pub commit: Option<String>,
    /// Its own account of what it did.
    pub report: Option<String>,
    /// What it changed, against the session's base.
    pub changes: Vec<String>,
    /// Why it was abandoned, if it was.
    pub epitaph: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Planned,
    Running,
    Done,
    /// Abandoned, and kept in the graph as something already tried.
    Sealed,
    Failed,
}

/// What a plan asks for, before any of it has run.
#[derive(Debug, Clone, Deserialize)]
struct Brief {
    title: String,
    brief: String,
}

/// A working session.
pub struct CoworkSession {
    pub id: String,
    pub bus: EventBus,
    shadow: Arc<Shadow>,
    llm: Arc<dyn LlmProvider>,
    config: CoworkConfig,
    steering: Arc<SharedSteering>,
    /// The snapshot every workstream branches from.
    base: Mutex<Option<String>>,
    workstreams: Mutex<Vec<Workstream>>,
}

impl CoworkSession {
    /// Open a session over a folder.
    ///
    /// The shadow repository is created if it is not there. What it declines
    /// to track is announced immediately rather than discovered later by a
    /// rewind that skipped half the folder.
    pub async fn open(
        id: impl Into<String>,
        llm: Arc<dyn LlmProvider>,
        config: CoworkConfig,
    ) -> Result<Self> {
        let id = id.into();
        let session_state = config.state_dir.join(&id);
        tokio::fs::create_dir_all(&session_state).await.ok();

        let (shadow, excluded) =
            Shadow::open(session_state.join("shadow.git"), &config.folder).await?;
        let bus = EventBus::new();

        if !excluded.repositories.is_empty() || excluded.scan_truncated {
            bus.publish(Event::new(
                kinds::NOT_TRACKED,
                json!({
                    "repositories": excluded.repositories,
                    "scan_truncated": excluded.scan_truncated,
                    "detail": "these have their own history and their own undo, so cowork \
                               neither snapshots nor restores them",
                }),
            ))
            .await
            .ok();
        }

        let steering = Arc::new(SharedSteering::new(config.steering));
        crate::steering::announce(&bus, config.steering).await;

        Ok(Self {
            id,
            bus,
            shadow: Arc::new(shadow),
            llm,
            config,
            steering,
            base: Mutex::new(None),
            workstreams: Mutex::new(Vec::new()),
        })
    }

    /// Change how much happens without asking, mid-session.
    pub async fn steer(&self, mode: Steering) {
        self.steering.set(mode);
        crate::steering::announce(&self.bus, mode).await;
    }

    pub fn steering(&self) -> Steering {
        self.steering.get()
    }

    pub async fn workstreams(&self) -> Vec<Workstream> {
        self.workstreams.lock().await.clone()
    }

    /// Take on a goal: snapshot, plan, and run the parts.
    pub async fn run(&self, goal: &str) -> Result<Vec<Workstream>> {
        self.bus
            .publish(Event::new(kinds::GOAL_SET, json!({ "goal": goal })))
            .await?;

        // Everything branches from here, so it is recorded before anything
        // has a chance to change the folder.
        let base = self.shadow.snapshot("base").await?;
        *self.base.lock().await = Some(base.clone());

        let briefs = self.plan(goal).await;
        self.bus
            .publish(Event::new(
                kinds::PLAN_PROPOSED,
                json!({
                    "goal": goal,
                    "base": base,
                    "workstreams": briefs.iter().map(|b| json!({
                        "title": b.title, "brief": b.brief
                    })).collect::<Vec<_>>(),
                }),
            ))
            .await?;

        let planned: Vec<Workstream> = briefs
            .into_iter()
            .map(|b| Workstream {
                id: uuid::Uuid::new_v4().to_string()[..8].to_string(),
                title: b.title,
                brief: b.brief,
                status: Status::Planned,
                commit: None,
                report: None,
                changes: Vec::new(),
                epitaph: None,
            })
            .collect();
        *self.workstreams.lock().await = planned.clone();

        // Bounded, and the bound is deliberate — see `max_parallel`.
        let limit = Arc::new(tokio::sync::Semaphore::new(self.config.max_parallel.max(1)));
        let mut running = tokio::task::JoinSet::new();
        for stream in planned {
            let permit = Arc::clone(&limit);
            let base = base.clone();
            let shadow = Arc::clone(&self.shadow);
            let llm = Arc::clone(&self.llm);
            let steering = Arc::clone(&self.steering);
            let session_bus = self.bus.clone();
            let state = self.config.state_dir.join(&self.id);
            let budget = self.config.token_budget;
            running.spawn(async move {
                let _slot = permit.acquire().await.expect("the limiter stays open");
                run_workstream(
                    stream,
                    base,
                    shadow,
                    llm,
                    steering,
                    session_bus,
                    state,
                    budget,
                )
                .await
            });
        }

        let mut finished = Vec::new();
        while let Some(joined) = running.join_next().await {
            match joined {
                Ok(stream) => finished.push(stream),
                Err(e) => warn!("a workstream task panicked: {e}"),
            }
        }
        finished.sort_by(|a, b| a.title.cmp(&b.title));
        *self.workstreams.lock().await = finished.clone();
        Ok(finished)
    }

    /// Split a goal into workstreams.
    ///
    /// Falls back to a single stream carrying the goal verbatim when the model
    /// does not answer with something parseable. A session that refused to
    /// start because the planning step returned prose would be worse than one
    /// that simply does the work in one piece.
    async fn plan(&self, goal: &str) -> Vec<Brief> {
        let single = || {
            vec![Brief {
                title: "the task".into(),
                brief: goal.to_string(),
            }]
        };

        let prompt = format!(
            "You are planning a piece of work in a folder of documents.\n\n\
             GOAL: {goal}\n\n\
             Split this into at most {} independent workstreams that can run at the same \
             time without needing each other's output. Independent means they touch \
             different files, or answer different questions. If the goal is genuinely one \
             indivisible piece of work, return exactly one workstream — that is a good \
             answer, not a failure.\n\n\
             Reply with JSON only, no prose, no code fence:\n\
             [{{\"title\": \"short label\", \"brief\": \"what this stream should do, in full\"}}]",
            self.config.max_workstreams
        );

        let response = match self
            .llm
            .complete(vec![eventage::llm::ChatMessage::user(&prompt)], vec![])
            .await
        {
            Ok(r) => r,
            Err(e) => {
                warn!("could not plan the goal, running it as one piece: {e}");
                return single();
            }
        };

        let Some(text) = response.content else {
            return single();
        };
        match parse_plan(&text) {
            Some(briefs) if !briefs.is_empty() => briefs
                .into_iter()
                .take(self.config.max_workstreams.max(1))
                .collect(),
            _ => {
                warn!("the plan was not parseable JSON, running the goal as one piece");
                single()
            }
        }
    }

    /// Apply one workstream's result to the folder.
    pub async fn adopt(&self, workstream_id: &str) -> Result<Vec<FileChange>> {
        let base = self
            .base
            .lock()
            .await
            .clone()
            .context("this session has not run anything yet")?;
        let streams = self.workstreams.lock().await.clone();
        let stream = streams
            .iter()
            .find(|w| w.id == workstream_id)
            .context("no workstream with that id")?;
        let commit = stream
            .commit
            .as_ref()
            .context("that workstream produced nothing to adopt")?;

        let changes = self.shadow.adopt(commit, &base).await?;
        self.bus
            .publish(Event::new(
                kinds::ADOPTED,
                json!({
                    "id": stream.id,
                    "title": stream.title,
                    "changes": changes.iter().map(|c| json!({
                        "path": c.path, "status": c.status.as_str()
                    })).collect::<Vec<_>>(),
                }),
            ))
            .await?;
        Ok(changes)
    }

    /// Abandon a workstream, and record why.
    ///
    /// The reasoning is not deleted. Cowork and the Codex app both let you
    /// reject a result, and at that point what produced it is gone; here it
    /// stays in the graph, and a later attempt is told about it.
    pub async fn seal(&self, workstream_id: &str, why: &str) -> Result<()> {
        let mut streams = self.workstreams.lock().await;
        let stream = streams
            .iter_mut()
            .find(|w| w.id == workstream_id)
            .context("no workstream with that id")?;
        stream.status = Status::Sealed;
        stream.epitaph = Some(why.to_string());

        self.bus
            .publish(Event::new(
                kinds::WORKSTREAM_SEALED,
                json!({ "id": stream.id, "title": stream.title, "epitaph": why }),
            ))
            .await?;
        Ok(())
    }

    /// What has already been tried and abandoned, for a later attempt to read.
    pub async fn lessons(&self) -> Vec<String> {
        self.workstreams
            .lock()
            .await
            .iter()
            .filter(|w| w.status == Status::Sealed)
            .filter_map(|w| w.epitaph.as_ref().map(|e| format!("{}: {e}", w.title)))
            .collect()
    }

    /// Put the folder back to the state the session started from.
    pub async fn revert(&self) -> Result<Vec<FileChange>> {
        let base = self
            .base
            .lock()
            .await
            .clone()
            .context("this session has not run anything yet")?;
        self.shadow.restore(&base).await
    }
}

/// Pull a JSON array out of a model's reply.
///
/// Models fence JSON, prefix it with a sentence, or do both, whatever the
/// prompt says. Finding the array is a two-line job and refusing on a code
/// fence would be a strange thing to be strict about.
fn parse_plan(text: &str) -> Option<Vec<Brief>> {
    let start = text.find('[')?;
    let end = text.rfind(']')?;
    if end <= start {
        return None;
    }
    serde_json::from_str(&text[start..=end]).ok()
}

/// Run one workstream to completion in its own copy of the folder.
#[allow(clippy::too_many_arguments)]
async fn run_workstream(
    mut stream: Workstream,
    base: String,
    shadow: Arc<Shadow>,
    llm: Arc<dyn LlmProvider>,
    steering: Arc<SharedSteering>,
    session_bus: EventBus,
    state_dir: PathBuf,
    token_budget: u64,
) -> Workstream {
    let worktree = state_dir.join("workstreams").join(&stream.id);
    tokio::fs::create_dir_all(worktree.parent().unwrap())
        .await
        .ok();

    if let Err(e) = shadow.worktree(&worktree, &base).await {
        warn!(id = %stream.id, "could not create a working copy: {e:#}");
        stream.status = Status::Failed;
        stream.report = Some(format!("could not create a working copy: {e:#}"));
        return stream;
    }

    stream.status = Status::Running;
    let _ = session_bus
        .publish(Event::new(
            kinds::WORKSTREAM_STARTED,
            json!({
                "id": stream.id,
                "title": stream.title,
                "brief": stream.brief,
                "worktree": worktree.display().to_string(),
            }),
        ))
        .await;

    let outcome = work(
        &stream,
        &worktree,
        Arc::clone(&llm),
        Arc::clone(&steering),
        session_bus.clone(),
        token_budget,
    )
    .await;

    match outcome {
        Ok(report) => {
            stream.report = Some(report);
            match shadow
                .snapshot_tree(&worktree, &format!("ws-{}", stream.id))
                .await
            {
                Ok(commit) => {
                    let changes = shadow.diff(&base, &commit).await.unwrap_or_default();
                    stream.changes = changes.iter().map(|c| c.path.clone()).collect();
                    stream.commit = Some(commit);
                    stream.status = Status::Done;
                    let _ = session_bus
                        .publish(Event::new(
                            kinds::WORKSTREAM_FINISHED,
                            json!({
                                "id": stream.id,
                                "title": stream.title,
                                "report": stream.report,
                                "commit": stream.commit,
                                "changes": changes.iter().map(|c| json!({
                                    "path": c.path, "status": c.status.as_str()
                                })).collect::<Vec<_>>(),
                            }),
                        ))
                        .await;
                }
                Err(e) => {
                    warn!(id = %stream.id, "could not record the result: {e:#}");
                    stream.status = Status::Failed;
                }
            }
        }
        Err(e) => {
            warn!(id = %stream.id, "workstream failed: {e:#}");
            stream.status = Status::Failed;
            stream.report = Some(format!("{e:#}"));
        }
    }

    // The checkout has served its purpose; the result is in the snapshot.
    let _ = shadow.remove_worktree(&worktree).await;
    stream
}

/// Build an agent over the working copy and run the brief.
async fn work(
    stream: &Workstream,
    worktree: &PathBuf,
    llm: Arc<dyn LlmProvider>,
    steering: Arc<SharedSteering>,
    session_bus: EventBus,
    token_budget: u64,
) -> Result<String> {
    let ws = Arc::new(Workspace::open(worktree)?);
    // No language server for a folder of documents; the editing tools take
    // one and simply have nothing to tell it.
    let lsp = Arc::new(LspPool::new(worktree));

    // Its own bus, so the workstreams do not share a conversation, bridged
    // to the session's so the user still sees one trace. Only the events a
    // watcher needs cross over: forwarding everything would put each stream's
    // whole context assembly into the session log.
    let bus = EventBus::new();
    let bridge = tokio::spawn({
        let bus = bus.clone();
        let target = session_bus.clone();
        async move {
            let _ = eventage::WorkerSet::new()
                .add_worker(eventage::BusBridge::new(target).filter_kinds(vec![
                    ev::ASSISTANT_MESSAGE.to_string(),
                    ev::TOOL_CALL_PROPOSED.to_string(),
                    ev::TOOL_RESULT.to_string(),
                    ev::AGENT_STUCK.to_string(),
                ]))
                .run_on(bus)
                .await;
        }
    });

    let system = format!(
        "You are one workstream in a larger piece of work, working in your own private \
         copy of the user's folder. Do only what your brief asks; another stream is \
         handling the rest.\n\n\
         Your brief: {}\n\n\
         Everything you write lands in your copy and is reviewed against the others \
         before anything reaches the user's folder, so work concretely rather than \
         proposing. When you are done, report what you changed and what you did not, \
         plainly. If the brief turns out to be the wrong thing to do, say so and stop \
         rather than doing it anyway — being told an approach was wrong is more useful \
         than being handed the wrong result.",
        stream.brief
    );

    let agent = AgentBuilder::new()
        .agent_id(format!("ws-{}", stream.id))
        .bus(bus.clone())
        .llm_arc(Arc::clone(&llm))
        .context(DefaultContextAssembler::new(system))
        .hook(SteeringGate::new(session_bus.clone(), steering))
        // The bound that keeps a fan-out from becoming an invoice.
        .hook(TokenBudgetHook::new(token_budget))
        .tool(eventage_code::tools::ReadFile {
            ws: Arc::clone(&ws),
            client: None,
        })
        .tool(eventage_code::tools::WriteFile {
            ws: Arc::clone(&ws),
            client: None,
            lsp: Arc::clone(&lsp),
        })
        .tool(eventage_code::tools::EditFile {
            ws: Arc::clone(&ws),
            client: None,
            lsp: Arc::clone(&lsp),
        })
        .tool(eventage_code::tools::MultiEdit {
            ws: Arc::clone(&ws),
            client: None,
            lsp: Arc::clone(&lsp),
        })
        .tool(eventage_code::tools::Glob {
            ws: Arc::clone(&ws),
        })
        .tool(eventage_code::tools::Grep {
            ws: Arc::clone(&ws),
        })
        .tool(eventage_code::tools::ListDirectory {
            ws: Arc::clone(&ws),
        })
        .tool(eventage::WebSearchTool::new())
        .tool(eventage::WebFetchTool::new())
        .strategy(ReactStrategy::default())
        .build();

    bus.publish(Event::new(
        ev::USER_MESSAGE,
        json!({ "text": stream.brief }),
    ))
    .await?;
    agent.cycle().await?;

    // The bridge drains when the bus closes; the workstream is over either
    // way, so it is not waited on beyond its own shutdown.
    bus.close();
    let _ = tokio::time::timeout(std::time::Duration::from_secs(2), bridge).await;

    let log = bus.log().await;
    let report = log
        .iter()
        .rev()
        .find(|e| e.kind == ev::ASSISTANT_MESSAGE)
        .and_then(|e| e.payload.get("content").and_then(|c| c.as_str()))
        .unwrap_or("the workstream ended without a report")
        .to_string();
    info!(id = %stream.id, "workstream finished");
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eventage::llm::MockLlmProvider;

    #[test]
    fn a_plan_survives_a_code_fence_and_a_preamble() {
        // Models fence JSON and introduce it whatever the prompt says.
        // Refusing on that would be a strange thing to be strict about.
        let fenced = "Here is the plan:\n```json\n[{\"title\":\"a\",\"brief\":\"do a\"},\
                      {\"title\":\"b\",\"brief\":\"do b\"}]\n```\nHope that helps!";
        let plan = parse_plan(fenced).expect("the array was found");
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].title, "a");
        assert_eq!(plan[1].brief, "do b");
    }

    #[test]
    fn prose_is_not_mistaken_for_a_plan() {
        assert!(parse_plan("I think we should start by reading the files.").is_none());
        assert!(parse_plan("[not json at all]").is_none());
    }

    async fn session(folder: &tempfile::TempDir, replies: Vec<&str>) -> Option<CoworkSession> {
        let state = tempfile::tempdir().unwrap();
        let mut config = CoworkConfig::new(folder.path());
        config.state_dir = state.path().to_path_buf();
        // Leaked deliberately: the session outlives this handle and the files
        // must still be there when it runs.
        std::mem::forget(state);
        CoworkSession::open(
            "test-session",
            Arc::new(MockLlmProvider::with_texts(replies)),
            config,
        )
        .await
        .ok()
    }

    #[tokio::test]
    async fn an_unplannable_goal_still_runs_as_one_piece() {
        // A session that refused to start because the planning step returned
        // prose would be worse than one that simply does the work in a single
        // stream.
        let folder = tempfile::tempdir().unwrap();
        std::fs::write(folder.path().join("notes.md"), "some notes\n").unwrap();
        let Some(session) = session(&folder, vec!["I'd rather just chat about it."]).await else {
            return;
        };
        let briefs = session.plan("tidy the notes").await;
        assert_eq!(briefs.len(), 1);
        assert_eq!(briefs[0].brief, "tidy the notes");
    }

    #[tokio::test]
    async fn opening_a_session_announces_what_it_will_not_track() {
        // A rewind that quietly skipped half the folder would be worse than
        // one that refused, so the exclusions are said out loud at the start.
        let folder = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(folder.path().join("checkout/.git")).unwrap();
        std::fs::write(folder.path().join("mine.md"), "mine\n").unwrap();

        let Some(session) = session(&folder, vec!["[]"]).await else {
            return;
        };
        let log = session.bus.log().await;
        let notice = log
            .iter()
            .find(|e| e.kind == kinds::NOT_TRACKED)
            .expect("the exclusion was announced");
        assert_eq!(notice.payload["repositories"], json!(["checkout"]));
    }

    #[tokio::test]
    async fn sealing_keeps_the_reason_for_a_later_attempt_to_read() {
        // The difference from rejecting a diff: what produced the result is
        // still there, and the next attempt is told about it.
        let folder = tempfile::tempdir().unwrap();
        let Some(session) = session(&folder, vec!["[]"]).await else {
            return;
        };
        session.workstreams.lock().await.push(Workstream {
            id: "abc".into(),
            title: "rewrite in the 2026 template".into(),
            brief: "…".into(),
            status: Status::Done,
            commit: None,
            report: None,
            changes: vec![],
            epitaph: None,
        });

        session
            .seal(
                "abc",
                "the 2026 template is for external reports, not internal ones",
            )
            .await
            .unwrap();

        let lessons = session.lessons().await;
        assert_eq!(lessons.len(), 1);
        assert!(lessons[0].contains("external reports"), "{lessons:?}");
        assert!(session
            .bus
            .log()
            .await
            .iter()
            .any(|e| e.kind == kinds::WORKSTREAM_SEALED));
    }

    #[tokio::test]
    async fn steering_can_be_changed_while_a_session_is_open() {
        let folder = tempfile::tempdir().unwrap();
        let Some(session) = session(&folder, vec!["[]"]).await else {
            return;
        };
        assert_eq!(session.steering(), Steering::Auto);
        session.steer(Steering::Manual).await;
        assert_eq!(session.steering(), Steering::Manual);

        let announced = session
            .bus
            .log()
            .await
            .iter()
            .filter(|e| e.kind == kinds::STEERING_CHANGED)
            .count();
        assert_eq!(
            announced, 2,
            "the change and the initial mode are both said"
        );
    }
}
