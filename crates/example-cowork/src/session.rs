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
use crate::shadow::{Adoption, FileChange, Shadow};
use crate::steering::{SharedSteering, Steering, SteeringGate};
use anyhow::{Context, Result};
use eventage::agent::{DefaultContextAssembler, TokenBudgetHook};
use eventage::event::kinds as ev;
use eventage::llm::LlmProvider;
use eventage::observability::BusObserver;
use eventage::sqlite::{SqliteEventStore, SqliteExporter};
use eventage::{AgentBuilder, Event, EventBus, ReactStrategy};
use eventage_code::lsp::LspPool;
use eventage_code::workspace::Workspace;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
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
    /// The task writing events to disk, and its running failure count.
    persistence: Mutex<Option<tokio::task::JoinHandle<usize>>>,
    export_failures: Arc<std::sync::atomic::AtomicUsize>,
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
        Self::build(id.into(), llm, config, false).await
    }

    /// Reopen a session recorded earlier, with its workstreams intact.
    ///
    /// Everything needed is already on the bus — that was the bet the whole
    /// design makes, and this is where it pays. The plan, each stream's
    /// identity and outcome, what it changed, why one was abandoned, and the
    /// base snapshot they all branch from were published as events because a
    /// surface needed to render them; the same events rebuild the session.
    /// There is no second copy of the state to keep in step.
    ///
    /// The snapshots survive too, because they are referenced under
    /// `refs/cowork/` rather than left dangling — an unreferenced commit is
    /// collectable, and a resumed session that offered to adopt a workstream
    /// whose tree `git gc` had taken would be worse than one that offered
    /// nothing.
    pub async fn resume(
        id: impl Into<String>,
        llm: Arc<dyn LlmProvider>,
        config: CoworkConfig,
    ) -> Result<Self> {
        Self::build(id.into(), llm, config, true).await
    }

    async fn build(
        id: String,
        llm: Arc<dyn LlmProvider>,
        config: CoworkConfig,
        restore: bool,
    ) -> Result<Self> {
        // The id becomes a directory name directly beneath the state
        // directory, so an id from an untrusted caller is a path traversal.
        if !eventage_code::config::is_valid_session_id(&id) {
            anyhow::bail!(
                "'{id}' is not a valid session id (expected a UUID); refusing to turn \
                 it into a file path"
            );
        }
        let session_state = config.state_dir.join(&id);
        tokio::fs::create_dir_all(&session_state).await.ok();

        let (shadow, excluded) =
            Shadow::open(session_state.join("shadow.git"), &config.folder).await?;
        let bus = EventBus::new();

        // ── persistence ──────────────────────────────────────────────────
        let db = session_state.join("events.db");
        let recorded = match restore {
            false => Vec::new(),
            true => {
                let saved = SqliteEventStore::new(&db).await?.load_all().await?;
                info!(events = saved.len(), "reopening cowork session");
                bus.restore_from(saved.clone()).await;
                saved
            }
        };

        // Subscribed here rather than inside the task: a subscription made
        // after the spawn misses whatever is published before the task is
        // first polled, and those are exactly the events a resume needs.
        let observer = BusObserver::new(bus.clone()).add_exporter(SqliteExporter::new(&db).await?);
        let events = observer.subscribe();
        let export_failures = observer.failures();
        let persistence = tokio::spawn(observer.run_with(events));

        let (restored_base, restored_streams, restored_steering) = rebuild(&recorded);

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

        // A resumed session keeps the mode it was left in; the configured
        // one is only a starting point for a session that has none.
        let mode = restored_steering.unwrap_or(config.steering);
        // Announced on the session bus, so Studio shows what is loaded rather
        // than leaving the user to infer it from a changed system prompt.
        let host = eventage_code::agent::load_plugins(&config.folder.display().to_string());
        if !host.plugins().is_empty() {
            eventage_code::agent::announce_plugins(&bus, &host).await;
        }

        let steering = Arc::new(SharedSteering::new(mode));
        crate::steering::announce(&bus, mode).await;

        if restore {
            info!(
                workstreams = restored_streams.len(),
                base = restored_base.is_some(),
                "cowork session reopened"
            );
        }

        Ok(Self {
            id,
            bus,
            shadow: Arc::new(shadow),
            llm,
            config,
            steering,
            base: Mutex::new(restored_base),
            workstreams: Mutex::new(restored_streams),
            persistence: Mutex::new(Some(persistence)),
            export_failures,
        })
    }

    /// How many events failed to reach the log.
    ///
    /// Non-zero means this session's history is incomplete and reopening it
    /// will be missing something.
    pub fn export_failures(&self) -> usize {
        self.export_failures
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Close the session and wait for its log to be written.
    ///
    /// Dropping the persistence task instead loses whatever it had not yet
    /// flushed — the last events of the session, which are exactly the ones a
    /// resume needs.
    pub async fn close(&self) -> usize {
        self.bus.close();
        let handle = self.persistence.lock().await.take();
        match handle {
            None => self.export_failures(),
            Some(task) => match tokio::time::timeout(Duration::from_secs(10), task).await {
                Ok(Ok(failures)) => failures,
                Ok(Err(e)) => {
                    warn!("the persistence task failed: {e}");
                    self.export_failures() + 1
                }
                Err(_) => {
                    warn!("persistence did not finish flushing within 10s");
                    self.export_failures() + 1
                }
            },
        }
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

        // What earlier attempts in this session tried and abandoned. Sealing
        // recorded a reason and nothing ever read it, which made the whole
        // mechanism decorative: the graph kept the evidence and the planner
        // went on proposing the same thing.
        let lessons = self.lessons().await;
        let briefs = self.plan(goal, &lessons).await;
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
        // Sealed streams are carried across rounds; everything else is
        // replaced. Wholesale replacement dropped them, so `lessons()` came
        // back empty on the second goal and the planner stopped being told
        // what had already failed — the feedback worked within one round and
        // silently stopped at the boundary, which is the worst place for it
        // to stop.
        //
        // Finished streams are *not* carried, and that is deliberate: each
        // round takes a fresh base snapshot, so an earlier result diffs
        // against a tree that no longer exists and adopting it would apply
        // the wrong change set.
        {
            let mut held = self.workstreams.lock().await;
            let carried: Vec<Workstream> = held
                .drain(..)
                .filter(|w| w.status == Status::Sealed)
                .collect();
            *held = carried.into_iter().chain(planned.clone()).collect();
        }

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
        {
            let mut held = self.workstreams.lock().await;
            let carried: Vec<Workstream> = held
                .drain(..)
                .filter(|w| w.status == Status::Sealed)
                .collect();
            *held = carried.into_iter().chain(finished.clone()).collect();
        }
        Ok(finished)
    }

    /// Split a goal into workstreams.
    ///
    /// Falls back to a single stream carrying the goal verbatim when the model
    /// does not answer with something parseable. A session that refused to
    /// start because the planning step returned prose would be worse than one
    /// that simply does the work in one piece.
    async fn plan(&self, goal: &str, lessons: &[String]) -> Vec<Brief> {
        let single = || {
            vec![Brief {
                title: "the task".into(),
                brief: goal.to_string(),
            }]
        };

        let prompt = self.planning_prompt(goal, lessons);

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

    /// The prompt the planner is given.
    ///
    /// Separate from the call so what the planner is told can be checked
    /// without spending a model request on it — the interesting part is the
    /// history, and a test that had to call a model to see it would not be
    /// written.
    fn planning_prompt(&self, goal: &str, lessons: &[String]) -> String {
        // Stated as findings rather than prohibitions. A sealed workstream
        // means "this was tried and was wrong", which is a fact about the
        // problem; turning it into "never do X" would outlive its reason.
        let history = match lessons.is_empty() {
            true => String::new(),
            false => format!(
                "ALREADY TRIED IN THIS SESSION, AND ABANDONED:\n{}\n\nDo not \
                 propose these again unless the goal has changed to ask for them. \
                 Say briefly how your plan differs.\n\n",
                lessons
                    .iter()
                    .map(|l| format!("- {l}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
        };

        format!(
            "You are planning a piece of work in a folder of documents.\n\n\
             GOAL: {goal}\n\n\
             {history}\
             Split this into at most {} independent workstreams that can run at the same \
             time without needing each other's output. Independent means they touch \
             different files, or answer different questions. If the goal is genuinely one \
             indivisible piece of work, return exactly one workstream — that is a good \
             answer, not a failure.\n\n\
             Reply with JSON only, no prose, no code fence:\n\
             [{{\"title\": \"short label\", \"brief\": \"what this stream should do, in full\"}}]",
            self.config.max_workstreams
        )
    }

    /// Apply one workstream's result to the folder.
    ///
    /// Refuses if the folder changed underneath the workstream — see
    /// [`Shadow::adopt`]. The refusal is published, because the user now has
    /// two versions of the same files and has to choose between them.
    pub async fn adopt(&self, workstream_id: &str) -> Result<Adoption> {
        self.adopt_with(workstream_id, false).await
    }

    /// Apply it anyway, discarding the folder's own changes to those paths.
    pub async fn adopt_overriding(&self, workstream_id: &str) -> Result<Adoption> {
        self.adopt_with(workstream_id, true).await
    }

    async fn adopt_with(&self, workstream_id: &str, override_conflicts: bool) -> Result<Adoption> {
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

        let outcome = match override_conflicts {
            true => self.shadow.adopt_overriding(commit, &base).await?,
            false => self.shadow.adopt(commit, &base).await?,
        };

        let event = match outcome.applied.is_empty() && !outcome.conflicts.is_empty() {
            true => Event::new(
                kinds::ADOPTION_BLOCKED,
                json!({
                    "id": stream.id,
                    "title": stream.title,
                    "conflicts": outcome.conflicts.iter().map(|c| json!({
                        "path": c.path,
                        "workstream": c.workstream.as_str(),
                        "live": c.live.as_str(),
                    })).collect::<Vec<_>>(),
                }),
            ),
            false => Event::new(
                kinds::ADOPTED,
                json!({
                    "id": stream.id,
                    "title": stream.title,
                    "overrode": override_conflicts,
                    "changes": outcome.applied.iter().map(|c| json!({
                        "path": c.path, "status": c.status.as_str()
                    })).collect::<Vec<_>>(),
                }),
            ),
        };
        self.bus.publish(event).await?;
        Ok(outcome)
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

    /// Run goals that arrive from somewhere other than the caller.
    ///
    /// The HTTP channel and the scheduler both publish
    /// [`GOAL_REQUESTED`](crate::kinds::GOAL_REQUESTED); until this existed,
    /// nothing consumed it. The endpoint answered "accepted" and did nothing,
    /// and an automation reported itself fired every time it came due without
    /// ever doing the work — both advertised, both hollow.
    ///
    /// Serialised deliberately. Requests queue behind whatever is running
    /// rather than fanning out on top of it: a session is one folder, and two
    /// goals adopting into it at once is not a thing to discover in
    /// production.
    ///
    /// Subscribes **before** returning its future, so a request published
    /// immediately after the call cannot be missed. Subscribing inside the
    /// spawned task instead left a window between `tokio::spawn` and the
    /// subscription in which a goal simply vanished — rare in a real session,
    /// where the consumer starts long before anyone asks for anything, and
    /// certain in a test that asks straight away.
    ///
    /// The returned future completes when the bus closes, so a caller can
    /// spawn it and abort the handle at shutdown.
    pub fn serve_requests(self: Arc<Self>) -> impl std::future::Future<Output = ()> + Send {
        let mut rx = self.bus.subscribe();
        async move { self.consume_requests(&mut rx).await }
    }

    async fn consume_requests(&self, rx: &mut eventage::bus::BusReceiver) {
        while let Some(event) = rx.recv().await {
            if event.kind != kinds::GOAL_REQUESTED {
                continue;
            }
            let Some(goal) = event
                .payload
                .get("goal")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .filter(|g| !g.trim().is_empty())
            else {
                warn!("a goal request arrived with nothing to do");
                continue;
            };
            let source = event
                .payload
                .get("source")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            info!(%source, "running a requested goal");
            if let Err(e) = self.run(&goal).await {
                warn!(%source, "the requested goal failed: {e:#}");
            }
        }
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

/// Rebuild a session's state from the events it recorded.
///
/// The reconstruction is possible because nothing was ever kept only in
/// memory: every fact a surface needed to render — the plan, each stream's
/// identity and outcome, what it changed, why one was abandoned — was
/// published, and the same events answer "what was this session doing?" for a
/// process that was not there.
///
/// Planned order is preserved separately from the map, for the same reason
/// the UI does it: streams *start* in whatever order the concurrency limiter
/// admits them, and a session that came back in a different order each time
/// would be disorienting for no reason.
fn rebuild(events: &[Event]) -> (Option<String>, Vec<Workstream>, Option<Steering>) {
    use std::collections::HashMap;

    let mut base = None;
    let mut steering = None;
    let mut by_id: HashMap<String, Workstream> = HashMap::new();
    let mut order: Vec<String> = Vec::new();

    let str_at = |e: &Event, key: &str| {
        e.payload
            .get(key)
            .and_then(|v| v.as_str())
            .map(str::to_string)
    };
    let place = |order: &mut Vec<String>, key: &str| {
        if !order.iter().any(|k| k == key) {
            order.push(key.to_string());
        }
    };

    for event in events {
        match event.kind.as_str() {
            kinds::PLAN_PROPOSED => {
                if let Some(commit) = str_at(event, "base") {
                    base = Some(commit);
                }
                let planned = event
                    .payload
                    .get("workstreams")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                for part in planned {
                    let Some(title) = part.get("title").and_then(|v| v.as_str()) else {
                        continue;
                    };
                    // A stream has no id until it starts, so the title is the
                    // only handle the plan gives.
                    let key = format!("planned:{title}");
                    place(&mut order, &key);
                    by_id.insert(
                        key,
                        Workstream {
                            id: String::new(),
                            title: title.to_string(),
                            brief: part
                                .get("brief")
                                .and_then(|v| v.as_str())
                                .unwrap_or_default()
                                .to_string(),
                            status: Status::Planned,
                            commit: None,
                            report: None,
                            changes: Vec::new(),
                            epitaph: None,
                        },
                    );
                }
            }
            kinds::WORKSTREAM_STARTED => {
                let (Some(id), Some(title)) = (str_at(event, "id"), str_at(event, "title")) else {
                    continue;
                };
                let placeholder = format!("planned:{title}");
                let planned = by_id.remove(&placeholder);
                match order.iter().position(|k| *k == placeholder) {
                    Some(at) => order[at] = id.clone(),
                    None => place(&mut order, &id),
                }
                by_id.insert(
                    id.clone(),
                    Workstream {
                        id,
                        title,
                        brief: str_at(event, "brief")
                            .or_else(|| planned.map(|p| p.brief))
                            .unwrap_or_default(),
                        // Running when the log ended means the process died
                        // mid-turn: it did not finish, and saying it did
                        // would offer a result that is not there.
                        status: Status::Failed,
                        commit: None,
                        report: Some("interrupted before it finished".into()),
                        changes: Vec::new(),
                        epitaph: None,
                    },
                );
            }
            kinds::WORKSTREAM_FINISHED => {
                let Some(id) = str_at(event, "id") else {
                    continue;
                };
                if let Some(stream) = by_id.get_mut(&id) {
                    stream.status = Status::Done;
                    stream.commit = str_at(event, "commit");
                    stream.report = str_at(event, "report");
                    stream.changes = event
                        .payload
                        .get("changes")
                        .and_then(|v| v.as_array())
                        .map(|cs| {
                            cs.iter()
                                .filter_map(|c| {
                                    c.get("path").and_then(|p| p.as_str()).map(str::to_string)
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                }
            }
            kinds::WORKSTREAM_FAILED => {
                let Some(id) = str_at(event, "id") else {
                    continue;
                };
                if let Some(stream) = by_id.get_mut(&id) {
                    stream.status = Status::Failed;
                    stream.report = str_at(event, "error");
                }
            }
            kinds::WORKSTREAM_SEALED => {
                let Some(id) = str_at(event, "id") else {
                    continue;
                };
                if let Some(stream) = by_id.get_mut(&id) {
                    stream.status = Status::Sealed;
                    stream.epitaph = str_at(event, "epitaph");
                }
            }
            kinds::STEERING_CHANGED => {
                if let Some(mode) = str_at(event, "steering")
                    .as_deref()
                    .and_then(Steering::from_id)
                {
                    steering = Some(mode);
                }
            }
            _ => {}
        }
    }

    let streams = order
        .iter()
        .filter_map(|key| by_id.remove(key))
        .collect::<Vec<_>>();
    (base, streams, steering)
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
        fail(
            &mut stream,
            &session_bus,
            format!("could not create a working copy: {e:#}"),
        )
        .await;
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
                    fail(
                        &mut stream,
                        &session_bus,
                        format!("could not record the result: {e:#}"),
                    )
                    .await;
                }
            }
        }
        Err(e) => {
            warn!(id = %stream.id, "workstream failed: {e:#}");
            fail(&mut stream, &session_bus, format!("{e:#}")).await;
        }
    }

    // The checkout has served its purpose; the result is in the snapshot.
    let _ = shadow.remove_worktree(&worktree).await;
    stream
}

/// Record a workstream's failure, on the bus as well as in memory.
async fn fail(stream: &mut Workstream, bus: &EventBus, why: String) {
    stream.status = Status::Failed;
    stream.report = Some(why.clone());
    let _ = bus
        .publish(Event::new(
            kinds::WORKSTREAM_FAILED,
            json!({ "id": stream.id, "title": stream.title, "error": why }),
        ))
        .await;
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

    // Plugins apply here as much as to a coding session: a house-style
    // plugin is as relevant to writing a report as to writing code. Installed
    // per workstream because each has its own registry.
    let plugin_tools = eventage::agent::ToolRegistry::new();
    let host = eventage_code::agent::load_plugins(&worktree.display().to_string());
    let mut system = system;
    if !host.plugins().is_empty() {
        match host.install(&plugin_tools).await {
            Ok(fragment) if !fragment.trim().is_empty() => {
                system.push_str("\n\n");
                system.push_str(fragment.trim());
            }
            Ok(_) => {}
            Err(e) => warn!("could not install a plugin: {e:#}"),
        }
    }

    let mut agent = AgentBuilder::new()
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
        .strategy(ReactStrategy::default());
    for tool in plugin_tools.all_tools() {
        agent = agent.tool_arc(tool);
    }
    let agent = agent.build();

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
    async fn what_was_abandoned_reaches_the_next_plan() {
        // Sealing recorded a reason and nothing read it, so the graph kept
        // the evidence and the planner went on proposing the same thing. The
        // lesson is now in the prompt the planner sees.
        let folder = tempfile::tempdir().unwrap();
        let Some(session) = session(&folder, vec!["[]", "[]"]).await else {
            return;
        };
        session.workstreams.lock().await.push(Workstream {
            id: "abc".into(),
            title: "rewrite in the 2026 template".into(),
            brief: "…".into(),
            status: Status::Sealed,
            commit: None,
            report: None,
            changes: vec![],
            epitaph: Some("the 2026 template is for external reports".into()),
        });

        let lessons = session.lessons().await;
        assert_eq!(lessons.len(), 1);

        // The planner is told, and a fresh session is not.
        let with = session.planning_prompt("tidy the reports", &lessons);
        assert!(with.contains("ALREADY TRIED"), "{with}");
        assert!(with.contains("external reports"), "{with}");

        let without = session.planning_prompt("tidy the reports", &[]);
        assert!(!without.contains("ALREADY TRIED"), "{without}");
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
        let briefs = session.plan("tidy the notes", &[]).await;
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
