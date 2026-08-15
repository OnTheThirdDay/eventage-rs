//! Session assembly: one bus + one agent per ACP session.

use crate::acp::{prompt_to_payload, wire::ContentBlock, ClientFs};
use crate::config::{PermissionMode, Provider, SessionConfig};
use crate::lsp::LspPool;
use crate::prompt::build_system_prompt;
use crate::tools::{self, intel};
use crate::workspace::Workspace;
use anyhow::Result;
use eventage::agent::recovery::{reconcile_interrupted_tools, ToolRecovery};
use eventage::agent::{
    load_project_context_walkup, DefaultContextAssembler, DynamicHookChain, SkillTool,
    SkillsLibrary, ToolResultClearingAssembler,
};
use eventage::event::kinds;
use eventage::llm::{AnthropicProvider, LlmProvider, OpenAiProvider, OpenAiResponsesProvider};
use eventage::observability::BusObserver;
use eventage::sqlite::{SqliteEventStore, SqliteExporter};
use eventage::{
    Agent, AgentBuilder, Event, EventBus, ReactStrategy, SummarizingContextAssembler,
    TokenBudgetHook,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{info, warn};

/// How much of the context window the repository map may occupy.
///
/// This buys *detail*, not coverage: every file is named whatever the budget,
/// and the number decides how many of them also get their symbols listed. A
/// whole multi-crate workspace fits comfortably here — this repository is
/// about 3,200 tokens complete — and the map sits in the stable system prefix
/// where a prompt cache reads it back at a fraction of list price.
///
/// It was 1,800 once, chosen without measuring, which cut a third of the
/// files and led an agent to report a capability missing because the file
/// defining it had been dropped.
const REPO_MAP_TOKENS: usize = 8_000;

/// Connect one MCP server and load its tools.
///
/// The bus is handed to the client so server-initiated elicitation surfaces
/// as `mcp.elicitation.request` events — which the ACP bridge can forward to
/// the editor like any other approval.
async fn connect_mcp(
    spec: &crate::config::McpServerConfig,
    bus: &EventBus,
) -> Result<eventage::mcp::McpToolset> {
    use eventage::mcp::{McpClient, McpToolset};

    let client = if let Some(command) = &spec.command {
        McpClient::connect_stdio(command, spec.args.clone(), spec.env.clone()).await?
    } else if let Some(url) = &spec.url {
        McpClient::connect_http(url).await?
    } else {
        anyhow::bail!("MCP server '{}' has neither command nor url", spec.name);
    };

    Ok(
        McpToolset::from_client(client.with_bus(bus.clone(), &spec.name))
            .await?
            .with_prefix(&spec.name),
    )
}

/// One live coding session.
pub struct CodingSession {
    pub id: String,
    pub bus: EventBus,
    agent: Agent,
    hooks: DynamicHookChain,
    cancelled: Arc<AtomicBool>,
    /// Wakes the running turn so its futures are dropped rather than merely
    /// flagged. Separate from `cancelled`, which records that it happened.
    cancel_tx: tokio::sync::watch::Sender<bool>,
    cancel_rx: tokio::sync::watch::Receiver<bool>,
    /// One turn at a time, per session.
    turn_gate: tokio::sync::Mutex<()>,
    config: SessionConfig,
    /// The mode in force, readable by tools and subagents at call time.
    mode: Arc<crate::config::SharedMode>,
    /// One checkpoint per turn, newest last — the anchors for `rewind`.
    checkpoints: tokio::sync::Mutex<Vec<eventage::EventId>>,
    /// The subagents this session started, kept alive between calls so the
    /// agent can go back to one instead of briefing a fresh copy. Dropping
    /// the session drops them, along with any worktrees they hold.
    subagents: Arc<tools::task::SubagentRegistry>,
    /// Everything that was ever recorded for this session, as loaded from
    /// disk — including the events `restore_from` leaves off the branch.
    history: Vec<Event>,
    /// The task writing events to disk, and its running failure count.
    persistence: tokio::sync::Mutex<Option<tokio::task::JoinHandle<usize>>>,
    export_failures: Arc<std::sync::atomic::AtomicUsize>,
}

impl CodingSession {
    /// Build a fresh session rooted at `config.cwd`.
    ///
    /// `client` routes file I/O through the editor when it advertised the
    /// capability; pass `None` for headless runs, which use the disk.
    pub async fn create(
        id: String,
        config: SessionConfig,
        client: Option<ClientFs>,
    ) -> Result<Self> {
        Self::build(id, config, false, client).await
    }

    /// Reopen a persisted session, replaying its event log.
    pub async fn resume(id: &str, config: SessionConfig, client: Option<ClientFs>) -> Result<Self> {
        Self::build(id.to_string(), config, true, client).await
    }

    async fn build(
        id: String,
        config: SessionConfig,
        restore: bool,
        client: Option<ClientFs>,
    ) -> Result<Self> {
        let bus = EventBus::new();
        let ws = Arc::new(Workspace::open(&config.cwd)?);
        let lsp = Arc::new(LspPool::new(&config.cwd));

        // Persistence: one SQLite log per session, so `session/load` works.
        if !crate::config::is_valid_session_id(&id) {
            anyhow::bail!(
                "'{id}' is not a valid session id (expected a UUID); refusing to \
                 turn it into a file path"
            );
        }
        let state_dir = config.state_dir();
        tokio::fs::create_dir_all(&state_dir).await.ok();
        let db = state_dir.join(format!("{id}.db"));

        // What this session was recorded against, checked on the way back in.
        //
        // Resuming carries a conversation full of file paths, diffs and tool
        // results into a new process, and nothing until now confirmed it was
        // the same workspace — a session recorded in one checkout could be
        // reopened in another and would reason confidently about files that
        // were never there. The model is recorded too, because a transcript
        // full of one model's reasoning handed to another is worth knowing
        // about even though it is not an error.
        let manifest_path = state_dir.join(format!("{id}.session.json"));
        let identity = serde_json::json!({
            "workspace": std::fs::canonicalize(&config.cwd)
                .unwrap_or_else(|_| std::path::PathBuf::from(&config.cwd))
                .display()
                .to_string(),
            "model": config.model.model,
        });
        if restore {
            if let Ok(text) = tokio::fs::read_to_string(&manifest_path).await {
                if let Ok(recorded) = serde_json::from_str::<serde_json::Value>(&text) {
                    let was = recorded["workspace"].as_str().unwrap_or_default();
                    let now = identity["workspace"].as_str().unwrap_or_default();
                    if !was.is_empty() && was != now {
                        anyhow::bail!(
                            "session {id} was recorded in '{was}' but this is '{now}'. \
                             Its history describes files that may not exist here; \
                             refusing to resume it against a different workspace."
                        );
                    }
                    let model_was = recorded["model"].as_str().unwrap_or_default();
                    if !model_was.is_empty() && model_was != config.model.model {
                        warn!(
                            recorded = model_was,
                            now = %config.model.model,
                            "resuming a session recorded with a different model"
                        );
                    }
                }
            }
        }
        if let Ok(text) = serde_json::to_string_pretty(&identity) {
            let _ = tokio::fs::write(&manifest_path, text).await;
        }
        let store = SqliteEventStore::new(&db).await?;

        // Kept whole, separately from what goes back onto the bus.
        //
        // `restore_from` rebuilds the *conversation*, and deliberately drops
        // events that were only ever fanned out to observers — streaming
        // deltas, context assemblies — because replaying them onto the active
        // branch would produce nonsense history. But those are exactly what a
        // trace is for, so the record of them has to survive somewhere: this.
        let history = if restore {
            let saved = store.load_all().await?;
            info!(events = saved.len(), "restoring session");
            bus.restore_from(saved.clone()).await;
            saved
        } else {
            Vec::new()
        };

        let exporter = SqliteExporter::new(&db).await?;
        // Subscribed here, not inside the task: a subscription made after the
        // spawn misses whatever is published before the task is first polled,
        // and those are exactly the events a resume needs.
        let observer = BusObserver::new(bus.clone()).add_exporter(exporter);
        let events = observer.subscribe();
        // Retained rather than detached, so the session can wait for the log
        // to be written before it says it is closed, and can report that
        // persistence went wrong instead of looking healthy with a hole in
        // its history.
        let export_failures = observer.failures();
        let persistence = tokio::spawn(observer.run_with(events));

        // ── Model ────────────────────────────────────────────────────────────
        let llm: Arc<dyn LlmProvider> = match config.model.provider {
            Provider::Anthropic => {
                let mut p = AnthropicProvider::new(&config.model.api_key, &config.model.model)
                    .with_max_tokens(config.model.max_tokens)
                    // A gateway is the same API at a different address with
                    // its own routing headers, so it needs no provider of its
                    // own — Portkey, LiteLLM, Helicone and Bedrock proxies all
                    // configure exactly this way.
                    .with_bearer_auth(config.model.bearer_auth)
                    .with_headers(config.model.headers.clone());
                if let Some(url) = &config.model.base_url {
                    p = p.with_base_url(url.trim_end_matches('/'));
                }
                if let Some(budget) = config.model.thinking_tokens {
                    p = p.with_thinking(budget);
                }
                Arc::new(eventage::RetryProvider::new(p))
            }
            Provider::OpenAiResponses => Arc::new(eventage::RetryProvider::new(
                OpenAiResponsesProvider::new(&config.model.api_key, &config.model.model)
                    .with_base_url(config.model.base_url())
                    .with_reasoning_effort("high"),
            )),
            Provider::Qwen => Arc::new(eventage::RetryProvider::new(
                eventage::llm::QwenProvider::new(&config.model.api_key, &config.model.model)
                    .with_base_url(config.model.base_url())
                    .with_thinking(true),
            )),
            Provider::OpenAiChat => Arc::new(eventage::RetryProvider::new(OpenAiProvider::new(
                config.model.base_url(),
                &config.model.api_key,
                &config.model.model,
            ))),
        };

        // ── Context: project instructions + skills, then editing + summarizing ──
        let mut system_prompt = build_system_prompt(&config.cwd);

        // A map of the workspace, so the first question the agent has to
        // answer is "which file" rather than "where do I even start". Built
        // on a blocking thread: it walks the tree and reads source files, and
        // it must not stall the runtime while it does.
        let map_root = std::path::PathBuf::from(&config.cwd);
        let map =
            tokio::task::spawn_blocking(move || crate::repomap::build(&map_root, REPO_MAP_TOKENS))
                .await
                .unwrap_or_default();
        if !map.is_empty() {
            info!(bytes = map.len(), "repository map built");
            system_prompt.push_str("\n\n");
            system_prompt.push_str(&map);
            // Said plainly, because the map is a snapshot and stays one: it
            // lives in the cacheable prefix, so rebuilding it every turn
            // would cost more than it is worth. `repo_map` gets a fresh one.
            system_prompt.push_str(
                "\n\nThis map was taken when the session started. If you add, delete \
                 or rename files — or find a path in it that does not exist — call \
                 `repo_map` for a current one rather than trusting it.",
            );
        }

        for ctx in load_project_context_walkup(&config.cwd) {
            system_prompt.push_str("\n\n");
            system_prompt.push_str(&ctx.system_prompt_section());
        }

        let mut skills = SkillsLibrary::new();
        skills.add_dir(std::path::Path::new(&config.cwd).join(".eventage/skills"))?;
        skills.add_dir(std::path::Path::new(&config.cwd).join(".claude/skills"))?;
        if !skills.is_empty() {
            system_prompt.push_str("\n\n");
            system_prompt.push_str(&skills.system_prompt_section());
        }

        let base = DefaultContextAssembler::new(system_prompt);
        let clearing = ToolResultClearingAssembler::new(
            Arc::new(base),
            (config.context_tokens as f64 * 0.6) as usize,
        );
        let assembler = SummarizingContextAssembler::new(
            Arc::new(clearing),
            Arc::clone(&llm),
            config.context_tokens,
            id.clone(),
        )
        // Compaction goes on the log: it survives reopening the session, the
        // trace shows what was folded away, and a summary that lost something
        // can be replaced without restarting.
        .with_bus(bus.clone());

        // ── Tools ────────────────────────────────────────────────────────────
        let jobs = Arc::new(tools::BackgroundJobs::default());
        let jobs_handle = jobs.clone();
        // Anything a previous run was killed before cleaning up.
        tools::task::sweep_orphans(std::path::Path::new(&config.cwd)).await;
        let plan_state = Arc::new(intel::PlanState::default());
        let mode = Arc::new(crate::config::SharedMode::new(config.mode));
        // Held by the session so its subagents die when it does.
        let subagents = Arc::new(tools::task::SubagentRegistry::new());
        let hooks = DynamicHookChain::new();
        hooks.add_arc(Arc::new(config.mode.policy()));
        if config.token_budget > 0 {
            hooks.add_hook(TokenBudgetHook::new(config.token_budget));
        }

        let mut builder = AgentBuilder::new()
            .agent_id(format!("code-{id}"))
            .bus(bus.clone())
            .llm_arc(Arc::clone(&llm))
            .context(assembler)
            .hook(hooks.clone())
            .strategy(ReactStrategy {
                max_steps: 100,
                max_concurrent_tools: 4,
                stream: true,
                ..Default::default()
            })
            .tool(tools::ReadFile {
                ws: ws.clone(),
                client: client.clone(),
            })
            .tool(tools::WriteFile {
                ws: ws.clone(),
                client: client.clone(),
                lsp: lsp.clone(),
            })
            .tool(tools::EditFile {
                ws: ws.clone(),
                client: client.clone(),
                lsp: lsp.clone(),
            })
            .tool(tools::patch::ApplyPatch {
                ws: ws.clone(),
                client: client.clone(),
                lsp: lsp.clone(),
            })
            .tool(tools::MultiEdit {
                ws: ws.clone(),
                client: client.clone(),
                lsp: lsp.clone(),
            })
            .tool(tools::Glob { ws: ws.clone() })
            .tool(tools::Grep { ws: ws.clone() })
            .tool(tools::ListDirectory { ws: ws.clone() })
            .tool(tools::Bash {
                ws: ws.clone(),
                jobs,
                // Contained by default. A permission prompt is not a boundary:
                // nobody can audit a generated pipeline, and Yolo removes the
                // prompt entirely.
                containment: config.shell,
                container_image: config.container_image.clone(),
            })
            .tool(tools::Jobs { jobs: jobs_handle })
            .tool(tools::Verify {
                ws: ws.clone(),
                containment: config.shell,
            })
            .tool(intel::LspDiagnostics {
                ws: ws.clone(),
                lsp: lsp.clone(),
            })
            .tool(intel::LspDefinition {
                ws: ws.clone(),
                lsp: lsp.clone(),
            })
            .tool(intel::LspReferences {
                ws: ws.clone(),
                lsp: lsp.clone(),
            })
            .tool(intel::LspRename {
                ws: ws.clone(),
                lsp: lsp.clone(),
                client: client.clone(),
            })
            .tool(intel::LspHover {
                ws: ws.clone(),
                lsp: lsp.clone(),
            })
            .tool(intel::LspSymbols {
                ws: ws.clone(),
                lsp: lsp.clone(),
            })
            .tool(tools::vision::ViewImage { ws: ws.clone() })
            .tool(crate::repomap::RepoMap { ws: ws.clone() })
            .tool(intel::Plan {
                state: plan_state.clone(),
            })
            // Reaching the network is gated by `RISKY_TOOLS`, and the tool
            // itself refuses loopback and private addresses.
            //
            // No `web_search`: the only keyless option is scraping a search
            // engine's HTML, which already returns one result and an empty
            // snippet here. A search tool the model trusts and that quietly
            // returns nothing is worse than no search tool, because the model
            // reads the empty result as "nothing exists".
            .tool(eventage::agent::web::WebFetchTool::new())
            .tool(tools::git::Git { ws: ws.clone() })
            .tool(tools::git::CreatePullRequest { ws: ws.clone() })
            // Two names for two capabilities. Exploring costs nothing and is
            // allowed everywhere; delegating work that may edit — and that
            // creates a git worktree to do it in — is gated like any other
            // edit. One tool could only be classified as one or the other,
            // and both answers were wrong.
            .tool(tools::task::Task {
                ws: ws.clone(),
                llm: Arc::clone(&llm),
                depth: 0,
                active: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                mode: mode.clone(),
                bus: bus.clone(),
                registry: subagents.clone(),
                read_only: true,
            })
            .tool(tools::task::Task {
                ws: ws.clone(),
                llm: Arc::clone(&llm),
                depth: 0,
                active: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                mode: mode.clone(),
                bus: bus.clone(),
                // Owned by the session, so the subagents it started die with
                // it — which is what makes keeping them alive safe.
                registry: subagents.clone(),
                read_only: false,
            });

        if !skills.is_empty() {
            builder = builder.tool(SkillTool::new(skills));
        }

        let registry = builder.tool_registry();
        let agent = builder.build();

        // MCP servers the editor configured: their tools join the registry
        // name-prefixed, so two servers exposing `search` cannot collide.
        for spec in &config.mcp_servers {
            match connect_mcp(spec, &bus).await {
                Ok(toolset) => {
                    toolset.add_to_registry(&registry);
                    info!(server = %spec.name, tools = toolset.len(), "MCP server connected");
                }
                // A broken MCP server must not stop the session from starting.
                Err(e) => warn!(server = %spec.name, "MCP server unavailable: {e}"),
            }
        }

        // Rewind anchors live in the log, so rebuild them when reopening —
        // otherwise a resumed session reports no turns to undo even though
        // its history is right there.
        let mut checkpoints: Vec<eventage::EventId> = Vec::new();
        if restore {
            checkpoints = bus
                .log()
                .await
                .iter()
                .filter(|e| e.kind == kinds::CHECKPOINT)
                .map(|e| e.id)
                .collect();
            info!(anchors = checkpoints.len(), "restored rewind anchors");
        }

        // A crash mid-tool leaves an orphaned call; resolve it before resuming.
        if restore {
            let policy = ToolRecovery::new()
                .replay("read_file")
                .replay("grep")
                .replay("glob")
                .replay("lsp_*");
            reconcile_interrupted_tools(&bus, &policy, Some(&registry)).await?;
        }

        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(false);
        Ok(Self {
            id,
            bus,
            agent,
            hooks,
            cancelled: Arc::new(AtomicBool::new(false)),
            cancel_tx,
            cancel_rx,
            turn_gate: tokio::sync::Mutex::new(()),
            mode,
            config,
            checkpoints: tokio::sync::Mutex::new(checkpoints),
            subagents,
            history,
            persistence: tokio::sync::Mutex::new(Some(persistence)),
            export_failures,
        })
    }

    /// Swap the permission policy for a new mode.
    pub async fn set_mode(&self, mode: PermissionMode) {
        // Built first, installed in one call. Clearing and then re-adding left
        // an interval — short, but real — in which a tool preflight could see
        // an empty hook chain and find nothing gating it.
        let mut replacement: Vec<Arc<dyn eventage::agent::CycleHook>> =
            vec![Arc::new(mode.policy())];
        if self.config.token_budget > 0 {
            replacement.push(Arc::new(TokenBudgetHook::new(self.config.token_budget)));
        }
        self.hooks.replace_all(replacement);

        // Subagents derive their policy from this, so it has to be somewhere
        // they can read *now* rather than a value copied when they were built.
        self.mode.store(mode);
        info!(mode = mode.id(), "permission mode changed");
    }

    /// The mode currently in force.
    pub fn mode(&self) -> PermissionMode {
        self.mode.load()
    }

    /// Publish the user's prompt (text and images) onto the bus.
    ///
    /// Each turn opens with a DAG checkpoint, which is what makes
    /// [`rewind`](Self::rewind) possible: undoing a turn is a graph operation,
    /// not a hand-rolled message-array edit.
    pub async fn submit_prompt(&self, blocks: &[ContentBlock]) -> Result<()> {
        self.cancelled.store(false, Ordering::SeqCst);
        let _ = self.cancel_tx.send(false);
        let checkpoint = self.bus.checkpoint().await?;
        self.checkpoints.lock().await.push(checkpoint);
        self.bus
            .publish(Event::new(kinds::USER_MESSAGE, prompt_to_payload(blocks)))
            .await?;
        Ok(())
    }

    /// Roll the conversation back by `turns` (default 1).
    ///
    /// The discarded trajectory is sealed as a rejected branch rather than
    /// deleted, so the agent can still be told "you tried that and it did not
    /// work" on the next attempt.
    pub async fn rewind(&self, turns: usize) -> Result<usize> {
        let turns = turns.max(1);
        let mut checkpoints = self.checkpoints.lock().await;
        if checkpoints.is_empty() {
            anyhow::bail!("nothing to rewind: this session has no completed turns");
        }
        let keep = checkpoints.len().saturating_sub(turns);
        let target = checkpoints[keep];
        self.bus.rollback(target).await?;
        checkpoints.truncate(keep);
        info!(turns, "rewound session");
        Ok(checkpoints.len())
    }

    /// Roll back to a specific checkpoint.
    ///
    /// Counting turns backwards is fine for "undo that", but a session with
    /// ten turns is easier to navigate by pointing at the moment you want to
    /// return to — which is what the timeline's checkpoint flags are.
    pub async fn rewind_to(&self, checkpoint: eventage::EventId) -> Result<usize> {
        let mut checkpoints = self.checkpoints.lock().await;
        let position = checkpoints
            .iter()
            .position(|&id| id == checkpoint)
            .ok_or_else(|| anyhow::anyhow!("that checkpoint is not part of this session"))?;
        self.bus.rollback(checkpoint).await?;
        checkpoints.truncate(position);
        info!(remaining = checkpoints.len(), "rewound to checkpoint");
        Ok(checkpoints.len())
    }

    /// The checkpoints this session can rewind to, oldest first.
    pub async fn checkpoints(&self) -> Vec<eventage::EventId> {
        self.checkpoints.lock().await.clone()
    }

    /// How many events failed to reach the log.
    ///
    /// Non-zero means this session's history is incomplete and a resume will
    /// be missing something. Exporter errors were previously only logged, so
    /// nothing above could tell.
    pub fn export_failures(&self) -> usize {
        self.export_failures
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Close the session and wait for its log to be written.
    ///
    /// Without this the persistence task was simply dropped with whatever it
    /// had not yet flushed — the last events of a session, which are exactly
    /// the ones a resume needs.
    pub async fn close(&self) -> usize {
        self.bus.close();
        let handle = self.persistence.lock().await.take();
        match handle {
            Some(task) => {
                match tokio::time::timeout(std::time::Duration::from_secs(10), task).await {
                    Ok(Ok(failures)) => failures,
                    Ok(Err(e)) => {
                        warn!("persistence task failed: {e}");
                        self.export_failures() + 1
                    }
                    Err(_) => {
                        warn!("persistence did not finish flushing within 10s");
                        self.export_failures() + 1
                    }
                }
            }
            None => self.export_failures(),
        }
    }

    /// Everything ever recorded for this session, as it was loaded.
    ///
    /// Use this and not `bus.log()` to seed a trace: the bus holds the
    /// conversation, which is a strict subset — a resumed session's bus has
    /// no streaming deltas and no context assemblies in it.
    pub fn history(&self) -> &[Event] {
        &self.history
    }

    /// The subagents this session has running.
    ///
    /// They live as long as the session and no longer, so this is also what
    /// bounds their worktrees.
    pub fn subagents(&self) -> &Arc<tools::task::SubagentRegistry> {
        &self.subagents
    }

    /// Run one reasoning cycle to completion, or until cancelled.
    ///
    /// Two things are load-bearing here.
    ///
    /// **One turn at a time.** The gate is what stops two prompts arriving
    /// close together from running against the same agent — which interleaved
    /// user messages and assistant events, let one turn assemble context from
    /// another's half-finished state, and created checkpoints in an order
    /// rewind could not undo. Editors retry and reconnect, so overlapping
    /// requests are ordinary rather than exotic.
    ///
    /// **Cancellation actually cancels.** `select!` drops the losing future,
    /// and dropping the cycle drops the in-flight HTTP request to the model
    /// and every running tool future with it. Setting a flag and waiting for
    /// the cycle to finish — which is what this used to do — meant a
    /// cancelled turn kept spending tokens and kept editing the repository,
    /// and "cancelled" was only the label on the outcome.
    pub async fn run_cycle(&self) -> Result<()> {
        let _turn = self.turn_gate.lock().await;
        let mut cancelled = self.cancel_rx.clone();
        // Mark the current value seen, so only a *new* cancellation fires.
        cancelled.borrow_and_update();

        tokio::select! {
            result = self.agent.cycle() => {
                result?;
                Ok(())
            }
            _ = cancelled.changed() => {
                // The cycle future is dropped at this point.
                info!("turn cancelled: model request and tools dropped");

                // Dropping tool futures mid-flight leaves calls with no
                // results, and an assistant message whose tool calls go
                // unanswered is not a conversation any provider will accept —
                // the *next* turn would fail, somewhere else, for a reason
                // that looks nothing like a cancellation. Closing them out
                // here means it holds wherever the cancel came from rather
                // than only where a caller remembered to do it.
                let policy = ToolRecovery::new();
                match reconcile_interrupted_tools(&self.bus, &policy, None).await {
                    Ok(report) if !report.is_empty() => {
                        info!(closed = report.total(), "closed out interrupted tool calls");
                    }
                    Ok(_) => {}
                    Err(e) => warn!("could not reconcile interrupted tools: {e}"),
                }
                self.bus
                    .publish(Event::new(
                        kinds::SYSTEM_MESSAGE,
                        serde_json::json!({
                            "content": "[the user stopped the previous turn before it finished]"
                        }),
                    ))
                    .await?;
                Ok(())
            }
        }
    }

    /// Stop the turn in flight.
    ///
    /// Deliberately takes no lock the turn holds, so it cannot deadlock
    /// against the thing it is trying to stop.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        // A send with no receivers is fine: nothing is running.
        let _ = self.cancel_tx.send(true);
    }

    pub fn was_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}
