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
    load_project_context_walkup, DefaultContextAssembler, DynamicHookChain,
    NegativeAwareContextAssembler, SkillTool, SkillsLibrary, ToolResultClearingAssembler,
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
use tracing::{debug, info, warn};

/// A point a session can be rewound to.
#[derive(Debug, Clone)]
struct Checkpoint {
    /// The event the conversation rolls back to.
    id: eventage::EventId,
    /// The working tree as the turn found it, if the workspace is a git
    /// repository. `None` means a rewind can restore the conversation and
    /// nothing else.
    tree: Option<String>,
}

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

/// How many rolled-back attempts stay in memory as events.
///
/// Small on purpose. Beyond this they are evicted, and `EpitaphStrategy`
/// leaves a one-line summary in their place — which is what a later attempt
/// actually needs from an old failure. Unbounded retention was the previous
/// default and meant a long session held every trajectory it had ever
/// abandoned.
const MAX_RETAINED_BRANCHES: usize = 8;

/// Put the loaded plugins on the log, so a surface can show them.
pub async fn announce_plugins(bus: &EventBus, host: &eventage::PluginHost) {
    let plugins: Vec<serde_json::Value> = host
        .plugins()
        .iter()
        .map(|p| {
            serde_json::json!({
                "name": p.name,
                "description": p.description,
                "skills": p.skills.len(),
                "mcp_servers": p.mcp_servers.len(),
                "adds_prompt": p.prompt_fragment.is_some(),
            })
        })
        .collect();
    let _ = bus
        .publish(Event::new(
            kinds::SYSTEM_PLUGINS,
            serde_json::json!({ "plugins": plugins }),
        ))
        .await;
}

/// Find and load the plugins available to a session.
///
/// Two places, in this order: the workspace's own `.eventage/plugins/`, then
/// the user's `~/.eventage/plugins/`. A repository's plugins come first so a
/// project can pin the version of a tool its instructions assume.
///
/// A directory without a manifest is not a plugin and is skipped in silence —
/// people keep all sorts of things in a plugins folder. A directory *with* a
/// manifest that fails to load is reported, because that one was meant to
/// work.
pub fn load_plugins(cwd: &str) -> eventage::PluginHost {
    let mut host = eventage::PluginHost::new();
    let roots = [
        std::path::PathBuf::from(cwd).join(".eventage/plugins"),
        dirs::home_dir()
            .unwrap_or_default()
            .join(".eventage/plugins"),
    ];

    for root in roots {
        let Ok(entries) = std::fs::read_dir(&root) else {
            continue;
        };
        for entry in entries.flatten() {
            let dir = entry.path();
            if !dir.is_dir() || !dir.join(eventage::plugin::MANIFEST_NAME).is_file() {
                continue;
            }
            if let Err(e) = host.load(&dir) {
                warn!(path = %dir.display(), "could not load plugin: {e}");
            }
        }
    }
    host
}

/// How the agent is told about attempts that were rolled back.
///
/// Two sources of different weight. Retained branches are rendered as events
/// by the framework's own formatter, so the agent sees what it did; branches
/// evicted earlier survive only as epitaphs, one sentence each, written by
/// the model when they aged out. Recent mistakes in detail, older ones as a
/// line — which is roughly how a person remembers their own.
fn negative_summary(branches: &[Vec<Event>], epitaphs: &eventage::EpitaphStore) -> String {
    let mut summary = eventage::agent::default_negative_context_format(branches);

    let older: Vec<String> = epitaphs
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .values()
        .cloned()
        .collect();
    if !older.is_empty() {
        summary.push_str("\n\nEarlier attempts, summarised:\n");
        for lesson in older {
            summary.push_str("  - ");
            summary.push_str(lesson.trim());
            summary.push('\n');
        }
    }
    summary
}

/// Published when a rewind leaves edited files on disk.
///
/// Rewinding undoes the *conversation*. Where the workspace is a git
/// repository the working tree is put back too ([`WORKING_TREE_RESTORED`]);
/// where it is not, there is nothing to put it back from, and saying so is
/// the difference between a limitation and a trap.
pub const WORKING_TREE_UNCHANGED: &str = "session.working_tree_unchanged";

/// Published when a rewind put the working tree back as well.
///
/// Carries `undo`, the commit id of a snapshot taken immediately before the
/// restore, so anything written since the checkpoint can be recovered with
/// `git checkout <undo> -- .`.
pub const WORKING_TREE_RESTORED: &str = "session.working_tree_restored";

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

/// Does this model want the adaptive thinking shape rather than a budget?
///
/// A deny-list of the older families rather than an allow-list of the newer
/// ones, so a model released next month gets the current shape by default
/// instead of the legacy one. Gateway-prefixed ids
/// (`@bedrock-au/au.anthropic.claude-opus-4-8`) match on the family substring
/// like any other.
fn uses_adaptive_thinking(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    /// Families that require `{type: "enabled", budget_tokens}` and reject
    /// `adaptive`. One gateway can front both generations, so the choice is
    /// per model and not per session.
    const LEGACY_BUDGET_FAMILIES: &[&str] = &[
        "sonnet-4-5",
        "haiku-4-5",
        "opus-4-5",
        "opus-4-1",
        "opus-4-0",
        "sonnet-4-0",
        "claude-3",
        "claude-2",
    ];
    !LEGACY_BUDGET_FAMILIES.iter().any(|fam| m.contains(fam))
}

/// Build the provider a [`ModelConfig`](crate::config::ModelConfig) describes.
///
/// Free-standing because it is not the coding agent's: cowork resolves a model
/// the same way, and two copies of the gateway handling would drift the first
/// time one of them learned something. Every provider is wrapped in
/// `RetryProvider`, since a transient 429 in the middle of a long session is
/// the most likely failure any of them will see.
pub fn provider_for(model: &crate::config::ModelConfig) -> Arc<dyn LlmProvider> {
    match model.provider {
        Provider::Anthropic => {
            let mut p = AnthropicProvider::new(&model.api_key, &model.model)
                .with_max_tokens(model.max_tokens)
                // A gateway is the same API at a different address with
                // its own routing headers, so it needs no provider of its
                // own — Portkey, LiteLLM, Helicone and Bedrock proxies all
                // configure exactly this way.
                .with_bearer_auth(model.bearer_auth)
                .with_headers(model.headers.clone());
            if let Some(url) = &model.base_url {
                p = p.with_base_url(url.trim_end_matches('/'));
            }
            // The `[1m]` marker and anything else the model string asked for
            // travel as headers, never as part of the model id.
            for beta in &model.betas {
                p = p.with_beta(beta.clone());
            }
            if let Some(budget) = model.thinking_tokens {
                match uses_adaptive_thinking(&model.model) {
                    true => p = p.with_adaptive_thinking("high"),
                    false => p = p.with_thinking(budget),
                }
            }
            Arc::new(eventage::RetryProvider::new(p))
        }
        Provider::OpenAiResponses => Arc::new(eventage::RetryProvider::new(
            OpenAiResponsesProvider::new(&model.api_key, &model.model)
                .with_base_url(model.base_url())
                .with_reasoning_effort("high"),
        )),
        Provider::Qwen => Arc::new(eventage::RetryProvider::new(
            eventage::llm::QwenProvider::new(&model.api_key, &model.model)
                .with_base_url(model.base_url())
                .with_thinking(true),
        )),
        Provider::OpenAiChat => Arc::new(eventage::RetryProvider::new(OpenAiProvider::new(
            model.base_url(),
            &model.api_key,
            &model.model,
        ))),
    }
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
    /// One checkpoint per turn, newest last — the anchors for `rewind`,
    /// each paired with a snapshot of the working tree as the turn found it.
    checkpoints: tokio::sync::Mutex<Vec<Checkpoint>>,
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
        // Resolved first: the bus's eviction strategy needs a model to write
        // epitaphs with, and building the provider is pure.
        let llm: Arc<dyn LlmProvider> = provider_for(&config.model);

        // Rejected branches are bounded, and what falls off the end leaves a
        // sentence behind rather than vanishing.
        //
        // Unbounded retention meant a long session accumulated every attempt
        // it had ever rolled back; `EpitaphStrategy` asks the model to
        // summarise each branch as it is evicted, so the lesson outlives the
        // events. `PruneStrategy` — the default — simply deletes them.
        let epitaphs = eventage::EpitaphStrategy::new(Arc::clone(&llm));
        let epitaph_store = epitaphs.epitaphs();
        let strategy = Arc::new(epitaphs);
        let bus = EventBus::with_config(eventage::BusConfig {
            max_retained_branches: MAX_RETAINED_BRANCHES,
            eviction_strategy: Arc::clone(&strategy) as Arc<dyn eventage::BranchEvictionStrategy>,
            ..Default::default()
        });
        // Now that the bus exists, epitaphs go onto it as well as into the
        // store — so they survive reopening and the trace shows them.
        strategy.publish_to(bus.clone());
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

        // ── Plugins ──────────────────────────────────────────────────────
        //
        // A plugin bundles a prompt fragment, skills and MCP servers in one
        // directory, which is how somebody extends the agent without forking
        // it. Installed into a registry of its own rather than the agent's,
        // because `install` returns the prompt fragment and the system prompt
        // has to be finished before the assembler is built — the agent does
        // not exist yet at this point. Its tools are handed to the builder
        // below.
        let plugin_tools = eventage::agent::ToolRegistry::new();
        let plugin_prompt = match load_plugins(&config.cwd) {
            host if host.plugins().is_empty() => String::new(),
            host => match host.install(&plugin_tools).await {
                Ok(prompt) => {
                    info!(plugins = host.plugins().len(), "plugins installed");
                    // A plugin silently changes the system prompt and the
                    // tool list. Announcing it is the difference between the
                    // user being able to see that and having to infer it from
                    // behaviour they did not expect.
                    announce_plugins(&bus, &host).await;
                    prompt
                }
                Err(e) => {
                    // A broken plugin is not a reason to refuse to start; the
                    // session is still perfectly usable without it.
                    warn!("could not install a plugin: {e}");
                    String::new()
                }
            },
        };
        if !plugin_prompt.trim().is_empty() {
            system_prompt.push_str("\n\n");
            system_prompt.push_str(plugin_prompt.trim());
        }

        let base = DefaultContextAssembler::new(system_prompt);
        let clearing = ToolResultClearingAssembler::new(
            Arc::new(base),
            (config.context_tokens as f64 * 0.6) as usize,
        );
        let summarizing = SummarizingContextAssembler::new(
            Arc::new(clearing),
            Arc::clone(&llm),
            config.context_tokens,
            id.clone(),
        )
        // Compaction goes on the log: it survives reopening the session, the
        // trace shows what was folded away, and a summary that lost something
        // can be replaced without restarting.
        .with_bus(bus.clone());

        // Outermost, so what it injects is not summarised away by the layer
        // below: a warning about an approach that failed is worth more than
        // the transcript of the approach.
        //
        // Two sources, deliberately different in weight. Branches still
        // retained are shown as events, so the agent sees what it actually
        // did; branches long since evicted are one line each, written by
        // `EpitaphStrategy` when they aged out. Recent mistakes in detail,
        // old ones as a sentence.
        let assembler = NegativeAwareContextAssembler::new(summarizing)
            .with_formatter(move |branches| negative_summary(branches, &epitaph_store));

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
            .tool(tools::git::Git {
                ws: ws.clone(),
                containment: config.shell,
            })
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
                containment: config.shell,
                container_image: config.container_image.clone(),
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
                containment: config.shell,
                container_image: config.container_image.clone(),
            });

        if !skills.is_empty() {
            builder = builder.tool(SkillTool::new(skills));
        }
        for tool in plugin_tools.all_tools() {
            builder = builder.tool_arc(tool);
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
        let mut checkpoints: Vec<Checkpoint> = Vec::new();
        if restore {
            checkpoints = bus
                .log()
                .await
                .iter()
                .filter(|e| e.kind == kinds::CHECKPOINT)
                // No tree: the snapshot commits from the previous process
                // are unreferenced and `git gc` may already have collected
                // them, so a resumed session rewinds the conversation and
                // says plainly that the files stayed. Claiming a restore
                // against a commit id that no longer resolves would be worse
                // than not offering one.
                .map(|e| Checkpoint {
                    id: e.id,
                    tree: None,
                })
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
    ///
    /// **Not gated.** It publishes a message and clears the cancellation
    /// flag; the gate is taken by [`run_cycle`](Self::run_cycle). A caller
    /// that submits and then runs, without holding something of its own in
    /// between, has a window where a second prompt can publish a second user
    /// message and reset the cancellation of the turn already in flight.
    /// Prefer [`prompt_turn`](Self::prompt_turn), which holds the gate across
    /// both. This stays public for callers that genuinely drive the two
    /// halves apart, such as a CLI with one turn and no concurrency.
    pub async fn submit_prompt(&self, blocks: &[ContentBlock]) -> Result<()> {
        self.cancelled.store(false, Ordering::SeqCst);
        let _ = self.cancel_tx.send(false);
        let checkpoint = self.bus.checkpoint().await?;

        // Published before the snapshot, and the order is the whole point.
        //
        // `snapshot::capture` runs `git add -A` over the working tree, which
        // on a large or untracked-heavy repository takes seconds. Studio has
        // no optimistic echo — the message appears when its event reaches the
        // SSE feed — so publishing afterwards meant *your own typing* sat
        // invisible for the length of a git walk, which reads as the app
        // having missed the keystroke.
        //
        // The checkpoint still comes first, so a rewind to it still lands
        // before the message and rewind semantics are unchanged. And the
        // snapshot is identical either way: nothing mutates the tree between
        // these two lines, because the reasoning cycle is started by the
        // caller only after this returns.
        self.bus
            .publish(Event::new(kinds::USER_MESSAGE, prompt_to_payload(blocks)))
            .await?;

        // Recorded with the checkpoint so a rewind can put the files back as
        // well as the conversation. Best-effort: a workspace that is not a
        // git repository simply has no snapshot, and `rewind` says so rather
        // than pretending.
        let tree = match crate::snapshot::capture(std::path::Path::new(&self.config.cwd)).await {
            Ok(commit) => Some(commit),
            Err(e) => {
                debug!("no working-tree snapshot for this turn: {e:#}");
                None
            }
        };
        self.checkpoints.lock().await.push(Checkpoint {
            id: checkpoint,
            tree,
        });
        Ok(())
    }

    /// Roll the conversation back by `turns` (default 1).
    ///
    /// The discarded trajectory is sealed as a rejected branch rather than
    /// deleted, so the agent can still be told "you tried that and it did not
    /// work" on the next attempt.
    ///
    /// **The working tree is not restored.** Rewinding is a graph operation
    /// on the conversation; the files the rewound turns wrote are still on
    /// disk exactly as the agent left them. Undoing those too would mean
    /// snapshotting the tree at every checkpoint, which is a feature and not
    /// a line of code. What this does instead is refuse to be quiet about it:
    /// [`WORKING_TREE_UNCHANGED`] is published naming every file the
    /// discarded turns modified, so the user is told what to revert rather
    /// than left believing the undo covered it.
    pub async fn rewind(&self, turns: usize) -> Result<usize> {
        // Refused while a turn is running. Rewinding rolls back the event DAG
        // *and* rewrites the working tree; doing that underneath a live turn
        // means the model's next tool call lands on files that moved and its
        // events attach to a branch that was sealed while it was thinking.
        // The gate has existed since `prompt_turn` and this simply never took
        // it — ACP can call `session/rewind` at any moment, and nothing
        // stopped it.
        let _gate = self.turn_gate.try_lock().map_err(|_| {
            anyhow::anyhow!(
                "this session is working on something; stop the current turn before \
                 rewinding it"
            )
        })?;
        let turns = turns.max(1);
        let mut checkpoints = self.checkpoints.lock().await;
        if checkpoints.is_empty() {
            anyhow::bail!("nothing to rewind: this session has no completed turns");
        }
        let keep = checkpoints.len().saturating_sub(turns);
        let target = checkpoints[keep].clone();
        let touched = self.files_touched_since(target.id).await;
        self.bus.rollback(target.id).await?;
        checkpoints.truncate(keep);
        self.restore_working_tree(&target, touched).await;
        info!(turns, "rewound session");
        Ok(checkpoints.len())
    }

    /// Paths written by the events that a rollback to `target` will discard.
    ///
    /// Read off the tool results themselves rather than a separate ledger:
    /// every editing tool reports `_diff.path`, which is what the editor
    /// already uses to draw a diff card, so anything that writes a file is
    /// necessarily here.
    async fn files_touched_since(&self, target: eventage::EventId) -> Vec<String> {
        let log = self.bus.log().await;
        let from = log
            .iter()
            .position(|e| e.id == target)
            .map(|i| i + 1)
            .unwrap_or(0);
        let mut paths: Vec<String> = log[from..]
            .iter()
            .filter(|e| e.kind == kinds::TOOL_RESULT)
            .filter_map(|e| {
                e.payload
                    .get("result")
                    .and_then(|r| r.get("_diff"))
                    .and_then(|d| d.get("path"))
                    .and_then(|p| p.as_str())
                    .map(str::to_string)
            })
            .collect();
        paths.sort();
        paths.dedup();
        paths
    }

    /// Put the working tree back, or say why it could not be.
    ///
    /// Both outcomes are announced, because the failure mode this closes was
    /// silence: a turn vanished from the transcript and the files it wrote
    /// stayed, which reads as an undo that half worked.
    ///
    /// Ephemeral either way — a note to the person watching, not a fact the
    /// model should carry into the next turn's context.
    async fn restore_working_tree(&self, target: &Checkpoint, touched: Vec<String>) {
        let Some(tree) = &target.tree else {
            if touched.is_empty() {
                return;
            }
            self.bus.broadcast(Event::new(
                WORKING_TREE_UNCHANGED,
                serde_json::json!({
                    "paths": touched,
                    "detail": "the conversation was rewound, but this workspace is not a \
                               git repository, so there was no snapshot to put the files \
                               back from — they are still as the agent left them",
                }),
            ));
            return;
        };

        match crate::snapshot::restore(std::path::Path::new(&self.config.cwd), tree).await {
            Ok(restored) if restored.paths.is_empty() => {}
            Ok(restored) => {
                info!(files = restored.paths.len(), "restored the working tree");
                self.bus.broadcast(Event::new(
                    WORKING_TREE_RESTORED,
                    serde_json::json!({
                        "paths": restored.paths,
                        "undo": restored.undo,
                        "detail": "the working tree was put back as it was before these \
                                   turns; `git checkout <undo> -- .` recovers what they wrote",
                    }),
                ));
            }
            Err(e) => {
                warn!("could not restore the working tree: {e:#}");
                self.bus.broadcast(Event::new(
                    WORKING_TREE_UNCHANGED,
                    serde_json::json!({
                        "paths": touched,
                        "detail": format!(
                            "the conversation was rewound, but the working tree could not \
                             be restored ({e:#}) — these files are still as the agent left them"
                        ),
                    }),
                ));
            }
        }
    }

    /// Roll back to a specific checkpoint.
    ///
    /// Counting turns backwards is fine for "undo that", but a session with
    /// ten turns is easier to navigate by pointing at the moment you want to
    /// return to — which is what the timeline's checkpoint flags are.
    pub async fn rewind_to(&self, checkpoint: eventage::EventId) -> Result<usize> {
        // Same reasoning as `rewind`: not while a turn is in flight.
        let _gate = self.turn_gate.try_lock().map_err(|_| {
            anyhow::anyhow!(
                "this session is working on something; stop the current turn before \
                 rewinding it"
            )
        })?;
        let mut checkpoints = self.checkpoints.lock().await;
        let position = checkpoints
            .iter()
            .position(|c| c.id == checkpoint)
            .ok_or_else(|| anyhow::anyhow!("that checkpoint is not part of this session"))?;
        let target = checkpoints[position].clone();
        let touched = self.files_touched_since(checkpoint).await;
        self.bus.rollback(checkpoint).await?;
        checkpoints.truncate(position);
        self.restore_working_tree(&target, touched).await;
        info!(remaining = checkpoints.len(), "rewound to checkpoint");
        Ok(checkpoints.len())
    }

    /// The checkpoints this session can rewind to, oldest first.
    pub async fn checkpoints(&self) -> Vec<eventage::EventId> {
        self.checkpoints.lock().await.iter().map(|c| c.id).collect()
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
        self.cycle_locked().await
    }

    /// Submit a prompt and run the turn it opens, as one indivisible step.
    ///
    /// The gate is held across both halves. Taken only by `run_cycle`, a
    /// second `session/prompt` arriving mid-turn could publish its user
    /// message into the running conversation and clear the in-flight turn's
    /// cancellation flag before ever reaching the gate. The Studio backend
    /// happens to be safe because it holds a mutex of its own around the
    /// pair; the ACP server had no such thing, and an editor that pipelines
    /// requests is not doing anything unusual.
    ///
    /// Refuses rather than queues when a turn is already running, so the
    /// caller is told what happened instead of watching a request sit.
    pub async fn prompt_turn(&self, blocks: &[ContentBlock]) -> Result<()> {
        let Ok(_turn) = self.turn_gate.try_lock() else {
            anyhow::bail!(
                "this session is already working on something; cancel the current turn first"
            );
        };
        self.submit_prompt(blocks).await?;
        self.cycle_locked().await
    }

    /// Is a turn running right now?
    pub fn is_busy(&self) -> bool {
        self.turn_gate.try_lock().is_err()
    }

    /// The body of a turn, with the gate already held by the caller.
    async fn cycle_locked(&self) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinking_shape_is_chosen_by_model_family() {
        // One gateway can front both generations — a settings file mapping
        // `opus` to 4.8 and `sonnet` to 4.5 needs a different request body per
        // model, so this cannot be a per-session setting.
        for legacy in [
            "claude-sonnet-4-5",
            "claude-haiku-4-5-20251001",
            "au.anthropic.claude-sonnet-4-5-v1",
            "claude-opus-4-5",
            "claude-3-7-sonnet",
        ] {
            assert!(
                !uses_adaptive_thinking(legacy),
                "{legacy} rejects adaptive and needs budget_tokens"
            );
        }

        // Newer families, including gateway-prefixed ids — and anything
        // unrecognised, because the deny-list is of the *old* shapes so a
        // model released next month gets the current one.
        for adaptive in [
            "claude-opus-4-8",
            "@bedrock-au/au.anthropic.claude-opus-4-8",
            "claude-sonnet-5",
            "claude-opus-5",
            "some-model-nobody-has-heard-of",
        ] {
            assert!(
                uses_adaptive_thinking(adaptive),
                "{adaptive} should use the adaptive shape"
            );
        }
    }

    #[test]
    fn old_failures_are_a_line_and_recent_ones_are_the_events() {
        // Two sources of different weight. A branch still retained is shown
        // as what the agent actually did; one evicted long ago survives only
        // as the sentence `EpitaphStrategy` wrote when it aged out. Showing
        // only one of the two would either flood the context with old
        // transcripts or lose the lesson entirely.
        let retained = vec![vec![Event::new(
            kinds::ASSISTANT_MESSAGE,
            serde_json::json!({ "content": "I will rewrite the parser from scratch." }),
        )]];

        let store: eventage::EpitaphStore = Default::default();
        store.lock().unwrap().insert(
            uuid::Uuid::new_v4(),
            "rewriting the lexer broke the tests".into(),
        );

        let summary = negative_summary(&retained, &store);

        // The recent attempt, in its own words.
        assert!(summary.contains("rewrite the parser"), "{summary}");
        // The old one, as a line under its own heading.
        assert!(
            summary.contains("Earlier attempts, summarised:"),
            "{summary}"
        );
        assert!(
            summary.contains("rewriting the lexer broke the tests"),
            "{summary}"
        );
    }

    #[test]
    fn with_nothing_evicted_no_summary_section_appears() {
        // The heading is worth nothing on its own, and an empty section in
        // every request is exactly the kind of thing that gets ignored.
        let store: eventage::EpitaphStore = Default::default();
        let summary = negative_summary(&[], &store);
        assert!(!summary.contains("Earlier attempts"), "{summary}");
    }

    #[test]
    fn plugins_are_looked_for_in_the_workspace_first() {
        // A project pinning its own version of a tool has to win over one the
        // user happens to have installed globally.
        let dir = tempfile::tempdir().unwrap();
        let host = load_plugins(dir.path().to_str().unwrap());
        // Nothing there: an absent plugins directory is the normal case and
        // must not be an error.
        assert!(host.plugins().is_empty());
    }
}
