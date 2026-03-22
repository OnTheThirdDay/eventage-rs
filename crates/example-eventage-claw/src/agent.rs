//! ClawAgent and ClawAgentBuilder.
//!
//! Each configured group gets its own isolated `GroupAgent` with a per-group
//! `EventBus`. A shared bus carries IPC events, heartbeats, and schedule events.
//! The `WorkerSet` on the shared bus runs `RelayWorker` and `SchedulerWorker`.

use crate::assembler::{GroupMemoryAssembler, SkillsAssembler, SummarizingAssembler, UserCorrectionsAssembler};
use crate::config::{ClawConfig, GroupConfig};
use crate::error::ClawError;
use crate::hooks::{HumanApprovalHook, SecurityGateHook};
use crate::prompt::build_system_prompt;
use crate::streaming::StreamingOpenAiProvider;
use crate::tools::{
    AddTaskTool, AgentSpawner, BrowserTool, CancelTaskTool, CompleteTaskTool,
    DockerRunCommandTool, EditFileTool, GlobTool, GrepTool, GroupRegistry, ListGroupsTool,
    ListSessionTasksTool, ListTasksTool, LsTool, MessageGroupTool, PauseTaskTool, ReadFileTool,
    RegisterGroupTool, ScheduleState, ScheduleTaskTool, SpawnGroupTool, TaskState, UpdateTaskTool,
    WebFetchTool, WebSearchTool, WriteFileTool, load_tasks, new_group_registry, new_task_state,
};
use crate::workers::{
    ChannelOutputWorker, DelegationReplyWorker, GroupBuses, RelayWorker, SchedulerWorker,
};
use eventage::{
    agent::{ContextAssembler, DefaultContextAssembler},
    llm::OpenAiProvider,
    secrets_masking_transform, AgentBuilder, EventBus, RateLimitedProvider, ReactStrategy,
    WorkerSet,
};
use std::collections::HashMap;
use std::sync::{atomic::AtomicBool, Arc};
use tokio::sync::{Mutex, RwLock};
use tracing::info;
use uuid::Uuid;

// ── GroupAgent ────────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub struct GroupAgent {
    pub name: String,
    pub is_main: bool,
    pub agent: eventage::Agent,
    /// Per-group isolated bus.
    pub bus: EventBus,
    pub worker_set: WorkerSet,
    pub cancelled: Arc<AtomicBool>,
}

// ── BusHook ───────────────────────────────────────────────────────────────────

/// Called with each newly-spawned group `EventBus` so observability workers
/// (JSONL exporter, replay server, etc.) can be attached at runtime.
///
/// `main.rs` populates this slot after the exporter is created; the default
/// no-op means observability is simply not attached when the slot is empty.
pub type BusHook = Arc<dyn Fn(EventBus) + Send + Sync>;

// ── ClawAgent ─────────────────────────────────────────────────────────────────

#[allow(dead_code)]
pub struct ClawAgent {
    pub groups: HashMap<String, GroupAgent>,
    /// Shared bus: IPC events, heartbeats, schedule events.
    pub shared_bus: EventBus,
    pub shared_workers: WorkerSet,
    pub schedule_state: ScheduleState,
    pub config: ClawConfig,
    pub active_group: Arc<Mutex<String>>,
    /// Hook invoked for every dynamically-spawned group bus so `main.rs` can
    /// attach observability (exporter, replay) without restarting.
    pub spawner_bus_hook: Arc<std::sync::Mutex<Option<BusHook>>>,
}

#[allow(dead_code)]
impl ClawAgent {
    fn active_group_name(&self) -> String {
        self.active_group.try_lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Returns a reference to the active group's bus (for TUI subscription).
    pub fn active_bus(&self) -> &EventBus {
        self.groups
            .get(&self.active_group_name())
            .map(|g| &g.bus)
            .unwrap_or(&self.shared_bus)
    }

    /// Returns the cancelled flag for the active group (for TUI Ctrl+X).
    pub fn active_cancelled(&self) -> Arc<AtomicBool> {
        self.groups
            .get(&self.active_group_name())
            .map(|g| g.cancelled.clone())
            .unwrap_or_else(|| Arc::new(AtomicBool::new(false)))
    }

    /// Run all group agents + shared workers concurrently.
    pub async fn run(self) -> Result<(), ClawError> {
        let shared_bus = self.shared_bus.clone();
        let shared_workers = self.shared_workers;

        // Spawn each group agent in its own task
        let mut handles = vec![];
        for (name, group_agent) in self.groups {
            let bus = group_agent.bus.clone();
            let agent = group_agent.agent;
            let ws = group_agent.worker_set;
            let log_name = name.clone();
            let handle = tokio::spawn(async move {
                spawn_group_workers(name, ws, bus);
                agent.run().await.map_err(ClawError::Agent)
            });
            info!(group = %log_name, "ClawAgent: group agent spawned");
            handles.push(handle);
        }

        // Run shared workers on the shared bus
        let shared_handle = tokio::spawn(async move {
            shared_workers
                .run_on(shared_bus)
                .await
                .map_err(ClawError::Worker)
        });

        // Wait for any to finish (or error).
        // select_all panics on an empty iterator — fall through to shared_handle only.
        if handles.is_empty() {
            return shared_handle
                .await
                .unwrap_or_else(|e| Err(ClawError::Tool(e.to_string())));
        }

        tokio::select! {
            Some(r) = async {
                futures_util::future::select_all(handles).await.0.ok()
            } => {
                r?
            }
            r = shared_handle => {
                r.unwrap_or_else(|e| Err(ClawError::Tool(e.to_string())))?
            }
        };

        Ok(())
    }
}

// ── ArcAssembler ─────────────────────────────────────────────────────────────

struct ArcAssembler(Arc<dyn ContextAssembler>);

#[async_trait::async_trait]
impl ContextAssembler for ArcAssembler {
    async fn assemble(
        &self,
        context: &eventage::agent::AssemblyContext<'_>,
    ) -> Vec<eventage::llm::ChatMessage> {
        self.0.assemble(context).await
    }
}

// ── ClawAgentBuilder ──────────────────────────────────────────────────────────

pub struct ClawAgentBuilder {
    config: ClawConfig,
    tui_mode: bool,
    session_id_prefix: String,
}

impl ClawAgentBuilder {
    pub fn new(config: ClawConfig) -> Self {
        Self {
            config,
            tui_mode: true,
            session_id_prefix: Uuid::new_v4().to_string(),
        }
    }

    pub fn tui_mode(mut self, enabled: bool) -> Self {
        self.tui_mode = enabled;
        self
    }

    pub fn build(self) -> ClawAgent {
        let config = self.config;
        let shared_bus = EventBus::new();

        // Restore scheduled tasks from disk (survives restarts).
        let tasks_path = config.tasks_path();
        let persisted_tasks = load_tasks(&tasks_path);
        if !persisted_tasks.is_empty() {
            info!(count = persisted_tasks.len(), "ClawAgentBuilder: restored scheduled tasks from disk");
        }
        let schedule_state = Arc::new(tokio::sync::Mutex::new(persisted_tasks));

        let group_names: Vec<String> = config.groups.iter().map(|g| g.name.clone()).collect();
        let group_registry: GroupRegistry = new_group_registry(group_names.clone());

        // Collect main-group names for IPC authorization.
        let main_groups: Vec<String> = config.groups
            .iter()
            .filter(|g| g.is_main)
            .map(|g| g.name.clone())
            .collect();

        // Build per-group buses into a plain map first — no lock needed during init.
        // EventBus is Arc-backed so cloning into group_buses shares the same bus.
        let bus_map: HashMap<String, EventBus> = config.groups
            .iter()
            .map(|g| (g.name.clone(), EventBus::new()))
            .collect();

        // Wrap in Arc<RwLock<>> for runtime routing (dynamically spawned groups
        // can be inserted without restarting).
        let group_buses: GroupBuses = Arc::new(RwLock::new(
            bus_map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        ));

        // Build shared workers
        let shared_workers = WorkerSet::new()
            .add_worker(RelayWorker {
                group_buses: group_buses.clone(),
                main_groups: main_groups.clone(),
            })
            .add_worker(SchedulerWorker {
                state: schedule_state.clone(),
                group_buses: group_buses.clone(),
            });

        // Shared hook slot: populated by main.rs after the exporter is ready.
        // Both ClawGroupSpawner and ClawAgent hold the same Arc so main.rs can
        // set it once via claw.spawner_bus_hook and the spawner sees it immediately.
        let spawner_bus_hook: Arc<std::sync::Mutex<Option<BusHook>>> =
            Arc::new(std::sync::Mutex::new(None));

        // The first main group is the default target for scheduled tasks created
        // by sub-agents (they can't reach the user directly).
        let main_group_name = main_groups.first()
            .or_else(|| group_names.first())
            .cloned()
            .unwrap_or_default();

        // Spawner used by SpawnGroupTool — holds everything needed to build a
        // new GroupAgent at runtime and insert it into the live routing table.
        let spawner: Arc<dyn AgentSpawner> = Arc::new(ClawGroupSpawner {
            config: Arc::new(config.clone()),
            shared_bus: shared_bus.clone(),
            group_buses: group_buses.clone(),
            group_registry: group_registry.clone(),
            schedule_state: schedule_state.clone(),
            session_id_prefix: self.session_id_prefix.clone(),
            tui_mode: self.tui_mode,
            bus_hook: spawner_bus_hook.clone(),
            main_group: main_group_name,
        });

        // Build each group agent
        let mut groups: HashMap<String, GroupAgent> = HashMap::new();
        let active_group_name = config.groups.first().map(|g| g.name.clone()).unwrap_or_default();

        for group_config in &config.groups {
            let Some(group_bus) = bus_map.get(&group_config.name) else {
                tracing::error!(group = %group_config.name, "bug: missing bus for group — skipping");
                continue;
            };
            let group_bus = group_bus.clone();
            let session_id = format!("{}-{}", self.session_id_prefix, group_config.name);
            let task_state = new_task_state();

            let group_agent = build_group_agent(
                group_config,
                &config,
                group_bus.clone(),
                shared_bus.clone(),
                schedule_state.clone(),
                group_registry.clone(),
                &group_names,
                self.tui_mode,
                session_id,
                task_state,
                spawner.clone(),
                true, // configured groups are user-facing
                None,
            );

            groups.insert(group_config.name.clone(), group_agent);
        }

        ClawAgent {
            groups,
            shared_bus,
            shared_workers,
            schedule_state,
            active_group: Arc::new(Mutex::new(active_group_name)),
            config,
            spawner_bus_hook,
        }
    }
}

// ── ClawGroupSpawner ──────────────────────────────────────────────────────────

/// Implements [`AgentSpawner`] for the claw runtime.
///
/// Held by `SpawnGroupTool` (main group only). On `spawn()`, it builds a full
/// `GroupAgent`, inserts the new bus into the shared routing table, and starts
/// the agent task — all without restarting the process.
struct ClawGroupSpawner {
    config: Arc<ClawConfig>,
    shared_bus: EventBus,
    group_buses: GroupBuses,
    group_registry: GroupRegistry,
    schedule_state: ScheduleState,
    session_id_prefix: String,
    tui_mode: bool,
    /// Shared slot populated by `main.rs` after the exporter is created.
    /// Called with the new bus so observability workers can be attached.
    bus_hook: Arc<std::sync::Mutex<Option<BusHook>>>,
    /// Name of the main (user-facing) group. Scheduled tasks created by
    /// sub-agents default to firing here so the main agent can deliver
    /// reminders to the user via its ChannelOutputWorker.
    main_group: String,
}

#[async_trait::async_trait]
impl AgentSpawner for ClawGroupSpawner {
    async fn spawn(&self, name: &str, system_prompt: Option<&str>) -> Result<(), String> {
        let group_bus = EventBus::new();
        let session_id = format!("{}-{}", self.session_id_prefix, name);
        let task_state = new_task_state();

        let group_config = GroupConfig {
            name: name.to_string(),
            is_main: false,
            system_prompt_suffix: system_prompt.map(|s| s.to_string()),
            human_approval_tools: vec![],
            require_approve_all: false,
            work_dir: None,
            allowed_senders: vec![],
        };

        // Snapshot current group names for the new agent's MessageGroupTool hint.
        let known_groups: Vec<String> = self.group_registry.lock().await.clone();

        // No recursive spawning from spawned sub-agents — pass a no-op spawner.
        let no_spawn: Arc<dyn AgentSpawner> = Arc::new(NoopSpawner);

        let group_agent = build_group_agent(
            &group_config,
            &self.config,
            group_bus.clone(),
            self.shared_bus.clone(),
            self.schedule_state.clone(),
            self.group_registry.clone(),
            &known_groups,
            self.tui_mode,
            session_id,
            task_state,
            no_spawn,
            false, // spawned sub-agents are internal; only delegation replies allowed
            Some(self.main_group.as_str()),
        );

        // Atomically check for duplicates and register — single write lock prevents
        // TOCTOU races from concurrent spawn calls.
        {
            let mut buses = self.group_buses.write().await;
            if buses.contains_key(name) {
                return Err(format!("group '{name}' already exists"));
            }
            buses.insert(name.to_string(), group_bus);
        }
        self.group_registry.lock().await.push(name.to_string());

        let group_name = name.to_string();
        let agent = group_agent.agent;
        let ws = group_agent.worker_set;
        let bus = group_agent.bus;

        // Attach observability (JSONL exporter, replay) to the new bus if the
        // hook has been populated by main.rs.
        if let Some(hook) = self.bus_hook.lock().unwrap().as_ref() {
            hook(bus.clone());
        }

        tokio::spawn(async move {
            spawn_group_workers(group_name.clone(), ws, bus);
            if let Err(e) = agent.run().await {
                tracing::warn!(group = %group_name, "spawned group agent exited: {e}");
            }
        });

        info!(group = %name, "ClawGroupSpawner: sub-agent spawned");
        Ok(())
    }
}

/// Placeholder spawner for sub-agents that should not spawn further agents.
struct NoopSpawner;

#[async_trait::async_trait]
impl AgentSpawner for NoopSpawner {
    async fn spawn(&self, name: &str, _system_prompt: Option<&str>) -> Result<(), String> {
        Err(format!("sub-agent cannot spawn further agents (requested: '{name}')"))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Spawns a group's `WorkerSet` as a fire-and-forget background task.
fn spawn_group_workers(name: String, ws: WorkerSet, bus: EventBus) {
    tokio::spawn(async move {
        if let Err(e) = ws.run_on(bus).await {
            tracing::warn!(group = %name, "group worker error: {e}");
        }
    });
}

// ── build_group_agent ─────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn build_group_agent(
    group_config: &GroupConfig,
    config: &ClawConfig,
    group_bus: EventBus,
    shared_bus: EventBus,
    schedule_state: ScheduleState,
    group_registry: GroupRegistry,
    known_groups: &[String],
    tui_mode: bool,
    session_id: String,
    task_state: TaskState,
    spawner: Arc<dyn AgentSpawner>,
    // True for user-facing configured groups; false for ephemeral sub-agents.
    with_channel_output: bool,
    // If set, scheduled tasks created by this group fire via relay and their
    // result routes back here. None for main/configured groups (no relay needed).
    schedule_reply_group: Option<&str>,
) -> GroupAgent {
    let work_dir = config.group_work_dir(&group_config.name);
    let _ = std::fs::create_dir_all(&work_dir);

    let screenshots_dir = work_dir.join("screenshots");

    let system_prompt = build_system_prompt(group_config);

    // ── LLM provider ─────────────────────────────────────────────────────────
    let (llm, cancelled): (Arc<dyn eventage::llm::LlmProvider>, Arc<AtomicBool>) = if tui_mode {
        let streaming = StreamingOpenAiProvider::new(
            &config.llm_url,
            &config.api_key,
            &config.model,
            group_bus.clone(),
        );
        let flag = streaming.cancelled.clone();
        let base: Arc<dyn eventage::llm::LlmProvider> = Arc::new(streaming);
        let llm = if config.requests_per_minute > 0 {
            Arc::new(RateLimitedProvider::from_arc(base, config.requests_per_minute))
                as Arc<dyn eventage::llm::LlmProvider>
        } else {
            base
        };
        (llm, flag)
    } else {
        let base_llm = OpenAiProvider::new(&config.llm_url, &config.api_key, &config.model);
        let llm: Arc<dyn eventage::llm::LlmProvider> = if config.requests_per_minute > 0 {
            Arc::new(RateLimitedProvider::new(base_llm, config.requests_per_minute))
        } else {
            Arc::new(base_llm)
        };
        (llm, Arc::new(AtomicBool::new(false)))
    };

    // ── Assembler chain ───────────────────────────────────────────────────────
    let base: Arc<dyn ContextAssembler> = Arc::new(DefaultContextAssembler::new(&system_prompt));

    let with_corrections: Arc<dyn ContextAssembler> =
        Arc::new(UserCorrectionsAssembler::new(base, llm.clone()));

    let with_summary: Arc<dyn ContextAssembler> = if config.max_tokens > 0 {
        Arc::new(
            SummarizingAssembler::new(
                with_corrections,
                llm.clone(),
                config.max_tokens,
                &session_id,
            )
            .with_archive_dir("/tmp/claw-history"),
        )
    } else {
        with_corrections
    };

    let with_memory: Arc<dyn ContextAssembler> = Arc::new(GroupMemoryAssembler::new(
        with_summary,
        &config.global_memory_path(),
        &config.group_memory_path(&group_config.name),
    ));

    // Always wrap in SkillsAssembler — it handles a missing directory gracefully
    // and re-scans on every cycle, enabling new skills to be picked up immediately.
    let skills_dir = config.skills_dir();
    let final_assembler: Arc<dyn ContextAssembler> =
        Arc::new(SkillsAssembler::new(with_memory, &skills_dir).with_llm(llm.clone()));

    // ── Secrets masking ───────────────────────────────────────────────────────
    // Redact the API key from all events stored in the bus (JSONL log, subscribers).
    if !config.api_key.is_empty() {
        group_bus.add_publish_transform(secrets_masking_transform(vec![config.api_key.clone()]));
    }

    // ── AgentBuilder ──────────────────────────────────────────────────────────
    let mut builder = AgentBuilder::new()
        .bus(group_bus.clone())
        .llm_arc(llm.clone())
        .context(ArcAssembler(final_assembler))
        .strategy(ReactStrategy {
            max_steps: config.max_steps,
            max_concurrent_tools: 4,
        });

    // ── Standard tools (all groups) ───────────────────────────────────────────
    builder = builder
        .tool(LsTool { work_dir: work_dir.clone() })
        .tool(ReadFileTool { work_dir: work_dir.clone() })
        .tool(WriteFileTool { work_dir: work_dir.clone() })
        .tool(EditFileTool { work_dir: work_dir.clone() })
        .tool(GlobTool { work_dir: work_dir.clone() })
        .tool(GrepTool { work_dir: work_dir.clone() });

    // ── Docker tool (opt-in) ──────────────────────────────────────────────────
    if config.docker_enabled {
        let docker_tool = DockerRunCommandTool::new(work_dir.clone(), &config.docker_image)
            .with_network(config.docker_network.clone());
        builder = builder.tool(docker_tool);
    }

    let tasks_path = Some(config.tasks_path());

    builder = builder
        .tool(WebSearchTool::new())
        .tool(WebFetchTool::new())
        .tool(BrowserTool::new(screenshots_dir))
        .tool(ScheduleTaskTool {
            bus: shared_bus.clone(),
            state: schedule_state.clone(),
            default_group: group_config.name.clone(),
            reply_group: schedule_reply_group.map(|s| s.to_string()),
            tasks_path: tasks_path.clone(),
        })
        .tool(ListTasksTool { state: schedule_state.clone() })
        .tool(CancelTaskTool {
            bus: shared_bus.clone(),
            state: schedule_state.clone(),
            tasks_path: tasks_path.clone(),
        })
        .tool(PauseTaskTool {
            bus: shared_bus.clone(),
            state: schedule_state.clone(),
            tasks_path: tasks_path.clone(),
        })
        .tool(UpdateTaskTool {
            bus: shared_bus.clone(),
            state: schedule_state.clone(),
            tasks_path: tasks_path.clone(),
        })
        .tool(MessageGroupTool {
            shared_bus: shared_bus.clone(),
            known_groups: known_groups.to_vec(),
            source_group: group_config.name.clone(),
        })
        .tool(AddTaskTool { state: task_state.clone(), bus: group_bus.clone() })
        .tool(CompleteTaskTool { state: task_state.clone(), bus: group_bus.clone() })
        .tool(ListSessionTasksTool { state: task_state });

    // ── Admin tools (main group only) ─────────────────────────────────────────
    if group_config.is_main {
        builder = builder
            .tool(RegisterGroupTool {
                bus: shared_bus.clone(),
                registry: group_registry.clone(),
            })
            .tool(ListGroupsTool {
                registry: group_registry.clone(),
            })
            .tool(SpawnGroupTool { spawner });
    }

    // ── Hook ──────────────────────────────────────────────────────────────────
    if tui_mode {
        if group_config.require_approve_all {
            builder = builder.hook(SecurityGateHook::all_tools(group_bus.clone()));
        } else if !group_config.human_approval_tools.is_empty() {
            builder = builder.hook(SecurityGateHook::watched(
                group_bus.clone(),
                group_config.human_approval_tools.clone(),
            ));
        }
    } else if group_config.require_approve_all {
        builder = builder.hook(HumanApprovalHook::all_tools());
    } else if !group_config.human_approval_tools.is_empty() {
        builder = builder.hook(HumanApprovalHook::new(group_config.human_approval_tools.clone()));
    }

    let agent = builder.build();

    info!(
        group = %group_config.name,
        is_main = group_config.is_main,
        model = %config.model,
        tui_mode,
        "GroupAgent ready"
    );

    // ── Per-group workers ──────────────────────────────────────────────────────
    let mut group_workers = WorkerSet::new();
    group_workers = group_workers.add_worker(DelegationReplyWorker::new(
        shared_bus.clone(),
        &group_config.name,
    ));
    if with_channel_output {
        if let Some(ref webhook_url) = config.webhook_url {
            group_workers = group_workers.add_worker(ChannelOutputWorker::new(
                webhook_url.clone(),
                &group_config.name,
            ));
        }
    }

    GroupAgent {
        name: group_config.name.clone(),
        is_main: group_config.is_main,
        agent,
        bus: group_bus,
        worker_set: group_workers,
        cancelled,
    }
}
