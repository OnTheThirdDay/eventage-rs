use crate::assembler::{MemoryAssembler, SkillsAssembler, SummarizingAssembler, UserCorrectionsAssembler};
use crate::error::CodingAgentError;
use crate::hooks::{HumanApprovalHook, SecurityGateHook};
use crate::prompt::build_system_prompt;
use crate::streaming::StreamingOpenAiProvider;
use crate::tools::{
    AddTodoTool, ApplyPatchTool, CheckAsyncTaskTool, CompleteTodoTool, EditFileTool, GlobTool,
    GrepTool, LaunchAsyncTaskTool, ListTodosTool, LsTool, ReadFileTool, RunCommandTool,
    SubAgentSpec, TaskTool, TodoState, WriteFileTool,
};
use crate::workers::{SubAgentWorker, TurnDiffWorker};
use crate::workspace::Workspace;
use eventage::{
    agent::{ContextAssembler, DefaultContextAssembler},
    event::{kinds, Event},
    llm::OpenAiProvider,
    AgentBuilder, EventBus, RateLimitedProvider, ReactStrategy, WorkerSet,
};
use serde_json::json;
use std::path::PathBuf;
use std::sync::{atomic::AtomicBool, Arc};
use tracing::info;
use uuid::Uuid;

// ── CodingAgent ───────────────────────────────────────────────────────────────

pub struct CodingAgent {
    agent: eventage::Agent,
    bus: EventBus,
    worker_set: Option<WorkerSet>,
    /// Cancellation flag (only Some in TUI mode, from StreamingOpenAiProvider).
    pub cancelled: Option<Arc<AtomicBool>>,
    #[allow(dead_code)]
    pub workspace: Arc<Workspace>,
    pub model: String,
    pub session_id: String,
}

impl CodingAgent {
    pub fn bus(&self) -> &EventBus {
        &self.bus
    }

    /// Publish a user message, run one agent cycle, return the assistant's response.
    pub async fn chat(&self, msg: &str) -> Result<String, CodingAgentError> {
        self.bus
            .publish(Event::new(kinds::USER_MESSAGE, json!({ "text": msg })))
            .await
            .map_err(|e| CodingAgentError::Tool(e.to_string()))?;

        self.agent.cycle().await?;

        let log = self.bus.log().await;
        let response = log
            .iter()
            .rev()
            .find(|e| e.kind == kinds::ASSISTANT_MESSAGE)
            .and_then(|e| e.payload["content"].as_str().map(|s| s.to_string()))
            .unwrap_or_else(String::new);

        Ok(response)
    }

    /// Run the agent reactively. Workers run concurrently on the same bus.
    pub async fn run(self) -> Result<(), CodingAgentError> {
        let bus = self.bus.clone();
        let agent = self.agent;

        if let Some(ws) = self.worker_set {
            tokio::select! {
                r = agent.run() => r.map_err(CodingAgentError::Agent),
                r = ws.run_on(bus) => r.map_err(CodingAgentError::Worker),
            }
        } else {
            agent.run().await.map_err(CodingAgentError::Agent)
        }
    }
}

// ── CodingAgentBuilder ────────────────────────────────────────────────────────

pub struct CodingAgentBuilder {
    llm_url: String,
    api_key: String,
    model: String,
    system_prompt: Option<String>,
    max_steps: usize,
    max_tokens: usize,
    memory_paths: Vec<PathBuf>,
    skill_dirs: Vec<PathBuf>,
    work_dir: Option<PathBuf>,
    human_approval_tools: Vec<String>,
    require_approve_all: bool,
    subagent_specs: Vec<SubAgentSpec>,
    extra_tools: Vec<Arc<dyn eventage::Tool>>,
    async_subagents: bool,
    session_id: String,
    /// 0 = unlimited
    requests_per_minute: u32,
    /// When true, use StreamingOpenAiProvider + SecurityGateHook (TUI mode).
    /// When false, use OpenAiProvider + HumanApprovalHook (REPL mode).
    tui_mode: bool,
}

impl Default for CodingAgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
impl CodingAgentBuilder {
    pub fn new() -> Self {
        Self {
            llm_url: "http://localhost:11434/v1".into(),
            api_key: "ollama".into(),
            model: "qwen3:4b".into(),
            system_prompt: None,
            max_steps: 30,
            max_tokens: 120_000,
            memory_paths: vec![],
            skill_dirs: vec![],
            work_dir: None,
            human_approval_tools: vec![],
            require_approve_all: false,
            subagent_specs: vec![],
            extra_tools: vec![],
            async_subagents: true,
            session_id: Uuid::new_v4().to_string(),
            requests_per_minute: 0,
            tui_mode: true,
        }
    }

    pub fn ollama(mut self, model: impl Into<String>) -> Self {
        self.llm_url = "http://localhost:11434/v1".into();
        self.api_key = "ollama".into();
        self.model = model.into();
        self
    }

    pub fn openai(mut self, api_key: impl Into<String>, model: impl Into<String>) -> Self {
        self.llm_url = "https://api.openai.com/v1".into();
        self.api_key = api_key.into();
        self.model = model.into();
        self
    }

    pub fn model(
        mut self,
        url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        self.llm_url = url.into();
        self.api_key = api_key.into();
        self.model = model.into();
        self
    }

    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn system_prompt_opt(mut self, prompt: Option<String>) -> Self {
        self.system_prompt = prompt;
        self
    }

    pub fn work_dir_opt(mut self, path: Option<PathBuf>) -> Self {
        self.work_dir = path;
        self
    }

    pub fn max_steps(mut self, n: usize) -> Self {
        self.max_steps = n;
        self
    }

    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    pub fn memory(mut self, paths: Vec<PathBuf>) -> Self {
        self.memory_paths = paths;
        self
    }

    pub fn skills(mut self, dirs: Vec<PathBuf>) -> Self {
        self.skill_dirs = dirs;
        self
    }

    pub fn work_dir(mut self, path: PathBuf) -> Self {
        self.work_dir = Some(path);
        self
    }

    pub fn human_approval_for(mut self, tools: Vec<String>) -> Self {
        self.human_approval_tools = tools;
        self
    }

    pub fn require_approve_all(mut self, enabled: bool) -> Self {
        self.require_approve_all = enabled;
        self
    }

    pub fn subagent(mut self, spec: SubAgentSpec) -> Self {
        self.subagent_specs.push(spec);
        self
    }

    pub fn tool(mut self, tool: impl eventage::Tool + 'static) -> Self {
        self.extra_tools.push(Arc::new(tool));
        self
    }

    pub fn requests_per_minute(mut self, rpm: u32) -> Self {
        self.requests_per_minute = rpm;
        self
    }

    pub fn async_subagents(mut self, enabled: bool) -> Self {
        self.async_subagents = enabled;
        self
    }

    pub fn tui_mode(mut self, enabled: bool) -> Self {
        self.tui_mode = enabled;
        self
    }

    pub fn build(self) -> CodingAgent {
        let work_dir = self
            .work_dir
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

        let workspace = Arc::new(
            Workspace::open(&work_dir)
                .unwrap_or_else(|_| Workspace::open(".").expect("could not open workspace")),
        );

        let bus = EventBus::new();

        // ── LLM provider (branches on tui_mode) ─────────────────────────────
        let (llm, cancelled): (Arc<dyn eventage::llm::LlmProvider>, Option<Arc<AtomicBool>>) =
            if self.tui_mode {
                let streaming = StreamingOpenAiProvider::new(
                    &self.llm_url,
                    &self.api_key,
                    &self.model,
                    bus.clone(),
                );
                let flag = streaming.cancelled.clone();
                let base: Arc<dyn eventage::llm::LlmProvider> = Arc::new(streaming);
                let llm = if self.requests_per_minute > 0 {
                    Arc::new(RateLimitedProvider::from_arc(
                        base,
                        self.requests_per_minute,
                    )) as Arc<dyn eventage::llm::LlmProvider>
                } else {
                    base
                };
                (llm, Some(flag))
            } else {
                let base_llm = OpenAiProvider::new(&self.llm_url, &self.api_key, &self.model);
                let llm: Arc<dyn eventage::llm::LlmProvider> = if self.requests_per_minute > 0 {
                    Arc::new(RateLimitedProvider::new(base_llm, self.requests_per_minute))
                } else {
                    Arc::new(base_llm)
                };
                (llm, None)
            };

        let todo_state = TodoState::new();

        // Build sub-agent spec list
        let mut specs = vec![SubAgentSpec::general_purpose()];
        specs.extend(self.subagent_specs);

        let system_prompt = build_system_prompt(self.system_prompt.as_deref());

        // ── Assembler chain ──────────────────────────────────────────────────
        let base: Arc<dyn ContextAssembler> =
            Arc::new(DefaultContextAssembler::new(&system_prompt));

        // Use the LLM to classify user follow-up messages as behavioral instructions
        // and inject them as a sticky system message that survives summarization.
        let with_corrections: Arc<dyn ContextAssembler> =
            Arc::new(UserCorrectionsAssembler::new(base, llm.clone()));

        let with_summary: Arc<dyn ContextAssembler> = if self.max_tokens > 0 {
            Arc::new(SummarizingAssembler::new(
                with_corrections,
                llm.clone(),
                self.max_tokens,
                &self.session_id,
            ))
        } else {
            with_corrections
        };

        let with_skills: Arc<dyn ContextAssembler> = if self.skill_dirs.is_empty() {
            with_summary
        } else {
            Arc::new(SkillsAssembler::new(with_summary, &self.skill_dirs))
        };

        let final_assembler: Arc<dyn ContextAssembler> = if self.memory_paths.is_empty() {
            with_skills
        } else {
            Arc::new(MemoryAssembler::new(with_skills, &self.memory_paths))
        };

        // ── Build AgentBuilder ───────────────────────────────────────────────
        let mut builder = AgentBuilder::new()
            .bus(bus.clone())
            .llm_arc(llm.clone())
            .context(ArcAssembler(final_assembler))
            .strategy(ReactStrategy {
                max_steps: self.max_steps,
                max_concurrent_tools: 4,
                ..Default::default()
            });

        // ── Register standard tools ──────────────────────────────────────────
        builder = builder
            .tool(LsTool {
                work_dir: work_dir.clone(),
            })
            .tool(ReadFileTool {
                work_dir: work_dir.clone(),
            })
            .tool(WriteFileTool {
                work_dir: work_dir.clone(),
            })
            .tool(EditFileTool {
                work_dir: work_dir.clone(),
            })
            .tool(ApplyPatchTool {
                work_dir: work_dir.clone(),
            })
            .tool(GlobTool {
                work_dir: work_dir.clone(),
            })
            .tool(GrepTool {
                work_dir: work_dir.clone(),
            })
            .tool(RunCommandTool {
                work_dir: work_dir.clone(),
            })
            .tool(AddTodoTool {
                state: todo_state.clone(),
            })
            .tool(CompleteTodoTool {
                state: todo_state.clone(),
            })
            .tool(ListTodosTool { state: todo_state });

        // ── Sync task tool ───────────────────────────────────────────────────
        builder = builder.tool(TaskTool {
            llm: llm.clone(),
            base_system_prompt: system_prompt.clone(),
            specs: specs.clone(),
            max_steps: self.max_steps / 2,
            work_dir: work_dir.clone(),
        });

        // ── Async task tools (if enabled) ────────────────────────────────────
        if self.async_subagents {
            builder = builder
                .tool(LaunchAsyncTaskTool {
                    bus: bus.clone(),
                    specs: specs.clone(),
                })
                .tool(CheckAsyncTaskTool { bus: bus.clone() });
        }

        // ── Extra user-supplied tools ────────────────────────────────────────
        for tool in self.extra_tools {
            builder = builder.tool_arc(tool);
        }

        // ── Hook ─────────────────────────────────────────────────────────────
        // Same flags control both modes; only the UI implementation differs:
        // TUI uses an event-driven overlay, REPL uses a stdin prompt.
        if self.tui_mode {
            if self.require_approve_all {
                builder = builder.hook(SecurityGateHook::all_tools(bus.clone()));
            } else if !self.human_approval_tools.is_empty() {
                builder = builder.hook(SecurityGateHook::watched(bus.clone(), self.human_approval_tools));
            }
        } else if self.require_approve_all {
            builder = builder.hook(HumanApprovalHook::all_tools());
        } else if !self.human_approval_tools.is_empty() {
            builder = builder.hook(HumanApprovalHook::new(self.human_approval_tools));
        }

        let agent = builder.build();

        // ── Worker set ───────────────────────────────────────────────────────
        let mut ws = WorkerSet::new().add_worker(TurnDiffWorker::new(workspace.clone()));

        if self.async_subagents {
            ws = ws.add_worker(SubAgentWorker {
                llm: llm.clone(),
                specs,
                base_system_prompt: system_prompt,
                max_steps: self.max_steps / 2,
                work_dir: work_dir.clone(),
            });
        }

        info!(
            model = %self.model,
            work_dir = %work_dir.display(),
            tui_mode = self.tui_mode,
            async_subagents = self.async_subagents,
            "coding agent ready"
        );

        CodingAgent {
            agent,
            bus,
            worker_set: Some(ws),
            cancelled,
            workspace,
            model: self.model,
            session_id: self.session_id,
        }
    }
}

// ── ArcAssembler ─────────────────────────────────────────────────────────────

/// Newtype so Arc<dyn ContextAssembler> can be passed to AgentBuilder::context().
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
