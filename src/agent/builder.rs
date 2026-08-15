use super::context::{ContextAssembler, RawContextAssembler, SystemPromptAssembler};
use super::core::Agent;
use super::hook::{CycleHook, HookChain};
use super::strategy::ExecutionStrategy;
use super::tool::{Tool, ToolRegistry, ToolSelector};
use crate::bus::EventBus;
use crate::llm::LlmProvider;
use std::sync::Arc;

/// Fluent builder for constructing an [`Agent`].
///
/// Requires an [`ExecutionStrategy`] to be provided.
///
/// # Example
///
/// ```rust,no_run
/// use eventage::AgentBuilder;
/// use eventage::EventBus;
/// use eventage::llm::MockLlmProvider;
/// use eventage::ReactStrategy;
///
/// let agent = AgentBuilder::new()
///     .agent_id("my-agent")
///     .bus(EventBus::default())
///     .llm(MockLlmProvider::with_texts(Vec::<&str>::new()))
///     .system_prompt("You are a helpful assistant.")
///     .strategy(ReactStrategy::default())
///     .build();
/// ```
#[derive(Default)]
pub struct AgentBuilder {
    agent_id: Option<String>,
    bus: Option<EventBus>,
    llm: Option<Arc<dyn LlmProvider>>,
    context: Option<Arc<dyn ContextAssembler>>,
    tools: ToolRegistry,
    tool_selector: Option<Arc<dyn ToolSelector>>,
    system_prompt: Option<String>,
    hooks: Vec<Arc<dyn CycleHook>>,
    strategy: Option<Arc<dyn ExecutionStrategy>>,
}

impl AgentBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets a stable identifier (defaults to a random UUID).
    pub fn agent_id(mut self, id: impl Into<String>) -> Self {
        self.agent_id = Some(id.into());
        self
    }

    pub fn bus(mut self, bus: EventBus) -> Self {
        self.bus = Some(bus);
        self
    }

    pub fn llm(mut self, llm: impl LlmProvider + 'static) -> Self {
        self.llm = Some(Arc::new(llm));
        self
    }

    pub fn llm_arc(mut self, llm: Arc<dyn LlmProvider>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Sets a basic system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Provides a custom [`ContextAssembler`], overriding `system_prompt`.
    pub fn context(mut self, context: impl ContextAssembler + 'static) -> Self {
        self.context = Some(Arc::new(context));
        self
    }

    /// Register a tool.
    pub fn tool(self, tool: impl Tool + 'static) -> Self {
        self.tools.register(Arc::new(tool));
        self
    }

    /// Register a pre-boxed tool.
    pub fn tool_arc(self, tool: Arc<dyn Tool>) -> Self {
        self.tools.register(tool);
        self
    }

    /// Attaches a [`ToolSelector`] to dynamically filter tools per step.
    pub fn tool_selector(mut self, selector: impl ToolSelector + 'static) -> Self {
        self.tool_selector = Some(Arc::new(selector));
        self
    }

    /// Returns a shared handle to the builder's tool registry.
    ///
    /// ```rust,no_run
    /// use eventage::AgentBuilder;
    /// use eventage::llm::MockLlmProvider;
    ///
    /// let builder = AgentBuilder::new()
    ///     .llm(MockLlmProvider::with_texts(Vec::<&str>::new()));
    ///
    /// let tools = builder.tool_registry(); // keep for runtime use
    /// let agent = builder.build();
    ///
    /// // tools.add_tool(NewTool);  // agent sees it immediately
    /// ```
    pub fn tool_registry(&self) -> ToolRegistry {
        self.tools.clone()
    }

    /// Attaches a [`CycleHook`]. Hooks run in registration order.
    pub fn hook(mut self, hook: impl CycleHook + 'static) -> Self {
        self.hooks.push(Arc::new(hook));
        self
    }

    /// Sets the required [`ExecutionStrategy`] (e.g. ReAct).
    pub fn strategy(mut self, strategy: impl ExecutionStrategy + 'static) -> Self {
        self.strategy = Some(Arc::new(strategy));
        self
    }

    /// Builds the agent. Panics if `llm` or `strategy` are missing.
    pub fn build(self) -> Agent {
        let agent_id = self
            .agent_id
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let bus = self.bus.unwrap_or_default();
        let llm = self.llm.expect("AgentBuilder: llm provider is required");
        let context: Arc<dyn ContextAssembler> = self.context.unwrap_or_else(|| {
            if let Some(prompt) = self.system_prompt {
                Arc::new(SystemPromptAssembler {
                    system_prompt: prompt,
                })
            } else {
                Arc::new(RawContextAssembler)
            }
        });
        let hooks: Arc<dyn CycleHook> = Arc::new(HookChain::new(self.hooks));
        let strategy = self.strategy.expect(
            "AgentBuilder: an execution strategy is required. \
             Call `.strategy(ReactStrategy::default())` to use the built-in ReAct strategy.",
        );

        Agent::new(
            agent_id,
            bus,
            llm,
            context,
            self.tools,
            self.tool_selector,
            hooks,
            strategy,
        )
    }
}
