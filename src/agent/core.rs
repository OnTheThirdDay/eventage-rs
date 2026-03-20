use super::context::{AssemblyContext, ContextAssembler};
use super::error::AgentError;
use super::hook::CycleHook;
use super::strategy::{AgentContext, ExecutionStrategy};
use super::tool::{ToolRegistry, ToolSelector};
use crate::event::{kinds, meta_keys, Event};
use crate::bus::EventBus;
use crate::llm::LlmProvider;
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use tracing::instrument;
use uuid::Uuid;

/// The orchestrator driving tool execution and LLM reasoning.
///
/// Emits `agent.cycle.start` and `agent.cycle.end` around execution.
/// The reasoning logic is delegated to an [`ExecutionStrategy`].
/// Construct via [`crate::AgentBuilder`].
pub struct Agent {
    /// Unique identifier for this agent.
    pub agent_id: String,
    bus: EventBus,
    llm: Arc<dyn LlmProvider>,
    context: Arc<dyn ContextAssembler>,
    tools: ToolRegistry,
    tool_selector: Option<Arc<dyn ToolSelector>>,
    hooks: Arc<dyn CycleHook>,
    strategy: Arc<dyn ExecutionStrategy>,
}

impl Agent {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        agent_id: String,
        bus: EventBus,
        llm: Arc<dyn LlmProvider>,
        context: Arc<dyn ContextAssembler>,
        tools: ToolRegistry,
        tool_selector: Option<Arc<dyn ToolSelector>>,
        hooks: Arc<dyn CycleHook>,
        strategy: Arc<dyn ExecutionStrategy>,
    ) -> Self {
        Self {
            agent_id,
            bus,
            llm,
            context,
            tools,
            tool_selector,
            hooks,
            strategy,
        }
    }

    /// Returns a shared handle to the agent's [`ToolRegistry`].
    /// Mutating this registry affects the agent immediately.
    pub fn tools(&self) -> ToolRegistry {
        self.tools.clone()
    }

    /// Executes a single reasoning cycle via the configured strategy.
    #[instrument(skip(self), fields(agent_id = %self.agent_id), name = "agent_cycle")]
    pub async fn cycle(&self) -> Result<(), AgentError> {
        // Quick check: if there is no context to work with, skip the cycle.
        let initial_events = self.bus.log().await;
        let initial_ctx = AssemblyContext::new(&initial_events);
        let initial_messages = self.context.assemble(&initial_ctx).await;
        if initial_messages.is_empty() {
            return Ok(());
        }

        let trace_id = Uuid::new_v4().to_string();
        let started_at = Instant::now();

        let cycle_start = Event::new(kinds::AGENT_CYCLE_START, json!({}))
            .with_meta(meta_keys::AGENT_ID, json!(self.agent_id))
            .with_meta(meta_keys::TRACE_ID, json!(&trace_id));
        self.bus.publish(cycle_start).await?;

        let agent_ctx = AgentContext {
            agent_id: self.agent_id.clone(),
            trace_id: trace_id.clone(),
            bus: self.bus.clone(),
            llm: self.llm.clone(),
            assembler: self.context.clone(),
            tools: self.tools.clone(),
            tool_selector: self.tool_selector.clone(),
            hooks: self.hooks.clone(),
        };

        let result = self.strategy.execute(&agent_ctx).await;

        let elapsed_ms = started_at.elapsed().as_millis() as u64;
        let cycle_end = Event::new(kinds::AGENT_CYCLE_END, json!({}))
            .with_meta(meta_keys::AGENT_ID, json!(self.agent_id))
            .with_meta(meta_keys::TRACE_ID, json!(&trace_id))
            .with_meta(meta_keys::ELAPSED_MS, json!(elapsed_ms));
        self.bus.publish(cycle_end).await?;

        result
    }

    /// Continuously listens and reacts to incoming events.
    pub async fn run(&self) -> Result<(), AgentError> {
        let mut rx = self.bus.subscribe();
        while let Some(event) = rx.recv().await {
            let wake = match event.kind.as_str() {
                kinds::USER_MESSAGE | kinds::SYSTEM_HEARTBEAT => true,
                kinds::AGENT_MESSAGE => event
                    .metadata
                    .get(meta_keys::TO_AGENT_ID)
                    .and_then(|v| v.as_str())
                    .is_none_or(|to| to == self.agent_id),
                _ => false,
            };
            if wake {
                self.cycle().await?;
            }
        }
        Ok(())
    }

    pub fn bus(&self) -> &EventBus {
        &self.bus
    }
}
