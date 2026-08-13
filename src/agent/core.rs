use super::context::{AssemblyContext, ContextAssembler};
use super::error::AgentError;
use super::hook::CycleHook;
use super::strategy::{AgentContext, ExecutionStrategy};
use super::stuck::detect_stuck;
use super::tool::{ToolRegistry, ToolSelector};
use crate::event::{kinds, meta_keys, Event};
use crate::bus::EventBus;
use crate::llm::LlmProvider;
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{error, instrument, warn};
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

        // ── Stuck detection ───────────────────────────────────────────────────
        // Check the recent event log for loop patterns before committing to a
        // new cycle. If a pattern is found, publish a hint event so the LLM
        // can see it in context and try a different approach.
        if let Some(analysis) = detect_stuck(&initial_events, 10) {
            warn!(kind = ?analysis.kind, repeat_count = analysis.repeat_count, "stuck pattern detected");
            let _ = self
                .bus
                .publish(
                    Event::new(
                        kinds::AGENT_STUCK,
                        json!({
                            "kind": format!("{:?}", analysis.kind),
                            "repeat_count": analysis.repeat_count,
                            "hint": "You appear to be repeating the same action or error. \
                                     Try a different approach, use different arguments, \
                                     or ask the user for clarification.",
                        }),
                    )
                    .with_meta(meta_keys::AGENT_ID, json!(self.agent_id)),
                )
                .await;
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
        // Subscribe before inspecting the log so we cannot miss events published
        // between the log read and the subscribe call.
        let mut rx = self.bus.subscribe();

        // Consecutive-error backoff state.  After N errors in a row we skip wake
        // events until at least `2^(N-1)` seconds have elapsed since the last
        // failure.  This prevents hammering a failing LLM on every heartbeat or
        // rapid user message burst.  Caps at 32 s (2^5) so the agent can always
        // recover within one heartbeat interval.
        let mut consecutive_errors: u32 = 0;
        let mut last_error_at: Option<Instant> = None;

        let run_cycle = |consecutive_errors: &mut u32,
                         last_error_at: &mut Option<Instant>|
         -> bool {
            if *consecutive_errors == 0 {
                return true;
            }
            let wait = Duration::from_secs(2u64.pow((*consecutive_errors).min(6) - 1));
            last_error_at.is_none_or(|t| t.elapsed() >= wait)
        };

        // If wake events were published to the bus before this subscription was
        // registered — e.g. an HTTP message that arrived during startup before
        // the agent task was scheduled — run an initial cycle.
        let has_pending = self
            .bus
            .log()
            .await
            .iter()
            .any(|e| matches!(e.kind.as_str(), kinds::USER_MESSAGE | kinds::SYSTEM_HEARTBEAT | kinds::SYSTEM_MESSAGE));
        if has_pending {
            if let Err(e) = self.cycle().await {
                if matches!(e, AgentError::Bus(_)) {
                    return Err(e);
                }
                consecutive_errors += 1;
                last_error_at = Some(Instant::now());
                error!(error = %e, consecutive_errors, "agent cycle error — recovering");
                let _ = self
                    .bus
                    .publish(Event::new(
                        kinds::ASSISTANT_MESSAGE,
                        json!({ "content": format!("⚠️ Error: {e}. Ready for your next message.") }),
                    ))
                    .await;
            }
        }

        while let Some(event) = rx.recv().await {
            let wake = match event.kind.as_str() {
                kinds::USER_MESSAGE | kinds::SYSTEM_HEARTBEAT | kinds::SYSTEM_MESSAGE => true,
                kinds::AGENT_MESSAGE => event
                    .metadata
                    .get(meta_keys::TO_AGENT_ID)
                    .and_then(|v| v.as_str())
                    .is_none_or(|to| to == self.agent_id),
                _ => false,
            };
            if wake {
                if !run_cycle(&mut consecutive_errors, &mut last_error_at) {
                    continue;
                }
                if let Err(e) = self.cycle().await {
                    if matches!(e, AgentError::Bus(_)) {
                        return Err(e);
                    }
                    consecutive_errors += 1;
                    last_error_at = Some(Instant::now());
                    error!(error = %e, consecutive_errors, "agent cycle error — recovering");
                    let _ = self
                        .bus
                        .publish(Event::new(
                            kinds::ASSISTANT_MESSAGE,
                            json!({ "content": format!("⚠️ Error: {e}. Ready for your next message.") }),
                        ))
                        .await;
                } else {
                    consecutive_errors = 0;
                    last_error_at = None;
                }
            }
        }
        Ok(())
    }

    pub fn bus(&self) -> &EventBus {
        &self.bus
    }
}
