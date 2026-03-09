//! Built-in execution strategies (`ReactStrategy`, `SingleShotStrategy`).

use async_trait::async_trait;
use eventage_agent::context::AssemblyContext;
use eventage_agent::error::AgentError;
use eventage_agent::hook::{HookAction, HookContext};
use eventage_agent::strategy::{execute_tools, AgentContext, ExecutionStrategy};
use eventage_core::kinds;
use serde_json::{json, Value};
use tracing::{info, warn};

/// Default max LLM sub-calls per cycle.
pub const DEFAULT_MAX_REACT_STEPS: usize = 20;

/// Default max concurrent tools per ReAct step.
pub const DEFAULT_MAX_CONCURRENT_TOOLS: usize = 4;

// ── ReactStrategy ─────────────────────────────────────────────────────────────

/// ReAct (Reason + Act) loop strategy.
///
/// Repeats `context → LLM → execute tools` until no tool calls or limit reached.
///
/// # Example
///
/// ```rust,no_run
/// use eventage_agent::AgentBuilder;
/// use eventage_provided_impl::ReactStrategy;
/// use eventage_llm::MockLlmProvider;
///
/// let agent = AgentBuilder::new()
///     .llm(MockLlmProvider::with_texts(Vec::<&str>::new()))
///     .strategy(ReactStrategy { max_steps: 5, max_concurrent_tools: 2 })
///     .build();
/// ```
pub struct ReactStrategy {
    /// Hard cap on LLM sub-calls per cycle.
    pub max_steps: usize,
    /// Maximum tools executing concurrently in one react step.
    pub max_concurrent_tools: usize,
}

impl Default for ReactStrategy {
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_MAX_REACT_STEPS,
            max_concurrent_tools: DEFAULT_MAX_CONCURRENT_TOOLS,
        }
    }
}

#[async_trait]
impl ExecutionStrategy for ReactStrategy {
    async fn execute(&self, ctx: &AgentContext) -> Result<(), AgentError> {
        let mut step = 0usize;
        loop {
            step += 1;
            if step > self.max_steps {
                warn!(
                    max_steps = self.max_steps,
                    "ReactStrategy: step limit reached — aborting cycle"
                );
                return Err(AgentError::MaxStepsReached(self.max_steps));
            }

            let hook_ctx = HookContext {
                agent_id: &ctx.agent_id,
                trace_id: &ctx.trace_id,
                step,
                bus: &ctx.bus,
            };

            // ── before_step hook ──────────────────────────────────────────────
            match ctx.hooks.before_step(&hook_ctx).await {
                HookAction::Continue => {}
                _ => return Ok(()),
            }

            // ── Assemble context ──────────────────────────────────────────────
            let events = ctx.bus.log().await;
            let assembly_ctx = AssemblyContext::new(&events);
            let mut messages = ctx.assembler.assemble(&assembly_ctx).await;
            if messages.is_empty() {
                return Ok(());
            }

            // ── before_llm hook (may mutate messages) ─────────────────────────
            match ctx.hooks.before_llm(&hook_ctx, &mut messages).await {
                HookAction::Continue => {}
                _ => return Ok(()),
            }

            // ── Select tools for this step ────────────────────────────────────
            let tool_defs = if ctx.tools.is_empty() {
                vec![]
            } else if let Some(sel) = &ctx.tool_selector {
                let all = ctx.tools.all_tools();
                let selected = sel.select(&all, &messages).await;
                selected.iter().map(|t| t.definition()).collect()
            } else {
                ctx.tools.definitions()
            };

            // ── LLM call ──────────────────────────────────────────────────────
            let response = ctx.llm.complete(messages, tool_defs).await?;

            let tool_calls_json: Vec<Value> = response
                .tool_calls
                .iter()
                .map(|tc| {
                    json!({
                        "id": tc.id,
                        "type": tc.kind,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
                })
                .collect();

            ctx.bus
                .publish(ctx.event(
                    kinds::ASSISTANT_MESSAGE,
                    json!({
                        "content": response.content,
                        "tool_calls": tool_calls_json
                    }),
                ))
                .await?;

            if let Some(text) = &response.content {
                info!("assistant: {}", text);
            }

            if response.tool_calls.is_empty() {
                return Ok(());
            }

            // ── Execute tools (hooks + bounded concurrency) ───────────────────
            let had_terminal = execute_tools(
                ctx,
                &response.tool_calls,
                &hook_ctx,
                self.max_concurrent_tools,
            )
            .await?;

            if had_terminal {
                return Ok(());
            }
            // Loop: tool results are now in the log; re-assemble and call LLM.
        }
    }
}

// ── SingleShotStrategy ────────────────────────────────────────────────────────

/// A single LLM call with no tool execution.
///
/// Suitable for classification, summarization, or initial planning.
///
/// # Example
///
/// ```rust,no_run
/// use eventage_agent::AgentBuilder;
/// use eventage_provided_impl::SingleShotStrategy;
/// use eventage_llm::MockLlmProvider;
///
/// let agent = AgentBuilder::new()
///     .llm(MockLlmProvider::with_texts(Vec::<&str>::new()))
///     .strategy(SingleShotStrategy)
///     .build();
/// ```
pub struct SingleShotStrategy;

#[async_trait]
impl ExecutionStrategy for SingleShotStrategy {
    async fn execute(&self, ctx: &AgentContext) -> Result<(), AgentError> {
        let events = ctx.bus.log().await;
        let assembly_ctx = AssemblyContext::new(&events);
        let messages = ctx.assembler.assemble(&assembly_ctx).await;
        if messages.is_empty() {
            return Ok(());
        }

        let tool_defs = if ctx.tools.is_empty() {
            vec![]
        } else {
            ctx.tools.definitions()
        };

        let response = ctx.llm.complete(messages, tool_defs).await?;

        let tool_calls_json: Vec<Value> = response
            .tool_calls
            .iter()
            .map(|tc| {
                json!({
                    "id": tc.id,
                    "type": tc.kind,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })
            })
            .collect();

        ctx.bus
            .publish(ctx.event(
                kinds::ASSISTANT_MESSAGE,
                json!({
                    "content": response.content,
                    "tool_calls": tool_calls_json
                }),
            ))
            .await?;

        if let Some(text) = &response.content {
            info!("assistant: {}", text);
        }

        Ok(())
    }
}
