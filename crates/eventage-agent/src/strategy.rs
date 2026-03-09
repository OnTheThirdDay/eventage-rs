//! Execution strategies orchestrating the agent's core loop.
//!
//! An [`ExecutionStrategy`] receives handles via [`AgentContext`] to drive execution.

use crate::context::ContextAssembler;
use crate::error::AgentError;
use crate::hook::{HookAction, HookContext};
use crate::tool::{ToolRegistry, ToolSelector};
use async_trait::async_trait;
use eventage_core::{kinds, meta_keys, Event, EventBus};
use eventage_llm::{types::ToolCall, LlmProvider};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::warn;

// ── AgentContext ──────────────────────────────────────────────────────────────

/// A complete set of cheap-to-clone handles needed to execute a reasoning cycle.
pub struct AgentContext {
    /// Stable agent identifier — appears in every event's metadata.
    pub agent_id: String,
    /// Per-cycle correlation UUID — stamped on every event emitted this cycle.
    pub trace_id: String,
    /// The shared event bus: publish events, query the log, use `wait_for`.
    pub bus: EventBus,
    /// LLM provider for chat completions.
    pub llm: Arc<dyn LlmProvider>,
    /// Context assembler: converts the event log into a `Vec<ChatMessage>`.
    pub assembler: Arc<dyn ContextAssembler>,
    /// Live tool registry — strategies can add/remove tools at runtime.
    pub tools: ToolRegistry,
    /// Optional per-step tool filter (may narrow what the LLM sees).
    pub tool_selector: Option<Arc<dyn ToolSelector>>,
    /// Lifecycle hooks — before/after step, LLM, and tool calls.
    pub hooks: Arc<dyn crate::hook::CycleHook>,
}

impl AgentContext {
    /// Build an event pre-tagged with this cycle's `agent_id` and `trace_id`.
    pub fn event(&self, kind: &str, payload: Value) -> Event {
        Event::new(kind, payload)
            .with_meta(meta_keys::AGENT_ID, json!(self.agent_id))
            .with_meta(meta_keys::TRACE_ID, json!(self.trace_id))
    }

    /// Spawns a new agent operating on the same event bus.
    pub fn spawn_agent(
        &self,
        agent: crate::agent::Agent,
    ) -> tokio::task::JoinHandle<Result<(), crate::error::AgentError>> {
        let bus = self.bus.clone();
        let agent_id = agent.agent_id.clone();
        tokio::spawn(async move {
            bus.publish(eventage_core::Event::new(
                eventage_core::kinds::AGENT_SPAWNED,
                serde_json::json!({ "agent_id": agent_id }),
            ))
            .await
            .ok();
            agent.run().await
        })
    }
}

// ── ExecutionStrategy trait ───────────────────────────────────────────────────

/// Defines the cognitive architecture of an agent (e.g., ReAct, Planning).
///
/// Called per cycle to orchestrate the LLM, tools, and message bus.
#[async_trait]
pub trait ExecutionStrategy: Send + Sync {
    /// Drive reasoning for one cycle using the provided context.
    async fn execute(&self, ctx: &AgentContext) -> Result<(), AgentError>;
}

// ── Shared tool execution logic ────────────────────────────────────────────────

/// Executes tool calls concurrently, returning `true` if any tool was terminal.
pub async fn execute_tools(
    ctx: &AgentContext,
    calls: &[ToolCall],
    hook_ctx: &HookContext<'_>,
    max_concurrent: usize,
) -> Result<bool, AgentError> {
    struct ToolPlan {
        id: String,
        name: String,
        args: Value,
        skipped: bool,
        is_terminal: bool,
    }

    // ── Phase 1: sequential pre-flight ────────────────────────────────────
    let mut plan: Vec<ToolPlan> = Vec::with_capacity(calls.len());
    for tc in calls {
        ctx.bus
            .publish(ctx.event(
                kinds::TOOL_CALL_PROPOSED,
                json!({
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }),
            ))
            .await?;

        let args: Value = serde_json::from_str(&tc.function.arguments).unwrap_or(Value::Null);

        let action = ctx
            .hooks
            .before_tool(hook_ctx, &tc.function.name, &args)
            .await;
        let skipped = !matches!(action, HookAction::Continue);

        let is_terminal = !skipped
            && ctx
                .tools
                .get(&tc.function.name)
                .is_some_and(|t| t.is_terminal());

        plan.push(ToolPlan {
            id: tc.id.clone(),
            name: tc.function.name.clone(),
            args,
            skipped,
            is_terminal,
        });
    }

    // ── Phase 2: bounded concurrent execution ─────────────────────────────
    let sem = Arc::new(Semaphore::new(max_concurrent));
    let mut join_set: JoinSet<(usize, Value)> = JoinSet::new();

    for (i, p) in plan.iter().enumerate() {
        if p.skipped {
            continue;
        }
        let sem = sem.clone();
        let tool = ctx.tools.get(&p.name);
        let tc_id = p.id.clone();
        let tc_name = p.name.clone();
        let args = p.args.clone();

        join_set.spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let payload = match tool {
                None => {
                    warn!("tool '{}' not found in registry", tc_name);
                    json!({
                        "tool_call_id": tc_id,
                        "name": tc_name,
                        "error": format!("tool '{}' not registered", tc_name)
                    })
                }
                Some(t) => match t.execute(args).await {
                    Ok(r) => json!({
                        "tool_call_id": tc_id,
                        "name": tc_name,
                        "result": r
                    }),
                    Err(e) => {
                        warn!("tool '{}' returned error: {}", tc_name, e);
                        json!({
                            "tool_call_id": tc_id,
                            "name": tc_name,
                            "error": e.to_string()
                        })
                    }
                },
            };
            (i, payload)
        });
    }

    // Collect by original index (preserves order for Phase 3).
    let mut exec_results: HashMap<usize, Value> = HashMap::new();
    while let Some(join_result) = join_set.join_next().await {
        let (idx, payload) = join_result.map_err(|e| AgentError::Tool(e.to_string()))?;
        exec_results.insert(idx, payload);
    }

    // ── Phase 3: after_tool hooks + publish results in original order ──────
    let mut had_terminal = false;
    for (i, p) in plan.iter().enumerate() {
        let result_payload = if p.skipped {
            json!({
                "tool_call_id": p.id,
                "name": p.name,
                "result": { "skipped": true, "reason": "vetoed by hook" }
            })
        } else {
            exec_results.remove(&i).unwrap_or_else(|| {
                json!({
                    "tool_call_id": p.id,
                    "name": p.name,
                    "error": "execution result missing"
                })
            })
        };

        let result_val = result_payload
            .get("result")
            .cloned()
            .unwrap_or(result_payload.clone());
        ctx.hooks.after_tool(hook_ctx, &p.name, &result_val).await;

        ctx.bus
            .publish(ctx.event(kinds::TOOL_RESULT, result_payload))
            .await?;

        if p.is_terminal {
            had_terminal = true;
        }
    }

    Ok(had_terminal)
}
