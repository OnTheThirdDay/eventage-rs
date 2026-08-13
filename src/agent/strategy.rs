//! Execution strategies orchestrating the agent's core loop.
//!
//! An [`ExecutionStrategy`] receives handles via [`AgentContext`] to drive execution.

use super::context::{AssemblyContext, ContextAssembler};
use super::error::AgentError;
use super::hook::{CycleHook, HookAction, HookContext};
use super::tool::{ToolRegistry, ToolSelector};
use crate::bus::EventBus;
use crate::event::{kinds, meta_keys, Event, EventId};
use crate::llm::{types::ToolCall, LlmProvider};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Serialize a tool call to the JSON shape stored in bus events.
/// Preserves `extra_content` so provider-specific metadata (e.g. Gemini's
/// thought_signature) is round-tripped correctly through the event bus.
fn tool_call_to_json(tc: &ToolCall) -> Value {
    let mut obj = json!({
        "id": tc.id,
        "type": tc.kind,
        "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments
        }
    });
    if let Some(ref extra) = tc.extra_content {
        obj["extra_content"] = extra.clone();
    }
    obj
}
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::{info, warn};

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
    pub hooks: Arc<dyn CycleHook>,
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
        agent: super::core::Agent,
    ) -> tokio::task::JoinHandle<Result<(), AgentError>> {
        let bus = self.bus.clone();
        let agent_id = agent.agent_id.clone();
        tokio::spawn(async move {
            bus.publish(Event::new(
                kinds::AGENT_SPAWNED,
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

/// Runtime guardrails applied by [`execute_tools`] to every tool call.
#[derive(Debug, Clone)]
pub struct ToolExecOptions {
    /// Maximum tools executing concurrently in one step.
    pub max_concurrent: usize,
    /// Wall-clock limit per tool call. A tool exceeding it produces an error
    /// `tool.result` (visible to the model) instead of hanging the cycle.
    /// `None` disables the limit.
    pub timeout: Option<std::time::Duration>,
    /// Maximum serialized size (in chars) of a tool result kept in the
    /// context payload. Oversized results are middle-truncated with an
    /// explanatory marker; the size cap also protects the event log from
    /// pathological outputs. `None` disables truncation.
    pub max_result_chars: Option<usize>,
    /// Validate parsed arguments against the tool's JSON Schema
    /// (`type`/`required`/`properties`/`items`/`enum`) before execution.
    /// Violations are fed back to the model as tool errors.
    pub validate_args: bool,
}

impl Default for ToolExecOptions {
    fn default() -> Self {
        Self {
            max_concurrent: DEFAULT_MAX_CONCURRENT_TOOLS,
            timeout: Some(std::time::Duration::from_secs(DEFAULT_TOOL_TIMEOUT_SECS)),
            max_result_chars: Some(DEFAULT_MAX_TOOL_RESULT_CHARS),
            validate_args: true,
        }
    }
}

/// Middle-truncate `s` to at most `max_chars`, keeping the head and tail
/// (where the useful signal of most tool outputs lives) and inserting a
/// marker stating how much was elided.
pub fn truncate_middle(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        return s.to_string();
    }
    // Keep ~70% head, ~30% tail (both clamped to char boundaries).
    let head_target = (max_chars * 7) / 10;
    let tail_target = max_chars.saturating_sub(head_target);

    let mut head_end = head_target.min(s.len());
    while !s.is_char_boundary(head_end) {
        head_end -= 1;
    }
    let mut tail_start = s.len().saturating_sub(tail_target);
    while !s.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    let elided = tail_start.saturating_sub(head_end);
    format!(
        "{}\n…[{} of {} chars elided by the harness — full output is in the event log; re-run the tool with narrower arguments if you need the elided part]…\n{}",
        &s[..head_end],
        elided,
        s.len(),
        &s[tail_start..]
    )
}

/// Apply `max_result_chars` to a successful tool result value.
fn cap_result_value(value: Value, max_chars: Option<usize>) -> Value {
    let Some(max) = max_chars else { return value };
    // Fast path: small values pass through untouched.
    let serialized = value.to_string();
    if serialized.len() <= max {
        return value;
    }
    // Truncate the *rendered* form: models see `result.to_string()` anyway,
    // and a truncated string stays valid JSON in the payload.
    let rendered = match value {
        Value::String(s) => s,
        other => other.to_string(),
    };
    Value::String(truncate_middle(&rendered, max))
}

/// Executes tool calls concurrently, returning `true` if any tool was terminal.
pub async fn execute_tools(
    ctx: &AgentContext,
    calls: &[ToolCall],
    hook_ctx: &HookContext<'_>,
    opts: &ToolExecOptions,
) -> Result<bool, AgentError> {
    /// Why a planned call will not be executed.
    enum Veto {
        /// Hook returned `Skip`/`AbortCycle` — generic veto.
        Skipped,
        /// Hook returned `Deny(reason)` — reason is surfaced to the model.
        Denied(String),
        /// Arguments were not valid JSON — parse error surfaced to the model.
        BadArgs(String),
    }

    struct ToolPlan {
        id: String,
        name: String,
        args: Value,
        veto: Option<Veto>,
        is_terminal: bool,
        /// ID of the TOOL_CALL_PROPOSED event published for this call.
        /// Used to set `parent_event_id` on the corresponding TOOL_RESULT event,
        /// maintaining the causal chain in the event DAG.
        proposed_event_id: EventId,
    }

    // ── Phase 1: sequential pre-flight ────────────────────────────────────
    let mut plan: Vec<ToolPlan> = Vec::with_capacity(calls.len());
    for tc in calls {
        let proposed = ctx.event(
            kinds::TOOL_CALL_PROPOSED,
            json!({
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments
            }),
        );
        let proposed_event_id = proposed.id;
        ctx.bus.publish(proposed).await?;

        // Malformed arguments never reach the tool — the parse error goes back
        // to the model as a tool error so it can correct itself on the next step.
        let (args, veto) = match serde_json::from_str::<Value>(&tc.function.arguments) {
            Ok(v) => (v, None),
            Err(e) => (
                Value::Null,
                Some(Veto::BadArgs(format!(
                    "invalid JSON in tool arguments: {e}. \
                     Re-issue the call with arguments as a single valid JSON object."
                ))),
            ),
        };

        // Schema validation: catch structurally wrong arguments before they
        // reach the tool, and phrase the violation for the model.
        let veto = match veto {
            Some(v) => Some(v),
            None if opts.validate_args => ctx.tools.get(&tc.function.name).and_then(|tool| {
                let schema = tool.definition().function.parameters;
                crate::schema::validate_args(&schema, &args).err().map(|e| {
                    Veto::BadArgs(format!("{e}. Re-issue the call with corrected arguments."))
                })
            }),
            None => None,
        };

        let veto = match veto {
            Some(v) => Some(v),
            None => match ctx
                .hooks
                .before_tool(hook_ctx, &tc.function.name, &args)
                .await
            {
                HookAction::Continue => None,
                HookAction::Deny(reason) => Some(Veto::Denied(reason)),
                _ => Some(Veto::Skipped),
            },
        };

        let is_terminal = veto.is_none()
            && ctx
                .tools
                .get(&tc.function.name)
                .is_some_and(|t| t.is_terminal());

        plan.push(ToolPlan {
            id: tc.id.clone(),
            name: tc.function.name.clone(),
            args,
            veto,
            is_terminal,
            proposed_event_id,
        });
    }

    // ── Phase 2: bounded concurrent execution ─────────────────────────────
    let sem = Arc::new(Semaphore::new(opts.max_concurrent));
    let mut join_set: JoinSet<(usize, Value)> = JoinSet::new();

    for (i, p) in plan.iter().enumerate() {
        if p.veto.is_some() {
            continue;
        }
        let sem = sem.clone();
        let tool = ctx.tools.get(&p.name);
        let tc_id = p.id.clone();
        let tc_name = p.name.clone();
        let args = p.args.clone();
        let timeout = opts.timeout;
        let max_result_chars = opts.max_result_chars;

        join_set.spawn(async move {
            // `_permit` keeps the semaphore slot held for the duration of this task.
            // Note: `let _permit = x` (named binding) is different from `let _ = x` (immediate drop).
            let _permit = sem.acquire().await.expect("concurrency semaphore closed");
            let payload = match tool {
                None => {
                    warn!("tool '{}' not found in registry", tc_name);
                    json!({
                        "tool_call_id": tc_id,
                        "name": tc_name,
                        "error": format!("tool '{}' not registered", tc_name)
                    })
                }
                Some(t) => {
                    let exec = t.execute(args);
                    let outcome = match timeout {
                        Some(limit) => match tokio::time::timeout(limit, exec).await {
                            Ok(r) => r,
                            Err(_) => Err(AgentError::ToolTimeout {
                                name: tc_name.clone(),
                                secs: limit.as_secs(),
                            }),
                        },
                        None => exec.await,
                    };
                    match outcome {
                        Ok(r) => json!({
                            "tool_call_id": tc_id,
                            "name": tc_name,
                            "result": cap_result_value(r, max_result_chars)
                        }),
                        Err(e) => {
                            warn!("tool '{}' returned error: {}", tc_name, e);
                            json!({
                                "tool_call_id": tc_id,
                                "name": tc_name,
                                "error": e.to_string()
                            })
                        }
                    }
                }
            };
            (i, payload)
        });
    }

    // Collect by original index (preserves order for Phase 3).
    // JoinErrors (task panics) are logged and skipped — Phase 3 will synthesize an error
    // tool.result for any index missing from this map, keeping the message history valid.
    let mut exec_results: HashMap<usize, Value> = HashMap::new();
    while let Some(join_result) = join_set.join_next().await {
        match join_result {
            Ok((idx, payload)) => {
                exec_results.insert(idx, payload);
            }
            Err(join_err) => warn!("tool task panicked: {}", join_err),
        }
    }

    // ── Phase 3: after_tool hooks + publish results in original order ──────
    let mut had_terminal = false;
    for (i, p) in plan.iter().enumerate() {
        let result_payload = match &p.veto {
            Some(Veto::Skipped) => json!({
                "tool_call_id": p.id,
                "name": p.name,
                "result": { "skipped": true, "reason": "vetoed by hook" }
            }),
            Some(Veto::Denied(reason)) => json!({
                "tool_call_id": p.id,
                "name": p.name,
                "result": { "denied": true, "reason": reason }
            }),
            Some(Veto::BadArgs(error)) => json!({
                "tool_call_id": p.id,
                "name": p.name,
                "error": error
            }),
            None => exec_results.remove(&i).unwrap_or_else(|| {
                json!({
                    "tool_call_id": p.id,
                    "name": p.name,
                    "error": "tool execution panicked and produced no result"
                })
            }),
        };

        let result_val = result_payload
            .get("result")
            .cloned()
            .unwrap_or(result_payload.clone());
        ctx.hooks.after_tool(hook_ctx, &p.name, &result_val).await;

        // Link this result back to its originating call in the event DAG.
        let result_event = ctx
            .event(kinds::TOOL_RESULT, result_payload)
            .with_parent(p.proposed_event_id);
        ctx.bus.publish(result_event).await?;

        if p.is_terminal {
            had_terminal = true;
        }
    }

    Ok(had_terminal)
}

// ── ReactStrategy ─────────────────────────────────────────────────────────────

/// Default max LLM sub-calls per cycle.
pub const DEFAULT_MAX_REACT_STEPS: usize = 20;

/// Default max concurrent tools per ReAct step.
pub const DEFAULT_MAX_CONCURRENT_TOOLS: usize = 4;

/// Default wall-clock limit per tool call (seconds).
pub const DEFAULT_TOOL_TIMEOUT_SECS: u64 = 300;

/// Default cap on the serialized size of one tool result (chars).
pub const DEFAULT_MAX_TOOL_RESULT_CHARS: usize = 30_000;

/// ReAct (Reason + Act) loop strategy.
///
/// Repeats `context → LLM → execute tools` until no tool calls or limit reached.
///
/// Guardrails are on by default: per-tool timeouts, tool-result size caps,
/// and a graceful wrap-up turn when the step budget is exhausted. Set the
/// corresponding fields to `None` / `false` to opt out.
///
/// # Example
///
/// ```rust,no_run
/// use eventage::AgentBuilder;
/// use eventage::ReactStrategy;
/// use eventage::llm::MockLlmProvider;
///
/// let agent = AgentBuilder::new()
///     .llm(MockLlmProvider::with_texts(Vec::<&str>::new()))
///     .strategy(ReactStrategy { max_steps: 5, max_concurrent_tools: 2, ..Default::default() })
///     .build();
/// ```
pub struct ReactStrategy {
    /// Hard cap on LLM sub-calls per cycle.
    pub max_steps: usize,
    /// Maximum tools executing concurrently in one react step.
    pub max_concurrent_tools: usize,
    /// Wall-clock limit per tool call. Exceeding it produces an error
    /// `tool.result` visible to the model instead of hanging the cycle.
    pub tool_timeout: Option<std::time::Duration>,
    /// Cap on the serialized size of a single tool result kept in context.
    /// Oversized results are middle-truncated with an explanatory marker.
    pub max_tool_result_chars: Option<usize>,
    /// When the step budget runs out, make one final tool-free LLM call so
    /// the agent wraps up with a coherent answer (progress, remaining work,
    /// blockers) instead of erroring with [`AgentError::MaxStepsReached`].
    pub finalize_on_max_steps: bool,
    /// Stream completions via [`LlmProvider::complete_stream`], broadcasting
    /// **ephemeral** `assistant.delta` events (see [`EventBus::broadcast`]) as
    /// tokens arrive. The durable `assistant.message` event is unchanged.
    /// Providers without native streaming fall back to one delta per response.
    pub stream: bool,
}

impl Default for ReactStrategy {
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_MAX_REACT_STEPS,
            max_concurrent_tools: DEFAULT_MAX_CONCURRENT_TOOLS,
            tool_timeout: Some(std::time::Duration::from_secs(DEFAULT_TOOL_TIMEOUT_SECS)),
            max_tool_result_chars: Some(DEFAULT_MAX_TOOL_RESULT_CHARS),
            finalize_on_max_steps: true,
            stream: false,
        }
    }
}

/// Run the LLM call for a step, streaming deltas as ephemeral
/// `assistant.delta` broadcasts when `stream` is enabled.
///
/// Returns the response together with the harness's pre-call token estimate,
/// which is recorded on the resulting event so
/// [`TokenCalibration`](super::tokens::TokenCalibration) can learn the
/// estimator's error from the provider's real usage numbers.
async fn call_llm(
    ctx: &AgentContext,
    messages: Vec<crate::llm::ChatMessage>,
    tool_defs: Vec<crate::llm::ToolDefinition>,
    stream: bool,
) -> Result<(crate::llm::LlmResponse, usize), AgentError> {
    let estimated = super::tokens::messages_token_count(&messages);
    if !stream {
        let response = ctx.llm.complete(messages, tool_defs).await?;
        return Ok((response, estimated));
    }
    let bus = ctx.bus.clone();
    let agent_id = ctx.agent_id.clone();
    let trace_id = ctx.trace_id.clone();
    let on_delta: crate::llm::types::DeltaHandler = Arc::new(move |delta| {
        let mut payload = json!({});
        if let Some(text) = delta.content {
            payload["content"] = json!(text);
        }
        if let Some(text) = delta.reasoning_content {
            payload["reasoning_content"] = json!(text);
        }
        bus.broadcast(
            Event::new(kinds::ASSISTANT_DELTA, payload)
                .with_meta(meta_keys::AGENT_ID, json!(agent_id))
                .with_meta(meta_keys::TRACE_ID, json!(trace_id)),
        );
    });
    let response = ctx
        .llm
        .complete_stream(messages, tool_defs, on_delta)
        .await?;
    Ok((response, estimated))
}

/// Publish an `assistant.message` event for `response`, carrying reasoning
/// content (when present) and token-usage metadata.
async fn publish_assistant_message(
    ctx: &AgentContext,
    response: &crate::llm::LlmResponse,
    estimated_input_tokens: usize,
    extra: Option<(&str, Value)>,
) -> Result<(), AgentError> {
    let tool_calls_json: Vec<Value> = response.tool_calls.iter().map(tool_call_to_json).collect();

    let mut payload = json!({
        "content": response.content,
        "tool_calls": tool_calls_json
    });
    if let Some(reasoning) = &response.reasoning_content {
        payload["reasoning_content"] = json!(reasoning);
    }
    if let Some(provider_extra) = &response.provider_extra {
        payload["provider_extra"] = provider_extra.clone();
    }
    if let Some((key, value)) = extra {
        payload[key] = value;
    }

    let mut msg_event = ctx
        .event(kinds::ASSISTANT_MESSAGE, payload)
        .with_meta(
            meta_keys::LLM_INPUT_TOKENS,
            json!(response.input_tokens.unwrap_or(0)),
        )
        .with_meta(
            meta_keys::LLM_OUTPUT_TOKENS,
            json!(response.output_tokens.unwrap_or(0)),
        )
        .with_meta(
            meta_keys::LLM_ESTIMATED_INPUT_TOKENS,
            json!(estimated_input_tokens),
        );
    if let Some(cached) = response.cached_input_tokens {
        msg_event = msg_event.with_meta(meta_keys::LLM_CACHED_INPUT_TOKENS, json!(cached));
    }
    ctx.bus.publish(msg_event).await?;
    Ok(())
}

#[async_trait]
impl ExecutionStrategy for ReactStrategy {
    async fn execute(&self, ctx: &AgentContext) -> Result<(), AgentError> {
        let mut step = 0usize;
        loop {
            step += 1;
            if step > self.max_steps {
                if !self.finalize_on_max_steps {
                    warn!(
                        max_steps = self.max_steps,
                        "ReactStrategy: step limit reached — aborting cycle"
                    );
                    return Err(AgentError::MaxStepsReached(self.max_steps));
                }
                warn!(
                    max_steps = self.max_steps,
                    "ReactStrategy: step limit reached — requesting final wrap-up answer"
                );
                return self.finalize(ctx).await;
            }

            let opts = ToolExecOptions {
                max_concurrent: self.max_concurrent_tools,
                timeout: self.tool_timeout,
                max_result_chars: self.max_tool_result_chars,
                ..Default::default()
            };
            match run_react_step(ctx, step, &opts, self.stream).await? {
                StepOutcome::Done => return Ok(()),
                // Tool results are now in the log; re-assemble and call again.
                StepOutcome::Continue => {}
            }
        }
    }
}

/// What one ReAct step decided about the cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOutcome {
    /// The cycle is finished (final answer, terminal tool, or hook veto).
    Done,
    /// Tool results were produced; another step should run.
    Continue,
}

/// Run a single ReAct step: hooks → assemble → LLM → publish → execute tools.
///
/// This is the primitive the ReAct loop iterates, exposed so alternative
/// search strategies (see [`beam_search`](super::speculate::beam_search)) can
/// drive step-by-step exploration over forked buses.
pub async fn run_react_step(
    ctx: &AgentContext,
    step: usize,
    opts: &ToolExecOptions,
    stream: bool,
) -> Result<StepOutcome, AgentError> {
    let hook_ctx = HookContext {
        agent_id: &ctx.agent_id,
        trace_id: &ctx.trace_id,
        step,
        bus: &ctx.bus,
    };

    // ── before_step hook ──────────────────────────────────────────────
    match ctx.hooks.before_step(&hook_ctx).await {
        HookAction::Continue => {}
        _ => return Ok(StepOutcome::Done),
    }

    // ── Assemble context ──────────────────────────────────────────────
    let events = ctx.bus.log().await;
    let assembly_ctx = AssemblyContext::new(&events);
    let mut messages = ctx.assembler.assemble(&assembly_ctx).await;
    if messages.is_empty() {
        return Ok(StepOutcome::Done);
    }

    // ── before_llm hook (may mutate messages) ─────────────────────────
    match ctx.hooks.before_llm(&hook_ctx, &mut messages).await {
        HookAction::Continue => {}
        _ => return Ok(StepOutcome::Done),
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
    let (response, estimated) = call_llm(ctx, messages, tool_defs, stream).await?;

    publish_assistant_message(ctx, &response, estimated, None).await?;

    if let Some(text) = &response.content {
        info!("assistant: {}", text);
    }

    if response.tool_calls.is_empty() {
        return Ok(StepOutcome::Done);
    }

    // ── Execute tools (hooks + bounded concurrency) ───────────────────
    let had_terminal = execute_tools(ctx, &response.tool_calls, &hook_ctx, opts).await?;

    Ok(if had_terminal {
        StepOutcome::Done
    } else {
        StepOutcome::Continue
    })
}

impl ReactStrategy {
    /// Step budget exhausted: run one last LLM call *without tools*, nudged to
    /// wrap up, and publish the answer. The nudge is appended only to the
    /// in-flight message list (not the event log) so it cannot re-wake agents
    /// or pollute future turns.
    async fn finalize(&self, ctx: &AgentContext) -> Result<(), AgentError> {
        let events = ctx.bus.log().await;
        let assembly_ctx = AssemblyContext::new(&events);
        let mut messages = ctx.assembler.assemble(&assembly_ctx).await;
        if messages.is_empty() {
            return Err(AgentError::MaxStepsReached(self.max_steps));
        }
        messages.push(
            crate::llm::ChatMessage::user(
                "[harness] The step budget for this task is exhausted; no more tool \
                 calls will be executed. Write your final answer now: state what was \
                 accomplished, what remains unfinished, and any blockers.",
            )
            .with_name("harness"),
        );

        let (response, estimated) = call_llm(ctx, messages, vec![], self.stream).await?;
        // Tool calls (if any slipped through) are recorded but never executed.
        publish_assistant_message(
            ctx,
            &response,
            estimated,
            Some(("finalized_due_to", json!("max_steps"))),
        )
        .await?;

        if let Some(text) = &response.content {
            info!("assistant (wrap-up): {}", text);
        }
        Ok(())
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
/// use eventage::AgentBuilder;
/// use eventage::SingleShotStrategy;
/// use eventage::llm::MockLlmProvider;
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

        let estimated = super::tokens::messages_token_count(&messages);
        let response = ctx.llm.complete(messages, tool_defs).await?;

        publish_assistant_message(ctx, &response, estimated, None).await?;

        if let Some(text) = &response.content {
            info!("assistant: {}", text);
        }

        Ok(())
    }
}
