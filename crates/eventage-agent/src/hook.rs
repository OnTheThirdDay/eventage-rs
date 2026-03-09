//! Lifecycle hooks for intercepting an agent's reasoning cycle.
//!
//! Extend agent behavior (e.g., approvals, rate limits) by implementing
//! [`CycleHook`] and registering via [`crate::AgentBuilder::hook`].

use async_trait::async_trait;
use eventage_core::EventBus;
use eventage_llm::types::ChatMessage;
use serde_json::Value;
use std::sync::Arc;

// ── HookContext ───────────────────────────────────────────────────────────────

/// Context passed to every [`CycleHook`] method during an agent cycle.
pub struct HookContext<'a> {
    /// The agent's stable identifier.
    pub agent_id: &'a str,
    /// Per-cycle UUID appearing in all events for this cycle.
    pub trace_id: &'a str,
    /// 1-based step index (incremented per tool execution round).
    pub step: usize,
    /// Shared event bus for inspecting state or publishing control events.
    pub bus: &'a EventBus,
}

// ── HookAction ────────────────────────────────────────────────────────────────

/// Returned by [`CycleHook`] methods to influence the agent's execution flow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookAction {
    /// Proceed normally.
    Continue,

    /// Skips the current operation without error.
    ///
    /// Depending on the hook point, this cleanly ends the cycle or injects
    /// a synthetic skipped result for a vetoed tool call.
    Skip,

    /// Aborts the current cycle immediately.
    AbortCycle,
}

// ── CycleHook trait ───────────────────────────────────────────────────────────

/// Intercepts key moments in an agent's reasoning cycle.
///
/// Use default no-op implementations to only override needed methods.
///
/// # Example — human approval gate
///
/// ```rust,no_run
/// use eventage_agent::hook::{CycleHook, HookAction, HookContext};
/// use async_trait::async_trait;
/// use serde_json::Value;
///
/// pub struct HumanApprovalGate;
///
/// #[async_trait]
/// impl CycleHook for HumanApprovalGate {
///     async fn before_tool(
///         &self,
///         _ctx: &HookContext<'_>,
///         name: &str,
///         _args: &Value,
///     ) -> HookAction {
///         eprint!("Allow tool '{}'? [y/N]: ", name);
///         let mut line = String::new();
///         std::io::stdin().read_line(&mut line).ok();
///         if line.trim().eq_ignore_ascii_case("y") {
///             HookAction::Continue
///         } else {
///             HookAction::Skip
///         }
///     }
/// }
/// ```
///
/// # Example — pause every N steps and wait for a resume event
///
/// ```rust,no_run
/// use eventage_agent::hook::{CycleHook, HookAction, HookContext};
/// use eventage_core::Event;
/// use async_trait::async_trait;
///
/// pub struct PauseEveryNSteps(pub usize);
///
/// #[async_trait]
/// impl CycleHook for PauseEveryNSteps {
///     async fn before_step(&self, ctx: &HookContext<'_>) -> HookAction {
///         if ctx.step > 1 && (ctx.step - 1) % self.0 == 0 {
///             eprintln!("Agent paused after {} steps. Publish 'system.resume' to continue.", ctx.step - 1);
///             ctx.bus.wait_for(|e: &Event| e.kind == "system.resume").await;
///         }
///         HookAction::Continue
///     }
/// }
/// ```
#[async_trait]
pub trait CycleHook: Send + Sync {
    /// Called at the start of each react step.
    async fn before_step(&self, _ctx: &HookContext<'_>) -> HookAction {
        HookAction::Continue
    }

    /// Called just before LLM invocation, allowing `messages` mutation.
    async fn before_llm(
        &self,
        _ctx: &HookContext<'_>,
        _messages: &mut Vec<ChatMessage>,
    ) -> HookAction {
        HookAction::Continue
    }

    /// Called before executing an individual tool call.
    async fn before_tool(&self, _ctx: &HookContext<'_>, _name: &str, _args: &Value) -> HookAction {
        HookAction::Continue
    }

    /// Called after a tool executes, fails, or is skipped.
    async fn after_tool(&self, _ctx: &HookContext<'_>, _name: &str, _result: &Value) {}
}

// ── HookChain ─────────────────────────────────────────────────────────────────

/// Runs a sequence of [`CycleHook`]s, short-circuiting on the first
/// non-Continue action.
pub(crate) struct HookChain {
    hooks: Vec<Arc<dyn CycleHook>>,
}

impl HookChain {
    pub fn new(hooks: Vec<Arc<dyn CycleHook>>) -> Self {
        Self { hooks }
    }
}

/// Internal implementation allowing the chain to act as a single hook.
#[async_trait]
impl CycleHook for HookChain {
    async fn before_step(&self, ctx: &HookContext<'_>) -> HookAction {
        for hook in &self.hooks {
            match hook.before_step(ctx).await {
                HookAction::Continue => {}
                action => return action,
            }
        }
        HookAction::Continue
    }

    async fn before_llm(
        &self,
        ctx: &HookContext<'_>,
        messages: &mut Vec<ChatMessage>,
    ) -> HookAction {
        for hook in &self.hooks {
            match hook.before_llm(ctx, messages).await {
                HookAction::Continue => {}
                action => return action,
            }
        }
        HookAction::Continue
    }

    async fn before_tool(&self, ctx: &HookContext<'_>, name: &str, args: &Value) -> HookAction {
        for hook in &self.hooks {
            match hook.before_tool(ctx, name, args).await {
                HookAction::Continue => {}
                action => return action,
            }
        }
        HookAction::Continue
    }

    async fn after_tool(&self, ctx: &HookContext<'_>, name: &str, result: &Value) {
        for hook in &self.hooks {
            hook.after_tool(ctx, name, result).await;
        }
    }
}
