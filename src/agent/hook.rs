//! Lifecycle hooks for intercepting an agent's reasoning cycle.
//!
//! Extend agent behavior (e.g., approvals, rate limits) by implementing
//! [`CycleHook`] and registering via [`crate::AgentBuilder::hook`].

use async_trait::async_trait;
use crate::bus::EventBus;
use crate::llm::types::ChatMessage;
use serde_json::Value;
use std::sync::{Arc, RwLock};

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
/// use eventage::agent::hook::{CycleHook, HookAction, HookContext};
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

// ── DynamicHookChain ──────────────────────────────────────────────────────────

/// A [`CycleHook`] whose list of inner hooks can be mutated at runtime.
///
/// Changes take effect on the next ReAct step. Clones share the same hook list.
///
/// # Example
///
/// ```rust,no_run
/// use eventage::{AgentBuilder, agent::hook::{CycleHook, HookAction, HookContext}};
/// use eventage::agent::DynamicHookChain;
/// use eventage::ReactStrategy;
/// use eventage::llm::MockLlmProvider;
/// use async_trait::async_trait;
///
/// struct LogHook;
///
/// #[async_trait]
/// impl CycleHook for LogHook {
///     async fn before_step(&self, _ctx: &HookContext<'_>) -> HookAction {
///         println!("step started");
///         HookAction::Continue
///     }
/// }
///
/// let dyn_hooks = DynamicHookChain::new();
/// let handle = dyn_hooks.clone();
///
/// let agent = AgentBuilder::new()
///     .llm(MockLlmProvider::with_texts(Vec::<&str>::new()))
///     .hook(dyn_hooks)
///     .strategy(ReactStrategy::default())
///     .build();
///
/// handle.add_hook(LogHook);
/// handle.remove_all();
/// ```
#[derive(Clone, Default)]
pub struct DynamicHookChain {
    hooks: Arc<RwLock<Vec<Arc<dyn CycleHook>>>>,
}

impl DynamicHookChain {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a hook. The agent sees it on the next ReAct step.
    pub fn add_hook(&self, hook: impl CycleHook + 'static) {
        self.hooks.write().unwrap_or_else(|e| e.into_inner()).push(Arc::new(hook));
    }

    /// Appends a pre-boxed hook.
    pub fn add_arc(&self, hook: Arc<dyn CycleHook>) {
        self.hooks.write().unwrap_or_else(|e| e.into_inner()).push(hook);
    }

    /// Removes all hooks.
    pub fn remove_all(&self) {
        self.hooks.write().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// Returns the current number of registered hooks.
    pub fn len(&self) -> usize {
        self.hooks.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    pub fn is_empty(&self) -> bool {
        self.hooks.read().unwrap_or_else(|e| e.into_inner()).is_empty()
    }
}

#[async_trait]
impl CycleHook for DynamicHookChain {
    async fn before_step(&self, ctx: &HookContext<'_>) -> HookAction {
        let hooks: Vec<Arc<dyn CycleHook>> = self.hooks.read().unwrap_or_else(|e| e.into_inner()).clone();
        for hook in &hooks {
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
        let hooks: Vec<Arc<dyn CycleHook>> = self.hooks.read().unwrap_or_else(|e| e.into_inner()).clone();
        for hook in &hooks {
            match hook.before_llm(ctx, messages).await {
                HookAction::Continue => {}
                action => return action,
            }
        }
        HookAction::Continue
    }

    async fn before_tool(&self, ctx: &HookContext<'_>, name: &str, args: &Value) -> HookAction {
        let hooks: Vec<Arc<dyn CycleHook>> = self.hooks.read().unwrap_or_else(|e| e.into_inner()).clone();
        for hook in &hooks {
            match hook.before_tool(ctx, name, args).await {
                HookAction::Continue => {}
                action => return action,
            }
        }
        HookAction::Continue
    }

    async fn after_tool(&self, ctx: &HookContext<'_>, name: &str, result: &Value) {
        let hooks: Vec<Arc<dyn CycleHook>> = self.hooks.read().unwrap_or_else(|e| e.into_inner()).clone();
        for hook in &hooks {
            hook.after_tool(ctx, name, result).await;
        }
    }
}
