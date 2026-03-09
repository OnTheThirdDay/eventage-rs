//! Dynamic hook management.
//!
//! Provides [`DynamicHookChain`] for runtime-mutable hooks.

use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use eventage_agent::{CycleHook, HookAction, HookContext};
use eventage_llm::ChatMessage;
use serde_json::Value;

/// A [`CycleHook`] whose list of inner hooks can be mutated at runtime.
///
/// Changes take effect on the next ReAct step. Clones share the same hook list.
///
/// # Example
///
/// ```rust,no_run
/// use eventage_agent::{AgentBuilder, hook::{CycleHook, HookAction, HookContext}};
/// use eventage_provided_impl::{DynamicHookChain, ReactStrategy};
/// use eventage_llm::MockLlmProvider;
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
        self.hooks.write().unwrap().push(Arc::new(hook));
    }

    /// Appends a pre-boxed hook.
    pub fn add_arc(&self, hook: Arc<dyn CycleHook>) {
        self.hooks.write().unwrap().push(hook);
    }

    /// Removes all hooks.
    pub fn remove_all(&self) {
        self.hooks.write().unwrap().clear();
    }

    /// Returns the current number of registered hooks.
    pub fn len(&self) -> usize {
        self.hooks.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.hooks.read().unwrap().is_empty()
    }
}

#[async_trait]
impl CycleHook for DynamicHookChain {
    async fn before_step(&self, ctx: &HookContext<'_>) -> HookAction {
        let hooks: Vec<Arc<dyn CycleHook>> = self.hooks.read().unwrap().clone();
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
        let hooks: Vec<Arc<dyn CycleHook>> = self.hooks.read().unwrap().clone();
        for hook in &hooks {
            match hook.before_llm(ctx, messages).await {
                HookAction::Continue => {}
                action => return action,
            }
        }
        HookAction::Continue
    }

    async fn before_tool(&self, ctx: &HookContext<'_>, name: &str, args: &Value) -> HookAction {
        let hooks: Vec<Arc<dyn CycleHook>> = self.hooks.read().unwrap().clone();
        for hook in &hooks {
            match hook.before_tool(ctx, name, args).await {
                HookAction::Continue => {}
                action => return action,
            }
        }
        HookAction::Continue
    }

    async fn after_tool(&self, ctx: &HookContext<'_>, name: &str, result: &Value) {
        let hooks: Vec<Arc<dyn CycleHook>> = self.hooks.read().unwrap().clone();
        for hook in &hooks {
            hook.after_tool(ctx, name, result).await;
        }
    }
}
