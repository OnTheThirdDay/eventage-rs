//! Runtime components: capabilities an agent can gain and lose while it runs.
//!
//! A long-lived agent's capabilities are not fixed at startup. A plugin is
//! installed mid-session, an MCP server dies, a skill set is swapped for a
//! different task. Everything such a capability contributes lands in shared,
//! mutable state — the [`ToolRegistry`], the hook chain, background tasks on
//! the bus — so removing it cleanly is the hard part. Two rules make that
//! safe:
//!
//! - **Nothing outlives its owner.** When a component goes away, every tool
//!   it registered, hook it installed, task it spawned, and service it
//!   published goes with it. A stale tool the model can still call, or a task
//!   still publishing to the bus, is a bug the agent will eventually trip on.
//! - **Dependencies are declared, not assumed.** A component states what it
//!   needs; the host starts it only once those needs are met and stops it if
//!   they later disappear. An agent should never hold a tool whose backing
//!   service is gone.
//!
//! # One list, maintained by construction
//!
//! The rule that makes the first guarantee checkable: **a registration and
//! its undo are produced by the same call**. `ctx.tool(x)` registers the tool
//! *and* records how to remove it. Splitting cleanup into a separate teardown
//! method would mean two lists that must be kept in step by hand, and nothing
//! but discipline stopping them from drifting apart. Here there is one list,
//! and you cannot add to it without also saying how to undo.
//!
//! ```no_run
//! # use eventage::component::{Component, ComponentContext, ComponentError, ComponentHost};
//! # use eventage::{EventBus, ToolRegistry};
//! # use async_trait::async_trait;
//! struct Search;
//!
//! #[async_trait]
//! impl Component for Search {
//!     fn name(&self) -> &str { "search" }
//!     // Cannot run without an index, so the host holds it until one exists.
//!     fn requires(&self) -> Vec<String> { vec!["index".into()] }
//!
//!     async fn start(&self, ctx: &mut ComponentContext) -> Result<(), ComponentError> {
//!         // Registered *and* scheduled for removal in one call.
//!         // ctx.tool(SearchTool { .. });
//!         ctx.on_dispose(|| println!("search withdrawn"));
//!         Ok(())
//!     }
//! }
//! ```
//!
//! # What this does not undo
//!
//! Only what the component registered through its context: tools, hooks,
//! tasks, services, and any undo it recorded itself. Effects that leave the
//! process — files written, rows inserted, messages sent — are not reversible
//! here and should not be presented as if they were. Those are handled
//! honestly by [`ToolRecovery`](crate::agent::recovery::ToolRecovery), with
//! explicit at-most-once semantics.

use crate::agent::hook::{CycleHook, DynamicHookChain, HookId};
use crate::agent::tool::{Tool, ToolRegistry};
use crate::agent::worker::EventWorker;
use crate::bus::EventBus;
use crate::event::Event;
use async_trait::async_trait;
use serde_json::json;
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use thiserror::Error;
use tracing::{debug, info, warn};

/// Event kinds emitted by the component runtime.
pub mod kinds {
    /// A component started. Payload: `{ "component", "provides" }`.
    pub const COMPONENT_LOADED: &str = "component.loaded";
    /// A component stopped. Payload: `{ "component", "reason", "effects_reverted" }`
    /// where `effects_reverted` counts the registrations undone.
    pub const COMPONENT_UNLOADED: &str = "component.unloaded";
    /// A component is waiting on unmet dependencies.
    /// Payload: `{ "component", "missing" }`.
    pub const COMPONENT_PENDING: &str = "component.pending";
}

#[derive(Debug, Error)]
pub enum ComponentError {
    #[error("component '{0}' is already registered")]
    Duplicate(String),
    #[error("unknown component '{0}'")]
    Unknown(String),
    #[error("component failed to start: {0}")]
    Start(String),
}

/// Undoes exactly one registration.
type Disposer = Box<dyn FnOnce() + Send>;

// ── Services ──────────────────────────────────────────────────────────────────

/// Shared services components publish for one another, keyed by name so a
/// dependency can be declared as plain configuration.
#[derive(Clone, Default)]
pub struct ServiceRegistry {
    inner: Arc<Mutex<HashMap<String, Arc<dyn Any + Send + Sync>>>>,
}

impl ServiceRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Publish a service under `name`, replacing any previous value.
    pub fn provide(&self, name: impl Into<String>, value: Arc<dyn Any + Send + Sync>) {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(name.into(), value);
    }

    /// Withdraw a service. Returns `true` if it was present.
    pub fn withdraw(&self, name: &str) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(name)
            .is_some()
    }

    pub fn has(&self, name: &str) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(name)
    }

    /// Fetch a service and downcast it to `T`.
    pub fn get<T: Any + Send + Sync>(&self, name: &str) -> Option<Arc<T>> {
        let value = self
            .inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(name)
            .cloned()?;
        value.downcast::<T>().ok()
    }

    pub fn names(&self) -> Vec<String> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .cloned()
            .collect()
    }
}

// ── Context ───────────────────────────────────────────────────────────────────

/// Handed to a component at start. Every registration made through it is
/// recorded together with the step that undoes it, so the host can take the
/// component apart completely later.
pub struct ComponentContext {
    name: String,
    bus: EventBus,
    tools: ToolRegistry,
    hooks: DynamicHookChain,
    services: ServiceRegistry,
    /// Undo steps, applied in reverse order on unload.
    disposers: Vec<Disposer>,
    /// Services this component published, for dependency bookkeeping.
    provided: Vec<String>,
}

impl ComponentContext {
    fn new(
        name: String,
        bus: EventBus,
        tools: ToolRegistry,
        hooks: DynamicHookChain,
        services: ServiceRegistry,
    ) -> Self {
        Self {
            name,
            bus,
            tools,
            hooks,
            services,
            disposers: Vec::new(),
            provided: Vec::new(),
        }
    }

    /// This component's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The shared event bus.
    pub fn bus(&self) -> &EventBus {
        &self.bus
    }

    /// Register a tool; it is removed from the registry on unload.
    pub fn tool(&mut self, tool: impl Tool + 'static) {
        let name = tool.definition().function.name;
        self.tools.register(Arc::new(tool));
        let registry = self.tools.clone();
        self.disposers.push(Box::new(move || {
            registry.remove(&name);
        }));
    }

    /// Install a lifecycle hook; it is withdrawn on unload.
    pub fn hook(&mut self, hook: impl CycleHook + 'static) {
        let id: HookId = self.hooks.add_hook(hook);
        let hooks = self.hooks.clone();
        self.disposers.push(Box::new(move || {
            hooks.remove(id);
        }));
    }

    /// Spawn a background task; it is aborted on unload.
    pub fn spawn<F>(&mut self, future: F)
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        let handle = tokio::spawn(future);
        self.disposers.push(Box::new(move || handle.abort()));
    }

    /// Run an [`EventWorker`] against the bus for as long as the component
    /// is loaded.
    pub fn worker(&mut self, worker: impl EventWorker + 'static) {
        let bus = self.bus.clone();
        let worker = Arc::new(worker);
        self.spawn(async move {
            let kinds = worker.subscribed_kinds();
            let mut rx = bus.subscribe();
            while let Some(event) = rx.recv().await {
                let interested = kinds.is_empty() || kinds.iter().any(|k| k == &event.kind);
                if interested {
                    if let Err(e) = worker.handle(&event, &bus).await {
                        warn!(error = %e, "component worker error");
                    }
                }
            }
        });
    }

    /// Publish a service other components can depend on. Withdrawn on
    /// unload, which stops anything that declared it in `requires`.
    pub fn provide<T: Any + Send + Sync>(&mut self, name: impl Into<String>, value: Arc<T>) {
        let name = name.into();
        self.services.provide(name.clone(), value);
        self.provided.push(name.clone());
        let services = self.services.clone();
        self.disposers.push(Box::new(move || {
            services.withdraw(&name);
        }));
    }

    /// Look up a service this component declared in `requires`.
    pub fn service<T: Any + Send + Sync>(&self, name: &str) -> Option<Arc<T>> {
        self.services.get::<T>(name)
    }

    /// Record an undo step for something the host cannot see — a connection
    /// to close, a cache to drop, a flag to reset.
    pub fn on_dispose(&mut self, f: impl FnOnce() + Send + 'static) {
        self.disposers.push(Box::new(f));
    }

    /// How many undo steps are pending (used by tests and diagnostics).
    pub fn effect_count(&self) -> usize {
        self.disposers.len()
    }

    /// Run every undo step, most recent first, and report how many ran.
    ///
    /// Reverse order matters: later registrations may rely on earlier ones,
    /// so a component must come apart in the opposite order it was built up.
    fn dispose(self) -> usize {
        let count = self.disposers.len();
        for disposer in self.disposers.into_iter().rev() {
            disposer();
        }
        count
    }
}

// ── Component ─────────────────────────────────────────────────────────────────

/// A capability that can be added to, and removed from, a running agent.
#[async_trait]
pub trait Component: Send + Sync {
    /// Stable identifier, unique within a host.
    fn name(&self) -> &str;

    /// Services that must exist before this component can start. While any
    /// are missing it stays [`Pending`](ComponentState::Pending); if one is
    /// withdrawn later, the component is stopped again.
    fn requires(&self) -> Vec<String> {
        Vec::new()
    }

    /// Register everything this component contributes. Anything acquired
    /// here must go through `ctx`, or the host cannot take it back.
    async fn start(&self, ctx: &mut ComponentContext) -> Result<(), ComponentError>;
}

/// Whether a registered component is currently running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentState {
    /// Running; its tools and hooks are live.
    Active,
    /// Registered but waiting on a service it declared.
    Pending,
}

struct Registered {
    component: Arc<dyn Component>,
    state: ComponentState,
    /// Present only while active; holds the pending undo steps.
    context: Option<ComponentContext>,
}

// ── Host ──────────────────────────────────────────────────────────────────────

/// Loads, unloads, and reconciles components.
///
/// The host tracks which components depend on which services: starting one
/// publishes its services, which may release others that were waiting;
/// stopping one withdraws them, which **cascades** to everything that
/// declared them.
pub struct ComponentHost {
    bus: EventBus,
    tools: ToolRegistry,
    hooks: DynamicHookChain,
    services: ServiceRegistry,
    components: Mutex<HashMap<String, Registered>>,
    /// Registration order, so reconciliation is deterministic.
    order: Mutex<Vec<String>>,
}

impl ComponentHost {
    pub fn new(bus: EventBus, tools: ToolRegistry, hooks: DynamicHookChain) -> Self {
        Self {
            bus,
            tools,
            hooks,
            services: ServiceRegistry::new(),
            components: Mutex::new(HashMap::new()),
            order: Mutex::new(Vec::new()),
        }
    }

    /// The shared service registry.
    pub fn services(&self) -> ServiceRegistry {
        self.services.clone()
    }

    /// Current state of a component, if registered.
    pub fn state(&self, name: &str) -> Option<ComponentState> {
        self.components
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(name)
            .map(|r| r.state)
    }

    /// Names of all active components.
    pub fn active(&self) -> Vec<String> {
        let components = self.components.lock().unwrap_or_else(|e| e.into_inner());
        let mut names: Vec<String> = components
            .iter()
            .filter(|(_, r)| r.state == ComponentState::Active)
            .map(|(name, _)| name.clone())
            .collect();
        names.sort();
        names
    }

    /// Services a component is missing right now.
    fn missing_for(&self, component: &dyn Component) -> Vec<String> {
        component
            .requires()
            .into_iter()
            .filter(|dep| !self.services.has(dep))
            .collect()
    }

    /// Register a component and start it if its dependencies are met.
    pub async fn load(
        &self,
        component: Arc<dyn Component>,
    ) -> Result<ComponentState, ComponentError> {
        let name = component.name().to_string();
        {
            let components = self.components.lock().unwrap_or_else(|e| e.into_inner());
            if components.contains_key(&name) {
                return Err(ComponentError::Duplicate(name));
            }
        }
        {
            let mut components = self.components.lock().unwrap_or_else(|e| e.into_inner());
            components.insert(
                name.clone(),
                Registered {
                    component: Arc::clone(&component),
                    state: ComponentState::Pending,
                    context: None,
                },
            );
            self.order
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(name.clone());
        }

        let state = self.try_start(&name).await?;
        // Starting may have published services that unblock others.
        if state == ComponentState::Active {
            self.reconcile().await;
        }
        Ok(state)
    }

    /// Attempt to start one registered component.
    async fn try_start(&self, name: &str) -> Result<ComponentState, ComponentError> {
        let component = {
            let components = self.components.lock().unwrap_or_else(|e| e.into_inner());
            let entry = components
                .get(name)
                .ok_or_else(|| ComponentError::Unknown(name.to_string()))?;
            if entry.state == ComponentState::Active {
                return Ok(ComponentState::Active);
            }
            Arc::clone(&entry.component)
        };

        let missing = self.missing_for(component.as_ref());
        if !missing.is_empty() {
            debug!(component = name, ?missing, "component pending on services");
            let _ = self
                .bus
                .publish(Event::new(
                    kinds::COMPONENT_PENDING,
                    json!({ "component": name, "missing": missing }),
                ))
                .await;
            return Ok(ComponentState::Pending);
        }

        let mut ctx = ComponentContext::new(
            name.to_string(),
            self.bus.clone(),
            self.tools.clone(),
            self.hooks.clone(),
            self.services.clone(),
        );

        // All-or-nothing: a component that fails halfway must not leave a
        // half-registered tool set behind.
        if let Err(e) = component.start(&mut ctx).await {
            let reverted = ctx.dispose();
            warn!(
                component = name,
                reverted, "start failed; partial effects reverted"
            );
            return Err(e);
        }

        let provides = ctx.provided.clone();
        {
            let mut components = self.components.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(entry) = components.get_mut(name) {
                entry.state = ComponentState::Active;
                entry.context = Some(ctx);
            }
        }
        info!(component = name, ?provides, "component loaded");
        let _ = self
            .bus
            .publish(Event::new(
                kinds::COMPONENT_LOADED,
                json!({ "component": name, "provides": provides }),
            ))
            .await;
        Ok(ComponentState::Active)
    }

    /// Stop a component and undo everything it registered.
    ///
    /// Components left without a service they declared are stopped too,
    /// transitively.
    pub async fn unload(&self, name: &str) -> Result<(), ComponentError> {
        {
            let components = self.components.lock().unwrap_or_else(|e| e.into_inner());
            if !components.contains_key(name) {
                return Err(ComponentError::Unknown(name.to_string()));
            }
        }
        self.stop(name, "unloaded").await;
        self.cascade().await;
        Ok(())
    }

    /// Remove a component entirely (stopping it first).
    pub async fn remove(&self, name: &str) -> Result<(), ComponentError> {
        self.unload(name).await?;
        self.components
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(name);
        self.order
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|n| n != name);
        Ok(())
    }

    /// Stop then start a component — hot replacement.
    pub async fn reload(&self, name: &str) -> Result<ComponentState, ComponentError> {
        self.stop(name, "reloading").await;
        self.cascade().await;
        let state = self.try_start(name).await?;
        self.reconcile().await;
        Ok(state)
    }

    /// Undo one component's registrations. Idempotent.
    async fn stop(&self, name: &str, reason: &str) {
        let context = {
            let mut components = self.components.lock().unwrap_or_else(|e| e.into_inner());
            match components.get_mut(name) {
                Some(entry) if entry.state == ComponentState::Active => {
                    entry.state = ComponentState::Pending;
                    entry.context.take()
                }
                _ => None,
            }
        };
        let Some(context) = context else { return };

        let reverted = context.dispose();
        info!(component = name, reverted, reason, "component unloaded");
        let _ = self
            .bus
            .publish(Event::new(
                kinds::COMPONENT_UNLOADED,
                json!({
                    "component": name,
                    "reason": reason,
                    "effects_reverted": reverted,
                }),
            ))
            .await;
    }

    /// Stop every active component whose declared services are no longer
    /// available, repeating until nothing is left dangling.
    async fn cascade(&self) {
        loop {
            let casualties: Vec<String> = {
                let components = self.components.lock().unwrap_or_else(|e| e.into_inner());
                components
                    .iter()
                    .filter(|(_, r)| r.state == ComponentState::Active)
                    .filter(|(_, r)| !self.missing_for(r.component.as_ref()).is_empty())
                    .map(|(name, _)| name.clone())
                    .collect()
            };
            if casualties.is_empty() {
                return;
            }
            for name in casualties {
                self.stop(&name, "dependency withdrawn").await;
            }
        }
    }

    /// Start any waiting component whose services are now available,
    /// repeating while progress is made.
    async fn reconcile(&self) {
        let mut attempted: HashSet<String> = HashSet::new();
        loop {
            let candidates: Vec<String> = {
                let components = self.components.lock().unwrap_or_else(|e| e.into_inner());
                let order = self.order.lock().unwrap_or_else(|e| e.into_inner());
                order
                    .iter()
                    .filter(|name| {
                        components
                            .get(*name)
                            .is_some_and(|r| r.state == ComponentState::Pending)
                    })
                    .filter(|name| !attempted.contains(*name))
                    .cloned()
                    .collect()
            };
            if candidates.is_empty() {
                return;
            }

            let mut progressed = false;
            for name in candidates {
                attempted.insert(name.clone());
                if let Ok(ComponentState::Active) = self.try_start(&name).await {
                    progressed = true;
                    // A newly published service may unblock earlier entries.
                    attempted.clear();
                    break;
                }
            }
            if !progressed {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::hook::{HookAction, HookContext};
    use crate::llm::ToolDefinition;
    use serde_json::Value;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn host() -> ComponentHost {
        ComponentHost::new(
            EventBus::new(),
            ToolRegistry::new(),
            DynamicHookChain::new(),
        )
    }

    struct NoopTool(String);

    #[async_trait]
    impl Tool for NoopTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(&self.0, "noop", json!({ "type": "object" }))
        }
        async fn execute(&self, _args: Value) -> Result<Value, crate::agent::AgentError> {
            Ok(json!({}))
        }
    }

    struct NoopHook;

    #[async_trait]
    impl CycleHook for NoopHook {
        async fn before_step(&self, _ctx: &HookContext<'_>) -> HookAction {
            HookAction::Continue
        }
    }

    /// Registers one of each kind, so a test can assert nothing survives.
    struct Kitchen {
        ticks: Arc<AtomicUsize>,
        disposed: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Component for Kitchen {
        fn name(&self) -> &str {
            "kitchen"
        }
        async fn start(&self, ctx: &mut ComponentContext) -> Result<(), ComponentError> {
            ctx.tool(NoopTool("kitchen_tool".into()));
            ctx.hook(NoopHook);
            let ticks = Arc::clone(&self.ticks);
            ctx.spawn(async move {
                loop {
                    ticks.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                }
            });
            let disposed = Arc::clone(&self.disposed);
            ctx.on_dispose(move || {
                disposed.fetch_add(1, Ordering::SeqCst);
            });
            Ok(())
        }
    }

    #[tokio::test]
    async fn unload_removes_everything_the_component_registered() {
        let host = host();
        let ticks = Arc::new(AtomicUsize::new(0));
        let disposed = Arc::new(AtomicUsize::new(0));

        host.load(Arc::new(Kitchen {
            ticks: Arc::clone(&ticks),
            disposed: Arc::clone(&disposed),
        }))
        .await
        .unwrap();

        assert!(host.tools.get("kitchen_tool").is_some());
        assert_eq!(host.hooks.len(), 1);
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        assert!(ticks.load(Ordering::SeqCst) > 0, "task should be running");

        host.unload("kitchen").await.unwrap();

        // Registrations gone.
        assert!(
            host.tools.get("kitchen_tool").is_none(),
            "tool must be removed"
        );
        assert_eq!(host.hooks.len(), 0, "hook must be withdrawn");
        assert_eq!(
            disposed.load(Ordering::SeqCst),
            1,
            "custom undo step must run"
        );

        // Task really stopped: the counter must stop advancing.
        let before = ticks.load(Ordering::SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(40)).await;
        assert_eq!(
            ticks.load(Ordering::SeqCst),
            before,
            "spawned task must be aborted"
        );
    }

    /// Records the order in which undo steps run.
    struct Ordered(Arc<Mutex<Vec<u8>>>);

    #[async_trait]
    impl Component for Ordered {
        fn name(&self) -> &str {
            "ordered"
        }
        async fn start(&self, ctx: &mut ComponentContext) -> Result<(), ComponentError> {
            for i in 1u8..=3 {
                let log = Arc::clone(&self.0);
                ctx.on_dispose(move || log.lock().unwrap().push(i));
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn registrations_are_undone_in_reverse_order() {
        let host = host();
        let log = Arc::new(Mutex::new(Vec::new()));
        host.load(Arc::new(Ordered(Arc::clone(&log))))
            .await
            .unwrap();
        host.unload("ordered").await.unwrap();
        assert_eq!(*log.lock().unwrap(), vec![3, 2, 1]);
    }

    struct Failing;

    #[async_trait]
    impl Component for Failing {
        fn name(&self) -> &str {
            "failing"
        }
        async fn start(&self, ctx: &mut ComponentContext) -> Result<(), ComponentError> {
            ctx.tool(NoopTool("doomed_tool".into()));
            Err(ComponentError::Start("boom".into()))
        }
    }

    #[tokio::test]
    async fn a_failed_start_leaves_nothing_behind() {
        let host = host();
        let err = host.load(Arc::new(Failing)).await.unwrap_err();
        assert!(matches!(err, ComponentError::Start(_)));
        assert!(
            host.tools.get("doomed_tool").is_none(),
            "a failed start must leave no registrations behind"
        );
    }

    struct Provider;

    #[async_trait]
    impl Component for Provider {
        fn name(&self) -> &str {
            "provider"
        }
        async fn start(&self, ctx: &mut ComponentContext) -> Result<(), ComponentError> {
            ctx.provide("index", Arc::new(42usize));
            Ok(())
        }
    }

    struct Consumer;

    #[async_trait]
    impl Component for Consumer {
        fn name(&self) -> &str {
            "consumer"
        }
        fn requires(&self) -> Vec<String> {
            vec!["index".into()]
        }
        async fn start(&self, ctx: &mut ComponentContext) -> Result<(), ComponentError> {
            // The dependency must really be resolvable, not just present.
            let index: Arc<usize> = ctx
                .service("index")
                .ok_or_else(|| ComponentError::Start("index missing".into()))?;
            assert_eq!(*index, 42);
            ctx.tool(NoopTool("consumer_tool".into()));
            Ok(())
        }
    }

    #[tokio::test]
    async fn a_component_waits_for_its_dependency_then_starts() {
        let host = host();
        // Consumer arrives first and must park.
        assert_eq!(
            host.load(Arc::new(Consumer)).await.unwrap(),
            ComponentState::Pending
        );
        assert!(host.tools.get("consumer_tool").is_none());

        // Providing the service starts it reactively.
        host.load(Arc::new(Provider)).await.unwrap();
        assert_eq!(host.state("consumer"), Some(ComponentState::Active));
        assert!(host.tools.get("consumer_tool").is_some());
    }

    #[tokio::test]
    async fn withdrawing_a_service_cascades_to_dependents() {
        let host = host();
        host.load(Arc::new(Provider)).await.unwrap();
        host.load(Arc::new(Consumer)).await.unwrap();
        assert_eq!(host.active(), vec!["consumer", "provider"]);

        // Unloading the provider must stop the consumer too, and revert its
        // effects — not leave a tool pointing at a dead dependency.
        host.unload("provider").await.unwrap();
        assert_eq!(host.state("consumer"), Some(ComponentState::Pending));
        assert!(
            host.tools.get("consumer_tool").is_none(),
            "a dependent's tools must go when its service goes"
        );
        assert!(host.active().is_empty());
    }

    #[tokio::test]
    async fn reload_restores_a_working_component() {
        let host = host();
        host.load(Arc::new(Provider)).await.unwrap();
        host.load(Arc::new(Consumer)).await.unwrap();

        host.reload("consumer").await.unwrap();
        assert_eq!(host.state("consumer"), Some(ComponentState::Active));
        assert!(host.tools.get("consumer_tool").is_some());
        // Exactly one registration survives a reload — no duplicates.
        assert_eq!(
            host.tools
                .names()
                .iter()
                .filter(|n| *n == "consumer_tool")
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn unload_is_idempotent_and_reports_unknown_components() {
        let host = host();
        host.load(Arc::new(Provider)).await.unwrap();
        host.unload("provider").await.unwrap();
        // Second unload is a no-op, not a panic or double-dispose.
        host.unload("provider").await.unwrap();
        assert!(matches!(
            host.unload("nope").await,
            Err(ComponentError::Unknown(_))
        ));
    }

    #[tokio::test]
    async fn duplicate_registration_is_rejected() {
        let host = host();
        host.load(Arc::new(Provider)).await.unwrap();
        assert!(matches!(
            host.load(Arc::new(Provider)).await,
            Err(ComponentError::Duplicate(_))
        ));
    }

    #[tokio::test]
    async fn lifecycle_is_observable_on_the_bus() {
        let host = host();
        let mut rx = host.bus.subscribe();
        host.load(Arc::new(Provider)).await.unwrap();
        host.unload("provider").await.unwrap();

        let mut seen = Vec::new();
        while let Ok(Some(event)) =
            tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv()).await
        {
            seen.push(event.kind);
        }
        assert!(seen.iter().any(|k| k == kinds::COMPONENT_LOADED));
        assert!(seen.iter().any(|k| k == kinds::COMPONENT_UNLOADED));
    }
}
