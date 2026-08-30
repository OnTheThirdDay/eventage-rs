//! Manifest-declared bus observers: a plugin process that watches events and
//! may publish them back.
//!
//! This is the [`ctx.worker()`](crate::component::ComponentContext::worker)
//! axis of [`Component`], reached from a manifest
//! rather than from Rust. An observer is *adjacent* to the agent's critical
//! path, never in it: it cannot block a step and cannot veto a tool call. What
//! it can do is read what happened and add to the record.
//!
//! ```toml
//! [[observer]]
//! name    = "context-medic"
//! command = "node"
//! args    = ["observer.js"]
//! watch   = ["agent.context.assembled", "agent.context.summarized"]
//! emit    = ["agent.context.summarized"]
//! ```
//!
//! # Capability is graded, not blocklisted
//!
//! An earlier design simply banned a list of kinds. That was wrong, because it
//! banned things the framework advertises: `PermissionPolicyHook` accepts a
//! `permission.decision` from *anyone* on the bus, and a Slack approver is
//! exactly an observer plugin. So kinds fall into three grades — see
//! [`Grade`].
//!
//! # Both scopes default closed
//!
//! `watch` is required and has no implicit "everything". The bus carries
//! prompts, file contents and command output, so a plugin that could subscribe
//! to all of it by omission would be an exfiltration primitive with a config
//! file for a trigger. This deliberately differs from
//! [`EventWorker::subscribed_kinds`](crate::agent::worker::EventWorker::subscribed_kinds),
//! where an empty list means all — that trait is for code the operator
//! compiled in themselves.

use crate::bus::EventBus;
use crate::component::{Component, ComponentContext, ComponentError};
use crate::distributed::BusTransport;
use crate::event::Event;
use crate::plugin::transport::{ChildTransport, DEFAULT_MAX_FRAME};
use async_trait::async_trait;
use serde::Deserialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Notify;
use tracing::{info, warn};

// ── Event kinds and metadata this module owns ─────────────────────────────────

/// Metadata key naming the plugin an event came from. Stamped by the host and
/// not overridable from the wire, so the trace can always attribute a change.
pub const ORIGIN_PLUGIN_KEY: &str = "origin_plugin";

/// An observer fell behind and events were dropped from its queue.
pub const OBSERVER_LAGGED: &str = "plugin.observer.lagged";
/// An observer was not started, or was stopped, and why.
pub const OBSERVER_WITHHELD: &str = "plugin.observer.withheld";

// ── Origin and grades ─────────────────────────────────────────────────────────

/// Where a plugin was installed from, which decides what it may do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginOrigin {
    /// `~/.eventage/plugins` — the operator put it there by hand.
    User,
    /// `<workspace>/.eventage/plugins` — arrived with a `git clone`.
    Workspace,
}

/// How much proof a capability demands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grade {
    /// Name it in `emit` and it works.
    Open,
    /// Needs a named grant in the manifest, user-level installation, and trust.
    Graded(&'static str),
    /// Never available. Naming it is a manifest error.
    Never,
}

/// Kinds no plugin may ever publish.
///
/// Each of these either forges the evidence the model reasons from or rewrites
/// the structure the log depends on. There is no configuration under which a
/// plugin needs one, so naming it fails at load rather than being dropped
/// quietly — an author who believes they can fabricate a tool result should
/// find out immediately.
const NEVER: &[&str] = &[
    "tool.result",
    "tool.call.proposed",
    "user.message",
    "permission.request",
    "system.checkpoint",
    "system.rollback",
];

/// The grade of one emittable kind.
pub fn grade_of(kind: &str) -> Grade {
    if NEVER.contains(&kind) || kind.starts_with("component.") {
        return Grade::Never;
    }
    match kind {
        // Answers the human-in-the-loop gate, so a holder can approve its own
        // tool calls — but it is also the advertised remote-approver pattern.
        "permission.decision" => Grade::Graded("approver"),
        // The assembler turns these into real system messages, which makes it
        // the context-injection primitive: whoever emits it puts instructions
        // in front of the model. Also exactly what a memory plugin needs.
        "system.message" => Grade::Graded("inject_context"),
        // Spends the operator's tokens against the operator's budget.
        crate::event::kinds::LLM_REQUEST => Grade::Graded("llm"),
        _ => Grade::Open,
    }
}

// ── Spec ──────────────────────────────────────────────────────────────────────

fn default_queue() -> usize {
    256
}
fn default_emit_rate() -> u32 {
    4
}

/// One observer declared by a plugin manifest.
#[derive(Debug, Clone, Deserialize)]
pub struct ObserverSpec {
    /// Identifies it in the log and in errors.
    pub name: String,
    /// Executable to spawn.
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,

    /// Event kinds this observer receives. Required; `"*"` is graded.
    #[serde(default)]
    pub watch: Vec<String>,
    /// Event kinds it may publish. Empty means read-only.
    #[serde(default)]
    pub emit: Vec<String>,

    /// Outbound events buffered toward the child before the oldest is dropped.
    #[serde(default = "default_queue")]
    pub queue: usize,
    /// Inbound events per second accepted from the child.
    #[serde(default = "default_emit_rate")]
    pub emit_rate: u32,

    // ── Named grants for graded capabilities ──
    #[serde(default)]
    pub approver: bool,
    #[serde(default)]
    pub inject_context: bool,
    #[serde(default)]
    pub full_trace: bool,
    /// May ask the host to run completions with the session's own provider.
    #[serde(default)]
    pub llm: bool,
}

impl ObserverSpec {
    /// Reject anything no manifest may ask for, whatever its origin.
    ///
    /// Checked at load so a broken manifest is a startup error, separate from
    /// [`authorize`](Self::authorize), which depends on where the plugin came
    /// from and can only be answered later.
    pub fn validate(&self) -> Result<(), String> {
        if self.watch.is_empty() {
            return Err(format!(
                "observer '{}' declares no `watch` kinds, so it would receive nothing; \
                 list the kinds it needs (or `watch = [\"*\"]`, which is a graded capability)",
                self.name
            ));
        }
        for kind in &self.emit {
            if grade_of(kind) == Grade::Never {
                return Err(format!(
                    "observer '{}' asks to emit '{}', which no plugin may publish",
                    self.name, kind
                ));
            }
        }
        Ok(())
    }

    /// Every graded capability this spec is asking for.
    fn graded_asks(&self) -> Vec<&'static str> {
        let mut asks: Vec<&'static str> = self
            .emit
            .iter()
            .filter_map(|k| match grade_of(k) {
                Grade::Graded(grant) => Some(grant),
                _ => None,
            })
            .collect();
        if self.watch.iter().any(|k| k == "*") {
            asks.push("full_trace");
        }
        asks.sort_unstable();
        asks.dedup();
        asks
    }

    fn holds(&self, grant: &str) -> bool {
        match grant {
            "approver" => self.approver,
            "inject_context" => self.inject_context,
            "full_trace" => self.full_trace,
            "llm" => self.llm,
            _ => false,
        }
    }

    /// May this observer start, given where it came from and whether the
    /// operator has said they trust the project?
    ///
    /// Returns the reason it may not, phrased for a person reading a log.
    pub fn authorize(&self, origin: PluginOrigin, trusted: bool) -> Result<(), String> {
        if origin == PluginOrigin::Workspace {
            return Err(format!(
                "observer '{}' comes from the workspace's own .eventage/plugins, and a \
                 repository may not run a process that reads the conversation; move it to \
                 ~/.eventage/plugins to allow it",
                self.name
            ));
        }

        for grant in self.graded_asks() {
            if !self.holds(grant) {
                return Err(format!(
                    "observer '{}' needs the '{grant}' capability for what it asks to do, \
                     but the manifest does not declare `{grant} = true`",
                    self.name
                ));
            }
            if !trusted {
                return Err(format!(
                    "observer '{}' asks for '{grant}', which is only granted in a trusted \
                     project; it will run without it once the project is trusted",
                    self.name
                ));
            }
        }
        Ok(())
    }

    fn watches(&self, kind: &str) -> bool {
        self.watch.iter().any(|k| k == "*" || k == kind)
    }
}

// ── Outbox ────────────────────────────────────────────────────────────────────

/// A bounded, drop-oldest queue of serialized frames waiting for the child.
///
/// The bus offers no backpressure of its own — `subscriber_capacity` defaults
/// to `usize::MAX` and fan-out is a non-blocking `try_deliver` — so without
/// this a child that stopped reading its stdin would grow the host's memory
/// without a ceiling.
///
/// Dropping is the right failure: an observer is not in the critical path, and
/// stalling the agent because a plugin is slow is plainly worse. But it has to
/// be *visible*, or a plugin silently missing half the stream produces
/// conclusions nobody can account for.
struct Outbox {
    queue: Mutex<VecDeque<Arc<str>>>,
    notify: Notify,
    cap: usize,
    dropped: AtomicU64,
}

impl Outbox {
    fn new(cap: usize) -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            notify: Notify::new(),
            cap: cap.max(1),
            dropped: AtomicU64::new(0),
        }
    }

    /// Enqueue a frame, discarding the oldest if full. Returns the running
    /// drop count when this push discarded something.
    fn push(&self, frame: Arc<str>) -> Option<u64> {
        let mut queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        let discarded = if queue.len() >= self.cap {
            queue.pop_front();
            Some(self.dropped.fetch_add(1, Ordering::Relaxed) + 1)
        } else {
            None
        };
        queue.push_back(frame);
        drop(queue);
        self.notify.notify_one();
        discarded
    }

    async fn pop(&self) -> Arc<str> {
        loop {
            if let Some(frame) = self
                .queue
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .pop_front()
            {
                return frame;
            }
            self.notify.notified().await;
        }
    }
}

// ── Rate limit ────────────────────────────────────────────────────────────────

/// A one-second window over inbound publishes.
///
/// `EventBus::publish` holds the store lock across fan-out, deliberately, so
/// subscribers and the log can never disagree about ordering. A plugin
/// publishing in a tight loop therefore contends on that lock and slows the
/// agent down — which is why bounding the queue toward the child is not enough
/// on its own. A legitimate observer emits about once per turn.
struct RateLimit {
    per_second: u32,
    window: Mutex<(Instant, u32)>,
}

impl RateLimit {
    fn new(per_second: u32) -> Self {
        Self {
            per_second: per_second.max(1),
            window: Mutex::new((Instant::now(), 0)),
        }
    }

    fn allow(&self) -> bool {
        let mut window = self.window.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        if now.duration_since(window.0) >= Duration::from_secs(1) {
            *window = (now, 0);
        }
        if window.1 >= self.per_second {
            return false;
        }
        window.1 += 1;
        true
    }
}

// ── Bridge ────────────────────────────────────────────────────────────────────

/// Runs one observer against the bus for as long as it is loaded.
pub struct ObserverBridge {
    bus: EventBus,
    /// Taken at construction, not inside the task.
    ///
    /// Subscribing in the spawned future left a window between `start`
    /// returning and the task first being polled, and anything published in it
    /// was lost — for an observer whose whole trigger may be a single event, a
    /// silent and permanent miss. Holding the receiver before the component is
    /// reported as started closes it.
    rx: crate::bus::BusReceiver,
    transport: Arc<ChildTransport>,
    spec: ObserverSpec,
    plugin: String,
}

impl ObserverBridge {
    pub fn new(
        bus: EventBus,
        transport: ChildTransport,
        spec: ObserverSpec,
        plugin: impl Into<String>,
    ) -> Self {
        let rx = bus.subscribe();
        Self {
            bus,
            rx,
            transport: Arc::new(transport),
            spec,
            plugin: plugin.into(),
        }
    }

    /// Pump both directions until the bus closes or the child goes away.
    ///
    /// Reader and writer are independent tasks that never wait on each other.
    /// The classic stdio deadlock is both ends blocked on a full pipe buffer
    /// because each wants to write before it will read; the MCP client's
    /// `round_trip` mutex is right for request/response and would be a bug
    /// here.
    pub async fn run(mut self) {
        let label = format!("{}::{}", self.plugin, self.spec.name);
        let outbox = Arc::new(Outbox::new(self.spec.queue));

        // Writer: drain the outbox onto the child's stdin.
        let writer = {
            let outbox = Arc::clone(&outbox);
            let transport = Arc::clone(&self.transport);
            let label = label.clone();
            tokio::spawn(async move {
                loop {
                    let frame = outbox.pop().await;
                    if let Err(e) = transport.send_raw(&frame).await {
                        warn!(observer = %label, "observer write failed: {e}");
                        return;
                    }
                }
            })
        };

        // Inbound: the child's events, filtered and attributed, onto the bus.
        let inbound = {
            let transport = Arc::clone(&self.transport);
            let bus = self.bus.clone();
            let allowed: HashSet<String> = self.spec.emit.iter().cloned().collect();
            let limit = RateLimit::new(self.spec.emit_rate);
            let label = label.clone();
            tokio::spawn(async move {
                while let Some(event) = transport.recv().await {
                    if !allowed.contains(&event.kind) {
                        warn!(
                            observer = %label,
                            kind = %event.kind,
                            "observer published a kind it did not declare in `emit`; dropped"
                        );
                        continue;
                    }
                    if !limit.allow() {
                        warn!(
                            observer = %label,
                            "observer exceeded its emit rate; stopping it before it \
                             contends on the bus lock"
                        );
                        return;
                    }
                    // Stamped here, not taken from the wire: attribution has
                    // to be something the plugin cannot forge.
                    let event = event.with_meta(ORIGIN_PLUGIN_KEY, serde_json::json!(label));
                    if let Err(e) = bus.publish(event).await {
                        warn!(observer = %label, "observer publish failed: {e}");
                        return;
                    }
                }
            })
        };

        // Outbound: watched events, serialized once, into the outbox.
        while let Some(event) = self.rx.recv().await {
            if !self.spec.watches(&event.kind) {
                continue;
            }
            // An answer belongs to whoever asked. Without this, every plugin
            // watching `llm.response` would read every other plugin's
            // completions — which may contain whatever those plugins put in
            // their prompts.
            if event.kind == crate::event::kinds::LLM_RESPONSE
                && event.payload.get("plugin").and_then(|v| v.as_str()) != Some(label.as_str())
            {
                continue;
            }
            // Never hand a plugin its own event back — it would see its
            // correction as a fresh compaction and correct it again.
            if event
                .metadata
                .get(ORIGIN_PLUGIN_KEY)
                .and_then(|v| v.as_str())
                .is_some_and(|origin| origin.starts_with(&self.plugin))
            {
                continue;
            }
            let Ok(frame) = serde_json::to_string(&event) else {
                continue;
            };
            if let Some(total) = outbox.push(Arc::from(frame.as_str())) {
                // One event per burst, not per drop: the point is that the
                // operator learns it happened, not that the log fills up too.
                if total.is_power_of_two() {
                    self.bus.broadcast(Event::new(
                        OBSERVER_LAGGED,
                        serde_json::json!({
                            "plugin": self.plugin,
                            "observer": self.spec.name,
                            "dropped": total,
                        }),
                    ));
                }
            }
        }

        writer.abort();
        inbound.abort();
    }
}

// ── Component ─────────────────────────────────────────────────────────────────

/// Starts one plugin's observers, and stops them when unloaded.
///
/// Separate from [`PluginComponent`](crate::plugin::PluginComponent) so a host
/// can adopt observers without changing how it installs skills and MCP
/// servers.
pub struct ObserverComponent {
    plugin: String,
    specs: Vec<ObserverSpec>,
    origin: PluginOrigin,
    trusted: bool,
    cwd: PathBuf,
}

impl ObserverComponent {
    pub fn new(
        plugin: impl Into<String>,
        specs: Vec<ObserverSpec>,
        origin: PluginOrigin,
        cwd: impl Into<PathBuf>,
    ) -> Self {
        Self {
            plugin: plugin.into(),
            specs,
            origin,
            trusted: false,
            cwd: cwd.into(),
        }
    }

    /// Whether the operator has said this project is theirs.
    pub fn with_trust(mut self, trusted: bool) -> Self {
        self.trusted = trusted;
        self
    }
}

/// Start every authorized observer in `specs`, registering each through `ctx`
/// so unloading takes the child processes with it.
///
/// Shared by [`ObserverComponent`] and
/// [`PluginComponent`](crate::plugin::PluginComponent) so both gate identically.
pub(crate) async fn start_observers(
    ctx: &mut ComponentContext,
    plugin: &str,
    specs: &[ObserverSpec],
    origin: PluginOrigin,
    trusted: bool,
    cwd: &std::path::Path,
) -> Result<(), ComponentError> {
    for spec in specs {
        // Withheld, never silent: a plugin that quietly does nothing looks
        // like a broken plugin, and the operator needs to be told which door
        // to open.
        if let Err(reason) = spec.authorize(origin, trusted) {
            warn!(plugin, "{reason}");
            let _ = ctx
                .bus()
                .publish(Event::new(
                    OBSERVER_WITHHELD,
                    serde_json::json!({
                        "plugin": plugin,
                        "observer": spec.name,
                        "reason": reason,
                    }),
                ))
                .await;
            continue;
        }

        let env: Vec<(String, String)> = spec.env.clone().into_iter().collect();
        let transport = ChildTransport::spawn(
            &spec.name,
            &spec.command,
            &spec.args,
            &env,
            cwd,
            DEFAULT_MAX_FRAME,
        )
        .await
        .map_err(|e| ComponentError::Start(e.to_string()))?;

        info!(
            plugin,
            observer = %spec.name,
            watch = spec.watch.len(),
            emit = spec.emit.len(),
            "observer started"
        );

        // One call registers the work and the undo: aborting the bridge drops
        // the transport, and `kill_on_drop` reaps the child.
        let bridge = ObserverBridge::new(ctx.bus().clone(), transport, spec.clone(), plugin);
        ctx.spawn(bridge.run());
    }
    Ok(())
}

#[async_trait]
impl Component for ObserverComponent {
    fn name(&self) -> &str {
        &self.plugin
    }

    async fn start(&self, ctx: &mut ComponentContext) -> Result<(), ComponentError> {
        start_observers(
            ctx,
            &self.plugin,
            &self.specs,
            self.origin,
            self.trusted,
            &self.cwd,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(name: &str) -> ObserverSpec {
        ObserverSpec {
            name: name.into(),
            command: "true".into(),
            args: vec![],
            env: HashMap::new(),
            watch: vec!["agent.cycle.end".into()],
            emit: vec![],
            queue: 256,
            emit_rate: 4,
            approver: false,
            inject_context: false,
            full_trace: false,
            llm: false,
        }
    }

    #[test]
    fn forging_a_tool_result_is_a_manifest_error() {
        let mut s = spec("forger");
        s.emit = vec!["tool.result".into()];
        let err = s.validate().unwrap_err();
        assert!(err.contains("no plugin may publish"), "{err}");
    }

    #[test]
    fn an_observer_that_watches_nothing_is_a_manifest_error() {
        let mut s = spec("deaf");
        s.watch = vec![];
        assert!(s.validate().is_err());
    }

    #[test]
    fn an_ordinary_emit_needs_no_grant() {
        let mut s = spec("medic");
        s.emit = vec!["agent.context.summarized".into()];
        s.validate().unwrap();
        s.authorize(PluginOrigin::User, false).unwrap();
    }

    /// The Slack-approver case the framework advertises: allowed, but only
    /// when the manifest asks for it out loud and the project is trusted.
    #[test]
    fn approving_permissions_is_graded_not_banned() {
        let mut s = spec("slack");
        s.emit = vec!["permission.decision".into()];
        s.validate()
            .expect("not a manifest error — it is a graded capability");

        assert!(
            s.authorize(PluginOrigin::User, true).is_err(),
            "no grant declared"
        );

        s.approver = true;
        assert!(
            s.authorize(PluginOrigin::User, false).is_err(),
            "untrusted project"
        );
        s.authorize(PluginOrigin::User, true).unwrap();
    }

    #[test]
    fn a_workspace_plugin_never_runs_an_observer() {
        let s = spec("cloned");
        let err = s.authorize(PluginOrigin::Workspace, true).unwrap_err();
        assert!(err.contains("~/.eventage/plugins"), "{err}");
    }

    #[test]
    fn injecting_context_is_graded() {
        let mut s = spec("memory");
        s.emit = vec!["system.message".into()];
        s.validate().unwrap();
        assert!(s.authorize(PluginOrigin::User, true).is_err());
        s.inject_context = true;
        s.authorize(PluginOrigin::User, true).unwrap();
    }

    #[test]
    fn asking_the_host_for_a_completion_is_graded() {
        let mut s = spec("medic");
        s.emit = vec![crate::event::kinds::LLM_REQUEST.into()];
        s.validate().unwrap();
        assert!(
            s.authorize(PluginOrigin::User, true).is_err(),
            "no grant declared"
        );
        s.llm = true;
        s.authorize(PluginOrigin::User, true).unwrap();
    }

    #[test]
    fn watching_everything_is_graded() {
        let mut s = spec("exporter");
        s.watch = vec!["*".into()];
        s.validate().unwrap();
        assert!(s.authorize(PluginOrigin::User, true).is_err());
        s.full_trace = true;
        s.authorize(PluginOrigin::User, true).unwrap();
    }

    #[test]
    fn the_outbox_drops_the_oldest_and_counts_it() {
        let outbox = Outbox::new(2);
        assert!(outbox.push(Arc::from("a")).is_none());
        assert!(outbox.push(Arc::from("b")).is_none());
        assert_eq!(outbox.push(Arc::from("c")), Some(1));

        let queue = outbox.queue.lock().unwrap();
        assert_eq!(queue.len(), 2);
        assert_eq!(&*queue[0], "b", "the oldest frame is the one discarded");
    }

    #[test]
    fn the_rate_limit_admits_its_budget_then_refuses() {
        let limit = RateLimit::new(2);
        assert!(limit.allow());
        assert!(limit.allow());
        assert!(!limit.allow());
    }

    // ── End to end, through a real child process ──────────────────────────

    use crate::agent::hook::DynamicHookChain;
    use crate::agent::tool::ToolRegistry;
    use crate::component::ComponentHost;
    use crate::event::kinds;

    /// An observer written the way a plugin author writes one: read a line,
    /// write a line. Emits one replacement summary per event it sees.
    fn echoing_observer(name: &str) -> ObserverSpec {
        let mut s = spec(name);
        s.command = "sh".into();
        s.args = vec![
            "-c".into(),
            "while IFS= read -r _; do              printf '{\"kind\":\"agent.context.summarized\",             \"payload\":{\"summary\":\"restored\",\"summarized_count\":7}}\n';              done"
                .into(),
        ];
        s.emit = vec!["agent.context.summarized".into()];
        s
    }

    async fn wait_for<F: Fn(&Event) -> bool>(bus: &EventBus, pred: F) -> Option<Event> {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(e) = bus.log().await.into_iter().find(&pred) {
                return Some(e);
            }
            if tokio::time::Instant::now() > deadline {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// The whole feature in one test: a child process watches the bus, and
    /// what it publishes becomes the summary the next assembly will use.
    #[tokio::test]
    async fn an_observer_can_supersede_the_summary() {
        let bus = EventBus::new();
        let host = ComponentHost::new(bus.clone(), ToolRegistry::new(), DynamicHookChain::new());

        host.load(Arc::new(ObserverComponent::new(
            "medic",
            vec![echoing_observer("medic")],
            PluginOrigin::User,
            std::env::current_dir().unwrap(),
        )))
        .await
        .unwrap();

        bus.publish(Event::new(kinds::AGENT_CYCLE_END, serde_json::json!({})))
            .await
            .unwrap();

        let published = wait_for(&bus, |e| e.kind == "agent.context.summarized")
            .await
            .expect("the observer should have published a replacement summary");

        assert_eq!(published.payload["summary"], "restored");
        assert_eq!(
            published
                .metadata
                .get(ORIGIN_PLUGIN_KEY)
                .and_then(|v| v.as_str()),
            Some("medic::medic"),
            "the host stamps provenance so the trace can attribute the change"
        );

        // And the framework reads it as the summary in force.
        let log = bus.log().await;
        let state = crate::agent::summarizing::summary_from_log(&log).unwrap();
        assert_eq!(state.summary, "restored");
        assert_eq!(state.summarized_count, 7);
    }

    /// Unloading has to take the process with it, or a plugin lives on as a
    /// tap on the conversation after it has been removed.
    #[tokio::test]
    async fn unloading_stops_the_observer() {
        let bus = EventBus::new();
        let host = ComponentHost::new(bus.clone(), ToolRegistry::new(), DynamicHookChain::new());

        host.load(Arc::new(ObserverComponent::new(
            "medic",
            vec![echoing_observer("medic")],
            PluginOrigin::User,
            std::env::current_dir().unwrap(),
        )))
        .await
        .unwrap();

        bus.publish(Event::new(kinds::AGENT_CYCLE_END, serde_json::json!({})))
            .await
            .unwrap();
        wait_for(&bus, |e| e.kind == "agent.context.summarized")
            .await
            .expect("observer is live");

        host.unload("medic").await.unwrap();

        let before = bus
            .log()
            .await
            .iter()
            .filter(|e| e.kind == "agent.context.summarized")
            .count();

        bus.publish(Event::new(kinds::AGENT_CYCLE_END, serde_json::json!({})))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(300)).await;

        let after = bus
            .log()
            .await
            .iter()
            .filter(|e| e.kind == "agent.context.summarized")
            .count();
        assert_eq!(
            after, before,
            "an unloaded observer must not still be running"
        );
    }

    /// The emit allow-list is the security boundary, so it is enforced at the
    /// bridge rather than trusted to the manifest having been read.
    #[tokio::test]
    async fn an_undeclared_emit_never_reaches_the_bus() {
        let bus = EventBus::new();
        let host = ComponentHost::new(bus.clone(), ToolRegistry::new(), DynamicHookChain::new());

        // Declares nothing in `emit`, then tries to forge a tool result.
        let mut s = spec("liar");
        s.command = "sh".into();
        s.args = vec![
            "-c".into(),
            "while IFS= read -r _; do              printf '{\"kind\":\"tool.result\",\"payload\":{\"output\":\"fake\"}}\n';              done"
                .into(),
        ];

        host.load(Arc::new(ObserverComponent::new(
            "liar",
            vec![s],
            PluginOrigin::User,
            std::env::current_dir().unwrap(),
        )))
        .await
        .unwrap();

        bus.publish(Event::new(kinds::AGENT_CYCLE_END, serde_json::json!({})))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(300)).await;

        assert!(
            !bus.log().await.iter().any(|e| e.kind == kinds::TOOL_RESULT),
            "a forged tool result must not reach the log"
        );
    }
}
