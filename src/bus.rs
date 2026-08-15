use crate::error::BusError;
use crate::event::kinds;
use crate::event::{meta_keys, Event, EventId};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, instrument};
use uuid::Uuid;

pub type BranchId = Uuid;

// ── Branch Eviction ──────────────────────────────────────────────────────────

/// A rejected event branch passed to [`BranchEvictionStrategy::on_evict`].
///
/// The bus automatically reclaims the branch's memory after `on_evict` returns.
#[derive(Debug, Clone)]
pub struct BranchData {
    /// Unique branch identifier.
    pub id: BranchId,
    /// ID of the last event shared with the active branch. `None` if diverging from the root.
    pub parent_event_id: Option<EventId>,
    /// Chronological sequence of events in this branch.
    pub events: Vec<Event>,
}

/// Callback invoked when a rejected branch is evicted from memory.
///
/// Implementations receive full ownership of the [`BranchData`] to log, persist, or forward.
/// The bus immediately reclaims the branch's memory afterward.
///
/// **Important:** `on_evict` runs synchronously under the bus's write lock.
/// It must execute quickly; offload I/O or heavy computation to background tasks.
///
/// Default strategy: [`PruneStrategy`] (drops the branch).
pub trait BranchEvictionStrategy: Send + Sync {
    /// Synchronously consumes the evicted branch. Must not block.
    fn on_evict(&self, branch: BranchData);

    /// Human-readable strategy name.
    fn name(&self) -> &'static str {
        "custom"
    }
}

/// Default eviction strategy: silently discards evicted branches.
#[derive(Debug, Default, Clone)]
pub struct PruneStrategy;

impl BranchEvictionStrategy for PruneStrategy {
    fn on_evict(&self, _branch: BranchData) {}
    fn name(&self) -> &'static str {
        "prune"
    }
}

// ── Internal DAG store ────────────────────────────────────────────────────────

/// An immutable branch sealed after a rollback.
struct RejectedBranch {
    id: BranchId,
    /// ID of the last event shared with the active branch.
    parent_event_id: Option<EventId>,
    event_ids: Vec<EventId>,
}

struct DagStore {
    /// Global event registry.
    nodes: HashMap<EventId, Event>,
    /// Sequential EventIds forming the current active path.
    active_path: Vec<EventId>,
    rejected_branches: Vec<RejectedBranch>,
}

impl DagStore {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            active_path: Vec::new(),
            rejected_branches: Vec::new(),
        }
    }

    fn active_tip(&self) -> Option<EventId> {
        self.active_path.last().copied()
    }

    /// Evicts branches beyond `max`. Returns `(evicted_branches, evicted_nodes)`.
    fn evict_excess_branches(
        &mut self,
        max: usize,
        strategy: &dyn BranchEvictionStrategy,
    ) -> (usize, usize) {
        let count = self.rejected_branches.len();
        if count <= max {
            return (0, 0);
        }
        let excess = count - max;

        // Drain the oldest `excess` branches from the front, notify the strategy.
        let evicted: Vec<RejectedBranch> = self.rejected_branches.drain(..excess).collect();
        let evicted_count = evicted.len();

        for branch in evicted {
            let branch_data = BranchData {
                id: branch.id,
                parent_event_id: branch.parent_event_id,
                events: branch
                    .event_ids
                    .iter()
                    .filter_map(|id| self.nodes.get(id).cloned())
                    .collect(),
            };
            strategy.on_evict(branch_data);
        }

        // GC: keep nodes referenced by the active path or any retained branch.
        let retained: HashSet<EventId> = self
            .active_path
            .iter()
            .copied()
            .chain(
                self.rejected_branches
                    .iter()
                    .flat_map(|b| b.event_ids.iter().copied()),
            )
            .collect();

        let before = self.nodes.len();
        self.nodes.retain(|id, _| retained.contains(id));
        let evicted_nodes = before.saturating_sub(self.nodes.len());

        (evicted_count, evicted_nodes)
    }
}

// ── BusConfig ─────────────────────────────────────────────────────────────────

/// Configuration for the [`EventBus`] affecting memory limits and queue behavior.
///
/// * **Branches:** `max_retained_branches` limits sealed branches in memory. Excess triggers eviction and a `system.pruned` event.
/// * **Queues:** `subscriber_capacity` bounds subscriber channels. Overflow drops events for that subscriber without blocking publishers.
#[derive(Clone)]
pub struct BusConfig {
    /// Limit on sealed branches retained in memory. Default: `usize::MAX`.
    pub max_retained_branches: usize,

    /// Capacity of per-subscriber channels. Overflow drops events. Default: `usize::MAX`.
    pub subscriber_capacity: usize,

    /// Callback for evicted rejected branches. Default: [`PruneStrategy`].
    pub eviction_strategy: Arc<dyn BranchEvictionStrategy>,
}

impl std::fmt::Debug for BusConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BusConfig")
            .field("max_retained_branches", &self.max_retained_branches)
            .field("subscriber_capacity", &self.subscriber_capacity)
            .field("eviction_strategy", &self.eviction_strategy.name())
            .finish()
    }
}

impl Default for BusConfig {
    fn default() -> Self {
        Self {
            max_retained_branches: usize::MAX,
            subscriber_capacity: usize::MAX,
            eviction_strategy: Arc::new(PruneStrategy),
        }
    }
}

// ── BusReceiver ───────────────────────────────────────────────────────────────

enum ReceiverInner {
    Unbounded(mpsc::UnboundedReceiver<Event>),
    Bounded(mpsc::Receiver<Event>),
}

enum SenderInner {
    Unbounded(mpsc::UnboundedSender<Event>),
    Bounded(mpsc::Sender<Event>),
}

impl SenderInner {
    /// Synchronously attempts delivery. Returns `false` if the receiver has dropped.
    /// Bounded queues drop the event and return `true` on overflow.
    fn try_deliver(&self, event: Event) -> bool {
        match self {
            SenderInner::Unbounded(tx) => tx.send(event).is_ok(),
            SenderInner::Bounded(tx) => match tx.try_send(event) {
                Ok(_) => true,
                Err(mpsc::error::TrySendError::Full(_)) => {
                    tracing::warn!(
                        "subscriber queue full; event dropped. \
                         Use bus.log() / bus.log_since() to reconstruct missed events."
                    );
                    true // keep sender — subscriber is still alive
                }
                Err(mpsc::error::TrySendError::Closed(_)) => false, // prune
            },
        }
    }
}

/// Receiving end of an [`EventBus`] subscription.
///
/// Missed events from a bounded queue can be recovered using [`EventBus::log_since`].
pub struct BusReceiver(ReceiverInner);

impl BusReceiver {
    /// Yields the next available event, or `None` if the bus is dropped.
    pub async fn recv(&mut self) -> Option<Event> {
        match &mut self.0 {
            ReceiverInner::Unbounded(rx) => rx.recv().await,
            ReceiverInner::Bounded(rx) => rx.recv().await,
        }
    }
}

// ── EventBus ─────────────────────────────────────────────────────────────────

/// An asynchronous, DAG-structured broadcast bus.
///
/// Events automatically link to form a Directed Acyclic Graph. The active branch
/// remains in memory, while [`checkpoint`][Self::checkpoint] and [`rollback`][Self::rollback]
/// isolate abandoned paths into rejected branches.
#[derive(Clone)]
pub struct EventBus {
    inner: Arc<BusInner>,
}

struct BusInner {
    store: RwLock<DagStore>,
    /// Active subscriber senders. Dead connections are pruned on publish.
    subs: Mutex<Vec<SenderInner>>,
    config: BusConfig,
    /// Synchronous transforms applied to each event at publish time, in order.
    #[allow(clippy::type_complexity)]
    transforms: Mutex<Vec<Box<dyn Fn(Event) -> Event + Send + Sync>>>,
}

impl EventBus {
    pub fn new() -> Self {
        Self::with_config(BusConfig::default())
    }

    /// Instantiates with specific [`BusConfig`] limits.
    pub fn with_config(config: BusConfig) -> Self {
        Self {
            inner: Arc::new(BusInner {
                store: RwLock::new(DagStore::new()),
                subs: Mutex::new(Vec::new()),
                config,
                transforms: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Register a synchronous transform applied to every event at publish time,
    /// before it is stored in the DAG and fanned out to subscribers.
    ///
    /// Transforms are applied in registration order and are composable. Typical
    /// uses include secrets masking, payload normalization, or tagging.
    ///
    /// ```no_run
    /// use eventage::{EventBus, secrets_masking_transform};
    ///
    /// let bus = EventBus::new();
    /// bus.add_publish_transform(secrets_masking_transform(vec!["sk-secret".to_string()]));
    /// ```
    pub fn add_publish_transform(
        &self,
        f: impl Fn(Event) -> Event + Send + Sync + 'static,
    ) -> &Self {
        self.inner
            .transforms
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(Box::new(f));
        self
    }

    /// Dispatches an event to all subscribers, pruning dropped connections.
    fn fan_out(subs: &mut Vec<SenderInner>, event: Event) {
        // Count how many are alive before retain so we can detect drops.
        let before = subs.len();
        subs.retain(|tx| tx.try_deliver(event.clone()));
        let dropped = before - subs.len();
        if dropped > 0 {
            tracing::debug!(dropped, "pruned dead subscriber(s)");
        }
    }

    // ── Core publish / subscribe ──────────────────────────────────────────────

    /// Appends an event to the active branch and distributes it.
    ///
    /// If the event already has `parent_event_id` set (via [`Event::with_parent`]),
    /// that value is preserved. Otherwise the current active-branch tip is used.
    ///
    /// Registered publish transforms (see [`add_publish_transform`]) are applied
    /// to the event before storage and fan-out.
    #[instrument(skip(self, event), fields(kind = %event.kind, id = %event.id))]
    pub async fn publish(&self, mut event: Event) -> Result<(), BusError> {
        debug!("publishing event");

        {
            let transforms = self
                .inner
                .transforms
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            for transform in transforms.iter() {
                event = transform(event);
            }
        }

        {
            let mut store = self.inner.store.write().await;
            if event.parent_event_id.is_none() {
                event.parent_event_id = store.active_tip();
            }
            let id = event.id;
            store.nodes.insert(id, event.clone());
            store.active_path.push(id);
        }
        let mut subs = self.inner.subs.lock().unwrap_or_else(|e| e.into_inner());
        Self::fan_out(&mut subs, event);
        Ok(())
    }

    /// Broadcasts an **ephemeral** event to subscribers without storing it in
    /// the DAG log.
    ///
    /// Use for high-frequency signals that must never enter the LLM context or
    /// the persisted history: streaming deltas (`assistant.delta`), progress
    /// ticks, UI hints. Durable facts belong in [`publish`](Self::publish).
    pub fn broadcast(&self, event: Event) {
        let mut subs = self.inner.subs.lock().unwrap_or_else(|e| e.into_inner());
        Self::fan_out(&mut subs, Self::mark_ephemeral(event));
    }

    /// Stamp an event as never having been on the active branch.
    ///
    /// Observers cannot otherwise tell a broadcast from a published event —
    /// both arrive on the same channel — and an exporter that persists both
    /// would hand `restore_from` a log it could not reassemble correctly.
    fn mark_ephemeral(event: Event) -> Event {
        event.with_meta(meta_keys::EPHEMERAL, serde_json::json!(true))
    }

    /// Grants a new subscription channel for future events.
    pub fn subscribe(&self) -> BusReceiver {
        let cap = self.inner.config.subscriber_capacity;
        if cap == usize::MAX {
            let (tx, rx) = mpsc::unbounded_channel();
            self.inner
                .subs
                .lock()
                .unwrap()
                .push(SenderInner::Unbounded(tx));
            BusReceiver(ReceiverInner::Unbounded(rx))
        } else {
            let (tx, rx) = mpsc::channel(cap);
            self.inner
                .subs
                .lock()
                .unwrap()
                .push(SenderInner::Bounded(tx));
            BusReceiver(ReceiverInner::Bounded(rx))
        }
    }

    // ── Log access ────────────────────────────────────────────────────────────

    /// Fully chronologically reconstructs the active branch.
    pub async fn log(&self) -> Vec<Event> {
        let store = self.inner.store.read().await;
        store
            .active_path
            .iter()
            .filter_map(|id| store.nodes.get(id).cloned())
            .collect()
    }

    /// Chronologically yields active-branch events occurring at or after `index`.
    pub async fn log_since(&self, index: usize) -> Vec<Event> {
        let store = self.inner.store.read().await;
        let start = index.min(store.active_path.len());
        store.active_path[start..]
            .iter()
            .filter_map(|id| store.nodes.get(id).cloned())
            .collect()
    }

    /// Number of events on the active branch.
    pub async fn log_len(&self) -> usize {
        self.inner.store.read().await.active_path.len()
    }

    // ── DAG operations ────────────────────────────────────────────────────────

    /// Emits a `system.checkpoint` event, returning its [`EventId`].
    pub async fn checkpoint(&self) -> Result<EventId, BusError> {
        let event = Event::new(kinds::CHECKPOINT, serde_json::json!({}));
        let id = event.id;
        self.publish(event).await?;
        Ok(id)
    }

    /// Reverts the active branch to immediately prior to `checkpoint_event_id`.
    ///
    /// Converts discarded events into an immutable rejected branch and triggers GC.
    /// Appends a durable `system.rollback` tombstone to the new active tip so
    /// persisted logs can be restored with the correct branch topology (see
    /// [`restore_from`](Self::restore_from)).
    /// Returns the new `BranchId`. Err if the checkpoint is undiscoverable.
    pub async fn rollback(&self, checkpoint_event_id: EventId) -> Result<BranchId, BusError> {
        let branch_id;
        let parent_event_id;
        let rejected_event_ids;
        {
            let mut store = self.inner.store.write().await;

            let pos = store
                .active_path
                .iter()
                .position(|&id| id == checkpoint_event_id)
                .ok_or(BusError::CheckpointNotFound(checkpoint_event_id))?;

            parent_event_id = if pos > 0 {
                Some(store.active_path[pos - 1])
            } else {
                None
            };

            rejected_event_ids = store.active_path[pos..].to_vec();

            branch_id = BranchId::new_v4();
            store.rejected_branches.push(RejectedBranch {
                id: branch_id,
                parent_event_id,
                event_ids: rejected_event_ids.clone(),
            });

            store.active_path.truncate(pos);
        }

        // Evict excess rejected branches and GC orphaned nodes.
        let (evicted_branches, evicted_nodes) = {
            let mut store = self.inner.store.write().await;
            store.evict_excess_branches(
                self.inner.config.max_retained_branches,
                self.inner.config.eviction_strategy.as_ref(),
            )
        };

        // Broadcast observability events (not appended to active log).
        {
            let mut subs = self.inner.subs.lock().unwrap_or_else(|e| e.into_inner());
            Self::fan_out(
                &mut subs,
                Self::mark_ephemeral(Event::new(
                    kinds::BRANCH_SEALED,
                    serde_json::json!({
                        "branch_id": branch_id.to_string(),
                        "checkpoint_event_id": checkpoint_event_id.to_string(),
                        "reason": "rejected_trajectory"
                    }),
                )),
            );
            if evicted_branches > 0 {
                Self::fan_out(
                    &mut subs,
                    Self::mark_ephemeral(Event::new(
                        kinds::SYSTEM_PRUNED,
                        serde_json::json!({
                            "evicted_branches": evicted_branches,
                            "evicted_nodes": evicted_nodes,
                        }),
                    )),
                );
            }
        }

        // Durable tombstone: records the branch topology in the active log so
        // exporters persist it and `restore_from` can replay the rollback
        // instead of resurrecting rejected events onto the active branch.
        self.publish(Event::new(
            kinds::SYSTEM_ROLLBACK,
            serde_json::json!({
                "branch_id": branch_id.to_string(),
                "checkpoint_event_id": checkpoint_event_id.to_string(),
                "parent_event_id": parent_event_id.map(|id| id.to_string()),
                "rejected_event_ids": rejected_event_ids
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>(),
            }),
        ))
        .await?;

        Ok(branch_id)
    }

    /// Retrieves all rejected branches rooted at `anchor_event_id`.
    pub async fn rejected_branches_from(&self, anchor_event_id: EventId) -> Vec<Vec<Event>> {
        let store = self.inner.store.read().await;
        store
            .rejected_branches
            .iter()
            .filter(|b| b.parent_event_id == Some(anchor_event_id))
            .map(|b| {
                b.event_ids
                    .iter()
                    .filter_map(|id| store.nodes.get(id).cloned())
                    .collect()
            })
            .collect()
    }

    /// Dumps all retained rejected branches alongside their IDs.
    pub async fn all_rejected_branches(&self) -> Vec<(BranchId, Vec<Event>)> {
        let store = self.inner.store.read().await;
        store
            .rejected_branches
            .iter()
            .map(|b| {
                let events = b
                    .event_ids
                    .iter()
                    .filter_map(|id| store.nodes.get(id).cloned())
                    .collect();
                (b.id, events)
            })
            .collect()
    }

    // ── Restore ───────────────────────────────────────────────────────────────

    /// Silently ingests a persisted event stream, preserving `parent_event_id`
    /// linkages **and replaying rollbacks**.
    ///
    /// `system.rollback` tombstones (written by [`rollback`](Self::rollback))
    /// are honored: the events they reference are moved off the active path
    /// into a reconstructed rejected branch, so a restored session never
    /// resurrects trajectories that were rolled away. Broadcast-only
    /// observability kinds (`system.branch_sealed`, `system.pruned`) that an
    /// exporter may have captured are skipped.
    ///
    pub async fn restore_from(&self, events: Vec<Event>) {
        use std::collections::HashSet;

        let mut store = self.inner.store.write().await;
        for event in events {
            // Events that were only ever fanned out to observers must not be
            // resurrected onto the active branch. Exporters persist them —
            // streaming deltas are worth replaying — but rebuilding history
            // from them would produce a branch made of message fragments with
            // no user message and no tool results, which is both wrong and
            // unusable as LLM context.
            //
            // The kind list covers logs written before the marker existed.
            let ephemeral = event
                .metadata
                .get(meta_keys::EPHEMERAL)
                .and_then(|v| v.as_bool())
                == Some(true);
            if ephemeral
                || matches!(
                    event.kind.as_str(),
                    kinds::ASSISTANT_DELTA | kinds::BRANCH_SEALED | kinds::SYSTEM_PRUNED
                )
            {
                continue;
            }

            if event.kind == kinds::SYSTEM_ROLLBACK {
                let rejected: HashSet<EventId> = event
                    .payload
                    .get("rejected_event_ids")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .filter_map(|s| Uuid::parse_str(s).ok())
                            .collect()
                    })
                    .unwrap_or_default();

                if !rejected.is_empty() {
                    let branch_id = event
                        .payload
                        .get("branch_id")
                        .and_then(|v| v.as_str())
                        .and_then(|s| Uuid::parse_str(s).ok())
                        .unwrap_or_else(BranchId::new_v4);
                    let parent_event_id = event
                        .payload
                        .get("parent_event_id")
                        .and_then(|v| v.as_str())
                        .and_then(|s| Uuid::parse_str(s).ok());

                    // Move the rejected events (in order) off the active path.
                    let event_ids: Vec<EventId> = store
                        .active_path
                        .iter()
                        .copied()
                        .filter(|id| rejected.contains(id))
                        .collect();
                    store.active_path.retain(|id| !rejected.contains(id));
                    store.rejected_branches.push(RejectedBranch {
                        id: branch_id,
                        parent_event_id,
                        event_ids,
                    });
                }
                // The tombstone itself lives on the active path, as on the
                // original bus.
            }

            let id = event.id;
            store.nodes.insert(id, event);
            store.active_path.push(id);
        }
    }

    // ── Utility ───────────────────────────────────────────────────────────────

    /// Blocks until the next event matching `predicate` arrives.
    ///
    /// Returns `Err(BusError::ChannelClosed)` if the bus is dropped while waiting.
    pub async fn wait_for<F>(&self, predicate: F) -> Result<Event, BusError>
    where
        F: Fn(&Event) -> bool + Send,
    {
        let mut rx = self.subscribe();
        loop {
            match rx.recv().await {
                Some(event) if predicate(&event) => return Ok(event),
                Some(_) => continue,
                None => return Err(BusError::ChannelClosed),
            }
        }
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

// ── Official publish transforms ───────────────────────────────────────────────

/// Returns a publish transform that replaces each secret string with `"[REDACTED]"`
/// in the serialized event payload before it is stored or fanned out.
///
/// Register it with [`EventBus::add_publish_transform`]:
///
/// ```no_run
/// use eventage::{EventBus, secrets_masking_transform};
///
/// let bus = EventBus::new();
/// bus.add_publish_transform(secrets_masking_transform(vec!["sk-secret".to_string()]));
/// ```
///
/// The in-memory DAG stores the masked copy; the original secret value is
/// never written to the JSONL event log or visible to subscribers.
pub fn secrets_masking_transform(
    secrets: Vec<String>,
) -> impl Fn(Event) -> Event + Send + Sync + 'static {
    move |mut event: Event| {
        let non_empty: Vec<&str> = secrets
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| s.as_str())
            .collect();
        if non_empty.is_empty() {
            return event;
        }
        // Serialize → string-replace → deserialize so all nested fields are covered.
        if let Ok(mut text) = serde_json::to_string(&event.payload) {
            for secret in &non_empty {
                text = text.replace(secret, "[REDACTED]");
            }
            if let Ok(masked) = serde_json::from_str(&text) {
                event.payload = masked;
            }
        }
        event
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::kinds;
    use serde_json::json;

    #[tokio::test]
    async fn publish_appends_to_log() {
        let bus = EventBus::new();
        bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "hi"})))
            .await
            .unwrap();
        bus.publish(Event::new(kinds::SYSTEM_HEARTBEAT, json!({})))
            .await
            .unwrap();
        assert_eq!(bus.log_len().await, 2);
    }

    #[tokio::test]
    async fn subscribe_receives_future_events() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe();

        bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "hello"})))
            .await
            .unwrap();

        let received = rx.recv().await.unwrap();
        assert_eq!(received.kind, kinds::USER_MESSAGE);
    }

    #[tokio::test]
    async fn parent_event_id_chain() {
        let bus = EventBus::new();
        bus.publish(Event::new(kinds::USER_MESSAGE, json!({})))
            .await
            .unwrap();
        bus.publish(Event::new(kinds::ASSISTANT_MESSAGE, json!({})))
            .await
            .unwrap();

        let log = bus.log().await;
        assert!(
            log[0].parent_event_id.is_none(),
            "first event has no parent"
        );
        assert_eq!(
            log[1].parent_event_id,
            Some(log[0].id),
            "second event points to first"
        );
    }

    #[tokio::test]
    async fn checkpoint_publishes_event() {
        let bus = EventBus::new();
        bus.publish(Event::new(kinds::USER_MESSAGE, json!({})))
            .await
            .unwrap();

        let cp_id = bus.checkpoint().await.unwrap();
        let log = bus.log().await;

        assert_eq!(log.len(), 2);
        assert_eq!(log[1].id, cp_id);
        assert_eq!(log[1].kind, kinds::CHECKPOINT);
    }

    #[tokio::test]
    async fn rollback_seals_rejected_branch() {
        let bus = EventBus::new();

        bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "q"})))
            .await
            .unwrap();
        let cp_id = bus.checkpoint().await.unwrap();
        bus.publish(Event::new(
            kinds::ASSISTANT_MESSAGE,
            json!({"content": "bad"}),
        ))
        .await
        .unwrap();
        assert_eq!(bus.log_len().await, 3);

        bus.rollback(cp_id).await.unwrap();

        // Active branch: the user message plus the durable rollback tombstone.
        let log = bus.log().await;
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].kind, kinds::USER_MESSAGE);
        assert_eq!(log[1].kind, kinds::SYSTEM_ROLLBACK);

        let rejected = bus.rejected_branches_from(log[0].id).await;
        assert_eq!(rejected.len(), 1);
        assert_eq!(rejected[0].len(), 2);
        assert_eq!(rejected[0][0].kind, kinds::CHECKPOINT);
        assert_eq!(rejected[0][1].kind, kinds::ASSISTANT_MESSAGE);
    }

    #[tokio::test]
    async fn new_events_after_rollback_continue_from_anchor() {
        let bus = EventBus::new();

        bus.publish(Event::new(kinds::USER_MESSAGE, json!({})))
            .await
            .unwrap();
        let cp_id = bus.checkpoint().await.unwrap();
        bus.publish(Event::new(
            kinds::ASSISTANT_MESSAGE,
            json!({"content": "bad"}),
        ))
        .await
        .unwrap();

        bus.rollback(cp_id).await.unwrap();

        bus.publish(Event::new(
            kinds::ASSISTANT_MESSAGE,
            json!({"content": "retry"}),
        ))
        .await
        .unwrap();

        // [user, rollback tombstone, retry] — the retry chains off the tombstone.
        let log = bus.log().await;
        assert_eq!(log.len(), 3);
        assert_eq!(log[1].kind, kinds::SYSTEM_ROLLBACK);
        assert_eq!(log[2].payload["content"], "retry");
        assert_eq!(log[2].parent_event_id, Some(log[1].id));
    }

    #[tokio::test]
    async fn streaming_deltas_do_not_come_back_as_history() {
        // An exporter observes broadcasts as well as publishes, so a
        // persisted log contains both. Restoring must rebuild only what was
        // actually on the active branch: a session reopened from a streaming
        // run would otherwise consist of message fragments with no user
        // message and no tool results — unusable as context, and silently so.
        let bus = EventBus::new();
        let mut observed = bus.subscribe();

        bus.publish(Event::new(kinds::USER_MESSAGE, json!({ "text": "hi" })))
            .await
            .unwrap();
        bus.broadcast(Event::new(
            kinds::ASSISTANT_DELTA,
            json!({ "content": "he" }),
        ));
        bus.broadcast(Event::new(
            kinds::ASSISTANT_DELTA,
            json!({ "content": "llo" }),
        ));
        bus.publish(Event::new(
            kinds::ASSISTANT_MESSAGE,
            json!({ "content": "hello" }),
        ))
        .await
        .unwrap();

        // Everything an exporter would have written down.
        let mut persisted = Vec::new();
        for _ in 0..4 {
            persisted.push(observed.recv().await.expect("observer sees every event"));
        }
        assert_eq!(persisted.len(), 4, "observers see broadcasts too");

        let reopened = EventBus::new();
        reopened.restore_from(persisted).await;
        let log = reopened.log().await;

        let kinds_restored: Vec<&str> = log.iter().map(|e| e.kind.as_str()).collect();
        assert_eq!(
            kinds_restored,
            vec![kinds::USER_MESSAGE, kinds::ASSISTANT_MESSAGE],
            "only durable events belong on the restored branch"
        );
    }

    #[tokio::test]
    async fn broadcasts_are_marked_so_any_consumer_can_tell() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe();
        bus.broadcast(Event::new(kinds::ASSISTANT_DELTA, json!({})));
        let event = rx.recv().await.unwrap();
        assert_eq!(
            event
                .metadata
                .get(meta_keys::EPHEMERAL)
                .and_then(|v| v.as_bool()),
            Some(true)
        );
    }

    #[tokio::test]
    async fn restore_replays_rollbacks_instead_of_resurrecting_them() {
        // Simulate an exporter capturing every published + broadcast event.
        let bus = EventBus::new();
        let mut tap = bus.subscribe();

        bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "q"})))
            .await
            .unwrap();
        let cp = bus.checkpoint().await.unwrap();
        bus.publish(Event::new(
            kinds::ASSISTANT_MESSAGE,
            json!({"content": "bad"}),
        ))
        .await
        .unwrap();
        bus.rollback(cp).await.unwrap();
        bus.publish(Event::new(
            kinds::ASSISTANT_MESSAGE,
            json!({"content": "good"}),
        ))
        .await
        .unwrap();

        let mut captured = Vec::new();
        while let Ok(Some(e)) =
            tokio::time::timeout(std::time::Duration::from_millis(50), tap.recv()).await
        {
            captured.push(e);
        }

        // Restore the captured stream into a fresh bus.
        let restored = EventBus::new();
        restored.restore_from(captured).await;

        let original_log: Vec<EventId> = bus.log().await.iter().map(|e| e.id).collect();
        let restored_log: Vec<EventId> = restored.log().await.iter().map(|e| e.id).collect();
        assert_eq!(
            restored_log, original_log,
            "restored active path must match the original exactly (no resurrected events)"
        );

        // The rejected branch is reconstructed too.
        let anchor = bus.log().await[0].id;
        let rejected = restored.rejected_branches_from(anchor).await;
        assert_eq!(rejected.len(), 1);
        assert_eq!(rejected[0].len(), 2, "checkpoint + bad assistant message");
        assert_eq!(rejected[0][1].payload["content"], "bad");
    }

    #[tokio::test]
    async fn checkpoint_not_found_returns_error() {
        let bus = EventBus::new();
        let fake_id = EventId::new_v4();
        let result = bus.rollback(fake_id).await;
        assert!(matches!(result, Err(BusError::CheckpointNotFound(_))));
    }

    #[tokio::test]
    async fn wait_for_matches_predicate() {
        let bus = EventBus::new();
        let bus2 = bus.clone();

        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            bus2.publish(Event::new(kinds::TOOL_RESULT, json!({"id": "42"})))
                .await
                .unwrap();
        });

        let event = bus
            .wait_for(|e| e.kind == kinds::TOOL_RESULT)
            .await
            .unwrap();
        assert_eq!(event.kind, kinds::TOOL_RESULT);
    }

    #[tokio::test]
    async fn bounded_branches_evict_oldest() {
        let bus = EventBus::with_config(BusConfig {
            max_retained_branches: 2,
            ..BusConfig::default()
        });

        async fn make_rejected_branch(bus: &EventBus) -> EventId {
            use serde_json::json;
            bus.publish(Event::new(
                kinds::ASSISTANT_MESSAGE,
                json!({"content": "bad"}),
            ))
            .await
            .unwrap();
            let cp = bus.checkpoint().await.unwrap();
            bus.publish(Event::new(
                kinds::ASSISTANT_MESSAGE,
                json!({"content": "wrong"}),
            ))
            .await
            .unwrap();
            bus.rollback(cp).await.unwrap();
            cp
        }

        bus.publish(Event::new(
            kinds::USER_MESSAGE,
            serde_json::json!({"text": "q"}),
        ))
        .await
        .unwrap();

        let cp1 = make_rejected_branch(&bus).await;
        let cp2 = make_rejected_branch(&bus).await;
        let cp3 = make_rejected_branch(&bus).await;

        let _ = (cp1, cp2, cp3);

        let all = bus.all_rejected_branches().await;
        assert_eq!(all.len(), 2, "oldest branch should have been evicted");
    }

    #[tokio::test]
    async fn pruned_event_is_broadcast() {
        let bus = EventBus::with_config(BusConfig {
            max_retained_branches: 1,
            ..BusConfig::default()
        });
        let mut rx = bus.subscribe();

        bus.publish(Event::new(
            kinds::USER_MESSAGE,
            serde_json::json!({"text": "q"}),
        ))
        .await
        .unwrap();

        let cp1 = bus.checkpoint().await.unwrap();
        bus.publish(Event::new(
            kinds::ASSISTANT_MESSAGE,
            serde_json::json!({"content": "a"}),
        ))
        .await
        .unwrap();
        bus.rollback(cp1).await.unwrap();

        let cp2 = bus.checkpoint().await.unwrap();
        bus.publish(Event::new(
            kinds::ASSISTANT_MESSAGE,
            serde_json::json!({"content": "b"}),
        ))
        .await
        .unwrap();
        bus.rollback(cp2).await.unwrap();

        let _ = cp2;

        let mut found_pruned = false;
        while let Ok(event) =
            tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv()).await
        {
            if let Some(e) = event {
                if e.kind == kinds::SYSTEM_PRUNED {
                    found_pruned = true;
                }
            }
        }
        assert!(
            found_pruned,
            "system.pruned should be broadcast on eviction"
        );
    }

    #[tokio::test]
    async fn bounded_subscriber_drops_on_overflow() {
        let bus = EventBus::with_config(BusConfig {
            subscriber_capacity: 2,
            ..BusConfig::default()
        });
        let mut rx = bus.subscribe();

        for i in 0..5usize {
            bus.publish(Event::new(kinds::USER_MESSAGE, json!({ "i": i })))
                .await
                .unwrap();
        }

        let mut count = 0usize;
        while tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv())
            .await
            .ok()
            .flatten()
            .is_some()
        {
            count += 1;
        }
        assert_eq!(
            count, 2,
            "bounded subscriber should receive exactly capacity events"
        );

        assert_eq!(
            bus.log_len().await,
            5,
            "bus.log() always has the full history"
        );
    }

    #[tokio::test]
    async fn no_capacity_limit_high_volume() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe();

        let n = 2000usize;
        for i in 0..n {
            bus.publish(Event::new(kinds::USER_MESSAGE, json!({ "i": i })))
                .await
                .unwrap();
        }

        let mut count = 0usize;
        while count < n {
            let event = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
                .await
                .expect("timed out waiting for event")
                .expect("channel closed unexpectedly");
            assert_eq!(event.kind, kinds::USER_MESSAGE);
            count += 1;
        }
        assert_eq!(count, n);
    }
}
