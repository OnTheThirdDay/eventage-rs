//! Branch eviction strategies.
//!
//! Determines action when a rejected branch exceeds [`crate::BusConfig::max_retained_branches`].
//!
//! # Provided Strategies
//! - [`PruneStrategy`](crate::PruneStrategy) (core): Deletes immediately. Memory-efficient.
//! - [`EpitaphStrategy`]: Uses an LLM to generate a concise summary before deleting.
//!
//! # Custom Strategies
//! Implement [`BranchEvictionStrategy`] for custom logic (e.g., database persistence).

use crate::bus::{BranchData, BranchEvictionStrategy, BranchId};
use crate::llm::{ChatMessage, LlmProvider};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use tokio::sync::mpsc;

/// Thread-safe map of [`BranchId`] to LLM-generated epitaph strings.
///
/// Updated asynchronously after eviction.
pub type EpitaphStore = Arc<Mutex<HashMap<BranchId, String>>>;

/// Uses an LLM to generate a concise epitaph for evicted branches.
///
/// Processes eviction asynchronously to avoid blocking. The generated epitaph
/// acts as a "hard negative" memory, steering the LLM away from failed approaches.
/// Bounded queue depth for pending epitaph generation requests.
const EPITAPH_QUEUE_DEPTH: usize = 256;

pub struct EpitaphStrategy {
    tx: mpsc::Sender<BranchData>,
    store: EpitaphStore,
    /// Set by [`publish_to`](Self::publish_to) once the bus exists.
    bus: Arc<OnceLock<crate::EventBus>>,
}

impl EpitaphStrategy {
    /// Creates an `EpitaphStrategy` backed by `llm`.
    ///
    /// Spawns a background task to process evicted branches. The queue is
    /// bounded to `EPITAPH_QUEUE_DEPTH` entries; excess evictions are dropped
    /// with a warning rather than growing the queue unboundedly.
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        let (tx, rx) = mpsc::channel::<BranchData>(EPITAPH_QUEUE_DEPTH);
        let store: EpitaphStore = Arc::new(Mutex::new(HashMap::new()));
        let bus: Arc<OnceLock<crate::EventBus>> = Arc::new(OnceLock::new());
        tokio::spawn(epitaph_task(rx, llm, store.clone(), Arc::clone(&bus)));
        Self { tx, store, bus }
    }

    /// Also publish each epitaph onto `bus`, as `system.epitaph`.
    ///
    /// Attached after construction rather than taken by the constructor,
    /// because the bus this belongs to is usually the one being built *with*
    /// this strategy — a bus takes its eviction policy at construction, so a
    /// constructor argument would force the caller to invent a second bus and
    /// publish into one nothing reads.
    ///
    /// Worth doing at all because an epitaph is otherwise only in memory: the
    /// branch's events have been deleted, this sentence is all that remains
    /// of them, and reopening the session would lose it.
    ///
    /// Idempotent; the first bus attached wins.
    pub fn publish_to(&self, bus: crate::EventBus) {
        let _ = self.bus.set(bus);
    }

    /// Returns a shared handle to the asynchronously-populated epitaph map.
    pub fn epitaphs(&self) -> EpitaphStore {
        self.store.clone()
    }
}

impl BranchEvictionStrategy for EpitaphStrategy {
    /// Non-blocking forward of `branch` to the background summarisation task.
    /// Drops the branch silently with a warning if the queue is full.
    fn on_evict(&self, branch: BranchData) {
        if let Err(e) = self.tx.try_send(branch) {
            tracing::warn!(
                branch_id = %e.into_inner().id,
                "EpitaphStrategy: queue full ({EPITAPH_QUEUE_DEPTH} pending) — epitaph dropped"
            );
        }
    }

    fn name(&self) -> &'static str {
        "epitaph"
    }
}

// ── Background task ───────────────────────────────────────────────────────────

async fn epitaph_task(
    mut rx: mpsc::Receiver<BranchData>,
    llm: Arc<dyn LlmProvider>,
    store: EpitaphStore,
    bus: Arc<OnceLock<crate::EventBus>>,
) {
    while let Some(branch) = rx.recv().await {
        let branch_id = branch.id;
        let events_lost = branch.events.len();
        let epitaph = generate_epitaph(llm.as_ref(), &branch).await;
        tracing::debug!(branch_id = %branch_id, "epitaph generated");

        if let Some(bus) = bus.get() {
            // Durable: the branch's events are gone and this sentence is all
            // that remains of them.
            let _ = bus
                .publish(crate::Event::new(
                    crate::event::kinds::SYSTEM_EPITAPH,
                    serde_json::json!({
                        "branch_id": branch_id.to_string(),
                        "events_lost": events_lost,
                        "epitaph": epitaph,
                    }),
                ))
                .await;
        }

        store
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(branch_id, epitaph);
    }
}

async fn generate_epitaph(llm: &dyn LlmProvider, branch: &BranchData) -> String {
    let event_summary = branch
        .events
        .iter()
        .map(|e| {
            let payload = e.payload.as_object().is_some_and(|m| !m.is_empty());
            if payload {
                format!("[{}] {}", e.kind, e.payload)
            } else {
                format!("[{}]", e.kind)
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    let parent = branch
        .parent_event_id
        .map(|id| id.to_string())
        .unwrap_or_else(|| "root".into());

    let messages = vec![
        ChatMessage::system(
            "You generate concise epitaphs for rejected agent execution branches. \
             Given a chronological list of events from a failed or abandoned trajectory, \
             write one or two sentences that capture what was attempted and why it failed \
             or was abandoned. Be factual and brief — this summary will be stored as \
             negative context to guide future attempts away from the same mistake.",
        ),
        ChatMessage::user(format!(
            "Branch {branch_id} (diverged from {parent}):\n\n{event_summary}",
            branch_id = branch.id,
        )),
    ];

    match llm.complete(messages, vec![]).await {
        Ok(resp) => resp
            .content
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| format!("Branch {} — no summary generated.", branch.id)),
        Err(e) => {
            tracing::warn!(
                branch_id = %branch.id,
                "EpitaphStrategy: LLM call failed: {e}"
            );
            format!("Branch {} — epitaph generation failed: {e}", branch.id)
        }
    }
}
