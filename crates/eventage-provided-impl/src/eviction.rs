//! Branch eviction strategies.
//!
//! Determines action when a rejected branch exceeds [`BusConfig::max_retained_branches`].
//!
//! # Provided Strategies
//! - [`PruneStrategy`] (core): Deletes immediately. Memory-efficient.
//! - [`EpitaphStrategy`]: Uses an LLM to generate a concise summary before deleting.
//!
//! # Custom Strategies
//! Implement [`BranchEvictionStrategy`] for custom logic (e.g., database persistence).
//!
//! ```no_run
//! use eventage_provided_impl::{BranchData, BranchEvictionStrategy};
//!
//! struct AuditStrategy;
//!
//! impl BranchEvictionStrategy for AuditStrategy {
//!     fn on_evict(&self, branch: BranchData) {
//!         // Must be fast — runs under the DAG write lock.
//!         // Delegate I/O to a background channel.
//!         tracing::info!(branch_id = %branch.id, "branch evicted");
//!     }
//! }
//! ```

use eventage_core::{BranchData, BranchEvictionStrategy, BranchId};
use eventage_llm::{ChatMessage, LlmProvider};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// Thread-safe map of [`BranchId`] to LLM-generated epitaph strings.
///
/// Updated asynchronously after eviction.
pub type EpitaphStore = Arc<Mutex<HashMap<BranchId, String>>>;

/// Uses an LLM to generate a concise epitaph for evicted branches.
///
/// Processes eviction asynchronously to avoid blocking. The generated epitaph
/// acts as a "hard negative" memory, steering the LLM away from failed approaches.
///
/// # Example
///
/// ```no_run
/// use eventage_provided_impl::EpitaphStrategy;
/// use eventage_provided_impl::eventage_core::{BusConfig, EventBus};
/// use eventage_llm::OpenAiProvider;
/// use std::sync::Arc;
///
/// let llm = Arc::new(OpenAiProvider::ollama("qwen3:4b"));
/// let strategy = EpitaphStrategy::new(llm);
/// let epitaphs = strategy.epitaphs(); // shared handle — read at any point
///
/// let bus = EventBus::with_config(BusConfig {
///     max_retained_branches: 5,
///     eviction_strategy: Arc::new(strategy),
///     ..Default::default()
/// });
///
/// // Later — read generated epitaphs:
/// let map = epitaphs.lock().unwrap();
/// for (branch_id, text) in map.iter() {
///     println!("{branch_id}: {text}");
/// }
/// ```
pub struct EpitaphStrategy {
    tx: mpsc::UnboundedSender<BranchData>,
    store: EpitaphStore,
}

impl EpitaphStrategy {
    /// Creates an `EpitaphStrategy` backed by `llm`.
    ///
    /// Spawns a background task to process evicted branches.
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        let (tx, rx) = mpsc::unbounded_channel::<BranchData>();
        let store: EpitaphStore = Arc::new(Mutex::new(HashMap::new()));
        tokio::spawn(epitaph_task(rx, llm, store.clone()));
        Self { tx, store }
    }

    /// Returns a shared handle to the asynchronously-populated epitaph map.
    pub fn epitaphs(&self) -> EpitaphStore {
        self.store.clone()
    }
}

impl BranchEvictionStrategy for EpitaphStrategy {
    /// Non-blocking forward of `branch` to the background summarisation task.
    fn on_evict(&self, branch: BranchData) {
        let _ = self.tx.send(branch);
    }

    fn name(&self) -> &'static str {
        "epitaph"
    }
}

// ── Background task ───────────────────────────────────────────────────────────

async fn epitaph_task(
    mut rx: mpsc::UnboundedReceiver<BranchData>,
    llm: Arc<dyn LlmProvider>,
    store: EpitaphStore,
) {
    while let Some(branch) = rx.recv().await {
        let branch_id = branch.id;
        let epitaph = generate_epitaph(llm.as_ref(), &branch).await;
        tracing::debug!(branch_id = %branch_id, "epitaph generated");
        store.lock().unwrap().insert(branch_id, epitaph);
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
