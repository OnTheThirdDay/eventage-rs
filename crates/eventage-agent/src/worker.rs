//! Other e.g. non-LLM participants on the event bus via the [`EventWorker`] trait.
//!
//! Workers execute arbitrary async logic in response to subscribed events.
//! Use them for workflows, external API integration, human-in-the-loop, or memory.

use async_trait::async_trait;
use eventage_core::{Event, EventBus};
use thiserror::Error;

// ── WorkerError ───────────────────────────────────────────────────────────────

/// Errors executing an [`EventWorker`].
#[derive(Debug, Error)]
pub enum WorkerError {
    #[error("bus error: {0}")]
    Bus(#[from] eventage_core::BusError),
    #[error("worker error: {0}")]
    Worker(String),
}

// ── EventWorker trait ─────────────────────────────────────────────────────────

/// A participant that reacts to events with arbitrary async code.
///
/// Workers provide a clean mechanism to integrate deterministic logic alongside
/// LLM agents. They filter traffic via [`subscribed_kinds`](Self::subscribed_kinds).
/// Returning an empty list means subscribing to all events.
///
/// # Example — workflow sequencer
///
/// ```rust,no_run
/// use eventage_agent::worker::{EventWorker, WorkerError};
/// use eventage_core::{Event, EventBus, kinds};
/// use async_trait::async_trait;
///
/// /// Publishes the next workflow step trigger after every completed agent cycle.
/// pub struct StepAdvancer {
///     pub next_step: String,
/// }
///
/// #[async_trait]
/// impl EventWorker for StepAdvancer {
///     fn subscribed_kinds(&self) -> Vec<String> {
///         vec![kinds::AGENT_CYCLE_END.to_string()]
///     }
///
///     async fn handle(&self, _event: &Event, bus: &EventBus) -> Result<(), WorkerError> {
///         bus.publish(eventage_core::Event::new(
///             "workflow.step.ready",
///             serde_json::json!({ "step": self.next_step }),
///         ))
///         .await
///         .map_err(WorkerError::Bus)
///     }
/// }
/// ```
///
/// # Example — human approval bridge
///
/// ```rust,no_run
/// use eventage_agent::worker::{EventWorker, WorkerError};
/// use eventage_core::{Event, EventBus};
/// use async_trait::async_trait;
///
/// pub struct HumanApprovalBridge;
///
/// #[async_trait]
/// impl EventWorker for HumanApprovalBridge {
///     fn subscribed_kinds(&self) -> Vec<String> {
///         vec!["system.approval_needed".to_string()]
///     }
///
///     async fn handle(&self, event: &Event, bus: &EventBus) -> Result<(), WorkerError> {
///         let description = event.payload["description"].as_str().unwrap_or("action");
///         eprint!("Approve '{}'? [y/N]: ", description);
///         let mut line = String::new();
///         std::io::stdin().read_line(&mut line).ok();
///         let kind = if line.trim().eq_ignore_ascii_case("y") {
///             "system.approved"
///         } else {
///             "system.rejected"
///         };
///         bus.publish(eventage_core::Event::new(kind, serde_json::json!({})))
///             .await
///             .map_err(WorkerError::Bus)
///     }
/// }
/// ```
#[async_trait]
pub trait EventWorker: Send + Sync {
    /// Kinds of events to handle. Return empty to handle all events.
    fn subscribed_kinds(&self) -> Vec<String> {
        vec![]
    }

    /// Processes a single matching event.
    ///
    /// Can publish new events. Return `Err` only for fatal failures.
    async fn handle(&self, event: &Event, bus: &EventBus) -> Result<(), WorkerError>;
}
