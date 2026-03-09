//! [`DynamicWorkerHandle`] — add [`EventWorker`]s to a running system at runtime.
//!
//! Unlike [`WorkerSet`] (which requires all workers to be registered before
//! `run_on()` is called), `DynamicWorkerHandle` allows you to spawn new workers
//! at any time during execution — useful for loading MCP servers on demand,
//! enabling optional monitoring, or activating pipeline stages conditionally.
//!
//! # Example
//!
//! ```rust,no_run
//! use eventage_agent::worker::{EventWorker, WorkerError};
//! use eventage_core::{Event, EventBus};
//! use eventage_provided_impl::DynamicWorkerHandle;
//! use async_trait::async_trait;
//!
//! struct MetricsCollector;
//!
//! #[async_trait]
//! impl EventWorker for MetricsCollector {
//!     async fn handle(&self, _e: &Event, _bus: &EventBus) -> Result<(), WorkerError> {
//!         Ok(())
//!     }
//! }
//!
//! # async fn example() {
//! let bus = EventBus::new();
//! let handle = DynamicWorkerHandle::new(bus.clone());
//!
//! // Start a worker later — perhaps in response to a config event.
//! handle.add_worker(MetricsCollector);
//! # }
//! ```

use eventage_agent::worker::{EventWorker, WorkerError};
use eventage_core::EventBus;
use std::sync::Arc;

/// Spawns [`EventWorker`]s as background tasks on a shared [`EventBus`].
///
/// Each call to [`add`][Self::add] creates a new subscription and Tokio task.
/// The task runs until the bus closes (all `EventBus` clones dropped).
///
/// `DynamicWorkerHandle` is `Clone` — all clones spawn tasks on the same bus.
#[derive(Clone)]
pub struct DynamicWorkerHandle {
    bus: EventBus,
}

impl DynamicWorkerHandle {
    /// Create a handle that spawns workers on `bus`.
    pub fn new(bus: EventBus) -> Self {
        Self { bus }
    }

    /// Spawn `worker` as a background task. Returns a
    /// [`tokio::task::JoinHandle`]; dropping it does **not** cancel the task.
    pub fn add_worker<W: EventWorker + 'static>(
        &self,
        worker: W,
    ) -> tokio::task::JoinHandle<Result<(), WorkerError>> {
        let bus = self.bus.clone();
        let worker = Arc::new(worker);
        tokio::spawn(run_worker(worker, bus))
    }

    /// Spawn a pre-boxed worker.
    pub fn add_worker_arc(
        &self,
        worker: Arc<dyn EventWorker>,
    ) -> tokio::task::JoinHandle<Result<(), WorkerError>> {
        let bus = self.bus.clone();
        tokio::spawn(run_worker(worker, bus))
    }
}

async fn run_worker(worker: Arc<dyn EventWorker>, bus: EventBus) -> Result<(), WorkerError> {
    let kinds = worker.subscribed_kinds();
    let mut rx = bus.subscribe();
    while let Some(event) = rx.recv().await {
        let interested = kinds.is_empty() || kinds.iter().any(|k| k == &event.kind);
        if interested {
            worker.handle(&event, &bus).await?;
        }
    }
    Ok(())
}
