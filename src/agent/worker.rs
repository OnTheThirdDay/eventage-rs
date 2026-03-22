//! Other e.g. non-LLM participants on the event bus via the [`EventWorker`] trait.
//!
//! Workers execute arbitrary async logic in response to subscribed events.
//! Use them for workflows, external API integration, human-in-the-loop, or memory.

use async_trait::async_trait;
use crate::bus::EventBus;
use crate::event::Event;
use crate::error::BusError;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::task::JoinSet;
use tracing::warn;

// ── WorkerError ───────────────────────────────────────────────────────────────

/// Errors executing an [`EventWorker`].
#[derive(Debug, Error)]
pub enum WorkerError {
    #[error("bus error: {0}")]
    Bus(#[from] BusError),
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
/// use eventage::agent::worker::{EventWorker, WorkerError};
/// use eventage::{Event, EventBus, kinds};
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
///         bus.publish(Event::new(
///             "workflow.step.ready",
///             serde_json::json!({ "step": self.next_step }),
///         ))
///         .await
///         .map_err(WorkerError::Bus)
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

// ── WorkerSet ─────────────────────────────────────────────────────────────────

/// Runs multiple [`EventWorker`]s concurrently on a shared [`EventBus`].
///
/// Each worker runs in its own task, subscribing independently to avoid head-of-line blocking.
///
/// # Example
///
/// ```rust,no_run
/// use eventage::agent::worker::{EventWorker, WorkerError};
/// use eventage::agent::WorkerSet;
/// use eventage::{Event, EventBus};
/// use async_trait::async_trait;
///
/// struct MyWorker;
///
/// #[async_trait]
/// impl EventWorker for MyWorker {
///     async fn handle(&self, _e: &Event, _b: &EventBus) -> Result<(), WorkerError> { Ok(()) }
/// }
///
/// # async fn example() {
/// let bus = EventBus::default();
/// eventage::agent::WorkerSet::new()
///     .add_worker(MyWorker)
///     .run_on(bus)
///     .await
///     .unwrap();
/// # }
/// ```
#[derive(Default)]
pub struct WorkerSet {
    workers: Vec<Arc<dyn EventWorker>>,
}

impl WorkerSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a worker to the set.
    pub fn add_worker(mut self, worker: impl EventWorker + 'static) -> Self {
        self.workers.push(Arc::new(worker));
        self
    }

    /// Add a pre-boxed worker.
    pub fn add_worker_arc(mut self, worker: Arc<dyn EventWorker>) -> Self {
        self.workers.push(worker);
        self
    }

    /// Spawn all workers and drive them until the bus closes.
    ///
    /// Each worker runs under supervision: if it panics or returns a non-bus
    /// error it is automatically restarted after a 1-second delay.  Only a
    /// bus-closed signal (all `EventBus` clones dropped) causes workers to stop
    /// cleanly.  This prevents any single misbehaving worker from bringing down
    /// the whole set.
    pub async fn run_on(self, bus: EventBus) -> Result<(), WorkerError> {
        let mut set: JoinSet<()> = JoinSet::new();

        for worker in self.workers {
            let bus = bus.clone();
            set.spawn(run_worker_supervised(worker, bus));
        }

        while set.join_next().await.is_some() {}
        Ok(())
    }
}

// ── DynamicWorkerHandle ───────────────────────────────────────────────────────

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

/// Run a worker under supervision: restart it on panic or transient error.
///
/// Stops only when the bus closes (`rx.recv()` returns `None` → `Ok(())`) or
/// when the worker returns a `WorkerError::Bus` (channel gone).
async fn run_worker_supervised(worker: Arc<dyn EventWorker>, bus: EventBus) {
    loop {
        let result = tokio::spawn(run_worker(worker.clone(), bus.clone())).await;
        match result {
            // Bus closed — exit cleanly.
            Ok(Ok(())) => break,
            Ok(Err(WorkerError::Bus(_))) => break,
            // Transient error — log and restart.
            Ok(Err(e)) => {
                warn!(error = %e, "event worker returned error — restarting in 1 s");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            // Panic in the worker task — log and restart.
            Err(panic_err) => {
                warn!(error = %panic_err, "event worker panicked — restarting in 1 s");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    }
}
