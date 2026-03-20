//! Other e.g. non-LLM participants on the event bus via the [`EventWorker`] trait.
//!
//! Workers execute arbitrary async logic in response to subscribed events.
//! Use them for workflows, external API integration, human-in-the-loop, or memory.

use async_trait::async_trait;
use crate::bus::EventBus;
use crate::event::Event;
use crate::error::BusError;
use std::sync::Arc;
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

    /// Spawn all workers and drive them until the bus closes or a fatal error occurs.
    pub async fn run_on(self, bus: EventBus) -> Result<(), WorkerError> {
        let mut set: JoinSet<Result<(), WorkerError>> = JoinSet::new();

        for worker in self.workers {
            let bus = bus.clone();
            set.spawn(async move {
                let kinds = worker.subscribed_kinds();
                let mut rx = bus.subscribe();
                while let Some(event) = rx.recv().await {
                    let interested = kinds.is_empty() || kinds.iter().any(|k| k == &event.kind);
                    if interested {
                        worker.handle(&event, &bus).await?;
                    }
                }
                Ok(())
            });
        }

        let mut first_err: Option<WorkerError> = None;
        while let Some(result) = set.join_next().await {
            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    warn!("event worker exited with error: {e}");
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                    set.abort_all();
                }
                Err(join_err) => {
                    warn!("event worker task panicked: {join_err}");
                }
            }
        }

        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
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
