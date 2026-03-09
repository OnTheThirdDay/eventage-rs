//! [`WorkerSet`] — runs multiple [`EventWorker`]s concurrently.

use eventage_agent::worker::{EventWorker, WorkerError};
use eventage_core::EventBus;
use std::sync::Arc;
use tokio::task::JoinSet;
use tracing::warn;

// ── WorkerSet ─────────────────────────────────────────────────────────────────

/// Runs multiple [`EventWorker`]s concurrently on a shared [`EventBus`].
///
/// Each worker runs in its own task, subscribing independently to avoid head-of-line blocking.
///
/// # Example
///
/// ```rust,no_run
/// use eventage_agent::worker::{EventWorker, WorkerError};
/// use eventage_provided_impl::WorkerSet;
/// use eventage_core::{Event, EventBus};
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
/// WorkerSet::new()
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
