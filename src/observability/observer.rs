use super::error::ObsError;
use super::exporter::ObservabilityExporter;
use crate::bus::EventBus;
use crate::event::Event;
use std::sync::Arc;
use tracing::error;

/// Subscribes to an [`EventBus`] and dispenses events to registered [`ObservabilityExporter`]s.
///
/// Typically runs as a background task alongside agents.
pub struct BusObserver {
    bus: EventBus,
    exporters: Vec<Arc<dyn ObservabilityExporter>>,
    /// Export failures so far. Shared so a caller can watch it while the
    /// loop is still running.
    failures: Arc<std::sync::atomic::AtomicUsize>,
}

impl BusObserver {
    pub fn new(bus: EventBus) -> Self {
        Self {
            bus,
            exporters: vec![],
            failures: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    pub fn add_exporter(mut self, exporter: impl ObservabilityExporter + 'static) -> Self {
        self.exporters.push(Arc::new(exporter));
        self
    }

    pub fn add_exporter_arc(mut self, exporter: Arc<dyn ObservabilityExporter>) -> Self {
        self.exporters.push(exporter);
        self
    }

    /// Subscribe now, so nothing published before the loop starts is missed.
    ///
    /// [`run`](Self::run) subscribes inside the future, which is only polled
    /// once the task is scheduled — anything published in between is never
    /// seen. For a persistence exporter that means the opening events of a
    /// session can be absent from the log it will later be resumed from, and
    /// nothing reports it. Callers that spawn the observer should subscribe
    /// here first and hand the receiver to [`run_with`](Self::run_with).
    pub fn subscribe(&self) -> crate::bus::BusReceiver {
        self.bus.subscribe()
    }

    /// Drives the observer loop until the bus closes, forwarding events concurrently.
    /// Exporter errors are logged but do not disrupt the run.
    pub async fn run(self) {
        let rx = self.bus.subscribe();
        self.run_with(rx).await;
    }

    /// As [`run`](Self::run), against a receiver obtained earlier.
    ///
    /// Returns the number of export failures seen. They are logged as they
    /// happen and do not stop the loop — one exporter should not take a
    /// session down — but a caller that persists through an exporter needs to
    /// be able to find out, because a session whose log is incomplete looks
    /// perfectly healthy until somebody tries to resume it.
    pub async fn run_with(self, mut rx: crate::bus::BusReceiver) -> usize {
        while let Some(event) = rx.recv().await {
            self.dispatch(&event).await;
        }
        // Flush all exporters on shutdown.
        let mut failures = self.failures.load(std::sync::atomic::Ordering::Relaxed);
        for exp in &self.exporters {
            if let Err(e) = exp.flush().await {
                error!("exporter flush error: {e}");
                failures += 1;
            }
        }
        failures
    }

    /// Replays a snapshot of the current bus event log through registered exporters.
    pub async fn export_snapshot(&self) -> Result<(), ObsError> {
        let events: Vec<Event> = self.bus.log().await;
        for event in &events {
            self.dispatch(event).await;
        }
        Ok(())
    }

    /// A live count of export failures, readable while the loop runs.
    pub fn failures(&self) -> Arc<std::sync::atomic::AtomicUsize> {
        Arc::clone(&self.failures)
    }

    async fn dispatch(&self, event: &Event) {
        for exp in &self.exporters {
            if let Err(e) = exp.export(event).await {
                error!("exporter error: {e}");
                self.failures
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
}
