use crate::error::ObsError;
use crate::exporter::ObservabilityExporter;
use eventage_core::{Event, EventBus};
use std::sync::Arc;
use tracing::error;

/// Subscribes to an [`EventBus`] and dispenses events to registered [`ObservabilityExporter`]s.
///
/// Typically runs as a background task alongside agents.
pub struct BusObserver {
    bus: EventBus,
    exporters: Vec<Arc<dyn ObservabilityExporter>>,
}

impl BusObserver {
    pub fn new(bus: EventBus) -> Self {
        Self {
            bus,
            exporters: vec![],
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

    /// Drives the observer loop until the bus closes, forwarding events concurrently.
    /// Exporter errors are logged but do not disrupt the run.
    pub async fn run(self) {
        let mut rx = self.bus.subscribe();
        while let Some(event) = rx.recv().await {
            self.dispatch(&event).await;
        }
        // Flush all exporters on shutdown.
        for exp in &self.exporters {
            if let Err(e) = exp.flush().await {
                error!("exporter flush error: {e}");
            }
        }
    }

    /// Replays a snapshot of the current bus event log through registered exporters.
    pub async fn export_snapshot(&self) -> Result<(), ObsError> {
        let events: Vec<Event> = self.bus.log().await;
        for event in &events {
            self.dispatch(event).await;
        }
        Ok(())
    }

    async fn dispatch(&self, event: &Event) {
        for exp in &self.exporters {
            if let Err(e) = exp.export(event).await {
                error!("exporter error: {e}");
            }
        }
    }
}
