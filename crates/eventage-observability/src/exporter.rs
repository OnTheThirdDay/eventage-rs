use crate::error::ObsError;
use async_trait::async_trait;
use eventage_core::Event;

/// A destination for events emitted by a [`BusObserver`].
///
/// Implement this to forward events to custom backends (e.g., databases, tracing systems).
///
/// # Thread Safety
/// Must be `Send + Sync` as methods are called concurrently from the async observer task.
#[async_trait]
pub trait ObservabilityExporter: Send + Sync {
    /// Persists or forwards a single event observed on the bus.
    async fn export(&self, event: &Event) -> Result<(), ObsError>;

    /// Flushes any internal buffers. Called cleanly when the observer shuts down.
    async fn flush(&self) -> Result<(), ObsError> {
        Ok(())
    }
}
