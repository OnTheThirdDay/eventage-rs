use crate::event::EventId;
use thiserror::Error;

/// Errors returned by [`EventBus`](crate::EventBus) operations.
#[derive(Debug, Error)]
pub enum BusError {
    /// The underlying event channel was dropped or closed.
    #[error("Event bus channel closed")]
    ChannelClosed,
    /// Rollback failed: the specified checkpoint is not in the active branch history.
    #[error("Checkpoint event {0} not found in the active branch")]
    CheckpointNotFound(EventId),
}

/// Top-level errors for the Eventage core crate.
#[derive(Debug, Error)]
pub enum CoreError {
    /// A bus-related error occurred.
    #[error("Bus error: {0}")]
    Bus(#[from] BusError),
    /// Failed to serialize or deserialize event payload or metadata.
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}
