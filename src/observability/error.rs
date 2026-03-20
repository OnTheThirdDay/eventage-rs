use thiserror::Error;

/// Errors that can occur within the observability pipeline.
#[derive(Debug, Error)]
pub enum ObsError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("Channel closed")]
    ChannelClosed,
    #[error("{0}")]
    Other(String),
}
