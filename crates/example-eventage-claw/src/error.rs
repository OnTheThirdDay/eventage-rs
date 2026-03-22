//! Unified error type for eventage-claw.

#[derive(Debug, thiserror::Error)]
pub enum ClawError {
    #[error("agent error: {0}")]
    Agent(#[from] eventage::AgentError),

    #[error("worker error: {0}")]
    Worker(#[from] eventage::agent::worker::WorkerError),

    #[error("bus error: {0}")]
    Bus(#[from] eventage::BusError),

    #[allow(dead_code)]
    #[error("config error: {0}")]
    Config(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("tool error: {0}")]
    Tool(String),
}
