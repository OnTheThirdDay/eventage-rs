use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodingAgentError {
    #[error("agent error: {0}")]
    Agent(#[from] eventage::AgentError),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("tool error: {0}")]
    Tool(String),

    #[error("worker error: {0}")]
    Worker(#[from] eventage::WorkerError),
}
