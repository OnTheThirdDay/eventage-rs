use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("Bus error: {0}")]
    Bus(#[from] crate::error::BusError),
    #[error("LLM error: {0}")]
    Llm(#[from] crate::llm::LlmError),
    #[error("Tool error: {0}")]
    Tool(String),
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("React loop exceeded maximum steps ({0})")]
    MaxStepsReached(usize),
    #[error("tool '{name}' timed out after {secs}s")]
    ToolTimeout { name: String, secs: u64 },
}
