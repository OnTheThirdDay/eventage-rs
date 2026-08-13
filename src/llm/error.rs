use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("API error {status}: {body}")]
    Api { status: u16, body: String },
    #[error("Empty response from LLM")]
    EmptyResponse,
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("Structured output error: {0}")]
    Structured(String),
}
