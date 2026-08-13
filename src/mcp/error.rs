use thiserror::Error;

#[derive(Debug, Error)]
pub enum McpError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("MCP protocol error (code={code}): {message}")]
    Protocol { code: i64, message: String },

    #[error("tool execution error: {0}")]
    Tool(String),

    #[error("no result returned by tool")]
    NoResult,

    #[error("transport error: {0}")]
    Transport(String),
}
