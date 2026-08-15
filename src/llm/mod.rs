pub mod anthropic;
pub mod content;
pub mod error;
pub mod mock;
pub mod openai;
pub mod provider;
pub mod rate_limit;
pub mod qwen;
pub mod responses;
pub mod retry;
pub mod structured;
pub mod types;

/// Extract the payload of an SSE `data:` line.
///
/// The single space after the colon is **optional** in the SSE spec and real
/// gateways differ (OpenAI sends `data: {...}`, some Aliyun/vLLM endpoints
/// send `data:{...}`), so requiring it silently drops every event.
pub(crate) fn sse_data(line: &str) -> Option<&str> {
    line.strip_prefix("data:")
        .map(|d| d.strip_prefix(' ').unwrap_or(d))
}

pub use anthropic::AnthropicProvider;
pub use content::{parts_to_text, ContentPart, ImageSource};
pub use error::LlmError;
pub use mock::MockLlmProvider;
pub use openai::OpenAiProvider;
pub use provider::LlmProvider;
pub use rate_limit::RateLimitedProvider;
pub use qwen::QwenProvider;
pub use responses::OpenAiResponsesProvider;
pub use retry::RetryProvider;
pub use structured::{extract_json, StructuredExt};
pub use types::{
    ChatMessage, FunctionCall, FunctionDefinition, LlmResponse, Role, ToolCall, ToolDefinition,
};

#[cfg(test)]
mod sse_tests {
    use super::sse_data;

    #[test]
    fn accepts_both_spacings() {
        assert_eq!(sse_data("data: {\"a\":1}"), Some("{\"a\":1}"));
        assert_eq!(sse_data("data:{\"a\":1}"), Some("{\"a\":1}"));
        assert_eq!(sse_data("data:[DONE]"), Some("[DONE]"));
        assert_eq!(sse_data("event:response.created"), None);
        assert_eq!(sse_data(":comment"), None);
    }
}
