use super::error::LlmError;
use super::types::{ChatMessage, DeltaHandler, LlmResponse, ToolDefinition};
use async_trait::async_trait;

/// Unified interface for chat completion providers.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
    ) -> Result<LlmResponse, LlmError>;

    /// Streaming completion: `on_delta` fires for each incremental chunk and
    /// the assembled final response is returned at the end of the stream.
    ///
    /// The default implementation performs a regular [`complete`](Self::complete)
    /// and emits the whole answer as a single delta, so wrapper providers
    /// (retry, rate-limit, mocks) and non-streaming backends work unchanged.
    async fn complete_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
        on_delta: DeltaHandler,
    ) -> Result<LlmResponse, LlmError> {
        let response = self.complete(messages, tools).await?;
        if response.content.is_some() || response.reasoning_content.is_some() {
            on_delta(super::types::StreamDelta {
                content: response.content.clone(),
                reasoning_content: response.reasoning_content.clone(),
            });
        }
        Ok(response)
    }

    fn model(&self) -> &str;
}
