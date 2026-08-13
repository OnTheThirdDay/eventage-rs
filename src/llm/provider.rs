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

    /// Complete with the output constrained to a JSON Schema.
    ///
    /// The default implementation is provider-agnostic: it appends schema
    /// instructions to the prompt and extracts JSON from the reply, so every
    /// provider (including local models) supports structured output. Native
    /// providers override this to constrain decoding server-side.
    ///
    /// Prefer the typed [`complete_as`](super::StructuredExt::complete_as)
    /// wrapper, which also validates the result against the schema.
    async fn complete_structured(
        &self,
        mut messages: Vec<ChatMessage>,
        schema_name: &str,
        schema: serde_json::Value,
    ) -> Result<serde_json::Value, LlmError> {
        messages.push(super::structured::json_instruction(schema_name, &schema));
        let response = self.complete(messages, vec![]).await?;
        let text = response.content.unwrap_or_default();
        super::structured::extract_json(&text).ok_or_else(|| {
            LlmError::Structured(format!(
                "model did not return JSON for schema '{schema_name}': {}",
                text.chars().take(200).collect::<String>()
            ))
        })
    }

    fn model(&self) -> &str;
}
