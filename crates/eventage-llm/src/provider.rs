use crate::error::LlmError;
use crate::types::{ChatMessage, LlmResponse, ToolDefinition};
use async_trait::async_trait;

/// Unified interface for chat completion providers.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
    ) -> Result<LlmResponse, LlmError>;

    fn model(&self) -> &str;
}
