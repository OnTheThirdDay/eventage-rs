use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A single message in the chat history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Provider-opaque blocks round-tripped through the event log so
    /// stateful reasoning survives multi-step tool loops — e.g. Anthropic
    /// thinking blocks (with signatures) or OpenAI Responses reasoning items
    /// (with encrypted content). Never serialized onto the OpenAI-compatible
    /// wire; native providers read it explicitly.
    #[serde(skip_serializing, default)]
    pub provider_extra: Option<serde_json::Value>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            provider_extra: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            provider_extra: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            provider_extra: None,
        }
    }

    pub fn assistant_with_tool_calls(content: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            name: None,
            provider_extra: None,
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            name: None,
            provider_extra: None,
        }
    }

    /// Set the `name` field to identify the message sender.
    ///
    /// Used for `user`-role messages to distinguish between human input,
    /// agent delegations (`agent_<group>`), async replies (`agent_reply_<group>`),
    /// and scheduled tasks (`scheduler`).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Attach provider-opaque blocks (see [`ChatMessage::provider_extra`]).
    pub fn with_provider_extra(mut self, extra: serde_json::Value) -> Self {
        self.provider_extra = Some(extra);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
    /// Provider-specific extra fields (e.g. Gemini's `extra_content` containing
    /// `thought_signature`). Serialized as-is when present so providers that
    /// require round-tripping custom metadata (like thought signatures) work
    /// correctly without any provider-specific logic elsewhere in the stack.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_content: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// JSON-encoded arguments string.
    pub arguments: String,
}

/// Tool definition exposing a function to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema for parameters.
    pub parameters: serde_json::Value,
}

impl ToolDefinition {
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            kind: "function".to_string(),
            function: FunctionDefinition {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

/// An incremental piece of a streaming completion.
///
/// Delivered to the `on_delta` callback of
/// [`LlmProvider::complete_stream`](crate::llm::LlmProvider::complete_stream)
/// as tokens arrive. The full response is still returned at the end of the
/// stream as a normal [`LlmResponse`].
#[derive(Debug, Clone, Default)]
pub struct StreamDelta {
    /// New completion text, if this chunk carried any.
    pub content: Option<String>,
    /// New reasoning/thinking text, if this chunk carried any.
    pub reasoning_content: Option<String>,
}

/// Callback invoked for each [`StreamDelta`] during a streaming completion.
pub type DeltaHandler = std::sync::Arc<dyn Fn(StreamDelta) + Send + Sync>;

/// The parsed response from an LLM completion.
#[derive(Debug, Clone, Default)]
pub struct LlmResponse {
    pub content: Option<String>,
    /// Chain-of-thought / reasoning text emitted by thinking models
    /// (`reasoning_content` or `reasoning` in OpenAI-compatible responses).
    ///
    /// Preserved on the event bus for observability and replay, but not fed
    /// back into subsequent completion requests (most providers reject or
    /// ignore replayed reasoning).
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: String,
    /// Prompt tokens consumed by this completion, when reported by the provider.
    pub input_tokens: Option<u32>,
    /// Completion tokens generated by this completion, when reported by the provider.
    pub output_tokens: Option<u32>,
    /// Prompt tokens served from the provider's prompt cache, when reported
    /// (`usage.prompt_tokens_details.cached_tokens` for OpenAI,
    /// `usage.cache_read_input_tokens` for Anthropic). Lets harnesses track
    /// cache hit-rate and cost without provider-specific plumbing.
    pub cached_input_tokens: Option<u32>,
    /// Provider-opaque blocks that must be replayed on the next request for
    /// stateful reasoning (Anthropic thinking blocks, OpenAI Responses
    /// reasoning items). The strategy stores this on the `assistant.message`
    /// event and the context assembler restores it into
    /// [`ChatMessage::provider_extra`].
    pub provider_extra: Option<serde_json::Value>,
}
