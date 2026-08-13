use super::content::ContentPart;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A single message in the chat history.
///
/// Content is either plain [`content`](Self::content) text or, for multimodal
/// messages, an ordered list of [`parts`](Self::parts). When `parts` is
/// non-empty it takes precedence: serialization emits the OpenAI-style
/// content array, and native providers map the parts to their own block
/// formats.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: Option<String>,
    /// Multimodal content parts (text + images). Empty for text-only messages.
    pub parts: Vec<ContentPart>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
    /// Provider-opaque blocks round-tripped through the event log so
    /// stateful reasoning survives multi-step tool loops — e.g. Anthropic
    /// thinking blocks (with signatures) or OpenAI Responses reasoning items
    /// (with encrypted content). Never serialized onto the OpenAI-compatible
    /// wire; native providers read it explicitly.
    pub provider_extra: Option<serde_json::Value>,
}

// Serialization targets the OpenAI Chat Completions wire format: `content` is
// a bare string for text-only messages and an array of typed parts for
// multimodal ones. `provider_extra` is harness-internal and never sent.
impl Serialize for ChatMessage {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let has_content = self.content.is_some() || !self.parts.is_empty();
        let field_count = 1
            + usize::from(has_content)
            + usize::from(self.tool_calls.is_some())
            + usize::from(self.tool_call_id.is_some())
            + usize::from(self.name.is_some());

        let mut state = serializer.serialize_struct("ChatMessage", field_count)?;
        state.serialize_field("role", &self.role)?;
        if !self.parts.is_empty() {
            let parts: Vec<serde_json::Value> =
                self.parts.iter().map(|p| p.to_openai_json()).collect();
            state.serialize_field("content", &parts)?;
        } else if let Some(text) = &self.content {
            state.serialize_field("content", text)?;
        }
        if let Some(tool_calls) = &self.tool_calls {
            state.serialize_field("tool_calls", tool_calls)?;
        }
        if let Some(id) = &self.tool_call_id {
            state.serialize_field("tool_call_id", id)?;
        }
        if let Some(name) = &self.name {
            state.serialize_field("name", name)?;
        }
        state.end()
    }
}

/// Accepts `content` as either a string or an array of content parts.
#[derive(Deserialize)]
#[serde(untagged)]
enum ContentField {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Deserialize)]
struct ChatMessageRepr {
    role: Role,
    #[serde(default)]
    content: Option<ContentField>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    provider_extra: Option<serde_json::Value>,
}

impl<'de> Deserialize<'de> for ChatMessage {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let repr = ChatMessageRepr::deserialize(deserializer)?;
        let (content, parts) = match repr.content {
            Some(ContentField::Text(text)) => (Some(text), Vec::new()),
            Some(ContentField::Parts(parts)) => {
                let text = super::content::parts_to_text(&parts);
                ((!text.is_empty()).then_some(text), parts)
            }
            None => (None, Vec::new()),
        };
        Ok(ChatMessage {
            role: repr.role,
            content,
            parts,
            tool_calls: repr.tool_calls,
            tool_call_id: repr.tool_call_id,
            name: repr.name,
            provider_extra: repr.provider_extra,
        })
    }
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            parts: Vec::new(),
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
            parts: Vec::new(),
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
            parts: Vec::new(),
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
            parts: Vec::new(),
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
            parts: Vec::new(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            name: None,
            provider_extra: None,
        }
    }

    /// A user message with multimodal content (text and/or images).
    ///
    /// ```
    /// use eventage::llm::{ChatMessage, ContentPart};
    ///
    /// let msg = ChatMessage::user_with_parts(vec![
    ///     ContentPart::text("Describe this:"),
    ///     ContentPart::image_url("https://example.com/a.png"),
    /// ]);
    /// ```
    pub fn user_with_parts(parts: Vec<ContentPart>) -> Self {
        let text = super::content::parts_to_text(&parts);
        Self {
            role: Role::User,
            parts,
            content: (!text.is_empty()).then_some(text),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            provider_extra: None,
        }
    }

    /// Replace this message's content with multimodal parts.
    pub fn with_parts(mut self, parts: Vec<ContentPart>) -> Self {
        let text = super::content::parts_to_text(&parts);
        self.content = (!text.is_empty()).then_some(text);
        self.parts = parts;
        self
    }

    /// `true` if this message carries any non-text content.
    pub fn is_multimodal(&self) -> bool {
        self.parts
            .iter()
            .any(|p| !matches!(p, ContentPart::Text { .. }))
    }

    /// The message's content as content parts — synthesizing a single text
    /// part for plain-text messages. Empty when there is no content.
    pub fn content_parts(&self) -> Vec<ContentPart> {
        if !self.parts.is_empty() {
            return self.parts.clone();
        }
        match self.content.as_deref() {
            Some(text) if !text.is_empty() => vec![ContentPart::text(text)],
            _ => Vec::new(),
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
