use crate::error::LlmError;
use crate::provider::LlmProvider;
use crate::types::{ChatMessage, FunctionCall, LlmResponse, ToolCall, ToolDefinition};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

/// OpenAI-compatible provider (works with Ollama, Groq, Mistral, Azure, etc.).
pub struct OpenAiProvider {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
    /// Optional `tool_choice` sent when tools are present.
    /// Values: "auto", "none", "required", or specific function `{"type":"function","function":{"name":"..."}}`.
    /// Omitted if `None`.
    tool_choice: Option<serde_json::Value>,
}

impl OpenAiProvider {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
            tool_choice: None,
        }
    }

    /// Convenience constructor for a local Ollama instance.
    pub fn ollama(model: impl Into<String>) -> Self {
        Self::new("http://localhost:11434/v1", "ollama", model)
    }

    /// Convenience constructor for OpenAI.
    pub fn openai(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new("https://api.openai.com/v1", api_key, model)
    }

    /// Sets the `tool_choice` field for requests including tools.
    pub fn with_tool_choice(mut self, choice: serde_json::Value) -> Self {
        self.tool_choice = Some(choice);
        self
    }
}

// ── request / response serde structs ────────────────────────────────────────

#[derive(Serialize)]
struct CompletionRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDefinition>,
    /// Omitted when `None` or no tools are sent.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a serde_json::Value>,
}

#[derive(Deserialize, Debug)]
struct CompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    message: ChoiceMessage,
    finish_reason: String,
}

#[derive(Deserialize, Debug)]
struct ChoiceMessage {
    content: Option<String>,
    tool_calls: Option<Vec<RawToolCall>>,
}

#[derive(Deserialize, Debug)]
struct RawToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: RawFunctionCall,
}

#[derive(Deserialize, Debug)]
struct RawFunctionCall {
    name: String,
    arguments: String,
}

// ── LlmProvider impl ────────────────────────────────────────────────────────

#[async_trait]
impl LlmProvider for OpenAiProvider {
    #[instrument(skip(self, messages, tools), fields(model = %self.model, messages = messages.len()))]
    async fn complete(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
    ) -> Result<LlmResponse, LlmError> {
        let url = format!("{}/chat/completions", self.base_url);
        // Only include tool_choice when tools are actually being sent.
        let effective_tool_choice = if tools.is_empty() {
            None
        } else {
            self.tool_choice.as_ref()
        };
        let request = CompletionRequest {
            model: &self.model,
            messages: &messages,
            tool_choice: effective_tool_choice,
            tools,
        };

        debug!("sending completion request");

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .map_err(LlmError::Http)?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api {
                status: status.as_u16(),
                body,
            });
        }

        let completion: CompletionResponse = resp.json().await.map_err(LlmError::Http)?;

        debug!("completion response: {:?}", completion);

        let choice = completion
            .choices
            .into_iter()
            .next()
            .ok_or(LlmError::EmptyResponse)?;

        let tool_calls = choice
            .message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|tc| ToolCall {
                id: tc.id,
                kind: tc.kind,
                function: FunctionCall {
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                },
            })
            .collect();

        Ok(LlmResponse {
            content: choice.message.content,
            tool_calls,
            finish_reason: choice.finish_reason,
        })
    }

    fn model(&self) -> &str {
        &self.model
    }
}
