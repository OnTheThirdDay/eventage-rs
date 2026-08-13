use super::error::LlmError;
use super::provider::LlmProvider;
use super::types::{ChatMessage, FunctionCall, LlmResponse, ToolCall, ToolDefinition};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
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
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<u32>,
    stop: Vec<String>,
    seed: Option<i64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    reasoning_effort: Option<String>,
    parallel_tool_calls: Option<bool>,
    response_format: Option<serde_json::Value>,
    /// Extra top-level request fields merged into every completion request.
    /// Escape hatch for provider-specific options (`prompt_cache_key`,
    /// `logprobs`, ...).
    extra_body: serde_json::Map<String, serde_json::Value>,
}

impl OpenAiProvider {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        let client = Client::builder()
            // End-to-end timeout: covers connection + reading the full response body.
            // LLM calls can be slow on large contexts, but if the server hangs
            // entirely we must not block the agent loop forever.
            .timeout(Duration::from_secs(180))
            // Separate short timeout for the TCP handshake so a dead host fails fast.
            .connect_timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_default();
        Self {
            client,
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: Vec::new(),
            seed: None,
            frequency_penalty: None,
            presence_penalty: None,
            reasoning_effort: None,
            parallel_tool_calls: None,
            response_format: None,
            extra_body: serde_json::Map::new(),
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

    /// Sets the sampling temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets nucleus sampling `top_p`.
    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Caps completion length (`max_tokens`).
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets stop sequences.
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = stop;
        self
    }

    /// Sets the sampling seed for (best-effort) determinism.
    pub fn with_seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_frequency_penalty(mut self, penalty: f64) -> Self {
        self.frequency_penalty = Some(penalty);
        self
    }

    pub fn with_presence_penalty(mut self, penalty: f64) -> Self {
        self.presence_penalty = Some(penalty);
        self
    }

    /// Reasoning effort for reasoning models: `"minimal"`, `"low"`,
    /// `"medium"`, or `"high"`.
    pub fn with_reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    /// Enable or disable parallel tool calls.
    pub fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    /// Constrain output to a JSON Schema (structured outputs).
    ///
    /// ```no_run
    /// # use eventage::llm::OpenAiProvider;
    /// let provider = OpenAiProvider::openai("sk-...", "gpt-5-mini")
    ///     .with_json_schema("verdict", serde_json::json!({
    ///         "type": "object",
    ///         "properties": { "ok": { "type": "boolean" } },
    ///         "required": ["ok"],
    ///         "additionalProperties": false
    ///     }));
    /// ```
    pub fn with_json_schema(mut self, name: impl Into<String>, schema: serde_json::Value) -> Self {
        self.response_format = Some(serde_json::json!({
            "type": "json_schema",
            "json_schema": { "name": name.into(), "strict": true, "schema": schema }
        }));
        self
    }

    /// Constrain output to syntactically valid JSON (`json_object` mode).
    pub fn with_json_mode(mut self) -> Self {
        self.response_format = Some(serde_json::json!({ "type": "json_object" }));
        self
    }

    /// Adds an arbitrary top-level field to every completion request.
    ///
    /// Escape hatch for provider-specific options without waiting for a
    /// dedicated setter, e.g.:
    ///
    /// ```no_run
    /// # use eventage::llm::OpenAiProvider;
    /// let provider = OpenAiProvider::openai("sk-...", "gpt-5-mini")
    ///     .with_body_param("reasoning_effort", serde_json::json!("high"))
    ///     .with_body_param("prompt_cache_key", serde_json::json!("session-42"));
    /// ```
    pub fn with_body_param(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.extra_body.insert(key.into(), value);
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
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<&'a serde_json::Value>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
struct CompletionResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize, Debug)]
struct Usage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Deserialize, Debug)]
struct PromptTokensDetails {
    cached_tokens: Option<u32>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    message: ChoiceMessage,
    finish_reason: String,
}

#[derive(Deserialize, Debug)]
struct ChoiceMessage {
    content: Option<String>,
    /// Reasoning text from thinking models. Servers disagree on the field
    /// name: DeepSeek/vLLM/Ollama use `reasoning_content`, OpenRouter uses
    /// `reasoning`. Both are captured.
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    tool_calls: Option<Vec<RawToolCall>>,
}

#[derive(Deserialize, Debug)]
struct RawToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: RawFunctionCall,
    /// Preserved verbatim for providers that require round-tripping extra data
    /// (e.g. Gemini's thought_signature).
    extra_content: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
struct RawFunctionCall {
    name: String,
    arguments: String,
}

// ── LlmProvider impl ────────────────────────────────────────────────────────

impl OpenAiProvider {
    /// Build the JSON request body, merging `extra_body` on top of the typed request.
    fn build_body(
        &self,
        messages: &[ChatMessage],
        tools: Vec<ToolDefinition>,
        stream: bool,
    ) -> Result<serde_json::Value, LlmError> {
        // Only include tool_choice when tools are actually being sent.
        let effective_tool_choice = if tools.is_empty() {
            None
        } else {
            self.tool_choice.as_ref()
        };
        let request = CompletionRequest {
            model: &self.model,
            messages,
            tool_choice: effective_tool_choice,
            tools,
            temperature: self.temperature,
            top_p: self.top_p,
            max_tokens: self.max_tokens,
            stop: self.stop.clone(),
            seed: self.seed,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            reasoning_effort: self.reasoning_effort.as_deref(),
            parallel_tool_calls: self.parallel_tool_calls,
            response_format: self.response_format.as_ref(),
            stream,
            // Ask for usage stats in the final SSE chunk (OpenAI-compatible
            // servers that don't support this simply ignore it).
            stream_options: stream.then(|| serde_json::json!({ "include_usage": true })),
        };

        let mut body = serde_json::to_value(&request).map_err(LlmError::Serde)?;
        if !self.extra_body.is_empty() {
            if let Some(obj) = body.as_object_mut() {
                for (k, v) in &self.extra_body {
                    obj.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }
        }
        Ok(body)
    }

    async fn post(&self, body: &serde_json::Value) -> Result<reqwest::Response, LlmError> {
        let url = format!("{}/chat/completions", self.base_url);
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(body)
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
        Ok(resp)
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    #[instrument(skip(self, messages, tools), fields(model = %self.model, messages = messages.len()))]
    async fn complete(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
    ) -> Result<LlmResponse, LlmError> {
        let body = self.build_body(&messages, tools, false)?;

        debug!("sending completion request");

        let resp = self.post(&body).await?;

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
                extra_content: tc.extra_content,
            })
            .collect();

        let (input_tokens, output_tokens, cached_input_tokens) = completion
            .usage
            .map(|u| {
                let cached = u.prompt_tokens_details.and_then(|d| d.cached_tokens);
                (u.prompt_tokens, u.completion_tokens, cached)
            })
            .unwrap_or((None, None, None));

        let reasoning_content = choice
            .message
            .reasoning_content
            .or(choice.message.reasoning)
            .filter(|s| !s.is_empty());

        Ok(LlmResponse {
            content: choice.message.content,
            reasoning_content,
            tool_calls,
            finish_reason: choice.finish_reason,
            input_tokens,
            output_tokens,
            cached_input_tokens,
            provider_extra: None,
        })
    }

    #[instrument(skip(self, messages, tools, on_delta), fields(model = %self.model, messages = messages.len()))]
    async fn complete_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
        on_delta: super::types::DeltaHandler,
    ) -> Result<LlmResponse, LlmError> {
        use futures_util::StreamExt;

        let body = self.build_body(&messages, tools, true)?;

        debug!("opening SSE completion stream");

        let resp = self.post(&body).await?;

        let mut content = String::new();
        let mut reasoning = String::new();
        let mut tool_acc: std::collections::BTreeMap<usize, ToolCallAcc> =
            std::collections::BTreeMap::new();
        let mut finish_reason = String::from("stop");
        let mut usage: Option<Usage> = None;
        let mut line_buf = String::new();

        let mut byte_stream = resp.bytes_stream();
        'stream: while let Some(chunk_result) = byte_stream.next().await {
            let bytes = chunk_result.map_err(LlmError::Http)?;
            line_buf.push_str(&String::from_utf8_lossy(&bytes));

            while let Some(newline_pos) = line_buf.find('\n') {
                let line = line_buf[..newline_pos].trim_end_matches('\r').to_string();
                line_buf.drain(..=newline_pos);

                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };
                if data == "[DONE]" {
                    break 'stream;
                }

                let chunk: StreamChunk = match serde_json::from_str(data) {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!("failed to parse SSE chunk: {e}");
                        continue;
                    }
                };

                if let Some(u) = chunk.usage {
                    usage = Some(u);
                }
                let Some(choice) = chunk.choices.into_iter().next() else {
                    continue;
                };
                if let Some(reason) = choice.finish_reason {
                    if !reason.is_empty() {
                        finish_reason = reason;
                    }
                }

                let delta = choice.delta;
                let delta_reasoning = delta.reasoning_content.or(delta.reasoning);
                if delta.content.is_some() || delta_reasoning.is_some() {
                    if let Some(text) = &delta.content {
                        content.push_str(text);
                    }
                    if let Some(text) = &delta_reasoning {
                        reasoning.push_str(text);
                    }
                    on_delta(super::types::StreamDelta {
                        content: delta.content,
                        reasoning_content: delta_reasoning,
                    });
                }

                for d in delta.tool_calls.unwrap_or_default() {
                    let acc = tool_acc.entry(d.index).or_default();
                    if let Some(id) = d.id {
                        acc.id = id;
                    }
                    if let Some(kind) = d.kind {
                        acc.kind = kind;
                    }
                    if let Some(func) = d.function {
                        if let Some(name) = func.name {
                            acc.name.push_str(&name);
                        }
                        if let Some(args) = func.arguments {
                            acc.arguments.push_str(&args);
                        }
                    }
                }
            }
        }

        let tool_calls: Vec<ToolCall> = tool_acc
            .into_values()
            .map(|acc| ToolCall {
                id: acc.id,
                kind: if acc.kind.is_empty() {
                    "function".to_string()
                } else {
                    acc.kind
                },
                function: FunctionCall {
                    name: acc.name,
                    arguments: acc.arguments,
                },
                extra_content: None,
            })
            .collect();

        let (input_tokens, output_tokens, cached_input_tokens) = usage
            .map(|u| {
                let cached = u.prompt_tokens_details.and_then(|d| d.cached_tokens);
                (u.prompt_tokens, u.completion_tokens, cached)
            })
            .unwrap_or((None, None, None));

        Ok(LlmResponse {
            content: (!content.is_empty()).then_some(content),
            reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
            tool_calls,
            finish_reason,
            input_tokens,
            output_tokens,
            cached_input_tokens,
            provider_extra: None,
        })
    }

    fn model(&self) -> &str {
        &self.model
    }
}

// ── SSE stream serde structs ─────────────────────────────────────────────────

#[derive(Deserialize, Debug)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize, Debug)]
struct StreamChoice {
    delta: StreamDeltaRaw,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
struct StreamDeltaRaw {
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Deserialize, Debug)]
struct ToolCallDelta {
    #[serde(default)]
    index: usize,
    id: Option<String>,
    #[serde(rename = "type")]
    kind: Option<String>,
    function: Option<FunctionDelta>,
}

#[derive(Deserialize, Debug)]
struct FunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Default)]
struct ToolCallAcc {
    id: String,
    kind: String,
    name: String,
    arguments: String,
}
