//! Qwen provider — the Responses API as served by Alibaba Cloud's
//! `compatible-mode` gateway (Model Studio / MaaS).
//!
//! Qwen speaks the Responses *shape* but not its OpenAI-only extensions, and
//! its event stream differs in ways that silently break a strict client. This
//! lives in its own provider rather than as conditionals inside
//! [`OpenAiResponsesProvider`](super::OpenAiResponsesProvider), so neither
//! dialect distorts the other:
//!
//! | Difference | Handling |
//! |---|---|
//! | `store` / `include` / `reasoning_effort` unsupported | never sent |
//! | reasoning streams as `response.reasoning_text.delta` | mapped to reasoning deltas |
//! | no terminal `response.completed` event | response rebuilt from `output_item.done` |
//! | no `encrypted_content` on reasoning items | reasoning is not replayed across steps |
//!
//! Because reasoning cannot be round-tripped, tool loops rely on the visible
//! transcript alone — correct here, since Qwen does not require prior
//! reasoning to be echoed back.
//!
//! ```no_run
//! use eventage::llm::QwenProvider;
//!
//! let llm = QwenProvider::new(
//!     std::env::var("QWEN_API_KEY").unwrap(),
//!     "qwen3.7-max",
//! )
//! .with_base_url("https://<your-endpoint>/compatible-mode/v1");
//! ```

use super::error::LlmError;
use super::provider::LlmProvider;
use super::responses::{convert_messages, parse_response};
use super::types::{ChatMessage, DeltaHandler, LlmResponse, StreamDelta, ToolDefinition};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Map, Value};
use std::time::Duration;
use tracing::{debug, instrument, warn};

/// Alibaba Cloud's default public Model Studio endpoint.
const DEFAULT_BASE_URL: &str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1";

/// Qwen (Responses-compatible) provider. See the [module docs](self).
pub struct QwenProvider {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_output_tokens: Option<u32>,
    parallel_tool_calls: Option<bool>,
    tool_choice: Option<Value>,
    /// Qwen exposes thinking via `enable_thinking`, not `reasoning.effort`.
    enable_thinking: Option<bool>,
    extra_body: Map<String, Value>,
}

impl QwenProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(600))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_default();
        Self {
            client,
            base_url: DEFAULT_BASE_URL.into(),
            api_key: api_key.into(),
            model: model.into(),
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            parallel_tool_calls: None,
            tool_choice: None,
            enable_thinking: None,
            extra_body: Map::new(),
        }
    }

    /// Point at a specific gateway (regional MaaS endpoints differ per
    /// deployment). Must end at `/v1`; `/responses` is appended.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
        self
    }

    pub fn with_top_p(mut self, p: f64) -> Self {
        self.top_p = Some(p);
        self
    }

    pub fn with_max_output_tokens(mut self, n: u32) -> Self {
        self.max_output_tokens = Some(n);
        self
    }

    /// Toggle Qwen's thinking mode.
    pub fn with_thinking(mut self, enabled: bool) -> Self {
        self.enable_thinking = Some(enabled);
        self
    }

    pub fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    pub fn with_tool_choice(mut self, choice: Value) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Merge an arbitrary top-level field into every request.
    pub fn with_body_param(mut self, key: impl Into<String>, value: Value) -> Self {
        self.extra_body.insert(key.into(), value);
        self
    }

    fn build_body(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        stream: bool,
    ) -> Value {
        let (instructions, input) = convert_messages(messages);

        // Deliberately omits `store` and `include`: the gateway rejects or
        // ignores them, and there is no encrypted reasoning to preserve.
        let mut body = json!({ "model": self.model, "input": input });
        let obj = body.as_object_mut().expect("body is an object");

        if let Some(instructions) = instructions {
            obj.insert("instructions".into(), json!(instructions));
        }
        if !tools.is_empty() {
            let defs: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "name": t.function.name,
                        "description": t.function.description,
                        "parameters": t.function.parameters,
                    })
                })
                .collect();
            obj.insert("tools".into(), Value::Array(defs));
            if let Some(choice) = &self.tool_choice {
                obj.insert("tool_choice".into(), choice.clone());
            }
            if let Some(parallel) = self.parallel_tool_calls {
                obj.insert("parallel_tool_calls".into(), json!(parallel));
            }
        }
        if let Some(enabled) = self.enable_thinking {
            obj.insert("enable_thinking".into(), json!(enabled));
        }
        if let Some(t) = self.temperature {
            obj.insert("temperature".into(), json!(t));
        }
        if let Some(p) = self.top_p {
            obj.insert("top_p".into(), json!(p));
        }
        if let Some(n) = self.max_output_tokens {
            obj.insert("max_output_tokens".into(), json!(n));
        }
        if stream {
            obj.insert("stream".into(), json!(true));
        }
        for (k, v) in &self.extra_body {
            obj.entry(k.clone()).or_insert_with(|| v.clone());
        }
        body
    }

    async fn post(&self, body: &Value) -> Result<reqwest::Response, LlmError> {
        let resp = self
            .client
            .post(format!("{}/responses", self.base_url))
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
impl LlmProvider for QwenProvider {
    #[instrument(skip(self, messages, tools), fields(model = %self.model, messages = messages.len()))]
    async fn complete(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
    ) -> Result<LlmResponse, LlmError> {
        let body = self.build_body(&messages, &tools, false);
        debug!("sending qwen responses request");
        let resp = self.post(&body).await?;
        let parsed: Value = resp.json().await.map_err(LlmError::Http)?;
        Ok(parse_response(&parsed))
    }

    #[instrument(skip(self, messages, tools, on_delta), fields(model = %self.model, messages = messages.len()))]
    async fn complete_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
        on_delta: DeltaHandler,
    ) -> Result<LlmResponse, LlmError> {
        use futures_util::StreamExt;

        let body = self.build_body(&messages, &tools, true);
        debug!("opening qwen SSE stream");
        let resp = self.post(&body).await?;

        // The gateway ends the stream without `response.completed`, so the
        // finished output items are the source of truth.
        let mut items: Vec<Value> = Vec::new();
        let mut final_response: Option<Value> = None;
        let mut usage: Option<Value> = None;
        let mut line_buf = String::new();
        let mut byte_stream = resp.bytes_stream();

        while let Some(chunk) = byte_stream.next().await {
            let bytes = chunk.map_err(LlmError::Http)?;
            line_buf.push_str(&String::from_utf8_lossy(&bytes));

            while let Some(newline) = line_buf.find('\n') {
                let line = line_buf[..newline].trim_end_matches('\r').to_string();
                line_buf.drain(..=newline);
                let Some(data) = super::sse_data(&line) else {
                    continue;
                };
                if data == "[DONE]" {
                    continue;
                }
                let event: Value = match serde_json::from_str(data) {
                    Ok(v) => v,
                    Err(e) => {
                        warn!("failed to parse qwen SSE chunk: {e}");
                        continue;
                    }
                };

                match event.get("type").and_then(|t| t.as_str()) {
                    Some("response.output_text.delta") => {
                        if let Some(text) = event.get("delta").and_then(|d| d.as_str()) {
                            on_delta(StreamDelta {
                                content: Some(text.to_string()),
                                reasoning_content: None,
                            });
                        }
                    }
                    // Qwen streams thinking as `reasoning_text`.
                    Some("response.reasoning_text.delta") => {
                        if let Some(text) = event.get("delta").and_then(|d| d.as_str()) {
                            on_delta(StreamDelta {
                                content: None,
                                reasoning_content: Some(text.to_string()),
                            });
                        }
                    }
                    Some("response.output_item.done") => {
                        if let Some(item) = event.get("item") {
                            items.push(item.clone());
                        }
                    }
                    Some("response.completed") | Some("response.incomplete") => {
                        final_response = event.get("response").cloned();
                    }
                    Some("error") | Some("response.failed") => {
                        return Err(LlmError::Api {
                            status: 500,
                            body: event.to_string(),
                        });
                    }
                    _ => {
                        if let Some(u) = event.get("usage") {
                            usage = Some(u.clone());
                        }
                    }
                }
            }
        }

        let parsed = match final_response {
            Some(response) => response,
            None if !items.is_empty() => {
                let mut synthetic = json!({ "output": items });
                if let (Some(usage), Some(obj)) = (usage, synthetic.as_object_mut()) {
                    obj.insert("usage".into(), usage);
                }
                synthetic
            }
            None => return Err(LlmError::EmptyResponse),
        };
        Ok(parse_response(&parsed))
    }

    fn model(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider() -> QwenProvider {
        QwenProvider::new("key", "qwen3.7-max")
    }

    #[test]
    fn omits_openai_only_request_fields() {
        let body = provider().build_body(&[ChatMessage::user("hi")], &[], false);
        // These are the fields the gateway does not accept.
        assert!(body.get("store").is_none());
        assert!(body.get("include").is_none());
        assert!(body.get("reasoning").is_none());
        assert_eq!(body["model"], "qwen3.7-max");
        assert_eq!(body["input"][0]["role"], "user");
    }

    #[test]
    fn thinking_uses_qwen_flag() {
        let body =
            provider()
                .with_thinking(true)
                .build_body(&[ChatMessage::user("hi")], &[], false);
        assert_eq!(body["enable_thinking"], true);
        assert!(body.get("reasoning").is_none());
    }

    #[test]
    fn tools_use_the_flat_responses_shape() {
        let tool = ToolDefinition::function(
            "read_file",
            "Read a file",
            json!({ "type": "object", "properties": {} }),
        );
        let body = provider().build_body(&[ChatMessage::user("hi")], &[tool], false);
        assert_eq!(body["tools"][0]["type"], "function");
        // Flat `name`, not nested under `function`.
        assert_eq!(body["tools"][0]["name"], "read_file");
    }

    #[test]
    fn parses_a_response_without_a_terminal_event() {
        // Exactly what the gateway streams: items, no `response.completed`.
        let synthetic = json!({
            "output": [
                { "type": "reasoning", "summary": [{ "text": "thinking" }] },
                { "type": "message", "content": [{ "type": "output_text", "text": "OK" }] }
            ]
        });
        let resp = parse_response(&synthetic);
        assert_eq!(resp.content.as_deref(), Some("OK"));
        assert_eq!(resp.reasoning_content.as_deref(), Some("thinking"));
    }
}
