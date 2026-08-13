//! OpenAI **Responses API** provider — the native API for reasoning models.
//!
//! Where [`OpenAiProvider`](super::OpenAiProvider) speaks Chat Completions
//! (the universal compatibility surface), this provider speaks `/v1/responses`
//! for first-class reasoning support:
//!
//! - **Reasoning round-trip** — raw output items (including
//!   `reasoning` items with `encrypted_content`) are captured into
//!   [`LlmResponse::provider_extra`], persisted on the event bus, and
//!   replayed verbatim on the next request, so the model keeps its chain of
//!   thought across tool-loop steps without server-side state (`store: false`).
//! - **Reasoning effort** — [`with_reasoning_effort`](OpenAiResponsesProvider::with_reasoning_effort).
//! - **Streaming** — `response.output_text.delta` / reasoning-summary deltas.
//!
//! ```no_run
//! use eventage::llm::OpenAiResponsesProvider;
//!
//! let llm = OpenAiResponsesProvider::new(std::env::var("OPENAI_API_KEY").unwrap(), "gpt-5")
//!     .with_reasoning_effort("high");
//! ```

use super::error::LlmError;
use super::provider::LlmProvider;
use super::types::{
    ChatMessage, DeltaHandler, FunctionCall, LlmResponse, Role, StreamDelta, ToolCall,
    ToolDefinition,
};
use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Map, Value};
use std::time::Duration;
use tracing::{debug, instrument, warn};

/// Key under which raw Responses output items travel in `provider_extra`.
const ITEMS_KEY: &str = "openai_response_items";

/// Native OpenAI Responses API provider. See the [module docs](self).
pub struct OpenAiResponsesProvider {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_output_tokens: Option<u32>,
    reasoning_effort: Option<String>,
    parallel_tool_calls: Option<bool>,
    tool_choice: Option<Value>,
    extra_body: Map<String, Value>,
}

impl OpenAiResponsesProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(600))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_default();
        Self {
            client,
            base_url: "https://api.openai.com/v1".into(),
            api_key: api_key.into(),
            model: model.into(),
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            reasoning_effort: None,
            parallel_tool_calls: None,
            tool_choice: None,
            extra_body: Map::new(),
        }
    }

    /// Point at a different endpoint (proxy, gateway, Azure).
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

    /// Reasoning effort: `"minimal"`, `"low"`, `"medium"`, or `"high"`.
    pub fn with_reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    pub fn with_parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    /// Force tool choice, e.g. `json!("required")` or
    /// `json!({"type": "function", "name": "my_tool"})`.
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

        let mut body = json!({
            "model": self.model,
            "input": input,
            // Stateless operation: context is reassembled from the event log
            // every step; encrypted reasoning items carry the state instead.
            "store": false,
            "include": ["reasoning.encrypted_content"],
        });
        let obj = body.as_object_mut().expect("body is an object");

        if let Some(instructions) = instructions {
            obj.insert("instructions".into(), json!(instructions));
        }
        if !tools.is_empty() {
            // Responses API uses a flat function-tool shape.
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
        if let Some(effort) = &self.reasoning_effort {
            obj.insert("reasoning".into(), json!({ "effort": effort }));
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
        let url = format!("{}/responses", self.base_url);
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

// ── Conversion ────────────────────────────────────────────────────────────────

/// Convert harness messages to `(instructions, input items)`.
///
/// Assistant turns that carry raw Responses output items in `provider_extra`
/// are replayed **verbatim** (reasoning + message + function_call items in
/// their original order) — this is what keeps encrypted reasoning correctly
/// paired with its function calls. Turns without stored items are derived
/// from `content` / `tool_calls`.
fn convert_messages(messages: &[ChatMessage]) -> (Option<String>, Vec<Value>) {
    let mut instructions = String::new();
    let mut in_leading_system = true;
    let mut input: Vec<Value> = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                let text = msg.content.as_deref().unwrap_or_default();
                if in_leading_system {
                    if !instructions.is_empty() {
                        instructions.push_str("\n\n");
                    }
                    instructions.push_str(text);
                } else {
                    input.push(json!({ "role": "user", "content": format!("[system]\n{text}") }));
                }
            }
            Role::User => {
                in_leading_system = false;
                let text = msg.content.as_deref().unwrap_or_default();
                let text = match &msg.name {
                    Some(name) => format!("[{name}] {text}"),
                    None => text.to_string(),
                };
                input.push(json!({ "role": "user", "content": text }));
            }
            Role::Assistant => {
                in_leading_system = false;
                if let Some(items) = msg
                    .provider_extra
                    .as_ref()
                    .and_then(|e| e.get(ITEMS_KEY))
                    .and_then(|i| i.as_array())
                {
                    input.extend(items.iter().cloned());
                    continue;
                }
                if let Some(text) = msg.content.as_deref() {
                    if !text.is_empty() {
                        input.push(json!({ "role": "assistant", "content": text }));
                    }
                }
                for tc in msg.tool_calls.as_deref().unwrap_or(&[]) {
                    input.push(json!({
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }));
                }
            }
            Role::Tool => {
                in_leading_system = false;
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id.as_deref().unwrap_or_default(),
                    "output": msg.content.as_deref().unwrap_or_default(),
                }));
            }
        }
    }

    let instructions = (!instructions.is_empty()).then_some(instructions);
    (instructions, input)
}

/// Build an [`LlmResponse`] from a complete Responses API response object.
fn parse_response(parsed: &Value) -> LlmResponse {
    let output = parsed
        .get("output")
        .and_then(|o| o.as_array())
        .cloned()
        .unwrap_or_default();

    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::new();
    let mut replay_items: Vec<Value> = Vec::new();

    for item in &output {
        match item.get("type").and_then(|t| t.as_str()) {
            Some("message") => {
                for part in item
                    .get("content")
                    .and_then(|c| c.as_array())
                    .map(|a| a.as_slice())
                    .unwrap_or(&[])
                {
                    if part.get("type").and_then(|t| t.as_str()) == Some("output_text") {
                        if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                            content.push_str(t);
                        }
                    }
                }
                replay_items.push(item.clone());
            }
            Some("function_call") => {
                tool_calls.push(ToolCall {
                    id: item
                        .get("call_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    kind: "function".into(),
                    function: FunctionCall {
                        name: item
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        arguments: item
                            .get("arguments")
                            .and_then(|v| v.as_str())
                            .unwrap_or("{}")
                            .to_string(),
                    },
                    extra_content: None,
                });
                replay_items.push(item.clone());
            }
            Some("reasoning") => {
                for part in item
                    .get("summary")
                    .and_then(|s| s.as_array())
                    .map(|a| a.as_slice())
                    .unwrap_or(&[])
                {
                    if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                        if !reasoning.is_empty() {
                            reasoning.push('\n');
                        }
                        reasoning.push_str(t);
                    }
                }
                replay_items.push(item.clone());
            }
            _ => replay_items.push(item.clone()),
        }
    }

    let usage = parsed.get("usage");
    let read_u32 = |ptr: &str| {
        usage
            .and_then(|u| u.pointer(ptr))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
    };

    LlmResponse {
        content: (!content.is_empty()).then_some(content),
        reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
        finish_reason: if tool_calls.is_empty() {
            "stop".into()
        } else {
            "tool_calls".into()
        },
        tool_calls,
        input_tokens: read_u32("/input_tokens"),
        output_tokens: read_u32("/output_tokens"),
        cached_input_tokens: read_u32("/input_tokens_details/cached_tokens"),
        provider_extra: (!replay_items.is_empty()).then(|| json!({ ITEMS_KEY: replay_items })),
    }
}

// ── LlmProvider impl ──────────────────────────────────────────────────────────

#[async_trait]
impl LlmProvider for OpenAiResponsesProvider {
    #[instrument(skip(self, messages, tools), fields(model = %self.model, messages = messages.len()))]
    async fn complete(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
    ) -> Result<LlmResponse, LlmError> {
        let body = self.build_body(&messages, &tools, false);
        debug!("sending openai responses request");
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
        debug!("opening openai responses SSE stream");
        let resp = self.post(&body).await?;

        let mut final_response: Option<Value> = None;
        let mut line_buf = String::new();
        let mut byte_stream = resp.bytes_stream();

        while let Some(chunk_result) = byte_stream.next().await {
            let bytes = chunk_result.map_err(LlmError::Http)?;
            line_buf.push_str(&String::from_utf8_lossy(&bytes));

            while let Some(newline_pos) = line_buf.find('\n') {
                let line = line_buf[..newline_pos].trim_end_matches('\r').to_string();
                line_buf.drain(..=newline_pos);
                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };
                if data == "[DONE]" {
                    continue;
                }
                let event: Value = match serde_json::from_str(data) {
                    Ok(v) => v,
                    Err(e) => {
                        warn!("failed to parse responses SSE chunk: {e}");
                        continue;
                    }
                };

                match event.get("type").and_then(|t| t.as_str()) {
                    Some("response.output_text.delta") => {
                        if let Some(t) = event.get("delta").and_then(|v| v.as_str()) {
                            on_delta(StreamDelta {
                                content: Some(t.to_string()),
                                reasoning_content: None,
                            });
                        }
                    }
                    Some("response.reasoning_summary_text.delta") => {
                        if let Some(t) = event.get("delta").and_then(|v| v.as_str()) {
                            on_delta(StreamDelta {
                                content: None,
                                reasoning_content: Some(t.to_string()),
                            });
                        }
                    }
                    Some("response.completed") => {
                        final_response = event.get("response").cloned();
                    }
                    Some("response.failed") | Some("error") => {
                        return Err(LlmError::Api {
                            status: 500,
                            body: event.to_string(),
                        });
                    }
                    _ => {}
                }
            }
        }

        let parsed = final_response.ok_or(LlmError::EmptyResponse)?;
        Ok(parse_response(&parsed))
    }

    fn model(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_items_replay_verbatim() {
        let mut assistant = ChatMessage::assistant_with_tool_calls(
            None,
            vec![ToolCall {
                id: "call_1".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "search".into(),
                    arguments: "{}".into(),
                },
                extra_content: None,
            }],
        );
        assistant.provider_extra = Some(json!({
            ITEMS_KEY: [
                { "type": "reasoning", "id": "rs_1", "encrypted_content": "opaque..." },
                { "type": "function_call", "call_id": "call_1", "name": "search", "arguments": "{}" }
            ]
        }));

        let messages = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("find it"),
            assistant,
            ChatMessage::tool_result("call_1", "found"),
        ];
        let (instructions, input) = convert_messages(&messages);
        assert_eq!(instructions.as_deref(), Some("sys"));

        // user, reasoning (verbatim), function_call (verbatim), output.
        assert_eq!(input.len(), 4);
        assert_eq!(input[1]["type"], "reasoning");
        assert_eq!(input[1]["encrypted_content"], "opaque...");
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[3]["type"], "function_call_output");
        assert_eq!(input[3]["call_id"], "call_1");
    }

    #[test]
    fn parses_response_output() {
        let parsed = json!({
            "output": [
                { "type": "reasoning", "id": "rs_1", "encrypted_content": "xxx",
                  "summary": [{ "type": "summary_text", "text": "thought about it" }] },
                { "type": "message", "role": "assistant",
                  "content": [{ "type": "output_text", "text": "the answer" }] },
                { "type": "function_call", "call_id": "c1", "name": "f", "arguments": "{\"a\":1}" }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "input_tokens_details": { "cached_tokens": 80 }
            }
        });
        let resp = parse_response(&parsed);
        assert_eq!(resp.content.as_deref(), Some("the answer"));
        assert_eq!(resp.reasoning_content.as_deref(), Some("thought about it"));
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].id, "c1");
        assert_eq!(resp.finish_reason, "tool_calls");
        assert_eq!(resp.input_tokens, Some(100));
        assert_eq!(resp.cached_input_tokens, Some(80));
        let items = &resp.provider_extra.unwrap()[ITEMS_KEY];
        assert_eq!(items.as_array().unwrap().len(), 3);
    }
}
