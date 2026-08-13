//! Native Anthropic Messages API provider.
//!
//! Speaks the Messages API directly rather than an OpenAI-compatibility
//! shim, which unlocks the capabilities that define the platform:
//!
//! - **Prompt caching** — `cache_control` breakpoints are placed
//!   automatically (end of system prompt + end of the latest turn), so long
//!   agent sessions hit the cache on every step. On by default.
//! - **Extended thinking** — [`with_thinking`](AnthropicProvider::with_thinking)
//!   enables a thinking budget; thinking/redacted-thinking blocks (with
//!   signatures) are captured into [`LlmResponse::provider_extra`], stored on
//!   the event bus, and replayed on subsequent requests as the API requires
//!   for tool loops.
//! - **Streaming** — full SSE support including thinking deltas.
//!
//! ```no_run
//! use eventage::llm::AnthropicProvider;
//!
//! let llm = AnthropicProvider::new(std::env::var("ANTHROPIC_API_KEY").unwrap(), "claude-sonnet-4-5")
//!     .with_max_tokens(16_000)
//!     .with_thinking(4_096);
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

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 8192;

/// Key under which Anthropic thinking blocks travel in
/// [`ChatMessage::provider_extra`] / [`LlmResponse::provider_extra`].
const BLOCKS_KEY: &str = "anthropic_blocks";

/// Native Anthropic Messages API provider. See the [module docs](self).
pub struct AnthropicProvider {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
    max_tokens: u32,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u32>,
    stop_sequences: Vec<String>,
    /// Thinking budget in tokens; `None` disables extended thinking.
    thinking_budget: Option<u32>,
    /// Automatic `cache_control` breakpoints (default on).
    prompt_caching: bool,
    tool_choice: Option<Value>,
    beta_headers: Vec<String>,
    extra_body: Map<String, Value>,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(600))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_default();
        Self {
            client,
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            model: model.into(),
            max_tokens: DEFAULT_MAX_TOKENS,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: Vec::new(),
            thinking_budget: None,
            prompt_caching: true,
            tool_choice: None,
            beta_headers: Vec::new(),
            extra_body: Map::new(),
        }
    }

    /// Point at a different endpoint (proxy, gateway).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Required completion cap (`max_tokens`); defaults to 8192.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn with_stop_sequences(mut self, stops: Vec<String>) -> Self {
        self.stop_sequences = stops;
        self
    }

    /// Enable extended thinking with the given token budget.
    ///
    /// Thinking blocks are preserved across tool-loop steps via
    /// `provider_extra` as the API requires. Note: the API rejects
    /// `temperature`/`top_p`/`top_k` alongside thinking, so those are
    /// omitted from requests while thinking is enabled.
    pub fn with_thinking(mut self, budget_tokens: u32) -> Self {
        self.thinking_budget = Some(budget_tokens);
        self
    }

    /// Disable the automatic `cache_control` breakpoints.
    pub fn without_prompt_caching(mut self) -> Self {
        self.prompt_caching = false;
        self
    }

    /// Force tool choice, e.g. `json!({"type": "any"})` or
    /// `json!({"type": "tool", "name": "my_tool"})`.
    pub fn with_tool_choice(mut self, choice: Value) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Add an `anthropic-beta` header value (e.g. `"context-1m-2025-08-07"`).
    pub fn with_beta(mut self, beta: impl Into<String>) -> Self {
        self.beta_headers.push(beta.into());
        self
    }

    /// Merge an arbitrary top-level field into every request.
    pub fn with_body_param(mut self, key: impl Into<String>, value: Value) -> Self {
        self.extra_body.insert(key.into(), value);
        self
    }

    // ── Request construction ──────────────────────────────────────────────────

    fn build_body(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        stream: bool,
    ) -> Value {
        let (system, converted) = convert_messages(messages, self.prompt_caching);

        let mut body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": converted,
        });
        let obj = body.as_object_mut().expect("body is an object");

        if let Some(system) = system {
            obj.insert("system".into(), system);
        }
        if !tools.is_empty() {
            let defs: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.function.name,
                        "description": t.function.description,
                        "input_schema": t.function.parameters,
                    })
                })
                .collect();
            obj.insert("tools".into(), Value::Array(defs));
            if let Some(choice) = &self.tool_choice {
                obj.insert("tool_choice".into(), choice.clone());
            }
        }
        if let Some(budget) = self.thinking_budget {
            obj.insert(
                "thinking".into(),
                json!({ "type": "enabled", "budget_tokens": budget }),
            );
        } else {
            // Sampling params are incompatible with extended thinking.
            if let Some(t) = self.temperature {
                obj.insert("temperature".into(), json!(t));
            }
            if let Some(p) = self.top_p {
                obj.insert("top_p".into(), json!(p));
            }
            if let Some(k) = self.top_k {
                obj.insert("top_k".into(), json!(k));
            }
        }
        if !self.stop_sequences.is_empty() {
            obj.insert("stop_sequences".into(), json!(self.stop_sequences));
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
        let url = format!("{}/v1/messages", self.base_url);
        let mut req = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION);
        if !self.beta_headers.is_empty() {
            req = req.header("anthropic-beta", self.beta_headers.join(","));
        }
        let resp = req.json(body).send().await.map_err(LlmError::Http)?;

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

    fn response_from_blocks(
        blocks: Vec<Value>,
        stop_reason: Option<String>,
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
        cached_input_tokens: Option<u32>,
    ) -> LlmResponse {
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut tool_calls = Vec::new();
        let mut opaque_blocks = Vec::new();

        for block in blocks {
            match block.get("type").and_then(|t| t.as_str()) {
                Some("text") => {
                    if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                        content.push_str(t);
                    }
                }
                Some("thinking") => {
                    if let Some(t) = block.get("thinking").and_then(|t| t.as_str()) {
                        reasoning.push_str(t);
                    }
                    opaque_blocks.push(block);
                }
                Some("redacted_thinking") => opaque_blocks.push(block),
                Some("tool_use") => {
                    let id = block
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let name = block
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let args = block.get("input").cloned().unwrap_or(json!({}));
                    tool_calls.push(ToolCall {
                        id,
                        kind: "function".into(),
                        function: FunctionCall {
                            name,
                            arguments: args.to_string(),
                        },
                        extra_content: None,
                    });
                }
                _ => {}
            }
        }

        LlmResponse {
            content: (!content.is_empty()).then_some(content),
            reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
            tool_calls,
            finish_reason: stop_reason.unwrap_or_else(|| "end_turn".into()),
            input_tokens,
            output_tokens,
            cached_input_tokens,
            provider_extra: (!opaque_blocks.is_empty())
                .then(|| json!({ BLOCKS_KEY: opaque_blocks })),
        }
    }
}

// ── Message conversion ────────────────────────────────────────────────────────

/// Convert harness messages to `(system, messages)` in Anthropic format.
///
/// - Leading `system` messages become the top-level `system` parameter;
///   later ones become `[system]`-prefixed user turns (the API only accepts
///   system content up front).
/// - Consecutive same-role messages are merged (tool results share one user
///   turn, as required for parallel tool use).
/// - Assistant `provider_extra` thinking blocks are re-emitted **first** in
///   the assistant turn, as the API requires.
/// - With `caching`, `cache_control` breakpoints are placed on the system
///   prompt and the final content block.
fn convert_messages(messages: &[ChatMessage], caching: bool) -> (Option<Value>, Vec<Value>) {
    let mut system_text = String::new();
    let mut in_leading_system = true;
    let mut out: Vec<(String, Vec<Value>)> = Vec::new();

    let push = |role: &str, blocks: Vec<Value>, out: &mut Vec<(String, Vec<Value>)>| {
        if blocks.is_empty() {
            return;
        }
        if let Some((last_role, last_blocks)) = out.last_mut() {
            if last_role == role {
                last_blocks.extend(blocks);
                return;
            }
        }
        out.push((role.to_string(), blocks));
    };

    for msg in messages {
        match msg.role {
            Role::System => {
                let text = msg.content.as_deref().unwrap_or_default();
                if in_leading_system {
                    if !system_text.is_empty() {
                        system_text.push_str("\n\n");
                    }
                    system_text.push_str(text);
                } else {
                    push(
                        "user",
                        vec![json!({ "type": "text", "text": format!("[system]\n{text}") })],
                        &mut out,
                    );
                }
            }
            Role::User => {
                in_leading_system = false;
                if msg.is_multimodal() {
                    // Map each part to its Anthropic block, prefixing the
                    // sender name onto the first text block.
                    let mut blocks: Vec<Value> = Vec::new();
                    let mut named = msg.name.is_none();
                    for part in &msg.parts {
                        let mut block = part.to_anthropic_json();
                        if !named {
                            if let Some(text) = part.as_text() {
                                block = json!({
                                    "type": "text",
                                    "text": format!(
                                        "[{}] {text}",
                                        msg.name.as_deref().unwrap_or_default()
                                    )
                                });
                                named = true;
                            }
                        }
                        blocks.push(block);
                    }
                    push("user", blocks, &mut out);
                } else {
                    let text = msg.content.as_deref().unwrap_or_default();
                    let text = match &msg.name {
                        Some(name) => format!("[{name}] {text}"),
                        None => text.to_string(),
                    };
                    push(
                        "user",
                        vec![json!({ "type": "text", "text": text })],
                        &mut out,
                    );
                }
            }
            Role::Assistant => {
                in_leading_system = false;
                let mut blocks: Vec<Value> = Vec::new();
                // Thinking blocks must precede other content.
                if let Some(prior) = msg
                    .provider_extra
                    .as_ref()
                    .and_then(|e| e.get(BLOCKS_KEY))
                    .and_then(|b| b.as_array())
                {
                    blocks.extend(prior.iter().cloned());
                }
                if let Some(text) = msg.content.as_deref() {
                    if !text.is_empty() {
                        blocks.push(json!({ "type": "text", "text": text }));
                    }
                }
                for tc in msg.tool_calls.as_deref().unwrap_or(&[]) {
                    let input: Value =
                        serde_json::from_str(&tc.function.arguments).unwrap_or(json!({}));
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": input,
                    }));
                }
                push("assistant", blocks, &mut out);
            }
            Role::Tool => {
                in_leading_system = false;
                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id.as_deref().unwrap_or_default(),
                    "content": msg.content.as_deref().unwrap_or_default(),
                });
                push("user", vec![block], &mut out);
            }
        }
    }

    let mut converted: Vec<Value> = out
        .into_iter()
        .map(|(role, blocks)| json!({ "role": role, "content": blocks }))
        .collect();

    let system = if system_text.is_empty() {
        None
    } else if caching {
        Some(json!([{
            "type": "text",
            "text": system_text,
            "cache_control": { "type": "ephemeral" }
        }]))
    } else {
        Some(Value::String(system_text))
    };

    // Moving breakpoint: cache everything up to and including the latest turn.
    if caching {
        if let Some(last_msg) = converted.last_mut() {
            if let Some(last_block) = last_msg
                .get_mut("content")
                .and_then(|c| c.as_array_mut())
                .and_then(|blocks| blocks.last_mut())
                .and_then(|b| b.as_object_mut())
            {
                last_block.insert("cache_control".into(), json!({ "type": "ephemeral" }));
            }
        }
    }

    (system, converted)
}

// ── LlmProvider impl ──────────────────────────────────────────────────────────

#[async_trait]
impl LlmProvider for AnthropicProvider {
    #[instrument(skip(self, messages, tools), fields(model = %self.model, messages = messages.len()))]
    async fn complete(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
    ) -> Result<LlmResponse, LlmError> {
        let body = self.build_body(&messages, &tools, false);
        debug!("sending anthropic messages request");
        let resp = self.post(&body).await?;
        let parsed: Value = resp.json().await.map_err(LlmError::Http)?;

        let blocks = parsed
            .get("content")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();
        let stop_reason = parsed
            .get("stop_reason")
            .and_then(|v| v.as_str())
            .map(String::from);
        let usage = parsed.get("usage");
        let read_u32 = |key: &str| {
            usage
                .and_then(|u| u.get(key))
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
        };

        Ok(Self::response_from_blocks(
            blocks,
            stop_reason,
            read_u32("input_tokens"),
            read_u32("output_tokens"),
            read_u32("cache_read_input_tokens"),
        ))
    }

    /// Native structured output via forced single-tool use — the canonical
    /// Anthropic technique: the schema becomes a tool's `input_schema` and
    /// the model is required to call it, so the input *is* the result.
    #[instrument(skip(self, messages, schema), fields(model = %self.model, schema_name = schema_name))]
    async fn complete_structured(
        &self,
        messages: Vec<ChatMessage>,
        schema_name: &str,
        schema: Value,
    ) -> Result<Value, LlmError> {
        let (system, converted) = convert_messages(&messages, self.prompt_caching);
        let mut body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": converted,
            "tools": [{
                "name": schema_name,
                "description": format!("Return the structured '{schema_name}' result."),
                "input_schema": schema,
            }],
            "tool_choice": { "type": "tool", "name": schema_name },
        });
        if let Some(obj) = body.as_object_mut() {
            if let Some(system) = system {
                obj.insert("system".into(), system);
            }
            for (k, v) in &self.extra_body {
                obj.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }

        let resp = self.post(&body).await?;
        let parsed: Value = resp.json().await.map_err(LlmError::Http)?;

        parsed
            .get("content")
            .and_then(|c| c.as_array())
            .and_then(|blocks| {
                blocks
                    .iter()
                    .find(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_use"))
            })
            .and_then(|b| b.get("input").cloned())
            .ok_or_else(|| {
                LlmError::Structured(format!(
                    "model did not invoke the '{schema_name}' structured-output tool"
                ))
            })
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
        debug!("opening anthropic SSE stream");
        let resp = self.post(&body).await?;

        // Accumulate content blocks by index as deltas arrive.
        let mut blocks: Vec<Value> = Vec::new();
        let mut tool_json: Vec<String> = Vec::new();
        let mut stop_reason: Option<String> = None;
        let mut input_tokens = None;
        let mut output_tokens = None;
        let mut cached_input_tokens = None;

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
                let event: Value = match serde_json::from_str(data) {
                    Ok(v) => v,
                    Err(e) => {
                        warn!("failed to parse anthropic SSE chunk: {e}");
                        continue;
                    }
                };

                match event.get("type").and_then(|t| t.as_str()) {
                    Some("message_start") => {
                        let usage = event.pointer("/message/usage");
                        let read = |key: &str| {
                            usage
                                .and_then(|u| u.get(key))
                                .and_then(|v| v.as_u64())
                                .map(|v| v as u32)
                        };
                        input_tokens = read("input_tokens");
                        cached_input_tokens = read("cache_read_input_tokens");
                    }
                    Some("content_block_start") => {
                        let idx = event.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                        let block = event.get("content_block").cloned().unwrap_or(json!({}));
                        while blocks.len() <= idx {
                            blocks.push(json!({}));
                            tool_json.push(String::new());
                        }
                        blocks[idx] = block;
                    }
                    Some("content_block_delta") => {
                        let idx = event.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                        if idx >= blocks.len() {
                            continue;
                        }
                        let Some(delta) = event.get("delta") else {
                            continue;
                        };
                        match delta.get("type").and_then(|t| t.as_str()) {
                            Some("text_delta") => {
                                if let Some(t) = delta.get("text").and_then(|v| v.as_str()) {
                                    append_str(&mut blocks[idx], "text", t);
                                    on_delta(StreamDelta {
                                        content: Some(t.to_string()),
                                        reasoning_content: None,
                                    });
                                }
                            }
                            Some("thinking_delta") => {
                                if let Some(t) = delta.get("thinking").and_then(|v| v.as_str()) {
                                    append_str(&mut blocks[idx], "thinking", t);
                                    on_delta(StreamDelta {
                                        content: None,
                                        reasoning_content: Some(t.to_string()),
                                    });
                                }
                            }
                            Some("signature_delta") => {
                                if let Some(s) = delta.get("signature").and_then(|v| v.as_str()) {
                                    append_str(&mut blocks[idx], "signature", s);
                                }
                            }
                            Some("input_json_delta") => {
                                if let Some(j) = delta.get("partial_json").and_then(|v| v.as_str())
                                {
                                    tool_json[idx].push_str(j);
                                }
                            }
                            _ => {}
                        }
                    }
                    Some("message_delta") => {
                        if let Some(reason) =
                            event.pointer("/delta/stop_reason").and_then(|v| v.as_str())
                        {
                            stop_reason = Some(reason.to_string());
                        }
                        if let Some(out) = event
                            .pointer("/usage/output_tokens")
                            .and_then(|v| v.as_u64())
                        {
                            output_tokens = Some(out as u32);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Fill accumulated tool_use inputs.
        for (idx, json_str) in tool_json.iter().enumerate() {
            if json_str.is_empty() {
                continue;
            }
            if let Ok(input) = serde_json::from_str::<Value>(json_str) {
                if let Some(obj) = blocks[idx].as_object_mut() {
                    obj.insert("input".into(), input);
                }
            }
        }

        Ok(Self::response_from_blocks(
            blocks,
            stop_reason,
            input_tokens,
            output_tokens,
            cached_input_tokens,
        ))
    }

    fn model(&self) -> &str {
        &self.model
    }
}

/// Append `text` to the string field `key` of a JSON object block.
fn append_str(block: &mut Value, key: &str, text: &str) {
    let Some(obj) = block.as_object_mut() else {
        return;
    };
    match obj.get_mut(key) {
        Some(Value::String(s)) => s.push_str(text),
        _ => {
            obj.insert(key.to_string(), Value::String(text.to_string()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_tool_loop_with_merged_results() {
        let messages = vec![
            ChatMessage::system("be helpful"),
            ChatMessage::user("run both tools"),
            ChatMessage::assistant_with_tool_calls(
                None,
                vec![
                    ToolCall {
                        id: "t1".into(),
                        kind: "function".into(),
                        function: FunctionCall {
                            name: "a".into(),
                            arguments: r#"{"x":1}"#.into(),
                        },
                        extra_content: None,
                    },
                    ToolCall {
                        id: "t2".into(),
                        kind: "function".into(),
                        function: FunctionCall {
                            name: "b".into(),
                            arguments: "{}".into(),
                        },
                        extra_content: None,
                    },
                ],
            ),
            ChatMessage::tool_result("t1", "one"),
            ChatMessage::tool_result("t2", "two"),
        ];

        let (system, converted) = convert_messages(&messages, false);
        assert_eq!(system, Some(Value::String("be helpful".into())));
        assert_eq!(
            converted.len(),
            3,
            "user, assistant, merged tool-result user"
        );

        let assistant = &converted[1];
        let blocks = assistant["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "tool_use");
        assert_eq!(blocks[0]["input"]["x"], 1);

        let results = &converted[2];
        assert_eq!(results["role"], "user");
        let blocks = results["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2, "parallel tool results share one user turn");
        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[1]["tool_use_id"], "t2");
    }

    #[test]
    fn caching_places_breakpoints() {
        let messages = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("hi"),
            ChatMessage::assistant("hello"),
            ChatMessage::user("again"),
        ];
        let (system, converted) = convert_messages(&messages, true);

        let system = system.unwrap();
        assert_eq!(system[0]["cache_control"]["type"], "ephemeral");

        let last_blocks = converted.last().unwrap()["content"].as_array().unwrap();
        assert_eq!(
            last_blocks.last().unwrap()["cache_control"]["type"],
            "ephemeral",
            "moving breakpoint on the latest turn"
        );
        // Earlier messages carry no breakpoints.
        assert!(converted[0]["content"][0].get("cache_control").is_none());
    }

    #[test]
    fn thinking_blocks_replay_first() {
        let mut assistant = ChatMessage::assistant_with_tool_calls(
            Some("visible".into()),
            vec![ToolCall {
                id: "t1".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "a".into(),
                    arguments: "{}".into(),
                },
                extra_content: None,
            }],
        );
        assistant.provider_extra = Some(json!({
            BLOCKS_KEY: [{ "type": "thinking", "thinking": "hmm", "signature": "sig" }]
        }));

        let messages = vec![ChatMessage::user("q"), assistant];
        let (_, converted) = convert_messages(&messages, false);
        let blocks = converted[1]["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "thinking", "thinking must come first");
        assert_eq!(blocks[0]["signature"], "sig");
        assert_eq!(blocks[1]["type"], "text");
        assert_eq!(blocks[2]["type"], "tool_use");
    }

    #[test]
    fn late_system_messages_become_user_turns() {
        let messages = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("hi"),
            ChatMessage::system("negative context warning"),
        ];
        let (system, converted) = convert_messages(&messages, false);
        assert_eq!(system, Some(Value::String("sys".into())));
        // Merged into the preceding user turn as a [system]-prefixed block.
        assert_eq!(converted.len(), 1);
        let blocks = converted[0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert!(blocks[1]["text"].as_str().unwrap().starts_with("[system]"));
    }
}
