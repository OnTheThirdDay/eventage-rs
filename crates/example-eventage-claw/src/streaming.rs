//! Streaming LLM provider that publishes `claw.stream.chunk` events per token.
//! Adapted from example-coding-agent/src/streaming.rs.

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use async_trait::async_trait;
use eventage::llm::{
    types::{ChatMessage, FunctionCall, LlmResponse, ToolCall, ToolDefinition},
    LlmError, LlmProvider,
};
use eventage::{Event, EventBus};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{debug, warn};

use crate::kinds::CLAW_STREAM_CHUNK;

// ── SSE serde types ───────────────────────────────────────────────────────────

#[derive(Serialize)]
struct CompletionRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a Value>,
}

#[derive(Deserialize, Debug)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize, Debug)]
struct StreamChoice {
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
struct Delta {
    content: Option<String>,
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

// ── StreamingOpenAiProvider ───────────────────────────────────────────────────

/// An [`LlmProvider`] that streams tokens and publishes `CLAW_STREAM_CHUNK`
/// events onto the bus for the TUI to display in real time.
pub struct StreamingOpenAiProvider {
    client: Client,
    base_url: String,
    api_key: String,
    model: String,
    tool_choice: Option<Value>,
    bus: EventBus,
    /// Set to `true` (e.g., by TUI Ctrl+X) to abort the current stream.
    pub cancelled: Arc<AtomicBool>,
}

impl StreamingOpenAiProvider {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        bus: EventBus,
    ) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: model.into(),
            tool_choice: None,
            bus,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    #[allow(dead_code)]
    pub fn ollama(model: impl Into<String>, bus: EventBus) -> Self {
        Self::new("http://localhost:11434/v1", "ollama", model, bus)
    }
}

#[async_trait]
impl LlmProvider for StreamingOpenAiProvider {
    async fn complete(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
    ) -> Result<LlmResponse, LlmError> {
        let url = format!("{}/chat/completions", self.base_url);
        let effective_tool_choice = if tools.is_empty() {
            None
        } else {
            self.tool_choice.as_ref()
        };

        let request = CompletionRequest {
            model: &self.model,
            messages: &messages,
            stream: true,
            tools,
            tool_choice: effective_tool_choice,
        };

        debug!("opening SSE stream");
        self.cancelled.store(false, Ordering::Relaxed);

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Accept", "text/event-stream")
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

        let mut content = String::new();
        let mut tool_acc: HashMap<usize, ToolCallAcc> = HashMap::new();
        let mut finish_reason = String::from("stop");
        let mut line_buf = String::new();

        let mut byte_stream = resp.bytes_stream();

        while let Some(chunk_result) = byte_stream.next().await {
            if self.cancelled.load(Ordering::Relaxed) {
                debug!("streaming cancelled by user");
                break;
            }

            let bytes = chunk_result.map_err(LlmError::Http)?;
            let text = String::from_utf8_lossy(&bytes);
            line_buf.push_str(&text);

            while let Some(newline_pos) = line_buf.find('\n') {
                let line = line_buf[..newline_pos].trim_end_matches('\r').to_string();
                line_buf = line_buf[newline_pos + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                let data = if let Some(d) = line.strip_prefix("data: ") {
                    d
                } else {
                    continue;
                };

                if data == "[DONE]" {
                    break;
                }

                let chunk: StreamChunk = match serde_json::from_str(data) {
                    Ok(c) => c,
                    Err(e) => {
                        warn!("failed to parse SSE chunk: {e}");
                        continue;
                    }
                };

                let Some(choice) = chunk.choices.into_iter().next() else {
                    continue;
                };

                if let Some(reason) = choice.finish_reason {
                    if !reason.is_empty() {
                        finish_reason = reason;
                    }
                }

                let delta = choice.delta;

                if let Some(text) = delta.content {
                    if !text.is_empty() {
                        content.push_str(&text);
                        let _ = self
                            .bus
                            .publish(Event::new(CLAW_STREAM_CHUNK, json!({ "content": text })))
                            .await;
                    }
                }

                if let Some(tc_deltas) = delta.tool_calls {
                    for d in tc_deltas {
                        let acc = tool_acc.entry(d.index).or_default();
                        if let Some(id) = d.id { acc.id = id; }
                        if let Some(kind) = d.kind { acc.kind = kind; }
                        if let Some(func) = d.function {
                            if let Some(name) = func.name { acc.name.push_str(&name); }
                            if let Some(args) = func.arguments { acc.arguments.push_str(&args); }
                        }
                    }
                }
            }
        }

        let mut sorted_indices: Vec<usize> = tool_acc.keys().copied().collect();
        sorted_indices.sort();

        let tool_calls: Vec<ToolCall> = sorted_indices
            .into_iter()
            .filter_map(|i| {
                let acc = tool_acc.remove(&i)?;
                Some(ToolCall {
                    id: acc.id,
                    kind: if acc.kind.is_empty() { "function".to_string() } else { acc.kind },
                    function: FunctionCall {
                        name: acc.name,
                        arguments: acc.arguments,
                    },
                    extra_content: None,
                })
            })
            .collect();

        Ok(LlmResponse {
            content: if content.is_empty() { None } else { Some(content) },
            tool_calls,
            finish_reason,
        })
    }

    fn model(&self) -> &str {
        &self.model
    }
}
