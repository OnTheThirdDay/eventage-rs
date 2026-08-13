use super::error::LlmError;
use super::provider::LlmProvider;
use super::types::{ChatMessage, LlmResponse, ToolDefinition};
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

/// A deterministic mock LLM for testing.
///
/// Exhausts the pre-configured response queue sequentially and
/// repeats the final response once empty.
pub struct MockLlmProvider {
    responses: Arc<Mutex<Vec<LlmResponse>>>,
    index: Arc<Mutex<usize>>,
}

impl MockLlmProvider {
    pub fn new(responses: Vec<LlmResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            index: Arc::new(Mutex::new(0)),
        }
    }

    /// Initializes a mock returning sequential plain text responses.
    pub fn with_texts(texts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let responses = texts
            .into_iter()
            .map(|t| LlmResponse {
                content: Some(t.into()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                ..Default::default()
            })
            .collect();
        Self::new(responses)
    }
}

#[async_trait]
impl LlmProvider for MockLlmProvider {
    async fn complete(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Vec<ToolDefinition>,
    ) -> Result<LlmResponse, LlmError> {
        let responses = self.responses.lock().unwrap();
        if responses.is_empty() {
            return Ok(LlmResponse {
                content: Some("(mock empty response)".to_string()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                ..Default::default()
            });
        }
        let mut idx = self.index.lock().unwrap();
        let response = responses[*idx].clone();
        if *idx + 1 < responses.len() {
            *idx += 1;
        }
        Ok(response)
    }

    fn model(&self) -> &str {
        "mock"
    }
}
