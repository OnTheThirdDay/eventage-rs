use async_trait::async_trait;
use eventage_core::{kinds, Event};
use eventage_llm::types::{ChatMessage, FunctionCall, ToolCall};

// ── AssemblyContext ───────────────────────────────────────────────────────────

/// Context data provided to a [`ContextAssembler`].
///
/// Contains active branch `events` and optionally `rejected_branches` 
/// for injecting negative constraints ("what not to do").
///
/// # Example
/// ```no_run
/// # use eventage_agent::AssemblyContext;
/// # let active_events: Vec<eventage_core::Event> = vec![];
/// # let rejected: Vec<Vec<eventage_core::Event>> = vec![];
/// // Basic usage (no negative context):
/// let ctx = AssemblyContext::new(&active_events);
///
/// // With negative context from a rollback point:
/// let ctx = AssemblyContext::new(&active_events).with_rejected_branches(rejected);
/// ```
pub struct AssemblyContext<'a> {
    /// Events on the current active branch, in chronological order.
    pub events: &'a [Event],
    /// Events from sealed (rejected) branches that share an ancestor checkpoint
    /// with the current active branch.  Each inner `Vec<Event>` is one rejected branch.
    pub rejected_branches: Vec<Vec<Event>>,
}

impl<'a> AssemblyContext<'a> {
    pub fn new(events: &'a [Event]) -> Self {
        Self {
            events,
            rejected_branches: vec![],
        }
    }

    pub fn with_rejected_branches(mut self, branches: Vec<Vec<Event>>) -> Self {
        self.rejected_branches = branches;
        self
    }
}

// ── ContextAssembler trait ────────────────────────────────────────────────────

/// Transforms the event log into a list of LLM `ChatMessage`s.
///
/// Allows filtering, windowing, summarization, or injecting negative feedback
/// prior to LLM reasoning.
#[async_trait]
pub trait ContextAssembler: Send + Sync {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage>;
}

// ── Internal minimal assemblers (used by AgentBuilder defaults) ───────────────

/// A basic assembler that prepends a fixed system prompt.
pub(crate) struct SystemPromptAssembler {
    pub(crate) system_prompt: String,
}

#[async_trait]
impl ContextAssembler for SystemPromptAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let mut messages = vec![ChatMessage::system(&self.system_prompt)];
        messages.extend(events_to_messages(context.events));
        messages
    }
}

/// A transparent assembler with no system prompt.
pub(crate) struct RawContextAssembler;

#[async_trait]
impl ContextAssembler for RawContextAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        events_to_messages(context.events)
    }
}

// ── Shared event→message conversion ──────────────────────────────────────────

/// Converts raw events to `ChatMessage`s.
pub fn events_to_messages(events: &[Event]) -> Vec<ChatMessage> {
    let mut messages = Vec::new();

    for event in events {
        match event.kind.as_str() {
            kinds::USER_MESSAGE => {
                if let Some(text) = event.payload.get("text").and_then(|v| v.as_str()) {
                    messages.push(ChatMessage::user(text));
                }
            }
            kinds::ASSISTANT_MESSAGE => {
                let content = event
                    .payload
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let tool_calls: Vec<ToolCall> = event
                    .payload
                    .get("tool_calls")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|tc| {
                                Some(ToolCall {
                                    id: tc.get("id")?.as_str()?.to_string(),
                                    kind: tc
                                        .get("type")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("function")
                                        .to_string(),
                                    function: FunctionCall {
                                        name: tc
                                            .get("function")?
                                            .get("name")?
                                            .as_str()?
                                            .to_string(),
                                        arguments: tc
                                            .get("function")?
                                            .get("arguments")?
                                            .as_str()?
                                            .to_string(),
                                    },
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                if !tool_calls.is_empty() {
                    messages.push(ChatMessage::assistant_with_tool_calls(content, tool_calls));
                } else if let Some(text) = content {
                    messages.push(ChatMessage::assistant(text));
                }
            }
            kinds::TOOL_RESULT => {
                if let Some(id) = event.payload.get("tool_call_id").and_then(|v| v.as_str()) {
                    let result = event
                        .payload
                        .get("result")
                        .map(|v| v.to_string())
                        .or_else(|| event.payload.get("error").map(|e| format!("error: {}", e)))
                        .unwrap_or_default();
                    messages.push(ChatMessage::tool_result(id, result));
                }
            }
            _ => {}
        }
    }

    messages
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use eventage_core::Event;
    use eventage_llm::types::Role;
    use serde_json::json;

    #[tokio::test]
    async fn assembles_user_message() {
        let assembler = RawContextAssembler;
        let events = vec![Event::new(kinds::USER_MESSAGE, json!({"text": "hello"}))];
        let ctx = AssemblyContext::new(&events);
        let messages = assembler.assemble(&ctx).await;
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content.as_deref(), Some("hello"));
    }

    #[tokio::test]
    async fn skips_observability_events() {
        let assembler = RawContextAssembler;
        let events = vec![
            Event::new(kinds::AGENT_CYCLE_START, json!({})),
            Event::new(kinds::USER_MESSAGE, json!({"text": "hi"})),
            Event::new(kinds::AGENT_CYCLE_END, json!({})),
        ];
        let ctx = AssemblyContext::new(&events);
        let messages = assembler.assemble(&ctx).await;
        assert_eq!(messages.len(), 1);
    }

    #[tokio::test]
    async fn system_prompt_assembler_prepends_system() {
        let assembler = SystemPromptAssembler {
            system_prompt: "Be helpful.".to_string(),
        };
        let events = vec![Event::new(kinds::USER_MESSAGE, json!({"text": "hi"}))];
        let ctx = AssemblyContext::new(&events);
        let messages = assembler.assemble(&ctx).await;
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content.as_deref(), Some("Be helpful."));
    }
}
