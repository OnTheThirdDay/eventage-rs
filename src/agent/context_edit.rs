//! Assembly-time context editing: reclaim budget by clearing stale tool
//! results before resorting to LLM summarization.
//!
//! # Overview
//!
//! In most long agent sessions the bulk of the context is not conversation —
//! it is old tool output (file dumps, search results, command logs) that the
//! model consumed once and rarely needs again. [`ToolResultClearingAssembler`]
//! wraps any [`ContextAssembler`] and, once the assembled context crosses a
//! token trigger, replaces the *content* of the oldest tool results with a
//! short placeholder. The assistant's tool-call records stay intact, so the
//! message history remains valid for every provider.
//!
//! This is the cheap first line of context management:
//!
//! - **Zero LLM calls** — unlike summarization, clearing is free.
//! - **Lossless** — the full result still lives in the event log (the bus is
//!   the source of truth); only the LLM's *view* is edited. Replay,
//!   observability, and rollback are unaffected.
//! - **Monotonic** — once a result is cleared it stays cleared, so the edited
//!   prefix of the conversation is stable across steps and stays friendly to
//!   provider prompt caches.
//!
//! Compose it under a [`SummarizingContextAssembler`](crate::SummarizingContextAssembler)
//! so summarization only triggers when clearing alone cannot reclaim enough:
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use eventage::agent::{DefaultContextAssembler, ToolResultClearingAssembler};
//! # use eventage::SummarizingContextAssembler;
//! # let llm: Arc<dyn eventage::llm::LlmProvider> = todo!();
//! let base = DefaultContextAssembler::new("You are helpful.");
//! let clearing = ToolResultClearingAssembler::new(Arc::new(base), 24_000);
//! let assembler = SummarizingContextAssembler::new(Arc::new(clearing), llm, 32_000, "session");
//! ```

use async_trait::async_trait;
use std::sync::Arc;
use std::sync::Mutex;
use tracing::debug;

use super::context::{AssemblyContext, ContextAssembler};
use super::tokens::messages_token_count;
use crate::llm::types::{ChatMessage, Role};

/// Wraps a [`ContextAssembler`] and clears old tool-result content once the
/// assembled context exceeds a token trigger.
///
/// The most recent [`keep_recent`](Self::keep_recent) tool results are always
/// kept verbatim; clearing is a monotonic ratchet (a cleared result never
/// reappears), keeping the context prefix stable for prompt caching.
pub struct ToolResultClearingAssembler {
    inner: Arc<dyn ContextAssembler>,
    /// Token estimate above which clearing kicks in.
    pub trigger_tokens: usize,
    /// Number of most-recent tool results always kept verbatim (default 5).
    pub keep_recent: usize,
    /// Monotonic count of tool results (from the start of the conversation)
    /// whose content has been cleared.
    cleared: Mutex<usize>,
}

impl ToolResultClearingAssembler {
    pub fn new(inner: Arc<dyn ContextAssembler>, trigger_tokens: usize) -> Self {
        Self {
            inner,
            trigger_tokens,
            keep_recent: 5,
            cleared: Mutex::new(0),
        }
    }

    /// Sets how many of the most recent tool results are always kept verbatim.
    pub fn with_keep_recent(mut self, n: usize) -> Self {
        self.keep_recent = n;
        self
    }

    fn placeholder(original: &str) -> String {
        format!(
            "[cleared by harness: {} chars of stale tool output removed from context. \
             The full result remains in the event log; re-run the tool if you need it again.]",
            original.len()
        )
    }

    /// Replace the content of the first `n` tool-result messages.
    fn apply_clearing(messages: &mut [ChatMessage], n: usize) {
        let mut seen = 0usize;
        for m in messages.iter_mut() {
            if m.role != Role::Tool {
                continue;
            }
            if seen >= n {
                break;
            }
            seen += 1;
            let already_cleared = m
                .content
                .as_deref()
                .is_some_and(|c| c.starts_with("[cleared by harness:"));
            if !already_cleared {
                let original = m.content.take().unwrap_or_default();
                m.content = Some(Self::placeholder(&original));
            }
        }
    }
}

#[async_trait]
impl ContextAssembler for ToolResultClearingAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let mut messages = self.inner.assemble(context).await;

        let total_tool_results = messages.iter().filter(|m| m.role == Role::Tool).count();
        let mut cleared = self.cleared.lock().unwrap_or_else(|e| e.into_inner());

        // Always re-apply the ratchet so previously cleared results stay cleared.
        Self::apply_clearing(&mut messages, *cleared);

        if messages_token_count(&messages) > self.trigger_tokens {
            let target = total_tool_results.saturating_sub(self.keep_recent);
            if target > *cleared {
                debug!(
                    cleared_before = *cleared,
                    cleared_after = target,
                    total_tool_results,
                    "context over trigger — clearing stale tool results"
                );
                *cleared = target;
                Self::apply_clearing(&mut messages, *cleared);
            }
        }

        messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::context::DefaultContextAssembler;
    use crate::event::{kinds, Event};
    use serde_json::json;

    fn tool_turn(id: &str, big: &str) -> Vec<Event> {
        vec![
            Event::new(
                kinds::ASSISTANT_MESSAGE,
                json!({
                    "content": null,
                    "tool_calls": [{
                        "id": id,
                        "type": "function",
                        "function": { "name": "read_file", "arguments": "{}" }
                    }]
                }),
            ),
            Event::new(
                kinds::TOOL_RESULT,
                json!({ "tool_call_id": id, "name": "read_file", "result": big }),
            ),
        ]
    }

    #[tokio::test]
    async fn clears_old_tool_results_when_over_trigger() {
        let big = "x".repeat(4000);
        let mut events = vec![Event::new(kinds::USER_MESSAGE, json!({"text": "go"}))];
        for i in 0..4 {
            events.extend(tool_turn(&format!("c{i}"), &big));
        }

        let inner = DefaultContextAssembler::without_system_prompt();
        // ~4000 tokens of budget vs ~4 × 1000-token results.
        let assembler =
            ToolResultClearingAssembler::new(Arc::new(inner), 2_500).with_keep_recent(1);

        let ctx = AssemblyContext::new(&events);
        let messages = assembler.assemble(&ctx).await;

        let tool_msgs: Vec<&ChatMessage> =
            messages.iter().filter(|m| m.role == Role::Tool).collect();
        assert_eq!(tool_msgs.len(), 4);
        for cleared in &tool_msgs[..3] {
            assert!(
                cleared
                    .content
                    .as_deref()
                    .unwrap()
                    .starts_with("[cleared by harness:"),
                "old results should be cleared"
            );
        }
        assert!(
            tool_msgs[3].content.as_deref().unwrap().contains("xxx"),
            "most recent result must stay verbatim"
        );
    }

    #[tokio::test]
    async fn under_trigger_leaves_everything_verbatim() {
        let mut events = vec![Event::new(kinds::USER_MESSAGE, json!({"text": "go"}))];
        events.extend(tool_turn("c0", "small result"));

        let inner = DefaultContextAssembler::without_system_prompt();
        let assembler = ToolResultClearingAssembler::new(Arc::new(inner), 100_000);

        let ctx = AssemblyContext::new(&events);
        let messages = assembler.assemble(&ctx).await;
        let tool_msg = messages.iter().find(|m| m.role == Role::Tool).unwrap();
        assert_eq!(tool_msg.content.as_deref(), Some("\"small result\""));
    }

    #[tokio::test]
    async fn clearing_is_monotonic_across_calls() {
        let big = "y".repeat(4000);
        let mut events = vec![Event::new(kinds::USER_MESSAGE, json!({"text": "go"}))];
        for i in 0..4 {
            events.extend(tool_turn(&format!("c{i}"), &big));
        }

        let inner = DefaultContextAssembler::without_system_prompt();
        let assembler =
            ToolResultClearingAssembler::new(Arc::new(inner), 2_500).with_keep_recent(1);

        // First call ratchets the clearing boundary.
        let ctx = AssemblyContext::new(&events);
        assembler.assemble(&ctx).await;

        // Second call on the *same* events must keep the same results cleared,
        // even though the (now smaller) context is under the trigger.
        let messages = assembler.assemble(&ctx).await;
        let cleared_count = messages
            .iter()
            .filter(|m| m.role == Role::Tool)
            .filter(|m| {
                m.content
                    .as_deref()
                    .unwrap_or("")
                    .starts_with("[cleared by harness:")
            })
            .count();
        assert_eq!(cleared_count, 3, "ratchet must persist across calls");
    }
}
