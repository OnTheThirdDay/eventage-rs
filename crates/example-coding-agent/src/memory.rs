//! Context compaction — automatically summarises long conversations to keep the
//! LLM prompt within a token budget.
//!
//! [`CompactingContextAssembler`] approximates the token count of the assembled
//! message list. When it exceeds [`Self::compaction_threshold`] (default 85%)
//! of [`Self::max_tokens`], the assembler compacts: it calls the LLM to
//! produce a running summary, stores it in-memory, and builds future prompts
//! as `[system] + [summary] + [recent_window]` instead of the full history.
//!
//! The compacted summary is also published to the bus as a
//! [`kinds::CODING_CONTEXT_COMPACTED`] event for observability.

use std::sync::Arc;

use async_trait::async_trait;
use eventage::llm::{types::ChatMessage, LlmProvider};
use eventage::{agent::context::events_to_messages, AssemblyContext, ContextAssembler};
use eventage::{Event, EventBus};
use serde_json::json;
use tokio::sync::Mutex;
use tracing::info;

use crate::kinds::CODING_CONTEXT_COMPACTED;
use crate::workspace::Workspace;

// ── Token estimation ──────────────────────────────────────────────────────────

/// Rough token estimate: split on whitespace, multiply by 1.3 for sub-word
/// overhead. Good enough for deciding when to compact — no external tokenizer.
fn estimate_tokens(s: &str) -> usize {
    let words = s.split_whitespace().count();
    (words as f64 * 1.3).ceil() as usize
}

fn tokens_in_messages(msgs: &[ChatMessage]) -> usize {
    msgs.iter()
        .map(|m| {
            let content_tokens = m.content.as_deref().map(estimate_tokens).unwrap_or(0);
            let tool_call_tokens = m
                .tool_calls
                .as_deref()
                .map(|tcs| {
                    tcs.iter()
                        .map(|tc| estimate_tokens(&tc.function.arguments) + 10)
                        .sum::<usize>()
                })
                .unwrap_or(0);
            content_tokens + tool_call_tokens + 4 // per-message overhead
        })
        .sum()
}

// ── CompactingContextAssembler ────────────────────────────────────────────────

/// Summary state stored between compaction calls.
struct CompactionState {
    /// The LLM-generated summary of the conversation up to the compaction point.
    summary: String,
    /// Number of events included in the summary (used to find the "recent" tail).
    events_summarised: usize,
}

/// A [`ContextAssembler`] that compacts the conversation when the token budget
/// is exceeded.
///
/// Each call to `assemble`:
/// 1. Converts the event log to messages.
/// 2. Estimates the token count.
/// 3. If count < threshold: return normally.
/// 4. If count ≥ threshold and no compaction pending: fire a background LLM
///    call to summarise, cache the result, publish a `coding.context.compacted`
///    event, then return a compacted prompt.
/// 5. If a summary already exists: use it immediately to build the prompt.
pub struct CompactingContextAssembler {
    pub system_prompt: String,
    /// Hard token budget (e.g., 120_000 for GPT-4o).
    pub max_tokens: usize,
    /// Fraction of `max_tokens` at which to trigger compaction (default 0.85).
    pub compaction_threshold: f64,
    /// Number of most-recent conversation messages to keep verbatim after a
    /// compaction. The rest are replaced by the summary.
    pub recent_window: usize,
    llm: Arc<dyn LlmProvider>,
    bus: EventBus,
    workspace: Arc<Workspace>,
    state: Mutex<Option<CompactionState>>,
}

impl CompactingContextAssembler {
    pub fn new(
        system_prompt: impl Into<String>,
        max_tokens: usize,
        llm: Arc<dyn LlmProvider>,
        bus: EventBus,
        workspace: Arc<Workspace>,
    ) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            max_tokens,
            compaction_threshold: 0.85,
            recent_window: 20,
            llm,
            bus,
            workspace,
            state: Mutex::new(None),
        }
    }

    #[allow(dead_code)]
    pub fn with_compaction_threshold(mut self, threshold: f64) -> Self {
        self.compaction_threshold = threshold.clamp(0.5, 0.99);
        self
    }

    pub fn with_recent_window(mut self, n: usize) -> Self {
        self.recent_window = n.max(4);
        self
    }

    fn token_budget(&self) -> usize {
        (self.max_tokens as f64 * self.compaction_threshold) as usize
    }

    /// Build the workspace status line injected into every prompt.
    fn workspace_status(&self) -> String {
        match self.workspace.list_files() {
            Err(_) => "[workspace: could not read]".to_string(),
            Ok(files) if files.is_empty() => "[workspace: empty]".to_string(),
            Ok(files) => {
                let entries: Vec<String> = files
                    .iter()
                    .map(|f| format!("{} ({}B)", f.path, f.size_bytes))
                    .collect();
                format!("[workspace: {}]", entries.join(", "))
            }
        }
    }

    /// Call the LLM to produce a summary of `messages` and cache it.
    async fn compact(&self, messages: &[ChatMessage], events_summarised: usize) {
        let mut compaction_msgs = vec![ChatMessage::system(
            "You are a context compactor. Produce a dense, precise summary of the \
             conversation below. Keep all: task objectives, decisions made, files \
             written, commands run, errors encountered, and their resolutions. \
             Output only the summary — no preamble.",
        )];
        compaction_msgs.extend_from_slice(messages);
        compaction_msgs.push(ChatMessage::user(
            "Summarise the conversation above into a single assistant message.",
        ));

        match self.llm.complete(compaction_msgs, vec![]).await {
            Ok(resp) => {
                let summary = resp.content.unwrap_or_default();
                info!(
                    events_summarised,
                    summary_len = summary.len(),
                    "context compacted"
                );

                let _ = self
                    .bus
                    .publish(Event::new(
                        CODING_CONTEXT_COMPACTED,
                        json!({
                            "events_summarised": events_summarised,
                            "summary_tokens": estimate_tokens(&summary),
                        }),
                    ))
                    .await;

                let mut state = self.state.lock().await;
                *state = Some(CompactionState {
                    summary,
                    events_summarised,
                });
            }
            Err(e) => {
                tracing::warn!("context compaction LLM call failed: {e}");
            }
        }
    }
}

#[async_trait]
impl ContextAssembler for CompactingContextAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        // ── 1. System + workspace status ──────────────────────────────────────
        let mut base = vec![
            ChatMessage::system(&self.system_prompt),
            ChatMessage::system(self.workspace_status()),
        ];

        let all_conv = events_to_messages(context.events);
        if all_conv.is_empty() {
            return base;
        }

        // ── 2. Check if we have an existing compaction summary ────────────────
        let state_guard = self.state.lock().await;
        if let Some(ref state) = *state_guard {
            // Use: [system] + [summary] + [recent window from after compaction]
            let recent_start = state.events_summarised;
            let recent_conv = if recent_start < all_conv.len() {
                &all_conv[recent_start..]
            } else {
                &[]
            };

            base.push(ChatMessage::system(format!(
                "[Earlier conversation summary]\n{}",
                state.summary
            )));

            // Apply recent_window cap.
            let window = recent_conv.len().min(self.recent_window);
            let recent_slice = &recent_conv[recent_conv.len() - window..];
            base.extend_from_slice(recent_slice);
            drop(state_guard);
            return base;
        }
        drop(state_guard);

        // ── 3. No summary yet — check if we need to compact ──────────────────
        let total_tokens = tokens_in_messages(&base) + tokens_in_messages(&all_conv);

        if total_tokens < self.token_budget() {
            // Under budget — return the full conversation.
            base.extend(all_conv);
            return base;
        }

        // ── 4. Over budget — compact asynchronously, return window for now ────
        // We compact the older portion and keep the most recent window verbatim.
        let window = all_conv.len().min(self.recent_window);
        let compaction_slice = &all_conv[..all_conv.len() - window];
        let recent_slice = &all_conv[all_conv.len() - window..];

        // Kick off compaction without holding the state lock.
        let events_summarised = all_conv.len() - window;
        self.compact(compaction_slice, events_summarised).await;

        // Re-assemble with the newly stored summary.
        let state_guard = self.state.lock().await;
        if let Some(ref state) = *state_guard {
            base.push(ChatMessage::system(format!(
                "[Earlier conversation summary]\n{}",
                state.summary
            )));
        } else {
            // Compaction failed — inject a notice and continue with the window.
            base.push(ChatMessage::system(
                "[Note: context was too long but compaction failed. \
                 Showing recent messages only.]"
                    .to_string(),
            ));
        }
        drop(state_guard);

        base.extend_from_slice(recent_slice);
        base
    }
}
