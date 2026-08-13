//! Context assembler that automatically summarizes old conversation history
//! when the assembled context approaches a token budget.
//!
//! # Overview
//!
//! [`SummarizingContextAssembler`] wraps any [`ContextAssembler`]. On each
//! call to `assemble`, it checks whether the resulting message list exceeds a
//! configurable token threshold. When it does, the oldest conversation messages
//! are summarized with an LLM call, stored as a brief system block, and the
//! original messages are archived to disk (optionally).
//!
//! Summarization is **incremental**: if the summary plus the remaining recent
//! messages still exceeds the budget (e.g. after a very long session), the
//! assembler extends the existing summary to cover more messages rather than
//! failing.  The most recent [`SummarizingContextAssembler::keep_recent`]
//! conversation messages are always kept verbatim.
//!
//! # Usage
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use eventage::agent::DefaultContextAssembler;
//! # use eventage::SummarizingContextAssembler;
//! # let inner: Arc<DefaultContextAssembler> = todo!();
//! # let llm: Arc<dyn eventage::llm::LlmProvider> = todo!();
//! let assembler = SummarizingContextAssembler::new(inner, llm, 32_000, "my-session");
//! ```

use async_trait::async_trait;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, warn};

use crate::agent::context::{AssemblyContext, ContextAssembler};
use crate::event::kinds;
use crate::llm::types::{ChatMessage, Role};
use crate::llm::LlmProvider;

// ── Token estimation ──────────────────────────────────────────────────────────

use super::tokens::TokenCalibration;

fn split_by_role(messages: &[ChatMessage]) -> (Vec<ChatMessage>, Vec<ChatMessage>) {
    let (sys, conv): (Vec<_>, Vec<_>) = messages
        .iter()
        .cloned()
        .partition(|m| m.role == Role::System);
    (sys, conv)
}

/// Render a single `ChatMessage` to a human-readable line for the summarization
/// prompt.  Tool calls and tool results are serialized so the summarizer LLM
/// sees the actual data rather than a placeholder.
fn message_to_text(m: &ChatMessage) -> String {
    match m.role {
        Role::System => String::new(), // system messages are not part of the conversation narrative
        Role::User => {
            let text = m.content.as_deref().unwrap_or("").trim();
            if text.is_empty() {
                String::new()
            } else {
                format!("User: {text}\n")
            }
        }
        Role::Assistant => {
            let mut parts: Vec<String> = Vec::new();
            if let Some(text) = m.content.as_deref() {
                let t = text.trim();
                if !t.is_empty() {
                    parts.push(t.to_string());
                }
            }
            if let Some(calls) = &m.tool_calls {
                for tc in calls {
                    // Truncate very long arguments to keep the prompt manageable.
                    let args = truncate_str(&tc.function.arguments, 200);
                    parts.push(format!("[called {}({})]", tc.function.name, args));
                }
            }
            if parts.is_empty() {
                String::new()
            } else {
                format!("Assistant: {}\n", parts.join(" "))
            }
        }
        Role::Tool => {
            let result = m.content.as_deref().unwrap_or("").trim();
            format!("Tool result: {}\n", truncate_str(result, 300))
        }
    }
}

fn truncate_str(s: &str, max_chars: usize) -> &str {
    if s.chars().count() <= max_chars {
        return s;
    }
    let mut end = max_chars.min(s.len());
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

// ── Internal state ────────────────────────────────────────────────────────────

struct SummaryState {
    summary: String,
    /// Number of conversation messages (from the head of conv_msgs) that have
    /// been folded into this summary.
    summarized_count: usize,
}

// ── SummarizingContextAssembler ───────────────────────────────────────────────

/// Wraps any [`ContextAssembler`] and keeps the assembled context within a
/// token budget by summarizing old conversation history with the LLM.
///
/// When the token estimate of the assembled messages exceeds
/// `max_tokens × threshold`, the oldest conversation messages are summarized.
/// The most recent [`keep_recent`](Self::keep_recent) conversation messages are
/// always kept verbatim.
///
/// If the summary + recent messages still exceeds the budget (very long
/// sessions), the assembler extends the summary incrementally rather than
/// failing or returning an over-budget context.
pub struct SummarizingContextAssembler {
    inner: Arc<dyn ContextAssembler>,
    llm: Arc<dyn LlmProvider>,
    /// Hard token budget for the entire context.
    pub max_tokens: usize,
    /// Fraction of `max_tokens` at which summarization is triggered (default 0.85).
    pub threshold: f64,
    /// Minimum number of recent conversation messages to keep verbatim (default 20).
    ///
    /// Kept as a fixed count rather than a fraction so behaviour is predictable
    /// regardless of total conversation length.
    pub keep_recent: usize,
    /// Session identifier used as the archive filename.
    pub session_id: String,
    /// Directory where archived conversation history is written.
    /// `None` disables archiving.
    pub archive_dir: Option<PathBuf>,
    state: Mutex<Option<SummaryState>>,
    /// Learns the estimator's error from real provider usage.
    calibration: Arc<TokenCalibration>,
}

impl SummarizingContextAssembler {
    /// Create a new summarizing assembler.
    ///
    /// * `inner` — the wrapped assembler whose output is checked against the budget
    /// * `llm` — provider used to generate summaries
    /// * `max_tokens` — total token budget; `0` disables summarization entirely
    /// * `session_id` — used as the archive filename stem
    pub fn new(
        inner: Arc<dyn ContextAssembler>,
        llm: Arc<dyn LlmProvider>,
        max_tokens: usize,
        session_id: impl Into<String>,
    ) -> Self {
        Self {
            inner,
            llm,
            max_tokens,
            threshold: 0.85,
            keep_recent: 20,
            session_id: session_id.into(),
            archive_dir: None,
            calibration: Arc::new(TokenCalibration::new()),
            state: Mutex::new(None),
        }
    }

    /// Share a [`TokenCalibration`] with other components so they learn from
    /// the same samples.
    pub fn with_calibration(mut self, calibration: Arc<TokenCalibration>) -> Self {
        self.calibration = calibration;
        self
    }

    /// The calibration this assembler is using.
    pub fn calibration(&self) -> Arc<TokenCalibration> {
        Arc::clone(&self.calibration)
    }

    /// Set the directory where archived (summarized-away) conversation history
    /// is written.  Each summarization pass appends to
    /// `<dir>/<session_id>.md`.
    pub fn with_archive_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.archive_dir = Some(dir.into());
        self
    }

    /// Summarize `new_messages`, optionally extending `existing_summary`.
    ///
    /// When `existing_summary` is `Some`, the prompt asks the LLM to extend the
    /// prior summary rather than starting fresh, so important earlier facts are
    /// not lost across multiple compression passes.
    async fn do_summarize(
        &self,
        existing_summary: Option<&str>,
        new_messages: &[ChatMessage],
    ) -> String {
        let new_text: String = new_messages.iter().map(message_to_text).collect();

        let prompt = match existing_summary {
            Some(prev) => format!(
                "You have a summary of earlier conversation history:\n\n\
                 {prev}\n\n\
                 Extend this summary to also cover the following new messages. \
                 Your updated summary MUST start with a \"User instructions\" bullet \
                 list that quotes, verbatim, every short instruction or correction \
                 the user gave (including any from the previous summary). \
                 After the bullet list, write a concise narrative preserving all \
                 important context, decisions, and results.\n\n{new_text}"
            ),
            None => format!(
                "Summarize the following conversation history. \
                 Your summary MUST start with a \"User instructions\" bullet list \
                 that quotes, verbatim, every short instruction or correction the \
                 user gave. \
                 After the bullet list, write a concise narrative preserving all \
                 important context, decisions, and results.\n\n{new_text}"
            ),
        };

        match self
            .llm
            .complete(vec![ChatMessage::user(prompt)], vec![])
            .await
        {
            Ok(resp) => resp.content.unwrap_or_else(|| "(empty summary)".into()),
            Err(e) => {
                warn!("context summarization failed: {e}");
                // Fall back to the existing summary so we don't lose it.
                existing_summary
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("[Summary unavailable: {e}]"))
            }
        }
    }

    /// Append `messages` to the archive file (if archiving is enabled).
    async fn archive_to_file(&self, messages: &[ChatMessage]) {
        let dir = match &self.archive_dir {
            Some(d) => d.clone(),
            None => return,
        };
        if tokio::fs::create_dir_all(&dir).await.is_err() {
            return;
        }
        let path = dir.join(format!("{}.md", self.session_id));
        let text: String = messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::User => "User",
                    Role::Assistant => "Assistant",
                    Role::Tool => "Tool",
                    Role::System => "System",
                };
                format!(
                    "## {role}\n{}\n\n",
                    m.content.as_deref().unwrap_or("(tool calls)")
                )
            })
            .collect();
        // Append — each summarization pass adds to the archive so the full
        // history is preserved even across multiple compression rounds.
        use tokio::io::AsyncWriteExt;
        if let Ok(mut file) = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
        {
            let _ = file.write_all(text.as_bytes()).await;
            debug!("archived conversation history to {}", path.display());
        }
    }

    /// Build the candidate message list from system messages, an optional
    /// summary block, and the unsummarized tail of conversation messages.
    fn build_candidate(
        sys_msgs: &[ChatMessage],
        state: Option<&SummaryState>,
        conv_msgs: &[ChatMessage],
    ) -> Vec<ChatMessage> {
        let summarized_count = state.map(|s| s.summarized_count).unwrap_or(0);
        let mut msgs: Vec<ChatMessage> = sys_msgs.to_vec();
        if let Some(s) = state {
            if s.summarized_count > 0 {
                msgs.push(ChatMessage::system(format!(
                    "<conversation_summary>\n\
                     The following is a summary of the earlier conversation:\n\n\
                     {}\n\
                     </conversation_summary>",
                    s.summary
                )));
            }
        }
        msgs.extend_from_slice(&conv_msgs[summarized_count..]);
        msgs
    }
}

#[async_trait]
impl ContextAssembler for SummarizingContextAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        if self.max_tokens == 0 {
            return self.inner.assemble(context).await;
        }

        let messages = self.inner.assemble(context).await;
        // Learn from the provider's real prompt-token counts before deciding.
        self.calibration.observe_events(context.events);
        let (sys_msgs, conv_msgs) = split_by_role(&messages);
        let budget = (self.max_tokens as f64 * self.threshold) as usize;

        // Incremental compression loop.  Each iteration either returns (under
        // budget) or extends the summary to cover more messages (over budget).
        // Terminates in at most 2 iterations for typical sessions; a third can
        // only occur if the keep_recent window itself exceeds the budget, in
        // which case we return as-is (best effort).
        loop {
            let candidate = {
                let state = self.state.lock().await;
                Self::build_candidate(&sys_msgs, state.as_ref(), &conv_msgs)
            };

            if self.calibration.count(&candidate) < budget {
                return candidate;
            }

            // Still over budget — determine how many more messages to fold in.
            let current_summarized = {
                let state = self.state.lock().await;
                state.as_ref().map(|s| s.summarized_count).unwrap_or(0)
            };

            // The new cutoff keeps keep_recent messages verbatim.
            let new_cutoff = conv_msgs.len().saturating_sub(self.keep_recent);

            if new_cutoff <= current_summarized {
                // Can't compress further without touching the recent window.
                // Return what we have — last-resort fallback.
                warn!(
                    session = %self.session_id,
                    keep_recent = self.keep_recent,
                    "context still over budget after max summarization; returning as-is"
                );
                return candidate;
            }

            let to_summarize = &conv_msgs[current_summarized..new_cutoff];

            // Retrieve existing summary before releasing the lock.
            let existing_summary = {
                let state = self.state.lock().await;
                state.as_ref().map(|s| s.summary.clone())
            };

            self.archive_to_file(to_summarize).await;

            let new_summary = self
                .do_summarize(existing_summary.as_deref(), to_summarize)
                .await;

            debug!(
                session = %self.session_id,
                prev_summarized = current_summarized,
                new_summarized = new_cutoff,
                "context summarized"
            );

            {
                let mut state = self.state.lock().await;
                *state = Some(SummaryState {
                    summary: new_summary,
                    summarized_count: new_cutoff,
                });
            }
            // Loop: re-check budget with the updated summary.
        }
    }
}

// ── AGENT_CONTEXT_SUMMARIZED event helper ─────────────────────────────────────

/// Build the payload for an [`AGENT_CONTEXT_SUMMARIZED`](kinds::AGENT_CONTEXT_SUMMARIZED)
/// event. Callers that have access to the EventBus can publish this after detecting
/// that summarization occurred.
pub fn context_summarized_payload(
    summary_len: usize,
    summarized_events: usize,
    retained_events: usize,
) -> serde_json::Value {
    serde_json::json!({
        "summary_len": summary_len,
        "summarized_events": summarized_events,
        "retained_events": retained_events,
    })
}

pub use kinds::AGENT_CONTEXT_SUMMARIZED;
