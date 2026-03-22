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

/// Approximate token count for a string using a word-count heuristic.
fn estimate_tokens(s: &str) -> usize {
    let words = s.split_whitespace().count();
    ((words as f64) * 1.3).ceil() as usize + 4
}

fn messages_token_count(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .map(|m| {
            let content = m.content.as_deref().map(estimate_tokens).unwrap_or(0);
            let tool_calls: usize = m
                .tool_calls
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .map(|tc| {
                    estimate_tokens(&tc.function.arguments)
                        + estimate_tokens(&tc.function.name)
                        + 4
                })
                .sum();
            content + tool_calls + 4
        })
        .sum()
}

fn split_by_role(messages: &[ChatMessage]) -> (Vec<ChatMessage>, Vec<ChatMessage>) {
    let (sys, conv): (Vec<_>, Vec<_>) = messages
        .iter()
        .cloned()
        .partition(|m| m.role == Role::System);
    (sys, conv)
}

// ── Internal state ────────────────────────────────────────────────────────────

struct SummaryState {
    summary: String,
    /// Number of conversation messages that have been replaced by this summary.
    summarized_count: usize,
}

// ── SummarizingContextAssembler ───────────────────────────────────────────────

/// Wraps any [`ContextAssembler`] and keeps the assembled context within a
/// token budget by summarizing old conversation history with the LLM.
///
/// When the token estimate of the assembled messages exceeds
/// `max_tokens × threshold`, the oldest conversation messages (everything
/// except the most recent `keep_fraction`) are summarized and the originals
/// are optionally archived to `archive_dir/<session_id>.md`.
pub struct SummarizingContextAssembler {
    inner: Arc<dyn ContextAssembler>,
    llm: Arc<dyn LlmProvider>,
    /// Hard token budget for the entire context.
    pub max_tokens: usize,
    /// Fraction of `max_tokens` at which summarization is triggered (default 0.85).
    pub threshold: f64,
    /// Fraction of conversation messages to keep verbatim after summarizing (default 0.10).
    pub keep_fraction: f64,
    /// Session identifier used as the archive filename.
    pub session_id: String,
    /// Directory where archived conversation history is written.
    /// `None` disables archiving.
    pub archive_dir: Option<PathBuf>,
    state: Mutex<Option<SummaryState>>,
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
            keep_fraction: 0.10,
            session_id: session_id.into(),
            archive_dir: None,
            state: Mutex::new(None),
        }
    }

    /// Set the directory where archived (summarized-away) conversation history is written.
    ///
    /// Each session writes a single markdown file: `<dir>/<session_id>.md`.
    pub fn with_archive_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.archive_dir = Some(dir.into());
        self
    }

    async fn do_summarize(&self, to_summarize: &[ChatMessage]) -> String {
        let text: String = to_summarize
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::User => "User",
                    Role::Assistant => "Assistant",
                    Role::Tool => "Tool",
                    Role::System => "System",
                };
                format!(
                    "{role}: {}\n",
                    m.content.as_deref().unwrap_or("(tool calls)")
                )
            })
            .collect();

        let prompt = format!(
            "Summarize the following conversation history. \
             Your summary MUST start with a \"User instructions\" bullet list that quotes, \
             verbatim, every short instruction or correction the user gave. \
             After the bullet list, write a concise narrative preserving all important \
             context, decisions, and results.\n\n{text}"
        );

        match self
            .llm
            .complete(vec![ChatMessage::user(prompt)], vec![])
            .await
        {
            Ok(resp) => resp.content.unwrap_or_else(|| "(empty summary)".into()),
            Err(e) => {
                warn!("context summarization failed: {e}");
                format!("[Summary unavailable: {e}]")
            }
        }
    }

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
        let _ = tokio::fs::write(&path, text).await;
        debug!("archived conversation history to {}", path.display());
    }
}

#[async_trait]
impl ContextAssembler for SummarizingContextAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        if self.max_tokens == 0 {
            return self.inner.assemble(context).await;
        }

        let mut messages = self.inner.assemble(context).await;
        let (sys_msgs, conv_msgs) = split_by_role(&messages);
        let budget = (self.max_tokens as f64 * self.threshold) as usize;

        // If we already have a cached summary, apply it to the current messages.
        {
            let state = self.state.lock().await;
            if let Some(ref s) = *state {
                if conv_msgs.len() > s.summarized_count {
                    let summary_msg = ChatMessage::system(format!(
                        "<conversation_summary>\n\
                         The following is a summary of the earlier conversation:\n\n\
                         {}\n\
                         </conversation_summary>",
                        s.summary
                    ));
                    messages.clear();
                    messages.extend(sys_msgs.clone());
                    messages.push(summary_msg);
                    messages.extend_from_slice(&conv_msgs[s.summarized_count..]);
                }
                return messages;
            }
        }

        // Not yet over budget — return as-is.
        if messages_token_count(&messages) < budget {
            return messages;
        }

        // Decide the cutoff: keep the most recent `keep_fraction` of conversation messages.
        let keep = ((conv_msgs.len() as f64 * self.keep_fraction).ceil() as usize).max(4);
        let cutoff = conv_msgs.len().saturating_sub(keep);
        if cutoff == 0 {
            return messages;
        }

        let to_summarize = &conv_msgs[..cutoff];
        let to_keep = &conv_msgs[cutoff..];

        self.archive_to_file(to_summarize).await;
        let summary = self.do_summarize(to_summarize).await;

        {
            let mut state = self.state.lock().await;
            *state = Some(SummaryState {
                summary: summary.clone(),
                summarized_count: cutoff,
            });
        }

        let summarized_count = to_summarize.len();
        let retained_count = to_keep.len();
        debug!(
            summarized = summarized_count,
            retained = retained_count,
            session = %self.session_id,
            "context summarized"
        );

        let summary_msg = ChatMessage::system(format!(
            "<conversation_summary>\n\
             The following is a summary of the earlier conversation:\n\n\
             {summary}\n\
             </conversation_summary>"
        ));

        messages.clear();
        messages.extend(sys_msgs);
        messages.push(summary_msg);
        messages.extend_from_slice(to_keep);
        messages
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
