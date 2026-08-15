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
//! failing.
//!
//! How much conversation survives a pass is decided in *tokens*, not in
//! messages — twenty messages can be five thousand tokens of chat or a hundred
//! and fifty thousand tokens of file dumps, and only one of those fits. The
//! cut itself is then snapped to a turn boundary so a tool result is never
//! separated from the call that produced it.
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
use crate::event::{kinds, Event};
use crate::llm::content::{ContentPart, ImageSource};
use crate::llm::types::{ChatMessage, Role};
use crate::llm::LlmProvider;

// ── Token estimation ──────────────────────────────────────────────────────────

use super::prompts::{names, PromptLibrary};
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

/// How much of a message the assembly record carries.
///
/// Generous, because a trimmed preview answers "which messages" and not "what
/// did it actually say" — and the second is the question people have when a
/// long session goes wrong. Beyond this the text is cut and its true length
/// reported, so a 40 KB file dump does not multiply the log by the length of
/// the conversation while a paragraph of reasoning is still readable in full.
const MAX_RECORDED_CHARS: usize = 4_000;

/// What a message says, at length, for the assembly record.
///
/// A message carries its text in one of three places and the record has to
/// look in all of them, or a whole class of message shows up blank: `content`
/// for the ordinary case, `parts` once an image is attached, and `tool_calls`
/// for a call, whose content is empty but whose *call* is what identifies it.
fn recorded_text(m: &ChatMessage) -> (String, Option<usize>) {
    let mut sections: Vec<String> = Vec::new();

    if let Some(text) = &m.content {
        if !text.trim().is_empty() {
            sections.push(text.clone());
        }
    }

    // A multimodal message keeps its text here instead. Images are named
    // rather than dumped — a base64 payload is not something anyone reads,
    // but knowing one was sent is exactly what this panel is for.
    for part in &m.parts {
        match part {
            ContentPart::Text { text } if !text.trim().is_empty() => sections.push(text.clone()),
            ContentPart::Text { .. } => {}
            ContentPart::Image { source } => sections.push(match source {
                ImageSource::Url { url } => format!("[image: {url}]"),
                ImageSource::Base64 { media_type, data } => format!(
                    "[image: {media_type}, {} KB inline]",
                    (data.len() * 3 / 4).div_ceil(1024)
                ),
            }),
        }
    }

    if let Some(calls) = &m.tool_calls {
        for call in calls {
            sections.push(format!(
                "{}({})",
                call.function.name, call.function.arguments
            ));
        }
    }

    let body = sections.join("\n\n");
    let total = body.chars().count();
    if total <= MAX_RECORDED_CHARS {
        (body, None)
    } else {
        (
            truncate_str(&body, MAX_RECORDED_CHARS).to_string(),
            Some(total),
        )
    }
}

/// The first `max_chars` *characters* of `s`.
///
/// Counting bytes here would silently return a short record for any message
/// with non-ASCII in it, which is most of them once code or prose is involved.
fn truncate_str(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((end, _)) => &s[..end],
        None => s,
    }
}

// ── Internal state ────────────────────────────────────────────────────────────

pub struct SummaryState {
    pub summary: String,
    /// Number of conversation messages (from the head of conv_msgs) that have
    /// been folded into this summary.
    pub summarized_count: usize,
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
    /// Fraction of `max_tokens` to compress *down to* once summarization runs
    /// (default 0.5).
    ///
    /// Compressing to just under [`threshold`](Self::threshold) would put the
    /// context back over it within a few turns, and every pass costs an LLM
    /// call plus the prompt-cache tail after the cut. Compressing to a
    /// low-water mark buys a long stretch before the next pass.
    pub target: f64,
    /// Floor on how many recent conversation messages stay verbatim
    /// (default 20).
    ///
    /// Retention is chosen by token budget; this is only a backstop so a
    /// pathologically small budget cannot fold away the turn in progress.
    pub keep_recent: usize,
    /// Fraction of `max_tokens` above which summarization runs even in the
    /// middle of a task, because overflowing is worse than a bad cut point
    /// (default 0.95).
    pub hard_threshold: f64,
    /// Session identifier used as the archive filename.
    pub session_id: String,
    /// Directory where archived conversation history is written.
    /// `None` disables archiving.
    pub archive_dir: Option<PathBuf>,
    state: Mutex<Option<SummaryState>>,
    /// Learns the estimator's error from real provider usage.
    calibration: Arc<TokenCalibration>,
    /// Where summaries and assembly records are published.
    ///
    /// Without one the assembler still works, but its compaction is invisible
    /// and lives only in this process.
    bus: Option<crate::bus::EventBus>,
    /// How this assembler asks for a summary.
    ///
    /// Held by name rather than inline so an application can change the
    /// instruction — to preserve different things, or to answer in another
    /// language — without forking the assembler.
    prompts: PromptLibrary,
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
            hard_threshold: 0.95,
            target: 0.5,
            keep_recent: 20,
            session_id: session_id.into(),
            archive_dir: None,
            calibration: Arc::new(TokenCalibration::new()),
            state: Mutex::new(None),
            prompts: PromptLibrary::with_defaults(),
            bus: None,
        }
    }

    /// Set the fraction of the budget above which summarization runs even
    /// mid-task.
    pub fn with_hard_threshold(mut self, fraction: f64) -> Self {
        self.hard_threshold = fraction;
        self
    }

    /// Set the fraction of the budget that a summarization pass compresses
    /// down to.
    pub fn with_target(mut self, fraction: f64) -> Self {
        self.target = fraction;
        self
    }

    /// Publish compaction decisions onto `bus`.
    ///
    /// This is what makes compaction durable and inspectable rather than a
    /// private fact of one process: the summary becomes an event, so it
    /// survives a restart, shows up in a trace, and can be replaced by
    /// publishing a newer one.
    pub fn with_bus(mut self, bus: crate::bus::EventBus) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Use a different prompt library, e.g. one whose summarization
    /// instruction has been replaced.
    pub fn with_prompts(mut self, prompts: PromptLibrary) -> Self {
        self.prompts = prompts;
        self
    }

    /// The prompts this assembler will send.
    pub fn prompts(&self) -> PromptLibrary {
        self.prompts.clone()
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
            Some(prev) => self.prompts.render(
                names::SUMMARIZE_EXTEND,
                &[("summary", prev), ("conversation", &new_text)],
            ),
            None => self
                .prompts
                .render(names::SUMMARIZE_FRESH, &[("conversation", &new_text)]),
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

    /// Note what this request was made of, message by message.
    ///
    /// Counts alone are not transparency. "sixteen verbatim messages" does
    /// not tell you *which* sixteen, and the question people actually have
    /// when a long session goes strange is whether the thing they said an
    /// hour ago is still in front of the model or has been folded into a
    /// summary. So the record lists every message: what it is, how big, where
    /// it came from, and enough of its text to recognise.
    ///
    /// Previews rather than contents — the full text is already in the event
    /// log, and copying it here every step would multiply the log by the
    /// length of the conversation.
    fn record_assembly(
        &self,
        messages: &[ChatMessage],
        summary: Option<&SummaryState>,
        conv: &[ChatMessage],
    ) {
        let Some(bus) = &self.bus else { return };

        let summarized = summary.map(|s| s.summarized_count).unwrap_or(0);
        // The verbatim tail begins this many messages into the assembled
        // list: everything before it is system prefix and summary.
        let head = messages.len().saturating_sub(conv.len() - summarized);

        let manifest: Vec<serde_json::Value> = messages
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let tokens = self.calibration.count(std::slice::from_ref(m));
                let text = m.content.as_deref().unwrap_or("");
                let source = if m.role == Role::System {
                    if text.contains("<conversation_summary>") {
                        "summary"
                    } else {
                        "system"
                    }
                } else if text.starts_with("[cleared by harness:") {
                    "cleared"
                } else if i >= head {
                    "verbatim"
                } else {
                    "other"
                };
                let (text, truncated_from) = recorded_text(m);
                serde_json::json!({
                    "index": i,
                    "role": match m.role {
                        Role::System => "system",
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::Tool => "tool",
                    },
                    "tokens": tokens,
                    "source": source,
                    "text": text,
                    "truncated_from": truncated_from,
                })
            })
            .collect();

        let system_tokens: usize = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .map(|m| self.calibration.count(std::slice::from_ref(m)))
            .sum();

        bus.broadcast(Event::new(
            kinds::CONTEXT_ASSEMBLED,
            serde_json::json!({
                "session": self.session_id,
                "messages": messages.len(),
                "total_tokens": self.calibration.count(messages),
                "system_tokens": system_tokens,
                "verbatim_messages": conv.len().saturating_sub(summarized),
                "summarized_messages": summarized,
                "summary_tokens": summary
                    .map(|s| s.summary.len().div_ceil(4))
                    .unwrap_or(0),
                "compacted": summary.is_some(),
                "budget": self.max_tokens,
                "manifest": manifest,
            }),
        ));
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

        // Prefer to compress between tasks. Mid-task we hold off until the
        // context is close enough to the ceiling that running out would be
        // the worse outcome.
        let budget = if at_task_boundary(context.events) {
            (self.max_tokens as f64 * self.threshold) as usize
        } else {
            (self.max_tokens as f64 * self.hard_threshold) as usize
        };

        // The log seeds the state; the loop then tracks its own progress.
        //
        // Reading the log on every pass looks tidier and hangs: events
        // published during this call are not in the slice we were handed, so
        // the loop would never see its own work and would compact forever.
        let mut state: Option<SummaryState> = match summary_from_log(context.events) {
            Some(from_log) => Some(from_log),
            None => self.state.lock().await.as_ref().map(|s| SummaryState {
                summary: s.summary.clone(),
                summarized_count: s.summarized_count,
            }),
        };

        // Incremental compression loop.  Each iteration either returns (under
        // budget) or extends the summary to cover more messages (over budget).
        // Terminates in at most 2 iterations for typical sessions; a third can
        // only occur if the keep_recent window itself exceeds the budget, in
        // which case we return as-is (best effort).
        loop {
            let candidate = Self::build_candidate(&sys_msgs, state.as_ref(), &conv_msgs);

            if self.calibration.count(&candidate) < budget {
                self.record_assembly(&candidate, state.as_ref(), &conv_msgs);
                return candidate;
            }

            // Still over budget — determine how many more messages to fold in.
            // From the loop's own state, which the log seeded. Re-reading the
            // private field here left this at zero whenever a bus was present
            // — the guard below never fired and the loop compacted forever.
            let (current_summarized, existing_summary) = match &state {
                Some(s) => (s.summarized_count, Some(s.summary.clone())),
                None => (0, None),
            };

            let retain = retention_budget(
                self.max_tokens,
                self.target,
                &sys_msgs,
                existing_summary.as_deref(),
                &self.calibration,
            );
            let new_cutoff = choose_cutoff(
                &conv_msgs,
                current_summarized,
                retain,
                self.keep_recent,
                &self.calibration,
            );

            if new_cutoff <= current_summarized {
                // Can't compress further without touching the recent window.
                // Return what we have — last-resort fallback.
                warn!(
                    session = %self.session_id,
                    keep_recent = self.keep_recent,
                    "context still over budget after max summarization; returning as-is"
                );
                self.record_assembly(&candidate, state.as_ref(), &conv_msgs);
                return candidate;
            }

            let to_summarize = &conv_msgs[current_summarized..new_cutoff];

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

            // Record it. The log is where the summary lives from here on, so
            // a restart finds it, a trace shows it, and replacing it is a
            // matter of publishing a newer one.
            if let Some(bus) = &self.bus {
                let _ = bus
                    .publish(Event::new(
                        kinds::AGENT_CONTEXT_SUMMARIZED,
                        serde_json::json!({
                            "summary": new_summary,
                            "summarized_count": new_cutoff,
                            "replaced_messages": new_cutoff - current_summarized,
                            "session": self.session_id,
                        }),
                    ))
                    .await;
            } else {
                // No bus: keep it in this process so the assembler still
                // works, at the cost of losing it on restart.
                *self.state.lock().await = Some(SummaryState {
                    summary: new_summary.clone(),
                    summarized_count: new_cutoff,
                });
            }

            state = Some(SummaryState {
                summary: new_summary,
                summarized_count: new_cutoff,
            });
            // Loop: re-check budget with the updated summary.
        }
    }
}

/// Room reserved for a summary we have not generated yet.
const MIN_SUMMARY_RESERVE: usize = 1_024;

/// Token room for verbatim conversation once a summarization pass finishes.
///
/// A pass is not free: it costs an LLM call, and because it removes messages
/// from the middle of the prompt it invalidates every cached token after the
/// cut. Landing just under the trigger means paying both again a few turns
/// later, so we aim at a low-water mark instead — the same total loss, spread
/// over far fewer passes.
fn retention_budget(
    max_tokens: usize,
    target: f64,
    sys_msgs: &[ChatMessage],
    summary: Option<&str>,
    calibration: &TokenCalibration,
) -> usize {
    let low_water = (max_tokens as f64 * target) as usize;
    // The summary is about to grow, so reserve room for a larger one than the
    // one we are holding.
    let summary_room = summary
        .map(|s| calibration.count(&[ChatMessage::system(s)]) * 2)
        .unwrap_or(0)
        .max(MIN_SUMMARY_RESERVE);
    low_water.saturating_sub(calibration.count(sys_msgs) + summary_room)
}

/// How many leading conversation messages to fold into the summary.
///
/// The raw cut is wherever the retention budget runs out, walking back from
/// the newest message. That point lands wherever the arithmetic happens to
/// put it — between an assistant's tool call and the result it is waiting on,
/// or part-way through a tool loop — so it is snapped to a real boundary
/// before use. Folding half a turn strands tool results whose originating
/// call is on the other side of the cut, which providers reject outright, and
/// hands the model half a story besides.
fn choose_cutoff(
    conv: &[ChatMessage],
    already: usize,
    retain_tokens: usize,
    keep_recent: usize,
    calibration: &TokenCalibration,
) -> usize {
    // The newest messages survive however tight the budget gets.
    let ceiling = conv.len().saturating_sub(keep_recent.max(1));
    if ceiling <= already {
        return already;
    }

    // Walk back from the newest message, keeping whatever the budget affords.
    let mut used = calibration.count(&conv[ceiling..]);
    let mut raw = ceiling;
    for i in (already..ceiling).rev() {
        let cost = calibration.count(std::slice::from_ref(&conv[i]));
        if used + cost > retain_tokens {
            break;
        }
        used += cost;
        raw = i;
    }

    // Cut where a turn begins, or do not cut at all.
    //
    // This was four ranked fallbacks — nearest turn start after, then before,
    // then any point not opening on an orphaned tool result, in both
    // directions. Each was defensible on its own and the combination was
    // wrong twice, because a chain of "if that fails, try" is only as clear
    // as its least likely branch.
    //
    // Two independent choices, so four candidates rather than a ranking:
    //
    //                    at/after the budget point   before it
    //   turn boundary            (1) best              (2)
    //   any safe point           (3)                   (4)
    //
    // The column is "keep less than asked" versus "keep more than asked"; the
    // row is "cut where a turn begins" versus "cut anywhere that does not
    // strand a tool result from its call". A turn boundary is preferable
    // because neither side is left half-told, but a single long turn — one
    // prompt, forty tool steps — contains no second boundary, and refusing to
    // cut there would mean never compacting it at all. That case is why the
    // second row exists; the second column is why a budget too small for one
    // whole turn still makes progress.
    //
    // Collapsing this to one rule was tried and reverted: it left a long
    // single turn uncompactable. The shape is four branches because the
    // problem has two binary dimensions, not because it grew by accretion.
    let turn_start = |i: usize| conv.get(i).is_some_and(|m| m.role == Role::User);
    (raw..=ceiling)
        .find(|&i| turn_start(i))
        .or_else(|| {
            (already..=raw)
                .rev()
                .find(|&i| i > already && turn_start(i))
        })
        .or_else(|| (raw..=ceiling).find(|&i| starts_clean(conv, i)))
        .or_else(|| (already..=raw).rev().find(|&i| starts_clean(conv, i)))
        .unwrap_or(already)
}

/// Would the retained tail beginning at `i` stand on its own?
///
/// Only if it does not open with a tool result — that result's call would be
/// on the far side of the cut.
fn starts_clean(conv: &[ChatMessage], i: usize) -> bool {
    conv.get(i).is_none_or(|m| m.role != Role::Tool)
}

/// The newest summary recorded on the log, if any.
///
/// Reading this rather than a private field is what makes compaction
/// inspectable and overridable: whoever publishes the most recent
/// `agent.context.summarized` decides what the model sees, whether that is
/// this assembler, a different strategy, or a person who thought the summary
/// dropped something important.
pub fn summary_from_log(events: &[Event]) -> Option<SummaryState> {
    events
        .iter()
        .rev()
        .find(|e| e.kind == kinds::AGENT_CONTEXT_SUMMARIZED)
        .and_then(|e| {
            Some(SummaryState {
                summary: e.payload.get("summary")?.as_str()?.to_string(),
                summarized_count: e.payload.get("summarized_count")?.as_u64()? as usize,
            })
        })
}

/// Replace the summary the next assembly will use.
///
/// For when compaction lost something that mattered. The old summary stays in
/// the log — it is not edited, it is superseded — so the trace still shows
/// what was dropped and by whom.
pub async fn override_summary(
    bus: &crate::bus::EventBus,
    summary: impl Into<String>,
    summarized_count: usize,
) -> Result<(), crate::error::BusError> {
    bus.publish(Event::new(
        kinds::AGENT_CONTEXT_SUMMARIZED,
        serde_json::json!({
            "summary": summary.into(),
            "summarized_count": summarized_count,
            "source": "manual_override",
        }),
    ))
    .await
    .map(|_| ())
}

/// Is the agent between tasks rather than part-way through one?
///
/// Summarizing mid-task is destructive in a way that summarizing between
/// tasks is not: the details the agent is *currently* working from — the file
/// it just read, the error it is chasing — are exactly what a summary
/// discards, and it has no way to ask for them back. At a turn boundary the
/// working set is already spent, so the same compression costs far less.
///
/// A task is in flight while a cycle has started and not yet ended, so the
/// boundary is simply: no unmatched `agent.cycle.start`.
fn at_task_boundary(events: &[Event]) -> bool {
    let mut depth = 0i32;
    for event in events {
        match event.kind.as_str() {
            kinds::AGENT_CYCLE_START => depth += 1,
            kinds::AGENT_CYCLE_END => depth -= 1,
            _ => {}
        }
    }
    depth <= 0
}

// ── AGENT_CONTEXT_SUMMARIZED event helper ─────────────────────────────────────

/// Build the payload for an [`kinds::AGENT_CONTEXT_SUMMARIZED`]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::EventBus;
    use serde_json::json;

    fn ev(kind: &str) -> Event {
        Event::new(kind, json!({}))
    }

    /// An LLM that answers nothing useful but remembers what it was asked.
    struct Recorder {
        seen: Arc<std::sync::Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl LlmProvider for Recorder {
        fn model(&self) -> &str {
            "recorder"
        }

        async fn complete(
            &self,
            messages: Vec<ChatMessage>,
            _tools: Vec<crate::llm::types::ToolDefinition>,
        ) -> Result<crate::llm::types::LlmResponse, crate::llm::LlmError> {
            self.seen.lock().unwrap().push(
                messages
                    .iter()
                    .filter_map(|m| m.content.clone())
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
            Ok(crate::llm::types::LlmResponse {
                content: Some("a summary".into()),
                finish_reason: "stop".into(),
                ..Default::default()
            })
        }
    }

    #[tokio::test]
    async fn an_application_can_change_how_the_harness_asks_for_a_summary() {
        // The reason prompts are a registry rather than string literals: this
        // instruction should be changeable without forking this file.
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let llm = Arc::new(Recorder {
            seen: Arc::clone(&seen),
        });

        let mut events = vec![Event::new(kinds::USER_MESSAGE, json!({ "text": "go" }))];
        for i in 0..40 {
            events.push(Event::new(
                kinds::ASSISTANT_MESSAGE,
                json!({ "content": format!("a fairly wordy reply number {i} ").repeat(20) }),
            ));
        }
        events.push(Event::new(kinds::AGENT_CYCLE_END, json!({})));

        let prompts = PromptLibrary::with_defaults();
        prompts.set(
            names::SUMMARIZE_FRESH,
            "In Dutch, one sentence only:\n\n{conversation}",
        );

        let assembler = SummarizingContextAssembler::new(
            Arc::new(crate::agent::context::DefaultContextAssembler::without_system_prompt()),
            llm,
            600,
            "test",
        )
        .with_prompts(prompts);

        let ctx = AssemblyContext::new(&events);
        assembler.assemble(&ctx).await;

        let asked = seen.lock().unwrap();
        assert!(!asked.is_empty(), "summarization should have run");
        assert!(
            asked[0].starts_with("In Dutch, one sentence only:"),
            "the replaced prompt must be what reaches the model, got: {}",
            &asked[0][..asked[0].len().min(120)]
        );
        assert!(
            asked[0].contains("wordy reply"),
            "the conversation must still be substituted in"
        );
    }

    /// A conversation long enough to force compaction.
    fn wordy_session() -> Vec<Event> {
        let mut events = vec![Event::new(kinds::USER_MESSAGE, json!({ "text": "go" }))];
        for i in 0..40 {
            events.push(Event::new(
                kinds::ASSISTANT_MESSAGE,
                json!({ "content": format!("a fairly wordy reply number {i} ").repeat(20) }),
            ));
        }
        events.push(Event::new(kinds::AGENT_CYCLE_END, json!({})));
        events
    }

    fn compacting(bus: &EventBus, llm: Arc<dyn LlmProvider>) -> SummarizingContextAssembler {
        SummarizingContextAssembler::new(
            Arc::new(crate::agent::context::DefaultContextAssembler::without_system_prompt()),
            llm,
            600,
            "test",
        )
        .with_bus(bus.clone())
    }

    #[tokio::test]
    async fn a_summary_survives_reopening_the_session() {
        // It used to live in a field, so closing the app threw it away: the
        // reopened session either paid for a fresh summary or sent the whole
        // history it had already compacted. Putting it on the log fixes both.
        let bus = EventBus::new();
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let llm = Arc::new(Recorder {
            seen: Arc::clone(&seen),
        });

        for event in wordy_session() {
            bus.publish(event).await.unwrap();
        }
        let assembler = compacting(&bus, llm.clone());
        let events = bus.log().await;
        assembler.assemble(&AssemblyContext::new(&events)).await;
        assert_eq!(seen.lock().unwrap().len(), 1, "should have compacted once");

        // A different process, same log.
        let reopened = EventBus::new();
        reopened.restore_from(bus.log().await).await;
        let fresh = compacting(&reopened, llm.clone());
        let restored = reopened.log().await;
        let messages = fresh.assemble(&AssemblyContext::new(&restored)).await;

        assert_eq!(
            seen.lock().unwrap().len(),
            1,
            "the restored session must reuse the recorded summary, not buy another"
        );
        assert!(
            messages.iter().any(|m| m
                .content
                .as_deref()
                .is_some_and(|c| c.contains("conversation_summary"))),
            "the restored context should carry the summary"
        );
    }

    #[tokio::test]
    async fn a_person_can_replace_a_summary_that_dropped_something() {
        // The point of putting compaction on the log: when it loses something
        // that mattered, it can be corrected without restarting the session.
        let bus = EventBus::new();
        let llm = Arc::new(Recorder {
            seen: Arc::new(std::sync::Mutex::new(Vec::new())),
        });
        for event in wordy_session() {
            bus.publish(event).await.unwrap();
        }
        let assembler = compacting(&bus, llm);
        let events = bus.log().await;
        assembler.assemble(&AssemblyContext::new(&events)).await;

        override_summary(
            &bus,
            "The user's deploy key is in .secrets/, do not touch it.",
            30,
        )
        .await
        .unwrap();

        let events = bus.log().await;
        let messages = assembler.assemble(&AssemblyContext::new(&events)).await;
        assert!(
            messages.iter().any(|m| m
                .content
                .as_deref()
                .is_some_and(|c| c.contains("deploy key"))),
            "the override should be what the model sees"
        );
    }

    #[tokio::test]
    async fn superseding_a_summary_leaves_the_old_one_in_the_log() {
        // Corrections are additions, not edits: the trace should still show
        // what was dropped and by whom.
        let bus = EventBus::new();
        override_summary(&bus, "first attempt", 10).await.unwrap();
        override_summary(&bus, "second attempt", 20).await.unwrap();

        let events = bus.log().await;
        let recorded: Vec<&Event> = events
            .iter()
            .filter(|e| e.kind == kinds::AGENT_CONTEXT_SUMMARIZED)
            .collect();
        assert_eq!(recorded.len(), 2, "history is appended to, not rewritten");
        assert_eq!(
            summary_from_log(&events).unwrap().summary,
            "second attempt",
            "the newest one wins"
        );
    }

    #[tokio::test]
    async fn every_request_records_what_it_was_made_of() {
        // "What did the model actually see, and how much of it was a summary"
        // is not answerable from the event log alone — the log records what
        // happened, not what was selected to describe it.
        let bus = EventBus::new();
        let mut observed = bus.subscribe();
        let llm = Arc::new(Recorder {
            seen: Arc::new(std::sync::Mutex::new(Vec::new())),
        });
        for event in wordy_session() {
            bus.publish(event).await.unwrap();
        }
        let assembler = compacting(&bus, llm);
        let events = bus.log().await;
        assembler.assemble(&AssemblyContext::new(&events)).await;

        let mut record = None;
        // Drain what the subscriber saw; the assembly record is broadcast
        // after the durable events, so it arrives last.
        while let Ok(Some(event)) =
            tokio::time::timeout(std::time::Duration::from_millis(50), observed.recv()).await
        {
            if event.kind == kinds::CONTEXT_ASSEMBLED {
                record = Some(event);
                break;
            }
        }
        let record = record.expect("an assembly record should have been broadcast");
        assert_eq!(record.payload["compacted"], true);

        // Counts are not transparency: the record must name the messages.
        let manifest = record.payload["manifest"].as_array().unwrap();
        assert_eq!(
            manifest.len(),
            record.payload["messages"].as_u64().unwrap() as usize,
            "every message in the request should be listed"
        );
        assert!(
            manifest.iter().any(|m| m["source"] == "summary"),
            "the summary block should be identifiable: {manifest:?}"
        );
        assert!(
            manifest.iter().any(|m| m["source"] == "verbatim"),
            "so should the messages that survived verbatim"
        );
        for entry in manifest {
            assert!(entry["tokens"].as_u64().is_some(), "each needs a size");
            assert!(entry["role"].as_str().is_some());
        }
        assert!(
            manifest
                .iter()
                .any(|m| !m["text"].as_str().unwrap_or("").is_empty()),
            "and what each one actually said"
        );
        assert!(record.payload["summarized_messages"].as_u64().unwrap() > 0);
        assert!(record.payload["total_tokens"].as_u64().unwrap() > 0);
        assert!(record.payload["verbatim_messages"].as_u64().is_some());
    }

    #[test]
    fn a_long_message_is_recorded_at_length_and_says_what_it_cut() {
        // A ninety-character preview answers "which messages" but not "what
        // did it say", and the second is the question people actually have.
        let long = "x".repeat(MAX_RECORDED_CHARS + 500);
        let (text, truncated) = recorded_text(&ChatMessage::assistant(long.clone()));
        assert_eq!(text.chars().count(), MAX_RECORDED_CHARS);
        assert_eq!(truncated, Some(long.chars().count()));

        let short = ChatMessage::assistant("brief");
        assert_eq!(recorded_text(&short), ("brief".to_string(), None));
    }

    #[test]
    fn a_message_carrying_an_image_is_not_recorded_as_blank() {
        // Attaching an image moves the text into `parts` and leaves `content`
        // empty, so a record that reads only `content` shows the whole
        // message as "(no text)" — exactly the messages worth inspecting.
        let mut m = ChatMessage::user("");
        m.content = None;
        m.parts = vec![
            ContentPart::text("what is wrong with this screenshot?"),
            ContentPart::image_base64("image/png", "A".repeat(4096)),
        ];

        let (text, truncated) = recorded_text(&m);
        assert!(
            text.contains("what is wrong with this screenshot?"),
            "{text}"
        );
        assert!(
            text.contains("image/png"),
            "the image should be named: {text}"
        );
        assert!(!text.contains("AAAA"), "but not dumped: {text}");
        assert_eq!(truncated, None);
    }

    #[test]
    fn the_cut_counts_characters_not_bytes() {
        // A byte-indexed cut returns a third of the text for CJK and looks
        // like data loss in the panel.
        let wide = "\u{6c49}".repeat(MAX_RECORDED_CHARS + 100);
        let (text, truncated) = recorded_text(&ChatMessage::assistant(wide.clone()));
        assert_eq!(text.chars().count(), MAX_RECORDED_CHARS);
        assert_eq!(truncated, Some(wide.chars().count()));
    }

    #[test]
    fn a_tool_call_is_recorded_as_the_call_it_makes() {
        use crate::llm::types::{FunctionCall, ToolCall};
        let m = ChatMessage::assistant_with_tool_calls(
            Some("Let me look.".into()),
            vec![ToolCall {
                id: "c1".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "read_file".into(),
                    arguments: r#"{"path":"src/bus.rs"}"#.into(),
                },
                extra_content: None,
            }],
        );
        let (text, _) = recorded_text(&m);
        assert!(text.contains("Let me look."));
        assert!(
            text.contains(r#"read_file({"path":"src/bus.rs"})"#),
            "{text}"
        );
    }

    #[test]
    fn a_finished_turn_is_a_boundary() {
        assert!(at_task_boundary(&[]), "an empty log is idle");
        assert!(at_task_boundary(&[
            ev(kinds::USER_MESSAGE),
            ev(kinds::AGENT_CYCLE_START),
            ev(kinds::ASSISTANT_MESSAGE),
            ev(kinds::AGENT_CYCLE_END),
        ]));
    }

    #[test]
    fn a_running_turn_is_not_a_boundary() {
        // Mid tool-loop: started, tools running, no end yet.
        assert!(!at_task_boundary(&[
            ev(kinds::USER_MESSAGE),
            ev(kinds::AGENT_CYCLE_START),
            ev(kinds::TOOL_CALL_PROPOSED),
            ev(kinds::TOOL_RESULT),
        ]));
    }

    /// One turn: the user asks, the assistant calls a tool, the tool answers,
    /// the assistant replies. `bulk` pads the tool result so token budgets bite.
    fn turn(n: usize, bulk: usize) -> Vec<ChatMessage> {
        use crate::llm::types::{FunctionCall, ToolCall};
        vec![
            ChatMessage::user(format!("request {n}")),
            ChatMessage::assistant_with_tool_calls(
                None,
                vec![ToolCall {
                    id: format!("call{n}"),
                    kind: "function".into(),
                    function: FunctionCall {
                        name: "read_file".into(),
                        arguments: "{}".into(),
                    },
                    extra_content: None,
                }],
            ),
            ChatMessage::tool_result(format!("call{n}"), "x".repeat(bulk)),
            ChatMessage::assistant(format!("done {n}")),
        ]
    }

    fn conversation(turns: usize, bulk: usize) -> Vec<ChatMessage> {
        (0..turns).flat_map(|n| turn(n, bulk)).collect()
    }

    #[test]
    fn the_cut_lands_on_a_turn_boundary() {
        let cal = TokenCalibration::new();
        let conv = conversation(10, 4_000);

        // Sweep every plausible budget: whatever the arithmetic picks, the cut
        // must land where a turn begins, never part-way through one.
        for retain in (0..20_000).step_by(250) {
            let cut = choose_cutoff(&conv, 0, retain, 1, &cal);
            assert_eq!(
                conv[cut].role,
                Role::User,
                "cut at {cut} (retain={retain}) splits a turn"
            );
        }
    }

    #[test]
    fn a_tight_budget_still_cuts_cleanly_and_makes_progress() {
        let cal = TokenCalibration::new();
        let conv = conversation(10, 4_000);

        // Far too little room for even one turn.
        let cut = choose_cutoff(&conv, 0, 10, 1, &cal);
        assert!(cut > 0, "must fold something rather than give up");
        assert_eq!(
            conv[cut].role,
            Role::User,
            "should land on the start of a turn"
        );
    }

    #[test]
    fn keep_recent_is_a_floor_the_budget_cannot_cross() {
        let cal = TokenCalibration::new();
        let conv = conversation(10, 4_000);
        let cut = choose_cutoff(&conv, 0, 0, 8, &cal);
        assert!(
            cut <= conv.len() - 8,
            "at least keep_recent messages must survive"
        );
    }

    #[test]
    fn retention_aims_at_the_low_water_mark_not_the_trigger() {
        let cal = TokenCalibration::new();
        let sys = vec![ChatMessage::system("you are helpful")];

        // Compressing to 50% leaves far less standing than compressing to 85%,
        // which is the point: fewer passes, each buying a longer stretch.
        let low = retention_budget(100_000, 0.5, &sys, None, &cal);
        let shallow = retention_budget(100_000, 0.85, &sys, None, &cal);
        assert!(low < shallow);
        assert!(low > 0);

        // A budget swallowed entirely by the system prompt and summary reserve
        // yields no room rather than underflowing.
        assert_eq!(retention_budget(100, 0.5, &sys, None, &cal), 0);
    }

    #[test]
    fn a_bigger_summary_leaves_less_room_for_verbatim_history() {
        let cal = TokenCalibration::new();
        let sys = vec![ChatMessage::system("you are helpful")];
        let small = retention_budget(100_000, 0.5, &sys, Some("short"), &cal);
        let large = retention_budget(100_000, 0.5, &sys, Some(&"word ".repeat(5_000)), &cal);
        assert!(large < small, "summary growth must come out of retention");
    }

    #[test]
    fn boundary_survives_many_completed_turns() {
        let mut log = Vec::new();
        for _ in 0..3 {
            log.push(ev(kinds::USER_MESSAGE));
            log.push(ev(kinds::AGENT_CYCLE_START));
            log.push(ev(kinds::AGENT_CYCLE_END));
        }
        assert!(at_task_boundary(&log));
        // Opening a fourth turn closes the window again.
        log.push(ev(kinds::AGENT_CYCLE_START));
        assert!(!at_task_boundary(&log));
    }
}
