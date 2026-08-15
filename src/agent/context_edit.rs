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
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;
use tracing::debug;

use super::context::{AssemblyContext, ContextAssembler};
use super::tokens::TokenCalibration;
use crate::llm::types::{ChatMessage, Role};

/// Wraps a [`ContextAssembler`] and clears tool-result content once the
/// assembled context exceeds a token trigger.
///
/// Results are cleared by how much budget they hold, not by how old they are:
/// the point of the exercise is to reclaim tokens, and a stale 60 KB file dump
/// is worth more than a dozen 200-byte search hits put together. Clearing is a
/// monotonic ratchet — a cleared result never reappears — so the edited view
/// stays stable across steps.
pub struct ToolResultClearingAssembler {
    inner: Arc<dyn ContextAssembler>,
    /// Token estimate above which clearing kicks in.
    pub trigger_tokens: usize,
    /// Number of most-recent tool results never cleared (default 2).
    ///
    /// A small hard-protect window: the results the model is actively working
    /// from. Anything older is judged on size, so a large recent result is not
    /// shielded merely by its position.
    pub keep_recent: usize,
    /// Fraction of `trigger_tokens` a clearing pass reclaims down to
    /// (default 0.75), so the trigger is not tripped again on the next step.
    pub target: f64,
    /// Results smaller than this are never cleared (default 512 bytes).
    ///
    /// Below it, clearing frees nothing worth having and still costs the model
    /// whatever the result said.
    pub min_clear_bytes: usize,
    /// Ordinals (counted over tool results from the start of the conversation)
    /// whose content has been cleared. Grows only.
    cleared: Mutex<HashSet<usize>>,
    /// Learns the estimator's error from real provider usage.
    calibration: Arc<TokenCalibration>,
}

impl ToolResultClearingAssembler {
    pub fn new(inner: Arc<dyn ContextAssembler>, trigger_tokens: usize) -> Self {
        Self {
            inner,
            trigger_tokens,
            keep_recent: 2,
            target: 0.75,
            min_clear_bytes: 512,
            cleared: Mutex::new(HashSet::new()),
            calibration: Arc::new(TokenCalibration::new()),
        }
    }

    /// Sets how many of the most recent tool results are never cleared.
    pub fn with_keep_recent(mut self, n: usize) -> Self {
        self.keep_recent = n;
        self
    }

    /// Sets the fraction of the trigger a clearing pass reclaims down to.
    pub fn with_target(mut self, fraction: f64) -> Self {
        self.target = fraction;
        self
    }

    /// Sets the size below which a result is left alone.
    pub fn with_min_clear_bytes(mut self, bytes: usize) -> Self {
        self.min_clear_bytes = bytes;
        self
    }

    /// Share a [`TokenCalibration`] with other components (e.g. a
    /// [`SummarizingContextAssembler`](crate::SummarizingContextAssembler)
    /// wrapped around this one) so they learn from the same samples.
    pub fn with_calibration(mut self, calibration: Arc<TokenCalibration>) -> Self {
        self.calibration = calibration;
        self
    }

    /// The calibration this assembler is using.
    pub fn calibration(&self) -> Arc<TokenCalibration> {
        Arc::clone(&self.calibration)
    }

    fn placeholder(original: &str) -> String {
        format!(
            "[cleared by harness: {} chars of stale tool output removed from context. \
             The full result remains in the event log; re-run the tool if you need it again.]",
            original.len()
        )
    }

    /// Replace the content of every tool result whose ordinal is in `cleared`.
    fn apply_clearing(messages: &mut [ChatMessage], cleared: &HashSet<usize>) {
        let mut ordinal = 0usize;
        for m in messages.iter_mut() {
            if m.role != Role::Tool {
                continue;
            }
            let this = ordinal;
            ordinal += 1;
            if !cleared.contains(&this) {
                continue;
            }
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

    /// Choose which tool results to clear, biggest first, until `need` tokens
    /// have been reclaimed.
    ///
    /// Ranking is by size, nudged by age. Reclaiming budget is the whole point
    /// of the pass, so what matters about a result is how much of the budget it
    /// is holding; clearing a 200-byte search hit frees nothing and still takes
    /// a fact away from the model. Age only separates results of comparable
    /// size, where the older one is the safer of the two to drop.
    fn select_for_clearing(
        &self,
        messages: &[ChatMessage],
        cleared: &HashSet<usize>,
        need: usize,
    ) -> Vec<usize> {
        let results: Vec<&ChatMessage> = messages.iter().filter(|m| m.role == Role::Tool).collect();
        let total = results.len();
        // The newest few are off limits — the model is still working from them.
        let open = total.saturating_sub(self.keep_recent);

        // (ordinal, score, tokens reclaimed)
        let mut candidates: Vec<(usize, f64, usize)> = results
            .iter()
            .enumerate()
            .take(open)
            .filter(|(i, _)| !cleared.contains(i))
            .filter_map(|(i, m)| {
                let bytes = m.content.as_deref().map_or(0, str::len);
                if bytes < self.min_clear_bytes {
                    return None;
                }
                // 1.0 for the newest clearable result, 2.0 for the oldest.
                let age = 1.0 + (total - i) as f64 / total.max(1) as f64;
                let tokens = self.calibration.count(std::slice::from_ref(*m));
                Some((i, bytes as f64 * age, tokens))
            })
            .collect();

        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut picked = Vec::new();
        let mut reclaimed = 0usize;
        for (ordinal, _, tokens) in candidates {
            if reclaimed >= need {
                break;
            }
            picked.push(ordinal);
            reclaimed += tokens;
        }
        picked
    }
}

#[async_trait]
impl ContextAssembler for ToolResultClearingAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let mut messages = self.inner.assemble(context).await;

        // Learn from the provider's real prompt-token counts before deciding.
        self.calibration.observe_events(context.events);

        let mut cleared = self.cleared.lock().unwrap_or_else(|e| e.into_inner());

        // Always re-apply the ratchet so previously cleared results stay cleared.
        Self::apply_clearing(&mut messages, &cleared);

        let estimate = self.calibration.count(&messages);
        if estimate > self.trigger_tokens {
            // Reclaim past the trigger, not merely to it, so the next few steps
            // do not each trip another pass.
            let low_water = (self.trigger_tokens as f64 * self.target) as usize;
            let need = estimate.saturating_sub(low_water);
            let picked = self.select_for_clearing(&messages, &cleared, need);
            if !picked.is_empty() {
                debug!(
                    cleared_before = cleared.len(),
                    newly_cleared = picked.len(),
                    tokens = estimate,
                    need,
                    "context over trigger — clearing the largest stale tool results"
                );
                cleared.extend(picked);
                Self::apply_clearing(&mut messages, &cleared);
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

    /// A tool turn whose result is `bytes` long.
    fn sized_turn(id: &str, bytes: usize) -> Vec<Event> {
        tool_turn(id, &"x".repeat(bytes))
    }

    #[tokio::test]
    async fn the_big_results_go_first_whatever_their_age() {
        // Oldest to newest: tiny, huge, tiny, tiny, huge.
        let mut events = vec![Event::new(kinds::USER_MESSAGE, json!({"text": "go"}))];
        events.extend(sized_turn("t0", 600));
        events.extend(sized_turn("t1", 40_000));
        events.extend(sized_turn("t2", 600));
        events.extend(sized_turn("t3", 600));
        events.extend(sized_turn("t4", 40_000));

        let inner = DefaultContextAssembler::without_system_prompt();
        let assembler = ToolResultClearingAssembler::new(Arc::new(inner), 2_000)
            .with_keep_recent(1)
            .with_min_clear_bytes(1_000);

        let ctx = AssemblyContext::new(&events);
        let messages = assembler.assemble(&ctx).await;

        let tool_msgs: Vec<&ChatMessage> =
            messages.iter().filter(|m| m.role == Role::Tool).collect();
        let is_cleared = |m: &ChatMessage| {
            m.content
                .as_deref()
                .unwrap_or("")
                .starts_with("[cleared by harness:")
        };

        assert!(is_cleared(tool_msgs[1]), "the 40 KB dump must be reclaimed");
        for (i, small) in [0usize, 2, 3].iter().enumerate() {
            assert!(
                !is_cleared(tool_msgs[*small]),
                "small result #{i} frees nothing — clearing it is pure loss"
            );
        }
    }

    #[tokio::test]
    async fn a_large_recent_result_is_not_shielded_by_its_position() {
        // The single biggest result is also the newest but one. Under an
        // age-ordered policy it would be untouchable while small old results
        // were destroyed around it.
        let mut events = vec![Event::new(kinds::USER_MESSAGE, json!({"text": "go"}))];
        events.extend(sized_turn("t0", 600));
        events.extend(sized_turn("t1", 600));
        events.extend(sized_turn("t2", 80_000));
        events.extend(sized_turn("t3", 600));

        let inner = DefaultContextAssembler::without_system_prompt();
        let assembler = ToolResultClearingAssembler::new(Arc::new(inner), 2_000)
            .with_keep_recent(1)
            .with_min_clear_bytes(1_000);

        let ctx = AssemblyContext::new(&events);
        let messages = assembler.assemble(&ctx).await;
        let tool_msgs: Vec<&ChatMessage> =
            messages.iter().filter(|m| m.role == Role::Tool).collect();

        assert!(tool_msgs[2]
            .content
            .as_deref()
            .unwrap()
            .starts_with("[cleared by harness:"));
    }

    #[tokio::test]
    async fn results_under_the_floor_are_left_alone() {
        // Nothing here is worth clearing: the context is over the trigger only
        // because there are many small results, and clearing them would cost
        // the model facts while freeing nothing.
        let mut events = vec![Event::new(kinds::USER_MESSAGE, json!({"text": "go"}))];
        for i in 0..30 {
            events.extend(sized_turn(&format!("s{i}"), 300));
        }

        let inner = DefaultContextAssembler::without_system_prompt();
        let assembler = ToolResultClearingAssembler::new(Arc::new(inner), 500)
            .with_keep_recent(1)
            .with_min_clear_bytes(1_000);

        let ctx = AssemblyContext::new(&events);
        let messages = assembler.assemble(&ctx).await;
        assert!(
            messages.iter().filter(|m| m.role == Role::Tool).all(|m| !m
                .content
                .as_deref()
                .unwrap_or("")
                .starts_with("[cleared")),
            "sub-floor results must survive"
        );
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
