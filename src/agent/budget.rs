//! Token-budget enforcement for agent sessions.
//!
//! [`TokenBudgetHook`] tracks LLM token usage from the event log (every
//! `assistant.message` carries `llm_input_tokens` / `llm_output_tokens`
//! metadata) and enforces a hard ceiling:
//!
//! - Crossing the **warn threshold** (default 80%) publishes a
//!   `budget.warning` event and injects a one-line harness note into the
//!   prompt so the model starts wrapping up.
//! - Crossing the **budget** publishes `budget.exhausted` and aborts the
//!   cycle before the next LLM call.
//!
//! Because usage is recomputed from the bus, the accounting survives process
//! restarts when the log is restored from persistence, and multiple agents on
//! a shared bus can be governed by one session-wide budget.

use async_trait::async_trait;
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::warn;

use super::hook::{CycleHook, HookAction, HookContext};
use crate::event::{kinds, meta_keys, Event};
use crate::llm::ChatMessage;

/// Which events count against the budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetScope {
    /// All LLM usage on the bus (every agent in the session).
    Session,
    /// Only usage attributed to the agent this hook runs inside.
    Agent,
}

/// A [`CycleHook`] enforcing a token budget over the event log.
pub struct TokenBudgetHook {
    /// Hard ceiling on input + output tokens.
    pub max_tokens: u64,
    /// Fraction of `max_tokens` at which the warning fires (default 0.8).
    pub warn_fraction: f64,
    /// Whether usage is counted per session or per agent (default session).
    pub scope: BudgetScope,
    warned: AtomicBool,
    exhausted_reported: AtomicBool,
}

impl TokenBudgetHook {
    pub fn new(max_tokens: u64) -> Self {
        Self {
            max_tokens,
            warn_fraction: 0.8,
            scope: BudgetScope::Session,
            warned: AtomicBool::new(false),
            exhausted_reported: AtomicBool::new(false),
        }
    }

    /// Sets the warning threshold as a fraction of the budget.
    pub fn with_warn_fraction(mut self, fraction: f64) -> Self {
        self.warn_fraction = fraction;
        self
    }

    /// Restrict accounting to the owning agent's usage only.
    pub fn agent_scoped(mut self) -> Self {
        self.scope = BudgetScope::Agent;
        self
    }

    /// Sum tracked token usage from the event log.
    async fn used_tokens(&self, ctx: &HookContext<'_>) -> u64 {
        let events = ctx.bus.log().await;
        events
            .iter()
            .filter(|e| match self.scope {
                BudgetScope::Session => true,
                BudgetScope::Agent => {
                    e.metadata.get(meta_keys::AGENT_ID).and_then(|v| v.as_str())
                        == Some(ctx.agent_id)
                }
            })
            .map(|e| {
                let read = |key: &str| e.metadata.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
                read(meta_keys::LLM_INPUT_TOKENS) + read(meta_keys::LLM_OUTPUT_TOKENS)
            })
            .sum()
    }
}

#[async_trait]
impl CycleHook for TokenBudgetHook {
    async fn before_llm(
        &self,
        ctx: &HookContext<'_>,
        messages: &mut Vec<ChatMessage>,
    ) -> HookAction {
        let used = self.used_tokens(ctx).await;

        if used >= self.max_tokens {
            if !self.exhausted_reported.swap(true, Ordering::SeqCst) {
                warn!(
                    used,
                    max = self.max_tokens,
                    "token budget exhausted — aborting cycle"
                );
                let _ = ctx
                    .bus
                    .publish(Event::new(
                        kinds::BUDGET_EXHAUSTED,
                        json!({ "used_tokens": used, "max_tokens": self.max_tokens }),
                    ))
                    .await;
            }
            return HookAction::AbortCycle;
        }

        let warn_at = (self.max_tokens as f64 * self.warn_fraction) as u64;
        if used >= warn_at {
            if !self.warned.swap(true, Ordering::SeqCst) {
                let _ = ctx
                    .bus
                    .publish(Event::new(
                        kinds::BUDGET_WARNING,
                        json!({ "used_tokens": used, "max_tokens": self.max_tokens }),
                    ))
                    .await;
            }
            messages.push(
                ChatMessage::user(format!(
                    "[harness] Token budget nearly exhausted ({used} of {} used). \
                     Finish the task now with what you have; avoid further tool calls \
                     unless strictly necessary.",
                    self.max_tokens
                ))
                .with_name("harness"),
            );
        }

        HookAction::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::EventBus;

    fn usage_event(agent_id: &str, input: u64, output: u64) -> Event {
        Event::new(kinds::ASSISTANT_MESSAGE, json!({"content": "x"}))
            .with_meta(meta_keys::AGENT_ID, json!(agent_id))
            .with_meta(meta_keys::LLM_INPUT_TOKENS, json!(input))
            .with_meta(meta_keys::LLM_OUTPUT_TOKENS, json!(output))
    }

    #[tokio::test]
    async fn aborts_when_budget_exhausted() {
        let bus = EventBus::new();
        bus.publish(usage_event("a", 900, 200)).await.unwrap();

        let hook = TokenBudgetHook::new(1000);
        let ctx = HookContext {
            agent_id: "a",
            trace_id: "t",
            step: 1,
            bus: &bus,
        };
        let mut messages = vec![];
        assert_eq!(
            hook.before_llm(&ctx, &mut messages).await,
            HookAction::AbortCycle
        );

        let log = bus.log().await;
        assert!(log.iter().any(|e| e.kind == kinds::BUDGET_EXHAUSTED));
    }

    #[tokio::test]
    async fn warns_and_injects_note_near_limit() {
        let bus = EventBus::new();
        bus.publish(usage_event("a", 700, 150)).await.unwrap();

        let hook = TokenBudgetHook::new(1000);
        let ctx = HookContext {
            agent_id: "a",
            trace_id: "t",
            step: 1,
            bus: &bus,
        };
        let mut messages = vec![ChatMessage::user("hi")];
        assert_eq!(
            hook.before_llm(&ctx, &mut messages).await,
            HookAction::Continue
        );
        assert_eq!(messages.len(), 2, "wrap-up note should be injected");
        assert!(messages[1]
            .content
            .as_deref()
            .unwrap()
            .contains("budget nearly exhausted"));

        let log = bus.log().await;
        assert!(log.iter().any(|e| e.kind == kinds::BUDGET_WARNING));
    }

    #[tokio::test]
    async fn agent_scope_ignores_other_agents() {
        let bus = EventBus::new();
        bus.publish(usage_event("other", 5000, 5000)).await.unwrap();
        bus.publish(usage_event("me", 10, 10)).await.unwrap();

        let hook = TokenBudgetHook::new(1000).agent_scoped();
        let ctx = HookContext {
            agent_id: "me",
            trace_id: "t",
            step: 1,
            bus: &bus,
        };
        let mut messages = vec![];
        assert_eq!(
            hook.before_llm(&ctx, &mut messages).await,
            HookAction::Continue
        );
        assert!(messages.is_empty(), "no warning expected under agent scope");
    }
}
