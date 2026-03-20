//! Security gate hook — intercepts dangerous tool calls and requires explicit
//! user approval before execution.
//!
//! The hook publishes a [`kinds::CODING_APPROVAL_REQUESTED`] event and then
//! suspends the current react step via `bus.wait_for()`. Meanwhile, the TUI
//! shows an approval overlay. When the user presses **y** or **n**, the TUI
//! publishes [`kinds::CODING_APPROVAL_GRANTED`] or [`kinds::CODING_APPROVAL_DENIED`],
//! and the hook unblocks to allow or veto the tool call.
//!
//! This is a zero-polling, pure-event-driven flow — no polling loops or
//! channels outside the bus are needed.

use async_trait::async_trait;
use eventage::agent::{CycleHook, HookAction, HookContext};
use eventage::{Event, EventBus};
use serde_json::{json, Value};
use std::time::Duration;
use tracing::warn;

use crate::kinds::{CODING_APPROVAL_DENIED, CODING_APPROVAL_GRANTED, CODING_APPROVAL_REQUESTED};

/// Tool names that always require explicit user approval.
const DANGEROUS_TOOLS: &[&str] = &["write_file", "apply_patch", "execute_shell"];

/// A [`CycleHook`] that intercepts dangerous tool calls and waits for user
/// approval via the event bus.
///
/// # Event flow
///
/// ```text
/// SecurityGateHook::before_tool
///   └─ publish  coding.approval.requested { tool, args, trace_id }
///   └─ wait_for coding.approval.granted OR coding.approval.denied
///        ↑ TUI shows overlay and publishes one of the above when user responds
/// ```
///
/// On approval timeout (default 5 minutes), the call is vetoed with [`HookAction::Skip`].
pub struct SecurityGateHook {
    bus: EventBus,
    /// How long to wait for user approval before auto-denying.
    timeout: Duration,
}

impl SecurityGateHook {
    pub fn new(bus: EventBus) -> Self {
        Self {
            bus,
            timeout: Duration::from_secs(300),
        }
    }

    #[allow(dead_code)]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[async_trait]
impl CycleHook for SecurityGateHook {
    async fn before_tool(&self, ctx: &HookContext<'_>, name: &str, args: &Value) -> HookAction {
        if !DANGEROUS_TOOLS.contains(&name) {
            return HookAction::Continue;
        }

        // Announce that approval is needed.
        let _ = self
            .bus
            .publish(Event::new(
                CODING_APPROVAL_REQUESTED,
                json!({
                    "tool": name,
                    "args": args,
                    "trace_id": ctx.trace_id,
                    "agent_id": ctx.agent_id,
                }),
            ))
            .await;

        // Suspend until the TUI publishes a grant or deny response.
        let bus = self.bus.clone();
        let result = tokio::time::timeout(self.timeout, async move {
            bus.wait_for(|e: &Event| {
                e.kind == CODING_APPROVAL_GRANTED || e.kind == CODING_APPROVAL_DENIED
            })
            .await
        })
        .await;

        match result {
            Ok(event) if event.kind == CODING_APPROVAL_GRANTED => HookAction::Continue,
            Ok(_) => {
                // Denied by user.
                HookAction::Skip
            }
            Err(_) => {
                // Timeout — auto-deny.
                warn!(
                    tool = name,
                    agent_id = ctx.agent_id,
                    "SecurityGateHook: approval timed out — tool call vetoed"
                );
                HookAction::Skip
            }
        }
    }
}
