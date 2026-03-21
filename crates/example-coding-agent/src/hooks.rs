//! Approval hooks for the coding agent.
//!
//! Two hooks are provided:
//!
//! - [`SecurityGateHook`] — TUI mode: event-driven approval via the bus, zero-polling.
//! - [`HumanApprovalHook`] — REPL mode: stdin-based approval prompt.

use async_trait::async_trait;
use eventage::agent::hook::{CycleHook, HookAction, HookContext};
use eventage::{Event, EventBus};
use serde_json::{json, Value};
use std::io::Write as _;
use std::time::Duration;
use tracing::warn;

use crate::kinds::{CODING_APPROVAL_DENIED, CODING_APPROVAL_GRANTED, CODING_APPROVAL_REQUESTED};

// ── SecurityGateHook ──────────────────────────────────────────────────────────

/// A [`CycleHook`] that intercepts tool calls and waits for user approval via
/// the event bus (TUI mode).
///
/// Publishes [`CODING_APPROVAL_REQUESTED`] and suspends until the TUI
/// publishes [`CODING_APPROVAL_GRANTED`] or [`CODING_APPROVAL_DENIED`].
///
/// On timeout (default 5 minutes), the call is auto-denied.
pub struct SecurityGateHook {
    bus: EventBus,
    timeout: Duration,
    /// Tools to watch. Empty = gate every tool call.
    watched: Vec<String>,
}

impl SecurityGateHook {
    /// Gate every tool call.
    pub fn all_tools(bus: EventBus) -> Self {
        Self { bus, timeout: Duration::from_secs(300), watched: vec![] }
    }

    /// Gate only the specified tools.
    pub fn watched(bus: EventBus, tools: Vec<String>) -> Self {
        Self { bus, timeout: Duration::from_secs(300), watched: tools }
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
        if !self.watched.is_empty() && !self.watched.iter().any(|t| t == name) {
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
            Ok(_) => HookAction::Skip,
            Err(_) => {
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

// ── HumanApprovalHook ─────────────────────────────────────────────────────────

/// Intercepts specified tool calls and asks the user for approval via stdin.
///
/// If `tools` is empty, all tool calls require approval.
pub struct HumanApprovalHook {
    /// Tool names that require approval. Empty = gate every tool call.
    pub tools: Vec<String>,
}

impl HumanApprovalHook {
    pub fn new(tools: Vec<String>) -> Self {
        Self { tools }
    }

    pub fn all_tools() -> Self {
        Self { tools: vec![] }
    }
}

#[async_trait]
impl CycleHook for HumanApprovalHook {
    async fn before_tool(
        &self,
        _ctx: &HookContext<'_>,
        name: &str,
        _args: &Value,
    ) -> HookAction {
        // Skip if tool not in the watch list
        if !self.tools.is_empty() && !self.tools.iter().any(|t| t == name) {
            return HookAction::Continue;
        }

        // Yield once so drain_cycle can process TOOL_CALL_PROPOSED and print
        // "[→ tool] ..." before we show the approval prompt.
        tokio::task::yield_now().await;

        // "[→ tool] ..." is already shown by drain_cycle — just ask for approval.
        eprint!("Approve? [y/N]: ");
        let _ = std::io::stderr().flush();

        // Read stdin on a blocking thread to avoid blocking the async runtime
        let line = tokio::task::spawn_blocking(|| {
            let mut s = String::new();
            std::io::stdin().read_line(&mut s).ok();
            s
        })
        .await
        .unwrap_or_default();

        if line.trim().eq_ignore_ascii_case("y") {
            HookAction::Continue
        } else {
            eprintln!("[coding-agent] Tool call vetoed: {name}");
            HookAction::Skip
        }
    }
}
