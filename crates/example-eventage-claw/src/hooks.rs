//! Approval hooks for eventage-claw.
//!
//! - [`SecurityGateHook`] — TUI mode: event-driven approval via the bus.
//! - [`HumanApprovalHook`] — REPL mode: stdin-based approval.
//!
//! Adapted from example-coding-agent/src/hooks.rs.

use async_trait::async_trait;
use eventage::agent::hook::{CycleHook, HookAction, HookContext};
use eventage::{Event, EventBus};
use serde_json::{json, Value};
use std::io::Write as _;
use std::time::Duration;
use tracing::warn;

use crate::kinds::{CLAW_APPROVAL_DENIED, CLAW_APPROVAL_GRANTED, CLAW_APPROVAL_REQUESTED};

// ── SecurityGateHook ──────────────────────────────────────────────────────────

/// TUI mode: publishes `CLAW_APPROVAL_REQUESTED` and waits for the TUI to
/// respond with `CLAW_APPROVAL_GRANTED` or `CLAW_APPROVAL_DENIED`.
pub struct SecurityGateHook {
    bus: EventBus,
    timeout: Duration,
    watched: Vec<String>,
}

impl SecurityGateHook {
    pub fn all_tools(bus: EventBus) -> Self {
        Self {
            bus,
            timeout: Duration::from_secs(300),
            watched: vec![],
        }
    }

    pub fn watched(bus: EventBus, tools: Vec<String>) -> Self {
        Self {
            bus,
            timeout: Duration::from_secs(300),
            watched: tools,
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
        if !self.watched.is_empty() && !self.watched.iter().any(|t| t == name) {
            return HookAction::Continue;
        }

        let _ = self
            .bus
            .publish(Event::new(
                CLAW_APPROVAL_REQUESTED,
                json!({
                    "tool": name,
                    "args": args,
                    "trace_id": ctx.trace_id,
                    "agent_id": ctx.agent_id,
                }),
            ))
            .await;

        let bus = self.bus.clone();
        let result = tokio::time::timeout(self.timeout, async move {
            bus.wait_for(|e: &Event| {
                e.kind == CLAW_APPROVAL_GRANTED || e.kind == CLAW_APPROVAL_DENIED
            })
            .await
        })
        .await;

        match result {
            Ok(Ok(event)) if event.kind == CLAW_APPROVAL_GRANTED => HookAction::Continue,
            Ok(Ok(_)) => HookAction::Skip,
            Ok(Err(_)) | Err(_) => {
                warn!(tool = name, "SecurityGateHook: approval timed out — vetoed");
                HookAction::Skip
            }
        }
    }
}

// ── HumanApprovalHook ─────────────────────────────────────────────────────────

/// REPL mode: prompts the user via stdin before executing watched tools.
pub struct HumanApprovalHook {
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
    async fn before_tool(&self, _ctx: &HookContext<'_>, name: &str, _args: &Value) -> HookAction {
        if !self.tools.is_empty() && !self.tools.iter().any(|t| t == name) {
            return HookAction::Continue;
        }

        tokio::task::yield_now().await;

        eprint!("Approve tool '{name}'? [y/N]: ");
        let _ = std::io::stderr().flush();

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
            eprintln!("[claw] Tool call vetoed: {name}");
            HookAction::Skip
        }
    }
}
