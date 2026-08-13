//! Crash recovery for interrupted tool calls.
//!
//! A process that dies between `tool.call.proposed` and its `tool.result`
//! leaves the restored log with an **orphaned call**: the model asked for a
//! side effect, and the harness cannot know whether it happened.
//!
//! # What this guarantees (and what it cannot)
//!
//! True exactly-once execution is impossible without cooperation from the
//! tool itself — the crash may have occurred after the side effect but
//! before the result was persisted. What this module provides is a
//! *reconciliation pass with an explicit, per-tool policy*:
//!
//! - [`ResumePolicy::ReportInterrupted`] (default) — **at-most-once**: the
//!   call is never re-executed; a synthetic error result tells the model the
//!   outcome is unknown so it can verify before retrying. Correct for tools
//!   with side effects.
//! - [`ResumePolicy::Replay`] — **at-least-once**: re-execute the call. Only
//!   safe for idempotent tools (reads, pure computations, or tools with
//!   their own idempotency keys); combined with tool-side idempotency this
//!   is what effectively-once looks like in practice.
//! - [`ResumePolicy::Fail`] — refuse to resume, for deployments that require
//!   a human to adjudicate.
//!
//! Reconciliation also repairs the message history: without a result for
//! every `tool_call_id`, most providers reject the next request outright.
//!
//! ```no_run
//! # use eventage::{EventBus, ToolRegistry};
//! # use eventage::agent::recovery::{reconcile_interrupted_tools, ToolRecovery};
//! # async fn example(bus: EventBus, registry: ToolRegistry) -> anyhow::Result<()> {
//! // After restoring a persisted log, before starting the agent:
//! let policy = ToolRecovery::new().replay("read_*").replay("search_*");
//! let report = reconcile_interrupted_tools(&bus, &policy, Some(&registry)).await?;
//! println!("reconciled {} interrupted call(s)", report.total());
//! # Ok(())
//! # }
//! ```

use super::error::AgentError;
use super::tool::ToolRegistry;
use crate::agent::permission::glob_match;
use crate::bus::EventBus;
use crate::event::{kinds, Event};
use serde_json::json;
use std::collections::HashSet;
use tracing::{info, warn};

/// How to handle an interrupted tool call on resume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumePolicy {
    /// Never re-execute; report the interruption to the model (at-most-once).
    ReportInterrupted,
    /// Re-execute the call (at-least-once; requires an idempotent tool).
    Replay,
    /// Refuse to resume while unresolved calls exist.
    Fail,
}

/// Per-tool resume policy, matched by glob in registration order.
pub struct ToolRecovery {
    rules: Vec<(String, ResumePolicy)>,
    default_policy: ResumePolicy,
}

impl Default for ToolRecovery {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRecovery {
    /// Defaults every tool to [`ResumePolicy::ReportInterrupted`] — the safe
    /// choice, since the harness cannot know whether a side effect landed.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            default_policy: ResumePolicy::ReportInterrupted,
        }
    }

    /// Re-execute tools matching `pattern` on resume (idempotent tools only).
    pub fn replay(mut self, pattern: impl Into<String>) -> Self {
        self.rules.push((pattern.into(), ResumePolicy::Replay));
        self
    }

    /// Explicitly report-only for tools matching `pattern`.
    pub fn report(mut self, pattern: impl Into<String>) -> Self {
        self.rules
            .push((pattern.into(), ResumePolicy::ReportInterrupted));
        self
    }

    /// Refuse to resume when a tool matching `pattern` was interrupted.
    pub fn fail(mut self, pattern: impl Into<String>) -> Self {
        self.rules.push((pattern.into(), ResumePolicy::Fail));
        self
    }

    /// Change the policy for tools not matched by any rule.
    pub fn with_default(mut self, policy: ResumePolicy) -> Self {
        self.default_policy = policy;
        self
    }

    pub fn policy_for(&self, tool: &str) -> ResumePolicy {
        self.rules
            .iter()
            .find(|(pattern, _)| glob_match(pattern, tool))
            .map(|(_, policy)| *policy)
            .unwrap_or(self.default_policy)
    }
}

/// A tool call that was proposed but never produced a result.
#[derive(Debug, Clone)]
pub struct OrphanedCall {
    pub tool_call_id: String,
    pub name: String,
    /// Raw JSON arguments string from the proposal event.
    pub arguments: String,
}

/// Scan an event log for proposals with no matching `tool.result`.
pub fn find_orphaned_tool_calls(events: &[Event]) -> Vec<OrphanedCall> {
    let answered: HashSet<&str> = events
        .iter()
        .filter(|e| e.kind == kinds::TOOL_RESULT)
        .filter_map(|e| e.payload.get("tool_call_id").and_then(|v| v.as_str()))
        .collect();

    events
        .iter()
        .filter(|e| e.kind == kinds::TOOL_CALL_PROPOSED)
        .filter_map(|e| {
            let id = e.payload.get("tool_call_id").and_then(|v| v.as_str())?;
            if answered.contains(id) {
                return None;
            }
            Some(OrphanedCall {
                tool_call_id: id.to_string(),
                name: e
                    .payload
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                arguments: e
                    .payload
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}")
                    .to_string(),
            })
        })
        .collect()
}

/// What a reconciliation pass did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct RecoveryReport {
    /// Calls re-executed under [`ResumePolicy::Replay`].
    pub replayed: usize,
    /// Calls reported to the model as interrupted.
    pub reported: usize,
}

impl RecoveryReport {
    pub fn total(&self) -> usize {
        self.replayed + self.reported
    }

    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }
}

/// Resolve every interrupted tool call on `bus` according to `policy`.
///
/// Call this after restoring a persisted log and before starting agents.
/// Each orphan gets a `tool.result` event so the history is valid again, and
/// a `system.recovered` event records what happened.
///
/// `registry` is needed only for [`ResumePolicy::Replay`]; without it (or
/// when the tool is no longer registered) a replay degrades to a reported
/// interruption.
pub async fn reconcile_interrupted_tools(
    bus: &EventBus,
    policy: &ToolRecovery,
    registry: Option<&ToolRegistry>,
) -> Result<RecoveryReport, AgentError> {
    let events = bus.log().await;
    let orphans = find_orphaned_tool_calls(&events);
    if orphans.is_empty() {
        return Ok(RecoveryReport::default());
    }

    let mut report = RecoveryReport::default();
    for orphan in &orphans {
        let decided = policy.policy_for(&orphan.name);
        if decided == ResumePolicy::Fail {
            return Err(AgentError::Tool(format!(
                "refusing to resume: tool '{}' (call {}) was interrupted by a restart \
                 and its policy is Fail",
                orphan.name, orphan.tool_call_id
            )));
        }

        let tool = registry.and_then(|r| r.get(&orphan.name));
        let replay = decided == ResumePolicy::Replay && tool.is_some();

        let payload = if replay {
            let tool = tool.expect("checked above");
            let args = serde_json::from_str(&orphan.arguments).unwrap_or(json!({}));
            info!(tool = %orphan.name, call = %orphan.tool_call_id, "replaying interrupted tool call");
            match tool.execute(args).await {
                Ok(result) => {
                    report.replayed += 1;
                    json!({
                        "tool_call_id": orphan.tool_call_id,
                        "name": orphan.name,
                        "result": result,
                        "recovered": "replayed"
                    })
                }
                Err(e) => {
                    report.replayed += 1;
                    json!({
                        "tool_call_id": orphan.tool_call_id,
                        "name": orphan.name,
                        "error": e.to_string(),
                        "recovered": "replayed"
                    })
                }
            }
        } else {
            warn!(tool = %orphan.name, call = %orphan.tool_call_id, "tool call interrupted by restart");
            report.reported += 1;
            json!({
                "tool_call_id": orphan.tool_call_id,
                "name": orphan.name,
                "error": format!(
                    "tool '{}' was interrupted by a restart; it is UNKNOWN whether it \
                     took effect. Verify the current state before retrying.",
                    orphan.name
                ),
                "recovered": "interrupted"
            })
        };

        bus.publish(Event::new(kinds::TOOL_RESULT, payload)).await?;
    }

    bus.publish(Event::new(
        kinds::SYSTEM_RECOVERED,
        json!({
            "interrupted_tool_calls": report.total(),
            "replayed": report.replayed,
            "reported": report.reported,
        }),
    ))
    .await?;

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tool::Tool;
    use crate::llm::types::ToolDefinition;
    use async_trait::async_trait;
    use serde_json::Value;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct CountingTool {
        name: String,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Tool for CountingTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(&self.name, "counts", json!({"type": "object"}))
        }
        async fn execute(&self, _args: Value) -> Result<Value, AgentError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(json!({"ok": true}))
        }
    }

    async fn bus_with_orphan(name: &str) -> EventBus {
        let bus = EventBus::new();
        bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "go"})))
            .await
            .unwrap();
        bus.publish(Event::new(
            kinds::TOOL_CALL_PROPOSED,
            json!({ "tool_call_id": "c1", "name": name, "arguments": "{}" }),
        ))
        .await
        .unwrap();
        bus
    }

    #[test]
    fn finds_only_unanswered_calls() {
        let events = vec![
            Event::new(
                kinds::TOOL_CALL_PROPOSED,
                json!({"tool_call_id": "a", "name": "t", "arguments": "{}"}),
            ),
            Event::new(kinds::TOOL_RESULT, json!({"tool_call_id": "a"})),
            Event::new(
                kinds::TOOL_CALL_PROPOSED,
                json!({"tool_call_id": "b", "name": "write_file", "arguments": "{\"p\":1}"}),
            ),
        ];
        let orphans = find_orphaned_tool_calls(&events);
        assert_eq!(orphans.len(), 1);
        assert_eq!(orphans[0].tool_call_id, "b");
        assert_eq!(orphans[0].name, "write_file");
    }

    #[tokio::test]
    async fn default_reports_without_executing() {
        let bus = bus_with_orphan("write_file").await;
        let calls = Arc::new(AtomicUsize::new(0));
        let registry = ToolRegistry::new();
        registry.add_tool(CountingTool {
            name: "write_file".into(),
            calls: Arc::clone(&calls),
        });

        let report = reconcile_interrupted_tools(&bus, &ToolRecovery::new(), Some(&registry))
            .await
            .unwrap();
        assert_eq!(report.reported, 1);
        assert_eq!(report.replayed, 0);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "side-effecting tool must NOT be re-executed"
        );

        let log = bus.log().await;
        let result = log.iter().find(|e| e.kind == kinds::TOOL_RESULT).unwrap();
        assert_eq!(result.payload["recovered"], "interrupted");
        assert!(result.payload["error"]
            .as_str()
            .unwrap()
            .contains("UNKNOWN whether it took effect"));
        assert!(log.iter().any(|e| e.kind == kinds::SYSTEM_RECOVERED));
    }

    #[tokio::test]
    async fn replay_policy_reexecutes_idempotent_tools() {
        let bus = bus_with_orphan("read_file").await;
        let calls = Arc::new(AtomicUsize::new(0));
        let registry = ToolRegistry::new();
        registry.add_tool(CountingTool {
            name: "read_file".into(),
            calls: Arc::clone(&calls),
        });

        let policy = ToolRecovery::new().replay("read_*");
        let report = reconcile_interrupted_tools(&bus, &policy, Some(&registry))
            .await
            .unwrap();
        assert_eq!(report.replayed, 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        let log = bus.log().await;
        let result = log.iter().find(|e| e.kind == kinds::TOOL_RESULT).unwrap();
        assert_eq!(result.payload["recovered"], "replayed");
        assert_eq!(result.payload["result"]["ok"], true);
    }

    #[tokio::test]
    async fn replay_without_registry_degrades_to_report() {
        let bus = bus_with_orphan("read_file").await;
        let policy = ToolRecovery::new().replay("read_*");
        let report = reconcile_interrupted_tools(&bus, &policy, None)
            .await
            .unwrap();
        assert_eq!(report.reported, 1);
        assert_eq!(report.replayed, 0);
    }

    #[tokio::test]
    async fn fail_policy_refuses_to_resume() {
        let bus = bus_with_orphan("transfer_funds").await;
        let policy = ToolRecovery::new().fail("transfer_*");
        let err = reconcile_interrupted_tools(&bus, &policy, None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("refusing to resume"));
    }

    #[tokio::test]
    async fn clean_log_is_a_no_op() {
        let bus = EventBus::new();
        bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "hi"})))
            .await
            .unwrap();
        let report = reconcile_interrupted_tools(&bus, &ToolRecovery::new(), None)
            .await
            .unwrap();
        assert!(report.is_empty());
        assert!(!bus
            .log()
            .await
            .iter()
            .any(|e| e.kind == kinds::SYSTEM_RECOVERED));
    }
}
