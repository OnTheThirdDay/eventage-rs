//! Rule-based tool permission policy with asynchronous, bus-native approval.
//!
//! [`PermissionPolicyHook`] is the enterprise governance layer for tool
//! execution: glob rules decide per tool name whether a call is allowed,
//! denied (with a reason the model can act on), or must be **asked** — in
//! which case a durable `permission.request` event is published and the hook
//! waits for a matching `permission.decision` event. Any approver can answer:
//! a TUI, a Slack bridge, a web dashboard, or another agent — anything that
//! can publish to the bus. Unanswered requests deny after a timeout.
//!
//! ```no_run
//! use eventage::agent::PermissionPolicyHook;
//!
//! let policy = PermissionPolicyHook::new()
//!     .allow("read_*")
//!     .allow("search_*")
//!     .ask("write_*")
//!     .deny("delete_*", "deletion is disabled in this environment")
//!     .deny_by_default("tool not in the allowlist for this deployment");
//! ```

use async_trait::async_trait;
use serde_json::{json, Value};
use std::time::Duration;
use tracing::{info, warn};
use uuid::Uuid;

use super::hook::{CycleHook, HookAction, HookContext};
use crate::event::{kinds, Event};

// ── Glob matching ─────────────────────────────────────────────────────────────

/// Match `text` against `pattern`, where `*` matches any (possibly empty)
/// substring. No other metacharacters.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let segments: Vec<&str> = pattern.split('*').collect();
    if segments.len() == 1 {
        return pattern == text;
    }

    let mut pos = 0usize;
    // First segment anchors at the start.
    if let Some(first) = segments.first() {
        if !text[pos..].starts_with(first) {
            return false;
        }
        pos += first.len();
    }
    // Middle segments must appear in order.
    for seg in &segments[1..segments.len() - 1] {
        if seg.is_empty() {
            continue;
        }
        match text[pos..].find(seg) {
            Some(found) => pos += found + seg.len(),
            None => return false,
        }
    }
    // Last segment anchors at the end.
    let last = segments[segments.len() - 1];
    text.len() >= pos + last.len() && text[pos..].ends_with(last)
}

// ── Policy types ──────────────────────────────────────────────────────────────

/// Verdict a rule applies to a matching tool call.
#[derive(Debug, Clone)]
pub enum PermissionVerdict {
    /// Execute without interaction.
    Allow,
    /// Refuse; the reason is surfaced to the model via [`HookAction::Deny`].
    Deny(String),
    /// Publish a `permission.request` event and wait for a
    /// `permission.decision` before executing.
    Ask,
}

struct PermissionRule {
    pattern: String,
    verdict: PermissionVerdict,
}

/// A [`CycleHook`] enforcing glob-based tool permissions with async approval.
///
/// Rules are evaluated in registration order; the first matching rule wins.
/// Unmatched tools fall through to the default verdict (initially
/// [`Allow`](PermissionVerdict::Allow); harden with
/// [`deny_by_default`](Self::deny_by_default) or
/// [`ask_by_default`](Self::ask_by_default)).
pub struct PermissionPolicyHook {
    rules: Vec<PermissionRule>,
    default_verdict: PermissionVerdict,
    /// How long an `Ask` waits for a decision before denying (default 120 s).
    ask_timeout: Duration,
}

impl Default for PermissionPolicyHook {
    fn default() -> Self {
        Self::new()
    }
}

impl PermissionPolicyHook {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            default_verdict: PermissionVerdict::Allow,
            ask_timeout: Duration::from_secs(120),
        }
    }

    /// Allow tools whose name matches `pattern` (first match wins).
    pub fn allow(mut self, pattern: impl Into<String>) -> Self {
        self.rules.push(PermissionRule {
            pattern: pattern.into(),
            verdict: PermissionVerdict::Allow,
        });
        self
    }

    /// Deny matching tools with a model-visible reason.
    pub fn deny(mut self, pattern: impl Into<String>, reason: impl Into<String>) -> Self {
        self.rules.push(PermissionRule {
            pattern: pattern.into(),
            verdict: PermissionVerdict::Deny(reason.into()),
        });
        self
    }

    /// Require asynchronous approval for matching tools.
    pub fn ask(mut self, pattern: impl Into<String>) -> Self {
        self.rules.push(PermissionRule {
            pattern: pattern.into(),
            verdict: PermissionVerdict::Ask,
        });
        self
    }

    /// Deny any tool not matched by an earlier rule.
    pub fn deny_by_default(mut self, reason: impl Into<String>) -> Self {
        self.default_verdict = PermissionVerdict::Deny(reason.into());
        self
    }

    /// Require approval for any tool not matched by an earlier rule.
    pub fn ask_by_default(mut self) -> Self {
        self.default_verdict = PermissionVerdict::Ask;
        self
    }

    /// Set how long an `Ask` waits for a `permission.decision` before denying.
    pub fn with_ask_timeout(mut self, timeout: Duration) -> Self {
        self.ask_timeout = timeout;
        self
    }

    /// The verdict this policy applies to `tool` (first matching rule wins).
    ///
    /// Exposed so callers can preview or test a policy without executing a
    /// tool call.
    pub fn verdict_for(&self, tool: &str) -> &PermissionVerdict {
        self.rules
            .iter()
            .find(|r| glob_match(&r.pattern, tool))
            .map(|r| &r.verdict)
            .unwrap_or(&self.default_verdict)
    }

    /// Publish a `permission.request` and wait for the matching decision.
    async fn ask_for_approval(
        &self,
        ctx: &HookContext<'_>,
        tool: &str,
        args: &Value,
    ) -> HookAction {
        let request_id = Uuid::new_v4().to_string();

        // Subscribe BEFORE publishing so a fast approver can't race us.
        let mut rx = ctx.bus.subscribe();

        let request = Event::new(
            kinds::PERMISSION_REQUEST,
            json!({
                "request_id": request_id,
                "tool": tool,
                "arguments": args,
                "agent_id": ctx.agent_id,
            }),
        );
        if let Err(e) = ctx.bus.publish(request).await {
            return HookAction::Deny(format!("permission request could not be published: {e}"));
        }
        info!(tool, request_id = %request_id, "waiting for permission decision");

        let deadline = tokio::time::Instant::now() + self.ask_timeout;
        loop {
            let event = match tokio::time::timeout_at(deadline, rx.recv()).await {
                Err(_) => {
                    warn!(tool, request_id = %request_id, "permission request timed out");
                    return HookAction::Deny(format!(
                        "approval for '{tool}' timed out after {}s; \
                         proceed without this tool or ask the user to respond",
                        self.ask_timeout.as_secs()
                    ));
                }
                Ok(None) => return HookAction::Deny("event bus closed".to_string()),
                Ok(Some(event)) => event,
            };

            if event.kind != kinds::PERMISSION_DECISION {
                continue;
            }
            if event.payload.get("request_id").and_then(|v| v.as_str()) != Some(&request_id) {
                continue;
            }

            let approved = event
                .payload
                .get("approve")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if approved {
                return HookAction::Continue;
            }
            let reason = event
                .payload
                .get("reason")
                .and_then(|v| v.as_str())
                .unwrap_or("request was declined by the approver");
            return HookAction::Deny(reason.to_string());
        }
    }
}

#[async_trait]
impl CycleHook for PermissionPolicyHook {
    async fn before_tool(&self, ctx: &HookContext<'_>, name: &str, args: &Value) -> HookAction {
        match self.verdict_for(name) {
            PermissionVerdict::Allow => HookAction::Continue,
            PermissionVerdict::Deny(reason) => HookAction::Deny(reason.clone()),
            PermissionVerdict::Ask => self.ask_for_approval(ctx, name, args).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_semantics() {
        assert!(glob_match("read_*", "read_file"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("exact", "exact"));
        assert!(glob_match("*_file", "write_file"));
        assert!(glob_match("a*b*c", "a-x-b-y-c"));
        assert!(!glob_match("read_*", "write_file"));
        assert!(!glob_match("exact", "exactly"));
        assert!(!glob_match("a*b*c", "a-c-b"));
    }

    #[test]
    fn first_matching_rule_wins() {
        let policy = PermissionPolicyHook::new()
            .deny("write_secrets", "secrets are protected")
            .allow("write_*")
            .deny_by_default("not allowlisted");

        assert!(matches!(
            policy.verdict_for("write_secrets"),
            PermissionVerdict::Deny(_)
        ));
        assert!(matches!(
            policy.verdict_for("write_file"),
            PermissionVerdict::Allow
        ));
        assert!(matches!(
            policy.verdict_for("launch_rocket"),
            PermissionVerdict::Deny(_)
        ));
    }
}
