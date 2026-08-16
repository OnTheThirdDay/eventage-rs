//! How much the agent may do without asking.
//!
//! The vocabulary is Claude Cowork's, because it is the one users of these
//! products already have: **Manual** pauses for approval, **Auto** checks each
//! action for safety and blocks what it judges unsafe, **Skip** does neither.
//!
//! What differs is where the judgement in `Auto` comes from. Cowork describes
//! Claude reviewing each action; that is a model deciding whether to trust
//! itself, and a repository or a document can talk to a model. Here the check
//! is code, applied to the *arguments* of a call rather than to the intent
//! behind it: deleting, overwriting a file nobody read, or reaching outside
//! the granted folder. A rule cannot be argued out of.
//!
//! One thing is not a mode. **Deletion always asks**, in every mode including
//! `Skip`, matching Cowork's own guarantee. A session where the user chose to
//! stop being interrupted is still a session where deleting somebody's work
//! uninvited is the wrong default, and it is the one action with no undo short
//! of a snapshot.

use eventage::agent::{CycleHook, HookAction, HookContext};
use eventage::{Event, EventBus};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tracing::warn;

use crate::kinds;

/// How much the agent may do without asking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Steering {
    /// Every action that changes something waits for approval.
    Manual,
    /// Actions are checked and the unsafe ones held for approval.
    Auto,
    /// No checks. Deletion still asks.
    Skip,
}

impl Steering {
    pub fn id(self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::Auto => "auto",
            Self::Skip => "skip",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "manual" => Some(Self::Manual),
            "auto" => Some(Self::Auto),
            "skip" => Some(Self::Skip),
            _ => None,
        }
    }

    pub fn describe(self) -> &'static str {
        match self {
            Self::Manual => "asks before every change",
            Self::Auto => "works on its own, asking only for risky actions",
            Self::Skip => "works without asking, except to delete",
        }
    }

    pub const NAMES: &'static str = "manual | auto | skip";

    fn code(self) -> u8 {
        match self {
            Self::Manual => 0,
            Self::Auto => 1,
            Self::Skip => 2,
        }
    }

    fn from_code(code: u8) -> Self {
        match code {
            0 => Self::Manual,
            1 => Self::Auto,
            _ => Self::Skip,
        }
    }
}

/// The steering mode in force, readable by a running turn.
///
/// Shared and atomic rather than copied into each hook, because the user
/// changes it *while* work is running — that is the point of a steering
/// control — and a copy taken when the session was built would go stale the
/// moment they did.
#[derive(Debug)]
pub struct SharedSteering(AtomicU8);

impl SharedSteering {
    pub fn new(mode: Steering) -> Self {
        Self(AtomicU8::new(mode.code()))
    }

    pub fn get(&self) -> Steering {
        Steering::from_code(self.0.load(Ordering::Relaxed))
    }

    pub fn set(&self, mode: Steering) {
        self.0.store(mode.code(), Ordering::Relaxed);
    }
}

/// Tools that only look at things.
const READ_ONLY: &[&str] = &[
    "read_file",
    "list_directory",
    "glob",
    "grep",
    "view_image",
    "web_search",
    "web_fetch",
    "plan",
];

/// Tools that destroy work outright, and always ask.
const DESTRUCTIVE: &[&str] = &["delete_file", "move_file"];

/// The approval gate.
///
/// Publishes a request onto the bus and waits for a decision, so any
/// surface — Studio, the HTTP channel, a test — answers the same way. A
/// timeout denies rather than allows: an unattended session must not become
/// an unsupervised one just because nobody was watching.
pub struct SteeringGate {
    bus: EventBus,
    steering: Arc<SharedSteering>,
    timeout: Duration,
}

impl SteeringGate {
    pub fn new(bus: EventBus, steering: Arc<SharedSteering>) -> Self {
        Self {
            bus,
            steering,
            // Long, because the person may be away from the machine — the
            // product's whole premise is that work continues while they are.
            timeout: Duration::from_secs(3600),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Does this call need a human, under the mode currently in force?
    ///
    /// Separate from asking, so the decision can be tested without a bus and
    /// read without a turn running.
    pub fn needs_approval(mode: Steering, tool: &str, args: &Value) -> Option<&'static str> {
        if DESTRUCTIVE.contains(&tool) {
            return Some("this removes work, and there is no undo but a snapshot");
        }
        if READ_ONLY.contains(&tool) {
            return None;
        }
        match mode {
            Steering::Manual => Some("this session asks before every change"),
            Steering::Skip => None,
            Steering::Auto => {
                // A write with no `old_string` replaces a whole file. The
                // editing tools match on surrounding text and so cannot
                // silently discard what they did not see; a blind write can.
                if tool == "write_file" && args.get("old_string").is_none() {
                    let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
                    if !path.is_empty() {
                        return Some("this replaces a whole file rather than editing part of it");
                    }
                }
                // A shell command is unreviewable by anyone in practice, and
                // the model proposing it may have been told what to propose.
                if tool == "shell" || tool == "bash" {
                    return Some("a generated shell command cannot be meaningfully audited");
                }
                None
            }
        }
    }
}

#[async_trait::async_trait]
impl CycleHook for SteeringGate {
    async fn before_tool(&self, ctx: &HookContext<'_>, name: &str, args: &Value) -> HookAction {
        let mode = self.steering.get();
        let Some(reason) = Self::needs_approval(mode, name, args) else {
            return HookAction::Continue;
        };

        let request_id = uuid::Uuid::new_v4().to_string();
        let request = Event::new(
            eventage::event::kinds::PERMISSION_REQUEST,
            json!({
                "request_id": request_id,
                "tool": name,
                "arguments": args,
                "reason": reason,
                "steering": mode.id(),
                "agent_id": ctx.agent_id,
            }),
        );
        if self.bus.publish(request).await.is_err() {
            return HookAction::Deny("the session is closing".into());
        }

        let wanted = request_id.clone();
        let decided = tokio::time::timeout(
            self.timeout,
            self.bus.wait_for(move |e| {
                e.kind == eventage::event::kinds::PERMISSION_DECISION
                    && e.payload.get("request_id").and_then(|v| v.as_str()) == Some(wanted.as_str())
            }),
        )
        .await;

        match decided {
            Ok(Ok(decision)) => match decision.payload.get("approve").and_then(|v| v.as_bool()) {
                Some(true) => HookAction::Continue,
                _ => HookAction::Deny(
                    decision
                        .payload
                        .get("reason")
                        .and_then(|v| v.as_str())
                        .unwrap_or("the user declined this action")
                        .to_string(),
                ),
            },
            // The bus closed: the session is going away, and a call that
            // slipped through on the way out is the one nobody would see.
            Ok(Err(_)) => HookAction::Deny("the session closed before this was answered".into()),
            Err(_) => {
                warn!(tool = name, "no decision within the approval timeout");
                HookAction::Deny(
                    "nobody answered the approval request in time, so the action was not \
                     taken. Say what you were about to do and why, and let the user decide."
                        .into(),
                )
            }
        }
    }
}

/// Announce a steering change so every surface and the trace agree.
pub async fn announce(bus: &EventBus, mode: Steering) {
    let _ = bus
        .publish(Event::new(
            kinds::STEERING_CHANGED,
            json!({ "steering": mode.id(), "describes": mode.describe() }),
        ))
        .await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deletion_asks_in_every_mode_including_skip() {
        // Cowork's own guarantee, and the reason it is not a mode: a user who
        // chose to stop being interrupted did not choose to have their work
        // deleted uninvited, and it is the one action a snapshot is the only
        // recovery from.
        for mode in [Steering::Manual, Steering::Auto, Steering::Skip] {
            assert!(
                SteeringGate::needs_approval(mode, "delete_file", &json!({"path": "a"})).is_some(),
                "{mode:?} allowed a deletion unasked"
            );
        }
    }

    #[test]
    fn reading_never_asks() {
        for mode in [Steering::Manual, Steering::Auto, Steering::Skip] {
            assert!(SteeringGate::needs_approval(mode, "read_file", &json!({})).is_none());
            assert!(SteeringGate::needs_approval(mode, "grep", &json!({})).is_none());
        }
    }

    #[test]
    fn manual_asks_for_every_change_and_skip_asks_for_none() {
        let write = json!({ "path": "notes.md", "content": "x" });
        assert!(SteeringGate::needs_approval(Steering::Manual, "write_file", &write).is_some());
        assert!(SteeringGate::needs_approval(Steering::Skip, "write_file", &write).is_none());
    }

    #[test]
    fn auto_holds_a_blind_overwrite_but_not_a_targeted_edit() {
        // The distinction that makes `Auto` worth having: an edit matching on
        // surrounding text cannot silently discard what it did not see, and a
        // whole-file write can.
        let blind = json!({ "path": "report.md", "content": "everything, replaced" });
        assert!(SteeringGate::needs_approval(Steering::Auto, "write_file", &blind).is_some());

        let targeted = json!({ "path": "report.md", "old_string": "a", "new_string": "b" });
        assert!(SteeringGate::needs_approval(Steering::Auto, "edit_file", &targeted).is_none());
    }

    #[test]
    fn auto_holds_a_shell_command() {
        assert!(
            SteeringGate::needs_approval(Steering::Auto, "shell", &json!({"command": "ls"}))
                .is_some()
        );
    }

    #[test]
    fn the_mode_can_change_while_work_is_running() {
        // The point of a steering control. A copy taken when the session was
        // built would go stale the moment the user moved the switch.
        let shared = SharedSteering::new(Steering::Manual);
        assert_eq!(shared.get(), Steering::Manual);
        shared.set(Steering::Skip);
        assert_eq!(shared.get(), Steering::Skip);
    }

    #[tokio::test]
    async fn an_unanswered_request_denies_rather_than_proceeds() {
        // An unattended session must not become an unsupervised one just
        // because nobody was watching.
        let bus = EventBus::new();
        let gate = SteeringGate::new(bus.clone(), Arc::new(SharedSteering::new(Steering::Manual)))
            .with_timeout(Duration::from_millis(50));
        let ctx = HookContext {
            agent_id: "a",
            trace_id: "t",
            step: 1,
            bus: &bus,
        };
        let action = gate
            .before_tool(&ctx, "write_file", &json!({"path": "x", "content": "y"}))
            .await;
        assert!(matches!(action, HookAction::Deny(_)), "{action:?}");
    }

    #[tokio::test]
    async fn an_approval_lets_the_call_through() {
        let bus = EventBus::new();
        let gate = SteeringGate::new(bus.clone(), Arc::new(SharedSteering::new(Steering::Manual)))
            .with_timeout(Duration::from_secs(5));

        // Subscribed before the gate runs. `subscribe` only delivers future
        // events, so answering from a task spawned alongside would race the
        // request it is meant to answer — a real surface subscribes when the
        // session opens, long before any turn.
        let mut rx = bus.subscribe();

        let calling = {
            let bus = bus.clone();
            tokio::spawn(async move {
                let ctx = HookContext {
                    agent_id: "a",
                    trace_id: "t",
                    step: 1,
                    bus: &bus,
                };
                gate.before_tool(&ctx, "write_file", &json!({"path": "x", "content": "y"}))
                    .await
            })
        };

        let request = loop {
            let event = rx.recv().await.expect("the bus stayed open");
            if event.kind == eventage::event::kinds::PERMISSION_REQUEST {
                break event;
            }
        };
        assert_eq!(request.payload["tool"], "write_file");
        assert_eq!(request.payload["steering"], "manual");

        let id = request.payload["request_id"].as_str().unwrap().to_string();
        bus.publish(Event::new(
            eventage::event::kinds::PERMISSION_DECISION,
            json!({ "request_id": id, "approve": true }),
        ))
        .await
        .unwrap();

        let action = calling.await.unwrap();
        assert!(matches!(action, HookAction::Continue), "{action:?}");
    }
}
