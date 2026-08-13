use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

pub type EventId = Uuid;

/// A structured, immutable event traversing the [`EventBus`](crate::EventBus).
///
/// Events form a Directed Acyclic Graph (DAG) linked by `parent_event_id`.
/// The bus automatically maintains this lineage to support checkpointing and branching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: EventId,
    pub timestamp: DateTime<Utc>,
    /// Dot-separated domain identifier (e.g., `"user.message"`).
    pub kind: String,
    pub payload: serde_json::Value,
    /// ID of the preceding event in the active branch.
    ///
    /// Set automatically by [`EventBus::publish`](crate::EventBus::publish). `None` for the root event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_event_id: Option<EventId>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Event {
    pub fn new(kind: impl Into<String>, payload: serde_json::Value) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            kind: kind.into(),
            payload,
            parent_event_id: None, // filled in by EventBus::publish
            metadata: HashMap::new(),
        }
    }

    pub fn with_meta(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set an explicit parent event ID on this event before publishing.
    ///
    /// By default [`EventBus::publish`](crate::EventBus::publish) sets `parent_event_id` to the
    /// current branch tip. Use this builder to override that with a specific event ID — for
    /// example, to link a tool result directly back to the call event that produced it.
    pub fn with_parent(mut self, parent: EventId) -> Self {
        self.parent_event_id = Some(parent);
        self
    }
}

/// Predefined event kinds used throughout the framework.
pub mod kinds {
    pub const USER_MESSAGE: &str = "user.message";
    pub const ASSISTANT_MESSAGE: &str = "assistant.message";
    pub const TOOL_CALL_PROPOSED: &str = "tool.call.proposed";
    pub const TOOL_RESULT: &str = "tool.result";
    pub const SYSTEM_HEARTBEAT: &str = "system.heartbeat";
    /// Message originating from system infrastructure (e.g. task scheduler,
    /// hooks, automated pipelines) — not from a human user or another agent.
    pub const SYSTEM_MESSAGE: &str = "system.message";
    pub const AGENT_CYCLE_START: &str = "agent.cycle.start";
    pub const AGENT_CYCLE_END: &str = "agent.cycle.end";
    /// Emitted by [`EventBus::checkpoint`](crate::EventBus::checkpoint) to mark a safe rollback point.
    pub const CHECKPOINT: &str = "system.checkpoint";
    /// Emitted after a successful rollback to indicate a branch was rejected.
    /// **Broadcast-only** (never stored in the DAG); see [`SYSTEM_ROLLBACK`]
    /// for the durable record.
    pub const BRANCH_SEALED: &str = "system.branch_sealed";
    /// Durable tombstone appended to the new active tip by
    /// [`EventBus::rollback`](crate::EventBus::rollback). Records the branch
    /// topology so persisted logs can be restored faithfully — without it, a
    /// restore would resurrect rolled-back events into the active branch.
    /// Payload: `{ "branch_id", "checkpoint_event_id", "parent_event_id"?,
    /// "rejected_event_ids": [...] }`.
    pub const SYSTEM_ROLLBACK: &str = "system.rollback";
    /// Emitted when evicted branches are removed from memory.
    ///
    /// Payload: `{ "evicted_branches": usize, "evicted_nodes": usize }`.
    pub const SYSTEM_PRUNED: &str = "system.pruned";

    // ── Multi-agent ───────────────────────────────────────────────────────────
    /// Directed message to an agent (or broadcast if `TO_AGENT_ID` is missing).
    pub const AGENT_MESSAGE: &str = "agent.message";
    /// Emitted when an agent spawns a child agent.
    pub const AGENT_SPAWNED: &str = "agent.spawned";
    /// Emitted when an agent's execution successfully completes.
    pub const AGENT_COMPLETED: &str = "agent.completed";

    // ── Self-correction ───────────────────────────────────────────────────────
    /// Emitted before a cycle when the agent detects it is repeating the same
    /// actions or errors. Payload includes `"kind"` and `"hint"` so the LLM
    /// can read the event and try a different approach.
    pub const AGENT_STUCK: &str = "agent.stuck";

    // ── Context management ────────────────────────────────────────────────────
    /// Emitted by [`SummarizingContextAssembler`](crate::SummarizingContextAssembler) when
    /// old conversation history is summarized to keep the context window bounded.
    /// Payload includes `"summary_len"`, `"summarized_events"`, and `"retained_events"`.
    pub const AGENT_CONTEXT_SUMMARIZED: &str = "agent.context.summarized";

    // ── Streaming (ephemeral — broadcast only, never stored in the DAG) ──────
    /// Incremental completion text emitted while an LLM response streams.
    /// Payload: `{ "content": Option<str>, "reasoning_content": Option<str> }`.
    /// Broadcast via [`EventBus::broadcast`](crate::EventBus::broadcast); the
    /// complete text still arrives as a durable `assistant.message`.
    pub const ASSISTANT_DELTA: &str = "assistant.delta";

    // ── Governance ────────────────────────────────────────────────────────────
    /// Emitted by `PermissionPolicyHook` when a tool call needs approval.
    /// Payload: `{ "request_id", "tool", "arguments" }`.
    pub const PERMISSION_REQUEST: &str = "permission.request";
    /// Approval verdict for a pending `permission.request`.
    /// Payload: `{ "request_id", "approve": bool, "reason"? }`.
    pub const PERMISSION_DECISION: &str = "permission.decision";
    /// Emitted by `TokenBudgetHook` when usage crosses the warn threshold.
    /// Payload: `{ "used_tokens", "max_tokens" }`.
    pub const BUDGET_WARNING: &str = "budget.warning";
    /// Emitted by `TokenBudgetHook` when the budget is exhausted; the cycle
    /// is aborted. Payload: `{ "used_tokens", "max_tokens" }`.
    pub const BUDGET_EXHAUSTED: &str = "budget.exhausted";

    // ── MCP ───────────────────────────────────────────────────────────────────
    /// An MCP server asked the user for structured input (2025-06-18
    /// elicitation). Payload: `{ "request_id", "server", "message", "schema" }`.
    /// Answer by publishing [`MCP_ELICITATION_RESPONSE`].
    pub const MCP_ELICITATION_REQUEST: &str = "mcp.elicitation.request";
    /// Answer to an [`MCP_ELICITATION_REQUEST`].
    /// Payload: `{ "request_id", "action": "accept"|"decline"|"cancel",
    /// "content": { ... } }`.
    pub const MCP_ELICITATION_RESPONSE: &str = "mcp.elicitation.response";
    /// An MCP server announced that its tool list changed; re-run
    /// `McpToolset::reload`. Payload: `{ "server" }`.
    pub const MCP_TOOLS_CHANGED: &str = "mcp.tools.changed";

    // ── Recovery ──────────────────────────────────────────────────────────────
    /// Emitted after a resume reconciles tool calls that were interrupted by
    /// a restart. Payload: `{ "interrupted_tool_calls", "replayed", "reported" }`.
    pub const SYSTEM_RECOVERED: &str = "system.recovered";

    // ── Speculation ───────────────────────────────────────────────────────────
    /// Emitted after a speculative best-of-N round completes.
    /// Payload: `{ "candidates", "winner_index", "scores" }`.
    pub const SPECULATION_COMPLETED: &str = "speculation.completed";
}

/// Standard metadata keys for [`Event::metadata`].
pub mod meta_keys {
    /// ID of the emitting agent.
    pub const AGENT_ID: &str = "agent_id";
    /// Trace ID grouping events within a single reasoning or execution cycle.
    pub const TRACE_ID: &str = "trace_id";
    /// Recipient agent ID for targeted messages.
    pub const TO_AGENT_ID: &str = "to_agent_id";
    /// Duration of a cycle in milliseconds.
    pub const ELAPSED_MS: &str = "elapsed_ms";
    /// Prompt tokens consumed in the last LLM interaction.
    pub const LLM_INPUT_TOKENS: &str = "llm_input_tokens";
    /// Completion tokens generated in the last LLM interaction.
    pub const LLM_OUTPUT_TOKENS: &str = "llm_output_tokens";
    /// Prompt tokens served from the provider's cache in the last LLM interaction.
    pub const LLM_CACHED_INPUT_TOKENS: &str = "llm_cached_input_tokens";
    /// Harness estimate of the prompt tokens for the request that produced
    /// this message. Paired with [`LLM_INPUT_TOKENS`] it lets
    /// [`TokenCalibration`](crate::agent::tokens::TokenCalibration) learn the
    /// estimator's error online.
    pub const LLM_ESTIMATED_INPUT_TOKENS: &str = "llm_estimated_input_tokens";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_roundtrip_json() {
        let e = Event::new(kinds::USER_MESSAGE, serde_json::json!({"text": "hello"}));
        let json = serde_json::to_string(&e).unwrap();
        let decoded: Event = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.kind, kinds::USER_MESSAGE);
        assert_eq!(decoded.payload["text"], "hello");
    }
}
