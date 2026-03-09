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
}

/// Predefined event kinds used throughout the framework.
pub mod kinds {
    pub const USER_MESSAGE: &str = "user.message";
    pub const ASSISTANT_MESSAGE: &str = "assistant.message";
    pub const TOOL_CALL_PROPOSED: &str = "tool.call.proposed";
    pub const TOOL_RESULT: &str = "tool.result";
    pub const SYSTEM_HEARTBEAT: &str = "system.heartbeat";
    pub const AGENT_CYCLE_START: &str = "agent.cycle.start";
    pub const AGENT_CYCLE_END: &str = "agent.cycle.end";
    /// Emitted by [`EventBus::checkpoint`](crate::EventBus::checkpoint) to mark a safe rollback point.
    pub const CHECKPOINT: &str = "system.checkpoint";
    /// Emitted after a successful rollback to indicate a branch was rejected.
    pub const BRANCH_SEALED: &str = "system.branch_sealed";
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
