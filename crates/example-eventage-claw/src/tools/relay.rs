//! Inter-group IPC tool — the key EventBus-as-IPC demonstration.
//!
//! `MessageGroupTool` lets an agent send a message to another named group.
//! It publishes a `CLAW_GROUP_MESSAGE` event on the shared bus; the
//! `RelayWorker` subscribes and routes it to the target group's per-group bus
//! as a `user.message` event.

use async_trait::async_trait;
use eventage::{AgentError, Event, EventBus, Tool, ToolDefinition};
use serde_json::{json, Value};
use std::time::Duration;
use uuid::Uuid;

use crate::kinds::CLAW_GROUP_MESSAGE;

pub struct MessageGroupTool {
    /// Shared bus — events published here are visible to all workers.
    pub shared_bus: EventBus,
    /// Names of all known groups — included in the schema so the LLM knows valid targets.
    pub known_groups: Vec<String>,
    /// Name of the group that owns this tool instance (source group).
    pub source_group: String,
}

#[async_trait]
impl Tool for MessageGroupTool {
    fn definition(&self) -> ToolDefinition {
        let groups_hint = if self.known_groups.is_empty() {
            String::new()
        } else {
            format!(" Known groups: {}.", self.known_groups.join(", "))
        };
        ToolDefinition::function(
            "message_group",
            format!(
                "Send a message to another named group's agent via the event bus. \
                 This demonstrates eventage's EventBus-as-IPC: no files, no sockets — \
                 just events.{groups_hint}"
            ),
            json!({
                "type": "object",
                "properties": {
                    "target_group": {
                        "type": "string",
                        "description": "Name of the target group (e.g. 'work', 'personal')."
                    },
                    "message": {
                        "type": "string",
                        "description": "Message to deliver to the target group's agent."
                    },
                    "await_reply": {
                        "type": "boolean",
                        "description": "If true, wait up to 30s for a reply from the target group (default: false)."
                    }
                },
                "required": ["target_group", "message"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let target = args["target_group"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'target_group'".into()))?;
        let message = args["message"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'message'".into()))?;
        let await_reply = args["await_reply"].as_bool().unwrap_or(false);

        let msg_id = Uuid::new_v4().to_string();

        // Publish the IPC event — RelayWorker will route this to target's bus.
        self.shared_bus
            .publish(Event::new(
                CLAW_GROUP_MESSAGE,
                json!({
                    "message_id": msg_id,
                    "target_group": target,
                    "source_group": self.source_group,
                    "content": message,
                }),
            ))
            .await
            .map_err(|e| AgentError::Tool(format!("bus publish failed: {e}")))?;

        if await_reply {
            let bus = self.shared_bus.clone();
            let target_owned = target.to_string();
            let msg_id_clone = msg_id.clone();

            let result = tokio::time::timeout(Duration::from_secs(30), async move {
                bus.wait_for(|e: &Event| {
                    e.kind == CLAW_GROUP_MESSAGE
                        && e.payload["source_group"].as_str() == Some(&target_owned)
                        && e.payload["in_reply_to"].as_str() == Some(&msg_id_clone)
                })
                .await
            })
            .await;

            match result {
                Ok(Ok(reply_event)) => Ok(json!({
                    "delivered": true,
                    "message_id": msg_id,
                    "reply": reply_event.payload["content"],
                })),
                Ok(Err(_)) | Err(_) => Ok(json!({
                    "delivered": true,
                    "message_id": msg_id,
                    "reply": null,
                    "timeout": true,
                })),
            }
        } else {
            Ok(json!({
                "delivered": true,
                "message_id": msg_id,
            }))
        }
    }
}
