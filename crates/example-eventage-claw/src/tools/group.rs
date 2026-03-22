//! Group management tools — only registered for the "main" group.

use async_trait::async_trait;
use eventage::{AgentError, Event, EventBus, Tool, ToolDefinition};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::kinds::CLAW_GROUP_REGISTER;

// ── AgentSpawner trait ────────────────────────────────────────────────────────

/// Abstracts over the runtime agent-spawning logic so `SpawnGroupTool` does
/// not need to import from `agent.rs` (which would create a circular dep).
#[async_trait]
pub trait AgentSpawner: Send + Sync {
    async fn spawn(&self, name: &str, system_prompt: Option<&str>) -> Result<(), String>;
}

/// Shared runtime group registry (name → bool for "active").
pub type GroupRegistry = Arc<Mutex<Vec<String>>>;

pub fn new_group_registry(names: Vec<String>) -> GroupRegistry {
    Arc::new(Mutex::new(names))
}

// ── RegisterGroupTool ─────────────────────────────────────────────────────────

pub struct RegisterGroupTool {
    pub bus: EventBus,
    pub registry: GroupRegistry,
}

#[async_trait]
impl Tool for RegisterGroupTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "register_group",
            "Register a new named group at runtime. The group gets its own isolated \
             context and event bus. Only the main group can register new groups.",
            json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique group name (e.g. 'research', 'work')."
                    },
                    "system_prompt_suffix": {
                        "type": "string",
                        "description": "Optional custom persona for this group."
                    }
                },
                "required": ["name"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let name = args["name"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'name'".into()))?;
        let suffix = args["system_prompt_suffix"].as_str();

        let mut registry = self.registry.lock().await;
        if registry.iter().any(|n| n == name) {
            return Ok(json!({
                "name": name,
                "registered": false,
                "message": "group already exists",
            }));
        }
        registry.push(name.to_string());

        let _ = self
            .bus
            .publish(Event::new(
                CLAW_GROUP_REGISTER,
                json!({
                    "name": name,
                    "system_prompt_suffix": suffix,
                }),
            ))
            .await;

        Ok(json!({
            "name": name,
            "registered": true,
            "message": format!("Group '{name}' registered. Restart claw to activate it, or use message_group to send it messages."),
        }))
    }
}

// ── ListGroupsTool ────────────────────────────────────────────────────────────

pub struct ListGroupsTool {
    pub registry: GroupRegistry,
}

#[async_trait]
impl Tool for ListGroupsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "list_groups",
            "List all active groups and their names.",
            json!({ "type": "object", "properties": {} }),
        )
    }

    async fn execute(&self, _args: Value) -> Result<Value, AgentError> {
        let registry = self.registry.lock().await;
        Ok(json!({
            "groups": *registry,
            "count": registry.len(),
        }))
    }
}

// ── SpawnGroupTool ────────────────────────────────────────────────────────────

/// Dynamically spawns a new sub-agent group at runtime.
///
/// The new agent gets its own isolated EventBus, the full tool set, and begins
/// listening immediately — no restart required.  `message_group` can reach it
/// the moment this tool returns.  Only the main group has this tool.
pub struct SpawnGroupTool {
    pub spawner: Arc<dyn AgentSpawner>,
}

#[async_trait]
impl Tool for SpawnGroupTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "spawn_group",
            "Spawn a new sub-agent group at runtime with an isolated context and event bus. \
             The agent starts immediately. Use message_group to send it a task and receive \
             its reply. Only usable by the main group.",
            json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Unique group name for the new agent (e.g. 'researcher', 'coder')."
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "System prompt that defines this agent's persona and task focus."
                    }
                },
                "required": ["name", "system_prompt"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let name = args["name"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'name'".into()))?;
        let prompt = args["system_prompt"].as_str();

        self.spawner
            .spawn(name, prompt)
            .await
            .map_err(AgentError::Tool)?;

        Ok(json!({
            "spawned": true,
            "name": name,
            "message": format!("Agent '{name}' is running. Use message_group to send it a task — it will block until the sub-agent replies (up to 30s)."),
        }))
    }
}
