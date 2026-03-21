use super::{build_sub_agent_tools, SubAgentSpec};
use async_trait::async_trait;
use eventage::{
    event::{kinds, Event},
    llm::LlmProvider,
    AgentBuilder, AgentError, EventBus, ReactStrategy, Tool, ToolDefinition,
};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

use crate::kinds::{SUBAGENT_TASK_COMPLETE, SUBAGENT_TASK_ERROR, SUBAGENT_TASK_LAUNCH};

// ── TaskTool (synchronous sub-agent) ─────────────────────────────────────────

/// Spawns an isolated sub-agent with a fresh EventBus and waits for completion.
/// The sub-agent's final assistant message is returned as the tool result.
pub struct TaskTool {
    pub llm: Arc<dyn LlmProvider>,
    pub base_system_prompt: String,
    pub specs: Vec<SubAgentSpec>,
    pub max_steps: usize,
    pub work_dir: PathBuf,
}

#[async_trait]
impl Tool for TaskTool {
    fn definition(&self) -> ToolDefinition {
        let spec_names: Vec<&str> = self.specs.iter().map(|s| s.name.as_str()).collect();
        let descriptions: String = self
            .specs
            .iter()
            .map(|s| format!("- {}: {}", s.name, s.description))
            .collect::<Vec<_>>()
            .join("\n");

        ToolDefinition::function(
            "task",
            format!(
                "Delegate a task to an ephemeral sub-agent with an isolated context window.\n\
                 The sub-agent runs to completion and returns its final response.\n\
                 Available sub-agent types:\n{descriptions}\n\n\
                 Usage notes:\n\
                 - Provide a fully self-contained task description (the sub-agent has no parent context).\n\
                 - Use this for independent subtasks that don't need the parent's conversation history.\n\
                 - Multiple task calls can be made concurrently."
            ),
            json!({
                "type": "object",
                "properties": {
                    "subagent_type": {
                        "type": "string",
                        "description": "Which sub-agent to use.",
                        "enum": spec_names
                    },
                    "description": {
                        "type": "string",
                        "description": "Complete, self-contained task for the sub-agent."
                    }
                },
                "required": ["subagent_type", "description"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let subagent_type = args["subagent_type"].as_str().unwrap_or("general-purpose");
        let description = args["description"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'description'".into()))?;

        let spec = self
            .specs
            .iter()
            .find(|s| s.name == subagent_type)
            .or_else(|| self.specs.first())
            .ok_or_else(|| AgentError::Tool("no sub-agent specs configured".into()))?;

        // Isolated EventBus — no link to parent conversation
        let sub_bus = EventBus::new();

        // System prompt: spec override takes precedence, then base
        let system_prompt = if spec.system_prompt.trim().is_empty() {
            self.base_system_prompt.clone()
        } else {
            format!("{}\n\n{}", spec.system_prompt, self.base_system_prompt)
        };

        // Build sub-agent with standard tool set (no recursive TaskTool)
        let tools = build_sub_agent_tools(&self.work_dir);
        let mut builder = AgentBuilder::new()
            .bus(sub_bus.clone())
            .llm_arc(self.llm.clone())
            .system_prompt(system_prompt)
            .strategy(ReactStrategy {
                max_steps: self.max_steps,
                max_concurrent_tools: 4,
            });

        for tool in tools {
            builder = builder.tool_arc(tool);
        }

        let agent = builder.build();

        // Seed the sub-bus with the task as a user message
        sub_bus
            .publish(Event::new(
                kinds::USER_MESSAGE,
                json!({ "text": description }),
            ))
            .await
            .map_err(|e| AgentError::Tool(format!("bus publish: {e}")))?;

        // Run until completion
        agent.cycle().await?;

        // Extract the last assistant message
        let log = sub_bus.log().await;
        let result = log
            .iter()
            .rev()
            .find(|e| e.kind == kinds::ASSISTANT_MESSAGE)
            .and_then(|e| e.payload["content"].as_str().map(|s| s.to_string()))
            .unwrap_or_else(|| "(sub-agent produced no response)".to_string());

        Ok(json!({ "subagent_type": subagent_type, "result": result }))
    }
}

// ── LaunchAsyncTaskTool ───────────────────────────────────────────────────────

/// Launches a sub-agent in the background via a bus event.
/// The SubAgentWorker (in workers.rs) handles the actual execution.
/// Returns a job_id to track the task with check_async_task.
pub struct LaunchAsyncTaskTool {
    pub bus: EventBus,
    pub specs: Vec<SubAgentSpec>,
}

#[async_trait]
impl Tool for LaunchAsyncTaskTool {
    fn definition(&self) -> ToolDefinition {
        let spec_names: Vec<&str> = self.specs.iter().map(|s| s.name.as_str()).collect();
        let descriptions: String = self
            .specs
            .iter()
            .map(|s| format!("- {}: {}", s.name, s.description))
            .collect::<Vec<_>>()
            .join("\n");

        ToolDefinition::function(
            "launch_async_task",
            format!(
                "Launch a sub-agent in the background. Returns a job_id immediately.\n\
                 Use check_async_task to poll for the result.\n\
                 Use this for independent tasks you want to run in parallel.\n\
                 Available sub-agent types:\n{descriptions}"
            ),
            json!({
                "type": "object",
                "properties": {
                    "subagent_type": {
                        "type": "string",
                        "description": "Which sub-agent to use.",
                        "enum": spec_names
                    },
                    "description": {
                        "type": "string",
                        "description": "Complete, self-contained task for the sub-agent."
                    }
                },
                "required": ["subagent_type", "description"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let subagent_type = args["subagent_type"].as_str().unwrap_or("general-purpose");
        let description = args["description"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'description'".into()))?;

        let job_id = Uuid::new_v4().to_string();

        self.bus
            .publish(Event::new(
                SUBAGENT_TASK_LAUNCH,
                json!({
                    "job_id": job_id,
                    "subagent_type": subagent_type,
                    "description": description
                }),
            ))
            .await
            .map_err(|e| AgentError::Tool(format!("bus publish: {e}")))?;

        Ok(json!({
            "job_id": job_id,
            "status": "launched",
            "message": "Use check_async_task to poll for the result."
        }))
    }
}

// ── CheckAsyncTaskTool ────────────────────────────────────────────────────────

/// Check the status/result of a background sub-agent job.
pub struct CheckAsyncTaskTool {
    pub bus: EventBus,
}

#[async_trait]
impl Tool for CheckAsyncTaskTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "check_async_task",
            "Check the status of a background sub-agent task launched with launch_async_task.",
            json!({
                "type": "object",
                "properties": {
                    "job_id": { "type": "string", "description": "The job_id returned by launch_async_task." }
                },
                "required": ["job_id"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let job_id = args["job_id"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'job_id'".into()))?;

        let log = self.bus.log().await;

        // Check for completion event
        if let Some(ev) = log.iter().find(|e| {
            e.kind == SUBAGENT_TASK_COMPLETE
                && e.payload.get("job_id").and_then(|v| v.as_str()) == Some(job_id)
        }) {
            let result = ev.payload["result"].as_str().unwrap_or("").to_string();
            return Ok(json!({ "job_id": job_id, "status": "complete", "result": result }));
        }

        // Check for error event
        if let Some(ev) = log.iter().find(|e| {
            e.kind == SUBAGENT_TASK_ERROR
                && e.payload.get("job_id").and_then(|v| v.as_str()) == Some(job_id)
        }) {
            let error = ev.payload["error"]
                .as_str()
                .unwrap_or("unknown error")
                .to_string();
            return Ok(json!({ "job_id": job_id, "status": "error", "error": error }));
        }

        // Check if the launch event exists (otherwise invalid job_id)
        let launched = log.iter().any(|e| {
            e.kind == SUBAGENT_TASK_LAUNCH
                && e.payload.get("job_id").and_then(|v| v.as_str()) == Some(job_id)
        });

        if launched {
            Ok(json!({ "job_id": job_id, "status": "running" }))
        } else {
            Err(AgentError::Tool(format!("unknown job_id: {job_id}")))
        }
    }
}
