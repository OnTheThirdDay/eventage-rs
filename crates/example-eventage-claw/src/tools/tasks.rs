//! In-session task tracking tools.
//!
//! Three tools share a single [`TaskState`] and let the agent maintain an
//! explicit todo list for the current session. This is distinct from the
//! scheduler (`schedule.rs`) which fires messages at future times — this is
//! a simple checklist visible to the LLM and the user.
//!
//! # Tools
//! - [`AddTaskTool`]      — add a top-level or nested task
//! - [`CompleteTaskTool`] — mark a task done by id or title
//! - `ListTasksTool`    — return the current task tree as formatted text

use async_trait::async_trait;
use eventage::{AgentError, Event, EventBus, Tool, ToolDefinition};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use crate::kinds::CLAW_TASK_UPDATED;

// ── Data model ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Task {
    pub id: String,
    pub title: String,
    pub done: bool,
    pub subtasks: Vec<Task>,
}

impl Task {
    fn new(title: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string()[..8].to_string(),
            title: title.into(),
            done: false,
            subtasks: vec![],
        }
    }

    fn find_mut(&mut self, id_or_title: &str) -> Option<&mut Task> {
        if self.id == id_or_title || self.title.eq_ignore_ascii_case(id_or_title) {
            return Some(self);
        }
        for sub in &mut self.subtasks {
            if let Some(found) = sub.find_mut(id_or_title) {
                return Some(found);
            }
        }
        None
    }

    fn render(&self, indent: usize) -> String {
        let mark = if self.done { "✓" } else { "○" };
        let prefix = " ".repeat(indent * 2);
        let mut out = format!("{prefix}{mark} [{}] {}\n", self.id, self.title);
        for sub in &self.subtasks {
            out.push_str(&sub.render(indent + 1));
        }
        out
    }
}

/// Shared in-session task list.
pub type TaskState = Arc<Mutex<Vec<Task>>>;

pub fn new_task_state() -> TaskState {
    Arc::new(Mutex::new(Vec::new()))
}

fn render_all(tasks: &[Task]) -> String {
    if tasks.is_empty() {
        return "No tasks in the current session.".to_string();
    }
    let done = count_done(tasks);
    let total = count_total(tasks);
    let mut out = format!("Session tasks ({done}/{total} done):\n");
    for task in tasks {
        out.push_str(&task.render(0));
    }
    out
}

fn count_done(tasks: &[Task]) -> usize {
    tasks
        .iter()
        .map(|t| (if t.done { 1 } else { 0 }) + count_done(&t.subtasks))
        .sum()
}

fn count_total(tasks: &[Task]) -> usize {
    tasks.iter().map(|t| 1 + count_total(&t.subtasks)).sum()
}

// ── AddTaskTool ───────────────────────────────────────────────────────────────

/// Add a top-level or nested task to the current session's task list.
pub struct AddTaskTool {
    pub state: TaskState,
    pub bus: EventBus,
}

#[async_trait]
impl Tool for AddTaskTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "add_task",
            "Add a task to the current session's todo list. \
             Use parent_id to nest it under an existing task.",
            json!({
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short description of the task."
                    },
                    "parent_id": {
                        "type": "string",
                        "description": "Optional id of the parent task (for subtasks)."
                    }
                },
                "required": ["title"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let title = args["title"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'title'".into()))?;
        let parent_id = args["parent_id"].as_str();

        let new_task = Task::new(title);
        let new_id = new_task.id.clone();

        {
            let mut tasks = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(pid) = parent_id {
                let mut found = false;
                for t in tasks.iter_mut() {
                    if let Some(parent) = t.find_mut(pid) {
                        parent.subtasks.push(new_task.clone());
                        found = true;
                        break;
                    }
                }
                if !found {
                    return Err(AgentError::Tool(format!("parent task '{pid}' not found")));
                }
            } else {
                tasks.push(new_task);
            }
        }

        let _ = self
            .bus
            .publish(Event::new(
                CLAW_TASK_UPDATED,
                json!({ "action": "add", "task_id": new_id }),
            ))
            .await;

        Ok(json!({ "task_id": new_id, "added": true }))
    }
}

// ── CompleteTaskTool ──────────────────────────────────────────────────────────

/// Mark a task done by id or (case-insensitive) title.
pub struct CompleteTaskTool {
    pub state: TaskState,
    pub bus: EventBus,
}

#[async_trait]
impl Tool for CompleteTaskTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "complete_task",
            "Mark a session task as done by its id or title.",
            json!({
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task id or title to mark done."
                    }
                },
                "required": ["task_id"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let key = args["task_id"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'task_id'".into()))?;

        let found = {
            let mut tasks = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let mut found = false;
            for t in tasks.iter_mut() {
                if let Some(task) = t.find_mut(key) {
                    task.done = true;
                    found = true;
                    break;
                }
            }
            found
        };

        if !found {
            return Err(AgentError::Tool(format!("task '{key}' not found")));
        }

        let _ = self
            .bus
            .publish(Event::new(
                CLAW_TASK_UPDATED,
                json!({ "action": "complete", "task_id": key }),
            ))
            .await;

        Ok(json!({ "completed": true }))
    }
}

// ── ListTasksTool ─────────────────────────────────────────────────────────────

/// Return the current session task list as formatted text.
pub struct ListSessionTasksTool {
    pub state: TaskState,
}

#[async_trait]
impl Tool for ListSessionTasksTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "list_session_tasks",
            "Show the current session's task list with completion status.",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        )
    }

    async fn execute(&self, _args: Value) -> Result<Value, AgentError> {
        let tasks = self.state.lock().unwrap_or_else(|e| e.into_inner());
        Ok(json!({ "tasks": render_all(&tasks) }))
    }
}
