use async_trait::async_trait;
use eventage::{AgentError, Tool, ToolDefinition};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};
use tokio::sync::Mutex;

// ── TodoItem ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    pub id: u32,
    pub text: String,
    pub completed: bool,
}

// ── TodoState ─────────────────────────────────────────────────────────────────

pub struct TodoState {
    inner: Mutex<Vec<TodoItem>>,
    next_id: AtomicU32,
}

impl TodoState {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(vec![]),
            next_id: AtomicU32::new(1),
        })
    }

    pub async fn add(&self, text: String) -> TodoItem {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let item = TodoItem {
            id,
            text,
            completed: false,
        };
        self.inner.lock().await.push(item.clone());
        item
    }

    pub async fn complete(&self, id: u32) -> Option<TodoItem> {
        let mut list = self.inner.lock().await;
        if let Some(item) = list.iter_mut().find(|i| i.id == id) {
            item.completed = true;
            Some(item.clone())
        } else {
            None
        }
    }

    pub async fn list(&self) -> Vec<TodoItem> {
        self.inner.lock().await.clone()
    }
}

// ── AddTodoTool ───────────────────────────────────────────────────────────────

pub struct AddTodoTool {
    pub state: Arc<TodoState>,
}

#[async_trait]
impl Tool for AddTodoTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "add_todo",
            "Add an item to the todo list.",
            json!({
                "type": "object",
                "properties": {
                    "todo": { "type": "string", "description": "The task to add." }
                },
                "required": ["todo"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let text = args["todo"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'todo'".into()))?
            .to_string();
        let item = self.state.add(text).await;
        Ok(json!({ "id": item.id, "text": item.text, "status": "added" }))
    }
}

// ── CompleteTodoTool ──────────────────────────────────────────────────────────

pub struct CompleteTodoTool {
    pub state: Arc<TodoState>,
}

#[async_trait]
impl Tool for CompleteTodoTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "complete_todo",
            "Mark a todo item as completed by its ID.",
            json!({
                "type": "object",
                "properties": {
                    "id": { "type": "integer", "description": "The todo item ID." }
                },
                "required": ["id"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let id = args["id"]
            .as_u64()
            .ok_or_else(|| AgentError::Tool("missing 'id'".into()))? as u32;

        match self.state.complete(id).await {
            Some(item) => Ok(json!({ "id": item.id, "text": item.text, "status": "completed" })),
            None => Err(AgentError::Tool(format!("no todo with id {id}"))),
        }
    }
}

// ── ListTodosTool ─────────────────────────────────────────────────────────────

pub struct ListTodosTool {
    pub state: Arc<TodoState>,
}

#[async_trait]
impl Tool for ListTodosTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "list_todos",
            "List all todo items (pending and completed).",
            json!({ "type": "object", "properties": {} }),
        )
    }

    async fn execute(&self, _args: Value) -> Result<Value, AgentError> {
        let items = self.state.list().await;
        let pending: Vec<_> = items.iter().filter(|i| !i.completed).collect();
        let done: Vec<_> = items.iter().filter(|i| i.completed).collect();
        Ok(json!({
            "pending": pending,
            "completed": done,
            "total": items.len()
        }))
    }
}
