pub mod fs;
pub mod patch;
pub mod shell;
pub mod task;
pub mod todo;

pub use fs::{EditFileTool, GlobTool, GrepTool, LsTool, ReadFileTool, WriteFileTool};
pub use patch::ApplyPatchTool;
pub use shell::RunCommandTool;
pub use task::{CheckAsyncTaskTool, LaunchAsyncTaskTool, TaskTool};
pub use todo::{AddTodoTool, CompleteTodoTool, ListTodosTool, TodoState};

use eventage::Tool;
use std::path::Path;
use std::sync::Arc;

/// A named sub-agent type available for delegation.
#[derive(Clone, Debug)]
pub struct SubAgentSpec {
    pub name: String,
    pub description: String,
    /// Custom system prompt for this sub-agent type. Empty = use the builder's base prompt.
    pub system_prompt: String,
}

#[allow(dead_code)]
impl SubAgentSpec {
    pub fn general_purpose() -> Self {
        Self {
            name: "general-purpose".into(),
            description: "A general-purpose agent with full tool access for any task.".into(),
            system_prompt: String::new(),
        }
    }

    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        system_prompt: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            system_prompt: system_prompt.into(),
        }
    }
}

/// Build the standard tool set for a sub-agent (no recursive task delegation).
pub fn build_sub_agent_tools(work_dir: &Path) -> Vec<Arc<dyn Tool>> {
    let todo_state = TodoState::new();
    vec![
        Arc::new(LsTool {
            work_dir: work_dir.to_path_buf(),
        }),
        Arc::new(ReadFileTool {
            work_dir: work_dir.to_path_buf(),
        }),
        Arc::new(WriteFileTool {
            work_dir: work_dir.to_path_buf(),
        }),
        Arc::new(EditFileTool {
            work_dir: work_dir.to_path_buf(),
        }),
        Arc::new(ApplyPatchTool {
            work_dir: work_dir.to_path_buf(),
        }),
        Arc::new(GlobTool {
            work_dir: work_dir.to_path_buf(),
        }),
        Arc::new(GrepTool {
            work_dir: work_dir.to_path_buf(),
        }),
        Arc::new(RunCommandTool {
            work_dir: work_dir.to_path_buf(),
        }),
        Arc::new(AddTodoTool {
            state: todo_state.clone(),
        }),
        Arc::new(CompleteTodoTool {
            state: todo_state.clone(),
        }),
        Arc::new(ListTodosTool { state: todo_state }),
    ]
}
