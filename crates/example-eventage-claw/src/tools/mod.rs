//! Tool implementations for eventage-claw.

pub mod browser;
pub mod docker;
pub mod fs;
pub mod group;
pub mod relay;
pub mod schedule;
pub mod shell;
pub mod tasks;
/// Web access now lives in the framework so the coding agent shares it.
pub use eventage::agent::web;

pub use browser::BrowserTool;
pub use docker::DockerRunCommandTool;
pub use fs::{EditFileTool, GlobTool, GrepTool, LsTool, ReadFileTool, WriteFileTool};
pub use group::{
    new_group_registry, AgentSpawner, GroupRegistry, ListGroupsTool, RegisterGroupTool,
    SpawnGroupTool,
};
pub use relay::MessageGroupTool;
pub use schedule::{
    load_tasks, CancelTaskTool, ListTasksTool, PauseTaskTool, ScheduleState, ScheduleTaskTool,
    UpdateTaskTool,
};
pub use tasks::{new_task_state, AddTaskTool, CompleteTaskTool, ListSessionTasksTool, TaskState};
pub use web::{WebFetchTool, WebSearchTool};
