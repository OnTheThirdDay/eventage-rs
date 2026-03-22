//! Custom event kind constants for eventage-claw.

/// Published by `StreamingOpenAiProvider` for each streamed token.
pub const CLAW_STREAM_CHUNK: &str = "claw.stream.chunk";

/// Published by `SecurityGateHook` when a tool call awaits approval.
pub const CLAW_APPROVAL_REQUESTED: &str = "claw.approval.requested";

/// Published by the TUI when the user grants a tool call.
pub const CLAW_APPROVAL_GRANTED: &str = "claw.approval.granted";

/// Published by the TUI when the user denies a tool call.
pub const CLAW_APPROVAL_DENIED: &str = "claw.approval.denied";

/// Published by `ScheduleTaskTool` when a new task is created.
pub const CLAW_SCHEDULE_CREATE: &str = "claw.schedule.create";

/// Published by `SchedulerWorker` when a task is due.
pub const CLAW_SCHEDULE_FIRE: &str = "claw.schedule.fire";

/// Published by `CancelTaskTool` / `PauseTaskTool` when a task changes.
pub const CLAW_SCHEDULE_UPDATE: &str = "claw.schedule.update";

/// Published by `MessageGroupTool` — the core EventBus-as-IPC demonstration.
/// `RelayWorker` subscribes and routes the message to the target group's bus.
pub const CLAW_GROUP_MESSAGE: &str = "claw.group.message";

/// Published by `DelegationReplyWorker` when a sub-agent completes a relay request.
/// `MessageGroupTool` (await_reply=true) waits for this on the shared bus.
/// Kept separate from `CLAW_GROUP_MESSAGE` so `RelayWorker` never re-routes replies.
pub const CLAW_GROUP_REPLY: &str = "claw.group.reply";

/// Published by the TUI when the user switches the active group.
pub const CLAW_GROUP_SWITCH: &str = "claw.group.switch";

/// Published by `RegisterGroupTool` when a new group is registered at runtime.
pub const CLAW_GROUP_REGISTER: &str = "claw.group.register";

/// Published by the session task tools when the in-session task list changes.
/// Distinct from `CLAW_SCHEDULE_*` events which track time-based scheduled tasks.
pub const CLAW_TASK_UPDATED: &str = "claw.task.updated";
