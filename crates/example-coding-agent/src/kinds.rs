//! Custom event kind constants for the coding agent.

/// Published by `StreamingOpenAiProvider` for each streamed token.
pub const CODING_STREAM_CHUNK: &str = "coding.stream.chunk";

/// Published by `SecurityGateHook` when a dangerous tool call awaits approval.
pub const CODING_APPROVAL_REQUESTED: &str = "coding.approval.requested";

/// Published by the TUI when the user grants a tool call.
pub const CODING_APPROVAL_GRANTED: &str = "coding.approval.granted";

/// Published by the TUI when the user denies a tool call.
pub const CODING_APPROVAL_DENIED: &str = "coding.approval.denied";

/// Published by `TurnDiffWorker` with unified diffs of all changed files.
pub const CODING_TURN_DIFF: &str = "coding.turn.diff";

/// Published by `SummarizingAssembler` when the context is summarised.
#[allow(dead_code)]
pub const CODING_CONTEXT_COMPACTED: &str = "coding.context.compacted";

/// Published by `LaunchAsyncTaskTool` to request a background sub-agent.
pub const SUBAGENT_TASK_LAUNCH: &str = "subagent.task.launch";

/// Published by `SubAgentWorker` when a background sub-agent completes.
pub const SUBAGENT_TASK_COMPLETE: &str = "subagent.task.complete";

/// Published by `SubAgentWorker` when a background sub-agent fails.
pub const SUBAGENT_TASK_ERROR: &str = "subagent.task.error";
