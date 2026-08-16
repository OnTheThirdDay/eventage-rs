//! Event kinds cowork adds to the framework's own.
//!
//! Everything a surface needs to render a session is on the bus: the goal, the
//! plan, each workstream's life, what it changed, and what was kept. Nothing
//! is held in a struct that only the process that made it can see — which is
//! what makes a session resumable, and what lets Studio show one it did not
//! start.

/// The task the user described. Payload: `{ goal }`.
pub const GOAL_SET: &str = "cowork.goal.set";

/// Somebody asked for work to start. Payload: `{ goal, source }`.
///
/// Distinct from [`GOAL_SET`], which records that a run *began*. This is the
/// request, and it is what the HTTP channel and the scheduler publish — so a
/// goal arriving from a phone and one arriving from a cron entry take exactly
/// the same path as one typed into Studio.
pub const GOAL_REQUESTED: &str = "cowork.goal.requested";

/// How the goal was split. Payload: `{ workstreams: [{ id, title, brief }] }`.
pub const PLAN_PROPOSED: &str = "cowork.plan.proposed";

/// A workstream began. Payload: `{ id, title, brief, worktree }`.
pub const WORKSTREAM_STARTED: &str = "cowork.workstream.started";

/// A workstream finished. Payload: `{ id, report, commit, changes }`.
pub const WORKSTREAM_FINISHED: &str = "cowork.workstream.finished";

/// A workstream could not finish. Payload: `{ id, title, error }`.
///
/// Published rather than only logged, because a resumed session has to know
/// that a stream was attempted and failed — otherwise it comes back looking
/// as though it never ran.
pub const WORKSTREAM_FAILED: &str = "cowork.workstream.failed";

/// A workstream was abandoned. Payload: `{ id, epitaph }`.
///
/// Sealed rather than deleted: its trajectory stays in the DAG as a rejected
/// branch, and the coordinator reads it back as something already tried.
pub const WORKSTREAM_SEALED: &str = "cowork.workstream.sealed";

/// An adoption was refused because the folder moved. Payload:
/// `{ id, title, conflicts: [{ path, workstream, live }] }`.
///
/// A refusal is a fact about the session, not an error to be swallowed: the
/// user has two versions of the same files and has to choose.
pub const ADOPTION_BLOCKED: &str = "cowork.adoption.blocked";

/// A workstream's result was applied to the folder. Payload: `{ id, changes }`.
pub const ADOPTED: &str = "cowork.adopted";

/// The steering mode changed. Payload: `{ steering, describes }`.
pub const STEERING_CHANGED: &str = "cowork.steering.changed";

/// Parts of the folder cowork is not tracking. Payload: `{ repositories }`.
///
/// Published once at session start. A rewind that quietly skipped half the
/// folder would be worse than one that refused.
pub const NOT_TRACKED: &str = "cowork.not_tracked";

// ── automations ───────────────────────────────────────────────────────────────

/// A recurring goal was created. Payload: `{ id, name, schedule }`.
pub const SCHEDULE_CREATE: &str = "cowork.schedule.create";

/// A recurring goal came due, and is about to run. Payload: `{ id, name }`.
pub const SCHEDULE_FIRE: &str = "cowork.schedule.fire";

/// A recurring goal was paused, resumed or cancelled. Payload: `{ id, state }`.
pub const SCHEDULE_UPDATE: &str = "cowork.schedule.update";
