//! Scheduled task tools.
//!
//! Agents create tasks by calling `schedule_task`; `SchedulerWorker` fires them
//! on heartbeat events. State is in-memory (`Arc<Mutex<Vec<ScheduledTask>>>`),
//! persisted across restarts via session JSONL.

use async_trait::async_trait;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use eventage::{AgentError, Event, EventBus, Tool, ToolDefinition};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::kinds::{CLAW_SCHEDULE_CREATE, CLAW_SCHEDULE_UPDATE};

// ── ScheduledTask ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduleKind {
    /// Run once at a specific time.
    Once { at: DateTime<Utc> },
    /// Repeat every N seconds.
    Interval { seconds: u64 },
    /// Cron expression (e.g. "0 9 * * 1-5").
    Cron { expression: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledTask {
    pub id: String,
    pub name: String,
    /// Injected as a `user.message` event when the task fires.
    pub description: String,
    pub schedule_kind: ScheduleKind,
    /// Target group name. None = fire in the creating group's context.
    pub target_group: Option<String>,
    /// If set, the task fires via the relay mechanism (`CLAW_GROUP_MESSAGE`)
    /// so the target group handles it and routes the reply back here.
    /// Used by sub-agents so their scheduled task results reach the main agent.
    #[serde(default)]
    pub reply_group: Option<String>,
    pub next_fire: DateTime<Utc>,
    pub paused: bool,
    pub fired_count: u64,
    pub created_at: DateTime<Utc>,
}

pub type ScheduleState = Arc<Mutex<Vec<ScheduledTask>>>;

// ── Persistence helpers ───────────────────────────────────────────────────────

/// Persist the current task list to `path` as pretty-printed JSON.
/// Logs a warning on failure — the in-memory state remains authoritative.
pub fn save_tasks(tasks: &[ScheduledTask], path: &Path) {
    if let Some(parent) = path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            tracing::warn!(
                "save_tasks: could not create directory {}: {e}",
                parent.display()
            );
        }
    }
    match serde_json::to_string_pretty(tasks) {
        Ok(json) => {
            if let Err(e) = std::fs::write(path, &json) {
                tracing::warn!("save_tasks: failed to write {}: {e}", path.display());
            }
        }
        Err(e) => tracing::warn!("save_tasks: serialization failed: {e}"),
    }
}

/// Persist `snapshot` to `tasks_path` if both are `Some`.
/// Called after releasing the state mutex so disk I/O does not block lock holders.
fn save_if_needed(snapshot: Option<Vec<ScheduledTask>>, tasks_path: &Option<PathBuf>) {
    if let (Some(tasks), Some(p)) = (snapshot, tasks_path.as_ref()) {
        save_tasks(&tasks, p);
    }
}

/// Load the task list from `path`. Returns an empty vec if the file does not
/// exist or cannot be parsed.
pub fn load_tasks(path: &Path) -> Vec<ScheduledTask> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

// ── Schedule string parser ────────────────────────────────────────────────────

/// Parse a schedule string into a `ScheduleKind` and the first `next_fire`.
///
/// Supported formats:
/// - ISO 8601 datetime → `Once`
/// - `"every Ns"` / `"every N seconds"` / `"every N minutes"` / `"every N hours"` → `Interval`
/// - Cron expression (5 fields) → `Cron`
pub fn parse_schedule(s: &str) -> Result<(ScheduleKind, DateTime<Utc>), String> {
    let s = s.trim();
    let now = Utc::now();

    // ISO 8601 datetime
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        let at = dt.with_timezone(&Utc);
        return Ok((ScheduleKind::Once { at }, at));
    }

    // "every N unit"
    let lower = s.to_lowercase();
    if lower.starts_with("every ") {
        let rest = lower.trim_start_matches("every ").trim();
        let parts: Vec<&str> = rest.splitn(2, ' ').collect();
        let n: u64 = parts[0]
            .trim_end_matches('s') // strip trailing 's' from e.g. "10s"
            .parse()
            .map_err(|_| format!("invalid interval: '{rest}'"))?;
        let unit = parts.get(1).copied().unwrap_or("s");
        let seconds = match unit {
            "s" | "sec" | "second" | "seconds" => n,
            "m" | "min" | "minute" | "minutes" => n * 60,
            "h" | "hr" | "hour" | "hours" => n * 3600,
            _ => n, // assume seconds
        };
        let secs_i64 = i64::try_from(seconds).unwrap_or(i64::MAX);
        let next = now + ChronoDuration::seconds(secs_i64);
        return Ok((ScheduleKind::Interval { seconds }, next));
    }

    // Cron expression (5 whitespace-separated fields)
    let fields: Vec<&str> = s.split_whitespace().collect();
    if fields.len() == 5 {
        // Validate by parsing once
        let next = next_cron_fire(s, now).map_err(|e| format!("invalid cron '{s}': {e}"))?;
        return Ok((
            ScheduleKind::Cron {
                expression: s.to_string(),
            },
            next,
        ));
    }

    Err(format!(
        "unrecognized schedule format: '{s}'. Use ISO 8601, 'every Ns', or a 5-field cron expression."
    ))
}

/// Compute the next fire time for a cron expression after `after`.
pub fn next_cron_fire(expression: &str, after: DateTime<Utc>) -> Result<DateTime<Utc>, String> {
    use cron::Schedule;
    use std::str::FromStr;
    // The `cron` crate uses 6-field (with seconds) or 7-field schedules.
    // Standard 5-field cron (min, hour, dom, month, dow) → prepend "0 " for seconds.
    let six_field = format!("0 {expression}");
    let schedule = Schedule::from_str(&six_field).map_err(|e| format!("cron parse error: {e}"))?;
    schedule
        .after(&after)
        .next()
        .ok_or_else(|| "cron schedule has no future fires".to_string())
}

/// Compute the next fire time after the current one.
pub fn advance_schedule(kind: &ScheduleKind, last_fire: DateTime<Utc>) -> Option<DateTime<Utc>> {
    match kind {
        ScheduleKind::Once { .. } => None, // fires once, then done
        ScheduleKind::Interval { seconds } => {
            Some(last_fire + ChronoDuration::seconds(*seconds as i64))
        }
        ScheduleKind::Cron { expression } => next_cron_fire(expression, last_fire).ok(),
    }
}

// ── ScheduleTaskTool ──────────────────────────────────────────────────────────

pub struct ScheduleTaskTool {
    pub bus: EventBus,
    pub state: ScheduleState,
    pub default_group: String,
    /// If set, scheduled tasks fire via relay and their result is routed back
    /// to this group. Used for sub-agents so reminders reach the main agent.
    pub reply_group: Option<String>,
    /// If set, the task list is saved to disk after every mutation.
    pub tasks_path: Option<PathBuf>,
}

#[async_trait]
impl Tool for ScheduleTaskTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "schedule_task",
            "Schedule a task to run at a specific time or on a recurring schedule. \
             When it fires, the description is sent as a new user message to the target group.",
            json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short name for the task."
                    },
                    "description": {
                        "type": "string",
                        "description": "What to do when the task fires (sent as a user message)."
                    },
                    "schedule": {
                        "type": "string",
                        "description": "When to run: ISO 8601 (once), 'every Ns/Nm/Nh' (interval), or cron '0 9 * * 1-5' (weekdays at 9am)."
                    },
                    "target_group": {
                        "type": "string",
                        "description": "Which group to run the task in. Defaults to the current group."
                    }
                },
                "required": ["name", "description", "schedule"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let name = args["name"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'name'".into()))?;
        let description = args["description"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'description'".into()))?;
        let schedule_str = args["schedule"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'schedule'".into()))?;
        let target_group = args["target_group"].as_str().map(|s| s.to_string());

        let (kind, next_fire) = parse_schedule(schedule_str).map_err(AgentError::Tool)?;

        let task = ScheduledTask {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            description: description.to_string(),
            schedule_kind: kind,
            target_group: target_group.or_else(|| Some(self.default_group.clone())),
            reply_group: self.reply_group.clone(),
            next_fire,
            paused: false,
            fired_count: 0,
            created_at: Utc::now(),
        };

        let task_id = task.id.clone();
        let next_fire_str = task.next_fire.to_rfc3339();

        let snapshot = {
            let mut state = self.state.lock().await;
            state.push(task.clone());
            self.tasks_path.as_ref().map(|_| state.clone())
        };
        save_if_needed(snapshot, &self.tasks_path);

        let _ = self
            .bus
            .publish(Event::new(
                CLAW_SCHEDULE_CREATE,
                json!({
                    "task_id": task_id,
                    "name": name,
                    "description": description,
                    "schedule": schedule_str,
                    "next_fire": next_fire_str,
                    "target_group": task.target_group,
                }),
            ))
            .await;

        Ok(json!({
            "task_id": task_id,
            "name": name,
            "next_fire": next_fire_str,
            "success": true,
        }))
    }
}

// ── ListTasksTool ─────────────────────────────────────────────────────────────

pub struct ListTasksTool {
    pub state: ScheduleState,
}

#[async_trait]
impl Tool for ListTasksTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "list_tasks",
            "List all scheduled tasks with their IDs, schedules, and next fire times.",
            json!({ "type": "object", "properties": {} }),
        )
    }

    async fn execute(&self, _args: Value) -> Result<Value, AgentError> {
        let tasks = self.state.lock().await;
        let list: Vec<Value> = tasks
            .iter()
            .map(|t| {
                let schedule_desc = match &t.schedule_kind {
                    ScheduleKind::Once { at } => format!("once at {}", at.to_rfc3339()),
                    ScheduleKind::Interval { seconds } => format!("every {seconds}s"),
                    ScheduleKind::Cron { expression } => format!("cron '{expression}'"),
                };
                json!({
                    "id": t.id,
                    "name": t.name,
                    "description": t.description,
                    "schedule": schedule_desc,
                    "next_fire": t.next_fire.to_rfc3339(),
                    "target_group": t.target_group,
                    "paused": t.paused,
                    "fired_count": t.fired_count,
                })
            })
            .collect();

        Ok(json!({ "tasks": list, "count": list.len() }))
    }
}

// ── CancelTaskTool ────────────────────────────────────────────────────────────

pub struct CancelTaskTool {
    pub bus: EventBus,
    pub state: ScheduleState,
    pub tasks_path: Option<PathBuf>,
}

#[async_trait]
impl Tool for CancelTaskTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "cancel_task",
            "Cancel and remove a scheduled task by its ID or name.",
            json!({
                "type": "object",
                "properties": {
                    "task_id": { "type": "string", "description": "Task ID (from list_tasks)." },
                    "name":    { "type": "string", "description": "Task name (alternative to task_id)." }
                }
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let task_id = args["task_id"].as_str().unwrap_or("");
        let name = args["name"].as_str().unwrap_or("");

        if task_id.is_empty() && name.is_empty() {
            return Err(AgentError::Tool("provide 'task_id' or 'name'".into()));
        }

        let (removed, snapshot) = {
            let mut state = self.state.lock().await;
            let before = state.len();
            state.retain(|t| t.id != task_id && t.name != name);
            let removed = before - state.len();
            let snapshot = if removed > 0 {
                self.tasks_path.as_ref().map(|_| state.clone())
            } else {
                None
            };
            (removed, snapshot)
        };
        save_if_needed(snapshot, &self.tasks_path);

        let _ = self
            .bus
            .publish(Event::new(
                CLAW_SCHEDULE_UPDATE,
                json!({ "action": "cancel", "task_id": task_id, "name": name }),
            ))
            .await;

        Ok(json!({ "removed": removed, "success": removed > 0 }))
    }
}

// ── PauseTaskTool ─────────────────────────────────────────────────────────────

pub struct PauseTaskTool {
    pub bus: EventBus,
    pub state: ScheduleState,
    pub tasks_path: Option<PathBuf>,
}

#[async_trait]
impl Tool for PauseTaskTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "pause_task",
            "Pause or resume a scheduled task. Paused tasks are skipped on heartbeat.",
            json!({
                "type": "object",
                "properties": {
                    "task_id": { "type": "string", "description": "Task ID." },
                    "pause":   { "type": "boolean", "description": "true = pause, false = resume." }
                },
                "required": ["task_id"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let task_id = args["task_id"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'task_id'".into()))?;
        let pause = args["pause"].as_bool().unwrap_or(true);

        let (result, snapshot) = {
            let mut state = self.state.lock().await;
            match state.iter_mut().find(|t| t.id == task_id) {
                Some(t) => {
                    t.paused = pause;
                    let snap = self.tasks_path.as_ref().map(|_| state.clone());
                    (Ok(()), snap)
                }
                None => (
                    Err(AgentError::Tool(format!("task not found: {task_id}"))),
                    None,
                ),
            }
        };
        save_if_needed(snapshot, &self.tasks_path);
        result?;

        let _ = self
            .bus
            .publish(Event::new(
                CLAW_SCHEDULE_UPDATE,
                json!({ "action": if pause { "pause" } else { "resume" }, "task_id": task_id }),
            ))
            .await;

        Ok(json!({ "task_id": task_id, "paused": pause, "success": true }))
    }
}

// ── UpdateTaskTool ────────────────────────────────────────────────────────────

pub struct UpdateTaskTool {
    pub bus: EventBus,
    pub state: ScheduleState,
    pub tasks_path: Option<PathBuf>,
}

#[async_trait]
impl Tool for UpdateTaskTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "update_task",
            "Update an existing scheduled task's name, description, or schedule.",
            json!({
                "type": "object",
                "properties": {
                    "task_id":     { "type": "string", "description": "Task ID to update." },
                    "name":        { "type": "string", "description": "New name (optional)." },
                    "description": { "type": "string", "description": "New description / prompt (optional)." },
                    "schedule":    { "type": "string", "description": "New schedule string: ISO 8601, 'every Ns', or cron expression (optional)." }
                },
                "required": ["task_id"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let task_id = args["task_id"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'task_id'".into()))?;

        let new_name = args["name"].as_str();
        let new_description = args["description"].as_str();
        let new_schedule = args["schedule"].as_str();

        // Parse new schedule before acquiring lock (may error)
        let parsed_schedule = if let Some(s) = new_schedule {
            Some(parse_schedule(s).map_err(AgentError::Tool)?)
        } else {
            None
        };

        let (next_fire_str, snapshot) = {
            let mut state = self.state.lock().await;
            let task = state
                .iter_mut()
                .find(|t| t.id == task_id)
                .ok_or_else(|| AgentError::Tool(format!("task not found: {task_id}")))?;

            if let Some(n) = new_name {
                task.name = n.to_string();
            }
            if let Some(d) = new_description {
                task.description = d.to_string();
            }
            if let Some((kind, next)) = parsed_schedule {
                task.schedule_kind = kind;
                task.next_fire = next;
            }

            let nf = task.next_fire.to_rfc3339();
            let snap = self.tasks_path.as_ref().map(|_| state.clone());
            (nf, snap)
        };
        save_if_needed(snapshot, &self.tasks_path);

        let _ = self
            .bus
            .publish(Event::new(
                CLAW_SCHEDULE_UPDATE,
                json!({ "action": "update", "task_id": task_id }),
            ))
            .await;

        Ok(json!({ "task_id": task_id, "next_fire": next_fire_str, "success": true }))
    }
}
