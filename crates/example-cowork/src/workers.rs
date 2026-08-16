//! Automations: goals that run without anyone opening a session.
//!
//! A cowork session normally starts because somebody described a task. Some
//! work is not like that — the weekly summary, the inbox triage — and both
//! Cowork and the Codex app treat scheduling as a first-class part of the
//! product rather than a scripting afterthought. The Codex app's own
//! limitation is instructive: its automations run on your machine and need it
//! awake. Ours has the same constraint today, and says so.
//!
//! This is deliberately one worker. Claw routed schedule firings between
//! per-group buses through a relay; cowork has one session bus per session, so
//! the routing layer had nothing left to route.

use crate::kinds;
use crate::tools::schedule::ScheduleState;
use async_trait::async_trait;
use chrono::Utc;
use eventage::{
    agent::worker::{EventWorker, WorkerError},
    event::{kinds as ev, Event},
    EventBus,
};
use serde_json::json;
use tracing::info;

/// Fires due goals on each heartbeat.
pub struct SchedulerWorker {
    pub state: ScheduleState,
}

#[async_trait]
impl EventWorker for SchedulerWorker {
    fn subscribed_kinds(&self) -> Vec<String> {
        vec![ev::SYSTEM_HEARTBEAT.to_string()]
    }

    async fn handle(&self, _event: &Event, bus: &EventBus) -> Result<(), WorkerError> {
        let now = Utc::now();
        let mut due = Vec::new();
        let mut spent = Vec::new();

        {
            // The lock is held only to decide and advance, never across a
            // publish: a slow subscriber would otherwise stall every other
            // automation behind it.
            let mut state = self.state.lock().await;
            for task in state.iter_mut() {
                if task.paused || task.next_fire > now {
                    continue;
                }
                due.push((task.id.clone(), task.name.clone(), task.description.clone()));
                match crate::tools::schedule::advance_schedule(&task.schedule_kind, task.next_fire)
                {
                    Some(next) => {
                        task.next_fire = next;
                        task.fired_count += 1;
                    }
                    // A one-shot has now happened.
                    None => {
                        task.fired_count += 1;
                        spent.push(task.id.clone());
                    }
                }
            }
            state.retain(|t| !spent.contains(&t.id));
        }

        for (id, name, description) in due {
            info!(id = %id, name = %name, "an automation came due");
            bus.publish(Event::new(
                kinds::SCHEDULE_FIRE,
                json!({ "id": id, "name": name, "goal": description }),
            ))
            .await
            .map_err(|e| WorkerError::Worker(e.to_string()))?;
        }
        Ok(())
    }
}
