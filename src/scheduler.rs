use crate::bus::EventBus;
use crate::event::{kinds, Event};
use serde_json::json;
use std::time::Duration;
use tokio::time;
use tracing::{debug, warn};

/// Periodically emits a `system.heartbeat` event.
///
/// Enables agents to act autonomously without waiting for external inputs.
///
/// # Cost note
///
/// Every heartbeat wakes subscribed agents into a full reasoning cycle — an
/// LLM call each. For long-lived deployments enable
/// [`skip_when_idle`](Self::skip_when_idle) so heartbeats are suppressed while
/// nothing new has happened on the bus.
pub struct HeartbeatScheduler {
    bus: EventBus,
    interval: Duration,
    skip_when_idle: bool,
}

impl HeartbeatScheduler {
    pub fn new(bus: EventBus, interval: Duration) -> Self {
        Self {
            bus,
            interval,
            skip_when_idle: false,
        }
    }

    /// Suppress heartbeats while the bus is idle.
    ///
    /// A tick is skipped when no *activity* event — `user.message`,
    /// `system.message`, `agent.message`, or `tool.result` — has been
    /// published since the previous heartbeat. Assistant replies alone do not
    /// count as activity, so an agent answering a heartbeat with "nothing to
    /// do" does not keep the heartbeat (and its LLM bill) alive forever.
    pub fn skip_when_idle(mut self) -> Self {
        self.skip_when_idle = true;
        self
    }

    /// Runs the scheduler loop forever. Cancel via task abort or `select!`.
    pub async fn run(&self) {
        let mut ticker = time::interval(self.interval);
        // Skip the immediate first tick so the agent has time to initialise.
        ticker.set_missed_tick_behavior(time::MissedTickBehavior::Delay);
        ticker.tick().await; // consume the first instant tick

        let mut last_len = 0usize;
        loop {
            ticker.tick().await;

            if self.skip_when_idle {
                let new_events = self.bus.log_since(last_len).await;
                let active = new_events.iter().any(|e| {
                    matches!(
                        e.kind.as_str(),
                        kinds::USER_MESSAGE
                            | kinds::SYSTEM_MESSAGE
                            | kinds::AGENT_MESSAGE
                            | kinds::TOOL_RESULT
                    )
                });
                if !active {
                    debug!("bus idle — skipping heartbeat");
                    continue;
                }
            }

            // Bookmark BEFORE publishing so the heartbeat itself (and the
            // cycle events it triggers) is judged by the next tick.
            last_len = self.bus.log_len().await;

            debug!("emitting heartbeat");
            if let Err(e) = self
                .bus
                .publish(Event::new(
                    kinds::SYSTEM_HEARTBEAT,
                    json!({ "timestamp": chrono::Utc::now().to_rfc3339() }),
                ))
                .await
            {
                warn!("HeartbeatScheduler: failed to publish heartbeat: {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(start_paused = true)]
    async fn idle_skip_suppresses_heartbeats_until_activity() {
        let bus = EventBus::new();
        let scheduler =
            HeartbeatScheduler::new(bus.clone(), Duration::from_secs(10)).skip_when_idle();

        tokio::spawn(async move { scheduler.run().await });
        // Let the scheduler task initialize its ticker at t=0.
        tokio::task::yield_now().await;

        // Publish initial activity, then let two ticks pass.
        bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "go"})))
            .await
            .unwrap();
        tokio::time::advance(Duration::from_secs(25)).await;
        tokio::time::sleep(Duration::from_millis(1)).await;

        let beats = bus
            .log()
            .await
            .iter()
            .filter(|e| e.kind == kinds::SYSTEM_HEARTBEAT)
            .count();
        assert_eq!(beats, 1, "second tick should be skipped: no new activity");

        // New activity re-enables the heartbeat.
        bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "more"})))
            .await
            .unwrap();
        tokio::time::advance(Duration::from_secs(10)).await;
        tokio::time::sleep(Duration::from_millis(1)).await;

        let beats = bus
            .log()
            .await
            .iter()
            .filter(|e| e.kind == kinds::SYSTEM_HEARTBEAT)
            .count();
        assert_eq!(beats, 2, "activity should re-enable heartbeats");
    }
}
