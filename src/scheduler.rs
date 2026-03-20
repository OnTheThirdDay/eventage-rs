use crate::event::{kinds, Event};
use crate::bus::EventBus;
use serde_json::json;
use std::time::Duration;
use tokio::time;
use tracing::debug;

/// Periodically emits a `system.heartbeat` event.
///
/// Enables agents to act autonomously without waiting for external inputs.
pub struct HeartbeatScheduler {
    bus: EventBus,
    interval: Duration,
}

impl HeartbeatScheduler {
    pub fn new(bus: EventBus, interval: Duration) -> Self {
        Self { bus, interval }
    }

    /// Runs the scheduler loop forever. Cancel via task abort or `select!`.
    pub async fn run(&self) {
        let mut ticker = time::interval(self.interval);
        // Skip the immediate first tick so the agent has time to initialise.
        ticker.set_missed_tick_behavior(time::MissedTickBehavior::Delay);
        ticker.tick().await; // consume the first instant tick
        loop {
            ticker.tick().await;
            debug!("emitting heartbeat");
            let _ = self
                .bus
                .publish(Event::new(
                    kinds::SYSTEM_HEARTBEAT,
                    json!({ "timestamp": chrono::Utc::now().to_rfc3339() }),
                ))
                .await;
        }
    }
}
