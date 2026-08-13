//! [`BusBridge`] — forwards events between isolated [`EventBus`]es.
//!
//! Enables inter-bus IPC by publishing matching events from a source bus to a target bus.
//!
//! # Example — connecting two isolated agent buses
//!
//! ```rust,no_run
//! use eventage::{BusBridge, EventBus, kinds};
//! use eventage::agent::WorkerSet;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let bus_a = EventBus::new();
//! let bus_b = EventBus::new();
//!
//! // Forward all agent.message events from bus_a to bus_b.
//! let bridge = BusBridge::new(bus_b.clone())
//!     .filter_kinds(vec![kinds::AGENT_MESSAGE]);
//!
//! // Run the bridge on bus_a (the source).
//! tokio::spawn(WorkerSet::new().add_worker(bridge).run_on(bus_a.clone()));
//! # Ok(())
//! # }
//! ```

use crate::agent::worker::{EventWorker, WorkerError};
use crate::bus::EventBus;
use crate::event::Event;
use async_trait::async_trait;

/// Metadata key counting how many bridges an event has crossed.
pub const BRIDGE_HOPS_KEY: &str = "bridge_hops";

/// An [`EventWorker`] that forwards events from a source [`EventBus`] to a target bus.
///
/// By default all events are forwarded. Restrict using [`filter_kinds`][Self::filter_kinds].
///
/// # Loop protection
///
/// Every forwarded event is stamped with a `bridge_hops` metadata counter.
/// Events that have already crossed [`max_hops`](Self::max_hops) bridges
/// (default **1**) are not re-forwarded, so a pair of bridges connecting two
/// buses in both directions cannot ping-pong an event forever. Raise the
/// limit only for deliberate multi-hop topologies (e.g. A → B → C).
pub struct BusBridge {
    target: EventBus,
    filter: Vec<String>,
    max_hops: u64,
}

impl BusBridge {
    /// Create a bridge that forwards all events to `target`.
    pub fn new(target: EventBus) -> Self {
        Self {
            target,
            filter: vec![],
            max_hops: 1,
        }
    }

    /// Restrict forwarding to events whose `kind` is in `kinds`.
    ///
    /// An empty list (the default) forwards all events.
    pub fn filter_kinds(mut self, kinds: Vec<impl Into<String>>) -> Self {
        self.filter = kinds.into_iter().map(|k| k.into()).collect();
        self
    }

    /// Allow events to cross up to `n` bridges (default 1).
    pub fn max_hops(mut self, n: u64) -> Self {
        self.max_hops = n.max(1);
        self
    }
}

#[async_trait]
impl EventWorker for BusBridge {
    fn subscribed_kinds(&self) -> Vec<String> {
        self.filter.clone()
    }

    async fn handle(&self, event: &Event, _source: &EventBus) -> Result<(), WorkerError> {
        let hops = event
            .metadata
            .get(BRIDGE_HOPS_KEY)
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        if hops >= self.max_hops {
            // Already crossed enough bridges — stop the echo.
            return Ok(());
        }
        let forwarded = event
            .clone()
            .with_meta(BRIDGE_HOPS_KEY, serde_json::json!(hops + 1));
        self.target
            .publish(forwarded)
            .await
            .map_err(WorkerError::Bus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::kinds;
    use serde_json::json;

    #[tokio::test]
    async fn bidirectional_bridges_do_not_echo_forever() {
        let bus_a = EventBus::new();
        let bus_b = EventBus::new();

        let a_to_b = BusBridge::new(bus_b.clone());
        let b_to_a = BusBridge::new(bus_a.clone());

        // Manually simulate the bridge round-trip.
        let original = Event::new(kinds::AGENT_MESSAGE, json!({"text": "hi"}));
        bus_a.publish(original.clone()).await.unwrap();

        // A→B forwards…
        a_to_b.handle(&original, &bus_a).await.unwrap();
        let on_b = bus_b.log().await;
        assert_eq!(on_b.len(), 1);
        assert_eq!(on_b[0].metadata[BRIDGE_HOPS_KEY], 1);

        // …but B→A refuses to bounce it back.
        b_to_a.handle(&on_b[0], &bus_b).await.unwrap();
        assert_eq!(
            bus_a.log().await.len(),
            1,
            "event must not return to its origin bus"
        );
    }
}
