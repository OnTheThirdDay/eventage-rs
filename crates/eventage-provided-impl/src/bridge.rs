//! [`BusBridge`] — forwards events between isolated [`EventBus`]es.
//!
//! Enables inter-bus IPC by publishing matching events from a source bus to a target bus.
//!
//! # Example — connecting two isolated agent buses
//!
//! ```rust,no_run
//! use eventage_provided_impl::{BusBridge, WorkerSet};
//! use eventage_core::{EventBus, kinds};
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

use async_trait::async_trait;
use eventage_agent::worker::{EventWorker, WorkerError};
use eventage_core::{Event, EventBus};

/// An [`EventWorker`] that forwards events from a source [`EventBus`] to a target bus.
///
/// By default all events are forwarded. Restrict using [`filter_kinds`][Self::filter_kinds].
pub struct BusBridge {
    target: EventBus,
    filter: Vec<String>,
}

impl BusBridge {
    /// Create a bridge that forwards all events to `target`.
    pub fn new(target: EventBus) -> Self {
        Self {
            target,
            filter: vec![],
        }
    }

    /// Restrict forwarding to events whose `kind` is in `kinds`.
    ///
    /// An empty list (the default) forwards all events.
    pub fn filter_kinds(mut self, kinds: Vec<impl Into<String>>) -> Self {
        self.filter = kinds.into_iter().map(|k| k.into()).collect();
        self
    }
}

#[async_trait]
impl EventWorker for BusBridge {
    fn subscribed_kinds(&self) -> Vec<String> {
        self.filter.clone()
    }

    async fn handle(&self, event: &Event, _source: &EventBus) -> Result<(), WorkerError> {
        self.target
            .publish(event.clone())
            .await
            .map_err(WorkerError::Bus)
    }
}
