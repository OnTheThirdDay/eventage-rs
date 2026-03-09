//! Official provided implementations for the Eventage framework.
//!
//! Re-exports the full [`eventage_agent`] public API and provides ready-to-use
//! strategies, workers, context assemblers, and exporters.

mod agents;
mod bridge;
mod dynamic_worker;
pub mod eviction;
mod hooks;
mod observability_impl;
mod selectors;
mod session_impl;
mod strategy_impl;
mod workers;

pub mod context;

// ── Top-level re-exports ──────────────────────────────────────────────────────

// Re-export the full eventage-agent public API so users can depend on this
// crate as their single dependency.
pub use eventage_agent::*;
pub use eventage_core;

// Re-export bus eviction types from eventage-core so users who only depend on
// this crate get everything they need without an extra import.
pub use eventage_core::{BranchData, BranchEvictionStrategy, PruneStrategy};
pub use eviction::{EpitaphStore, EpitaphStrategy};

// Re-export the eventage-observability public API.
pub use eventage_observability::{BusObserver, ObsError, ObservabilityExporter};
pub use observability_impl::JsonlExporter;

#[cfg(feature = "opentelemetry")]
pub use observability_impl::otel::OtelExporter;

// Provided implementations — flat exports for ergonomic use.
pub use agents::AgentSet;
pub use bridge::BusBridge;
pub use context::{
    default_negative_context_format, DefaultContextAssembler, DynamicContextAssembler,
    NegativeAwareContextAssembler,
};
pub use dynamic_worker::DynamicWorkerHandle;
pub use hooks::DynamicHookChain;
pub use selectors::KeywordToolSelector;
pub use session_impl::{Session, SessionBuilder};
pub use strategy_impl::{
    ReactStrategy, SingleShotStrategy, DEFAULT_MAX_CONCURRENT_TOOLS, DEFAULT_MAX_REACT_STEPS,
};
pub use workers::WorkerSet;

// ── Submodule path aliases ────────────────────────────────────────────────────
// These shadow the re-exported eventage_agent submodules to add provided impls
// at the same paths users expect (e.g., `eventage_provided_impl::worker::WorkerSet`).

/// Extended `strategy` module: includes provided [`ReactStrategy`] and [`SingleShotStrategy`].
pub mod strategy {
    pub use super::strategy_impl::{
        ReactStrategy, SingleShotStrategy, DEFAULT_MAX_CONCURRENT_TOOLS, DEFAULT_MAX_REACT_STEPS,
    };
    pub use eventage_agent::strategy::*;
}

/// Extended `worker` module: includes provided [`WorkerSet`].
pub mod worker {
    pub use super::workers::WorkerSet;
    pub use eventage_agent::worker::*;
}

/// `multi` module: contains [`AgentSet`].
pub mod multi {
    pub use super::agents::AgentSet;
}

/// Extended `hook` module: includes provided [`DynamicHookChain`].
pub mod hook {
    pub use super::hooks::DynamicHookChain;
    pub use eventage_agent::hook::*;
}

/// `observability` module: includes [`BusObserver`], [`ObservabilityExporter`], [`ObsError`],
/// [`JsonlExporter`], and optionally [`OtelExporter`].
pub mod observability {
    pub use super::observability_impl::JsonlExporter;
    pub use eventage_observability::{BusObserver, ObsError, ObservabilityExporter};

    #[cfg(feature = "opentelemetry")]
    pub use super::observability_impl::otel::OtelExporter;
}
