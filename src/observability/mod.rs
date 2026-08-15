//! Core observability abstractions for the Eventage framework.
//!
//! # Components
//!
//! - [`BusObserver`] — subscribes to an [`crate::EventBus`] and fans events out to exporters.
//! - [`ObservabilityExporter`] — trait defining custom event export destinations.
//! - [`ObsError`] — error type for the observability pipeline.
//!
//! Concrete exporters (`JsonlExporter`, `OtelExporter`) are also in this module.

mod error;
mod exporter;
mod jsonl;
mod observer;

#[cfg(feature = "opentelemetry")]
pub mod otel;

pub use error::ObsError;
pub use exporter::ObservabilityExporter;
pub use jsonl::JsonlExporter;
pub use observer::BusObserver;

#[cfg(feature = "opentelemetry")]
pub use otel::OtelExporter;
