//! Core observability abstractions for the Eventage framework.
//!
//! # Components
//!
//! - [`BusObserver`] — subscribes to an [`eventage_core::EventBus`] and fans events out to exporters.
//! - [`ObservabilityExporter`] — trait defining custom event export destinations.
//! - [`ObsError`] — error type for the observability pipeline.
//!
//! Concrete exporters (`JsonlExporter`, `OtelExporter`) live in `eventage-provided-impl`.

pub mod error;
pub mod exporter;
pub mod observer;

pub use error::ObsError;
pub use exporter::ObservabilityExporter;
pub use observer::BusObserver;
