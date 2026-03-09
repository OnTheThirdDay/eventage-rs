//! # Eventage Core
//! 
//! Core abstractions for the Eventage framework, providing the central event bus
//! and fundamental data types.

pub mod bus;
pub mod error;
pub mod event;

pub use bus::{
    BranchData, BranchEvictionStrategy, BranchId, BusConfig, BusReceiver, EventBus, PruneStrategy,
};
pub use error::{BusError, CoreError};
pub use event::{kinds, meta_keys, Event, EventId};
