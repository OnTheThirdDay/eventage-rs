//! Eventage Studio: a desktop app for the eventage coding agent.
//!
//! The binary is a thin wrapper over these modules. They are public so the
//! HTTP surface can be exercised in tests against a stand-in backend, without
//! a model, an API key, or a network.

pub mod assets;
pub mod backend;
pub mod feed;
pub mod index;
pub mod protocol;
pub mod server;
