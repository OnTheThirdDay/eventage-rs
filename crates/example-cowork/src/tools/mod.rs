//! Tools cowork adds on top of the generic file tools.
//!
//! The reading, writing, editing and searching tools are the coding agent's,
//! used unchanged: none of them knows anything about code, and rewriting them
//! here would mean maintaining two copies of the workspace confinement, the
//! stale-write guard and the atomic writes. What is genuinely cowork's own is
//! scheduling — a goal that runs on its own, without anyone opening a session.

pub mod schedule;
