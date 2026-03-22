//! Input channels for eventage-claw.
//!
//! Each channel is an independent source of `user.message` events.
//!
//! - [`terminal`] — stdin/REPL input (no-TUI mode)
//! - [`http`] — HTTP POST /message → user.message (represents any webhook-based channel)

pub mod http;
pub mod terminal;
