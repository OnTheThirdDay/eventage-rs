//! eventage-code as a library.
//!
//! The `eventage-code` binary is a thin ACP server over these modules. They
//! are public so other front-ends can host a coding session directly instead
//! of speaking a protocol to a child process — which is what
//! `eventage-studio` does, so its trace panel can read the session's whole
//! event log rather than the summary a protocol would carry.

pub mod acp;
pub mod agent;
pub mod config;
pub mod lsp;
pub mod prompt;
pub mod repomap;
pub mod settings;
pub mod shell_sandbox;
pub mod tools;
pub mod workspace;
