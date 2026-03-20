//! MCP (Model Context Protocol) integration for Eventage.
//!
//! Connects to an MCP server over HTTP and exposes its tools as Eventage [`Tool`](crate::agent::Tool)s.
//!
//! # Static Loading
//!
//! Load tools at startup and register them on the agent builder:
//!
//! ```no_run
//! use eventage::{AgentBuilder, ReactStrategy, EventBus};
//! use eventage::mcp::McpToolset;
//! use eventage::llm::OpenAiProvider;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let bus = EventBus::default();
//!     let toolset = McpToolset::from_http("http://localhost:3000/mcp").await?;
//!
//!     let mut builder = AgentBuilder::new()
//!         .agent_id("mcp-agent")
//!         .bus(bus)
//!         .llm(OpenAiProvider::ollama("qwen3:4b"))
//!         .strategy(ReactStrategy::default());
//!
//!     for tool in toolset.into_tools() {
//!         builder = builder.tool(tool);
//!     }
//!
//!     let _agent = builder.build();
//!     Ok(())
//! }
//! ```

mod client;
mod error;
mod tool;

pub use client::McpClient;
pub use error::McpError;
pub use tool::{McpTool, McpToolset};
