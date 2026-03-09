//! MCP (Model Context Protocol) integration for Eventage.
//!
//! This crate connects to an MCP server over HTTP and exposes its tools as Eventage [`Tool`](eventage_agent::tool::Tool)s.
//!
//! # Static Loading
//!
//! Load tools at startup and register them on the agent builder:
//!
//! ```no_run
//! use eventage_provided_impl::{AgentBuilder, ReactStrategy};
//! use eventage_core::EventBus;
//! use eventage_mcp::McpToolset;
//! use eventage_llm::OpenAiProvider;
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
//!
//! # Dynamic Runtime Loading
//!
//! Inject MCP tools into a running agent dynamically using a [`ToolRegistry`](eventage_agent::tool::ToolRegistry).
//! The agent automatically picks up new tools on its next execution step.
//!
//! ```no_run
//! use eventage_provided_impl::{AgentBuilder, ReactStrategy};
//! use eventage_core::EventBus;
//! use eventage_mcp::McpToolset;
//! use eventage_llm::OpenAiProvider;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let bus = EventBus::default();
//!     let mut builder = AgentBuilder::new()
//!         .bus(bus)
//!         .llm(OpenAiProvider::ollama("qwen3:4b"))
//!         .strategy(ReactStrategy::default());
//!
//!     // Obtain a live registry handle *before* building.
//!     let registry = builder.tool_registry();
//!     let _agent = builder.build();
//!
//!     // Later: connect to an MCP server and inject tools at runtime.
//!     let toolset = McpToolset::from_http("http://localhost:3000/mcp").await?;
//!     toolset.add_to_registry(&registry);
//!
//!     // If the server gains new tools later, reload and re-register.
//!     let refreshed = toolset.reload().await?;
//!     refreshed.add_to_registry(&registry);
//!
//!     Ok(())
//! }
//! ```

mod client;
mod error;
mod tool;

pub use client::McpClient;
pub use error::McpError;
pub use tool::{McpTool, McpToolset};
