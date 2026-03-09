use crate::client::McpClient;
use crate::error::McpError;
use async_trait::async_trait;
use eventage_agent::tool::{Tool, ToolRegistry};
use eventage_agent::AgentError;
use eventage_llm::types::ToolDefinition;
use serde_json::Value;
use std::sync::Arc;

/// An Eventage [`Tool`] powered by a remote MCP server.
///
/// Generally created via [`McpToolset::from_http`].
#[derive(Clone)]
pub struct McpTool {
    definition: ToolDefinition,
    client: Arc<McpClient>,
}

#[async_trait]
impl Tool for McpTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        self.client
            .call_tool(&self.definition.function.name, args)
            .await
            .map_err(|e| AgentError::Tool(e.to_string()))
    }
}

// ── McpToolset ────────────────────────────────────────────────────────────────

/// Loads tools from an MCP server as [`McpTool`]s.
///
/// # Static Loading
///
/// Load tools at startup and register them on the agent builder:
///
/// ```no_run
/// use eventage_provided_impl::{AgentBuilder, ReactStrategy};
/// use eventage_mcp::McpToolset;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let toolset = McpToolset::from_http("http://localhost:3000/mcp").await?;
///
/// let agent = AgentBuilder::new()
///     .strategy(ReactStrategy::default())
///     /* .llm(...) .bus(...) */
///     .build();
/// # Ok(())
/// # }
/// ```
///
/// # Dynamic Runtime Loading
///
/// Inject MCP tools into a running agent via a [`ToolRegistry`]. The agent picks up new tools automatically.
///
/// ```no_run
/// use eventage_provided_impl::{AgentBuilder, ReactStrategy};
/// use eventage_core::EventBus;
/// use eventage_mcp::McpToolset;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let bus = EventBus::new();
/// let mut builder = AgentBuilder::new()
///     .bus(bus)
///     .strategy(ReactStrategy::default());
///     /* .llm(...) */
///
/// // Obtain a registry handle before building.
/// let registry = builder.tool_registry();
/// let agent = builder.build();
///
/// // Later — connect to an MCP server and inject its tools at runtime.
/// let toolset = McpToolset::from_http("http://localhost:3000/mcp").await?;
/// toolset.add_to_registry(&registry);
/// // The agent now sees the new tools on its next LLM call.
/// # Ok(())
/// # }
/// ```
///
/// # Reloading Tools
///
/// If the server's available tools change, use [`McpToolset::reload`] to fetch the latest tools:
///
/// ```no_run
/// use eventage_provided_impl::AgentBuilder;
/// use eventage_mcp::McpToolset;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let toolset = McpToolset::from_http("http://localhost:3000/mcp").await?;
///
/// // ... later, the server gained new tools ...
/// let refreshed = toolset.reload().await?;
/// // register refreshed tools to a registry
/// # Ok(())
/// # }
/// ```
pub struct McpToolset {
    client: Arc<McpClient>,
    tools: Vec<McpTool>,
}

impl McpToolset {
    /// Connects to an MCP server and loads all available tools.
    pub async fn from_http(url: impl Into<String>) -> Result<Self, McpError> {
        let client = Arc::new(McpClient::connect_http(url).await?);
        let definitions = client.list_tools().await?;

        let tools = definitions
            .into_iter()
            .map(|def| McpTool {
                definition: def,
                client: Arc::clone(&client),
            })
            .collect();

        Ok(Self { client, tools })
    }

    /// Registers all tools in this toolset into the given `registry`.
    ///
    /// This is ideal for dynamic runtime injection. The agent automatically detects new tools.
    pub fn add_to_registry(&self, registry: &ToolRegistry) {
        for tool in &self.tools {
            registry.register(Arc::new(tool.clone()));
        }
    }

    /// Reloads the tool list from the connected MCP server.
    ///
    /// Call [`McpToolset::add_to_registry`] on the updated toolset to inject new tools.
    pub async fn reload(&self) -> Result<Self, McpError> {
        let definitions = self.client.list_tools().await?;
        let tools = definitions
            .into_iter()
            .map(|def| McpTool {
                definition: def,
                client: Arc::clone(&self.client),
            })
            .collect();
        Ok(Self {
            client: Arc::clone(&self.client),
            tools,
        })
    }

    /// Consumes the toolset, returning the individual [`McpTool`]s.
    ///
    /// Prefer [`McpToolset::add_to_registry`] for dynamic injection.
    pub fn into_tools(self) -> Vec<McpTool> {
        self.tools
    }

    /// Returns the number of tools loaded from the server.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}
