use super::client::McpClient;
use super::error::McpError;
use async_trait::async_trait;
use crate::agent::tool::{Tool, ToolRegistry};
use crate::agent::error::AgentError;
use crate::llm::types::ToolDefinition;
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
    pub fn add_to_registry(&self, registry: &ToolRegistry) {
        for tool in &self.tools {
            registry.register(Arc::new(tool.clone()));
        }
    }

    /// Reloads the tool list from the connected MCP server.
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
