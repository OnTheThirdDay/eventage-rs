use super::client::McpClient;
use super::error::McpError;
use crate::agent::error::AgentError;
use crate::agent::tool::{Tool, ToolRegistry};
use crate::llm::types::ToolDefinition;
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

/// An Eventage [`Tool`] powered by a remote MCP server.
///
/// Generally created via [`McpToolset::from_http`] or [`McpToolset::from_stdio`].
#[derive(Clone)]
pub struct McpTool {
    /// Definition exposed to the LLM (name may carry a toolset prefix).
    definition: ToolDefinition,
    /// The tool's real name on the MCP server.
    remote_name: String,
    client: Arc<McpClient>,
}

#[async_trait]
impl Tool for McpTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        self.client
            .call_tool(&self.remote_name, args)
            .await
            .map_err(|e| AgentError::Tool(e.to_string()))
    }
}

// ── McpToolset ────────────────────────────────────────────────────────────────

/// Loads tools from an MCP server as [`McpTool`]s.
///
/// When registering tools from multiple servers, give each toolset a
/// [`prefix`](Self::with_prefix) — otherwise same-named tools from different
/// servers silently overwrite each other in the [`ToolRegistry`].
pub struct McpToolset {
    client: Arc<McpClient>,
    tools: Vec<McpTool>,
    prefix: Option<String>,
}

impl McpToolset {
    /// Connects to an MCP server over HTTP and loads all available tools.
    pub async fn from_http(url: impl Into<String>) -> Result<Self, McpError> {
        Self::from_client(McpClient::connect_http(url).await?).await
    }

    /// Spawns a local MCP server process and loads all available tools over stdio.
    pub async fn from_stdio(
        program: impl Into<String>,
        args: Vec<String>,
    ) -> Result<Self, McpError> {
        Self::from_client(McpClient::connect_stdio(program, args, vec![]).await?).await
    }

    /// Builds a toolset from an already-connected client.
    pub async fn from_client(client: McpClient) -> Result<Self, McpError> {
        let client = Arc::new(client);
        let tools = Self::load(&client, None).await?;
        Ok(Self {
            client,
            tools,
            prefix: None,
        })
    }

    /// Renames every tool to `<prefix>__<name>` in the LLM-facing definition
    /// while still calling the server with the original name. Prevents
    /// collisions when multiple MCP servers are registered together.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        let prefix = prefix.into();
        for tool in &mut self.tools {
            tool.definition.function.name = format!("{prefix}__{}", tool.remote_name);
        }
        self.prefix = Some(prefix);
        self
    }

    async fn load(client: &Arc<McpClient>, prefix: Option<&str>) -> Result<Vec<McpTool>, McpError> {
        let definitions = client.list_tools().await?;
        Ok(definitions
            .into_iter()
            .map(|mut def| {
                let remote_name = def.function.name.clone();
                if let Some(p) = prefix {
                    def.function.name = format!("{p}__{remote_name}");
                }
                McpTool {
                    definition: def,
                    remote_name,
                    client: Arc::clone(client),
                }
            })
            .collect())
    }

    /// Registers all tools in this toolset into the given `registry`.
    pub fn add_to_registry(&self, registry: &ToolRegistry) {
        for tool in &self.tools {
            registry.register(Arc::new(tool.clone()));
        }
    }

    /// Reloads the tool list from the connected MCP server.
    pub async fn reload(&self) -> Result<Self, McpError> {
        let tools = Self::load(&self.client, self.prefix.as_deref()).await?;
        Ok(Self {
            client: Arc::clone(&self.client),
            tools,
            prefix: self.prefix.clone(),
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
