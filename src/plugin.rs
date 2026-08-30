//! Manifest-driven plugins: distributable bundles of skills, MCP servers,
//! and prompt fragments.
//!
//! A plugin is a directory with an `eventage-plugin.toml` manifest:
//!
//! ```toml
//! [plugin]
//! name = "github-tools"
//! description = "GitHub workflows for the agent"
//! prompt = "prompts/system.md"   # optional system-prompt fragment
//! skills_dir = "skills"          # optional directory of SKILL.md bundles
//!
//! [[mcp]]
//! name = "github"
//! transport = "stdio"
//! command = "npx"
//! args = ["-y", "@modelcontextprotocol/server-github"]
//!
//! [[mcp]]
//! name = "search"
//! transport = "http"
//! url = "https://mcp.example.com"
//!
//! [[observer]]
//! name = "audit"
//! command = "node"
//! args = ["audit.js"]
//! watch = ["tool.result"]
//! ```
//!
//! An `[[observer]]` is a process that watches the event bus and may publish
//! back to it — see [`observer`] for what it may watch, what it may emit, and
//! why where it was installed decides which.
//!
//! [`PluginHost`] loads any number of plugins and installs them into an
//! agent in one call: MCP servers are connected and their tools registered
//! (name-prefixed per server to avoid collisions), skills from every plugin
//! are merged behind one `skill` tool, and the combined system-prompt
//! fragment is returned for the caller to append to its prompt.
//!
//! ```no_run
//! # use eventage::plugin::PluginHost;
//! # use eventage::ToolRegistry;
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut host = PluginHost::new();
//! host.load("./plugins/github-tools")?;
//! host.load("./plugins/data-analysis")?;
//!
//! let registry = ToolRegistry::new();
//! let prompt_fragment = host.install(&registry).await?;
//! # Ok(())
//! # }
//! ```

pub mod llm_service;
pub mod observer;
pub mod transport;

pub use llm_service::PluginLlmService;
pub use observer::{ObserverComponent, ObserverSpec, PluginOrigin};

use crate::agent::skills::{SkillTool, SkillsLibrary};
use crate::agent::tool::ToolRegistry;
use crate::component::{Component, ComponentContext, ComponentError};
use async_trait::async_trait;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::info;

/// Manifest filename looked up in a plugin directory.
pub const MANIFEST_NAME: &str = "eventage-plugin.toml";

#[derive(Debug, Error)]
pub enum PluginError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid manifest {path}: {message}")]
    Manifest { path: PathBuf, message: String },
    #[error("MCP server '{name}' failed: {message}")]
    Mcp { name: String, message: String },
    #[error("plugin declares MCP servers but eventage was built without the `mcp` feature")]
    McpFeatureDisabled,
    #[error("plugin '{plugin}': {message}")]
    Capability { plugin: String, message: String },
}

// ── Manifest schema ───────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct Manifest {
    plugin: ManifestPlugin,
    #[serde(default)]
    mcp: Vec<McpServerSpec>,
    #[serde(default)]
    observer: Vec<ObserverSpec>,
}

#[derive(Debug, Deserialize)]
struct ManifestPlugin {
    name: String,
    #[serde(default)]
    description: String,
    /// Relative path to a markdown file appended to the system prompt.
    #[serde(default)]
    prompt: Option<String>,
    /// Relative path to a directory of `SKILL.md` bundles.
    #[serde(default)]
    skills_dir: Option<String>,
}

/// One MCP server declared by a plugin.
#[derive(Debug, Clone, Deserialize)]
pub struct McpServerSpec {
    /// Registry prefix for this server's tools (`<name>__<tool>`).
    pub name: String,
    /// `"stdio"` or `"http"`.
    pub transport: String,
    /// stdio: executable to spawn.
    #[serde(default)]
    pub command: Option<String>,
    /// stdio: arguments.
    #[serde(default)]
    pub args: Vec<String>,
    /// stdio: extra environment variables for the server process.
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// http: endpoint URL.
    #[serde(default)]
    pub url: Option<String>,
    /// http: extra request headers (e.g. `Authorization`).
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

// ── Plugin ────────────────────────────────────────────────────────────────────

/// A loaded (but not yet installed) plugin.
#[derive(Debug)]
pub struct Plugin {
    pub name: String,
    pub description: String,
    /// System-prompt fragment from the manifest's `prompt` file.
    pub prompt_fragment: Option<String>,
    /// Skills bundled with this plugin.
    pub skills: SkillsLibrary,
    /// MCP servers to connect at install time.
    pub mcp_servers: Vec<McpServerSpec>,
    /// Bus observers to spawn, subject to [`PluginOrigin`] and trust.
    pub observers: Vec<ObserverSpec>,
    /// Where this plugin was installed from, which decides what its observers
    /// may do. Defaults to the least privileged answer.
    pub origin: PluginOrigin,
}

impl Plugin {
    /// Load a plugin from a directory containing [`MANIFEST_NAME`].
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, PluginError> {
        let dir = dir.as_ref();
        let manifest_path = dir.join(MANIFEST_NAME);
        let raw = std::fs::read_to_string(&manifest_path)?;
        let manifest: Manifest = toml::from_str(&raw).map_err(|e| PluginError::Manifest {
            path: manifest_path.clone(),
            message: e.to_string(),
        })?;

        let prompt_fragment = match &manifest.plugin.prompt {
            Some(rel) => {
                let content = std::fs::read_to_string(dir.join(rel))?;
                (!content.trim().is_empty()).then(|| content.trim().to_string())
            }
            None => None,
        };

        let mut skills = SkillsLibrary::new();
        if let Some(rel) = &manifest.plugin.skills_dir {
            skills.add_dir(dir.join(rel))?;
        }

        // Rejected here rather than at start: a manifest asking to forge a
        // tool result is broken however it was installed, and its author
        // should learn that at load instead of from a silent no-op.
        for spec in &manifest.observer {
            spec.validate().map_err(|message| PluginError::Capability {
                plugin: manifest.plugin.name.clone(),
                message,
            })?;
        }

        Ok(Self {
            name: manifest.plugin.name,
            description: manifest.plugin.description,
            prompt_fragment,
            skills,
            mcp_servers: manifest.mcp,
            observers: manifest.observer,
            // Least privilege by default; a caller that knows the plugin came
            // from the user's own directory says so explicitly.
            origin: PluginOrigin::Workspace,
        })
    }

    /// Record where this plugin was installed from.
    pub fn with_origin(mut self, origin: PluginOrigin) -> Self {
        self.origin = origin;
        self
    }
}

// ── PluginHost ────────────────────────────────────────────────────────────────

/// Loads plugins and installs them into an agent's tool registry.
#[derive(Default)]
pub struct PluginHost {
    plugins: Vec<Plugin>,
}

impl PluginHost {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load the plugin at `dir`, treating it as [`PluginOrigin::Workspace`].
    pub fn load(&mut self, dir: impl AsRef<Path>) -> Result<&Plugin, PluginError> {
        self.load_from(dir, PluginOrigin::Workspace)
    }

    /// Load the plugin at `dir`, recording where it came from.
    ///
    /// The origin is what decides whether the plugin's observers may run at
    /// all, so a caller that walks several directories must say which is
    /// which rather than letting them all inherit one answer.
    pub fn load_from(
        &mut self,
        dir: impl AsRef<Path>,
        origin: PluginOrigin,
    ) -> Result<&Plugin, PluginError> {
        let plugin = Plugin::load(dir)?.with_origin(origin);
        info!(
            plugin = %plugin.name,
            skills = plugin.skills.len(),
            mcp_servers = plugin.mcp_servers.len(),
            "plugin loaded"
        );
        self.plugins.push(plugin);
        Ok(self.plugins.last().expect("just pushed"))
    }

    /// Add an already-constructed plugin (e.g. built in code).
    pub fn add(&mut self, plugin: Plugin) {
        self.plugins.push(plugin);
    }

    pub fn plugins(&self) -> &[Plugin] {
        &self.plugins
    }

    /// Install every loaded plugin into `registry` and return the combined
    /// system-prompt fragment to append to the agent's prompt.
    ///
    /// - Each MCP server is connected and its tools registered under the
    ///   server's name prefix (`<server>__<tool>`).
    /// - Skills from all plugins are merged behind a single `skill` tool.
    /// - Prompt fragments are concatenated in load order, followed by the
    ///   skills discovery section.
    pub async fn install(&self, registry: &ToolRegistry) -> Result<String, PluginError> {
        let mut merged_skills = SkillsLibrary::new();
        let mut prompt = String::new();

        for plugin in &self.plugins {
            if let Some(fragment) = &plugin.prompt_fragment {
                if !prompt.is_empty() {
                    prompt.push_str("\n\n");
                }
                prompt.push_str(fragment);
            }
            for skill in plugin.skills.iter() {
                merged_skills.add_skill(skill.clone());
            }
            for server in &plugin.mcp_servers {
                self.connect_mcp(server, registry).await?;
            }
        }

        if !merged_skills.is_empty() {
            let section = merged_skills.system_prompt_section();
            registry.add_tool(SkillTool::new(merged_skills));
            if !prompt.is_empty() {
                prompt.push_str("\n\n");
            }
            prompt.push_str(&section);
        }

        Ok(prompt)
    }

    #[cfg(feature = "mcp")]
    async fn connect_mcp(
        &self,
        spec: &McpServerSpec,
        registry: &ToolRegistry,
    ) -> Result<(), PluginError> {
        use crate::mcp::{McpClient, McpToolset};

        let map_err = |e: crate::mcp::McpError| PluginError::Mcp {
            name: spec.name.clone(),
            message: e.to_string(),
        };

        let client = match spec.transport.as_str() {
            "stdio" => {
                let command = spec.command.as_deref().ok_or_else(|| PluginError::Mcp {
                    name: spec.name.clone(),
                    message: "stdio transport requires `command`".into(),
                })?;
                McpClient::connect_stdio(
                    command,
                    spec.args.clone(),
                    spec.env.clone().into_iter().collect(),
                )
                .await
                .map_err(map_err)?
            }
            "http" => {
                let url = spec.url.as_deref().ok_or_else(|| PluginError::Mcp {
                    name: spec.name.clone(),
                    message: "http transport requires `url`".into(),
                })?;
                McpClient::connect_http_with_headers(
                    url,
                    spec.headers.clone().into_iter().collect(),
                )
                .await
                .map_err(map_err)?
            }
            other => {
                return Err(PluginError::Mcp {
                    name: spec.name.clone(),
                    message: format!("unknown transport '{other}' (expected stdio|http)"),
                })
            }
        };

        let toolset = McpToolset::from_client(client)
            .await
            .map_err(map_err)?
            .with_prefix(&spec.name);
        toolset.add_to_registry(registry);
        info!(server = %spec.name, tools = toolset.len(), "MCP server installed");
        Ok(())
    }

    #[cfg(not(feature = "mcp"))]
    async fn connect_mcp(
        &self,
        _spec: &McpServerSpec,
        _registry: &ToolRegistry,
    ) -> Result<(), PluginError> {
        Err(PluginError::McpFeatureDisabled)
    }
}

// ── Plugins as components ─────────────────────────────────────────────────────

/// Adapts a [`Plugin`] to the [`Component`] lifecycle so it can be unloaded.
///
/// Everything the plugin contributes — its skills tool and each MCP server's
/// tools — is registered through the component context, so `unload` removes
/// all of it and drops the MCP clients (killing their child processes).
pub struct PluginComponent {
    plugin: Plugin,
    trusted: bool,
    cwd: PathBuf,
}

impl PluginComponent {
    pub fn new(plugin: Plugin) -> Self {
        Self {
            plugin,
            trusted: false,
            cwd: PathBuf::from("."),
        }
    }

    /// Whether the operator has said this project is theirs. Graded observer
    /// capabilities are withheld without it.
    pub fn with_trust(mut self, trusted: bool) -> Self {
        self.trusted = trusted;
        self
    }

    /// Working directory for any observer this plugin spawns.
    pub fn with_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = cwd.into();
        self
    }

    /// The system-prompt fragment this plugin contributes.
    pub fn prompt_fragment(&self) -> Option<&str> {
        self.plugin.prompt_fragment.as_deref()
    }
}

#[async_trait]
impl Component for PluginComponent {
    fn name(&self) -> &str {
        &self.plugin.name
    }

    async fn start(&self, ctx: &mut ComponentContext) -> Result<(), ComponentError> {
        if !self.plugin.skills.is_empty() {
            ctx.tool(SkillTool::new(self.plugin.skills.clone()));
        }

        #[cfg(feature = "mcp")]
        for spec in &self.plugin.mcp_servers {
            use crate::mcp::{McpClient, McpToolset};

            let client = match (&spec.command, &spec.url) {
                (Some(command), _) => McpClient::connect_stdio(
                    command,
                    spec.args.clone(),
                    spec.env.clone().into_iter().collect(),
                )
                .await
                .map_err(|e| ComponentError::Start(e.to_string()))?,
                (None, Some(url)) => McpClient::connect_http(url)
                    .await
                    .map_err(|e| ComponentError::Start(e.to_string()))?,
                _ => {
                    return Err(ComponentError::Start(format!(
                        "MCP server '{}' has neither command nor url",
                        spec.name
                    )))
                }
            };

            let toolset = McpToolset::from_client(client.with_bus(ctx.bus().clone(), &spec.name))
                .await
                .map_err(|e| ComponentError::Start(e.to_string()))?
                .with_prefix(&spec.name);

            // Register each tool through the context so unload withdraws it,
            // and keep the client alive for exactly this component's lifetime.
            for tool in toolset.into_tools() {
                ctx.tool(tool);
            }
        }

        observer::start_observers(
            ctx,
            &self.plugin.name,
            &self.plugin.observers,
            self.plugin.origin,
            self.trusted,
            &self.cwd,
        )
        .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_plugin(root: &Path) {
        std::fs::create_dir_all(root.join("skills/greet")).unwrap();
        std::fs::create_dir_all(root.join("prompts")).unwrap();
        std::fs::write(
            root.join(MANIFEST_NAME),
            r#"
[plugin]
name = "demo"
description = "demo plugin"
prompt = "prompts/system.md"
skills_dir = "skills"
"#,
        )
        .unwrap();
        std::fs::write(root.join("prompts/system.md"), "Always be polite.").unwrap();
        std::fs::write(
            root.join("skills/greet/SKILL.md"),
            "---\nname: greeting\ndescription: greet users warmly. Use for hellos.\n---\nSay hello twice.",
        )
        .unwrap();
    }

    #[tokio::test]
    async fn a_plugin_can_be_unloaded_completely() {
        use crate::agent::hook::DynamicHookChain;
        use crate::bus::EventBus;
        use crate::component::{ComponentHost, ComponentState};
        use std::sync::Arc;

        let tmp = tempfile::tempdir().unwrap();
        write_plugin(tmp.path());
        let plugin = Plugin::load(tmp.path()).unwrap();

        let tools = ToolRegistry::new();
        let host = ComponentHost::new(EventBus::new(), tools.clone(), DynamicHookChain::new());

        host.load(Arc::new(PluginComponent::new(plugin)))
            .await
            .unwrap();
        assert_eq!(host.state("demo"), Some(ComponentState::Active));
        assert!(
            tools.get("skill").is_some(),
            "plugin contributed its skill tool"
        );

        // The whole point: unloading takes its tools with it.
        host.unload("demo").await.unwrap();
        assert!(
            tools.get("skill").is_none(),
            "unloading a plugin must withdraw everything it registered"
        );
    }

    #[tokio::test]
    async fn loads_and_installs_plugin() {
        let tmp = tempfile::tempdir().unwrap();
        write_plugin(tmp.path());

        let mut host = PluginHost::new();
        let plugin = host.load(tmp.path()).unwrap();
        assert_eq!(plugin.name, "demo");
        assert_eq!(plugin.skills.len(), 1);
        assert_eq!(plugin.prompt_fragment.as_deref(), Some("Always be polite."));

        let registry = ToolRegistry::new();
        let prompt = host.install(&registry).await.unwrap();

        assert!(prompt.starts_with("Always be polite."));
        assert!(prompt.contains("greeting: greet users warmly"));
        assert!(
            registry.get("skill").is_some(),
            "merged skill tool must be registered"
        );
    }

    #[test]
    fn missing_manifest_is_io_error() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(matches!(Plugin::load(tmp.path()), Err(PluginError::Io(_))));
    }

    #[test]
    fn bad_manifest_is_reported_with_path() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(MANIFEST_NAME), "not [valid toml").unwrap();
        let err = Plugin::load(tmp.path()).unwrap_err();
        assert!(matches!(err, PluginError::Manifest { .. }));
    }
}
