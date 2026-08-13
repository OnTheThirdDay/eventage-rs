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
//! ```
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

use crate::agent::skills::{SkillTool, SkillsLibrary};
use crate::agent::tool::ToolRegistry;
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
}

// ── Manifest schema ───────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct Manifest {
    plugin: ManifestPlugin,
    #[serde(default)]
    mcp: Vec<McpServerSpec>,
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

        Ok(Self {
            name: manifest.plugin.name,
            description: manifest.plugin.description,
            prompt_fragment,
            skills,
            mcp_servers: manifest.mcp,
        })
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

    /// Load the plugin at `dir`.
    pub fn load(&mut self, dir: impl AsRef<Path>) -> Result<&Plugin, PluginError> {
        let plugin = Plugin::load(dir)?;
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
