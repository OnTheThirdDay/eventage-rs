//! Configuration for eventage-claw.
//!
//! Load order: `~/.claw/config.toml` → environment variables → CLI flags.

use serde::Deserialize;
use std::path::PathBuf;
use tracing::warn;

// ── GroupConfig ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct GroupConfig {
    /// Unique group name, e.g. "personal", "work".
    pub name: String,

    /// If true, this group gets admin tools (register_group, list_groups).
    #[serde(default)]
    pub is_main: bool,

    /// Optional suffix appended to the base system prompt for this group.
    pub system_prompt_suffix: Option<String>,

    /// Tool names that require user approval before execution.
    #[serde(default)]
    pub human_approval_tools: Vec<String>,

    /// If true, every tool call requires approval.
    #[serde(default)]
    pub require_approve_all: bool,

    /// Override the group's working directory. Defaults to `data_dir/groups/{name}`.
    pub work_dir: Option<PathBuf>,

    /// If non-empty, only these sender identifiers (WhatsApp JIDs, usernames, etc.)
    /// are allowed to send messages to this group via the HTTP channel.
    /// An empty list means all senders are accepted.
    #[serde(default)]
    pub allowed_senders: Vec<String>,
}

impl GroupConfig {
    pub fn default_personal() -> Self {
        Self {
            name: "personal".into(),
            is_main: true,
            system_prompt_suffix: None,
            human_approval_tools: vec!["run_command".into()],
            require_approve_all: false,
            work_dir: None,
            allowed_senders: vec![],
        }
    }
}

// ── ClawConfig ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct ClawConfig {
    /// OpenAI-compatible API key.
    #[serde(default = "default_api_key")]
    pub api_key: String,

    /// OpenAI-compatible base URL.
    #[serde(default = "default_llm_url")]
    pub llm_url: String,

    /// Model name.
    #[serde(default = "default_model")]
    pub model: String,

    /// Root data directory for sessions, memory files, skills, and group folders.
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,

    /// Max ReAct steps per cycle.
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,

    /// Token budget for conversation summarization (0 = disabled).
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Max LLM requests per minute (0 = unlimited).
    #[serde(default)]
    pub requests_per_minute: u32,

    /// Heartbeat interval in seconds for the task scheduler.
    #[serde(default = "default_heartbeat_secs")]
    pub heartbeat_secs: u64,

    /// If set, start an HTTP channel server on this port (accepts POST /message).
    pub http_channel_port: Option<u16>,

    /// Configured groups. Defaults to a single "personal" group if empty.
    #[serde(default)]
    pub groups: Vec<GroupConfig>,

    // ── Docker isolation ──────────────────────────────────────────────────────
    /// If true, register `run_in_docker` tool on all groups.
    #[serde(default)]
    pub docker_enabled: bool,

    /// Default Docker image for `run_in_docker`.
    #[serde(default = "default_docker_image")]
    pub docker_image: String,

    /// Memory limit for Docker containers (e.g. `"512m"`, `"2g"`).
    #[serde(default = "default_docker_memory")]
    #[allow(dead_code)]
    pub docker_memory: String,

    /// Default network mode: `"none"` (isolated) or `"bridge"`.
    #[serde(default = "default_docker_network")]
    pub docker_network: String,

    // ── Channel output webhook ────────────────────────────────────────────────
    /// Webhook URL where the `ChannelOutputWorker` POSTs agent responses.
    /// Set to the WhatsApp bridge's `/send` endpoint, e.g.
    /// `"http://localhost:3001/send"`.
    /// Also configurable via `CLAW_WEBHOOK_URL` env var.
    #[serde(default = "default_webhook_url")]
    pub webhook_url: Option<String>,
}

fn default_api_key() -> String {
    std::env::var("ANTHROPIC_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .unwrap_or_else(|_| "ollama".into())
}
fn default_llm_url() -> String {
    std::env::var("LLM_URL").unwrap_or_else(|_| "http://localhost:11434/v1".into())
}
fn default_model() -> String {
    std::env::var("LLM_MODEL").unwrap_or_else(|_| "qwen3:4b".into())
}
fn default_data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("CLAW_DATA_DIR") {
        return PathBuf::from(dir);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".claw")
}
fn default_webhook_url() -> Option<String> {
    std::env::var("CLAW_WEBHOOK_URL").ok()
}
fn default_max_steps() -> usize {
    30
}
fn default_max_tokens() -> usize {
    120_000
}
fn default_heartbeat_secs() -> u64 {
    60
}
fn default_docker_image() -> String {
    "ubuntu:22.04".into()
}
fn default_docker_memory() -> String {
    "512m".into()
}
fn default_docker_network() -> String {
    "none".into()
}

impl Default for ClawConfig {
    fn default() -> Self {
        Self {
            api_key: default_api_key(),
            llm_url: default_llm_url(),
            model: default_model(),
            data_dir: default_data_dir(),
            max_steps: default_max_steps(),
            max_tokens: default_max_tokens(),
            requests_per_minute: 0,
            heartbeat_secs: default_heartbeat_secs(),
            http_channel_port: None,
            groups: vec![GroupConfig::default_personal()],
            docker_enabled: false,
            docker_image: default_docker_image(),
            docker_memory: default_docker_memory(),
            docker_network: default_docker_network(),
            webhook_url: default_webhook_url(),
        }
    }
}

impl ClawConfig {
    /// Load config from `~/.claw/config.toml` (or the provided path), then
    /// overlay environment variables. Falls back to defaults if no file exists.
    pub fn load(path: Option<&std::path::Path>) -> Self {
        let file_path = path
            .map(|p| p.to_owned())
            .unwrap_or_else(|| default_data_dir().join("config.toml"));

        let mut config: ClawConfig = if file_path.exists() {
            match std::fs::read_to_string(&file_path) {
                Ok(content) => match toml::from_str(&content) {
                    Ok(c) => c,
                    Err(e) => {
                        warn!(
                            "could not parse {}: {e} — using defaults",
                            file_path.display()
                        );
                        ClawConfig::default()
                    }
                },
                Err(e) => {
                    warn!(
                        "could not read {}: {e} — using defaults",
                        file_path.display()
                    );
                    ClawConfig::default()
                }
            }
        } else {
            ClawConfig::default()
        };

        // Environment variable overrides (only apply when non-empty)
        if let Ok(k) =
            std::env::var("ANTHROPIC_API_KEY").or_else(|_| std::env::var("OPENAI_API_KEY"))
        {
            if !k.is_empty() {
                config.api_key = k;
            }
        }
        if let Ok(url) = std::env::var("LLM_URL") {
            if !url.is_empty() {
                config.llm_url = url;
            }
        }
        if let Ok(m) = std::env::var("LLM_MODEL") {
            if !m.is_empty() {
                config.model = m;
            }
        }
        if let Ok(url) = std::env::var("CLAW_WEBHOOK_URL") {
            if !url.is_empty() {
                config.webhook_url = Some(url);
            }
        }

        // CLAW_GROUPS=personal,alice,bob — add any groups not already in config.
        // The first listed group becomes is_main if no main group exists yet.
        if let Ok(groups_str) = std::env::var("CLAW_GROUPS") {
            for name in groups_str
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                if !config.groups.iter().any(|g| g.name == name) {
                    let is_main = !config.groups.iter().any(|g| g.is_main);
                    config.groups.push(GroupConfig {
                        name: name.to_string(),
                        is_main,
                        ..GroupConfig::default_personal()
                    });
                }
            }
        }

        // Ensure at least one group exists
        if config.groups.is_empty() {
            config.groups.push(GroupConfig::default_personal());
        }

        // Ensure exactly one main group (first if none is marked)
        if !config.groups.iter().any(|g| g.is_main) {
            config.groups[0].is_main = true;
        }

        config
    }

    /// Returns the config for a named group, or the first group as fallback.
    pub fn group(&self, name: &str) -> &GroupConfig {
        self.groups
            .iter()
            .find(|g| g.name == name)
            .unwrap_or(&self.groups[0])
    }

    /// Work dir for a group: `data_dir/groups/{name}` or custom override.
    pub fn group_work_dir(&self, name: &str) -> PathBuf {
        let g = self.group(name);
        g.work_dir
            .clone()
            .unwrap_or_else(|| self.data_dir.join("groups").join(name))
    }

    /// Path to the group-level AGENT.md memory file.
    pub fn group_memory_path(&self, name: &str) -> PathBuf {
        self.group_work_dir(name).join("AGENT.md")
    }

    /// Path to the global AGENT.md shared across all groups.
    pub fn global_memory_path(&self) -> PathBuf {
        self.data_dir.join("global").join("AGENT.md")
    }

    /// Directory containing SKILL.md files.
    pub fn skills_dir(&self) -> PathBuf {
        self.data_dir.join("skills")
    }

    /// Per-group session JSONL file path.
    #[allow(dead_code)]
    pub fn session_path(&self, group_name: &str) -> PathBuf {
        self.data_dir
            .join("sessions")
            .join(format!("{group_name}.jsonl"))
    }

    /// Path to the persisted scheduled tasks JSON file.
    pub fn tasks_path(&self) -> PathBuf {
        self.data_dir.join("tasks.json")
    }
}
