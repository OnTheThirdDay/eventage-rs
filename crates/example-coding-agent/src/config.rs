//! Session configuration: permission modes and tool rules.

use eventage::agent::PermissionPolicyHook;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Operating mode, surfaced to the editor through ACP `session/set_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum PermissionMode {
    /// Read-only research; edits and commands are refused with an explanation.
    Plan,
    /// Every mutating action asks the user.
    Ask,
    /// Workspace edits proceed; destructive/outbound actions ask.
    Auto,
    /// Nothing is gated.
    Yolo,
}

/// A permission mode readable from anywhere, updated when the user changes it.
///
/// Subagents and tools need the mode *now*, not the one that was in force when
/// they were constructed — a session switched from Auto to Plan mid-run was
/// still handing new subagents an Auto policy.
#[derive(Debug)]
pub struct SharedMode(std::sync::atomic::AtomicU8);

impl SharedMode {
    pub fn new(mode: PermissionMode) -> Self {
        Self(std::sync::atomic::AtomicU8::new(mode as u8))
    }

    pub fn load(&self) -> PermissionMode {
        PermissionMode::ALL[self.0.load(std::sync::atomic::Ordering::Relaxed) as usize]
    }

    pub fn store(&self, mode: PermissionMode) {
        self.0
            .store(mode as u8, std::sync::atomic::Ordering::Relaxed);
    }
}

impl PermissionMode {
    pub const ALL: [PermissionMode; 4] = [
        PermissionMode::Plan,
        PermissionMode::Ask,
        PermissionMode::Auto,
        PermissionMode::Yolo,
    ];

    pub fn id(&self) -> &'static str {
        match self {
            PermissionMode::Plan => "plan",
            PermissionMode::Ask => "ask",
            PermissionMode::Auto => "auto",
            PermissionMode::Yolo => "yolo",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            PermissionMode::Plan => "Plan",
            PermissionMode::Ask => "Ask every time",
            PermissionMode::Auto => "Auto-accept edits",
            PermissionMode::Yolo => "Full access",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            PermissionMode::Plan => "Read-only research and planning; no edits or commands",
            PermissionMode::Ask => "Approve each edit and command",
            PermissionMode::Auto => "Edits apply automatically; risky actions still ask",
            PermissionMode::Yolo => "No approval prompts at all",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.id() == id)
    }

    /// Tools that only read — always safe, never gated.
    pub const READ_ONLY_TOOLS: &'static [&'static str] = &[
        "read_file",
        "list_directory",
        "glob",
        "grep",
        "plan",
        // Named one by one rather than `lsp_*`: `lsp_rename` writes to disk,
        // and a wildcard here would have waved it through every gate.
        "lsp_diagnostics",
        "lsp_definition",
        "lsp_references",
        "lsp_hover",
        "lsp_symbols",
        // `task` is *not* here. An isolated run creates a git branch and a
        // worktree before the child's policy is applied, and creating
        // repository metadata is mutation whatever the child then does.
        "task_explore",
        "view_image",
        "repo_map",
        "jobs",
    ];

    /// Tools that modify the workspace.
    pub const EDIT_TOOLS: &'static [&'static str] = &[
        "write_file",
        "edit_file",
        "multi_edit",
        "apply_patch",
        "lsp_rename",
        // Delegating work that may edit, in a worktree it creates.
        "task",
    ];

    /// Tools that execute code, reach outside the workspace, or destroy work.
    ///
    /// `bash` is the *only* way to run anything. There used to be a second,
    /// `verify`, which took a program and arguments from a fixed list of
    /// build and test commands and ran without approval. It was added for a
    /// reason that has since expired — a subagent had no one to approve a
    /// shell command, so it could be told to verify its work and had no way
    /// to — and subagents reach the user now.
    ///
    /// What it left behind was worse than the gap it filled: a second
    /// execution path whose allow-list looked like a boundary and was not.
    /// `npm run` runs whatever the package names, `make test` runs whatever
    /// the recipe says, `cargo test` runs `build.rs` and the test bodies. It
    /// constrained the *spelling* of the command, never the code that ran —
    /// and being approval-free, it was the easiest way to execute a
    /// repository's code without anyone being asked.
    ///
    /// One path, gated once, is easier to reason about than two paths where
    /// the narrower one is narrower in name only. Repeated commands are for
    /// standing approvals to solve — those are scoped to exact arguments, so
    /// allowing `cargo test` allows `cargo test` and nothing else.
    pub const RISKY_TOOLS: &'static [&'static str] = &["bash", "git", "web_fetch"];

    /// Build the permission policy for this mode.
    ///
    /// `Plan` denies with a reason the model can act on (it should present a
    /// plan instead), rather than silently skipping.
    pub fn policy(&self) -> PermissionPolicyHook {
        let mut policy = PermissionPolicyHook::new();
        for pattern in Self::READ_ONLY_TOOLS {
            policy = policy.allow(*pattern);
        }
        match self {
            PermissionMode::Plan => {
                for pattern in Self::EDIT_TOOLS.iter().chain(Self::RISKY_TOOLS) {
                    policy = policy.deny(
                        *pattern,
                        "you are in PLAN mode: propose this change in your plan instead \
                         of performing it, then ask the user to switch to Ask or Auto mode",
                    );
                }
                policy.deny_by_default("not permitted in plan mode")
            }
            PermissionMode::Ask => {
                for pattern in Self::EDIT_TOOLS.iter().chain(Self::RISKY_TOOLS) {
                    policy = policy.ask(*pattern);
                }
                policy.ask_by_default()
            }
            PermissionMode::Auto => {
                for pattern in Self::EDIT_TOOLS {
                    policy = policy.allow(*pattern);
                }
                for pattern in Self::RISKY_TOOLS {
                    policy = policy.ask(*pattern);
                }
                policy.ask_by_default()
            }
            PermissionMode::Yolo => policy.allow("*"),
        }
    }

    /// The policy a subagent runs under.
    ///
    /// Subagents used to run with **no policy at all** — a `general` subagent
    /// could write files and run shell commands in any mode, including Plan.
    /// Then they inherited the parent's rules with every prompt turned into a
    /// refusal, because a subagent's bus had no UI on the other end and an
    /// `Ask` would have hung forever.
    ///
    /// Neither is true now. Permission requests are relayed to the parent's
    /// bus, where the editor is already listening, so a subagent can be asked
    /// about exactly like the agent that spawned it — which is what lets it
    /// have a shell at all. It therefore runs under the parent's own policy.
    ///
    /// The one adjustment: an **isolated** subagent may edit without asking,
    /// because it edits a throwaway git worktree rather than the user's files
    /// and its diff comes back for review rather than landing. Shell commands
    /// there still ask, because a worktree does not contain them.
    pub fn subagent_policy(&self, isolated: bool) -> PermissionPolicyHook {
        // Editing a disposable checkout is not editing the user's workspace,
        // and its diff comes back for review rather than landing, so an
        // isolated subagent does not prompt for every file it touches.
        if isolated && !matches!(self, PermissionMode::Plan) {
            let mut policy = PermissionPolicyHook::new();
            for pattern in Self::READ_ONLY_TOOLS {
                policy = policy.allow(*pattern);
            }
            for pattern in Self::EDIT_TOOLS {
                policy = policy.allow(*pattern);
            }
            for pattern in Self::RISKY_TOOLS {
                policy = policy.ask(*pattern);
            }
            return policy.ask_by_default();
        }
        self.policy()
    }
}

/// Which provider/model backs a session.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub provider: Provider,
    pub model: String,
    pub api_key: String,
    /// Extended-thinking budget (Anthropic) or reasoning effort mapping.
    pub thinking_tokens: Option<u32>,
    pub max_tokens: u32,
    /// Send the credential as `Authorization: Bearer` — what gateways take.
    pub bearer_auth: bool,
    /// Endpoint override, for a gateway sitting in front of the provider.
    pub base_url: Option<String>,
    /// Headers a gateway routes on, e.g. `x-portkey-provider`.
    pub headers: Vec<(String, String)>,
    /// Whether a real credential was found when this profile was resolved.
    ///
    /// Recorded here rather than re-checked later. Startup takes
    /// credential-shaped variables out of the environment, so anything asking
    /// "is `QWEN_API_KEY` set?" afterwards is asking a question whose answer
    /// has been deliberately erased — Studio's "no API key found" banner did
    /// exactly that, and told everyone with a working key that they had none.
    ///
    /// Not the same as `api_key.is_empty()`: the keyless fallback fills in a
    /// placeholder so an Ollama user still has something to send.
    pub credentialed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    OpenAiResponses,
    /// Alibaba Cloud Qwen via its `compatible-mode` Responses gateway.
    Qwen,
    OpenAiChat,
}

impl ModelConfig {
    /// Resolve provider and credentials from the environment.
    ///
    /// Prefers Anthropic, then the OpenAI Responses API, then any
    /// OpenAI-compatible endpoint (Ollama and friends need no key).
    pub fn from_env(model_override: Option<String>) -> Self {
        // `ANTHROPIC_AUTH_TOKEN` is what a gateway in front of the Messages
        // API is given, and it travels as a bearer token rather than as
        // `x-api-key` — the same split Anthropic's own SDKs make. Checked
        // first so that a workspace configured for a gateway uses the
        // gateway, not a stray key left in the environment.
        let bearer = std::env::var("ANTHROPIC_AUTH_TOKEN").ok();
        if let Some(key) = bearer
            .clone()
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
        {
            return Self {
                provider: Provider::Anthropic,
                model: model_override
                    .or_else(|| std::env::var("ANTHROPIC_MODEL").ok())
                    .unwrap_or_else(|| "claude-sonnet-4-5".into()),
                api_key: key,
                thinking_tokens: Some(8_000),
                max_tokens: 32_000,
                bearer_auth: bearer.is_some(),
                credentialed: true,
                base_url: std::env::var("ANTHROPIC_BASE_URL").ok(),
                headers: std::env::var("ANTHROPIC_CUSTOM_HEADERS")
                    .map(|raw| crate::settings::parse_custom_headers(&raw))
                    .unwrap_or_default(),
            };
        }
        // Captured here, with everything else, rather than read again later.
        let endpoint = std::env::var("OPENAI_BASE_URL").ok();

        // Qwen first: it needs its own dialect, not OpenAI's.
        if let Ok(key) = std::env::var("QWEN_API_KEY") {
            return Self {
                provider: Provider::Qwen,
                model: model_override.unwrap_or_else(|| "qwen3-max".into()),
                api_key: key,
                thinking_tokens: None,
                max_tokens: 32_000,
                credentialed: true,
                base_url: endpoint,
                ..Self::gatewayless()
            };
        }
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            return Self {
                provider: Provider::OpenAiResponses,
                model: model_override.unwrap_or_else(|| "gpt-5".into()),
                api_key: key,
                thinking_tokens: None,
                max_tokens: 32_000,
                credentialed: true,
                base_url: endpoint,
                ..Self::gatewayless()
            };
        }
        // Nothing was found: the fallback points at a local server and says
        // so, rather than pretending to be configured.
        Self {
            provider: Provider::OpenAiChat,
            model: model_override.unwrap_or_else(|| "qwen3:4b".into()),
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "ollama".into()),
            thinking_tokens: None,
            max_tokens: 8_000,
            base_url: endpoint,
            ..Self::gatewayless()
        }
    }

    /// Defaults for the gateway fields, which only Anthropic reads today.
    fn gatewayless() -> Self {
        Self {
            provider: Provider::OpenAiChat,
            model: String::new(),
            api_key: String::new(),
            thinking_tokens: None,
            max_tokens: 0,
            bearer_auth: false,
            base_url: None,
            headers: Vec::new(),
            credentialed: false,
        }
    }

    /// Base URL for OpenAI-compatible providers.
    ///
    /// `OPENAI_BASE_URL` points at any compatible gateway (Aliyun, Azure,
    /// OpenRouter, vLLM, Ollama…); the default is a local Ollama.
    /// The endpoint this config was resolved against.
    ///
    /// Read from the stored field, never from the environment. It used to
    /// re-read `OPENAI_BASE_URL` here, at provider-construction time — which
    /// is *after* startup takes credential-shaped variables out of the
    /// environment, and `OPENAI_*` is credential-shaped. A session pointed at
    /// a local vLLM, an Ollama instance or a private gateway therefore fell
    /// back to a hard-coded default, and for the Responses provider that
    /// default is `api.openai.com`: the operator's key and the repository's
    /// contents went to an endpoint they had not chosen.
    ///
    /// The scrub was added to stop credentials leaking. Reading configuration
    /// lazily out of the same namespace turned it into a credential-routing
    /// bug — which is why endpoint, credential, model, auth mode and headers
    /// are now captured together, once, and never re-derived.
    pub fn base_url(&self) -> String {
        self.base_url
            .clone()
            .unwrap_or_else(|| match self.provider {
                Provider::OpenAiResponses => "https://api.openai.com/v1".into(),
                Provider::Qwen => "https://dashscope-intl.aliyuncs.com/compatible-mode/v1".into(),
                _ => "http://localhost:11434/v1".into(),
            })
    }
}

/// Eight hex characters identifying a workspace by its canonical path.
fn workspace_digest(cwd: &str) -> String {
    use std::hash::{Hash, Hasher};
    let canonical = std::fs::canonicalize(cwd)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| cwd.to_string());
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    canonical.hash(&mut hasher);
    format!("{:08x}", hasher.finish() as u32)
}

/// Is this a session identifier we are willing to turn into a file path?
///
/// The id arrives from the client — `session/load` over ACP, a resume request
/// in Studio — and is joined into the state directory to name a database. A
/// `..` in it walks out of that directory; a `/` puts the file somewhere else
/// entirely. Ids we mint are UUIDs, so requiring that shape costs nothing and
/// removes the question.
pub fn is_valid_session_id(id: &str) -> bool {
    uuid::Uuid::parse_str(id).is_ok()
}

/// An MCP server the editor asked us to connect.
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    /// Prefix for this server's tools in the registry.
    pub name: String,
    /// stdio transport.
    pub command: Option<String>,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    /// HTTP transport.
    pub url: Option<String>,
}

/// Session-scoped settings.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub cwd: String,
    pub mode: PermissionMode,
    pub model: ModelConfig,
    /// MCP servers supplied by the client on `session/new`.
    pub mcp_servers: Vec<McpServerConfig>,
    /// Token budget for the whole session (0 = unlimited).
    pub token_budget: u64,
    /// Context window to keep assembled context inside.
    pub context_tokens: usize,
    /// How shell commands are contained.
    pub shell: crate::tools::ShellContainment,
    /// Image for container containment.
    pub container_image: String,
}

impl SessionConfig {
    pub fn new(cwd: impl Into<String>, model: ModelConfig) -> Self {
        Self {
            cwd: cwd.into(),
            mode: PermissionMode::Ask,
            model,
            mcp_servers: Vec::new(),
            token_budget: 0,
            context_tokens: 160_000,
            shell: crate::tools::ShellContainment::Confined,
            container_image: crate::tools::DEFAULT_CONTAINER_IMAGE.to_string(),
        }
    }

    /// Directory holding session state and logs for this workspace.
    ///
    /// Named for the directory *and* a digest of its canonical path. The name
    /// alone is not an identity: everybody has more than one checkout called
    /// `api`, and they were sharing a state directory — one repository's
    /// sessions listed under another, resumable against the wrong code.
    ///
    /// A directory left by the older, name-only scheme is still used if it
    /// exists, so nobody's history disappears on upgrade.
    pub fn state_dir(&self) -> std::path::PathBuf {
        let base = dirs::data_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("eventage-code");

        let name: String = Path::new(&self.cwd)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("workspace")
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '-' })
            .collect();

        let legacy = base.join(&name);
        if legacy.is_dir() {
            return legacy;
        }
        base.join(format!("{name}-{}", workspace_digest(&self.cwd)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eventage::agent::PermissionVerdict;

    #[test]
    fn mode_ids_round_trip() {
        for mode in PermissionMode::ALL {
            assert_eq!(PermissionMode::from_id(mode.id()), Some(mode));
        }
        assert_eq!(PermissionMode::from_id("nope"), None);
    }

    #[test]
    fn plan_mode_denies_edits_with_actionable_reason() {
        let policy = PermissionMode::Plan.policy();
        match policy.verdict_for("edit_file") {
            PermissionVerdict::Deny(reason) => {
                assert!(reason.contains("PLAN mode"), "{reason}");
                assert!(reason.contains("plan instead"), "{reason}");
            }
            other => panic!("expected deny, got {other:?}"),
        }
        // Reads stay available so the agent can still investigate.
        assert!(matches!(
            policy.verdict_for("read_file"),
            PermissionVerdict::Allow
        ));
        assert!(matches!(
            policy.verdict_for("lsp_definition"),
            PermissionVerdict::Allow
        ));
    }

    #[test]
    fn renaming_is_gated_like_any_other_write() {
        // It lives among the `lsp_*` tools and reads like a query, but it
        // rewrites every file the symbol appears in.
        assert!(matches!(
            PermissionMode::Ask.policy().verdict_for("lsp_rename"),
            PermissionVerdict::Ask
        ));
        assert!(matches!(
            PermissionMode::Plan.policy().verdict_for("lsp_rename"),
            PermissionVerdict::Deny(_)
        ));
        // While its read-only neighbours stay ungated.
        assert!(matches!(
            PermissionMode::Ask.policy().verdict_for("lsp_references"),
            PermissionVerdict::Allow
        ));
    }

    #[test]
    fn a_subagent_is_asked_about_exactly_like_its_parent() {
        // This used to assert the opposite — that nothing may `Ask`, because
        // a subagent's bus had no UI on the other end and a prompt would hang
        // forever. Requests are relayed to the parent's bus now, so the
        // limitation is gone and with it the reason to deny.
        for mode in [PermissionMode::Ask, PermissionMode::Auto] {
            let parent = mode.policy();
            let child = mode.subagent_policy(false);
            for tool in ["write_file", "edit_file", "bash", "web_fetch", "read_file"] {
                assert_eq!(
                    format!("{:?}", child.verdict_for(tool)),
                    format!("{:?}", parent.verdict_for(tool)),
                    "{mode:?}/{tool}: a subagent should be governed like its parent"
                );
            }
        }

        // Plan still means plan, for the child as much as the parent.
        let plan = PermissionMode::Plan.subagent_policy(false);
        assert!(matches!(
            plan.verdict_for("write_file"),
            PermissionVerdict::Deny(_)
        ));
        assert!(matches!(
            plan.verdict_for("bash"),
            PermissionVerdict::Deny(_)
        ));
    }

    #[test]
    fn an_isolated_subagent_edits_its_throwaway_checkout_without_asking() {
        // Its diff comes back for review rather than landing, so prompting
        // per file would be noise. The shell is a different matter: a
        // worktree does not contain it.
        let isolated = PermissionMode::Auto.subagent_policy(true);
        assert!(matches!(
            isolated.verdict_for("edit_file"),
            PermissionVerdict::Allow
        ));
        assert!(matches!(
            isolated.verdict_for("bash"),
            PermissionVerdict::Ask
        ));

        // Even in Ask mode, where the parent would prompt for every edit.
        let cautious = PermissionMode::Ask.subagent_policy(true);
        assert!(matches!(
            cautious.verdict_for("edit_file"),
            PermissionVerdict::Allow
        ));

        // Plan means plan, isolated or not.
        assert!(matches!(
            PermissionMode::Plan
                .subagent_policy(true)
                .verdict_for("edit_file"),
            PermissionVerdict::Deny(_)
        ));
    }

    #[test]
    fn auto_mode_allows_edits_but_gates_shell() {
        let policy = PermissionMode::Auto.policy();
        assert!(matches!(
            policy.verdict_for("edit_file"),
            PermissionVerdict::Allow
        ));
        assert!(matches!(policy.verdict_for("bash"), PermissionVerdict::Ask));
    }

    #[test]
    fn ask_mode_gates_mutations_only() {
        let policy = PermissionMode::Ask.policy();
        assert!(matches!(
            policy.verdict_for("read_file"),
            PermissionVerdict::Allow
        ));
        assert!(matches!(
            policy.verdict_for("write_file"),
            PermissionVerdict::Ask
        ));
    }

    #[test]
    fn yolo_allows_everything() {
        let policy = PermissionMode::Yolo.policy();
        for tool in ["bash", "edit_file", "anything_else"] {
            assert!(matches!(policy.verdict_for(tool), PermissionVerdict::Allow));
        }
    }
}
