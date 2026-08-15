//! Session configuration: permission modes and tool rules.

use eventage::agent::PermissionPolicyHook;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Operating mode, surfaced to the editor through ACP `session/set_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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
        "task",
        "view_image",
    ];

    /// Tools that modify the workspace.
    pub const EDIT_TOOLS: &'static [&'static str] = &[
        "write_file",
        "edit_file",
        "multi_edit",
        "apply_patch",
        "lsp_rename",
    ];

    /// Tools that reach outside the workspace or can destroy work.
    pub const RISKY_TOOLS: &'static [&'static str] =
        &["bash", "git", "web_fetch"];

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
    /// They inherit the parent's rules now, with two adjustments that follow
    /// from what a subagent is:
    ///
    /// * There is nobody to ask. A subagent's bus has no UI on the other end,
    ///   so anything the parent would have prompted about is denied instead,
    ///   with a reason the subagent can put in its report.
    /// * An **isolated** subagent may still edit, because it edits a throwaway
    ///   git worktree rather than the user's files, and its diff comes back
    ///   for review rather than landing.
    pub fn subagent_policy(&self, isolated: bool) -> PermissionPolicyHook {
        const NO_ONE_TO_ASK: &str =
            "this needs the user's approval and a subagent has nobody to ask — \
             report what you need and let your caller do it";

        let mut policy = PermissionPolicyHook::new();
        for pattern in Self::READ_ONLY_TOOLS {
            policy = policy.allow(*pattern);
        }

        if matches!(self, PermissionMode::Yolo) {
            return policy.allow("*");
        }

        // Editing a disposable checkout is not editing the user's workspace.
        let edits_allowed = isolated && !matches!(self, PermissionMode::Plan);
        for pattern in Self::EDIT_TOOLS {
            policy = if edits_allowed {
                policy.allow(*pattern)
            } else if matches!(self, PermissionMode::Plan) {
                policy.deny(
                    *pattern,
                    "you are part of a session in PLAN mode: describe the change, \
                     do not make it",
                )
            } else {
                policy.deny(*pattern, NO_ONE_TO_ASK)
            };
        }
        for pattern in Self::RISKY_TOOLS {
            policy = policy.deny(*pattern, NO_ONE_TO_ASK);
        }
        policy.deny_by_default(NO_ONE_TO_ASK)
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
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            return Self {
                provider: Provider::Anthropic,
                model: model_override.unwrap_or_else(|| "claude-sonnet-4-5".into()),
                api_key: key,
                thinking_tokens: Some(8_000),
                max_tokens: 32_000,
            };
        }
        // Qwen first: it needs its own dialect, not OpenAI's.
        if let Ok(key) = std::env::var("QWEN_API_KEY") {
            return Self {
                provider: Provider::Qwen,
                model: model_override.unwrap_or_else(|| "qwen3-max".into()),
                api_key: key,
                thinking_tokens: None,
                max_tokens: 32_000,
            };
        }
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            return Self {
                provider: Provider::OpenAiResponses,
                model: model_override.unwrap_or_else(|| "gpt-5".into()),
                api_key: key,
                thinking_tokens: None,
                max_tokens: 32_000,
            };
        }
        Self {
            provider: Provider::OpenAiChat,
            model: model_override.unwrap_or_else(|| "qwen3:4b".into()),
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "ollama".into()),
            thinking_tokens: None,
            max_tokens: 8_000,
        }
    }

    /// Base URL for OpenAI-compatible providers.
    ///
    /// `OPENAI_BASE_URL` points at any compatible gateway (Aliyun, Azure,
    /// OpenRouter, vLLM, Ollama…); the default is a local Ollama.
    pub fn base_url(&self) -> String {
        std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| match self.provider {
            Provider::OpenAiResponses => "https://api.openai.com/v1".into(),
            Provider::Qwen => {
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1".into()
            }
            _ => "http://localhost:11434/v1".into(),
        })
    }


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
        }
    }

    /// Directory holding session state and logs for this workspace.
    pub fn state_dir(&self) -> std::path::PathBuf {
        let base = dirs::data_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("eventage-code");
        let slug: String = Path::new(&self.cwd)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("workspace")
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '-' })
            .collect();
        base.join(slug)
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
    fn a_subagent_cannot_do_what_its_parent_would_have_had_to_ask_about() {
        // Subagents ran with no policy at all: a `general` one could write
        // files and run shell commands in every mode, Plan included.
        let plan = PermissionMode::Plan.subagent_policy(false);
        assert!(matches!(plan.verdict_for("write_file"), PermissionVerdict::Deny(_)));
        assert!(matches!(plan.verdict_for("bash"), PermissionVerdict::Deny(_)));

        // Nothing may *ask*, because a subagent has no user attached — a
        // prompt would hang on a bus with no UI on the other end.
        for mode in [PermissionMode::Ask, PermissionMode::Auto, PermissionMode::Plan] {
            let policy = mode.subagent_policy(false);
            for tool in ["write_file", "edit_file", "bash", "web_fetch", "lsp_rename"] {
                let verdict = policy.verdict_for(tool);
                assert!(
                    !matches!(verdict, PermissionVerdict::Ask),
                    "{mode:?}/{tool} would hang waiting for an answer"
                );
            }
        }

        // Reading is always fine; that is what most subagents are for.
        assert!(matches!(
            PermissionMode::Ask.subagent_policy(false).verdict_for("grep"),
            PermissionVerdict::Allow
        ));
    }

    #[test]
    fn an_isolated_subagent_may_still_edit_its_throwaway_checkout() {
        // Otherwise implementation subagents stop working entirely — and
        // editing a disposable worktree whose diff comes back for review is
        // not editing the user's files.
        let isolated = PermissionMode::Auto.subagent_policy(true);
        assert!(matches!(isolated.verdict_for("edit_file"), PermissionVerdict::Allow));
        // The shell still is not, worktree or no worktree: it reaches
        // everything outside the checkout.
        assert!(matches!(isolated.verdict_for("bash"), PermissionVerdict::Deny(_)));

        // Plan mode means plan, isolated or not.
        assert!(matches!(
            PermissionMode::Plan.subagent_policy(true).verdict_for("edit_file"),
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
        assert!(matches!(
            policy.verdict_for("bash"),
            PermissionVerdict::Ask
        ));
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
