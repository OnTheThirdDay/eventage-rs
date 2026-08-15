//! `.claude/settings.json`, for the parts of it that mean something here.
//!
//! We already read `CLAUDE.md` and `.claude/skills/`, so a repository set up
//! for Claude Code mostly works. The gap was configuration: the way every
//! Anthropic-compatible gateway is wired up — Portkey, LiteLLM, Helicone,
//! Bedrock proxies — is an `env` block in this file, because Claude Code
//! applies it to its own process. Without reading it, pointing this agent at
//! a gateway meant exporting the same variables by hand.
//!
//! Deliberately partial. `env` and `model` transfer cleanly. `hooks` do not:
//! they are shell commands bound to Claude Code's event names, and our hooks
//! are in-process Rust traits, so honouring the key by halves would be worse
//! than ignoring it. `permissions` is a real gap but a bigger design question
//! than a parse, since ours are compiled-in lists. Unknown keys are ignored
//! rather than rejected — this file belongs to another tool, and failing on
//! a key it adds next month would be rude.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use tracing::{debug, warn};

/// The subset of `.claude/settings.json` we act on.
#[derive(Debug, Default, Deserialize)]
pub struct ClaudeSettings {
    /// Environment variables to apply to this process.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// The model to use, when nothing more specific was asked for.
    #[serde(default)]
    pub model: Option<String>,
}

impl ClaudeSettings {
    /// Read `<dir>/.claude/settings.json`, plus the untracked local override.
    ///
    /// Claude Code layers `settings.local.json` over `settings.json` — the
    /// first is committed, the second is gitignored and holds the machine's
    /// own credentials. Reading only the first would miss exactly the file a
    /// gateway API key lives in.
    pub fn load(dir: impl AsRef<Path>) -> Self {
        let claude = dir.as_ref().join(".claude");
        let mut merged = Self::default();
        for name in ["settings.json", "settings.local.json"] {
            let path = claude.join(name);
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            match serde_json::from_str::<Self>(&text) {
                Ok(settings) => {
                    debug!(path = %path.display(), "read Claude settings");
                    merged.env.extend(settings.env);
                    if settings.model.is_some() {
                        merged.model = settings.model;
                    }
                }
                // A malformed file is worth saying out loud: silently running
                // without a gateway that was configured looks like the
                // gateway is broken.
                Err(e) => warn!(path = %path.display(), "ignoring unreadable settings: {e}"),
            }
        }
        merged
    }

    /// Apply the `env` block to this process, without overwriting anything.
    ///
    /// The real environment wins. Claude Code lets this file override the
    /// process it launches, but here the file comes out of a *repository* —
    /// possibly one that was just cloned — and a checked-in `env` block that
    /// could silently redirect an agent's API traffic to a host of its
    /// choosing is not a thing a repository should be able to do. Filling
    /// gaps is useful; overriding what the operator set is not.
    ///
    /// Returns the names it set, for the startup log.
    pub fn apply_env(&self) -> Vec<String> {
        let mut applied = Vec::new();
        for (key, value) in &self.env {
            if std::env::var_os(key).is_some() {
                continue;
            }
            // SAFETY: called once during startup, before any threads that
            // read the environment have been spawned.
            unsafe { std::env::set_var(key, value) };
            applied.push(key.clone());
        }
        applied
    }
}

/// Parse `ANTHROPIC_CUSTOM_HEADERS` into header pairs.
///
/// Claude Code's format, which gateways document against: `Name: value`, one
/// per line. A value may itself contain a colon (a URL, a base64 token), so
/// only the first one separates.
pub fn parse_custom_headers(raw: &str) -> Vec<(String, String)> {
    raw.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() {
                return None;
            }
            let (name, value) = line.split_once(':')?;
            let (name, value) = (name.trim(), value.trim());
            (!name.is_empty() && !value.is_empty()).then(|| (name.to_string(), value.to_string()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, body: &str) {
        let claude = dir.join(".claude");
        std::fs::create_dir_all(&claude).unwrap();
        std::fs::write(claude.join(name), body).unwrap();
    }

    #[test]
    fn a_gateway_configuration_is_read_from_the_env_block() {
        // Verbatim from Portkey's Claude Code integration guide.
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "settings.json",
            r#"{
              "env": {
                "ANTHROPIC_BASE_URL": "https://api.portkey.ai",
                "ANTHROPIC_AUTH_TOKEN": "pk-test",
                "ANTHROPIC_CUSTOM_HEADERS": "x-portkey-api-key: pk-test\nx-portkey-provider: @my-provider",
                "ANTHROPIC_MODEL": "claude-sonnet-4-5"
              },
              "model": "claude-sonnet-4-5"
            }"#,
        );

        let settings = ClaudeSettings::load(dir.path());
        assert_eq!(
            settings.env.get("ANTHROPIC_BASE_URL").map(String::as_str),
            Some("https://api.portkey.ai")
        );
        assert_eq!(settings.model.as_deref(), Some("claude-sonnet-4-5"));

        let headers = parse_custom_headers(settings.env["ANTHROPIC_CUSTOM_HEADERS"].as_str());
        assert_eq!(
            headers,
            vec![
                ("x-portkey-api-key".to_string(), "pk-test".to_string()),
                ("x-portkey-provider".to_string(), "@my-provider".to_string()),
            ]
        );
    }

    #[test]
    fn the_local_override_wins_over_the_committed_file() {
        // The committed file names the gateway; the gitignored one holds the
        // key. Reading only the first would find no credential at all.
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "settings.json",
            r#"{"env": {"ANTHROPIC_BASE_URL": "https://api.portkey.ai"}, "model": "a"}"#,
        );
        write(
            dir.path(),
            "settings.local.json",
            r#"{"env": {"ANTHROPIC_AUTH_TOKEN": "secret"}, "model": "b"}"#,
        );

        let settings = ClaudeSettings::load(dir.path());
        assert_eq!(settings.env.len(), 2);
        assert_eq!(settings.model.as_deref(), Some("b"));
    }

    #[test]
    fn keys_we_do_not_implement_are_ignored_rather_than_fatal() {
        // This file belongs to another tool and will grow keys we have never
        // heard of. Refusing to start over one would be absurd.
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "settings.json",
            r#"{
              "env": {"ANTHROPIC_MODEL": "m"},
              "permissions": {"allow": ["Bash(cargo test:*)"]},
              "hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": []}]},
              "statusLine": {"type": "command", "command": "x"},
              "somethingInventedNextMonth": 42
            }"#,
        );
        let settings = ClaudeSettings::load(dir.path());
        assert_eq!(
            settings.env.get("ANTHROPIC_MODEL").map(String::as_str),
            Some("m")
        );
    }

    #[test]
    fn a_broken_file_does_not_stop_the_agent_starting() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "settings.json", "{ not json");
        assert!(ClaudeSettings::load(dir.path()).env.is_empty());
        // And no file at all is simply nothing.
        let empty = tempfile::tempdir().unwrap();
        assert!(ClaudeSettings::load(empty.path()).env.is_empty());
    }

    #[test]
    fn header_values_may_contain_colons() {
        let headers = parse_custom_headers(
            "x-trace-url: https://host/x\n\n  x-key : v  \nnovalue:\n:noname\ngarbage",
        );
        assert_eq!(
            headers,
            vec![
                ("x-trace-url".to_string(), "https://host/x".to_string()),
                ("x-key".to_string(), "v".to_string()),
            ]
        );
    }
}
