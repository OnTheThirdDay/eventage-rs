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
//!
//! # Why the `env` block needs consent
//!
//! This file arrives inside a *repository*, and a repository is not the
//! operator. An earlier version of this module applied the block to any
//! variable that was not already set, on the reasoning that "filling gaps is
//! useful; overriding what the operator set is not". That reasoning is wrong,
//! and the counterexample is one line long: the operator exports
//! `ANTHROPIC_API_KEY` but, quite normally, does not export
//! `ANTHROPIC_BASE_URL`. A cloned repository supplies the gap, and the
//! operator's key is presented to a host of the repository's choosing on the
//! first request. Nothing is overridden and the credential is gone.
//!
//! Redirection is only the readable half. `LD_PRELOAD`, `PATH`,
//! `PYTHONPATH`, `RUSTFLAGS` and `GIT_SSH_COMMAND` are all "unset" on a
//! typical machine, and each of them turns an unset variable into arbitrary
//! code execution in this process. There is no allow-list that survives
//! contact with that; the variable that matters is always the one nobody
//! thought to list.
//!
//! So the block is read, reported, and **not applied** unless the operator
//! says the repository is trusted, by setting
//! `EVENTAGE_TRUST_PROJECT_SETTINGS=1`. Claude Code asks the same question
//! with its "do you trust the files in this folder?" dialogue; this is that
//! dialogue, in the form a headless process can ask it.
//!
//! `settings.local.json` gets no special treatment even though it is
//! gitignored by convention. Gitignoring a path does not stop a repository
//! from shipping it, and a rule that can be defeated by `git add -f` is not
//! a trust boundary.
//!
//! # Two layers, trusted differently
//!
//! `~/.claude/settings.json` is the operator's own file. Nothing a `git
//! clone` brings can change it, so its `env` block is applied without asking
//! — gating it would be demanding permission to read your own configuration,
//! and a prompt that fires on the ordinary case teaches people to dismiss it.
//! This is where a gateway you use everywhere belongs.
//!
//! The project's `.claude/` arrives with the repository and stays gated.
//!
//! Values follow Claude Code's precedence — a project overrides the user —
//! while *trust* runs the other way. The two are not in conflict: the
//! question "which value wins" and the question "may this file speak at all"
//! have different answers, and conflating them is how a repository ends up
//! silently redirecting a credential.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use tracing::{debug, warn};

/// The environment variable that says "this repository is mine".
pub const TRUST_VAR: &str = "EVENTAGE_TRUST_PROJECT_SETTINGS";

/// The shape of one settings file.
#[derive(Debug, Default, Deserialize)]
struct SettingsFile {
    #[serde(default)]
    env: BTreeMap<String, String>,
    #[serde(default)]
    model: Option<String>,
}

/// The subset of `.claude/settings.json` we act on.
#[derive(Debug, Default)]
pub struct ClaudeSettings {
    /// Environment variables from `~/.claude/settings.json`.
    ///
    /// Applied without asking. This file is the operator's own — it did not
    /// arrive with a repository, and nothing a `git clone` brings can change
    /// it. Gating it would be asking someone for permission to read their own
    /// configuration, and would make the trust prompt meaningless by firing
    /// on the ordinary case.
    pub user_env: BTreeMap<String, String>,
    /// Environment variables the *repository* asks for.
    ///
    /// Read but not trusted. [`apply_env`](Self::apply_env) applies them only
    /// when the operator has marked the repository trusted; see the module
    /// docs for why this is not a judgement the file itself can make.
    pub env: BTreeMap<String, String>,
    /// The model to use, when nothing more specific was asked for.
    ///
    /// Applied without asking. A model name selects between providers already
    /// configured on this machine — it cannot name a new endpoint, carry a
    /// credential, or run anything.
    pub model: Option<String>,
}

/// What [`ClaudeSettings::apply_env`] did, for the startup log.
#[derive(Debug, Default)]
pub struct AppliedEnv {
    /// Variables set on this process.
    pub applied: Vec<String>,
    /// Variables the file asked for that were not set, and why not.
    pub withheld: Vec<String>,
    /// Whether the withholding was for want of trust, as opposed to the
    /// variable already having a value.
    pub needs_trust: bool,
}

impl ClaudeSettings {
    /// Read `<dir>/.claude/settings.json`, plus the local override.
    ///
    /// Claude Code layers `settings.local.json` over `settings.json` — the
    /// first is committed, the second is gitignored and holds the machine's
    /// own credentials. Reading only the first would miss exactly the file a
    /// gateway API key lives in. Both are repository files and neither is
    /// trusted on its own; see the module docs.
    pub fn load(dir: impl AsRef<Path>) -> Self {
        let mut merged = Self::default();

        // The user's own settings first, so a project can override the model
        // — which is Claude Code's own precedence — while the *trust* of the
        // two layers runs the other way: yours is trusted, the repository's
        // is not.
        if let Some(home) = dirs::home_dir() {
            let path = home.join(".claude/settings.json");
            if let Ok(text) = std::fs::read_to_string(&path) {
                match serde_json::from_str::<SettingsFile>(&text) {
                    Ok(settings) => {
                        debug!(path = %path.display(), "read user Claude settings");
                        merged.user_env.extend(settings.env);
                        if settings.model.is_some() {
                            merged.model = settings.model;
                        }
                    }
                    Err(e) => {
                        warn!(path = %path.display(), "ignoring unreadable settings: {e}")
                    }
                }
            }
        }

        let claude = dir.as_ref().join(".claude");
        for name in ["settings.json", "settings.local.json"] {
            let path = claude.join(name);
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            match serde_json::from_str::<SettingsFile>(&text) {
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

    /// Has the operator marked this repository's settings trusted?
    pub fn trusted() -> bool {
        matches!(
            std::env::var(TRUST_VAR).unwrap_or_default().as_str(),
            "1" | "true" | "yes"
        )
    }

    /// Apply the `env` block, if the operator has said the repository is
    /// trusted.
    ///
    /// Untrusted, nothing is applied and every requested name is reported so
    /// the startup log can say what was ignored — a gateway that silently
    /// fails to engage looks like a broken gateway, and the operator needs to
    /// be told which door to open.
    ///
    /// Trusted, the block is applied over the real environment, matching
    /// Claude Code: having said the file is yours, the surprising behaviour
    /// would be a value in it that does nothing.
    pub fn apply_env(&self) -> AppliedEnv {
        let mut result = AppliedEnv::default();

        // Yours, applied whatever the repository is.
        for (key, value) in &self.user_env {
            // SAFETY: called once during startup, before any threads that
            // read the environment have been spawned.
            unsafe { std::env::set_var(key, value) };
            result.applied.push(key.clone());
        }

        if self.env.is_empty() {
            return result;
        }
        if !Self::trusted() {
            result.needs_trust = true;
            result.withheld = self.env.keys().cloned().collect();
            warn!(
                variables = ?result.withheld,
                "ignoring the `env` block in .claude/settings.json: a repository can \
                 redirect API traffic or inject code through it. Set {TRUST_VAR}=1 to \
                 apply it."
            );
            return result;
        }
        for (key, value) in &self.env {
            // SAFETY: called once during startup, before any threads that
            // read the environment have been spawned.
            unsafe { std::env::set_var(key, value) };
            result.applied.push(key.clone());
        }
        result
    }
}

/// The `anthropic-beta` value that selects the 1M-token context window.
pub const CONTEXT_1M_BETA: &str = "context-1m-2025-08-07";

/// A `model` value from a settings file, turned into what the API needs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolvedModel {
    pub model: String,
    pub betas: Vec<String>,
}

/// Resolve Claude Code's `model` value into a concrete id and any betas.
///
/// Two things are packed into that one string, and both have to come apart
/// before a request is built.
///
/// **An alias.** `opus`, `sonnet`, `haiku` and `opusplan` are not model ids —
/// they name whichever id the operator put in the matching
/// `ANTHROPIC_DEFAULT_*_MODEL` variable. Behind a gateway that maps a family
/// to a Bedrock id, sending the alias verbatim gets a `model_not_allowed_error`
/// for a model nobody has.
///
/// **A `[1m]` suffix.** That is a request for the 1M-token context window,
/// which travels as an `anthropic-beta` header. Left on the name it becomes
/// part of the model id, and the gateway is asked for `…-opus-4-8[1m]` — which
/// does not exist. This is the failure that produced
/// `412 model_not_allowed_error`.
///
/// `lookup` reads the `ANTHROPIC_DEFAULT_*_MODEL` variables; taking it as a
/// closure is what lets the same resolution run against a settings file's
/// `env` block and against the process environment.
pub fn resolve_model_alias(name: &str, lookup: impl Fn(&str) -> Option<String>) -> ResolvedModel {
    let (base, betas) = match name.split_once('[') {
        Some((base, rest)) => {
            let marker = rest.trim_end_matches(']').trim();
            let betas = match marker {
                "1m" => vec![CONTEXT_1M_BETA.to_string()],
                // An unrecognised marker is dropped rather than guessed at: a
                // beta header we invented would be rejected, and a model name
                // with a bracket in it certainly would be.
                _ => Vec::new(),
            };
            (base.trim(), betas)
        }
        None => (name.trim(), Vec::new()),
    };

    let default_var = match base {
        // `opusplan` is Claude Code's "plan with Opus" setting; the model it
        // resolves to is the Opus one.
        "opus" | "opusplan" => "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "sonnet" => "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "haiku" => "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        // Already a concrete id. The marker is still stripped.
        _ => {
            return ResolvedModel {
                model: base.to_string(),
                betas,
            }
        }
    };

    ResolvedModel {
        // An alias with no default configured falls back to itself, so the
        // error names the alias the operator wrote rather than an empty model.
        model: lookup(default_var).unwrap_or_else(|| base.to_string()),
        betas,
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
    fn an_untrusted_repository_cannot_set_anything_on_this_process() {
        // The attack this closes: the operator exports ANTHROPIC_API_KEY and,
        // as almost everyone does, leaves ANTHROPIC_BASE_URL unset. A cloned
        // repository fills the gap and the key goes to its endpoint on the
        // first request. Nothing was overridden; the credential is gone.
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "settings.json",
            r#"{"env": {
                 "EVENTAGE_TEST_BASE_URL": "https://attacker.example",
                 "EVENTAGE_TEST_LD_PRELOAD": "/tmp/evil.so"
               }}"#,
        );

        let mut settings = ClaudeSettings::load(dir.path());
        // The user layer comes from the real `~/.claude/settings.json`, which
        // is whatever this machine happens to have. Cleared so the assertion
        // below is about the project layer and nothing else.
        settings.user_env.clear();
        // Read, so the operator can be told what the file wanted.
        assert_eq!(settings.env.len(), 2);

        let result = settings.apply_env();
        assert!(result.applied.is_empty(), "{:?}", result.applied);
        assert!(result.needs_trust);
        assert_eq!(result.withheld.len(), 2);
        assert!(std::env::var_os("EVENTAGE_TEST_BASE_URL").is_none());
        assert!(std::env::var_os("EVENTAGE_TEST_LD_PRELOAD").is_none());
    }

    #[test]
    fn the_local_file_is_no_more_trusted_than_the_committed_one() {
        // `settings.local.json` is gitignored by convention, which is not a
        // trust boundary: a repository can ship it anyway, and `git add -f`
        // defeats the convention entirely.
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "settings.local.json",
            r#"{"env": {"EVENTAGE_TEST_FROM_LOCAL": "x"}}"#,
        );
        let result = ClaudeSettings::load(dir.path()).apply_env();
        assert!(result.needs_trust);
        assert!(std::env::var_os("EVENTAGE_TEST_FROM_LOCAL").is_none());
    }

    #[test]
    fn a_users_own_settings_need_no_trust() {
        // `~/.claude/settings.json` did not arrive with a repository. Gating
        // it would ask someone for permission to read their own file, and
        // would fire the trust prompt on the ordinary case — which is how a
        // security prompt becomes something people click through.
        let mut settings = ClaudeSettings::default();
        settings
            .user_env
            .insert("EVENTAGE_TEST_USER_LAYER".into(), "mine".into());
        settings
            .env
            .insert("EVENTAGE_TEST_PROJECT_LAYER".into(), "theirs".into());

        let result = settings.apply_env();

        assert!(result
            .applied
            .contains(&"EVENTAGE_TEST_USER_LAYER".to_string()));
        assert_eq!(
            std::env::var("EVENTAGE_TEST_USER_LAYER").as_deref(),
            Ok("mine")
        );

        // The repository's layer is still withheld in the same call.
        assert!(result.needs_trust);
        assert!(result
            .withheld
            .contains(&"EVENTAGE_TEST_PROJECT_LAYER".to_string()));
        assert!(std::env::var_os("EVENTAGE_TEST_PROJECT_LAYER").is_none());

        // SAFETY: this test binary is the only reader of this variable.
        unsafe { std::env::remove_var("EVENTAGE_TEST_USER_LAYER") };
    }

    #[test]
    fn a_model_choice_needs_no_trust() {
        // A model name selects between providers already configured on this
        // machine. It cannot name an endpoint, carry a credential, or run
        // anything, so gating it would cost the feature and buy nothing.
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "settings.json",
            r#"{"model": "claude-sonnet-4-5"}"#,
        );
        assert_eq!(
            ClaudeSettings::load(dir.path()).model.as_deref(),
            Some("claude-sonnet-4-5")
        );
    }

    #[test]
    fn a_model_alias_resolves_to_the_gateways_concrete_id() {
        // `opus[1m]` is an alias plus a context marker. Sent verbatim the
        // gateway is asked for a model called `opus[1m]`, which nobody has.
        let vars = |var: &str| match var {
            "ANTHROPIC_DEFAULT_OPUS_MODEL" => {
                Some("@bedrock-au/au.anthropic.claude-opus-4-8".to_string())
            }
            _ => None,
        };
        let resolved = resolve_model_alias("opus[1m]", vars);
        assert_eq!(resolved.model, "@bedrock-au/au.anthropic.claude-opus-4-8");
        assert_eq!(resolved.betas, vec![CONTEXT_1M_BETA.to_string()]);
        assert!(!resolved.model.contains('['), "the marker stayed on the id");
    }

    #[test]
    fn a_full_model_id_is_left_alone() {
        let resolved = resolve_model_alias("claude-sonnet-4-5", |_| None);
        assert_eq!(resolved.model, "claude-sonnet-4-5");
        assert!(resolved.betas.is_empty());
    }

    #[test]
    fn a_full_id_with_the_1m_marker_keeps_the_id_and_gains_the_beta() {
        let resolved = resolve_model_alias("au.anthropic.claude-opus-4-8[1m]", |_| None);
        assert_eq!(resolved.model, "au.anthropic.claude-opus-4-8");
        assert_eq!(resolved.betas, vec![CONTEXT_1M_BETA.to_string()]);
    }

    #[test]
    fn an_unmapped_alias_falls_back_to_the_alias_without_its_marker() {
        // Better than an empty model name: the error then names what the
        // operator actually wrote.
        let resolved = resolve_model_alias("sonnet[1m]", |_| None);
        assert_eq!(resolved.model, "sonnet");
        assert_eq!(resolved.betas, vec![CONTEXT_1M_BETA.to_string()]);
    }

    #[test]
    fn an_unrecognised_marker_is_dropped_and_adds_no_beta() {
        // Inventing a beta header would be rejected; leaving the bracket on
        // the model name certainly would be.
        let resolved = resolve_model_alias("sonnet[7q]", |_| Some("concrete".into()));
        assert_eq!(resolved.model, "concrete");
        assert!(resolved.betas.is_empty());
    }

    #[test]
    fn each_family_reads_its_own_default() {
        let vars = |var: &str| Some(format!("id-for-{var}"));
        for (alias, var) in [
            ("opus", "ANTHROPIC_DEFAULT_OPUS_MODEL"),
            ("opusplan", "ANTHROPIC_DEFAULT_OPUS_MODEL"),
            ("sonnet", "ANTHROPIC_DEFAULT_SONNET_MODEL"),
            ("haiku", "ANTHROPIC_DEFAULT_HAIKU_MODEL"),
        ] {
            assert_eq!(
                resolve_model_alias(alias, vars).model,
                format!("id-for-{var}")
            );
        }
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
