//! Choosing a provider, endpoint and credential from the app.
//!
//! Until now the only way to point Studio at a model was to export the right
//! environment variables before starting it, which is fine on a workstation
//! you configured once and hostile everywhere else — a downloaded binary, a
//! second machine, someone trying a local Ollama for an afternoon.
//!
//! Three rules the rest of this module exists to keep.
//!
//! **The key never comes back out.** [`ModelView`] reports whether one is set,
//! never what it is. It is not in `AppInfo`, not in any event, and not in a
//! log line. A settings screen that echoes the credential back into the page
//! has put it somewhere new for no benefit — the person typing it already
//! knows it.
//!
//! **Persisting it is the user's decision, not ours.** Provider, model and
//! endpoint are remembered because they are not secrets. The credential is
//! written only when asked, and then to a file this process creates `0600`.
//!
//! **A change applies to the next session, not the running one.** The
//! provider is built into a live agent; swapping it underneath a turn would
//! mean a conversation half-answered by one model and half by another. New
//! sessions read the current settings when they open.

use anyhow::{Context, Result};
use eventage_code::config::{ModelConfig, Provider};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// Where the model configuration comes from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelSource {
    /// Typed into this screen.
    #[default]
    Manual,
    /// Read from `~/.claude/settings.json`, the file Claude Code uses.
    ///
    /// Re-read on every session rather than copied once, so editing that file
    /// takes effect without restarting Studio — and so the credential is not
    /// duplicated into a second place on disk.
    ClaudeSettings,
}

/// What a settings screen is shown. Deliberately without the credential.
#[derive(Debug, Clone, Serialize)]
pub struct ModelView {
    pub source: ModelSource,
    /// Whether `~/.claude/settings.json` currently resolves to a usable
    /// Anthropic profile, so the screen can offer that choice honestly
    /// instead of letting someone pick a source that turns out to be empty.
    pub claude_settings_available: bool,
    pub provider: String,
    pub model: String,
    /// Empty when the provider's default endpoint is in use.
    pub base_url: String,
    /// Whether a credential is configured — never which one.
    pub has_key: bool,
    /// Whether the credential is stored on disk, as opposed to held for this
    /// run only.
    pub key_remembered: bool,
    pub providers: Vec<ProviderChoice>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProviderChoice {
    pub id: String,
    pub label: String,
    pub endpoint_hint: String,
}

/// A change from the settings screen.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelUpdate {
    #[serde(default)]
    pub source: ModelSource,
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub base_url: String,
    /// Omitted to keep the existing credential; empty string to clear it.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Write the credential to disk as well as using it now.
    #[serde(default)]
    pub remember_key: bool,
}

/// What is written to disk. The credential only if asked for.
#[derive(Debug, Default, Serialize, Deserialize)]
struct Stored {
    #[serde(default)]
    source: ModelSource,
    provider: String,
    model: String,
    #[serde(default)]
    base_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    api_key: Option<String>,
}

/// The model configuration sessions are opened with.
/// A `std` lock rather than tokio's, deliberately: every read is a clone of a
/// small struct, and `info()` on the backend trait is synchronous. The only
/// slow thing here is writing the file, and that happens with no lock held.
pub struct ModelSettings {
    source: RwLock<ModelSource>,
    current: RwLock<ModelConfig>,
    /// True when the credential in `current` came from — and stays on — disk.
    remembered: RwLock<bool>,
    path: PathBuf,
}

impl ModelSettings {
    /// Load saved settings over `fallback`, which is what the environment
    /// resolved at startup.
    ///
    /// Saved settings win. Someone who chose a provider in the app expects it
    /// to still be chosen tomorrow, and an environment variable left over
    /// from a previous experiment is the more likely accident of the two.
    pub async fn load(fallback: ModelConfig, state_dir: &Path) -> Self {
        let path = state_dir.join("model.json");
        let mut current = fallback;
        let mut remembered = false;
        let mut source = ModelSource::Manual;

        if let Ok(text) = tokio::fs::read_to_string(&path).await {
            match serde_json::from_str::<Stored>(&text) {
                Ok(stored) => {
                    source = stored.source;
                    if let Some(provider) = Provider::from_id(&stored.provider) {
                        current.provider = provider;
                    }
                    if !stored.model.is_empty() {
                        current.model = stored.model;
                    }
                    current.base_url = (!stored.base_url.is_empty()).then_some(stored.base_url);
                    if let Some(key) = stored.api_key.filter(|k| !k.is_empty()) {
                        current.api_key = key;
                        current.credentialed = true;
                        remembered = true;
                    }
                }
                // Not fatal: the environment's answer is still a working one,
                // and refusing to start over a settings file would be worse
                // than ignoring it.
                Err(e) => {
                    tracing::warn!(path = %path.display(), "ignoring unreadable model settings: {e}")
                }
            }
        }

        Self {
            source: RwLock::new(source),
            current: RwLock::new(current),
            remembered: RwLock::new(remembered),
            path,
        }
    }

    /// What `~/.claude/settings.json` resolves to right now, if anything.
    ///
    /// Read each time rather than cached: the point of choosing this source
    /// is that the file stays authoritative, so editing it — or having Claude
    /// Code rewrite it — takes effect on the next session without restarting.
    fn from_claude_settings() -> Option<ModelConfig> {
        // An empty path is fine: `load` only reads `<dir>/.claude` for the
        // project layer, and the user layer comes from the home directory
        // regardless.
        let settings = eventage_code::settings::ClaudeSettings::load("");
        ModelConfig::from_claude_env(&settings.user_env, settings.model)
    }

    /// The configuration a new session should be opened with.
    pub fn get(&self) -> ModelConfig {
        match *self.source.read().unwrap_or_else(|e| e.into_inner()) {
            ModelSource::Manual => self.read().clone(),
            // Falls back to the manual profile if the file has since been
            // emptied or removed, rather than handing out a session with no
            // credential and letting the first request explain it.
            ModelSource::ClaudeSettings => {
                Self::from_claude_settings().unwrap_or_else(|| self.read().clone())
            }
        }
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, ModelConfig> {
        self.current.read().unwrap_or_else(|e| e.into_inner())
    }

    /// What the settings screen shows.
    pub fn view(&self) -> ModelView {
        let source = *self.source.read().unwrap_or_else(|e| e.into_inner());
        let from_claude = Self::from_claude_settings();
        // Describe what is actually in force, so the screen shows the
        // resolved model and endpoint rather than a stale manual profile the
        // session is not using.
        let effective = match source {
            ModelSource::ClaudeSettings => from_claude.clone(),
            ModelSource::Manual => None,
        };
        let manual = self.read();
        let current = effective.as_ref().unwrap_or(&manual);
        ModelView {
            source,
            claude_settings_available: from_claude.is_some(),
            provider: current.provider.id().to_string(),
            model: current.model.clone(),
            base_url: current.base_url.clone().unwrap_or_default(),
            // `credentialed` rather than a non-empty key: the keyless fallback
            // fills in a placeholder, so an empty check reports the opposite
            // of the truth for exactly the case that needs the warning.
            has_key: current.credentialed,
            key_remembered: *self.remembered.read().unwrap_or_else(|e| e.into_inner()),
            providers: Provider::ALL
                .into_iter()
                .map(|p| ProviderChoice {
                    id: p.id().to_string(),
                    label: p.label().to_string(),
                    endpoint_hint: p.endpoint_hint().to_string(),
                })
                .collect(),
        }
    }

    /// Apply a change, and persist what may be persisted.
    pub async fn set(&self, update: ModelUpdate) -> Result<ModelView> {
        let provider = Provider::from_id(&update.provider)
            .with_context(|| format!("unknown provider '{}'", update.provider))?;
        if update.model.trim().is_empty() {
            anyhow::bail!("give a model name");
        }

        *self.source.write().unwrap_or_else(|e| e.into_inner()) = update.source;
        {
            let mut current = self.current.write().unwrap_or_else(|e| e.into_inner());
            current.provider = provider;
            current.model = update.model.trim().to_string();
            current.base_url = match update.base_url.trim() {
                "" => None,
                url => Some(url.trim_end_matches('/').to_string()),
            };

            match update.api_key.as_deref() {
                // Not sent: the existing credential stands. A settings form
                // cannot round-trip a key it was never shown.
                None => {}
                Some("") => {
                    current.api_key.clear();
                    current.credentialed = false;
                }
                Some(key) => {
                    current.api_key = key.to_string();
                    current.credentialed = true;
                }
            }
        }

        // Snapshot under the lock, write with it released.
        let stored = {
            let current = self.read();
            Stored {
                source: update.source,
                provider: current.provider.id().to_string(),
                model: current.model.clone(),
                base_url: current.base_url.clone().unwrap_or_default(),
                api_key: match update.remember_key && current.credentialed {
                    true => Some(current.api_key.clone()),
                    false => None,
                },
            }
        };
        self.persist(stored).await?;
        *self.remembered.write().unwrap_or_else(|e| e.into_inner()) = update.remember_key;
        Ok(self.view())
    }

    async fn persist(&self, stored: Stored) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            tokio::fs::create_dir_all(parent).await.ok();
        }
        let text = serde_json::to_string_pretty(&stored)?;

        // Created 0600 *before* anything is written, not chmod'ed after:
        // between a default-mode create and a later chmod there is a moment
        // when a credential is world-readable.
        #[cfg(unix)]
        {
            use tokio::io::AsyncWriteExt;
            let mut file = tokio::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(&self.path)
                .await
                .with_context(|| format!("cannot write {}", self.path.display()))?;
            file.write_all(text.as_bytes()).await?;
            file.sync_all().await?;
        }
        #[cfg(not(unix))]
        tokio::fs::write(&self.path, text)
            .await
            .with_context(|| format!("cannot write {}", self.path.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bare() -> ModelConfig {
        let mut config = ModelConfig::from_env(Some("start-model".into()));
        config.api_key = String::new();
        config.credentialed = false;
        config.base_url = None;
        config
    }

    #[tokio::test]
    async fn the_key_is_never_reported_back() {
        // A settings screen that echoes the credential has put it somewhere
        // new for no benefit: the person typing it already knows it.
        let dir = tempfile::tempdir().unwrap();
        let settings = ModelSettings::load(bare(), dir.path()).await;

        let view = settings
            .set(ModelUpdate {
                source: ModelSource::Manual,
                provider: "qwen".into(),
                model: "qwen3-max".into(),
                base_url: "https://gateway.example/v1/".into(),
                api_key: Some("sk-secret-value".into()),
                remember_key: false,
            })
            .await
            .unwrap();

        assert!(view.has_key);
        let json = serde_json::to_string(&view).unwrap();
        assert!(!json.contains("sk-secret-value"), "{json}");

        // But the session actually gets it, and the trailing slash is gone.
        let config = settings.get();
        assert_eq!(config.api_key, "sk-secret-value");
        assert_eq!(config.provider, Provider::Qwen);
        assert_eq!(
            config.base_url.as_deref(),
            Some("https://gateway.example/v1")
        );
    }

    #[tokio::test]
    async fn a_key_is_only_written_to_disk_when_asked() {
        let dir = tempfile::tempdir().unwrap();
        let settings = ModelSettings::load(bare(), dir.path()).await;
        settings
            .set(ModelUpdate {
                source: ModelSource::Manual,
                provider: "qwen".into(),
                model: "qwen3-max".into(),
                base_url: String::new(),
                api_key: Some("sk-not-remembered".into()),
                remember_key: false,
            })
            .await
            .unwrap();

        let on_disk = std::fs::read_to_string(dir.path().join("model.json")).unwrap();
        assert!(on_disk.contains("qwen"), "{on_disk}");
        assert!(
            !on_disk.contains("sk-not-remembered"),
            "the credential was written without being asked for: {on_disk}"
        );
    }

    #[tokio::test]
    async fn a_remembered_key_comes_back_and_the_file_is_private() {
        let dir = tempfile::tempdir().unwrap();
        {
            let settings = ModelSettings::load(bare(), dir.path()).await;
            settings
                .set(ModelUpdate {
                    source: ModelSource::Manual,
                    provider: "openai-chat".into(),
                    model: "qwen3:4b".into(),
                    base_url: "http://localhost:11434/v1".into(),
                    api_key: Some("sk-remembered".into()),
                    remember_key: true,
                })
                .await
                .unwrap();
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(dir.path().join("model.json"))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(
                mode & 0o777,
                0o600,
                "the credential file is readable by others"
            );
        }

        // A fresh process, as a restart would build.
        let reopened = ModelSettings::load(bare(), dir.path()).await;
        let config = reopened.get();
        assert_eq!(config.api_key, "sk-remembered");
        assert_eq!(
            config.base_url.as_deref(),
            Some("http://localhost:11434/v1")
        );
        assert!(reopened.view().key_remembered);
    }

    #[tokio::test]
    async fn omitting_the_key_keeps_it_and_an_empty_string_clears_it() {
        // The form cannot round-trip a key it was never shown, so "not sent"
        // has to mean "leave it alone" — otherwise changing the model name
        // would silently sign you out.
        let dir = tempfile::tempdir().unwrap();
        let settings = ModelSettings::load(bare(), dir.path()).await;
        let base = |key: Option<String>| ModelUpdate {
            source: ModelSource::Manual,
            provider: "qwen".into(),
            model: "qwen3-max".into(),
            base_url: String::new(),
            api_key: key,
            remember_key: false,
        };

        settings.set(base(Some("sk-first".into()))).await.unwrap();
        settings.set(base(None)).await.unwrap();
        assert_eq!(settings.get().api_key, "sk-first");
        assert!(settings.view().has_key);

        settings.set(base(Some(String::new()))).await.unwrap();
        assert!(settings.get().api_key.is_empty());
        assert!(!settings.view().has_key);
    }

    #[tokio::test]
    async fn choosing_claude_settings_keeps_the_file_authoritative() {
        // The point of the choice is that `~/.claude/settings.json` stays the
        // source of truth: editing it, or letting Claude Code rewrite it,
        // takes effect on the next session. Copying its values into our own
        // file once would duplicate the credential onto disk and then go
        // stale.
        let home = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(home.path().join(".claude")).unwrap();
        std::fs::write(
            home.path().join(".claude/settings.json"),
            r#"{"env":{"ANTHROPIC_API_KEY":"sk-from-claude-code",
                       "ANTHROPIC_BASE_URL":"https://gw.example"},
                "model":"claude-sonnet-4-5"}"#,
        )
        .unwrap();

        // SAFETY: this test binary runs alone against the environment.
        let previous = std::env::var_os("HOME");
        unsafe { std::env::set_var("HOME", home.path()) };

        let dir = tempfile::tempdir().unwrap();
        let settings = ModelSettings::load(bare(), dir.path()).await;
        assert!(settings.view().claude_settings_available);

        settings
            .set(ModelUpdate {
                source: ModelSource::ClaudeSettings,
                provider: "qwen".into(),
                model: "ignored-while-that-source-is-chosen".into(),
                base_url: String::new(),
                api_key: None,
                remember_key: false,
            })
            .await
            .unwrap();

        // Sessions get what the file says, not what the form said.
        let config = settings.get();
        assert_eq!(config.provider, Provider::Anthropic);
        assert_eq!(config.api_key, "sk-from-claude-code");
        assert_eq!(config.base_url.as_deref(), Some("https://gw.example"));

        // Editing the file is picked up without restarting.
        std::fs::write(
            home.path().join(".claude/settings.json"),
            r#"{"env":{"ANTHROPIC_API_KEY":"sk-rotated"},"model":"claude-opus-4-6"}"#,
        )
        .unwrap();
        assert_eq!(settings.get().api_key, "sk-rotated");
        assert_eq!(settings.view().model, "claude-opus-4-6");

        // And the credential was never copied into our own file.
        let on_disk = std::fs::read_to_string(dir.path().join("model.json")).unwrap();
        assert!(!on_disk.contains("sk-from-claude-code"), "{on_disk}");
        assert!(!on_disk.contains("sk-rotated"), "{on_disk}");

        match previous {
            Some(v) => unsafe { std::env::set_var("HOME", v) },
            None => unsafe { std::env::remove_var("HOME") },
        }
    }

    #[tokio::test]
    async fn a_nonsense_provider_or_empty_model_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let settings = ModelSettings::load(bare(), dir.path()).await;
        assert!(settings
            .set(ModelUpdate {
                source: ModelSource::Manual,
                provider: "not-a-provider".into(),
                model: "m".into(),
                base_url: String::new(),
                api_key: None,
                remember_key: false,
            })
            .await
            .is_err());
        assert!(settings
            .set(ModelUpdate {
                source: ModelSource::Manual,
                provider: "qwen".into(),
                model: "   ".into(),
                base_url: String::new(),
                api_key: None,
                remember_key: false,
            })
            .await
            .is_err());
    }

    #[tokio::test]
    async fn a_broken_settings_file_does_not_stop_studio_starting() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.json"), "{ not json").unwrap();
        let settings = ModelSettings::load(bare(), dir.path()).await;
        assert_eq!(settings.get().model, "start-model");
    }
}
