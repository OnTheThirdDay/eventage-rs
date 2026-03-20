use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use eventage::{Event, EventBus};
use serde::{Deserialize, Serialize};
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};

/// Persisted metadata for a single session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub model: String,
    pub provider_url: String,
}

/// Manages the on-disk state for one agent session.
///
/// Layout under `sessions_root/{id}/`:
/// ```
/// session.json    — metadata
/// events.jsonl    — one serialised Event per line (append-only)
/// workspace/      — the agent's file sandbox
/// ```
pub struct Session {
    pub meta: SessionMeta,
    dir: PathBuf,
}

impl Session {
    /// Create a brand-new session directory and write initial metadata.
    pub fn create(sessions_root: &Path, meta: SessionMeta) -> Result<Self> {
        let dir = sessions_root.join(&meta.id);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating session directory {}", dir.display()))?;
        std::fs::create_dir_all(dir.join("workspace"))?;
        let s = Self { meta, dir };
        s.write_meta()?;
        Ok(s)
    }

    /// Open an existing session by ID.
    pub fn open(sessions_root: &Path, id: &str) -> Result<Self> {
        let dir = sessions_root.join(id);
        let meta: SessionMeta = serde_json::from_str(
            &std::fs::read_to_string(dir.join("session.json"))
                .with_context(|| format!("session '{id}' not found"))?,
        )?;
        Ok(Self { meta, dir })
    }

    /// List all sessions under `sessions_root`, ordered newest-first.
    pub fn list(sessions_root: &Path) -> Result<Vec<SessionMeta>> {
        if !sessions_root.exists() {
            return Ok(vec![]);
        }
        let mut metas = Vec::new();
        for entry in std::fs::read_dir(sessions_root)? {
            let path = entry?.path().join("session.json");
            if path.exists() {
                if let Ok(text) = std::fs::read_to_string(&path) {
                    if let Ok(meta) = serde_json::from_str::<SessionMeta>(&text) {
                        metas.push(meta);
                    }
                }
            }
        }
        metas.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(metas)
    }

    pub fn workspace_path(&self) -> PathBuf {
        self.dir.join("workspace")
    }

    fn events_path(&self) -> PathBuf {
        self.dir.join("events.jsonl")
    }

    fn write_meta(&self) -> Result<()> {
        std::fs::write(
            self.dir.join("session.json"),
            serde_json::to_string_pretty(&self.meta)?,
        )?;
        Ok(())
    }

    /// Replay all persisted events onto `bus`.  Returns the number of events loaded.
    pub async fn load_events(&self, bus: &EventBus) -> Result<usize> {
        let path = self.events_path();
        if !path.exists() {
            return Ok(0);
        }
        let text = std::fs::read_to_string(&path)?;
        let mut count = 0usize;
        for (line_no, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let event: Event = serde_json::from_str(line)
                .with_context(|| format!("parsing event at line {}", line_no + 1))?;
            bus.publish(event)
                .await
                .map_err(|e| anyhow::anyhow!("bus error: {e}"))?;
            count += 1;
        }
        Ok(count)
    }

    /// Append events that were published after `from_index` to the JSONL file.
    pub async fn append_events(&self, bus: &EventBus, from_index: usize) -> Result<()> {
        let events = bus.log_since(from_index).await;
        if events.is_empty() {
            return Ok(());
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.events_path())?;
        for event in &events {
            writeln!(file, "{}", serde_json::to_string(event)?)?;
        }
        Ok(())
    }
}
