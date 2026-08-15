//! A small on-disk index of past sessions.
//!
//! Session history lives in one SQLite log per session, which is the right
//! store for replay but the wrong one for drawing a sidebar: listing ten
//! sessions would mean opening and scanning ten event logs. The index keeps
//! just enough to render the list — title, workspace, when it was last
//! touched — and is rewritten whenever a session gains a title or runs a
//! turn. Losing it costs nothing: the logs are still the source of truth, and
//! a missing entry only means a session shows as untitled.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IndexEntry {
    pub cwd: String,
    pub title: String,
    pub updated_at: String,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct IndexFile {
    #[serde(default)]
    sessions: BTreeMap<String, IndexEntry>,
}

pub struct SessionIndex {
    path: PathBuf,
    file: Mutex<IndexFile>,
}

impl SessionIndex {
    /// Load the index beside the session logs, tolerating absence or damage.
    pub async fn load(dir: &Path) -> Self {
        let path = dir.join("studio-index.json");
        let file = tokio::fs::read_to_string(&path)
            .await
            .ok()
            .and_then(|text| serde_json::from_str::<IndexFile>(&text).ok())
            .unwrap_or_default();
        Self {
            path,
            file: Mutex::new(file),
        }
    }

    pub async fn record(&self, id: &str, entry: IndexEntry) {
        {
            let mut file = self.file.lock().await;
            file.sessions.insert(id.to_string(), entry);
        }
        self.flush().await;
    }

    /// Update the title only if the session does not have one yet, so a later
    /// message never renames a conversation out from under the user.
    pub async fn title_once(&self, id: &str, title: &str) {
        {
            let mut file = self.file.lock().await;
            let entry = file.sessions.entry(id.to_string()).or_default();
            entry.updated_at = chrono::Utc::now().to_rfc3339();
            if !entry.title.is_empty() {
                // Still worth persisting the fresh timestamp.
                drop(file);
                self.flush().await;
                return;
            }
            entry.title = title.to_string();
        }
        self.flush().await;
    }

    pub async fn get(&self, id: &str) -> Option<IndexEntry> {
        self.file.lock().await.sessions.get(id).cloned()
    }

    pub async fn remove(&self, id: &str) {
        {
            let mut file = self.file.lock().await;
            file.sessions.remove(id);
        }
        self.flush().await;
    }

    /// Write the index out. Failures are logged, never fatal — the index is a
    /// convenience, and the event logs it describes are unaffected.
    async fn flush(&self) {
        let snapshot = {
            let file = self.file.lock().await;
            serde_json::to_string_pretty(&*file)
        };
        let Ok(text) = snapshot else { return };
        if let Some(parent) = self.path.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }
        // Write-then-rename so a crash mid-write cannot leave a truncated
        // index that fails to parse on the next launch.
        let tmp = self.path.with_extension("json.tmp");
        if tokio::fs::write(&tmp, text).await.is_ok() {
            if let Err(e) = tokio::fs::rename(&tmp, &self.path).await {
                tracing::warn!("could not update the session index: {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_missing_index_is_an_empty_one() {
        let dir = tempfile::tempdir().unwrap();
        let index = SessionIndex::load(dir.path()).await;
        assert!(index.get("anything").await.is_none());
    }

    #[tokio::test]
    async fn a_damaged_index_does_not_stop_the_app_starting() {
        let dir = tempfile::tempdir().unwrap();
        tokio::fs::write(dir.path().join("studio-index.json"), "{not json")
            .await
            .unwrap();
        let index = SessionIndex::load(dir.path()).await;
        assert!(index.get("anything").await.is_none());
    }

    #[tokio::test]
    async fn entries_survive_a_reload() {
        let dir = tempfile::tempdir().unwrap();
        {
            let index = SessionIndex::load(dir.path()).await;
            index
                .record(
                    "abc",
                    IndexEntry {
                        cwd: "/repo".into(),
                        title: "fix the parser".into(),
                        updated_at: "2026-08-15T00:00:00Z".into(),
                    },
                )
                .await;
        }
        let reloaded = SessionIndex::load(dir.path()).await;
        assert_eq!(reloaded.get("abc").await.unwrap().title, "fix the parser");
    }

    #[tokio::test]
    async fn a_session_is_titled_once_and_never_renamed() {
        let dir = tempfile::tempdir().unwrap();
        let index = SessionIndex::load(dir.path()).await;
        index.title_once("s1", "first thing asked").await;
        index.title_once("s1", "a later message").await;
        assert_eq!(index.get("s1").await.unwrap().title, "first thing asked");
    }

    #[tokio::test]
    async fn forgetting_a_session_removes_it() {
        let dir = tempfile::tempdir().unwrap();
        let index = SessionIndex::load(dir.path()).await;
        index.title_once("s1", "gone soon").await;
        index.remove("s1").await;
        assert!(index.get("s1").await.is_none());
    }
}
