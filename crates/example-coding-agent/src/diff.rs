//! Turn-level diff tracking via [`TurnDiffWorker`].
//!
//! The worker listens for `agent.cycle.start` and `agent.cycle.end` events.
//! At cycle start it snapshots the current workspace files (path + SHA-256
//! hash of contents). At cycle end it compares the current state against the
//! snapshot and computes unified diffs for every changed or new file.
//!
//! The diff summary is published as a [`kinds::CODING_TURN_DIFF`] event so
//! other components (e.g., the TUI, an observability exporter, a test harness)
//! can react to file changes without polling the filesystem.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use eventage::agent::{EventWorker, WorkerError};
use eventage::{kinds as core_kinds, Event, EventBus};
use serde_json::json;
use sha2::{Digest, Sha256};
use similar::{ChangeTag, TextDiff};
use tokio::sync::Mutex;
use tracing::debug;

use crate::kinds::CODING_TURN_DIFF;
use crate::workspace::Workspace;

// ── FileSha ───────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct FileSnap {
    sha256: String,
    content: String,
}

// ── TurnDiffWorker ────────────────────────────────────────────────────────────

/// An [`EventWorker`] that snapshots workspace files at the start of each
/// agent cycle and publishes unified diffs at the end.
///
/// Published event payload shape:
/// ```json
/// {
///   "changed_files": 2,
///   "new_files": 1,
///   "deleted_files": 0,
///   "diffs": {
///     "src/main.py": "--- a/src/main.py\n+++ b/src/main.py\n@@ ...",
///     "README.md":   "--- /dev/null\n+++ b/README.md\n@@ ..."
///   }
/// }
/// ```
pub struct TurnDiffWorker {
    workspace: Arc<Workspace>,
    /// Baseline snapshot taken at `agent.cycle.start`.
    baseline: Mutex<HashMap<String, FileSnap>>,
}

impl TurnDiffWorker {
    pub fn new(workspace: Arc<Workspace>) -> Self {
        Self {
            workspace,
            baseline: Mutex::new(HashMap::new()),
        }
    }

    /// Snapshot all current workspace files.
    async fn snapshot(&self) -> HashMap<String, FileSnap> {
        let mut snaps = HashMap::new();
        let Ok(files) = self.workspace.list_files() else {
            return snaps;
        };
        for entry in &files {
            let abs = match self.workspace.resolve(&entry.path) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let Ok(content) = std::fs::read_to_string(&abs) else {
                continue; // skip binary files
            };
            let sha256 = hex::encode(Sha256::digest(content.as_bytes()));
            snaps.insert(entry.path.clone(), FileSnap { sha256, content });
        }
        snaps
    }
}

#[async_trait]
impl EventWorker for TurnDiffWorker {
    fn subscribed_kinds(&self) -> Vec<String> {
        vec![
            core_kinds::AGENT_CYCLE_START.to_string(),
            core_kinds::AGENT_CYCLE_END.to_string(),
        ]
    }

    async fn handle(&self, event: &Event, bus: &EventBus) -> Result<(), WorkerError> {
        match event.kind.as_str() {
            // ── Cycle start: take baseline snapshot ───────────────────────────
            k if k == core_kinds::AGENT_CYCLE_START => {
                let snap = self.snapshot().await;
                debug!(
                    files = snap.len(),
                    "TurnDiffWorker: baseline snapshot taken"
                );
                *self.baseline.lock().await = snap;
            }

            // ── Cycle end: diff against baseline ──────────────────────────────
            k if k == core_kinds::AGENT_CYCLE_END => {
                let baseline = self.baseline.lock().await.clone();
                let current = self.snapshot().await;

                let mut diffs: HashMap<String, String> = HashMap::new();
                let mut changed = 0usize;
                let mut new_files = 0usize;
                let mut deleted = 0usize;

                // Changed and new files.
                for (path, cur_snap) in &current {
                    match baseline.get(path) {
                        Some(old) if old.sha256 == cur_snap.sha256 => {
                            // Unchanged — skip.
                        }
                        Some(old) => {
                            // Changed file — compute unified diff.
                            changed += 1;
                            let diff = compute_unified_diff(
                                &format!("a/{path}"),
                                &format!("b/{path}"),
                                &old.content,
                                &cur_snap.content,
                            );
                            diffs.insert(path.clone(), diff);
                        }
                        None => {
                            // New file.
                            new_files += 1;
                            let diff = compute_unified_diff(
                                "/dev/null",
                                &format!("b/{path}"),
                                "",
                                &cur_snap.content,
                            );
                            diffs.insert(path.clone(), diff);
                        }
                    }
                }

                // Deleted files.
                for path in baseline.keys() {
                    if !current.contains_key(path) {
                        deleted += 1;
                        let old = &baseline[path];
                        let diff = compute_unified_diff(
                            &format!("a/{path}"),
                            "/dev/null",
                            &old.content,
                            "",
                        );
                        diffs.insert(path.clone(), diff);
                    }
                }

                if !diffs.is_empty() {
                    debug!(
                        changed,
                        new_files, deleted, "TurnDiffWorker: publishing diff"
                    );
                    bus.publish(Event::new(
                        CODING_TURN_DIFF,
                        json!({
                            "changed_files": changed,
                            "new_files": new_files,
                            "deleted_files": deleted,
                            "diffs": diffs,
                        }),
                    ))
                    .await
                    .map_err(WorkerError::Bus)?;
                }
            }

            _ => {}
        }
        Ok(())
    }
}

// ── Unified diff helper ───────────────────────────────────────────────────────

fn compute_unified_diff(old_label: &str, new_label: &str, old: &str, new: &str) -> String {
    let diff = TextDiff::from_lines(old, new);
    let mut out = format!("--- {old_label}\n+++ {new_label}\n");

    for group in diff.grouped_ops(3) {
        // Compute hunk header.
        let first = group.first().unwrap();
        let _last = group.last().unwrap();

        let old_start = first.old_range().start + 1;
        let old_len: usize = group.iter().map(|op| op.old_range().len()).sum();
        let new_start = first.new_range().start + 1;
        let new_len: usize = group.iter().map(|op| op.new_range().len()).sum();

        out.push_str(&format!(
            "@@ -{},{} +{},{} @@\n",
            old_start, old_len, new_start, new_len
        ));

        for op in &group {
            for change in diff.iter_changes(op) {
                let tag = match change.tag() {
                    ChangeTag::Delete => '-',
                    ChangeTag::Insert => '+',
                    ChangeTag::Equal => ' ',
                };
                out.push(tag);
                out.push_str(change.value());
                if change.missing_newline() {
                    out.push('\n');
                }
            }
        }
    }

    out
}
