//! The agent's tool suite: filesystem, search, execution, and planning.
//!
//! Tool results are plain JSON. Fields the editor cares about are named by
//! convention so the ACP bridge can turn them into rich UI without any
//! tool-specific knowledge:
//!
//! - `_diff: { path, old_text, new_text }` → a reviewable diff card
//! - `_locations: [{ path, line }]`        → "follow along" in the file tree
//! - `_plan: [...]`                        → the task checklist

pub mod git;
pub mod intel;
pub mod patch;
pub mod task;
pub mod vision;

use crate::acp::ClientFs;
use crate::lsp::LspPool;
use crate::workspace::Workspace;
use anyhow::Result;
use async_trait::async_trait;
use eventage::agent::{AgentError, Tool};
use eventage::llm::ToolDefinition;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Maximum bytes returned by a single file read.
const MAX_READ_BYTES: usize = 400_000;
/// Maximum matches returned by one search.
const MAX_MATCHES: usize = 200;

/// Read a file, preferring the editor's (possibly unsaved) buffer.
///
/// Takes the workspace and a *relative* path rather than an absolute one, so
/// the read goes through the workspace's directory handle. Handing an
/// absolute path to the ambient filesystem is what let a symlink out of the
/// repository.
///
/// The editor route is checked first and is a separate trust decision: the
/// path is confined before it is offered, and the client only ever returns
/// buffers for files it has open.
async fn read_source(
    ws: &Workspace,
    client: &Option<ClientFs>,
    rel: &str,
) -> Result<String, AgentError> {
    let abs = ws
        .resolve(rel)
        .map_err(|e| AgentError::Tool(e.to_string()))?;
    if let Some(client) = client {
        if let Some(text) = client.read(&abs.display().to_string()).await {
            // Recorded here as well as on the disk path. Without it the
            // stale-write guard was inert for exactly the files the editor
            // has open — the ones a human is most likely to be editing at
            // the same moment.
            ws.remember_source(rel, &text);
            return Ok(text);
        }
    }
    ws.read_to_string(rel)
        .await
        .map_err(|e| AgentError::Tool(format!("{e:#}")))
}

/// Fail if the file changed since it was last read, checking whichever
/// source the write will actually go to.
///
/// `Workspace::ensure_unchanged` reads from disk, which is the wrong
/// comparison when the editor holds an unsaved buffer: it would compare the
/// agent's remembered text against a file the editor has not written yet and
/// refuse a perfectly good edit. Asking the same source that will receive the
/// write keeps the two halves talking about the same bytes.
async fn ensure_unchanged_source(
    ws: &Workspace,
    client: &Option<ClientFs>,
    rel: &str,
) -> Result<(), AgentError> {
    if let Some(client) = client {
        let abs = ws
            .resolve(rel)
            .map_err(|e| AgentError::Tool(e.to_string()))?;
        if let Some(text) = client.read(&abs.display().to_string()).await {
            return ws
                .ensure_matches(rel, text.as_bytes())
                .map_err(|e| AgentError::Tool(format!("{e:#}")));
        }
    }
    ws.ensure_unchanged(rel)
        .await
        .map_err(|e| AgentError::Tool(format!("{e:#}")))
}

/// Write a file, preferring the editor so its buffer stays authoritative.
async fn write_source(
    ws: &Workspace,
    client: &Option<ClientFs>,
    rel: &str,
    content: &str,
) -> Result<(), AgentError> {
    write_source_inner(ws, client, rel, content).await
}

async fn write_source_inner(
    ws: &Workspace,
    client: &Option<ClientFs>,
    rel: &str,
    content: &str,
) -> Result<(), AgentError> {
    let abs = ws
        .resolve(rel)
        .map_err(|e| AgentError::Tool(e.to_string()))?;
    if let Some(client) = client {
        if client.write(&abs.display().to_string(), content).await {
            return Ok(());
        }
    }
    ws.write(rel, content)
        .await
        .map_err(|e| AgentError::Tool(format!("{e:#}")))
}

fn arg_str(args: &Value, key: &str) -> Result<String, AgentError> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| AgentError::Tool(format!("missing required argument '{key}'")))
}

/// Render a unified diff for display alongside the structured old/new text.
fn unified_diff(path: &str, old: &str, new: &str) -> String {
    use similar::TextDiff;
    TextDiff::from_lines(old, new)
        .unified_diff()
        .header(path, path)
        .to_string()
}

// ── read_file ─────────────────────────────────────────────────────────────────

pub struct ReadFile {
    pub ws: Arc<Workspace>,
    /// When present, file I/O is routed through the editor.
    pub client: Option<ClientFs>,
}

#[async_trait]
impl Tool for ReadFile {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "read_file",
            "Read a file from the workspace. Returns numbered lines. Always read a \
             file before editing it.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Workspace-relative path" },
                    "offset": { "type": "integer", "description": "1-based first line" },
                    "limit": { "type": "integer", "description": "Max lines to return" }
                },
                "required": ["path"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let path = arg_str(&args, "path")?;
        let abs = self
            .ws
            .resolve(&path)
            .map_err(|e| AgentError::Tool(e.to_string()))?;
        let text = read_source(&self.ws, &self.client, &path).await?;
        let offset = args
            .get("offset")
            .and_then(|v| v.as_u64())
            .unwrap_or(1)
            .max(1) as usize;
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(2000) as usize;

        let total = text.lines().count();
        let numbered: String = text
            .lines()
            .enumerate()
            .skip(offset - 1)
            .take(limit)
            .map(|(i, l)| format!("{:>6}\t{}\n", i + 1, l))
            .collect();

        // The cap belongs on what is *returned*, not on the file's size. It
        // used to be checked before the range was applied, and the error told
        // you to use a range — advice the check itself made impossible to
        // follow, so a large file could not be read at all.
        if numbered.len() > MAX_READ_BYTES {
            return Err(AgentError::Tool(format!(
                "lines {}– of {path} come to {} bytes, over the {MAX_READ_BYTES} limit. \
                 Ask for fewer lines with `limit`, or search with grep.",
                offset,
                numbered.len()
            )));
        }

        Ok(json!({
            "path": path,
            "total_lines": total,
            "content": numbered,
            "_locations": [{ "path": abs.display().to_string(), "line": offset }],
        }))
    }
}

// ── write_file ────────────────────────────────────────────────────────────────

pub struct WriteFile {
    pub ws: Arc<Workspace>,
    /// When present, file I/O is routed through the editor.
    pub client: Option<ClientFs>,
    pub lsp: Arc<LspPool>,
}

#[async_trait]
impl Tool for WriteFile {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "write_file",
            "Create a new file or completely replace an existing one. For targeted \
             changes to existing files prefer edit_file.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let path = arg_str(&args, "path")?;
        let content = arg_str(&args, "content")?;
        let abs = self
            .ws
            .resolve(&path)
            .map_err(|e| AgentError::Tool(e.to_string()))?;
        // Held across both, so a concurrent edit_file on the same path
        // cannot read the old text, be overwritten here, and then write its
        // own result back over this one.
        let _guard = self.ws.lock_path(&path).await;
        // Checked before the read below, which would otherwise re-observe the
        // change and report everything as fine. This replaces the whole file
        // without looking at what was in it; `edit_file` matches on
        // surrounding text instead, a finer check that does not need this.
        ensure_unchanged_source(&self.ws, &self.client, &path).await?;
        let old = read_source(&self.ws, &self.client, &path)
            .await
            .unwrap_or_default();
        write_source(&self.ws, &self.client, &path, &content).await?;
        self.lsp.notify_changed(&abs).await;

        Ok(json!({
            "path": path,
            "bytes": content.len(),
            "created": old.is_empty(),
            "_diff": { "path": abs.display().to_string(), "old_text": old, "new_text": content },
        }))
    }
}

// ── edit_file / multi_edit ────────────────────────────────────────────────────

pub struct EditFile {
    pub ws: Arc<Workspace>,
    /// When present, file I/O is routed through the editor.
    pub client: Option<ClientFs>,
    pub lsp: Arc<LspPool>,
}

/// Apply one exact-string replacement, enforcing uniqueness.
fn apply_replacement(
    content: &str,
    old: &str,
    new: &str,
    replace_all: bool,
) -> Result<String, AgentError> {
    if old.is_empty() {
        return Err(AgentError::Tool("old_string must not be empty".into()));
    }
    let count = content.matches(old).count();
    match count {
        0 => Err(AgentError::Tool(
            "old_string not found — read the file and copy the exact text, including \
             indentation"
                .into(),
        )),
        _ if count > 1 && !replace_all => Err(AgentError::Tool(format!(
            "old_string appears {count} times; add more surrounding context to make it \
             unique, or set replace_all"
        ))),
        _ => Ok(if replace_all {
            content.replace(old, new)
        } else {
            content.replacen(old, new, 1)
        }),
    }
}

#[async_trait]
impl Tool for EditFile {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "edit_file",
            "Replace an exact string in a file. The old_string must match the file \
             byte-for-byte (including indentation) and be unique unless replace_all is set.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "old_string": { "type": "string" },
                    "new_string": { "type": "string" },
                    "replace_all": { "type": "boolean" }
                },
                "required": ["path", "old_string", "new_string"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let path = arg_str(&args, "path")?;
        let old_string = arg_str(&args, "old_string")?;
        let new_string = arg_str(&args, "new_string")?;
        let replace_all = args
            .get("replace_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let abs = self
            .ws
            .resolve(&path)
            .map_err(|e| AgentError::Tool(e.to_string()))?;
        // Read, modify and write are one operation. Without this the four
        // tools the ReAct loop runs concurrently can each read the same
        // original and the last write wins, discarding the rest in silence.
        let _guard = self.ws.lock_path(&path).await;
        let original = read_source(&self.ws, &self.client, &path).await?;
        let updated = apply_replacement(&original, &old_string, &new_string, replace_all)?;
        write_source(&self.ws, &self.client, &path, &updated).await?;
        self.lsp.notify_changed(&abs).await;

        let abs_str = abs.display().to_string();
        Ok(json!({
            "path": path,
            "diff": unified_diff(&path, &original, &updated),
            "_diff": { "path": abs_str, "old_text": original, "new_text": updated },
            "_locations": [{ "path": abs_str }],
        }))
    }
}

pub struct MultiEdit {
    pub ws: Arc<Workspace>,
    /// When present, file I/O is routed through the editor.
    pub client: Option<ClientFs>,
    pub lsp: Arc<LspPool>,
}

#[async_trait]
impl Tool for MultiEdit {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "multi_edit",
            "Apply several exact-string replacements to one file atomically, in order. \
             If any replacement fails, the file is left untouched.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_string": { "type": "string" },
                                "new_string": { "type": "string" },
                                "replace_all": { "type": "boolean" }
                            },
                            "required": ["old_string", "new_string"]
                        }
                    }
                },
                "required": ["path", "edits"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let path = arg_str(&args, "path")?;
        let edits = args
            .get("edits")
            .and_then(|v| v.as_array())
            .ok_or_else(|| AgentError::Tool("edits must be an array".into()))?
            .clone();

        let abs = self
            .ws
            .resolve(&path)
            .map_err(|e| AgentError::Tool(e.to_string()))?;
        let _guard = self.ws.lock_path(&path).await;
        let original = read_source(&self.ws, &self.client, &path).await?;

        // Apply to a scratch copy so a late failure cannot leave a half-edit.
        let mut updated = original.clone();
        for (i, edit) in edits.iter().enumerate() {
            let old = edit
                .get("old_string")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let new = edit
                .get("new_string")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let all = edit
                .get("replace_all")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            updated = apply_replacement(&updated, old, new, all)
                .map_err(|e| AgentError::Tool(format!("edit {}: {e}", i + 1)))?;
        }

        write_source(&self.ws, &self.client, &path, &updated).await?;
        self.lsp.notify_changed(&abs).await;

        let abs_str = abs.display().to_string();
        Ok(json!({
            "path": path,
            "edits_applied": edits.len(),
            "diff": unified_diff(&path, &original, &updated),
            "_diff": { "path": abs_str, "old_text": original, "new_text": updated },
            "_locations": [{ "path": abs_str }],
        }))
    }
}

// ── glob / grep / list_directory ──────────────────────────────────────────────

pub struct Glob {
    pub ws: Arc<Workspace>,
}

#[async_trait]
impl Tool for Glob {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "glob",
            "Find files by path pattern (e.g. 'src/**/*.rs'). Respects .gitignore.",
            json!({
                "type": "object",
                "properties": { "pattern": { "type": "string" } },
                "required": ["pattern"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let pattern = arg_str(&args, "pattern")?;
        // Compiled here rather than inside the walk. It used to fall back to
        // a substring match when the pattern would not parse, which is a
        // different search wearing the same name — the caller gets results
        // for a question they did not ask and no indication of it.
        let matcher = glob::Pattern::new(&pattern)
            .map_err(|e| AgentError::Tool(format!("'{pattern}' is not a valid glob: {e}")))?;
        let root = self.ws.root().to_path_buf();
        let matches = tokio::task::spawn_blocking(move || {
            let matcher = Some(matcher);
            let mut found: Vec<String> = Vec::new();
            for entry in ignore::Walk::new(&root).flatten() {
                if !entry.file_type().is_some_and(|t| t.is_file()) {
                    continue;
                }
                let rel = entry
                    .path()
                    .strip_prefix(&root)
                    .unwrap_or(entry.path())
                    .to_string_lossy()
                    .to_string();
                let hit = match &matcher {
                    Some(m) => m.matches(&rel),
                    None => rel.contains(pattern.as_str()),
                };
                if hit {
                    found.push(rel);
                    if found.len() >= MAX_MATCHES {
                        break;
                    }
                }
            }
            found.sort();
            found
        })
        .await
        .map_err(|e| AgentError::Tool(e.to_string()))?;

        Ok(json!({ "count": matches.len(), "files": matches }))
    }
}

pub struct Grep {
    pub ws: Arc<Workspace>,
}

#[async_trait]
impl Tool for Grep {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "grep",
            "Search file contents with a regular expression. Respects .gitignore. Use \
             lsp_references instead when you want real symbol usages.",
            json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Rust regex" },
                    "path": { "type": "string", "description": "Subdirectory to search" },
                    "glob": { "type": "string", "description": "Only files matching this pattern" }
                },
                "required": ["pattern"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let pattern = arg_str(&args, "pattern")?;
        let sub = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let file_glob = args
            .get("glob")
            .and_then(|v| v.as_str())
            .map(glob::Pattern::new)
            .transpose()
            .map_err(|e| AgentError::Tool(format!("invalid 'glob' filter: {e}")))?;

        // Checked through the workspace handle: `Walk::new` follows a
        // symlinked root, so a search rooted at one would walk outside.
        let root = self
            .ws
            .confined_dir(sub)
            .await
            .map_err(|e| AgentError::Tool(format!("{e:#}")))?;
        let ws_root = self.ws.root().to_path_buf();

        let hits = tokio::task::spawn_blocking(move || -> Result<Vec<Value>, String> {
            let re = regex::Regex::new(&pattern).map_err(|e| e.to_string())?;
            let mut hits = Vec::new();
            'files: for entry in ignore::Walk::new(&root).flatten() {
                if !entry.file_type().is_some_and(|t| t.is_file()) {
                    continue;
                }
                let rel = entry
                    .path()
                    .strip_prefix(&ws_root)
                    .unwrap_or(entry.path())
                    .to_string_lossy()
                    .to_string();
                if let Some(g) = &file_glob {
                    if !g.matches(&rel) {
                        continue;
                    }
                }
                let Ok(content) = std::fs::read_to_string(entry.path()) else {
                    continue; // binary or unreadable
                };
                for (i, line) in content.lines().enumerate() {
                    if re.is_match(line) {
                        hits.push(json!({
                            "path": rel,
                            "line": i + 1,
                            "text": line.chars().take(300).collect::<String>(),
                        }));
                        if hits.len() >= MAX_MATCHES {
                            break 'files;
                        }
                    }
                }
            }
            Ok(hits)
        })
        .await
        .map_err(|e| AgentError::Tool(e.to_string()))?
        .map_err(AgentError::Tool)?;

        Ok(json!({ "count": hits.len(), "matches": hits }))
    }
}

pub struct ListDirectory {
    pub ws: Arc<Workspace>,
}

#[async_trait]
impl Tool for ListDirectory {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "list_directory",
            "List the entries of a directory in the workspace.",
            json!({
                "type": "object",
                "properties": { "path": { "type": "string" } }
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let rel = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let out: Vec<Value> = self
            .ws
            .read_dir(rel)
            .await
            .map_err(|e| AgentError::Tool(format!("{e:#}")))?
            .into_iter()
            .map(|(name, is_dir)| {
                json!({ "name": name, "type": if is_dir { "dir" } else { "file" } })
            })
            .collect();
        Ok(json!({ "path": rel, "entries": out }))
    }
}

// ── jobs ──────────────────────────────────────────────────────────────────────

/// Inspect and stop background commands.
pub struct Jobs {
    pub jobs: Arc<BackgroundJobs>,
}

#[async_trait]
impl Tool for Jobs {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "jobs",
            "List the background commands you started, with whether each is still \
             running, and stop one by pid. Anything still running is killed when the \
             session ends.",
            json!({
                "type": "object",
                "properties": {
                    "stop": {
                        "type": "integer",
                        "description": "Pid to stop; omit to just list"
                    }
                }
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        if let Some(pid) = args.get("stop").and_then(|v| v.as_i64()) {
            let stopped = self.jobs.stop_one(pid as i32).await;
            if !stopped {
                return Err(AgentError::Tool(format!(
                    "no background job with pid {pid} — call jobs with no arguments to \
                     see what is running"
                )));
            }
        }
        Ok(json!({ "jobs": self.jobs.list().await }))
    }
}

// ── bash ──────────────────────────────────────────────────────────────────────

/// Long-running commands the agent started and has not collected.
///
/// They used to be a pid and a log path, recorded and then forgotten: nothing
/// could stop one, nothing reported whether it was still alive, and when the
/// session ended they carried on running against the user's repository with
/// no owner. A job is a process *group* now, so stopping one stops what it
/// spawned, and the whole set is stopped when this is dropped.
#[derive(Default)]
pub struct BackgroundJobs {
    jobs: Mutex<Vec<BackgroundJob>>,
}

pub struct BackgroundJob {
    pub pid: i32,
    pub log: PathBuf,
    pub command: String,
    pub started: std::time::Instant,
    /// Kept so the process can be *waited on*, not merely signalled.
    ///
    /// The tool used to drop this and keep only the number. See
    /// [`BackgroundJobs::running`] for what that cost.
    child: Option<tokio::process::Child>,
}

impl BackgroundJobs {
    /// Is the process still running? Reaps it if it is not.
    ///
    /// `kill(pid, 0)` cannot answer this. A process that has exited but has
    /// not been waited on is a zombie, and a zombie still has a pid that
    /// accepts signal 0 — so with only a number to work from, a job reported
    /// itself as running until something else happened to reap it. That
    /// something was tokio's orphan queue, which collects a dropped `Child` on
    /// the next `SIGCHLD`: within 100ms on a developer machine, and *not*
    /// within the two seconds `stop` waits, on CI. The test that caught it
    /// stopped a `sleep 60` and was told it was still running.
    ///
    /// `try_wait` is the same question asked of `waitpid`, which both answers
    /// it and clears the zombie. It needs the handle, which is why the job
    /// keeps one.
    fn running(job: &mut BackgroundJob) -> bool {
        let Some(child) = job.child.as_mut() else {
            return false; // already reaped, and remembered as such
        };
        match child.try_wait() {
            Ok(None) => true,
            Ok(Some(_)) => {
                job.child = None;
                false
            }
            // Nothing more can be learned from this handle. Reporting it
            // finished is the safer error: the alternative is a job that
            // claims to be running for the rest of the session.
            Err(_) => {
                job.child = None;
                false
            }
        }
    }

    /// Stop a job and everything it started.
    ///
    /// Staged: `SIGTERM` first so a dev server can close its sockets and a
    /// test runner can print what it was doing, then `SIGKILL` for whatever
    /// ignored it. Signalled to the process *group*, so a job's children go
    /// with it, and to the process as well in case it never became a group
    /// leader — `setsid` is called in the child, and a `stop` that quietly did
    /// nothing because it failed would be worse than a redundant signal.
    ///
    /// Returns only once the process has actually been collected, or the wait
    /// timed out. `stop` returning while `running` still says yes is the bug
    /// this replaced.
    #[cfg(unix)]
    async fn stop(job: &mut BackgroundJob) {
        unsafe {
            libc::killpg(job.pid, libc::SIGTERM);
            libc::kill(job.pid, libc::SIGTERM);
        }
        for _ in 0..20 {
            if !Self::running(job) {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        unsafe {
            libc::killpg(job.pid, libc::SIGKILL);
            libc::kill(job.pid, libc::SIGKILL);
        }
        if let Some(child) = job.child.as_mut() {
            // Bounded: an unkillable process is a worse thing to wait forever
            // for than to report honestly as still running.
            let _ = tokio::time::timeout(std::time::Duration::from_secs(2), child.wait()).await;
        }
        let _ = Self::running(job);
    }

    /// As above, where there are no signals: the handle can still kill it,
    /// which is more than the previous version of this did — it was empty, so
    /// asking to stop a job on Windows silently did nothing at all.
    #[cfg(not(unix))]
    async fn stop(job: &mut BackgroundJob) {
        if let Some(child) = job.child.as_mut() {
            let _ = child.kill().await;
        }
        let _ = Self::running(job);
    }

    pub async fn record(&self, job: BackgroundJob) {
        self.jobs.lock().await.push(job);
    }

    /// Every job, with whether it is still running.
    pub async fn list(&self) -> Vec<Value> {
        self.jobs
            .lock()
            .await
            .iter_mut()
            .map(|job| {
                json!({
                    "pid": job.pid,
                    "command": job.command,
                    "running": Self::running(job),
                    "seconds": job.started.elapsed().as_secs(),
                    "log_file": job.log.display().to_string(),
                })
            })
            .collect()
    }

    pub async fn stop_one(&self, pid: i32) -> bool {
        // The lock is held across the stop so a `jobs` listing cannot observe
        // the job half-stopped, and so two stops cannot race on one handle.
        let mut jobs = self.jobs.lock().await;
        match jobs.iter_mut().find(|job| job.pid == pid) {
            Some(job) => {
                Self::stop(job).await;
                true
            }
            None => false,
        }
    }
}

impl Drop for BackgroundJobs {
    /// A watcher the agent started must not outlive the session that started
    /// it.
    ///
    /// Best-effort by nature: `Drop` cannot await, so this signals the
    /// process groups and returns without reaping them. What it no longer
    /// does is skip silently — it used to `try_lock`, which fails while
    /// anything else holds the mutex and then quietly left the jobs running.
    /// `get_mut` needs no lock at all: `Drop` has `&mut self`, so by
    /// definition nobody else holds a reference.
    fn drop(&mut self) {
        for job in self.jobs.get_mut().iter_mut() {
            // `try_wait`, not `kill(pid, 0)`: it tells the truth about a
            // process that has already exited, and skipping those is what
            // keeps this from signalling a pid the OS has since handed to
            // somebody else.
            if !Self::running(job) {
                continue;
            }
            // The group, not just the process: a dev server's children would
            // otherwise survive the shell that started them.
            #[cfg(unix)]
            unsafe {
                libc::killpg(job.pid, libc::SIGKILL);
            }
            #[cfg(not(unix))]
            if let Some(child) = job.child.as_mut() {
                let _ = child.start_kill();
            }
        }
    }
}

/// How a shell command is contained.
///
/// A permission prompt is not a boundary. Nobody can reliably audit a
/// generated shell pipeline, repository instructions can talk a model into
/// proposing one, and Yolo mode removes the prompt entirely. So containment
/// has to hold whether or not the command was approved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellContainment {
    /// Scrub credentials, run in its own process group, apply resource
    /// limits, and confine **writes** to the workspace and the toolchain
    /// caches with Landlock where the kernel offers it.
    ///
    /// Reads are **not** confined and the network is open. That is a
    /// deliberate trade, and the same one Codex's Linux sandbox makes:
    /// toolchains install themselves in unpredictable places under `$HOME`,
    /// and a read policy that guesses wrong makes `node` or `cargo` vanish.
    /// It stops a command *changing* anything outside the repository; it does
    /// not stop one reading `~/.ssh` and sending it somewhere. Use `Strict`
    /// when that matters.
    Confined,
    /// As `Confined`, and additionally: reads narrowed to the system, the
    /// named toolchain locations and the workspace; the network refused at
    /// the syscall; and a refusal to run at all if the kernel cannot enforce
    /// any of it.
    ///
    /// For a repository you do not trust, where running a command unconfined
    /// is worse than not running it. It is allowed to break an exotic
    /// toolchain, and a build that needs to fetch dependencies will fail —
    /// that is the mode working, not the mode broken.
    Strict,
    /// Run inside a throwaway container: **no network**, no host filesystem
    /// beyond the workspace, capabilities dropped, memory and pid capped.
    ///
    /// The only option here that is a boundary rather than a seatbelt.
    /// Landlock cannot take the network away and cannot stop a command
    /// reading anything the user can read outside the paths it was told
    /// about; a container can do both.
    ///
    /// The cost is the toolchain. A container only has what its image has, so
    /// `cargo test` inside a bare `ubuntu` image fails for want of cargo
    /// rather than for anything to do with the code. Point `container_image`
    /// at something that carries the project's toolchain when you need
    /// builds to work inside it.
    Container,
    /// The host, as it was: a **login** shell with the user's full
    /// environment, so `~/.profile` and friends are sourced and whatever they
    /// export — credentials included — is present.
    ///
    /// That is the point of it rather than an oversight. `Confined` uses a
    /// non-login shell precisely to stop a profile re-importing the
    /// credentials it just scrubbed; `Host` is the escape hatch for when a
    /// toolchain genuinely needs the user's real environment, and it is named
    /// so that choosing it is a visible decision.
    Host,
}

/// Environment variables a shell command has no business inheriting.
///
/// A command runs at the model's suggestion in a repository that may be
/// hostile, and the parent process holds the credential for the very model
/// that proposed it. Matching on shape rather than an allow-list of known
/// names, because the next provider's variable is not in any list.
pub fn is_credential(name: &str) -> bool {
    const MARKERS: [&str; 7] = [
        "KEY",
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "CREDENTIAL",
        "AUTH",
        "SESSION",
    ];
    let upper = name.to_ascii_uppercase();
    MARKERS.iter().any(|m| upper.contains(m))
        // Not a credential, but it names where to send them.
        || upper.starts_with("ANTHROPIC_")
        || upper.starts_with("OPENAI_")
        || upper.starts_with("QWEN_")
        || upper == "AWS_PROFILE"
}

/// The environment a confined command gets: the parent's, minus secrets.
///
/// Subtractive rather than an allow-list, because a shell needs far more than
/// anyone can enumerate — `PATH`, `HOME`, `CARGO_HOME`, locale, proxies — and
/// a build that fails for want of an unlisted variable would push people
/// straight back to the unconfined mode.
/// The environment a confined command gets: this process's, minus secrets.
///
/// Belt and braces since [`secrets::capture_and_scrub`](crate::secrets) —
/// after startup there is no credential left in the environment for this to
/// filter. It stays because the filter is the guarantee and the scrub is an
/// optimisation of it: the library is usable without `main` ever running, and
/// a tool that leaked the key when embedded would be a strange thing to ship.
pub fn scrubbed_env() -> Vec<(String, String)> {
    std::env::vars()
        .filter(|(name, _)| !is_credential(name))
        .collect()
}

pub struct Bash {
    pub ws: Arc<Workspace>,
    pub jobs: Arc<BackgroundJobs>,
    pub containment: ShellContainment,
    /// Image for [`ShellContainment::Container`]. Ignored otherwise.
    pub container_image: String,
}

/// The image a container-confined shell runs in when nothing else is chosen.
///
/// Plain Ubuntu: a shell, coreutils, and nothing else. It is the right
/// default because it is the honest one — a container's whole value is that
/// the command cannot reach anything you did not put in it, and an image
/// chosen to make builds convenient is an image full of things you did not
/// think about. Override it when you need the toolchain inside.
pub const DEFAULT_CONTAINER_IMAGE: &str = "ubuntu:24.04";

impl ShellContainment {
    /// Parse the name a user types.
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "host" => Some(Self::Host),
            "confined" => Some(Self::Confined),
            "strict" => Some(Self::Strict),
            "container" => Some(Self::Container),
            _ => None,
        }
    }

    /// What each one is, for `--help` and for the error when it is misspelt.
    pub const NAMES: &'static str = "host | confined | strict | container";

    /// The Landlock and seccomp policy this mode asks for.
    ///
    /// One place, so a tool cannot accidentally disagree with another about
    /// what `Strict` means — which is how `verify` came to run with the
    /// network open in a mode whose entire purpose is to deny it.
    fn policy(self) -> crate::shell_sandbox::Policy {
        match self {
            ShellContainment::Strict => crate::shell_sandbox::Policy::strict(),
            _ => crate::shell_sandbox::Policy::permissive(),
        }
    }

    /// Build a confined command for a trusted helper program, or `None` to
    /// run it plainly.
    ///
    /// For `git`: a known binary, run with its hooks disabled, that still
    /// needs the session's containment because a tool that starts a process
    /// and ignores the policy is a hole whatever the process is. Errors when
    /// the session demands confinement the kernel cannot give.
    ///
    /// `Container` falls back to the strongest *host* policy rather than a
    /// container. Running git inside the image would need git in the image
    /// and the repository's `.git` mounted into it, and a bare `ubuntu` has
    /// neither — the honest options were the host sandbox or no git at all.
    pub(crate) fn confined_command(
        self,
        program: &str,
        args: &[String],
        root: &Path,
    ) -> Result<Option<tokio::process::Command>, AgentError> {
        if self == ShellContainment::Host {
            return Ok(None);
        }
        let policy = match self {
            ShellContainment::Strict => crate::shell_sandbox::Policy::strict(),
            _ => crate::shell_sandbox::Policy::permissive(),
        };
        match crate::shell_sandbox::confined_argv(root, policy, program, args) {
            Some(helper) => Ok(Some(helper.into())),
            None if self == ShellContainment::Strict => Err(AgentError::Tool(format!(
                "this session requires filesystem confinement and it is unavailable \
                 (no Landlock, or no sandbox helper alongside this binary); refusing to \
                 run `{program}`"
            ))),
            None => Ok(None),
        }
    }

    /// How this mode should be described in a tool result.
    ///
    /// Stated rather than assumed, and stated accurately: a model told its
    /// filesystem is confined when it is not will reason from that.
    fn describe(self, confined: bool) -> String {
        match self {
            ShellContainment::Host => "none — running with full host access".into(),
            ShellContainment::Container => {
                "container — no network, no host filesystem beyond the workspace, \
                 capabilities dropped"
                    .into()
            }
            _ if !confined => "credentials scrubbed, own process group, memory/cpu/file-size \
                 limits — the kernel offers no Landlock, so the filesystem is NOT \
                 confined and there is no network isolation"
                .into(),
            ShellContainment::Confined => "credentials scrubbed, own process group, \
                 memory/cpu/file-size limits, writes confined to the workspace. Reads are \
                 NOT confined and the network is open."
                .into(),
            ShellContainment::Strict => "credentials scrubbed, own process group, \
                 memory/cpu/file-size limits, writes confined to the workspace, reads \
                 confined to the workspace and the system toolchain, network refused."
                .into(),
        }
    }
}

/// Run one command inside a throwaway container.
///
/// A fresh container per command, which is the executor's own behaviour and
/// the right default here: it means one command cannot leave anything behind
/// for the next, and the only state that survives is what was written to the
/// workspace — which is the thing the user actually wants to keep.
///
/// Free-standing rather than a method, because both `bash` and `verify` need
/// it. `verify` reaching the host sandbox while the session asked for a
/// container was the sharper half of the bug: it runs `cargo test` and `npm
/// run`, so it executes `build.rs`, lifecycle scripts and test bodies — all
/// repository-controlled — and it needs no approval in any mode.
async fn container_exec(
    image: &str,
    root: &Path,
    program: &str,
    args: Vec<String>,
    timeout_secs: u64,
) -> Result<eventage::sandbox::SandboxOutput, AgentError> {
    use eventage::sandbox::{DockerExecutor, SandboxExecutor, SandboxRequest};

    DockerExecutor::new(image)
        .execute(SandboxRequest {
            program: program.into(),
            args,
            // Nothing from the host: the executor starts from an empty
            // environment, and there is no profile to re-import from.
            env: Default::default(),
            stdin: None,
            timeout_ms: timeout_secs * 1_000,
            working_dir: root.to_path_buf(),
            readable_paths: vec![],
            writable_paths: vec![root.to_path_buf()],
        })
        .await
        .map_err(|e| {
            AgentError::Tool(format!(
                "could not run the command in a container ({image}): {e}. Docker must be \
                 running and the image present — `docker pull {image}` — or switch this \
                 session away from container containment."
            ))
        })
}

impl Bash {
    /// Run the command inside a throwaway container.
    async fn run_in_container(
        &self,
        script: &str,
        timeout_secs: u64,
        background: bool,
    ) -> Result<Value, AgentError> {
        if background {
            return Err(AgentError::Tool(
                "background commands are not available inside a container: the container \
                 is torn down when the command ends, so nothing would survive to watch"
                    .into(),
            ));
        }

        let output = container_exec(
            &self.container_image,
            self.ws.root(),
            "bash",
            vec!["-c".into(), script.to_string()],
            timeout_secs,
        )
        .await?;

        Ok(json!({
            "exit_code": output.exit_code,
            "stdout": output.stdout,
            "stderr": output.stderr,
            "timed_out": output.timed_out,
            "containment": format!(
                "container ({}) — {}",
                self.container_image,
                ShellContainment::Container.describe(true),
            ),
        }))
    }

    /// Build the command, contained according to the policy.
    fn command(
        &self,
        script: &str,
        background: bool,
    ) -> Result<tokio::process::Command, AgentError> {
        if self.containment == ShellContainment::Host {
            let mut cmd = tokio::process::Command::new("bash");
            cmd.arg("-lc")
                .arg(script)
                .current_dir(self.ws.root())
                .kill_on_drop(!background);
            return Ok(cmd);
        }

        // Re-executed through the sandbox helper where the kernel supports
        // it, so the ruleset is built in a fresh single-threaded process
        // rather than in the forked child — see `shell_sandbox`.
        let confined = crate::shell_sandbox::confined_command(
            self.ws.root(),
            script,
            self.containment.policy(),
        );
        if confined.is_none() && self.containment == ShellContainment::Strict {
            return Err(AgentError::Tool(
                "this session requires filesystem confinement and the kernel does not \
                 provide it (Landlock unavailable); refusing to run the command"
                    .into(),
            ));
        }
        // Whether the trampoline is doing the work decides where `setsid` and
        // the resource limits are applied — see below.
        let via_helper = confined.is_some();
        let mut cmd: tokio::process::Command = match confined {
            Some(helper) => helper.into(),
            None => {
                // `-c`, not `-lc`: a login shell sources the user's profile,
                // which re-imports exactly the credentials just scrubbed.
                let mut plain = std::process::Command::new("bash");
                plain.arg("-c").arg(script);
                plain.into()
            }
        };
        cmd.current_dir(self.ws.root())
            .kill_on_drop(!background)
            .env_clear()
            .envs(scrubbed_env());

        // Nothing at all runs between fork and exec when the trampoline is
        // used: it is single-threaded by construction, so it does `setsid`
        // and the limits itself, before it execs.
        //
        // This is the same reasoning that moved the Landlock ruleset out of
        // `pre_exec`, applied to what was left behind. `setsid` and
        // `setrlimit` are bare syscalls and *should* be safe in a forked
        // child — but a hung child with one thread parked on a futex says
        // otherwise, and the fork-in-a-threaded-process hazard is not worth
        // arguing with when a process that cannot have it is already there.
        //
        // The unconfined path keeps `pre_exec`, because it has no trampoline
        // to delegate to.
        #[cfg(unix)]
        if !via_helper {
            unsafe {
                cmd.pre_exec(move || {
                    // Its own process group, so a timeout or a cancellation
                    // kills the whole tree rather than the shell alone.
                    libc::setsid();
                    apply_resource_limits();
                    Ok(())
                });
            }
        }
        Ok(cmd)
    }
}

/// Bound what one command can consume.
///
/// A wall-clock timeout stops a command that hangs; it does nothing about one
/// that allocates until the machine swaps, forks until the process table is
/// full, or writes until the disk is. These are the cheap kernel-enforced
/// limits — generous enough that a real build or test run never notices, low
/// enough that a runaway is stopped by the kernel rather than by the user
/// rebooting.
///
/// Deliberately no `RLIMIT_NPROC`. It looks like the fork-bomb defence and is
/// not one: the limit counts processes for the whole *user*, not this command,
/// so a value low enough to matter makes the shell fail to fork whenever the
/// user happens to be busy — which showed up here as commands hanging under a
/// parallel test run. Containing a fork bomb needs a cgroup or a pid
/// namespace, which means a container.
///
/// Not a substitute for one either way: there is still no network isolation
/// and no accounting across the process tree.
#[cfg(unix)]
pub fn apply_resource_limits() {
    /// Address space. A large Rust build links with a lot of memory.
    const MAX_MEMORY_BYTES: u64 = 8 * 1024 * 1024 * 1024;
    /// A single file. Enough for any artifact, not enough to fill a disk.
    const MAX_FILE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
    /// CPU seconds, as a backstop for a spin loop that produces no output and
    /// so never trips an idle timeout.
    const MAX_CPU_SECONDS: u64 = 3600;

    // The first argument to `setrlimit` is spelled differently per platform:
    // glibc has its own `__rlimit_resource_t`, the BSDs (macOS included) take
    // a plain `c_int`. Naming the Linux type unconditionally compiled fine on
    // the machine it was written on and failed the macOS CI job.
    #[cfg(target_os = "linux")]
    type RlimitResource = libc::__rlimit_resource_t;
    #[cfg(not(target_os = "linux"))]
    type RlimitResource = libc::c_int;

    let set = |resource: RlimitResource, value: u64| {
        let limit = libc::rlimit {
            rlim_cur: value,
            rlim_max: value,
        };
        unsafe { libc::setrlimit(resource, &limit) };
    };
    // On macOS `RLIMIT_AS` shares its value with `RLIMIT_RSS` and is advisory
    // rather than enforced, so the memory cap is a Linux guarantee and a
    // best-effort elsewhere. Setting it costs nothing either way.
    set(libc::RLIMIT_AS, MAX_MEMORY_BYTES);
    set(libc::RLIMIT_FSIZE, MAX_FILE_BYTES);
    set(libc::RLIMIT_CPU, MAX_CPU_SECONDS);
    // No core dumps: they are large, and they contain whatever was in memory.
    set(libc::RLIMIT_CORE, 0);
}

#[cfg(not(unix))]
pub fn apply_resource_limits() {}

// Filesystem confinement of the shell is **not** applied here, and the reason
// is worth recording so nobody puts it back the same way.
//
// The obvious place for it is `pre_exec`, next to `setsid`. It was there, and
// it deadlocked: `pre_exec` runs in the child after `fork`, where only
// async-signal-safe work is allowed, and building a Landlock ruleset
// allocates and opens files. If any other thread held the allocator lock at
// the moment of the fork, the child blocks on it forever. Standalone the tool
// looked fine; under a parallel test run `bash -c true` hung. `setsid` and
// `setrlimit` are bare syscalls that allocate nothing, which is why they are
// safe to keep.
//
// Doing it properly means building the ruleset in the *parent* — all the
// allocation and the file opens — and calling only `restrict_self` in the
// child. That is the fix; it is not this one, and shipping a shell tool that
// intermittently deadlocks would have been much worse than shipping one
// without a seatbelt. `DockerExecutor` remains the answer for a repository
// that is actually untrusted.

#[async_trait]
impl Tool for Bash {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "bash",
            "Run a shell command in the workspace. Use background:true for long-running \
             processes (dev servers, watchers) so you stay responsive.",
            json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "timeout_secs": { "type": "integer" },
                    "background": { "type": "boolean" }
                },
                "required": ["command"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let command = arg_str(&args, "command")?;
        let background = args
            .get("background")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let timeout_secs = args
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(120);

        if self.containment == ShellContainment::Container {
            return self
                .run_in_container(&command, timeout_secs, background)
                .await;
        }

        let mut cmd = self.command(&command, background)?;

        if background {
            let name = format!(".eventage-code-job-{}.log", uuid::Uuid::new_v4());
            let log = self.ws.root().join(&name);
            let file = std::fs::File::create(&log)
                .map_err(|e| AgentError::Tool(format!("cannot create job log: {e}")))?;
            let errfile = file
                .try_clone()
                .map_err(|e| AgentError::Tool(e.to_string()))?;
            cmd.stdout(file).stderr(errfile);
            let child = cmd
                .spawn()
                .map_err(|e| AgentError::Tool(format!("spawn failed: {e}")))?;
            let pid = child.id().unwrap_or_default() as i32;
            self.jobs
                .record(BackgroundJob {
                    pid,
                    log: log.clone(),
                    command: command.clone(),
                    started: std::time::Instant::now(),
                    // Handed over rather than dropped: it is the only thing
                    // that can wait on the process, and without it "is this
                    // job running?" has no honest answer once the process dies.
                    child: Some(child),
                })
                .await;
            return Ok(json!({
                "background": true,
                "pid": pid,
                "log_file": log.display().to_string(),
                "note": "running in background; read log_file for progress, and use \
                         `jobs` to check on it or stop it. It is killed when the \
                         session ends.",
            }));
        }

        let output =
            tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), cmd.output())
                .await
                .map_err(|_| {
                    AgentError::Tool(format!(
                        "command timed out after {timeout_secs}s; re-run with background:true \
                 or a longer timeout_secs"
                    ))
                })?
                .map_err(|e| AgentError::Tool(format!("command failed to start: {e}")))?;

        Ok(json!({
            "exit_code": output.status.code().unwrap_or(-1),
            "stdout": String::from_utf8_lossy(&output.stdout),
            "stderr": String::from_utf8_lossy(&output.stderr),
            // Stated in the result so the containment is visible rather than
            // assumed — the Landlock half is best-effort by kernel.
            "containment": self.containment.describe(crate::shell_sandbox::available()),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_requires_a_unique_match() {
        let content = "let a = 1;\nlet b = 1;\n";
        // Ambiguous: appears twice.
        let err = apply_replacement(content, "= 1;", "= 2;", false).unwrap_err();
        assert!(err.to_string().contains("appears 2 times"), "{err}");

        // replace_all resolves it.
        let all = apply_replacement(content, "= 1;", "= 2;", true).unwrap();
        assert_eq!(all, "let a = 2;\nlet b = 2;\n");

        // Unique context works.
        let one = apply_replacement(content, "let a = 1;", "let a = 9;", false).unwrap();
        assert_eq!(one, "let a = 9;\nlet b = 1;\n");
    }

    #[test]
    fn missing_text_explains_how_to_fix() {
        let err = apply_replacement("abc", "xyz", "1", false).unwrap_err();
        assert!(err.to_string().contains("not found"), "{err}");
        assert!(err.to_string().contains("exact text"), "{err}");
    }

    #[test]
    fn empty_old_string_rejected() {
        assert!(apply_replacement("abc", "", "x", false).is_err());
    }

    #[tokio::test]
    async fn edit_is_atomic_across_multiple_replacements() {
        let dir = tempfile::tempdir().unwrap();
        let ws = Arc::new(Workspace::open(dir.path()).unwrap());
        let file = dir.path().join("a.txt");
        std::fs::write(&file, "one\ntwo\n").unwrap();

        let tool = MultiEdit {
            ws: ws.clone(),
            client: None,
            lsp: Arc::new(LspPool::disabled(dir.path())),
        };
        // Second edit cannot match — the whole call must fail and leave the
        // file untouched.
        let err = tool
            .execute(json!({
                "path": "a.txt",
                "edits": [
                    { "old_string": "one", "new_string": "1" },
                    { "old_string": "MISSING", "new_string": "x" }
                ]
            }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("edit 2"), "{err}");
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "one\ntwo\n");
    }

    #[tokio::test]
    async fn read_file_numbers_lines_and_reports_total() {
        let dir = tempfile::tempdir().unwrap();
        let ws = Arc::new(Workspace::open(dir.path()).unwrap());
        std::fs::write(dir.path().join("f.txt"), "a\nb\nc\n").unwrap();

        let tool = ReadFile { ws, client: None };
        let out = tool.execute(json!({ "path": "f.txt" })).await.unwrap();
        assert_eq!(out["total_lines"], 3);
        assert!(out["content"].as_str().unwrap().contains("     1\ta"));

        let ranged = tool
            .execute(json!({ "path": "f.txt", "offset": 2, "limit": 1 }))
            .await
            .unwrap();
        let content = ranged["content"].as_str().unwrap();
        assert!(content.contains("     2\tb"));
        assert!(!content.contains("\tc"));
    }

    #[tokio::test]
    async fn file_io_falls_back_to_disk_without_a_client() {
        let dir = tempfile::tempdir().unwrap();
        let ws = Workspace::open(dir.path()).unwrap();
        std::fs::write(dir.path().join("f.txt"), "on disk").unwrap();

        assert_eq!(read_source(&ws, &None, "f.txt").await.unwrap(), "on disk");

        write_source(&ws, &None, "f.txt", "written").await.unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
            "written"
        );

        // Parent directories are created on demand.
        write_source(&ws, &None, "a/b/c.txt", "deep").await.unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a/b/c.txt")).unwrap(),
            "deep"
        );
    }

    #[tokio::test]
    async fn client_delegation_is_gated_on_advertised_capabilities() {
        use crate::acp::{ClientFs, Peer};
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("f.txt");
        std::fs::write(&file, "on disk").unwrap();

        // A client that advertised nothing must never be called: these
        // return without touching the peer, so we fall back to disk.
        let no_caps = ClientFs::new(Arc::new(Peer::new()), "s1".into(), Default::default());
        assert!(no_caps.read(&file.display().to_string()).await.is_none());
        assert!(!no_caps.write(&file.display().to_string(), "x").await);

        // …and read_source therefore still yields the on-disk content.
        let ws = Workspace::open(dir.path()).unwrap();
        let text = read_source(&ws, &Some(no_caps), "f.txt").await.unwrap();
        assert_eq!(text, "on disk");
    }

    #[tokio::test]
    async fn bash_reports_exit_code_and_output() {
        let dir = tempfile::tempdir().unwrap();
        let ws = Arc::new(Workspace::open(dir.path()).unwrap());
        let tool = Bash {
            ws,
            jobs: Arc::new(BackgroundJobs::default()),
            containment: ShellContainment::Confined,
            container_image: DEFAULT_CONTAINER_IMAGE.into(),
        };
        let out = tool
            .execute(json!({ "command": "echo hi && exit 3" }))
            .await
            .unwrap();
        assert_eq!(out["exit_code"], 3);
        assert!(out["stdout"].as_str().unwrap().contains("hi"));
    }
}
