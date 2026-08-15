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
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Maximum bytes returned by a single file read.
const MAX_READ_BYTES: usize = 400_000;
/// Maximum matches returned by one search.
const MAX_MATCHES: usize = 200;

/// Read a file, preferring the editor's (possibly unsaved) buffer.
async fn read_source(
    client: &Option<ClientFs>,
    abs: &std::path::Path,
) -> Result<String, AgentError> {
    if let Some(client) = client {
        if let Some(text) = client.read(&abs.display().to_string()).await {
            return Ok(text);
        }
    }
    tokio::fs::read_to_string(abs)
        .await
        .map_err(|e| AgentError::Tool(format!("cannot read {}: {e}", abs.display())))
}

/// Write a file, preferring the editor so its buffer stays authoritative.
async fn write_source(
    client: &Option<ClientFs>,
    abs: &std::path::Path,
    content: &str,
) -> Result<(), AgentError> {
    if let Some(client) = client {
        if client.write(&abs.display().to_string(), content).await {
            return Ok(());
        }
    }
    if let Some(parent) = abs.parent() {
        tokio::fs::create_dir_all(parent).await.ok();
    }
    tokio::fs::write(abs, content)
        .await
        .map_err(|e| AgentError::Tool(format!("cannot write {}: {e}", abs.display())))
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
        let text = read_source(&self.client, &abs).await?;
        if text.len() > MAX_READ_BYTES {
            return Err(AgentError::Tool(format!(
                "{path} is {} bytes; read a range with offset/limit or use grep",
                text.len()
            )));
        }
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
        let old = read_source(&self.client, &abs).await.unwrap_or_default();
        write_source(&self.client, &abs, &content).await?;
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
        let original = read_source(&self.client, &abs).await?;
        let updated = apply_replacement(&original, &old_string, &new_string, replace_all)?;
        write_source(&self.client, &abs, &updated).await?;
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
        let original = read_source(&self.client, &abs).await?;

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

        write_source(&self.client, &abs, &updated).await?;
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
        let root = self.ws.root().to_path_buf();
        let matches = tokio::task::spawn_blocking(move || {
            let matcher = glob::Pattern::new(&pattern).ok();
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
            .and_then(|g| glob::Pattern::new(g).ok());

        let root = self
            .ws
            .resolve(sub)
            .map_err(|e| AgentError::Tool(e.to_string()))?;
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
        let abs = self
            .ws
            .resolve(rel)
            .map_err(|e| AgentError::Tool(e.to_string()))?;
        let mut entries = tokio::fs::read_dir(&abs)
            .await
            .map_err(|e| AgentError::Tool(format!("cannot list {rel}: {e}")))?;

        let mut out = Vec::new();
        while let Ok(Some(entry)) = entries.next_entry().await {
            let is_dir = entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false);
            out.push(json!({
                "name": entry.file_name().to_string_lossy(),
                "type": if is_dir { "dir" } else { "file" },
            }));
        }
        Ok(json!({ "path": rel, "entries": out }))
    }
}

// ── bash ──────────────────────────────────────────────────────────────────────

/// Handles for commands still running in the background.
#[derive(Default)]
pub struct BackgroundJobs {
    jobs: Mutex<Vec<(String, PathBuf)>>,
}

pub struct Bash {
    pub ws: Arc<Workspace>,
    pub jobs: Arc<BackgroundJobs>,
}

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

        let mut cmd = tokio::process::Command::new("bash");
        cmd.arg("-lc")
            .arg(&command)
            .current_dir(self.ws.root())
            .kill_on_drop(!background);

        if background {
            let log = self
                .ws
                .root()
                .join(format!(".eventage-code-job-{}.log", uuid::Uuid::new_v4()));
            let file = std::fs::File::create(&log)
                .map_err(|e| AgentError::Tool(format!("cannot create job log: {e}")))?;
            let errfile = file
                .try_clone()
                .map_err(|e| AgentError::Tool(e.to_string()))?;
            cmd.stdout(file).stderr(errfile);
            let child = cmd
                .spawn()
                .map_err(|e| AgentError::Tool(format!("spawn failed: {e}")))?;
            let id = child.id().map(|p| p.to_string()).unwrap_or_default();
            self.jobs.jobs.lock().await.push((id.clone(), log.clone()));
            return Ok(json!({
                "background": true,
                "pid": id,
                "log_file": log.display().to_string(),
                "note": "running in background; read log_file to check progress",
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
            lsp: Arc::new(LspPool::new(dir.path())),
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
        let file = dir.path().join("f.txt");
        std::fs::write(&file, "on disk").unwrap();

        assert_eq!(read_source(&None, &file).await.unwrap(), "on disk");

        write_source(&None, &file, "written").await.unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "written");

        // Parent directories are created on demand.
        let nested = dir.path().join("a/b/c.txt");
        write_source(&None, &nested, "deep").await.unwrap();
        assert_eq!(std::fs::read_to_string(&nested).unwrap(), "deep");
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
        let text = read_source(&Some(no_caps), &file).await.unwrap();
        assert_eq!(text, "on disk");
    }

    #[tokio::test]
    async fn bash_reports_exit_code_and_output() {
        let dir = tempfile::tempdir().unwrap();
        let ws = Arc::new(Workspace::open(dir.path()).unwrap());
        let tool = Bash {
            ws,
            jobs: Arc::new(BackgroundJobs::default()),
        };
        let out = tool
            .execute(json!({ "command": "echo hi && exit 3" }))
            .await
            .unwrap();
        assert_eq!(out["exit_code"], 3);
        assert!(out["stdout"].as_str().unwrap().contains("hi"));
    }
}
