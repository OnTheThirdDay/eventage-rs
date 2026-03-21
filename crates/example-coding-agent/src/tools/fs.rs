use async_trait::async_trait;
use eventage::{AgentError, Tool, ToolDefinition};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

// ── Path helpers ──────────────────────────────────────────────────────────────

/// Resolve a path against work_dir without escaping it. For read-only tools.
fn resolve(work_dir: &Path, input: &str) -> PathBuf {
    let p = Path::new(input);
    if p.is_absolute() {
        p.to_owned()
    } else {
        work_dir.join(p)
    }
}

/// Normalize a path without requiring it to exist (no `canonicalize`).
fn normalize_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            c => out.push(c),
        }
    }
    out
}

/// Resolve a path against work_dir, rejecting absolute paths and `../` escapes.
/// Used for write/edit tools to prevent filesystem escapes.
pub fn safe_resolve(work_dir: &Path, input: &str) -> Result<PathBuf, AgentError> {
    if input.starts_with('/') || input.starts_with('\\') {
        return Err(AgentError::Tool(format!(
            "absolute paths not allowed: {input}"
        )));
    }
    let candidate = work_dir.join(input);
    let normalised = normalize_path(&candidate);
    if !normalised.starts_with(work_dir) {
        return Err(AgentError::Tool(format!(
            "path escape detected: {input}"
        )));
    }
    Ok(normalised)
}

fn tool_err(msg: impl Into<String>) -> AgentError {
    AgentError::Tool(msg.into())
}

// ── LsTool ────────────────────────────────────────────────────────────────────

pub struct LsTool {
    pub work_dir: PathBuf,
}

#[async_trait]
impl Tool for LsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "ls",
            "List files and directories at a path.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Directory to list. Defaults to working directory." }
                }
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let path = args["path"].as_str().unwrap_or(".");
        let target = resolve(&self.work_dir, path);
        let mut entries = vec![];

        let mut rd = tokio::fs::read_dir(&target)
            .await
            .map_err(|e| tool_err(format!("ls {}: {e}", target.display())))?;

        while let Some(entry) = rd.next_entry().await.map_err(|e| tool_err(e.to_string()))? {
            let meta = entry
                .metadata()
                .await
                .map_err(|e| tool_err(e.to_string()))?;
            entries.push(json!({
                "name": entry.file_name().to_string_lossy(),
                "is_dir": meta.is_dir(),
                "size_bytes": meta.len(),
            }));
        }

        entries.sort_by(|a, b| {
            let a_name = a["name"].as_str().unwrap_or("");
            let b_name = b["name"].as_str().unwrap_or("");
            a_name.cmp(b_name)
        });

        Ok(json!({ "path": target.display().to_string(), "entries": entries }))
    }
}

// ── ReadFileTool ──────────────────────────────────────────────────────────────

pub struct ReadFileTool {
    pub work_dir: PathBuf,
}

#[async_trait]
impl Tool for ReadFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "read_file",
            "Read file contents with optional line-range selection.",
            json!({
                "type": "object",
                "properties": {
                    "path":   { "type": "string",  "description": "File path to read." },
                    "offset": { "type": "integer", "description": "First line to include (1-based, default 1)." },
                    "limit":  { "type": "integer", "description": "Max lines to return (default 200)." }
                },
                "required": ["path"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| tool_err("missing 'path'"))?;
        let target = resolve(&self.work_dir, path);

        let content = tokio::fs::read_to_string(&target)
            .await
            .map_err(|e| tool_err(format!("read {}: {e}", target.display())))?;

        let offset = args["offset"].as_u64().unwrap_or(1).saturating_sub(1) as usize;
        let limit = args["limit"].as_u64().unwrap_or(200) as usize;

        let lines: Vec<&str> = content.lines().collect();
        let start = offset.min(lines.len());
        let end = (start + limit).min(lines.len());
        let slice = &lines[start..end];

        let numbered: String = slice
            .iter()
            .enumerate()
            .map(|(i, l)| format!("{:>5} {}\n", start + i + 1, l))
            .collect();

        Ok(json!({
            "path": target.display().to_string(),
            "total_lines": lines.len(),
            "shown_lines": slice.len(),
            "content": numbered,
        }))
    }
}

// ── WriteFileTool ─────────────────────────────────────────────────────────────

pub struct WriteFileTool {
    pub work_dir: PathBuf,
}

#[async_trait]
impl Tool for WriteFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "write_file",
            "Write content to a file, creating it (and any parent directories) if needed.",
            json!({
                "type": "object",
                "properties": {
                    "path":    { "type": "string", "description": "File path to write." },
                    "content": { "type": "string", "description": "Content to write." }
                },
                "required": ["path", "content"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| tool_err("missing 'path'"))?;
        let content = args["content"]
            .as_str()
            .ok_or_else(|| tool_err("missing 'content'"))?;
        let target = safe_resolve(&self.work_dir, path)?;

        if let Some(parent) = target.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| tool_err(format!("mkdir {}: {e}", parent.display())))?;
        }

        tokio::fs::write(&target, content)
            .await
            .map_err(|e| tool_err(format!("write {}: {e}", target.display())))?;

        Ok(json!({
            "path": target.display().to_string(),
            "bytes_written": content.len(),
            "success": true
        }))
    }
}

// ── EditFileTool ──────────────────────────────────────────────────────────────

pub struct EditFileTool {
    pub work_dir: PathBuf,
}

#[async_trait]
impl Tool for EditFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "edit_file",
            "Replace the first occurrence of old_string with new_string in a file.",
            json!({
                "type": "object",
                "properties": {
                    "path":       { "type": "string", "description": "File to edit." },
                    "old_string": { "type": "string", "description": "Exact text to find." },
                    "new_string": { "type": "string", "description": "Replacement text." }
                },
                "required": ["path", "old_string", "new_string"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| tool_err("missing 'path'"))?;
        let old = args["old_string"]
            .as_str()
            .ok_or_else(|| tool_err("missing 'old_string'"))?;
        let new = args["new_string"]
            .as_str()
            .ok_or_else(|| tool_err("missing 'new_string'"))?;
        let target = safe_resolve(&self.work_dir, path)?;

        let content = tokio::fs::read_to_string(&target)
            .await
            .map_err(|e| tool_err(format!("read {}: {e}", target.display())))?;

        if !content.contains(old) {
            return Err(tool_err(format!(
                "old_string not found in {}",
                target.display()
            )));
        }

        let new_content = content.replacen(old, new, 1);
        tokio::fs::write(&target, &new_content)
            .await
            .map_err(|e| tool_err(format!("write {}: {e}", target.display())))?;

        Ok(json!({ "path": target.display().to_string(), "success": true }))
    }
}

// ── GlobTool ──────────────────────────────────────────────────────────────────

pub struct GlobTool {
    pub work_dir: PathBuf,
}

#[async_trait]
impl Tool for GlobTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "glob",
            "Find files matching a glob pattern.",
            json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Glob pattern, e.g. '**/*.rs'." },
                    "path":    { "type": "string", "description": "Base directory (default: working dir)." }
                },
                "required": ["pattern"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| tool_err("missing 'pattern'"))?;
        let base_str = args["path"].as_str().unwrap_or(".");
        let base = resolve(&self.work_dir, base_str);

        let full_pattern = base.join(pattern).to_string_lossy().to_string();

        let matches: Vec<String> = glob::glob(&full_pattern)
            .map_err(|e| tool_err(format!("invalid pattern: {e}")))?
            .filter_map(|r| r.ok())
            .map(|p| {
                p.strip_prefix(&self.work_dir)
                    .map(|rel| rel.to_string_lossy().to_string())
                    .unwrap_or_else(|_| p.to_string_lossy().to_string())
            })
            .collect();

        Ok(json!({ "pattern": pattern, "matches": matches, "count": matches.len() }))
    }
}

// ── GrepTool ──────────────────────────────────────────────────────────────────

pub struct GrepTool {
    pub work_dir: PathBuf,
}

#[async_trait]
impl Tool for GrepTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "grep",
            "Search for a regex pattern across files.",
            json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string", "description": "Regex pattern to search for." },
                    "path":    { "type": "string", "description": "Directory to search (default: working dir)." },
                    "include": { "type": "string", "description": "Glob to filter files, e.g. '*.rs'." }
                },
                "required": ["pattern"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| tool_err("missing 'pattern'"))?;
        let search_dir = resolve(&self.work_dir, args["path"].as_str().unwrap_or("."));
        let include_glob = args["include"].as_str();

        let re = regex::Regex::new(pattern).map_err(|e| tool_err(format!("invalid regex: {e}")))?;

        let file_pattern = include_glob.and_then(|g| glob::Pattern::new(g).ok());

        let mut results = vec![];
        const MAX_MATCHES: usize = 100;

        for entry in walkdir::WalkDir::new(&search_dir)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            if let Some(ref pat) = file_pattern {
                let fname = entry.file_name().to_string_lossy();
                if !pat.matches(&fname) {
                    continue;
                }
            }

            let Ok(content) = std::fs::read_to_string(entry.path()) else {
                continue;
            };
            let rel_path = entry
                .path()
                .strip_prefix(&self.work_dir)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| entry.path().to_string_lossy().to_string());

            for (line_num, line) in content.lines().enumerate() {
                if re.is_match(line) {
                    results.push(json!({
                        "file": rel_path,
                        "line": line_num + 1,
                        "text": line,
                    }));
                    if results.len() >= MAX_MATCHES {
                        return Ok(json!({
                            "pattern": pattern,
                            "matches": results,
                            "truncated": true
                        }));
                    }
                }
            }
        }

        Ok(json!({ "pattern": pattern, "matches": results, "count": results.len() }))
    }
}
