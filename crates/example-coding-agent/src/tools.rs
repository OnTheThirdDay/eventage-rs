//! Coding agent tools: ReadFile, WriteFile, ApplyPatch, ExecuteShell, ListDir.
//!
//! `WriteFile`, `ApplyPatch`, and `ExecuteShell` are intercepted by
//! [`crate::security::SecurityGateHook`] to require user approval.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use eventage_llm::types::ToolDefinition;
use eventage_provided_impl::{AgentError, Tool};
use eventage_sandbox::{SandboxExecutor, SandboxRequest};
use serde_json::{json, Value};
use tracing::instrument;

use crate::workspace::Workspace;

// ── Constants ─────────────────────────────────────────────────────────────────

const MAX_FILE_BYTES: usize = 200 * 1024; // 200 KB read limit
const MAX_OUTPUT_BYTES: usize = 32 * 1024; // 32 KB per stream

// ── Helpers ───────────────────────────────────────────────────────────────────

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!(
            "{}\n…[truncated, {} bytes omitted]",
            &s[..max],
            s.len() - max
        )
    }
}

fn require_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, AgentError> {
    args.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| AgentError::Tool(format!("missing required argument '{key}'")))
}

// ── ReadFile ──────────────────────────────────────────────────────────────────

pub struct ReadFile {
    pub workspace: Arc<Workspace>,
}

#[async_trait]
impl Tool for ReadFile {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "read_file",
            "Read the full contents of a file from the workspace.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path, e.g. 'src/main.py'."
                    }
                },
                "required": ["path"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let path = require_str(&args, "path")?;
        let abs = self
            .workspace
            .resolve(path)
            .map_err(|e| AgentError::Tool(e.to_string()))?;

        if !abs.exists() {
            return Ok(json!({ "success": false, "error": format!("'{path}' not found") }));
        }

        let raw = std::fs::read(&abs).map_err(|e| AgentError::Tool(e.to_string()))?;
        match std::str::from_utf8(&raw) {
            Ok(s) => Ok(json!({
                "path": path,
                "content": truncate(s, MAX_FILE_BYTES),
                "size_bytes": raw.len(),
                "success": true
            })),
            Err(_) => {
                Ok(json!({ "success": false, "error": "binary file — not readable as UTF-8" }))
            }
        }
    }
}

// ── WriteFile ─────────────────────────────────────────────────────────────────

/// **Dangerous** — intercepted by `SecurityGateHook`.
pub struct WriteFile {
    pub workspace: Arc<Workspace>,
}

#[async_trait]
impl Tool for WriteFile {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "write_file",
            "Create or overwrite a file in the workspace with the given content. \
             Requires security approval.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path, e.g. 'src/main.py'."
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file content to write."
                    }
                },
                "required": ["path", "content"]
            }),
        )
    }

    #[instrument(skip(self, args), fields(path = args["path"].as_str().unwrap_or("?")))]
    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let path = require_str(&args, "path")?;
        let content = require_str(&args, "content")?;

        let abs = self
            .workspace
            .resolve(path)
            .map_err(|e| AgentError::Tool(e.to_string()))?;

        if let Some(parent) = abs.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| AgentError::Tool(format!("mkdir failed: {e}")))?;
        }

        std::fs::write(&abs, content)
            .map_err(|e| AgentError::Tool(format!("write failed: {e}")))?;

        Ok(json!({
            "path": path,
            "bytes_written": content.len(),
            "success": true
        }))
    }
}

// ── ApplyPatch ────────────────────────────────────────────────────────────────

/// **Dangerous** — intercepted by `SecurityGateHook`.
///
/// Applies a unified diff patch to a file in the workspace. The patch must be
/// in standard unified diff format (as produced by `diff -u` or `git diff`).
pub struct ApplyPatch {
    pub workspace: Arc<Workspace>,
}

#[async_trait]
impl Tool for ApplyPatch {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "apply_patch",
            "Apply a unified diff patch to an existing file. \
             Use this to make targeted edits without rewriting the entire file. \
             Requires security approval.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path to the file to patch."
                    },
                    "patch": {
                        "type": "string",
                        "description": "Unified diff patch (e.g., from `diff -u` or `git diff`). \
                                       Must include the @@ header lines."
                    }
                },
                "required": ["path", "patch"]
            }),
        )
    }

    #[instrument(skip(self, args), fields(path = args["path"].as_str().unwrap_or("?")))]
    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let path = require_str(&args, "path")?;
        let patch_str = require_str(&args, "patch")?;

        let abs = self
            .workspace
            .resolve(path)
            .map_err(|e| AgentError::Tool(e.to_string()))?;

        let original = if abs.exists() {
            std::fs::read_to_string(&abs).map_err(|e| AgentError::Tool(e.to_string()))?
        } else {
            String::new()
        };

        let patch = diffy::Patch::from_str(patch_str)
            .map_err(|e| AgentError::Tool(format!("invalid patch: {e}")))?;

        let patched = diffy::apply(&original, &patch)
            .map_err(|e| AgentError::Tool(format!("patch failed to apply: {e}")))?;

        if let Some(parent) = abs.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| AgentError::Tool(format!("mkdir failed: {e}")))?;
        }

        std::fs::write(&abs, &patched).map_err(|e| AgentError::Tool(e.to_string()))?;

        Ok(json!({
            "path": path,
            "original_lines": original.lines().count(),
            "patched_lines": patched.lines().count(),
            "success": true
        }))
    }
}

// ── ExecuteShell ──────────────────────────────────────────────────────────────

/// **Dangerous** — intercepted by `SecurityGateHook`.
///
/// Runs a shell command inside the workspace via the configured sandbox.
pub struct ExecuteShell {
    pub workspace: Arc<Workspace>,
    pub executor: Arc<dyn SandboxExecutor>,
    pub default_timeout_ms: u64,
}

#[async_trait]
impl Tool for ExecuteShell {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "execute_shell",
            "Run a shell command inside the workspace. \
             The working directory is the workspace root. \
             Returns stdout, stderr, and exit code. \
             Requires security approval.",
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run, e.g. 'python main.py' or 'pytest tests/'."
                    },
                    "stdin": {
                        "type": "string",
                        "description": "Optional data to write to stdin."
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Max execution time in ms. Default: 15000."
                    }
                },
                "required": ["command"]
            }),
        )
    }

    #[instrument(skip(self, args), fields(cmd = args["command"].as_str().unwrap_or("?")))]
    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let command = require_str(&args, "command")?;
        let stdin = args.get("stdin").and_then(|v| v.as_str()).map(String::from);
        let timeout_ms = args
            .get("timeout_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.default_timeout_ms);

        let workspace_root = self.workspace.root().to_path_buf();

        let req = SandboxRequest {
            program: "/bin/sh".to_string(),
            args: vec!["-c".to_string(), command.to_string()],
            env: HashMap::new(),
            stdin,
            timeout_ms,
            working_dir: workspace_root.clone(),
            readable_paths: vec![workspace_root.clone()],
            writable_paths: vec![workspace_root],
        };

        let out = self
            .executor
            .execute(req)
            .await
            .map_err(|e| AgentError::Tool(format!("sandbox error: {e}")))?;

        Ok(json!({
            "success": out.success(),
            "timed_out": out.timed_out,
            "stdout": truncate(&out.stdout, MAX_OUTPUT_BYTES),
            "stderr": truncate(&out.stderr, MAX_OUTPUT_BYTES),
            "exit_code": out.exit_code,
            "sandbox": self.executor.name()
        }))
    }
}

// ── ListDir ───────────────────────────────────────────────────────────────────

pub struct ListDir {
    pub workspace: Arc<Workspace>,
}

#[async_trait]
impl Tool for ListDir {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "list_dir",
            "List all files in the workspace with their sizes in bytes.",
            json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        )
    }

    async fn execute(&self, _args: Value) -> Result<Value, AgentError> {
        let files = self
            .workspace
            .list_files()
            .map_err(|e| AgentError::Tool(e.to_string()))?;

        let entries: Vec<Value> = files
            .iter()
            .map(|f| json!({ "path": f.path, "size_bytes": f.size_bytes }))
            .collect();

        Ok(json!({ "files": entries, "count": entries.len() }))
    }
}
