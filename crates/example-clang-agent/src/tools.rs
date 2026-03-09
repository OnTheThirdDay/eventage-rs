use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use eventage_agent::{AgentError, Tool};
use eventage_llm::types::ToolDefinition;
use eventage_sandbox::{SandboxExecutor, SandboxRequest};
use serde_json::{json, Value};
use tracing::{debug, instrument};

use crate::permissions::PermissionGate;
use crate::workspace::Workspace;

// ── Constants ─────────────────────────────────────────────────────────────────

const MAX_FILE_READ_BYTES: usize = 100 * 1024; // 100 KB
const MAX_COMPILER_OUTPUT_BYTES: usize = 16 * 1024; // 16 KB
const MAX_EXEC_OUTPUT_BYTES: usize = 16 * 1024; // 16 KB per stream

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

// ─────────────────────────────────────────────────────────────────────────────
// WriteFile
// ─────────────────────────────────────────────────────────────────────────────

pub struct WriteFile {
    pub workspace: Arc<Workspace>,
    pub gate: Arc<dyn PermissionGate>,
}

#[async_trait]
impl Tool for WriteFile {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "write_file",
            "Write (create or overwrite) a file in the workspace. \
             Always call this before compile when creating or modifying source code.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path within the workspace, e.g. 'main.c' or 'src/utils.c'."
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
        let rel_path = require_str(&args, "path")?;
        let content = require_str(&args, "content")?;

        let abs_path = self
            .workspace
            .resolve(rel_path)
            .map_err(|e| AgentError::Tool(e.to_string()))?;

        self.gate.request("write file", rel_path).await?;

        if let Some(parent) = abs_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| AgentError::Tool(format!("failed to create directories: {e}")))?;
        }

        std::fs::write(&abs_path, content)
            .map_err(|e| AgentError::Tool(format!("write failed: {e}")))?;

        debug!("wrote {} bytes to {}", content.len(), rel_path);

        Ok(json!({
            "path": rel_path,
            "bytes_written": content.len(),
            "success": true
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ReadFile
// ─────────────────────────────────────────────────────────────────────────────

pub struct ReadFile {
    pub workspace: Arc<Workspace>,
}

#[async_trait]
impl Tool for ReadFile {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "read_file",
            "Read the contents of a file from the workspace.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path, e.g. 'main.c'."
                    }
                },
                "required": ["path"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let rel_path = require_str(&args, "path")?;

        let abs_path = self
            .workspace
            .resolve(rel_path)
            .map_err(|e| AgentError::Tool(e.to_string()))?;

        if !abs_path.exists() {
            return Ok(json!({
                "success": false,
                "error": format!("file '{rel_path}' does not exist")
            }));
        }

        let raw =
            std::fs::read(&abs_path).map_err(|e| AgentError::Tool(format!("read failed: {e}")))?;

        let content = match std::str::from_utf8(&raw) {
            Ok(s) => truncate(s, MAX_FILE_READ_BYTES),
            Err(_) => {
                return Ok(json!({
                    "success": false,
                    "error": format!("'{rel_path}' is not valid UTF-8 (binary file?)")
                }))
            }
        };

        Ok(json!({
            "path": rel_path,
            "content": content,
            "size_bytes": raw.len(),
            "success": true
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ListFiles
// ─────────────────────────────────────────────────────────────────────────────

pub struct ListFiles {
    pub workspace: Arc<Workspace>,
}

#[async_trait]
impl Tool for ListFiles {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "list_files",
            "List all files currently in the workspace with their sizes.",
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
            .map(|f| {
                json!({
                    "path": f.path,
                    "size_bytes": f.size_bytes
                })
            })
            .collect();

        Ok(json!({
            "files": entries,
            "count": entries.len()
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Compile  (sandbox-aware)
// ─────────────────────────────────────────────────────────────────────────────

pub struct Compile {
    pub workspace: Arc<Workspace>,
    pub executor: Arc<dyn SandboxExecutor>,
    pub gate: Arc<dyn PermissionGate>,
}

#[async_trait]
impl Tool for Compile {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "compile",
            "Compile a C source file with gcc. \
             -Wall and -Wextra are always enabled. \
             The output binary is placed in the workspace bin/ directory. \
             Returns success status, warnings, and errors.",
            json!({
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Relative path to the .c source file, e.g. 'main.c'."
                    },
                    "output": {
                        "type": "string",
                        "description": "Name for the output binary (no path), e.g. 'main'. \
                                       The binary will be written to bin/{output}."
                    },
                    "flags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Extra compiler flags, e.g. [\"-O2\", \"-lm\", \"-lpthread\", \"-g\"]."
                    }
                },
                "required": ["source", "output"]
            }),
        )
    }

    #[instrument(skip(self, args), fields(
        source = args["source"].as_str().unwrap_or("?"),
        output = args["output"].as_str().unwrap_or("?")
    ))]
    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let source = require_str(&args, "source")?;
        let output_name = require_str(&args, "output")?;

        // Reject obvious path tricks in the binary name.
        if output_name.contains('/') || output_name.contains('\\') || output_name.contains("..") {
            return Err(AgentError::Tool(
                "output name must be a simple filename without path separators".to_string(),
            ));
        }

        let source_path = self
            .workspace
            .resolve(source)
            .map_err(|e| AgentError::Tool(e.to_string()))?;

        if !source_path.exists() {
            return Ok(json!({
                "success": false,
                "errors": format!("source file '{source}' does not exist — did you call write_file first?"),
                "warnings": ""
            }));
        }

        self.workspace
            .ensure_bin_dir()
            .map_err(|e| AgentError::Tool(e.to_string()))?;

        let output_path = self
            .workspace
            .resolve(&format!("bin/{output_name}"))
            .map_err(|e| AgentError::Tool(e.to_string()))?;

        let extra_flags: Vec<String> = args
            .get("flags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        // Build gcc args.
        let mut gcc_args = vec![
            "-Wall".to_string(),
            "-Wextra".to_string(),
            "-g".to_string(),
            source_path.to_string_lossy().to_string(),
            "-o".to_string(),
            output_path.to_string_lossy().to_string(),
        ];
        gcc_args.extend(extra_flags);

        self.gate.request("compile", source).await?;

        let req = SandboxRequest {
            program: "gcc".to_string(),
            args: gcc_args,
            env: HashMap::new(),
            stdin: None,
            timeout_ms: 30_000,
            working_dir: self.workspace.root().to_path_buf(),
            readable_paths: vec![self.workspace.root().to_path_buf()],
            writable_paths: vec![self.workspace.root().join("bin")],
        };

        let out = self
            .executor
            .execute(req)
            .await
            .map_err(|e| AgentError::Tool(format!("sandbox error: {e}")))?;

        // gcc sends warnings/errors to stderr; stdout is usually empty.
        let warnings = truncate(&out.stdout, MAX_COMPILER_OUTPUT_BYTES);
        let errors = truncate(&out.stderr, MAX_COMPILER_OUTPUT_BYTES);
        let success = out.exit_code == 0 && !out.timed_out;

        debug!(success, exit_code = out.exit_code, "gcc finished");

        Ok(json!({
            "success": success,
            "output_path": format!("bin/{output_name}"),
            "exit_code": out.exit_code,
            "warnings": warnings,
            "errors": errors,
            "sandbox": self.executor.name()
        }))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Execute  (sandbox-aware)
// ─────────────────────────────────────────────────────────────────────────────

pub struct Execute {
    pub workspace: Arc<Workspace>,
    pub executor: Arc<dyn SandboxExecutor>,
    pub gate: Arc<dyn PermissionGate>,
    pub default_timeout_ms: u64,
}

#[async_trait]
impl Tool for Execute {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "execute",
            "Run a compiled binary from the workspace. \
             Returns stdout, stderr, exit code, and whether the process timed out. \
             The binary's working directory is the workspace root.",
            json!({
                "type": "object",
                "properties": {
                    "binary": {
                        "type": "string",
                        "description": "Relative path to the binary, e.g. 'bin/main'."
                    },
                    "args": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Command-line arguments to pass to the binary."
                    },
                    "stdin": {
                        "type": "string",
                        "description": "Data to write to the binary's standard input."
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Maximum execution time in milliseconds. Default: 10000."
                    }
                },
                "required": ["binary"]
            }),
        )
    }

    #[instrument(skip(self, args), fields(binary = args["binary"].as_str().unwrap_or("?")))]
    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let binary = require_str(&args, "binary")?;

        let binary_path = self
            .workspace
            .resolve(binary)
            .map_err(|e| AgentError::Tool(e.to_string()))?;

        if !binary_path.exists() {
            return Ok(json!({
                "success": false,
                "error": format!("binary '{binary}' does not exist — did you compile it first?")
            }));
        }

        let extra_args: Vec<String> = args
            .get("args")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let stdin_data: Option<String> = args
            .get("stdin")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(String::from);

        let timeout_ms = args
            .get("timeout_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.default_timeout_ms);

        self.gate.request("execute binary", binary).await?;

        let workspace_root = self.workspace.root().to_path_buf();

        let req = SandboxRequest {
            program: binary_path.to_string_lossy().to_string(),
            args: extra_args,
            env: HashMap::new(),
            stdin: stdin_data,
            timeout_ms,
            working_dir: workspace_root.clone(),
            readable_paths: vec![workspace_root.clone()],
            writable_paths: vec![workspace_root],
        };

        debug!(
            timeout_ms,
            sandbox = self.executor.name(),
            "executing binary"
        );

        let out = self
            .executor
            .execute(req)
            .await
            .map_err(|e| AgentError::Tool(format!("sandbox error: {e}")))?;

        let stdout = truncate(&out.stdout, MAX_EXEC_OUTPUT_BYTES);
        let stderr = truncate(&out.stderr, MAX_EXEC_OUTPUT_BYTES);

        debug!(exit_code = out.exit_code, "process finished");

        Ok(json!({
            "success": out.success(),
            "timed_out": out.timed_out,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": out.exit_code,
            "sandbox": self.executor.name()
        }))
    }
}
