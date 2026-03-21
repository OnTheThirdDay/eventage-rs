use async_trait::async_trait;
use eventage::{AgentError, Tool, ToolDefinition};
use serde_json::{json, Value};
use std::path::PathBuf;

use super::fs::safe_resolve;

/// Applies a unified diff patch to a file in the working directory.
pub struct ApplyPatchTool {
    pub work_dir: PathBuf,
}

#[async_trait]
impl Tool for ApplyPatchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "apply_patch",
            "Apply a unified diff patch to an existing file. \
             Use this to make targeted edits without rewriting the entire file.",
            json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to patch."
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

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'path'".into()))?;
        let patch_str = args["patch"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'patch'".into()))?;

        let abs = safe_resolve(&self.work_dir, path)?;

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
