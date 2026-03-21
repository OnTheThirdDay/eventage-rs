use async_trait::async_trait;
use eventage::{AgentError, Tool, ToolDefinition};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Duration;

pub struct RunCommandTool {
    pub work_dir: PathBuf,
}

#[async_trait]
impl Tool for RunCommandTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "run_command",
            "Execute a shell command in the working directory. Use with care.",
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run, e.g. 'cargo check', 'ls -la'."
                    }
                },
                "required": ["command"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let command = args["command"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'command'".into()))?;

        let result = tokio::time::timeout(
            Duration::from_secs(30),
            tokio::process::Command::new("sh")
                .arg("-c")
                .arg(command)
                .current_dir(&self.work_dir)
                .output(),
        )
        .await;

        match result {
            Err(_) => Err(AgentError::Tool(format!(
                "command timed out after 30s: {command}"
            ))),
            Ok(Err(e)) => Err(AgentError::Tool(format!("failed to spawn: {e}"))),
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let exit_code = output.status.code().unwrap_or(-1);

                Ok(json!({
                    "command": command,
                    "exit_code": exit_code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "success": output.status.success(),
                }))
            }
        }
    }
}
