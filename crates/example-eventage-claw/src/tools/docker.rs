//! Docker-isolated command execution tool.
//!
//! Runs commands inside a disposable Docker container with the group workspace
//! mounted at `/workspace`. Each call spawns a new `--rm` container; nothing
//! persists between calls except the shared `/workspace` mount.
//!
//! The agent process runs on the host but code execution is sandboxed. This
//! is sufficient for writing/running software safely; the agent itself does not
//! need to be in the container.

use async_trait::async_trait;
use eventage::{AgentError, Tool, ToolDefinition};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::Duration;
use uuid::Uuid;

pub struct DockerRunCommandTool {
    /// Default Docker image (overridable per call).
    pub image: String,
    /// Host directory mounted into the container at `/workspace`.
    pub work_dir: PathBuf,
    /// Memory limit string (e.g. `"512m"`, `"2g"`).
    pub memory_limit: String,
    /// Network mode: `"none"` (isolated) or `"bridge"` (internet access).
    pub network: String,
    /// CPU quota (e.g. `"1.0"` = one core).
    pub cpus: String,
    /// Per-call timeout in seconds.
    pub timeout_secs: u64,
}

impl DockerRunCommandTool {
    pub fn new(work_dir: PathBuf, image: impl Into<String>) -> Self {
        Self {
            image: image.into(),
            work_dir,
            memory_limit: "512m".into(),
            network: "none".into(),
            cpus: "1.0".into(),
            timeout_secs: 120,
        }
    }

    pub fn with_network(mut self, network: impl Into<String>) -> Self {
        self.network = network.into();
        self
    }
}

#[async_trait]
impl Tool for DockerRunCommandTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "run_in_docker",
            format!(
                "Execute a shell command in an isolated Docker container (default image: {}). \
                 The group workspace is mounted at /workspace (read-write). \
                 Use for compiling, testing, running scripts, or any code that should be \
                 sandboxed. The container is deleted after each call.",
                self.image
            ),
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute inside the container, e.g. 'python3 main.py', 'cargo test', 'npm run build'."
                    },
                    "image": {
                        "type": "string",
                        "description": "Docker image override (e.g. 'node:20-slim', 'python:3.12-slim', 'rust:latest', 'ubuntu:22.04'). Uses the default image if omitted."
                    },
                    "network": {
                        "type": "string",
                        "description": "'none' (default, fully isolated) or 'bridge' (internet access for installing packages).",
                        "enum": ["none", "bridge"]
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

        let image = args["image"].as_str().unwrap_or(&self.image).to_string();

        let network = args["network"]
            .as_str()
            .unwrap_or(&self.network)
            .to_string();

        // Use canonical path so Docker on Linux/Mac can resolve the bind mount.
        let work_dir = self
            .work_dir
            .canonicalize()
            .unwrap_or_else(|_| self.work_dir.clone());

        let container_name = format!("claw-{}", &Uuid::new_v4().to_string()[..8]);

        let result = tokio::time::timeout(
            Duration::from_secs(self.timeout_secs),
            tokio::process::Command::new("docker")
                .args([
                    "run",
                    "--rm",
                    "--name",
                    &container_name,
                    "-v",
                    &format!("{}:/workspace", work_dir.display()),
                    "-w",
                    "/workspace",
                    "--memory",
                    &self.memory_limit,
                    "--cpus",
                    &self.cpus,
                    "--network",
                    &network,
                    "--security-opt",
                    "no-new-privileges",
                    &image,
                    "sh",
                    "-c",
                    command,
                ])
                .output(),
        )
        .await;

        match result {
            Err(_) => Err(AgentError::Tool(format!(
                "docker command timed out after {}s — increase timeout or split the task",
                self.timeout_secs
            ))),
            Ok(Err(e)) if e.kind() == std::io::ErrorKind::NotFound => Err(AgentError::Tool(
                "Docker not found — is Docker installed and running?".into(),
            )),
            Ok(Err(e)) => Err(AgentError::Tool(format!("docker spawn failed: {e}"))),
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let exit_code = output.status.code().unwrap_or(-1);

                Ok(json!({
                    "command": command,
                    "image": image,
                    "exit_code": exit_code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "success": output.status.success(),
                }))
            }
        }
    }
}
