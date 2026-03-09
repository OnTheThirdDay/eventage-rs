//! Direct process execution with no isolation.
//!
//! **Do not use in production with untrusted agent-generated code.**
//! Suitable only for trusted environments where sandboxing overhead is unwanted.

use crate::{SandboxError, SandboxExecutor, SandboxOutput, SandboxRequest};
use async_trait::async_trait;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tracing::debug;

/// Runs the process directly on the host with no filesystem or syscall
/// restrictions.  Timeout and stdin support are still enforced.
pub struct UnsandboxedExecutor;

impl UnsandboxedExecutor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for UnsandboxedExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SandboxExecutor for UnsandboxedExecutor {
    fn name(&self) -> &str {
        "unsandboxed"
    }

    async fn execute(&self, req: SandboxRequest) -> Result<SandboxOutput, SandboxError> {
        debug!(program = %req.program, "unsandboxed execution");

        let mut cmd = tokio::process::Command::new(&req.program);
        cmd.args(&req.args)
            .current_dir(&req.working_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        if !req.env.is_empty() {
            cmd.envs(&req.env);
        }

        if req.stdin.is_some() {
            cmd.stdin(std::process::Stdio::piped());
        } else {
            cmd.stdin(std::process::Stdio::null());
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| SandboxError::Spawn(format!("failed to spawn '{}': {e}", req.program)))?;

        if let Some(data) = &req.stdin {
            if let Some(mut handle) = child.stdin.take() {
                let _ = handle.write_all(data.as_bytes()).await;
            }
        }

        match tokio::time::timeout(
            Duration::from_millis(req.timeout_ms),
            child.wait_with_output(),
        )
        .await
        {
            Err(_) => Ok(SandboxOutput {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: -1,
                timed_out: true,
            }),
            Ok(Err(e)) => Err(SandboxError::Io(e)),
            Ok(Ok(output)) => Ok(SandboxOutput {
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                exit_code: output.status.code().unwrap_or(-1),
                timed_out: false,
            }),
        }
    }
}
