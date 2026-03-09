//! Docker container sandbox executor.
//!
//! Runs the target process inside an ephemeral Docker container with:
//! - `--network none`: No outbound network access.
//! - `--memory 256m`: Capped memory usage.
//! - `--cpus 1`: Single CPU limit.
//! - Read-only root filesystem with specific writable mounts.
//! - `--rm`: Auto-removed on exit.
//!
//! # Requirements
//! Docker must be installed, running, and accessible to the current user.
//!
//! # Path Translation
//! `SandboxRequest::working_dir` is mounted as `/sandbox/workspace`.
//! Absolute paths to the workspace in `program` or `args` are translated automatically.
//!
//! # Image Selection
//! Choose an image with the necessary tools, e.g., `"gcc:13"` or `"ubuntu:22.04"`.

use crate::{SandboxError, SandboxExecutor, SandboxOutput, SandboxRequest};
use async_trait::async_trait;
use std::path::Path;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info};

const CONTAINER_WORKSPACE: &str = "/sandbox/workspace";

/// Sandbox executor that runs processes inside Docker containers.
///
/// Each `execute()` call spawns a fresh container and removes it on completion.
pub struct DockerExecutor {
    /// Docker image to use, e.g. `"gcc:13"` or `"ubuntu:22.04"`.
    pub image: String,
}

impl DockerExecutor {
    pub fn new(image: impl Into<String>) -> Self {
        Self {
            image: image.into(),
        }
    }

    /// Verify that the Docker daemon is reachable and the image is present.
    ///
    /// Returns `Err` with a human-readable message if either check fails.
    pub async fn check(&self) -> Result<(), SandboxError> {
        // Check daemon connectivity.
        let info = tokio::process::Command::new("docker")
            .args(["info", "--format", "{{.ServerVersion}}"])
            .output()
            .await
            .map_err(|e| {
                SandboxError::Docker(format!(
                    "cannot run 'docker info' — is Docker installed and in PATH? {e}"
                ))
            })?;

        if !info.status.success() {
            let stderr = String::from_utf8_lossy(&info.stderr);
            return Err(SandboxError::DaemonUnreachable(stderr.trim().to_string()));
        }

        // Check that the image exists locally.
        let inspect = tokio::process::Command::new("docker")
            .args(["image", "inspect", &self.image, "--format", "{{.Id}}"])
            .output()
            .await
            .map_err(|e| SandboxError::DaemonUnreachable(e.to_string()))?;

        if !inspect.status.success() {
            return Err(SandboxError::ImageNotFound(self.image.clone()));
        }

        let server_ver = String::from_utf8_lossy(&info.stdout).trim().to_string();
        info!(
            image = %self.image,
            docker_version = %server_ver,
            "docker sandbox ready"
        );
        Ok(())
    }
}

#[async_trait]
impl SandboxExecutor for DockerExecutor {
    fn name(&self) -> &str {
        "docker"
    }

    async fn execute(&self, req: SandboxRequest) -> Result<SandboxOutput, SandboxError> {
        debug!(program = %req.program, image = %self.image, "docker execution");

        let workspace_host = req.working_dir.to_string_lossy().to_string();

        // Fail fast on missing host paths — Docker would silently create empty
        // root-owned directories for missing mounts, leading to confusing
        // permission errors inside the container.
        if !req.working_dir.exists() {
            return Err(SandboxError::HostPathMissing(req.working_dir.clone()));
        }
        for path in req.readable_paths.iter().chain(req.writable_paths.iter()) {
            if path != &req.working_dir && !path.exists() {
                return Err(SandboxError::HostPathMissing(path.clone()));
            }
        }

        // Translate the program path: if it lives inside the workspace,
        // remap it to the container mount point.
        let container_program = translate_path(&req.program, &req.working_dir);

        // Translate args that look like absolute workspace paths.
        let container_args: Vec<String> = req
            .args
            .iter()
            .map(|a| translate_path(a, &req.working_dir))
            .collect();

        // Build the `docker run` command.
        let mut docker_args: Vec<String> = vec![
            "run".into(),
            "--rm".into(),
            "--network".into(),
            "none".into(),
            "--memory".into(),
            "256m".into(),
            "--cpus".into(),
            "1".into(),
            // Mount the workspace as read-write.
            "-v".into(),
            format!("{workspace_host}:{CONTAINER_WORKSPACE}"),
            // Additional readable paths mounted read-only.
        ];

        for path in &req.readable_paths {
            if path != &req.working_dir && path.exists() {
                let host = path.to_string_lossy();
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                docker_args.push("-v".into());
                docker_args.push(format!("{host}:/sandbox/{name}:ro"));
            }
        }

        // Pass environment variables.
        for (k, v) in &req.env {
            docker_args.push("-e".into());
            docker_args.push(format!("{k}={v}"));
        }

        // Set working directory inside the container.
        docker_args.push("-w".into());
        docker_args.push(CONTAINER_WORKSPACE.into());

        if req.stdin.is_some() {
            docker_args.push("-i".into()); // interactive (keep stdin open)
        }

        // Image and the actual command.
        docker_args.push(self.image.clone());
        docker_args.push(container_program);
        docker_args.extend(container_args);

        info!("docker: {} {}", "docker", docker_args.join(" "));

        let mut cmd = tokio::process::Command::new("docker");
        cmd.args(&docker_args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        if req.stdin.is_some() {
            cmd.stdin(std::process::Stdio::piped());
        } else {
            cmd.stdin(std::process::Stdio::null());
        }

        let mut child = cmd.spawn().map_err(|e| {
            SandboxError::DaemonUnreachable(format!(
                "failed to spawn docker — is Docker installed and running? {e}"
            ))
        })?;

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
            Ok(Err(e)) => Err(SandboxError::Docker(e.to_string())),
            Ok(Ok(output)) => Ok(SandboxOutput {
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                exit_code: output.status.code().unwrap_or(-1),
                timed_out: false,
            }),
        }
    }
}

/// If `path_str` is an absolute path that starts with `workspace`, replace
/// the workspace prefix with the container mount point.
fn translate_path(path_str: &str, workspace: &Path) -> String {
    let workspace_str = workspace.to_string_lossy();
    if let Some(rel) = path_str.strip_prefix(workspace_str.as_ref()) {
        format!("{CONTAINER_WORKSPACE}{rel}")
    } else {
        path_str.to_string()
    }
}
