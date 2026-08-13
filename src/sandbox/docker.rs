//! Docker container sandbox executor.

use super::{SandboxError, SandboxExecutor, SandboxOutput, SandboxRequest};
use async_trait::async_trait;
use std::path::Path;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn};

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
    pub async fn check(&self) -> Result<(), SandboxError> {
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

        if !req.working_dir.exists() {
            return Err(SandboxError::HostPathMissing(req.working_dir.clone()));
        }
        for path in req.readable_paths.iter().chain(req.writable_paths.iter()) {
            if path != &req.working_dir && !path.exists() {
                return Err(SandboxError::HostPathMissing(path.clone()));
            }
        }

        let container_program = translate_path(&req.program, &req.working_dir);

        let container_args: Vec<String> = req
            .args
            .iter()
            .map(|a| translate_path(a, &req.working_dir))
            .collect();

        // A unique name lets us kill the *container* on timeout — SIGKILLing
        // the `docker run` client alone leaves the container running.
        let container_name = format!("eventage-{}", uuid::Uuid::new_v4());

        let mut docker_args: Vec<String> = vec![
            "run".into(),
            "--rm".into(),
            "--name".into(),
            container_name.clone(),
            "--network".into(),
            "none".into(),
            "--memory".into(),
            "256m".into(),
            "--cpus".into(),
            "1".into(),
            "--pids-limit".into(),
            "256".into(),
            "--cap-drop".into(),
            "ALL".into(),
            "--security-opt".into(),
            "no-new-privileges".into(),
            "-v".into(),
            format!("{workspace_host}:{CONTAINER_WORKSPACE}"),
        ];

        for mount in mount_specs(&req) {
            docker_args.push("-v".into());
            docker_args.push(mount);
        }

        // Pass env var *names* only; `docker` reads each value from its own
        // process environment. Values never appear in argv (world-readable
        // via /proc) or in our logs.
        for k in req.env.keys() {
            docker_args.push("-e".into());
            docker_args.push(k.clone());
        }

        docker_args.push("-w".into());
        docker_args.push(CONTAINER_WORKSPACE.into());

        if req.stdin.is_some() {
            docker_args.push("-i".into());
        }

        docker_args.push(self.image.clone());
        docker_args.push(container_program);
        docker_args.extend(container_args);

        // Safe to log: env values are not part of the argument list.
        info!("docker: docker {}", docker_args.join(" "));

        let mut cmd = tokio::process::Command::new("docker");
        cmd.args(&docker_args)
            .envs(&req.env)
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
            Err(_) => {
                // Kill the container itself, not just the docker client.
                // `--rm` then removes it once it exits.
                let kill = tokio::process::Command::new("docker")
                    .args(["kill", &container_name])
                    .output()
                    .await;
                if let Err(e) = kill {
                    warn!(container = %container_name, "failed to kill timed-out container: {e}");
                }
                Ok(SandboxOutput {
                    stdout: String::new(),
                    stderr: String::new(),
                    exit_code: -1,
                    timed_out: true,
                })
            }
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

/// Build read-only mount specs for the extra readable paths, deduplicating
/// container-side names so two host paths with the same basename can't
/// silently shadow each other.
fn mount_specs(req: &SandboxRequest) -> Vec<String> {
    use std::collections::HashSet;
    let mut used: HashSet<String> = HashSet::new();
    let mut specs = Vec::new();
    for path in &req.readable_paths {
        if path == &req.working_dir || !path.exists() {
            continue;
        }
        let host = path.to_string_lossy();
        let base = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let mut name = base.clone();
        let mut n = 1usize;
        while !used.insert(name.clone()) {
            n += 1;
            name = format!("{base}-{n}");
        }
        specs.push(format!("{host}:/sandbox/{name}:ro"));
    }
    specs
}

/// Rewrite `path_str` from the host workspace prefix to the container
/// workspace. Only rewrites at a path-component boundary, so a sibling like
/// `/work/project-backup` is not mangled when the workspace is `/work/project`.
fn translate_path(path_str: &str, workspace: &Path) -> String {
    let workspace_str = workspace.to_string_lossy();
    let ws = workspace_str.trim_end_matches('/');
    if let Some(rel) = path_str.strip_prefix(ws) {
        if rel.is_empty() || rel.starts_with('/') {
            return format!("{CONTAINER_WORKSPACE}{rel}");
        }
    }
    path_str.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::path::PathBuf;

    #[test]
    fn translate_only_at_path_boundary() {
        let ws = Path::new("/work/project");
        assert_eq!(translate_path("/work/project", ws), CONTAINER_WORKSPACE);
        assert_eq!(
            translate_path("/work/project/src/main.c", ws),
            format!("{CONTAINER_WORKSPACE}/src/main.c")
        );
        assert_eq!(
            translate_path("/work/project-backup/x", ws),
            "/work/project-backup/x",
            "sibling directory must not be rewritten"
        );
        assert_eq!(translate_path("-O2", ws), "-O2");
    }

    #[test]
    fn mount_names_are_deduplicated() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a").join("data");
        let b = dir.path().join("b").join("data");
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();

        let req = SandboxRequest {
            program: "true".into(),
            args: vec![],
            env: HashMap::new(),
            stdin: None,
            timeout_ms: 1000,
            working_dir: PathBuf::from("/tmp"),
            readable_paths: vec![a, b],
            writable_paths: vec![],
        };
        let specs = mount_specs(&req);
        assert_eq!(specs.len(), 2);
        assert!(specs[0].ends_with(":/sandbox/data:ro"));
        assert!(specs[1].ends_with(":/sandbox/data-2:ro"));
    }
}
