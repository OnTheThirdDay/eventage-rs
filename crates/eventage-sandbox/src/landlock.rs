//! Linux Landlock sandbox executor.
//!
//! Restricts child-process filesystem access to an explicit whitelist using
//! the [Landlock LSM](https://landlock.io) (Linux 5.13+). Unlisted access is blocked.
//!
//! # Isolation Details
//! - **Filesystem**: Child processes can only read/write to specified paths.
//!   Basic system libraries (`/usr`, `/lib`, `/etc/ld.so.cache`) are readable.
//! - **Network access**: Not restricted by Landlock V1. Use `DockerExecutor`
//!   if network isolation is required.
//! - **Process spawning**: The child can still fork/exec.
//!
//! # Kernel Requirement
//! Requires Linux 5.13+. On older kernels, it returns a `SandboxError::Setup`
//! rather than failing open.

use crate::{SandboxError, SandboxExecutor, SandboxOutput, SandboxRequest};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tracing::debug;

/// Sandbox executor using Linux Landlock for filesystem isolation.
///
/// The executor itself is stateless; all configuration is taken from the
/// per-request [`SandboxRequest`].
pub struct LandlockExecutor;

impl LandlockExecutor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LandlockExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SandboxExecutor for LandlockExecutor {
    fn name(&self) -> &str {
        "landlock"
    }

    async fn execute(&self, req: SandboxRequest) -> Result<SandboxOutput, SandboxError> {
        debug!(program = %req.program, "landlock execution");

        let readable = req.readable_paths.clone();
        let writable = req.writable_paths.clone();

        let mut cmd = tokio::process::Command::new(&req.program);
        cmd.args(&req.args)
            .current_dir(&req.working_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        // Use a minimal environment unless the caller explicitly sets vars.
        if req.env.is_empty() {
            cmd.env_clear();
            // Preserve PATH so the program can locate system libraries/helpers.
            if let Ok(path) = std::env::var("PATH") {
                cmd.env("PATH", path);
            }
        } else {
            cmd.envs(&req.env);
        }

        if req.stdin.is_some() {
            cmd.stdin(std::process::Stdio::piped());
        } else {
            cmd.stdin(std::process::Stdio::null());
        }

        // Apply Landlock rules in the child process after fork but before exec.
        // SAFETY: pre_exec runs in the fork child (single-threaded at this
        // point).  We only make Landlock syscalls and open file descriptors —
        // no tokio runtime interaction.
        unsafe {
            cmd.pre_exec(move || {
                apply_landlock(&readable, &writable)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::PermissionDenied, e))
            });
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

// ── Landlock setup (runs in fork child) ───────────────────────────────────────

/// Apply Landlock filesystem restrictions to the current process.
///
/// Called inside `pre_exec` (fork child, before exec).  Must not interact
/// with the tokio runtime.
fn apply_landlock(readable: &[PathBuf], writable: &[PathBuf]) -> Result<(), String> {
    use landlock::{
        Access, AccessFs, PathBeneath, PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr,
        RulesetStatus, ABI,
    };

    let abi = ABI::V1;

    let mut ruleset = Ruleset::default()
        .handle_access(AccessFs::from_all(abi))
        .map_err(|e| format!("landlock handle_access: {e}"))?
        .create()
        .map_err(|e| format!("landlock create ruleset: {e}"))?;

    // Standard read-only system paths needed by most programs.
    let system_ro: &[&str] = &[
        "/usr",
        "/lib",
        "/lib64",
        "/etc/ld.so.cache",
        "/etc/ld.so.conf",
        "/etc/ld.so.conf.d",
        "/proc/self",
    ];

    for path_str in system_ro {
        let path = Path::new(path_str);
        if !path.exists() {
            continue;
        }
        let fd =
            PathFd::new(path).map_err(|e| format!("landlock open system path {path_str}: {e}"))?;
        ruleset = ruleset
            .add_rule(PathBeneath::new(fd, AccessFs::from_read(abi)))
            .map_err(|e| format!("landlock add system read rule for {path_str}: {e}"))?;
    }

    // Caller-specified readable paths.
    for path in readable {
        if !path.exists() {
            continue;
        }
        let fd = PathFd::new(path)
            .map_err(|e| format!("landlock open readable path {}: {e}", path.display()))?;
        ruleset = ruleset
            .add_rule(PathBeneath::new(fd, AccessFs::from_read(abi)))
            .map_err(|e| format!("landlock add read rule for {}: {e}", path.display()))?;
    }

    // Caller-specified writable paths (implies read as well).
    for path in writable {
        if !path.exists() {
            continue;
        }
        let fd = PathFd::new(path)
            .map_err(|e| format!("landlock open writable path {}: {e}", path.display()))?;
        ruleset = ruleset
            .add_rule(PathBeneath::new(fd, AccessFs::from_all(abi)))
            .map_err(|e| format!("landlock add write rule for {}: {e}", path.display()))?;
    }

    let status = ruleset
        .restrict_self()
        .map_err(|e| format!("landlock restrict_self: {e}"))?;

    if status.ruleset == RulesetStatus::NotEnforced {
        return Err(
            "Landlock is not supported by this kernel (requires Linux 5.13+). \
             Use --sandbox none or --sandbox docker."
                .to_string(),
        );
    }

    Ok(())
}
