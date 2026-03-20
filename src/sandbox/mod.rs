//! Modular sandboxed execution for Eventage agent tools.
//!
//! Provides the [`SandboxExecutor`] trait and implementations with varying
//! security and portability:
//!
//! | Executor | Isolation | Platform |
//! |---|---|---|
//! | [`UnsandboxedExecutor`] | None | All |
//! | [`LandlockExecutor`] | Filesystem (Landlock) | Linux 5.13+ |
//! | [`DockerExecutor`] | Full container | Docker installed |
//! | [`WasmExecutor`] | WASM / WASI | All (feature `sandbox-wasm`) |

pub mod docker;
pub mod unsandboxed;

#[cfg(target_os = "linux")]
pub mod landlock;

#[cfg(feature = "sandbox-wasm")]
pub mod wasm;

pub use docker::DockerExecutor;
pub use unsandboxed::UnsandboxedExecutor;

#[cfg(target_os = "linux")]
pub use landlock::LandlockExecutor;

#[cfg(feature = "sandbox-wasm")]
pub use wasm::WasmExecutor;

use async_trait::async_trait;
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum SandboxError {
    #[error("Failed to spawn process: {0}")]
    Spawn(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Docker daemon is unreachable (not running or not in PATH).
    #[error("Docker daemon unreachable: {0}")]
    DaemonUnreachable(String),
    /// Requested Docker image not found locally. Pull it before running.
    #[error("Docker image '{0}' not found locally — run `docker pull {0}`")]
    ImageNotFound(String),
    /// Host filesystem path missing.
    #[error("Host path does not exist and cannot be mounted: {0}")]
    HostPathMissing(std::path::PathBuf),
    #[error("Docker error: {0}")]
    Docker(String),
    #[error("Sandbox setup failed: {0}")]
    Setup(String),
}

// ── Request / Output ──────────────────────────────────────────────────────────

/// All parameters needed to run a single process inside a sandbox.
pub struct SandboxRequest {
    /// Executable to run (absolute path or name on PATH).
    pub program: String,
    /// Arguments to pass to the executable.
    pub args: Vec<String>,
    /// Additional environment variables merged over the sandbox baseline.
    pub env: HashMap<String, String>,
    /// Data to pipe to standard input.
    pub stdin: Option<String>,
    /// Execution timeout in milliseconds.
    pub timeout_ms: u64,
    /// Working directory for the child process.
    pub working_dir: PathBuf,
    /// Filesystem paths the process may **read** (system libraries are always included).
    pub readable_paths: Vec<PathBuf>,
    /// Filesystem paths the process may **read and write**.
    pub writable_paths: Vec<PathBuf>,
}

/// The result of running a process inside a sandbox.
pub struct SandboxOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub timed_out: bool,
}

impl SandboxOutput {
    pub fn success(&self) -> bool {
        !self.timed_out && self.exit_code == 0
    }
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// A pluggable execution backend for running processes with configurable
/// isolation guarantees.
#[async_trait]
pub trait SandboxExecutor: Send + Sync {
    /// Execute a process according to the provided [`SandboxRequest`].
    async fn execute(&self, req: SandboxRequest) -> Result<SandboxOutput, SandboxError>;

    /// Human-readable name of this sandbox implementation.
    fn name(&self) -> &str;
}
