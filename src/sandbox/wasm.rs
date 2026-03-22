//! WASM-based sandboxed execution via wasmtime + WASI.

use super::{SandboxError, SandboxExecutor, SandboxOutput, SandboxRequest};
use async_trait::async_trait;
use wasmtime::{Config, Engine, Linker, Module, Store};
use wasmtime_wasi::{
    pipe::{MemoryInputPipe, MemoryOutputPipe},
    WasiCtxBuilder,
};

/// WASM-based sandbox executor.
///
/// Runs `.wasm` binaries in an isolated wasmtime environment with WASI support.
/// Creates a fresh WASM instance per `execute` call to prevent state leaks.
pub struct WasmExecutor {
    engine: Engine,
}

impl WasmExecutor {
    /// Create a new executor with default wasmtime configuration.
    pub fn new() -> Self {
        let mut config = Config::new();
        config.async_support(false);
        let engine = Engine::new(&config).expect("failed to create wasmtime engine");
        Self { engine }
    }
}

impl Default for WasmExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SandboxExecutor for WasmExecutor {
    fn name(&self) -> &str {
        "wasm"
    }

    async fn execute(&self, req: SandboxRequest) -> Result<SandboxOutput, SandboxError> {
        let engine = self.engine.clone();

        tokio::task::spawn_blocking(move || run_wasm(engine, req))
            .await
            .map_err(|e| SandboxError::Spawn(e.to_string()))?
    }
}

fn run_wasm(engine: Engine, req: SandboxRequest) -> Result<SandboxOutput, SandboxError> {
    let module = Module::from_file(&engine, &req.program)
        .map_err(|e| SandboxError::Setup(format!("failed to load wasm module: {e}")))?;

    let stdout_pipe = MemoryOutputPipe::new(usize::MAX);
    let stderr_pipe = MemoryOutputPipe::new(usize::MAX);

    let mut builder = WasiCtxBuilder::new();
    builder.stdout(stdout_pipe.clone());
    builder.stderr(stderr_pipe.clone());

    if let Some(stdin_data) = req.stdin {
        builder.stdin(MemoryInputPipe::new(stdin_data.into_bytes()));
    }

    for (key, val) in &req.env {
        builder.env(key, val);
    }

    builder.arg(&req.program);
    for arg in &req.args {
        builder.arg(arg);
    }

    for path in &req.writable_paths {
        if path.exists() {
            builder
                .preopened_dir(
                    path,
                    path.to_string_lossy().as_ref(),
                    wasmtime_wasi::DirPerms::all(),
                    wasmtime_wasi::FilePerms::all(),
                )
                .map_err(|e| SandboxError::Setup(format!("preopened dir error: {e}")))?;
        }
    }
    for path in &req.readable_paths {
        if path.exists() {
            builder
                .preopened_dir(
                    path,
                    path.to_string_lossy().as_ref(),
                    wasmtime_wasi::DirPerms::READ,
                    wasmtime_wasi::FilePerms::READ,
                )
                .map_err(|e| SandboxError::Setup(format!("preopened dir error: {e}")))?;
        }
    }

    let wasi_ctx = builder.build_p1();

    let mut store = Store::new(&engine, wasi_ctx);

    let fuel_limit = req.timeout_ms.saturating_mul(1_000_000);
    store.set_fuel(fuel_limit).ok();

    let mut linker: Linker<wasmtime_wasi::preview1::WasiP1Ctx> = Linker::new(&engine);
    wasmtime_wasi::preview1::add_to_linker_sync(&mut linker, |s| s)
        .map_err(|e| SandboxError::Setup(format!("failed to add wasi to linker: {e}")))?;

    let instance = linker
        .instantiate(&mut store, &module)
        .map_err(|e| SandboxError::Spawn(format!("failed to instantiate wasm module: {e}")))?;

    let start = instance
        .get_typed_func::<(), ()>(&mut store, "_start")
        .map_err(|e| SandboxError::Spawn(format!("_start not found in wasm module: {e}")))?;

    let timed_out = match start.call(&mut store, ()) {
        Ok(()) => false,
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("fuel") || msg.contains("Timeout") {
                true
            } else if msg.contains("Exited with i32 exit status 0") || msg.contains("proc_exit(0)")
            {
                false
            } else {
                return Err(SandboxError::Spawn(msg));
            }
        }
    };

    let stdout = String::from_utf8_lossy(&stdout_pipe.contents()).into_owned();
    let stderr = String::from_utf8_lossy(&stderr_pipe.contents()).into_owned();

    Ok(SandboxOutput {
        stdout,
        stderr,
        exit_code: if timed_out { -1 } else { 0 },
        timed_out,
    })
}
