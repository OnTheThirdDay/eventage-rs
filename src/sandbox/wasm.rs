//! WASM-based sandboxed execution via wasmtime + WASI.

use super::{SandboxError, SandboxExecutor, SandboxOutput, SandboxRequest};
use async_trait::async_trait;
use wasmtime::{Config, Engine, Linker, Module, Store, StoreLimits, StoreLimitsBuilder};
use wasmtime_wasi::{
    pipe::{MemoryInputPipe, MemoryOutputPipe},
    WasiCtxBuilder,
};

/// Maximum bytes captured from guest stdout/stderr (each).
const MAX_PIPE_BYTES: usize = 8 * 1024 * 1024;

/// Maximum linear memory a guest instance may grow to.
const MAX_GUEST_MEMORY_BYTES: usize = 512 * 1024 * 1024;

/// WASM-based sandbox executor.
///
/// Runs `.wasm` binaries in an isolated wasmtime environment with WASI support.
/// Creates a fresh WASM instance per `execute` call to prevent state leaks.
/// Guests are bounded: linear memory is capped at 512 MiB, captured
/// stdout/stderr at 8 MiB each, and CPU via a fuel budget derived from
/// `timeout_ms` (note: fuel bounds *computation*, not wall-clock time).
pub struct WasmExecutor {
    engine: Engine,
}

/// Store data: WASI context plus resource limits enforced by wasmtime.
struct HostState {
    wasi: wasmtime_wasi::preview1::WasiP1Ctx,
    limits: StoreLimits,
}

impl WasmExecutor {
    /// Create a new executor with default wasmtime configuration.
    pub fn new() -> Self {
        let mut config = Config::new();
        config.async_support(false);
        // Required for `Store::set_fuel` to actually enforce the CPU budget —
        // without this the fuel limit is silently ignored.
        config.consume_fuel(true);
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

    let stdout_pipe = MemoryOutputPipe::new(MAX_PIPE_BYTES);
    let stderr_pipe = MemoryOutputPipe::new(MAX_PIPE_BYTES);

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

    let limits = StoreLimitsBuilder::new()
        .memory_size(MAX_GUEST_MEMORY_BYTES)
        .build();
    let mut store = Store::new(
        &engine,
        HostState {
            wasi: wasi_ctx,
            limits,
        },
    );
    store.limiter(|s| &mut s.limits);

    let fuel_limit = req.timeout_ms.saturating_mul(1_000_000);
    store.set_fuel(fuel_limit).ok();

    let mut linker: Linker<HostState> = Linker::new(&engine);
    wasmtime_wasi::preview1::add_to_linker_sync(&mut linker, |s| &mut s.wasi)
        .map_err(|e| SandboxError::Setup(format!("failed to add wasi to linker: {e}")))?;

    let instance = linker
        .instantiate(&mut store, &module)
        .map_err(|e| SandboxError::Spawn(format!("failed to instantiate wasm module: {e}")))?;

    let start = instance
        .get_typed_func::<(), ()>(&mut store, "_start")
        .map_err(|e| SandboxError::Spawn(format!("_start not found in wasm module: {e}")))?;

    let (exit_code, timed_out) = match start.call(&mut store, ()) {
        Ok(()) => (0, false),
        Err(e) => classify_wasm_error(e)?,
    };

    let stdout = String::from_utf8_lossy(&stdout_pipe.contents()).into_owned();
    let stderr = String::from_utf8_lossy(&stderr_pipe.contents()).into_owned();

    Ok(SandboxOutput {
        stdout,
        stderr,
        exit_code,
        timed_out,
    })
}

/// Map a wasmtime execution error to `(exit_code, timed_out)`.
///
/// A guest calling `proc_exit(N)` surfaces as an `I32Exit(N)` error — that is
/// a *normal program exit*, not a sandbox failure, and must be reported as
/// `exit_code: N` so build/test workflows can branch on it. Running out of
/// fuel is reported as a timeout. Anything else is a real failure.
fn classify_wasm_error(e: wasmtime::Error) -> Result<(i32, bool), SandboxError> {
    if let Some(exit) = e.downcast_ref::<wasmtime_wasi::I32Exit>() {
        return Ok((exit.0, false));
    }
    if let Some(trap) = e.downcast_ref::<wasmtime::Trap>() {
        if *trap == wasmtime::Trap::OutOfFuel {
            return Ok((-1, true));
        }
    }
    Err(SandboxError::Spawn(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proc_exit_maps_to_exit_code() {
        let ok = classify_wasm_error(wasmtime::Error::new(wasmtime_wasi::I32Exit(0))).unwrap();
        assert_eq!(ok, (0, false));
        let fail = classify_wasm_error(wasmtime::Error::new(wasmtime_wasi::I32Exit(1))).unwrap();
        assert_eq!(fail, (1, false), "non-zero exit is a result, not an error");
    }

    #[test]
    fn out_of_fuel_maps_to_timeout() {
        let res = classify_wasm_error(wasmtime::Error::new(wasmtime::Trap::OutOfFuel)).unwrap();
        assert_eq!(res, (-1, true));
    }

    #[test]
    fn real_traps_are_errors() {
        let res = classify_wasm_error(wasmtime::Error::new(wasmtime::Trap::UnreachableCodeReached));
        assert!(res.is_err());
    }
}
