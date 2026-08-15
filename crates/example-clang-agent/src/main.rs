mod context;
mod display;
mod permissions;
mod session;
mod tools;
mod workspace;

use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use eventage::agent::AgentBuilder;
use eventage::llm::OpenAiProvider;
use eventage::sandbox::{DockerExecutor, SandboxExecutor, UnsandboxedExecutor};
use eventage::{kinds, Event, EventBus};
use serde_json::json;
use tokio::io::AsyncBufReadExt;
use tracing::debug;

use context::CAgentContextAssembler;
use permissions::{AutoApproveGate, PermissionGate, StdinPermissionGate};
use session::{Session, SessionMeta};
use tools::{Compile, Execute, ListFiles, ReadFile, WriteFile};
use workspace::Workspace;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "example-clang-agent",
    about = "Production-grade AI assistant for C programming",
    long_about = "An event-driven AI agent that writes, compiles, and runs C code \
                 interactively via a local or remote LLM."
)]
struct Args {
    /// LLM model name.
    #[arg(short, long, default_value = "qwen3:4b-cw")]
    model: String,

    /// LLM provider base URL (OpenAI-compatible).
    #[arg(short = 'u', long, default_value = "http://localhost:11434/v1")]
    url: String,

    /// API key (use 'ollama' for local Ollama).
    #[arg(short = 'k', long, default_value = "ollama")]
    api_key: String,

    /// Resume a previous session by ID.
    #[arg(short, long, value_name = "SESSION_ID")]
    session: Option<String>,

    /// List saved sessions and exit.
    #[arg(long)]
    list_sessions: bool,

    /// Maximum number of conversation messages kept in the LLM context window.
    #[arg(long, default_value = "40")]
    window: usize,

    /// Binary execution timeout in milliseconds.
    #[arg(long, default_value = "10000")]
    exec_timeout: u64,

    /// tool_choice value sent to the LLM API. Common values: auto, required, none.
    #[arg(long, default_value = "auto")]
    tool_choice: String,

    /// Sandbox mode for compiler and binary execution.
    ///
    /// Choices:
    ///   none     — Direct execution, no isolation (default; fast, for trusted use).
    ///   landlock — Linux Landlock filesystem isolation (requires kernel 5.13+).
    ///   docker   — Full Docker container isolation (requires Docker to be installed).
    #[arg(long, default_value = "none", value_name = "MODE")]
    sandbox: String,

    /// Docker image to use when --sandbox docker is set.
    #[arg(long, default_value = "gcc:13")]
    docker_image: String,

    /// Tracing log level filter (e.g. "warn", "eventage=debug").
    #[arg(long, default_value = "warn")]
    log_level: String,
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(&args.log_level)
        .with_writer(std::io::stderr)
        .init();

    if let Err(e) = run(args).await {
        display::print_error(&e.to_string());
        std::process::exit(1);
    }
}

async fn run(args: Args) -> Result<()> {
    let sessions_root = sessions_root()?;

    // ── --list-sessions ───────────────────────────────────────────────────────
    if args.list_sessions {
        let metas = Session::list(&sessions_root)?;
        if metas.is_empty() {
            println!("No sessions found.");
        } else {
            println!("{:<36}  {:<20}  {:<16}", "SESSION ID", "CREATED", "MODEL");
            println!("{}", "─".repeat(72));
            for m in &metas {
                println!(
                    "{:<36}  {:<20}  {}",
                    m.id,
                    m.created_at.format("%Y-%m-%d %H:%M UTC"),
                    m.model
                );
            }
        }
        return Ok(());
    }

    // ── Preflight ─────────────────────────────────────────────────────────────
    check_gcc_available()?;

    // ── Sandbox executor + permission gate ────────────────────────────────────
    let executor: Arc<dyn SandboxExecutor> = build_executor(&args).await?;
    display::print_sandbox(executor.name());

    // When running inside a real sandbox the agent is auto-approved; otherwise
    // every dangerous operation requires explicit human confirmation.
    let gate: Arc<dyn PermissionGate> = if args.sandbox == "none" {
        Arc::new(StdinPermissionGate)
    } else {
        Arc::new(AutoApproveGate)
    };

    // ── Session + bus ─────────────────────────────────────────────────────────
    let bus = EventBus::new();
    let initial_cursor;

    let session = match args.session {
        Some(ref id) => {
            let s = Session::open(&sessions_root, id)
                .with_context(|| format!("could not open session '{id}'"))?;
            let n = s.load_events(&bus).await?;
            initial_cursor = bus.log_len().await;
            display::print_resumed(n);
            s
        }
        None => {
            initial_cursor = 0;
            let meta = SessionMeta {
                id: uuid::Uuid::new_v4().to_string(),
                created_at: chrono::Utc::now(),
                model: args.model.clone(),
                provider_url: args.url.clone(),
            };
            Session::create(&sessions_root, meta)?
        }
    };

    let session_id = session.meta.id.clone();
    let session = Arc::new(session);

    // ── Workspace ─────────────────────────────────────────────────────────────
    let workspace = Arc::new(Workspace::open(session.workspace_path())?);

    // ── Agent ─────────────────────────────────────────────────────────────────
    let llm = OpenAiProvider::new(&args.url, &args.api_key, &args.model)
        .with_tool_choice(json!(args.tool_choice));

    let assembler = CAgentContextAssembler {
        system_prompt: system_prompt(),
        max_messages: args.window,
        workspace: workspace.clone(),
    };

    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(llm)
        .context(assembler)
        .tool(WriteFile {
            workspace: workspace.clone(),
            gate: gate.clone(),
        })
        .tool(ReadFile {
            workspace: workspace.clone(),
        })
        .tool(ListFiles {
            workspace: workspace.clone(),
        })
        .tool(Compile {
            workspace: workspace.clone(),
            executor: executor.clone(),
            gate: gate.clone(),
        })
        .tool(Execute {
            workspace: workspace.clone(),
            executor: executor.clone(),
            gate: gate.clone(),
            default_timeout_ms: args.exec_timeout,
        })
        .build();

    display::print_banner(
        &session_id,
        &args.model,
        &workspace.root().display().to_string(),
    );

    // ── Shutdown signal ───────────────────────────────────────────────────────
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    {
        let tx = shutdown_tx.clone();
        tokio::spawn(async move {
            tokio::signal::ctrl_c().await.ok();
            let _ = tx.send(true);
        });
    }

    // ── Task 1: Agent loop ────────────────────────────────────────────────────
    // Run the agent. It subscribes to the bus and processes cycles sequentially
    // when `user.message` or `system.heartbeat` events arrive.
    {
        let mut sd = shutdown_rx.clone();
        tokio::spawn(async move {
            tokio::select! {
                r = agent.run() => { if let Err(e) = r { display::print_error(&e.to_string()); } }
                _ = sd.changed() => {}
            }
        });
    }

    // ── Task 2: Live display + session persistence ────────────────────────────
    // Subscribes to the bus to print live events and persist completed cycles.
    {
        let bus2 = bus.clone();
        let session2 = session.clone();
        let mut sd = shutdown_rx.clone();
        let mut save_cursor = initial_cursor;

        tokio::spawn(async move {
            let mut rx = bus2.subscribe();
            loop {
                let event = tokio::select! {
                    e = rx.recv() => match e {
                        Some(ev) => ev,
                        None => break,
                    },
                    _ = sd.changed() => break,
                };

                display::display_event_live(&event);

                // After each completed cycle, persist all new events to disk.
                if event.kind == kinds::AGENT_CYCLE_END {
                    if let Err(e) = session2.append_events(&bus2, save_cursor).await {
                        display::print_error(&format!("session save failed: {e}"));
                    }
                    save_cursor = bus2.log_len().await;
                }
            }
        });
    }

    // ── Task 3: Stdin publisher ───────────────────────────────────────────────
    // Reads stdin and publishes `user.message` events.
    // Waits for `agent.cycle.end` before prompting again to prevent overlapping cycles.
    // Subscribes to the bus before publishing to ensure events are not missed.
    let mut stdin = tokio::io::BufReader::new(tokio::io::stdin()).lines();
    let mut sd = shutdown_rx.clone();

    loop {
        if *sd.borrow() {
            break;
        }

        display::print_prompt();

        let line = tokio::select! {
            l = stdin.next_line() => l?,
            _ = sd.changed() => break,
        };

        let input = match line {
            None => break,
            Some(s) if s.trim().is_empty() => continue,
            Some(s) => s.trim().to_string(),
        };

        if handle_repl_command(&input, &bus, &workspace) {
            continue;
        }

        display::print_thinking();

        // Subscribe BEFORE publishing so we cannot miss AGENT_CYCLE_END.
        let mut cycle_rx = bus.subscribe();

        bus.publish(Event::new(kinds::USER_MESSAGE, json!({ "text": input })))
            .await?;

        debug!("waiting for cycle to complete");

        // Wait for the cycle to finish.  We also listen for shutdown so Ctrl-C
        // during a long LLM call exits cleanly after the current network request
        // times out rather than hanging the process.
        loop {
            tokio::select! {
                event = cycle_rx.recv() => match event {
                    Some(e) if e.kind == kinds::AGENT_CYCLE_END => break,
                    Some(_) => continue,
                    None => { let _ = shutdown_tx.send(true); break; }
                },
                _ = sd.changed() => break,
            }
        }
    }

    display::print_farewell(&session_id);
    Ok(())
}

// ── Built-in REPL commands ────────────────────────────────────────────────────

fn handle_repl_command(input: &str, bus: &EventBus, workspace: &Workspace) -> bool {
    match input.trim() {
        "/help" => {
            println!(
                "\nBuilt-in commands:\n\
                 \x20 /help    — show this message\n\
                 \x20 /files   — list workspace files\n\
                 \x20 /log     — show recent event log\n\
                 \x20 exit     — exit the agent\n"
            );
            true
        }
        "/files" => {
            match workspace.list_files() {
                Ok(files) if files.is_empty() => println!("  (workspace is empty)"),
                Ok(files) => {
                    for f in &files {
                        println!("  {} ({} B)", f.path, f.size_bytes);
                    }
                }
                Err(e) => display::print_error(&e.to_string()),
            }
            println!();
            true
        }
        "/log" => {
            let log = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(bus.log())
            });
            println!();
            for event in log.iter().rev().take(12).rev() {
                println!(
                    "  {} [{}]  {}",
                    event.timestamp.format("%H:%M:%S"),
                    event.kind,
                    display::summarise_payload(&event.payload),
                );
            }
            println!();
            true
        }
        "exit" | "quit" | "/exit" | "/quit" => {
            std::process::exit(0);
        }
        _ => false,
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn sessions_root() -> Result<std::path::PathBuf> {
    let base = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("could not determine home directory"))?
        .join(".example-clang-agent")
        .join("sessions");
    std::fs::create_dir_all(&base)?;
    Ok(base)
}

fn check_gcc_available() -> Result<()> {
    std::process::Command::new("gcc")
        .arg("--version")
        .output()
        .map(|_| ())
        .map_err(|_| {
            anyhow::anyhow!(
                "gcc not found on PATH.\n  \
                 Ubuntu/Debian: sudo apt install gcc\n  \
                 macOS:         xcode-select --install"
            )
        })
}

async fn build_executor(args: &Args) -> Result<Arc<dyn SandboxExecutor>> {
    match args.sandbox.as_str() {
        "none" => Ok(Arc::new(UnsandboxedExecutor::new()) as Arc<dyn SandboxExecutor>),

        #[cfg(target_os = "linux")]
        "landlock" => {
            use eventage::sandbox::LandlockExecutor;
            Ok(Arc::new(LandlockExecutor::new()) as Arc<dyn SandboxExecutor>)
        }

        #[cfg(not(target_os = "linux"))]
        "landlock" => {
            anyhow::bail!(
                "Landlock is a Linux-only sandbox. Use --sandbox none or --sandbox docker."
            )
        }

        "docker" => {
            let exec = DockerExecutor::new(&args.docker_image);
            exec.check().await.with_context(|| {
                format!(
                    "Docker pre-flight check failed (image: {})",
                    args.docker_image
                )
            })?;
            Ok(Arc::new(exec) as Arc<dyn SandboxExecutor>)
        }

        other => anyhow::bail!(
            "unknown sandbox mode '{other}'. Valid modes: none, landlock (Linux only), docker."
        ),
    }
}

// ── System prompt ─────────────────────────────────────────────────────────────

fn system_prompt() -> String {
    r#"You are a C code execution agent with direct access to a real filesystem, compiler, and runtime.

CRITICAL RULE: You operate through tool calls, not through text. Writing C code in your text response has ZERO effect — no file is created, nothing is compiled, nothing runs. The ONLY way to create a program is to call write_file.

MANDATORY SEQUENCE for every coding request — no exceptions:
  Step 1.  write_file(path, content)   — write the complete C source to a file
  Step 2.  compile(source, output)     — compile with gcc; fix errors and retry if needed
  Step 3.  execute(binary)             — run the binary; observe actual output
  Step 4.  (text response)             — briefly summarise what you built and its observed output

Do NOT produce a text response before completing steps 1–3. Your text response is the final step, not the first.

TOOLS
  write_file(path, content)
    Creates or overwrites a file. path is relative to workspace root, e.g. "main.c".
    Call this every time you need to create or modify source code.

  read_file(path)
    Reads a file. Use before editing to see the current content.

  list_files()
    Lists all files in the workspace.

  compile(source, output, flags?)
    Compiles source with: gcc -Wall -Wextra -g <source> -o bin/<output> [flags]
    flags examples: ["-lm"], ["-lpthread"], ["-O2"], ["-std=c11"]
    On failure: read the error output, fix the source with write_file, then call compile again.
    Fix ALL errors in one write_file call, not one at a time.

  execute(binary, args?, stdin?, timeout_ms?)
    Runs bin/<binary> with cwd = workspace root.
    stdin: pass input as a string if the program reads from stdin.
    timeout_ms: default 10000; increase for slow programs.

CODE STANDARDS
  - All necessary #include directives (#include <stdio.h>, <stdlib.h>, etc.)
  - Check and handle return values of malloc, fopen, and other fallible calls
  - Free every heap allocation before exit
  - Use const for read-only pointer parameters
  - Use size_t for sizes and indices
  - Zero warnings with -Wall -Wextra (treat every warning as an error)
  - One-line comment above non-obvious functions

RESPONSE (step 4 only)
  - State the program's purpose and key implementation choice in 1–2 sentences
  - Quote the actual stdout from the execute result
  - If execution failed, state why and what you would fix next
"#.to_string()
}
