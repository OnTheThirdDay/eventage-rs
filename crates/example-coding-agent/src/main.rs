//! example-coding-agent — enterprise-grade AI coding agent.
//!
//! Demonstrates the full Eventage feature set in a single cohesive example:
//!
//! - **Streaming LLM** — tokens published as events, live TUI typing effect
//! - **ratatui TUI** — modes: Idle / Working / AwaitingApproval
//! - **SecurityGateHook** — intercepts dangerous tools, event-driven approval
//! - **CompactingContextAssembler** — auto-summarises long conversations
//! - **TurnDiffWorker** — per-turn file-change diffs published as events
//! - **SQLite persistence** — full event log saved and resumable across runs
//! - **Sandbox integration** — shell execution with Landlock / Docker / none
//!
//! # Quick start
//!
//! ```sh
//! # With Ollama running locally:
//! cargo run -p example-coding-agent -- --model qwen3:4b
//!
//! # With OpenAI:
//! cargo run -p example-coding-agent -- \
//!   --url https://api.openai.com/v1 \
//!   --api-key sk-... \
//!   --model gpt-4o
//!
//! # Resume a previous session:
//! cargo run -p example-coding-agent -- --resume <SESSION_ID>
//!
//! # List saved sessions:
//! cargo run -p example-coding-agent -- --list-sessions
//! ```

mod diff;
mod kinds;
mod memory;
mod security;
mod streaming;
mod tools;
mod tui;
mod workspace;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use eventage::sandbox::{DockerExecutor, SandboxExecutor, UnsandboxedExecutor};
use eventage::sqlite::{SqliteEventStore, SqliteExporter};
use eventage::BusObserver;
use eventage::{agent::WorkerSet, AgentBuilder, ReactStrategy};
use eventage::{BusConfig, EventBus};
use tracing::info;

use diff::TurnDiffWorker;
use memory::CompactingContextAssembler;
use security::SecurityGateHook;
use streaming::StreamingOpenAiProvider;
use tools::{ApplyPatch, ExecuteShell, ListDir, ReadFile, WriteFile};
use workspace::Workspace;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "coding-agent",
    about = "Enterprise-grade AI coding agent powered by Eventage",
    long_about = "An event-driven coding agent with streaming LLM output, \
                 a ratatui TUI, security-gate approval, context compaction, \
                 and turn-level diff tracking. Sessions are persisted to SQLite."
)]
struct Args {
    /// LLM model name.
    #[arg(short, long, default_value = "qwen3:4b")]
    model: String,

    /// LLM provider base URL (OpenAI-compatible).
    #[arg(short = 'u', long, default_value = "http://localhost:11434/v1")]
    url: String,

    /// API key (`ollama` for local Ollama, or your OpenAI key).
    #[arg(short = 'k', long, default_value = "ollama")]
    api_key: String,

    /// Resume a previous session by its ID.
    #[arg(long, value_name = "SESSION_ID")]
    resume: Option<String>,

    /// List saved sessions and exit.
    #[arg(long)]
    list_sessions: bool,

    /// Context token budget (approx). Compaction triggers at 85% of this.
    #[arg(long, default_value = "120000")]
    max_tokens: usize,

    /// Number of recent conversation messages to keep after compaction.
    #[arg(long, default_value = "20")]
    recent_window: usize,

    /// Shell execution timeout in milliseconds.
    #[arg(long, default_value = "15000")]
    exec_timeout: u64,

    /// Sandbox mode: none | landlock (Linux) | docker.
    #[arg(long, default_value = "none")]
    sandbox: String,

    /// Docker image for --sandbox docker.
    #[arg(long, default_value = "python:3.12-slim")]
    docker_image: String,

    /// Max ReAct loop steps per conversation turn.
    #[arg(long, default_value = "30")]
    max_steps: usize,

    /// Tracing log level (written to ~/.coding-agent/logs/).
    #[arg(long, default_value = "info")]
    log_level: String,
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Initialise tracing to a file so TUI output is not polluted.
    if let Ok(log_dir) = agent_data_dir().map(|d| d.join("logs")) {
        let _ = std::fs::create_dir_all(&log_dir);
        // Best-effort: if we can't open the file, just use stderr.
        let log_file = log_dir.join("coding-agent.log");
        if let Ok(file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_file)
        {
            tracing_subscriber::fmt()
                .with_env_filter(&args.log_level)
                .with_writer(move || file.try_clone().expect("log file clone"))
                .init();
        } else {
            tracing_subscriber::fmt()
                .with_env_filter(&args.log_level)
                .with_writer(std::io::stderr)
                .init();
        }
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(&args.log_level)
            .with_writer(std::io::stderr)
            .init();
    }

    if let Err(e) = run(args).await {
        eprintln!("\nerror: {e:#}");
        std::process::exit(1);
    }
}

async fn run(args: Args) -> Result<()> {
    let data_dir = agent_data_dir()?;
    let sessions_dir = data_dir.join("sessions");
    std::fs::create_dir_all(&sessions_dir)?;

    // ── --list-sessions ───────────────────────────────────────────────────────
    if args.list_sessions {
        list_sessions(&sessions_dir)?;
        return Ok(());
    }

    // ── Session (SQLite) ──────────────────────────────────────────────────────
    let session_id = args
        .resume
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let db_path = sessions_dir.join(format!("{session_id}.db"));
    // ── Event bus ─────────────────────────────────────────────────────────────
    let bus = EventBus::with_config(BusConfig {
        // Observability subscribers (SQLite, TUI) get a bounded queue so
        // they never cause unbounded memory growth.
        subscriber_capacity: 2048,
        max_retained_branches: 10,
        ..BusConfig::default()
    });

    // Restore previous session events if resuming.
    if args.resume.is_some() {
        let store = SqliteEventStore::new(&db_path)
            .await
            .with_context(|| format!("could not open session database at {db_path:?}"))?;
        let events = store.load_all().await?;
        let count = events.len();
        bus.restore_from(events).await;
        info!(count, session_id, "restored session events");
    }

    // ── Workspace ─────────────────────────────────────────────────────────────
    let workspace_dir = sessions_dir.join(&session_id).join("workspace");
    let workspace = Arc::new(Workspace::open(&workspace_dir)?);

    // ── Sandbox executor ──────────────────────────────────────────────────────
    let executor: Arc<dyn SandboxExecutor> = build_executor(&args).await?;

    // ── Streaming LLM provider ────────────────────────────────────────────────
    let llm = StreamingOpenAiProvider::new(&args.url, &args.api_key, &args.model, bus.clone());
    let cancelled = llm.cancelled.clone();
    let llm = Arc::new(llm);

    // ── Context assembler (compacting) ────────────────────────────────────────
    let assembler = CompactingContextAssembler::new(
        system_prompt(),
        args.max_tokens,
        llm.clone(),
        bus.clone(),
        workspace.clone(),
    )
    .with_recent_window(args.recent_window);

    // ── Security gate hook ────────────────────────────────────────────────────
    let security_hook = SecurityGateHook::new(bus.clone());

    // ── Agent ─────────────────────────────────────────────────────────────────
    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm_arc(llm)
        .context(assembler)
        .hook(security_hook)
        .strategy(ReactStrategy {
            max_steps: args.max_steps,
            max_concurrent_tools: 4,
        })
        .tool(ReadFile {
            workspace: workspace.clone(),
        })
        .tool(WriteFile {
            workspace: workspace.clone(),
        })
        .tool(ApplyPatch {
            workspace: workspace.clone(),
        })
        .tool(ExecuteShell {
            workspace: workspace.clone(),
            executor: executor.clone(),
            default_timeout_ms: args.exec_timeout,
        })
        .tool(ListDir {
            workspace: workspace.clone(),
        })
        .build();

    info!(
        session_id,
        model = args.model,
        sandbox = args.sandbox,
        "agent ready"
    );

    // ── Background workers ────────────────────────────────────────────────────

    // Turn diff worker — publishes file change diffs.
    let diff_worker = TurnDiffWorker::new(workspace.clone());

    // SQLite exporter — persists every event to the session database.
    let sqlite_exporter = SqliteExporter::new(&db_path)
        .await
        .with_context(|| format!("could not open SQLite exporter at {db_path:?}"))?;

    let workers_bus = bus.clone();
    tokio::spawn(async move {
        WorkerSet::new()
            .add_worker(diff_worker)
            .run_on(workers_bus)
            .await
            .ok();
    });

    // Observability: SQLite export runs alongside TUI.
    let obs_bus = bus.clone();
    tokio::spawn(async move {
        BusObserver::new(obs_bus)
            .add_exporter(sqlite_exporter)
            .run()
            .await;
    });

    // ── Agent event loop ──────────────────────────────────────────────────────
    tokio::spawn(async move {
        if let Err(e) = agent.run().await {
            tracing::error!("agent error: {e}");
        }
    });

    // ── TUI ───────────────────────────────────────────────────────────────────
    tui::run_tui(bus, args.model.clone(), session_id.clone(), cancelled).await?;

    println!("\nSession saved: {session_id}");
    println!("Resume with:  coding-agent --resume {session_id}");

    Ok(())
}

// ── Session listing ───────────────────────────────────────────────────────────

fn list_sessions(dir: &std::path::Path) -> Result<()> {
    let mut sessions: Vec<(String, std::time::SystemTime)> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".db") {
                let id = name.trim_end_matches(".db").to_string();
                let modified = entry
                    .metadata()?
                    .modified()
                    .unwrap_or(std::time::UNIX_EPOCH);
                sessions.push((id, modified));
            }
        }
    }
    sessions.sort_by(|a, b| b.1.cmp(&a.1));

    if sessions.is_empty() {
        println!("No saved sessions found.");
    } else {
        println!("{:<36}  LAST USED", "SESSION ID");
        println!("{}", "─".repeat(60));
        for (id, modified) in &sessions {
            let dt: chrono::DateTime<chrono::Utc> = (*modified).into();
            println!("{id:<36}  {}", dt.format("%Y-%m-%d %H:%M UTC"));
        }
    }
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn agent_data_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("could not determine home dir"))?;
    Ok(home.join(".coding-agent"))
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
        "landlock" => anyhow::bail!("Landlock is Linux-only. Use --sandbox none or docker."),

        "docker" => {
            let exec = DockerExecutor::new(&args.docker_image);
            exec.check().await.with_context(|| {
                format!("Docker pre-flight failed (image: {})", args.docker_image)
            })?;
            Ok(Arc::new(exec) as Arc<dyn SandboxExecutor>)
        }

        other => {
            anyhow::bail!("unknown sandbox mode '{other}'. Valid: none, landlock (Linux), docker.")
        }
    }
}

// ── System prompt ─────────────────────────────────────────────────────────────

fn system_prompt() -> String {
    r#"You are an expert software engineer with direct access to a real filesystem and shell.

TOOLS AVAILABLE
  read_file(path)
    Read a file from the workspace. Always read before editing.

  write_file(path, content)
    Create or overwrite a file. Use for new files or complete rewrites.
    Requires user approval.

  apply_patch(path, patch)
    Apply a unified diff to make targeted edits without rewriting the full file.
    Requires user approval.

  execute_shell(command, stdin?, timeout_ms?)
    Run any shell command. Working dir is the workspace root.
    Requires user approval.

  list_dir()
    List all workspace files and their sizes.

WORKFLOW
  1. Understand the task — read relevant files first.
  2. Plan — outline what you'll do before writing code.
  3. Implement — write or patch files.
  4. Test — run the code and observe actual output.
  5. Iterate — fix any errors until tests pass.
  6. Report — summarise what was done and show key outputs.

STANDARDS
  - Write production-quality code with error handling.
  - Use the language's idiomatic patterns and standard library.
  - Prefer apply_patch for small targeted changes, write_file for new files.
  - Always run and test code — never assume it works.
  - Report the actual stdout/stderr from execute_shell in your final response.
"#
    .to_string()
}
