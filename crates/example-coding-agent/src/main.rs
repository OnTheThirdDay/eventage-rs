//! coding-agent — agentic capabilities with streaming TUI, security gate, and diff tracking.
//!
//! # Quick start
//!
//! ```sh
//! # With Ollama running locally (TUI mode):
//! cargo run -p example-coding-agent -- --model qwen3:4b
//!
//! # With OpenAI:
//! cargo run -p example-coding-agent -- \
//!   --url https://api.openai.com/v1 --api-key sk-... --model gpt-4o
//!
//! # REPL fallback (no TUI):
//! cargo run -p example-coding-agent -- --no-tui --model qwen3:4b
//!
//! # Resume a previous session:
//! cargo run -p example-coding-agent -- --session-file session.jsonl
//! ```

mod agent;
mod assembler;
mod error;
mod hooks;
mod kinds;
mod prompt;
mod streaming;
mod tools;
mod tui;
mod workers;
mod workspace;

use agent::CodingAgentBuilder;
use clap::Parser;
use eventage::event::kinds as core_kinds;
use eventage::observability::{BusObserver, JsonlExporter};
use eventage::replay::LiveReplayServer;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(
    name = "coding-agent",
    about = "Coding Agent — agentic capabilities with streaming TUI, security gate, and diff tracking",
    long_about = None
)]
struct Args {
    /// LLM base URL (OpenAI-compatible)
    #[arg(long, default_value = "http://localhost:11434/v1")]
    url: String,

    /// API key
    #[arg(long, default_value = "ollama")]
    api_key: String,

    /// Model name
    #[arg(long, short, default_value = "qwen3:4b")]
    model: String,

    /// Custom system prompt prefix
    #[arg(long)]
    system_prompt: Option<String>,

    /// Max ReAct steps per cycle
    #[arg(long, default_value_t = 30)]
    max_steps: usize,

    /// Approximate token budget for conversation summarization (0 = disabled)
    #[arg(long, default_value_t = 120_000)]
    max_tokens: usize,

    /// Path(s) to AGENTS.md memory files
    #[arg(long = "memory")]
    memory: Vec<PathBuf>,

    /// Directory(ies) containing SKILL.md skill files
    #[arg(long = "skills")]
    skills: Vec<PathBuf>,

    /// Working directory for filesystem tools (default: current directory)
    #[arg(long)]
    work_dir: Option<PathBuf>,

    /// Tool names that require human approval (REPL mode only)
    #[arg(long = "approve")]
    human_approval: Vec<String>,

    /// Require human approval before executing ANY tool call
    #[arg(long = "require-approve-all", default_value_t = false)]
    require_approve_all: bool,

    /// Disable async sub-agent worker
    #[arg(long, default_value_t = false)]
    no_async_subagents: bool,

    /// Max LLM requests per minute (0 = unlimited)
    #[arg(long, default_value_t = 0)]
    rpm: u32,

    /// Write all bus events to a JSONL file for replay/inspection
    #[arg(long)]
    log: Option<PathBuf>,

    /// Save/restore conversation history (JSONL format)
    #[arg(long)]
    session_file: Option<PathBuf>,

    /// Start a live replay UI server
    #[arg(long, default_value_t = false)]
    replay: bool,

    /// Port for the live replay UI server
    #[arg(long, default_value_t = 4567)]
    replay_port: u16,

    /// Use REPL mode instead of the TUI (no streaming, stdin/stdout interaction)
    #[arg(long, default_value_t = false)]
    no_tui: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let tui_mode = !args.no_tui;

    if tui_mode {
        // In TUI mode, log to file to keep the terminal clean.
        setup_file_tracing();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
            )
            .with_target(false)
            .init();

        eprintln!(
            "Coding Agent starting (model: {}, url: {})",
            args.model, args.url
        );
        if args.require_approve_all {
            eprintln!("Approval required for: all tools (--require-approve-all)");
        } else if !args.human_approval.is_empty() {
            eprintln!("Approval required for: {}", args.human_approval.join(", "));
        }
    }

    let agent = CodingAgentBuilder::new()
        .model(&args.url, &args.api_key, &args.model)
        .system_prompt_opt(args.system_prompt)
        .max_steps(args.max_steps)
        .max_tokens(args.max_tokens)
        .memory(args.memory)
        .skills(args.skills)
        .work_dir_opt(args.work_dir)
        .human_approval_for(args.human_approval)
        .require_approve_all(args.require_approve_all)
        .async_subagents(!args.no_async_subagents)
        .requests_per_minute(args.rpm)
        .tui_mode(tui_mode)
        .build();

    let bus = agent.bus().clone();
    let model = agent.model.clone();
    let session_id = agent.session_id.clone();
    let cancelled = agent
        .cancelled
        .clone()
        .unwrap_or_else(|| Arc::new(std::sync::atomic::AtomicBool::new(false)));

    // Restore previous session if --session-file is set
    if let Some(ref sf) = args.session_file {
        if sf.exists() {
            match load_session(&bus, sf).await {
                Ok(n) => {
                    if !tui_mode {
                        eprintln!("Resumed {} events from {}", n, sf.display());
                    }
                }
                Err(e) => eprintln!("Warning: could not load session: {e}"),
            }
        } else if !tui_mode {
            eprintln!("New session — will save to {}", sf.display());
        }
    }

    // Start live replay server if requested
    if args.replay {
        LiveReplayServer::new(bus.clone())
            .port(args.replay_port)
            .serve_background();
        if !tui_mode {
            eprintln!("Live replay UI: http://localhost:{}", args.replay_port);
        }
    }

    // Start observability exporter if --log was specified
    if let Some(log_path) = args.log {
        match JsonlExporter::new(&log_path).await {
            Ok(exporter) => {
                let observer = BusObserver::new(bus.clone()).add_exporter(exporter);
                tokio::spawn(observer.run());
                if !tui_mode {
                    eprintln!("Logging events to {}", log_path.display());
                }
            }
            Err(e) => eprintln!("Warning: could not open log file: {e}"),
        }
    }

    if tui_mode {
        // ── TUI mode ─────────────────────────────────────────────────────────
        tokio::spawn(async move {
            if let Err(e) = agent.run().await {
                tracing::error!("agent error: {e}");
            }
        });

        tui::run_tui(bus, model, session_id, cancelled).await?;
    } else if !args.no_async_subagents {
        // ── Reactive REPL ─────────────────────────────────────────────────────
        tokio::spawn(async move {
            if let Err(e) = agent.run().await {
                eprintln!("agent error: {e}");
            }
        });
        run_reactive_repl(bus, args.session_file).await?;
    } else {
        // ── Sync REPL ─────────────────────────────────────────────────────────
        run_sync_repl(agent, args.session_file).await?;
    }

    Ok(())
}

// ── Tracing setup ─────────────────────────────────────────────────────────────

fn setup_file_tracing() {
    let log_dir = dirs::home_dir()
        .map(|h| h.join(".coding-agent").join("logs"))
        .unwrap_or_else(|| PathBuf::from("/tmp/coding-agent-logs"));

    let _ = std::fs::create_dir_all(&log_dir);
    let log_file = log_dir.join("coding-agent.log");

    if let Ok(file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file)
    {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .with_writer(move || file.try_clone().expect("log file clone"))
            .try_init();
    } else {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::new("warn"))
            .with_writer(std::io::stderr)
            .try_init();
    }
}

// ── Session persistence ───────────────────────────────────────────────────────

async fn load_session(bus: &eventage::EventBus, path: &Path) -> anyhow::Result<usize> {
    let content = tokio::fs::read_to_string(path).await?;
    let mut count = 0;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let event: eventage::Event = serde_json::from_str(line)?;
        bus.publish(event).await?;
        count += 1;
    }
    Ok(count)
}

async fn save_session(bus: &eventage::EventBus, path: &Path) {
    let log = bus.log().await;
    let result = async {
        let mut content = String::new();
        for event in &log {
            if matches!(
                event.kind.as_str(),
                core_kinds::USER_MESSAGE | core_kinds::ASSISTANT_MESSAGE | core_kinds::TOOL_RESULT
            ) {
                content.push_str(&serde_json::to_string(event)?);
                content.push('\n');
            }
        }
        tokio::fs::write(path, content).await?;
        Ok::<(), anyhow::Error>(())
    }
    .await;

    if let Err(e) = result {
        eprintln!("Warning: could not save session: {e}");
    }
}

// ── REPL variants ─────────────────────────────────────────────────────────────

/// Reactive REPL: publishes USER_MESSAGE events; background agent processes them.
async fn run_reactive_repl(
    bus: eventage::EventBus,
    session_file: Option<PathBuf>,
) -> anyhow::Result<()> {
    use eventage::event::Event;
    use serde_json::json;

    eprintln!("Ready. Type your message and press Enter. (Ctrl+C to exit)\n");

    loop {
        // Read one line using rustyline for full readline editing.
        let line = tokio::task::spawn_blocking(|| {
            match rustyline::DefaultEditor::new() {
                Ok(mut rl) => rl.readline("").ok(),
                Err(_) => {
                    // Fallback to raw read_line
                    let mut s = String::new();
                    match io::stdin().lock().read_line(&mut s) {
                        Ok(0) => None,
                        Ok(_) => Some(s),
                        Err(_) => None,
                    }
                }
            }
        })
        .await?;

        let Some(raw) = line else { break };
        let trimmed = raw.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }

        // Subscribe BEFORE publishing so we catch every event of this cycle.
        let mut rx = bus.subscribe();

        bus.publish(Event::new(
            core_kinds::USER_MESSAGE,
            json!({ "text": trimmed }),
        ))
        .await?;

        drain_cycle(&mut rx).await;

        if let Some(ref sf) = session_file {
            save_session(&bus, sf).await;
        }
    }

    Ok(())
}

/// Synchronous REPL: calls agent.chat() directly for each message.
async fn run_sync_repl(
    agent: agent::CodingAgent,
    session_file: Option<PathBuf>,
) -> anyhow::Result<()> {
    let bus = agent.bus().clone();

    eprintln!("Ready. Type your message and press Enter. (Ctrl+C to exit)\n");

    loop {
        let line = tokio::task::spawn_blocking(|| match rustyline::DefaultEditor::new() {
            Ok(mut rl) => rl.readline("").ok(),
            Err(_) => {
                let mut s = String::new();
                match io::stdin().lock().read_line(&mut s) {
                    Ok(0) => None,
                    Ok(_) => Some(s),
                    Err(_) => None,
                }
            }
        })
        .await?;

        let Some(raw) = line else { break };
        let trimmed = raw.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }

        // Stream intermediate events while chat() runs.
        let mut rx = bus.subscribe();
        let display = tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                print_event(&event);
            }
        });

        match agent.chat(&trimmed).await {
            Ok(response) => {
                display.abort();
                if !response.is_empty() {
                    eprintln!("\nAssistant: {response}\n");
                }
            }
            Err(e) => {
                display.abort();
                eprintln!("error: {e}");
            }
        }

        if let Some(ref sf) = session_file {
            save_session(&bus, sf).await;
        }
    }

    Ok(())
}

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Drain bus events until the ReAct cycle fully completes.
async fn drain_cycle(rx: &mut eventage::BusReceiver) {
    let mut cycle_started = false;
    while let Some(event) = rx.recv().await {
        if event.kind == core_kinds::AGENT_CYCLE_START {
            cycle_started = true;
            continue;
        }
        if !cycle_started {
            continue;
        }
        if print_event(&event) {
            return;
        }
        if event.kind == core_kinds::AGENT_CYCLE_END {
            return;
        }
    }
}

/// Print a single bus event to the terminal. Returns true when the cycle is done.
fn print_event(event: &eventage::Event) -> bool {
    if event.kind == core_kinds::TOOL_CALL_PROPOSED {
        let name = event.payload["name"].as_str().unwrap_or("?");
        let args = &event.payload["arguments"];
        let args_str = args.to_string();
        let args_display = if args_str.len() > 120 {
            format!("{}…", &args_str[..120])
        } else {
            args_str
        };
        eprintln!("[→ {name}] {args_display}");
    } else if event.kind == core_kinds::TOOL_RESULT {
        let name = event.payload["name"].as_str().unwrap_or("?");
        if let Some(err) = event.payload.get("error") {
            eprintln!("[← {name} ERR] {err}");
        } else if let Some(result) = event.payload.get("result") {
            let s = result.to_string();
            let preview = if s.len() > 200 {
                format!("{}…", &s[..200])
            } else {
                s
            };
            eprintln!("[← {name}] {preview}");
        }
    } else if event.kind == core_kinds::ASSISTANT_MESSAGE {
        let content = event.payload["content"].as_str().unwrap_or("");
        if !content.is_empty() {
            eprintln!("\nAssistant: {content}\n");
        }
        let has_tool_calls = event.payload["tool_calls"]
            .as_array()
            .map(|a| !a.is_empty())
            .unwrap_or(false);
        if !has_tool_calls {
            return true;
        }
    }
    false
}
