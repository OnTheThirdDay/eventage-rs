//! eventage-claw — personal AI assistant.
//!
//! # Quick start
//!
//! ```sh
//! # With Ollama (TUI mode):
//! cargo run -p example-eventage-claw -- --model qwen3:4b
//!
//! # With Anthropic (TUI mode):
//! ANTHROPIC_API_KEY=sk-... cargo run -p example-eventage-claw -- \
//!   --url https://api.anthropic.com/v1 --model claude-sonnet-4-6
//!
//! # REPL mode (no TUI):
//! cargo run -p example-eventage-claw -- --no-tui --model qwen3:4b
//!
//! # Two groups defined in ~/.claw/config.toml:
//! cargo run -p example-eventage-claw
//!
//! # HTTP channel (curl to send messages):
//! cargo run -p example-eventage-claw -- --http-port 3000 &
//! curl -X POST localhost:3000/message \
//!   -H 'Content-Type: application/json' \
//!   -d '{"group":"personal","text":"hello"}'
//!
//! # JSONL event log + live replay UI:
//! cargo run -p example-eventage-claw -- \
//!   --log ~/.claw/events.jsonl --replay
//! ```

mod agent;
mod assembler;
mod channels;
mod config;
mod error;
mod hooks;
mod kinds;
mod prompt;
mod streaming;
mod tools;
mod tui;
mod workers;

use agent::ClawAgentBuilder;
use channels::http::HttpChannelWorker;
use clap::Parser;
use config::ClawConfig;
use eventage::observability::{BusObserver, JsonlExporter};
use eventage::replay::LiveReplayServer;
use eventage::scheduler::HeartbeatScheduler;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tracing_subscriber::EnvFilter;

// ── CLI args ──────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "claw",
    about = "eventage-claw — personal AI assistant",
    long_about = None,
)]
struct Args {
    /// Config file path (default: ~/.claw/config.toml)
    #[arg(long)]
    config: Option<PathBuf>,

    /// LLM base URL (overrides config / env LLM_URL)
    #[arg(long)]
    url: Option<String>,

    /// API key (overrides config / env ANTHROPIC_API_KEY)
    #[arg(long)]
    api_key: Option<String>,

    /// Model name (overrides config / env LLM_MODEL)
    #[arg(long, short)]
    model: Option<String>,

    /// Max ReAct steps per cycle
    #[arg(long)]
    max_steps: Option<usize>,

    /// Token budget for summarization (0 = disabled)
    #[arg(long)]
    max_tokens: Option<usize>,

    /// Max LLM requests per minute (0 = unlimited)
    #[arg(long)]
    rpm: Option<u32>,

    /// Heartbeat interval in seconds for the task scheduler
    #[arg(long)]
    heartbeat_secs: Option<u64>,

    /// Start HTTP channel server on this port (accepts POST /message)
    #[arg(long)]
    http_port: Option<u16>,

    /// Write all bus events to a JSONL file for replay/inspection
    #[arg(long)]
    log: Option<PathBuf>,

    /// Start a live replay UI server (default port 4567)
    #[arg(long, default_value_t = false)]
    replay: bool,

    /// Port for the live replay UI server
    #[arg(long, default_value_t = 4567)]
    replay_port: u16,

    /// Use REPL mode instead of the TUI
    #[arg(long, default_value_t = false)]
    no_tui: bool,
}

// ── main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let tui_mode = !args.no_tui;

    if tui_mode {
        setup_file_tracing();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
            )
            .with_target(false)
            .init();
    }

    // ── Load config ───────────────────────────────────────────────────────────
    let mut config = ClawConfig::load(args.config.as_deref());

    // CLI flags override config values.
    if let Some(url) = args.url {
        config.llm_url = url;
    }
    if let Some(key) = args.api_key {
        config.api_key = key;
    }
    if let Some(model) = args.model {
        config.model = model;
    }
    if let Some(ms) = args.max_steps {
        config.max_steps = ms;
    }
    if let Some(mt) = args.max_tokens {
        config.max_tokens = mt;
    }
    if let Some(rpm) = args.rpm {
        config.requests_per_minute = rpm;
    }
    if let Some(hb) = args.heartbeat_secs {
        config.heartbeat_secs = hb;
    }
    if let Some(port) = args.http_port {
        config.http_channel_port = Some(port);
    }

    if !tui_mode {
        eprintln!(
            "claw starting (model: {}, url: {}, groups: {})",
            config.model,
            config.llm_url,
            config
                .groups
                .iter()
                .map(|g| g.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // ── Build agent ───────────────────────────────────────────────────────────
    let claw = ClawAgentBuilder::new(config.clone())
        .tui_mode(tui_mode)
        .build();

    let shared_bus = claw.shared_bus.clone();
    let schedule_state = claw.schedule_state.clone();
    let active_group = claw.active_group.clone();
    let model = config.model.clone();

    // Collect group buses for the TUI switcher and HTTP channel.
    let group_buses: std::collections::HashMap<String, eventage::EventBus> = claw
        .groups
        .iter()
        .map(|(name, ga)| (name.clone(), ga.bus.clone()))
        .collect();

    let groups_list: Vec<String> = config.groups.iter().map(|g| g.name.clone()).collect();
    let default_group = groups_list.first().cloned().unwrap_or_default();

    // ── Observability ─────────────────────────────────────────────────────────
    if let Some(log_path) = &args.log {
        match JsonlExporter::new(log_path).await {
            Ok(exporter) => {
                let observer = BusObserver::new(shared_bus.clone()).add_exporter(exporter);
                tokio::spawn(observer.run());
                if !tui_mode {
                    eprintln!("Logging events to {}", log_path.display());
                }
            }
            Err(e) => eprintln!("Warning: could not open log file: {e}"),
        }
    }

    if args.replay {
        LiveReplayServer::new(shared_bus.clone())
            .port(args.replay_port)
            .serve_background();
        if !tui_mode {
            eprintln!("Live replay UI: http://localhost:{}", args.replay_port);
        }
    }

    // ── HTTP channel ──────────────────────────────────────────────────────────
    if let Some(port) = config.http_channel_port {
        let allowed_senders: std::collections::HashMap<String, Vec<String>> = config
            .groups
            .iter()
            .map(|g| (g.name.clone(), g.allowed_senders.clone()))
            .collect();
        HttpChannelWorker::new(port, group_buses.clone(), default_group.clone())
            .with_allowed_senders(allowed_senders)
            .serve_background();
        if !tui_mode {
            eprintln!("HTTP channel listening on port {port}  (POST /message)");
        }
    }

    // ── HeartbeatScheduler on shared bus ──────────────────────────────────────
    {
        let heartbeat_bus = shared_bus.clone();
        let interval = Duration::from_secs(config.heartbeat_secs);
        tokio::spawn(async move {
            HeartbeatScheduler::new(heartbeat_bus, interval).run().await;
        });
    }

    // ── Run agent ─────────────────────────────────────────────────────────────
    if tui_mode {
        // Spawn the agent in background; the TUI loop owns this thread.
        let cancelled = {
            let ag = claw
                .active_group
                .try_lock()
                .map(|g| g.clone())
                .unwrap_or_default();
            claw.groups
                .get(&ag)
                .map(|ga| ga.cancelled.clone())
                .unwrap_or_else(|| Arc::new(std::sync::atomic::AtomicBool::new(false)))
        };

        let initial_group_name = active_group.lock().await.clone();
        let initial_bus = group_buses
            .get(&initial_group_name)
            .cloned()
            .unwrap_or_else(|| shared_bus.clone());

        let group_buses_arc = Arc::new(group_buses);

        tokio::spawn(async move {
            if let Err(e) = claw.run().await {
                tracing::error!("claw agent error: {e}");
            }
        });

        let gb = group_buses_arc.clone();
        tui::run_tui(
            initial_bus,
            model,
            active_group,
            groups_list,
            schedule_state,
            move |name| gb.get(name).cloned(),
            cancelled,
        )
        .await?;
    } else {
        // REPL mode: run agent in background, read stdin in foreground.
        let active_bus = Arc::new(Mutex::new(
            group_buses
                .get(&default_group)
                .cloned()
                .unwrap_or_else(|| shared_bus.clone()),
        ));

        let active_bus_for_repl = active_bus.clone();
        tokio::spawn(async move {
            if let Err(e) = claw.run().await {
                eprintln!("claw agent error: {e}");
            }
        });

        channels::terminal::run_terminal_channel(active_bus_for_repl).await?;
    }

    Ok(())
}

// ── Tracing setup ─────────────────────────────────────────────────────────────

fn setup_file_tracing() {
    let log_dir = dirs::home_dir()
        .map(|h| h.join(".claw").join("logs"))
        .unwrap_or_else(|| PathBuf::from("/tmp/claw-logs"));

    let _ = std::fs::create_dir_all(&log_dir);
    let log_file = log_dir.join("claw.log");

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
