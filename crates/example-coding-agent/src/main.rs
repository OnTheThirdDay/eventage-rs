//! eventage-code — an ACP agent server.
//!
//! Editors speak the [Agent Client Protocol](https://agentclientprotocol.com)
//! to this binary over stdio, so the agent runs inside Zed, Kiro, JetBrains,
//! or any ACP-capable client with full streaming, diff review, permission
//! prompts, and a live task plan.
//!
//! ```sh
//! # As an editor agent (stdio JSON-RPC) — the default.
//! eventage-code
//!
//! # Headless, for scripts and CI.
//! eventage-code run -p "fix the failing test" --cwd /repo --mode auto
//! ```
//!
//! Credentials come from `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`; with
//! neither set it talks to a local OpenAI-compatible server (`OPENAI_BASE_URL`).

use anyhow::Result;
use clap::{Parser, Subcommand};
use eventage_code::acp::wire::ContentBlock;
use eventage_code::acp::AcpServer;
use eventage_code::agent::CodingSession;
use eventage_code::config::{ModelConfig, PermissionMode, SessionConfig};
use eventage::event::kinds;
use eventage::{Event, EventBus};
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "eventage-code", version, about = "LSP-aware coding agent (ACP server)")]
struct Cli {
    /// Model override (defaults per provider).
    #[arg(long, global = true)]
    model: Option<String>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Serve the Agent Client Protocol over stdio (default).
    Acp,
    /// Run a single prompt headlessly and print the result.
    Run {
        /// The prompt.
        #[arg(short = 'p', long)]
        prompt: String,
        /// Workspace root.
        #[arg(long, default_value = ".")]
        cwd: String,
        /// Permission mode: plan | ask | auto | yolo.
        #[arg(long, default_value = "auto")]
        mode: String,
        /// Emit the full event log as JSON instead of prose.
        #[arg(long)]
        json: bool,
        /// Approve every gated tool call without prompting (for CI).
        #[arg(long, short = 'y')]
        yes: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let model = ModelConfig::from_env(cli.model);

    match cli.command {
        // Logs must never touch stdout: it carries the protocol.
        None | Some(Command::Acp) => {
            tracing_subscriber::fmt()
                .with_env_filter(
                    EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| EnvFilter::new("eventage_code=info,eventage=warn")),
                )
                .with_writer(std::io::stderr)
                .init();
            Arc::new(AcpServer::new(model)).run().await
        }
        Some(Command::Run {
            prompt,
            cwd,
            mode,
            json,
            yes,
        }) => {
            tracing_subscriber::fmt()
                .with_env_filter(
                    EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| EnvFilter::new("eventage_code=info")),
                )
                .with_writer(std::io::stderr)
                .init();
            run_headless(model, prompt, cwd, mode, json, yes).await
        }
    }
}

/// Answer `permission.request` events when no editor is attached.
///
/// Without this, headless runs in `ask`/`auto` mode stall: the permission
/// hook publishes a request and waits on the bus for a decision that nobody
/// is there to give. We resolve it three ways, in order of preference:
///
/// - `--yes` approves everything (CI).
/// - An interactive terminal prompts the operator.
/// - Otherwise (a pipe, a CI job with no TTY) we **deny immediately** with an
///   actionable reason instead of hanging until the timeout.
fn spawn_headless_approver(bus: EventBus, auto_approve: bool) -> tokio::task::JoinHandle<()> {
    use std::io::IsTerminal;
    let interactive = std::io::stdin().is_terminal();
    let mut rx = bus.subscribe();

    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            if event.kind != kinds::PERMISSION_REQUEST {
                continue;
            }
            let request_id = event
                .payload
                .get("request_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let tool = event
                .payload
                .get("tool")
                .and_then(|v| v.as_str())
                .unwrap_or("tool")
                .to_string();

            let (approve, reason) = if auto_approve {
                (true, None)
            } else if interactive {
                let prompt_tool = tool.clone();
                let answer = tokio::task::spawn_blocking(move || {
                    use std::io::{stdin, stderr, Write};
                    let mut err = stderr();
                    let _ = write!(err, "\nAllow tool '{prompt_tool}'? [y/N]: ");
                    let _ = err.flush();
                    let mut line = String::new();
                    let _ = stdin().read_line(&mut line);
                    line.trim().eq_ignore_ascii_case("y")
                })
                .await
                .unwrap_or(false);
                (
                    answer,
                    (!answer).then(|| "the operator declined this action".to_string()),
                )
            } else {
                (
                    false,
                    Some(format!(
                        "'{tool}' needs approval but this run is non-interactive;                          re-run with --yes to approve automatically, or --mode yolo"
                    )),
                )
            };

            let _ = bus
                .publish(Event::new(
                    kinds::PERMISSION_DECISION,
                    serde_json::json!({
                        "request_id": request_id,
                        "approve": approve,
                        "reason": reason,
                    }),
                ))
                .await;
        }
    })
}

async fn run_headless(
    model: ModelConfig,
    prompt: String,
    cwd: String,
    mode: String,
    as_json: bool,
    auto_approve: bool,
) -> Result<()> {
    let cwd = std::fs::canonicalize(&cwd)?.display().to_string();
    let mut config = SessionConfig::new(cwd, model);
    config.mode = PermissionMode::from_id(&mode)
        .ok_or_else(|| anyhow::anyhow!("unknown mode '{mode}' (plan|ask|auto|yolo)"))?;

    let session = CodingSession::create(uuid::Uuid::new_v4().to_string(), config, None).await?;
    // No editor is attached, so stand in as the approver.
    let approver = spawn_headless_approver(session.bus.clone(), auto_approve);
    session
        .submit_prompt(&[ContentBlock::text(prompt)])
        .await?;
    let outcome = session.run_cycle().await;
    approver.abort();
    outcome?;

    let log = session.bus.log().await;
    if as_json {
        println!("{}", serde_json::to_string_pretty(&log)?);
        return Ok(());
    }

    let reply = log
        .iter()
        .rev()
        .find(|e| e.kind == kinds::ASSISTANT_MESSAGE)
        .and_then(|e| e.payload.get("content").and_then(|c| c.as_str()))
        .unwrap_or("(no response)");
    println!("{reply}");
    Ok(())
}
