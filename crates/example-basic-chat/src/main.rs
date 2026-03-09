//! Basic chat example — single agent, multi-turn REPL.
//!
//! Demonstrates two [`Session`] interaction modes:
//! - **Synchronous REPL (default)**: `session.chat(msg)` blocks until the ReAct cycle finishes. Ideal for terminals.
//! - **Reactive / Event-driven**: `session.run()` processes `user.message` events from the bus asynchronously. Ideal for GUIs or concurrent inputs (see `example-reactive-chat`).
//!
//! # Setup
//! Requires Ollama running locally with the `qwen3:4b` model:
//! ```bash
//! ollama pull qwen3:4b
//! ```
//!
//! # Run
//! ```bash
//! cargo run -p example-basic-chat
//! ```
//!
//! A live replay UI starts automatically at `http://localhost:4567`.
//! Events are also saved to `/tmp/basic-chat-events.jsonl` for replay:
//! ```bash
//! cargo run -p eventage-replay -- /tmp/basic-chat-events.jsonl
//! ```

use eventage_llm::OpenAiProvider;
use eventage_provided_impl::{BusObserver, JsonlExporter, Session};
use eventage_replay::LiveReplayServer;
use tokio::io::{AsyncBufReadExt, BufReader};

const EVENTS_LOG: &str = "/tmp/basic-chat-events.jsonl";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("warn")
        .with_writer(std::io::stderr)
        .init();

    // ── Build session ──────────────────────────────────────────────────────────
    // Session wraps an agent + event bus. Use chat() for a synchronous REPL,
    // or run() for a reactive event-driven loop.
    let mut session = Session::builder()
        .llm(OpenAiProvider::ollama("qwen3:4b"))
        .system_prompt("You are a concise, helpful assistant. Keep replies under three sentences.")
        .build();

    // ── Observability ──────────────────────────────────────────────────────────
    // Records every cycle and tool call. LiveReplayServer streams them to the browser;
    // JsonlExporter persists them for post-hoc replay.
    LiveReplayServer::new(session.bus().clone()).serve_background();
    let exporter = JsonlExporter::new(EVENTS_LOG).await?;
    let observer = BusObserver::new(session.bus().clone()).add_exporter(exporter);
    tokio::spawn(async move { observer.run().await });

    // ── Synchronous REPL ───────────────────────────────────────────────────────
    // chat() publishes a user.message event, blocks for the reasoning cycle,
    // and returns the text response. Enforces sequential interaction.
    eprintln!("Chat agent ready. Type your message and press Enter. Ctrl-C to exit.");
    eprintln!("Live replay:  http://localhost:4567");
    eprintln!("Session log:  {EVENTS_LOG}");
    eprintln!();

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    loop {
        eprint!("> ");
        let Some(line) = lines.next_line().await? else {
            break;
        };
        let input = line.trim().to_string();
        if input.is_empty() {
            continue;
        }
        if input == "exit" || input == "quit" {
            break;
        }

        let reply = session.chat(&input).await?;
        let display = strip_thinking(&reply);
        if !display.is_empty() {
            println!("\nAssistant: {display}\n");
        }
    }

    Ok(())
}

fn strip_thinking(s: &str) -> &str {
    if let Some(end) = s.find("</think>") {
        s[end + "</think>".len()..].trim_start()
    } else {
        s
    }
}
