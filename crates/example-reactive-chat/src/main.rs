//! Reactive chat example — event-driven interaction via `Session::run()`.
//!
//! Demonstrates asynchronous, background processing using `user.message` events.
//! Unlike `example-basic-chat` which uses a blocking `chat()` loop, this example
//! uses `Session::run()` to process input published to the bus independently.
//! Ideal for decoupled input/output such as GUIs, sockets, or streaming.
//!
//! # Setup
//! Requires Ollama running locally with the `qwen3:4b` model:
//! ```bash
//! ollama pull qwen3:4b
//! ```
//!
//! # Run
//! ```bash
//! cargo run -p example-reactive-chat
//! ```
//!
//! Live replay UI: `http://localhost:4568`
//! Event log: `/tmp/reactive-chat-events.jsonl`

use eventage_core::{kinds, Event};
use eventage_llm::OpenAiProvider;
use eventage_provided_impl::{BusObserver, JsonlExporter, Session};
use eventage_replay::LiveReplayServer;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, BufReader};

const EVENTS_LOG: &str = "/tmp/reactive-chat-events.jsonl";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("warn")
        .with_writer(std::io::stderr)
        .init();

    // ── Build session ──────────────────────────────────────────────────────────
    let session = Session::builder()
        .llm(OpenAiProvider::ollama("qwen3:4b"))
        .system_prompt("You are a concise, helpful assistant. Keep replies under three sentences.")
        .build();

    // ── Observability ──────────────────────────────────────────────────────────
    // Live replay streams events to the browser in real-time; JsonlExporter
    // persists them for post-hoc replay.
    LiveReplayServer::new(session.bus().clone()).port(4568).serve_background();
    let exporter = JsonlExporter::new(EVENTS_LOG).await?;
    let observer = BusObserver::new(session.bus().clone()).add_exporter(exporter);
    tokio::spawn(async move { observer.run().await });

    // ── Bus handles ───────────────────────────────────────────────────────────
    // Clone the bus before calling run() to publish and subscribe from other tasks.
    let bus = session.bus().clone();

    // Subscribe before spawning the agent so no response is missed regardless
    // of scheduling order.
    let mut reply_rx = bus.subscribe();

    // ── Reactive agent loop ───────────────────────────────────────────────────
    // Subscribes to the bus and cycles automatically on `user.message` events.
    tokio::spawn(async move {
        if let Err(e) = session.run().await {
            tracing::error!("agent error: {e}");
        }
    });

    // ── Stdin reader / REPL ───────────────────────────────────────────────────
    // Publishes input to the bus, which triggers the reactive agent.
    eprintln!("Reactive chat agent ready. Type your message and press Enter. Ctrl-C to exit.");
    eprintln!("Live replay:  http://localhost:4568");
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

        // Publish user input as a plain bus event — no session.chat() needed.
        bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": input})))
            .await?;

        // Wait for the assistant's response on the shared subscriber.
        while let Some(event) = reply_rx.recv().await {
            if event.kind == kinds::ASSISTANT_MESSAGE {
                let content = event.payload["content"].as_str().unwrap_or("");
                let display = strip_thinking(content);
                if !display.is_empty() {
                    println!("\nAssistant: {display}\n");
                    break;
                }
            }
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
