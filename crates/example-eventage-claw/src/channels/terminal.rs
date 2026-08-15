//! Terminal (stdin) channel for eventage-claw.
//!
//! Used in `--no-tui` mode. Reads lines from stdin and publishes them as
//! `user.message` events on the active group's bus.

use eventage::event::{kinds, Event};
use eventage::EventBus;
use serde_json::json;
use std::io::{self, BufRead};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Runs a REPL loop reading from stdin and publishing `user.message` events.
///
/// `active_group_bus` is a shared reference to the currently active group's
/// EventBus. The caller can update the pointer when the user switches groups.
pub async fn run_terminal_channel(active_group_bus: Arc<Mutex<EventBus>>) -> anyhow::Result<()> {
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

        let Some(raw) = line else {
            // stdin closed (headless/Docker) — block until killed so the HTTP
            // channel keeps running.
            std::future::pending::<()>().await;
            break;
        };
        let trimmed = raw.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }

        let bus = {
            let guard = active_group_bus.lock().await;
            guard.clone()
        };

        let mut rx = bus.subscribe();

        bus.publish(Event::new(kinds::USER_MESSAGE, json!({ "text": trimmed })))
            .await
            .ok();

        // Drain events until cycle ends, printing tool calls and responses.
        drain_cycle(&mut rx).await;
    }

    Ok(())
}

async fn drain_cycle(rx: &mut eventage::BusReceiver) {
    use eventage::event::kinds as k;

    let mut cycle_started = false;
    while let Some(event) = rx.recv().await {
        if event.kind == k::AGENT_CYCLE_START {
            cycle_started = true;
            continue;
        }
        if !cycle_started {
            continue;
        }
        if print_event(&event) {
            return;
        }
        if event.kind == k::AGENT_CYCLE_END {
            return;
        }
    }
}

/// Returns `true` when the cycle is effectively done (final assistant message
/// with no pending tool calls).
fn print_event(event: &eventage::Event) -> bool {
    use eventage::event::kinds as k;

    if event.kind == k::TOOL_CALL_PROPOSED {
        let name = event.payload["name"].as_str().unwrap_or("?");
        let args = event.payload["arguments"].to_string();
        let display = if args.chars().count() > 120 {
            let truncated: String = args.chars().take(120).collect();
            format!("{truncated}…")
        } else {
            args
        };
        eprintln!("[→ {name}] {display}");
    } else if event.kind == k::TOOL_RESULT {
        let name = event.payload["name"].as_str().unwrap_or("?");
        let result = event.payload["result"].to_string();
        let preview = if result.chars().count() > 200 {
            let truncated: String = result.chars().take(200).collect();
            format!("{truncated}…")
        } else {
            result
        };
        eprintln!("[← {name}] {preview}");
    } else if event.kind == k::ASSISTANT_MESSAGE {
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
