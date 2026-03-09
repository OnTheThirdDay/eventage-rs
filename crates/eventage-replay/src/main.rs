//! `eventage-replay` — interactive visual replay of Eventage agent sessions.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p eventage-replay -- path/to/events.jsonl [--port <port>]
//! ```
//!
//! Starts a local HTTP server and opens the UI in your browser.
//! The UI allows:
//! - Scrubbing through the timeline.
//! - Inspecting event payloads and metadata.
//! - Viewing swim lanes, cycle ranges, and event kinds.
//! - Play/pause functionality (0.5×–5× speed).
//! - Filtering events by kind.
//!
//! For live replay within an app, embed [`eventage_replay::LiveReplayServer`].

use axum::{extract::State, response::Html, routing::get, Json, Router};
use eventage_core::Event;
use eventage_replay::UI_HTML;
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use tokio::fs;
use tower_http::cors::CorsLayer;

#[derive(Clone)]
struct AppState {
    events: Arc<Vec<Event>>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter("eventage_replay=info")
        .init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        eprintln!(
            "Usage: eventage-replay <events.jsonl> [--port <port>]\n\
             \n\
             Opens an interactive visual replay of a recorded Eventage session.\n\
             The JSONL file is produced by eventage_observability::JsonlExporter.\n\
             \n\
             For live replay during execution, use eventage_replay::LiveReplayServer."
        );
        std::process::exit(1);
    }

    let path = PathBuf::from(&args[1]);
    let port: u16 = args
        .windows(2)
        .find(|w| w[0] == "--port")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(4567);

    // Parse the JSONL file.
    let content = fs::read_to_string(&path).await.unwrap_or_else(|e| {
        eprintln!("Error reading {:?}: {e}", path);
        std::process::exit(1);
    });

    let events: Vec<Event> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .filter_map(|(i, line)| {
            serde_json::from_str(line)
                .map_err(|e| {
                    eprintln!("Warning: line {} is not valid JSON — skipping ({e})", i + 1);
                })
                .ok()
        })
        .collect();

    let event_count = events.len();
    let state = AppState {
        events: Arc::new(events),
    };

    let app = Router::new()
        .route("/", get(serve_ui))
        .route("/events", get(serve_events))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let url = format!("http://localhost:{port}");

    eprintln!("Loaded {event_count} events from {:?}", path);
    eprintln!("Replay UI: {url}");

    // Try to open the browser automatically.
    let url_clone = url.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        open_browser(&url_clone);
    });

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .unwrap_or_else(|e| {
            eprintln!("Could not bind to {addr}: {e}");
            std::process::exit(1);
        });

    axum::serve(listener, app).await.unwrap();
}

async fn serve_ui() -> Html<&'static str> {
    Html(UI_HTML)
}

async fn serve_events(State(state): State<AppState>) -> Json<Vec<Event>> {
    Json((*state.events).clone())
}

fn open_browser(url: &str) {
    #[cfg(target_os = "linux")]
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(url).spawn();
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("cmd")
        .args(["/c", "start", url])
        .spawn();
}
