//! Embed a live replay server in your application.
//!
//! The [`LiveReplayServer`] serves the interactive web UI, reading events
//! directly from a running [`EventBus`]. Embed it to observe agent execution
//! in real-time.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use eventage::EventBus;
//! use eventage::replay::LiveReplayServer;
//!
//! # async fn example() {
//! let bus = EventBus::new();
//!
//! // Start the background server (http://localhost:4567)
//! LiveReplayServer::new(bus.clone()).serve_background();
//!
//! // Run your agents on `bus` — the UI updates in real-time.
//! # }
//! ```

use axum::{
    extract::State,
    response::{
        sse::{Event as SseEvent, KeepAlive, Sse},
        Html,
    },
    routing::get,
    Json, Router,
};
use crate::bus::EventBus;
use crate::event::Event;
use futures_util::stream;
use std::{convert::Infallible, net::SocketAddr, sync::Arc};
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;

/// Embedded UI HTML shared between the live server and CLI binary.
pub const UI_HTML: &str = include_str!("ui.html");

// ── LiveReplayServer ──────────────────────────────────────────────────────────

/// A live replay server streaming events from a running [`EventBus`].
///
/// Serves the same UI as `eventage-replay` CLI, powered by a live bus subscription.
///
/// # Endpoints
///
/// - `/`: Interactive web UI.
/// - `/events`: Active-branch snapshot (JSON array).
/// - `/events/stream`: Server-Sent Events (SSE) stream of live events.
///
/// Clients loading `/` automatically subscribe to `/events/stream` for real-time updates.
pub struct LiveReplayServer {
    buses: Vec<EventBus>,
    port: u16,
}

#[derive(Clone)]
struct LiveState {
    buses: Vec<Arc<EventBus>>,
}

impl LiveReplayServer {
    /// Create a new server attached to `bus`.
    pub fn new(bus: EventBus) -> Self {
        Self { buses: vec![bus], port: 4567 }
    }

    /// Attach additional buses so all their events appear in the replay UI.
    pub fn with_buses(mut self, buses: impl IntoIterator<Item = EventBus>) -> Self {
        self.buses.extend(buses);
        self
    }

    /// Override the listening port (default: `4567`).
    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Start the server in a background Tokio task.
    ///
    /// Returns a [`tokio::task::JoinHandle`]. Dropping it does not stop the server;
    /// it runs until the process exits or is explicitly aborted.
    pub fn serve_background(self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            self.serve().await;
        })
    }

    /// Start the server, blocking the current task until it exits.
    pub async fn serve(self) {
        let state = LiveState {
            buses: self.buses.into_iter().map(Arc::new).collect(),
        };

        let app = Router::new()
            .route("/", get(serve_ui))
            .route("/events", get(serve_snapshot))
            .route("/events/stream", get(serve_live_stream))
            .layer(CorsLayer::permissive())
            .with_state(state);

        let addr = SocketAddr::from(([127, 0, 0, 1], self.port));
        tracing::info!("Live replay UI: http://localhost:{}", self.port);

        let listener = match tokio::net::TcpListener::bind(addr).await {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("Could not bind replay server to {addr}: {e}");
                return;
            }
        };

        if let Err(e) = axum::serve(listener, app).await {
            tracing::error!("Replay server error: {e}");
        }
    }
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn serve_ui() -> Html<&'static str> {
    Html(UI_HTML)
}

async fn serve_snapshot(State(state): State<LiveState>) -> Json<Vec<Event>> {
    let mut all: Vec<Event> = Vec::new();
    for bus in &state.buses {
        all.extend(bus.log().await);
    }
    all.sort_by_key(|e| e.timestamp);
    Json(all)
}

async fn serve_live_stream(
    State(state): State<LiveState>,
) -> Sse<impl futures_util::stream::Stream<Item = Result<SseEvent, Infallible>>> {
    // Merge live events from all buses into one channel.
    let (tx, rx) = mpsc::unbounded_channel::<Event>();
    for bus in &state.buses {
        let mut bus_rx = bus.subscribe();
        let tx = tx.clone();
        tokio::spawn(async move {
            while let Some(event) = bus_rx.recv().await {
                let _ = tx.send(event);
            }
        });
    }
    let event_stream = stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|event| {
            let data = serde_json::to_string(&event).unwrap_or_default();
            let sse = SseEvent::default().data(data);
            (Ok::<_, Infallible>(sse), rx)
        })
    });
    Sse::new(event_stream).keep_alive(KeepAlive::default())
}
