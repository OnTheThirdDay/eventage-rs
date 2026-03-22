//! HTTP channel worker for eventage-claw.
//!
//! Runs an embedded `axum` HTTP server. Any external messaging channel
//! (WhatsApp webhook, Telegram bot webhook, Slack Events API, etc.) can POST
//! to `POST /message` and the matching group's agent will receive and respond.
//!
//! The external adapter just needs to POST to this endpoint.
//!
//! Request body:
//! ```json
//! { "group": "personal", "text": "Hello", "sender": "alice" }
//! ```
//!
//! Response:
//! ```json
//! { "ok": true }
//! ```

use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::IntoResponse,
    routing::post,
    Router,
};
use eventage::{event::kinds, Event, EventBus};
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

// ── Request/Response types ────────────────────────────────────────────────────

#[derive(Deserialize)]
struct MessageRequest {
    /// Target group name. Falls back to first configured group if omitted.
    #[serde(default)]
    group: String,
    /// Message text to deliver to the agent. May be empty if attachments are present.
    #[serde(default)]
    text: String,
    /// Optional sender identifier (logged, passed in payload).
    #[serde(default)]
    sender: String,
    /// Explicit reply address (WhatsApp JID, Telegram chat ID, etc.).
    /// When provided, overrides the sender-derived reply_to so the response
    /// is routed back to the correct destination (e.g. @s.whatsapp.net vs @lid).
    #[serde(default)]
    reply_to: String,
    /// Optional media attachments forwarded from the channel bridge.
    /// Each entry describes a file saved to disk by the bridge.
    /// The agent can use RunCommandTool to process them (e.g. whisper, pdftotext).
    #[serde(default)]
    attachments: Vec<serde_json::Value>,
}

// ── Server state ──────────────────────────────────────────────────────────────

#[derive(Clone)]
struct ChannelState {
    group_buses: Arc<Mutex<HashMap<String, EventBus>>>,
    default_group: String,
    /// Per-group sender allowlists. Empty vec = allow all senders.
    allowed_senders: HashMap<String, Vec<String>>,
}

// ── Handler ───────────────────────────────────────────────────────────────────

async fn post_message(
    State(state): State<ChannelState>,
    axum::Json(req): axum::Json<MessageRequest>,
) -> impl IntoResponse {
    let group = if req.group.is_empty() {
        state.default_group.clone()
    } else {
        req.group.clone()
    };

    // Sender allowlist check: if the group has a non-empty allowlist, the
    // sender must appear in it. An empty allowlist permits all senders.
    if !req.sender.is_empty() {
        if let Some(allowed) = state.allowed_senders.get(&group) {
            if !allowed.is_empty() && !allowed.iter().any(|a| a == &req.sender) {
                return (
                    StatusCode::FORBIDDEN,
                    axum::Json(json!({ "ok": false, "error": "sender not allowed" })),
                );
            }
        }
    }

    // Require at least text or an attachment.
    if req.text.is_empty() && req.attachments.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(json!({ "ok": false, "error": "text or attachments required" })),
        );
    }

    let buses = state.group_buses.lock().await;
    let bus = match buses.get(&group) {
        Some(b) => b.clone(),
        None => {
            return (
                StatusCode::NOT_FOUND,
                axum::Json(json!({ "ok": false, "error": format!("group '{group}' not found") })),
            );
        }
    };
    drop(buses);

    info!(group = %group, sender = %req.sender, attachments = req.attachments.len(), "HTTP channel: delivering message");

    // `reply_to` carries the sender address back through the event so
    // ChannelOutputWorker can route the response to the right destination.
    // Prefer explicit reply_to > sender > group (allows bridge to pass the
    // corrected JID, e.g. @s.whatsapp.net instead of @lid).
    let reply_to = if !req.reply_to.is_empty() {
        req.reply_to.clone()
    } else if !req.sender.is_empty() {
        req.sender.clone()
    } else {
        group.clone()
    };
    let payload = json!({
        "text": req.text,
        "source": "http",
        "sender": req.sender,
        "reply_to": reply_to,
        "group": group,
        "attachments": req.attachments,
    });

    match bus.publish(Event::new(kinds::USER_MESSAGE, payload)).await {
        Ok(_) => (StatusCode::OK, axum::Json(json!({ "ok": true }))),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(json!({ "ok": false, "error": e.to_string() })),
        ),
    }
}

async fn health() -> impl IntoResponse {
    axum::Json(json!({ "status": "ok" }))
}

// ── HttpChannelWorker ─────────────────────────────────────────────────────────

/// Runs an embedded axum HTTP server that bridges external webhooks to the
/// group EventBus.
///
/// Start it with [`HttpChannelWorker::serve_background`] to run it as a
/// background Tokio task.
pub struct HttpChannelWorker {
    port: u16,
    group_buses: HashMap<String, EventBus>,
    default_group: String,
    allowed_senders: HashMap<String, Vec<String>>,
}

impl HttpChannelWorker {
    pub fn new(port: u16, group_buses: HashMap<String, EventBus>, default_group: String) -> Self {
        Self {
            port,
            group_buses,
            default_group,
            allowed_senders: HashMap::new(),
        }
    }

    /// Attach per-group sender allowlists.
    pub fn with_allowed_senders(mut self, allowed_senders: HashMap<String, Vec<String>>) -> Self {
        self.allowed_senders = allowed_senders;
        self
    }

    /// Start the HTTP server as a background Tokio task.
    pub fn serve_background(self) {
        let group_buses = Arc::new(Mutex::new(self.group_buses));
        let state = ChannelState {
            group_buses,
            default_group: self.default_group,
            allowed_senders: self.allowed_senders,
        };

        let app = Router::new()
            .route("/message", post(post_message))
            .route("/health", axum::routing::get(health))
            .layer(DefaultBodyLimit::max(1024 * 1024)) // 1 MB — prevents OOM from oversized payloads
            .with_state(state);

        let addr = SocketAddr::from(([0, 0, 0, 0], self.port));
        info!(port = self.port, "HTTP channel listening on {addr}");

        tokio::spawn(async move {
            let listener = match tokio::net::TcpListener::bind(addr).await {
                Ok(l) => l,
                Err(e) => {
                    tracing::error!("HTTP channel failed to bind: {e}");
                    return;
                }
            };
            if let Err(e) = axum::serve(listener, app).await {
                tracing::error!("HTTP channel error: {e}");
            }
        });
    }
}
