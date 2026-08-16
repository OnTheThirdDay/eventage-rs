//! A small HTTP way in: set a goal, steer, or answer an approval.
//!
//! Not a second UI. It exists so a session can be reached from somewhere that
//! is not the machine it runs on — a phone, a cron entry, a chat integration —
//! which is the property both products sell and neither achieves with a
//! terminal. Everything it does is published onto the session bus, so a
//! request from here and a click in Studio are indistinguishable downstream.
//!
//! Bound to loopback and gated on a token. A local port that starts work in
//! somebody's documents folder is not a thing to leave open.

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use eventage::{event::kinds as ev, Event, EventBus};
use serde::Deserialize;
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::info;

#[derive(Clone)]
pub struct ChannelState {
    bus: EventBus,
    token: String,
}

#[derive(Deserialize)]
pub struct GoalRequest {
    /// What to work on.
    goal: String,
    token: String,
}

#[derive(Deserialize)]
pub struct DecisionRequest {
    request_id: String,
    approve: bool,
    #[serde(default)]
    reason: Option<String>,
    token: String,
}

/// Serve the channel on `port`, returning the address it bound to.
pub async fn serve(bus: EventBus, token: String, port: u16) -> anyhow::Result<SocketAddr> {
    let state = Arc::new(ChannelState { bus, token });
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/goal", post(set_goal))
        .route("/decision", post(decide))
        .with_state(state);

    // Loopback only. This endpoint starts work in the user's own files.
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    let addr = listener.local_addr()?;
    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            tracing::warn!("the http channel stopped: {e}");
        }
    });
    info!(%addr, "http channel listening");
    Ok(addr)
}

/// Constant-time enough for a local token, and never logged.
fn authorised(state: &ChannelState, offered: &str) -> bool {
    offered.len() == state.token.len()
        && offered
            .bytes()
            .zip(state.token.bytes())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0
}

async fn set_goal(
    State(state): State<Arc<ChannelState>>,
    Json(req): Json<GoalRequest>,
) -> Result<Json<Value>, StatusCode> {
    if !authorised(&state, &req.token) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    state
        .bus
        .publish(Event::new(ev::USER_MESSAGE, json!({ "text": req.goal })))
        .await
        .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?;
    Ok(Json(json!({ "accepted": true })))
}

async fn decide(
    State(state): State<Arc<ChannelState>>,
    Json(req): Json<DecisionRequest>,
) -> Result<Json<Value>, StatusCode> {
    if !authorised(&state, &req.token) {
        return Err(StatusCode::UNAUTHORIZED);
    }
    state
        .bus
        .publish(Event::new(
            ev::PERMISSION_DECISION,
            json!({
                "request_id": req.request_id,
                "approve": req.approve,
                "reason": req.reason,
            }),
        ))
        .await
        .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?;
    Ok(Json(json!({ "recorded": true })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(token: &str) -> ChannelState {
        ChannelState {
            bus: EventBus::new(),
            token: token.to_string(),
        }
    }

    #[test]
    fn a_wrong_token_is_refused_and_length_alone_does_not_pass() {
        let s = state("secret-token");
        assert!(authorised(&s, "secret-token"));
        assert!(!authorised(&s, "secret-tokeX"));
        assert!(!authorised(&s, "short"));
        assert!(!authorised(&s, ""));
    }
}
