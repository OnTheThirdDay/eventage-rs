//! Starting Studio's server without deciding how it will be looked at.
//!
//! `main` binds a port, prints a URL and opens a browser window. A desktop
//! shell wants the first of those and none of the rest: the server, on a port
//! it can find out, with a token it can put in a webview.
//!
//! So the binding lives here and the presentation lives in the caller. There
//! is no second copy of the wiring for a shell to drift away from, and the
//! HTTP API stays exactly what it was — which is what keeps the plain binary,
//! a browser, and a native window all talking to one implementation.

use crate::backend::Backend;
use crate::server::{self, AppState};
use anyhow::{Context, Result};
use std::net::SocketAddr;
use std::sync::Arc;

/// A running Studio server.
pub struct Running {
    /// Where it is listening.
    pub addr: SocketAddr,
    /// The URL to open, token included.
    pub url: String,
    /// The per-run token, for a client that sets the cookie itself.
    pub token: String,
    state: AppState,
    server: tokio::task::JoinHandle<()>,
}

impl Running {
    /// Stop serving and release what the sessions hold.
    ///
    /// Sessions own child processes — language servers, ACP agents — so a
    /// shell that closes its window without this leaves them running.
    pub async fn shutdown(self) {
        self.server.abort();
        self.state.shutdown().await;
    }
}

/// Resolve which model to use, from the workspace's settings and the
/// environment, then take the credentials out of the environment.
///
/// This is shared for the same reason `serve` is. The desktop shell first
/// resolved the model on its own, skipping `.claude/settings.json` — so the
/// same workspace that ran on `opus` in the plain binary silently ran on a
/// local Ollama endpoint in a window. Two ways to answer one question is one
/// too many.
///
/// `chosen` is an explicit choice — a `--model` flag — and wins over the
/// workspace's own setting.
pub fn resolve_model(cwd: &str, chosen: Option<String>) -> eventage_code::config::ModelConfig {
    // A workspace configured for an Anthropic-compatible gateway keeps its
    // endpoint, credential and routing headers in `.claude/settings.json`.
    // Read before the model is resolved. The block is only applied to this
    // process if the operator has said the repository is trusted — see the
    // `settings` module for what an untrusted one can otherwise do with it.
    let settings = eventage_code::settings::ClaudeSettings::load(cwd);
    let env = settings.apply_env();
    if !env.applied.is_empty() {
        // Names only. Several of these are credentials.
        tracing::info!(vars = ?env.applied, "applied .claude/settings.json");
    }

    // Resolved before the credentials are taken out of the environment;
    // everything downstream reads them from the config, not from `getenv`.
    let model = eventage_code::config::ModelConfig::from_env(chosen.or(settings.model));

    // A process's environment is a file any process of the same user can read,
    // so a confined command denied the key in its own environment could open
    // `/proc/<our pid>/environ` and have it anyway. Held in memory from here on.
    let held = eventage_code::secrets::capture_and_scrub();
    if !held.is_empty() {
        tracing::debug!(
            count = held.len(),
            "moved credentials out of the environment"
        );
    }

    model
}

/// Bind the server and start serving, returning where to point a client.
///
/// Port `0` asks the OS for a free one, which is what a desktop shell wants:
/// nothing has to agree on a number in advance, and two copies of the app do
/// not collide.
pub async fn serve(backend: Arc<dyn Backend>, port: u16) -> Result<Running> {
    let token = uuid::Uuid::new_v4().to_string();
    let state = AppState::new(backend, token.clone());
    let app = server::router(state.clone());

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .with_context(|| format!("could not bind 127.0.0.1:{port}"))?;
    let addr = listener.local_addr()?;
    let url = format!("http://{addr}/?t={token}");

    let server = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            tracing::warn!("the studio server stopped: {e}");
        }
    });

    Ok(Running {
        addr,
        url,
        token,
        state,
        server,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn serving_on_port_zero_reports_where_it_landed() {
        // What a desktop shell needs: no agreed port, no collision between two
        // copies of the app, and a URL it can hand straight to a webview.
        let dir = tempfile::tempdir().unwrap();
        let settings = Arc::new(
            crate::model_settings::ModelSettings::load(
                eventage_code::config::ModelConfig::from_env(Some("m".into())),
                dir.path(),
            )
            .await,
        );
        let backend = Arc::new(
            crate::backend::local::LocalBackend::new(settings, dir.path().display().to_string())
                .await,
        );

        let running = serve(backend, 0).await.unwrap();
        assert_ne!(running.addr.port(), 0, "the OS port was not reported back");
        assert!(running.url.contains(&running.token));
        assert!(running.url.starts_with("http://127.0.0.1:"));

        // It is actually answering.
        let response = reqwest::Client::new()
            .get(format!("http://{}/api/app", running.addr))
            .header("Cookie", format!("eventage_studio_token={}", running.token))
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());

        running.shutdown().await;
    }
}
