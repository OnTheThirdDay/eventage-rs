//! Eventage Studio — a desktop app for the eventage coding agent.
//!
//! Two panes: the conversation, and a live trace of everything behind it.
//! The trace is the reason the app exists in this shape — an event-sourced
//! harness knows exactly why it did what it did, and that is worth showing
//! rather than hiding behind a chat bubble.
//!
//! ```sh
//! # Host the coding agent in-process (default): full event DAG in the trace.
//! eventage-studio
//!
//! # Or drive any ACP agent, the way an editor would.
//! eventage-studio --acp eventage-code
//! ```
//!
//! Credentials come from the environment, exactly as for `eventage-code`:
//! `ANTHROPIC_API_KEY`, `QWEN_API_KEY`, or `OPENAI_API_KEY` with
//! `OPENAI_BASE_URL`.

use anyhow::{Context, Result};
use eventage_studio::backend::{acp::AcpBackend, local::LocalBackend, Backend};
use eventage_studio::server;
use clap::Parser;
use eventage_code::config::ModelConfig;
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "eventage-studio",
    version,
    about = "Desktop app for the eventage coding agent"
)]
struct Cli {
    /// Drive an external ACP agent instead of hosting one in-process.
    ///
    /// Everything after this flag is the command to run, e.g.
    /// `--acp eventage-code`.
    #[arg(long, num_args = 1.., value_name = "COMMAND")]
    acp: Option<Vec<String>>,

    /// Workspace to open. Defaults to the current directory.
    #[arg(long)]
    cwd: Option<String>,

    /// Port to listen on. 0 picks a free one, which is the sane default for a
    /// desktop app: two copies should not fight over a port.
    #[arg(long, default_value_t = 0)]
    port: u16,

    /// Model override, passed to the in-process agent.
    #[arg(long)]
    model: Option<String>,

    /// Print the URL instead of opening a window.
    #[arg(long)]
    no_open: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("eventage_studio=info,eventage=warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cwd = match &cli.cwd {
        Some(dir) => std::fs::canonicalize(dir)
            .with_context(|| format!("cannot open workspace '{dir}'"))?,
        None => std::env::current_dir()?,
    }
    .display()
    .to_string();

    let backend: Arc<dyn Backend> = match cli.acp {
        Some(command) => Arc::new(AcpBackend::new(command, cwd.clone())?),
        None => Arc::new(LocalBackend::new(ModelConfig::from_env(cli.model), cwd.clone()).await),
    };

    let info = backend.info();
    let token = uuid::Uuid::new_v4().to_string();
    let state = server::AppState::new(backend, token.clone());
    let app = server::router(state.clone());

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", cli.port))
        .await
        .with_context(|| format!("could not bind 127.0.0.1:{}", cli.port))?;
    let addr = listener.local_addr()?;
    let url = format!("http://{addr}/?t={token}");

    tracing::info!(
        backend = info.backend,
        model = %info.model,
        workspace = %cwd,
        "Eventage Studio"
    );
    println!("Eventage Studio → {url}");

    if !cli.no_open {
        open_window(&url);
    }

    // Ctrl-C must reach the sessions, or LSP servers and ACP children would
    // survive the app that started them.
    let shutdown = {
        let state = state.clone();
        async move {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
            state.shutdown().await;
        }
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    state.shutdown().await;
    Ok(())
}

/// Open the app in a browser window.
///
/// Chrome and Edge take `--app`, which drops the tab strip and address bar
/// and gives a window that reads as an application rather than a web page.
/// Without one of those we fall back to whatever handles http, which still
/// works — it just looks like a tab.
fn open_window(url: &str) {
    #[cfg(target_os = "macos")]
    let candidates: [(&str, &[&str]); 2] = [
        ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", &[]),
        ("open", &[]),
    ];
    #[cfg(target_os = "windows")]
    let candidates: [(&str, &[&str]); 2] = [("chrome", &[]), ("cmd", &["/C", "start", ""])];
    #[cfg(all(unix, not(target_os = "macos")))]
    let candidates: [(&str, &[&str]); 5] = [
        ("google-chrome", &[]),
        ("chromium", &[]),
        ("microsoft-edge", &[]),
        ("brave-browser", &[]),
        ("xdg-open", &[]),
    ];

    for (program, prefix) in candidates {
        // Only the Chromium family understands --app; anything else gets the
        // bare URL.
        let app_mode = !program.contains("open") && program != "cmd";
        let mut command = std::process::Command::new(program);
        command.args(prefix);
        if app_mode {
            command.arg(format!("--app={url}"));
        } else {
            command.arg(url);
        }
        command
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        if command.spawn().is_ok() {
            return;
        }
    }
    tracing::warn!("could not open a window automatically — open the URL above");
}
