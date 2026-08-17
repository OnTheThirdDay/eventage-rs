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
use clap::Parser;
use eventage_code::config::ModelConfig;
use eventage_studio::backend::{
    acp::AcpBackend, cowork::CoworkBackend, local::LocalBackend, Backend,
};
use eventage_studio::server;
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

    /// Run cowork instead of the coding agent.
    ///
    /// A goal is split into workstreams that run against independent copies
    /// of the folder; you compare what they produced and keep one. Nothing is
    /// written to the folder until you do.
    #[arg(long)]
    cowork: bool,

    /// Cowork: how many workstreams may run at once.
    #[arg(long, default_value_t = 3)]
    parallel: usize,

    /// Cowork: how many workstreams a goal may be split into.
    #[arg(long, default_value_t = 5)]
    split: usize,

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

/// Everything that touches the process environment, before the runtime exists.
///
/// `set_var` and `remove_var` are unsafe on Unix because another thread
/// calling `getenv` concurrently is undefined behaviour, and `#[tokio::main]`
/// has already built the runtime's worker threads by the time the async body
/// starts. Doing the settings block and the credential scrub here means the
/// process really is single-threaded while it happens.
fn prologue() -> Result<(Cli, String, ModelConfig)> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("eventage_studio=info,eventage=warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cwd = match &cli.cwd {
        Some(dir) => {
            std::fs::canonicalize(dir).with_context(|| format!("cannot open workspace '{dir}'"))?
        }
        None => std::env::current_dir()?,
    }
    .display()
    .to_string();

    // Settings, model and credential scrubbing, shared with the desktop shell
    // so the same workspace resolves to the same model in a window as in a
    // terminal.
    let model = eventage_studio::launch::resolve_model(&cwd, cli.model.clone());

    Ok((cli, cwd, model))
}

fn main() -> Result<()> {
    // Before anything else, including argument parsing: a process started
    // with the sandbox marker is not really this program, it is a trampoline
    // that confines itself and execs the real command. Never returns when it
    // matches.
    eventage_code::shell_sandbox::run_if_helper();

    let (cli, cwd, model) = prologue()?;
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run(cli, cwd, model))
}

async fn run(cli: Cli, cwd: String, model: ModelConfig) -> Result<()> {
    // Saved settings layer over what the environment resolved, so a provider
    // chosen in the app survives a restart. Built before the backend because
    // both of them read it at session-open time.
    let state_dir =
        eventage_code::config::SessionConfig::new(cwd.clone(), model.clone()).state_dir();
    tokio::fs::create_dir_all(&state_dir).await.ok();
    let settings =
        Arc::new(eventage_studio::model_settings::ModelSettings::load(model, &state_dir).await);

    let backend: Arc<dyn Backend> = match (cli.acp.clone(), cli.cowork) {
        (Some(command), _) => Arc::new(AcpBackend::new(command, cwd.clone())?),
        // Cowork over the same folder: a goal fans into workstreams that each
        // work in a private copy, and nothing lands until one is kept.
        (None, true) => Arc::new(
            CoworkBackend::new(Arc::clone(&settings), cwd.clone())
                .with_limits(cli.parallel, cli.split),
        ),
        (None, false) => Arc::new(LocalBackend::new(Arc::clone(&settings), cwd.clone()).await),
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
        (
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            &[],
        ),
        ("open", &[]),
    ];
    #[cfg(target_os = "windows")]
    let candidates: [(&str, &[&str]); 2] = [("chrome", &[]), ("cmd", &["/C", "start", ""])];
    #[cfg(all(unix, not(target_os = "macos")))]
    let candidates: Vec<(&str, &[&str])> = {
        let mut found: Vec<(&str, &[&str])> = vec![
            ("google-chrome", &[]),
            ("chromium", &[]),
            ("chromium-browser", &[]),
            ("microsoft-edge", &[]),
            ("brave-browser", &[]),
        ];

        // Under WSL there is usually no Linux browser at all, and the useful
        // one is on the Windows side. It understands `--app` like any other
        // Chromium, and `127.0.0.1` reaches back into WSL because WSL2
        // forwards localhost — so this gives a real chromeless window rather
        // than the "could not open anything" fallback.
        //
        // Tried before `xdg-open`, which on a browserless WSL install
        // succeeds at doing nothing and would otherwise win.
        if std::fs::read_to_string("/proc/version")
            .map(|v| v.to_ascii_lowercase().contains("microsoft"))
            .unwrap_or(false)
        {
            found.extend([
                (
                    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
                    &[] as &[&str],
                ),
                (
                    "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe",
                    &[] as &[&str],
                ),
                (
                    "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
                    &[] as &[&str],
                ),
                (
                    "/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe",
                    &[] as &[&str],
                ),
                // The idiomatic WSL opener, if wslu is installed. No app mode.
                ("wslview", &[]),
            ]);
        }
        found.push(("xdg-open", &[]));
        found
    };

    for (program, prefix) in candidates {
        // Only the Chromium family understands `--app`; anything else gets
        // the bare URL and opens a tab.
        let app_mode = !program.contains("open") && program != "cmd" && program != "wslview";
        // A path that is not there is not worth a spawn attempt, and skipping
        // it keeps the Windows candidates from masking `xdg-open`.
        if program.starts_with('/') && !std::path::Path::new(program).is_file() {
            continue;
        }
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
