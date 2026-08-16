//! cowork — hand a folder and a goal to a session you can navigate.
//!
//! ```sh
//! # Work on a folder, asking before anything risky (the default).
//! cowork --folder ~/reports "turn the Q3 notes into a summary and a chart"
//!
//! # Fan wider, and do not interrupt.
//! cowork --folder ~/reports --steering skip --parallel 5 "reorganise this"
//!
//! # Reachable from elsewhere while it runs.
//! cowork --folder ~/reports --port 4600 "draft the board update"
//! ```
//!
//! The session prints what each workstream produced and stops. Nothing is
//! written into the folder until a result is adopted — `--adopt` takes the
//! best one automatically, and without it the run is a comparison you decide
//! on.

use anyhow::{Context, Result};
use clap::Parser;
use cowork::session::{CoworkConfig, CoworkSession, Status};
use cowork::steering::Steering;
use eventage_code::config::ModelConfig;
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "cowork", version, about = "A working session you can navigate")]
struct Cli {
    /// What to work on.
    goal: Vec<String>,

    /// The folder to work in. Defaults to the current directory.
    #[arg(long)]
    folder: Option<String>,

    /// How much happens without asking: manual | auto | skip.
    #[arg(long, default_value = "auto")]
    steering: String,

    /// How many workstreams may run at once.
    ///
    /// The expensive dial. Fanning out is what makes these sessions fast and
    /// what makes them costly, so it is a flag rather than a hidden default.
    #[arg(long, default_value_t = 3)]
    parallel: usize,

    /// How many workstreams a goal may be split into.
    ///
    /// Separate from `--parallel`: how far a goal divides is a property of the
    /// goal, how much runs together is a property of your budget.
    #[arg(long, default_value_t = 5)]
    split: usize,

    /// Tokens one workstream may spend before it is stopped.
    #[arg(long, default_value_t = 200_000)]
    budget: u64,

    /// Apply the workstream that changed the most files, without asking.
    #[arg(long)]
    adopt: bool,

    /// Serve the HTTP channel on this port, for steering from elsewhere.
    #[arg(long)]
    port: Option<u16>,

    /// The model to use.
    #[arg(long)]
    model: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Before anything parses arguments: a confined command re-executes this
    // binary's sibling trampoline, and the check has to come first.
    eventage_code::shell_sandbox::run_if_helper();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "cowork=info".into()))
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let goal = cli.goal.join(" ");
    if goal.trim().is_empty() {
        anyhow::bail!("say what to work on, e.g. cowork \"summarise the Q3 notes\"");
    }

    let folder = match &cli.folder {
        Some(f) => std::path::PathBuf::from(f),
        None => std::env::current_dir()?,
    };

    let settings = eventage_code::settings::ClaudeSettings::load(&folder);
    let _ = settings.apply_env();
    let model = ModelConfig::from_env(cli.model.or(settings.model));
    // Everything that reads a credential has run; take them out of the
    // environment, where any process of this user could read them.
    let _ = eventage_code::secrets::capture_and_scrub();

    let steering = Steering::from_id(&cli.steering).with_context(|| {
        format!(
            "unknown --steering '{}' (expected {})",
            cli.steering,
            Steering::NAMES
        )
    })?;

    let mut config = CoworkConfig::new(&folder);
    config.steering = steering;
    config.max_parallel = cli.parallel;
    config.max_workstreams = cli.split;
    config.token_budget = cli.budget;

    let llm = eventage_code::agent::provider_for(&model);
    let session = Arc::new(
        CoworkSession::open(uuid::Uuid::new_v4().to_string(), llm, config)
            .await
            .context("could not open the session")?,
    );

    if let Some(port) = cli.port {
        let token = uuid::Uuid::new_v4().to_string();
        let addr = cowork::channels::http::serve(session.bus.clone(), token.clone(), port).await?;
        // The token goes to stderr with the address, never into the event log.
        eprintln!("steer from elsewhere: http://{addr}  token: {token}");
    }

    println!("folder:   {}", folder.display());
    println!("steering: {} — {}", steering.id(), steering.describe());
    println!("goal:     {goal}\n");

    let streams = session.run(&goal).await?;

    println!("\n── workstreams ──");
    for stream in &streams {
        let mark = match stream.status {
            Status::Done => "✓",
            Status::Failed => "✗",
            Status::Sealed => "—",
            _ => "·",
        };
        println!(
            "{mark} {}  [{}]  {} file(s) changed",
            stream.title,
            stream.id,
            stream.changes.len()
        );
        if let Some(report) = &stream.report {
            for line in report.lines().take(6) {
                println!("    {line}");
            }
        }
    }

    // Nothing reaches the folder unless it is adopted. The default is a
    // comparison the user decides on, because the whole point of running
    // several is that they are not equally good.
    match cli.adopt {
        false => {
            println!(
                "\nNothing has been written to your folder. Each result is kept in the \
                 session's shadow repository; adopt one to apply it."
            );
        }
        true => {
            let best = streams
                .iter()
                .filter(|s| s.status == Status::Done)
                .max_by_key(|s| s.changes.len());
            match best {
                None => println!("\nNo workstream produced anything to adopt."),
                Some(stream) => {
                    let changes = session.adopt(&stream.id).await?;
                    println!(
                        "\nAdopted '{}': {} file(s) written to your folder.",
                        stream.title,
                        changes.len()
                    );
                }
            }
        }
    }

    Ok(())
}
