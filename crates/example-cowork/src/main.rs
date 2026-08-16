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

    /// Reopen a session by id, with its workstreams intact.
    ///
    /// The results of a previous run are still in its shadow repository, so a
    /// session interrupted halfway can be picked up rather than restarted.
    #[arg(long, value_name = "SESSION_ID")]
    resume: Option<String>,
}

/// Everything that touches the process environment, before the runtime exists.
///
/// `set_var` and `remove_var` are unsafe on Unix because another thread
/// calling `getenv` concurrently is undefined behaviour. `#[tokio::main]`
/// builds the runtime — and its worker threads — before the first line of the
/// async body, so doing this there was exactly what the rule forbids.
fn prologue() -> Result<(Cli, ModelConfig)> {
    let cli = Cli::parse();
    let folder = match &cli.folder {
        Some(f) => std::path::PathBuf::from(f),
        None => std::env::current_dir()?,
    };
    let settings = eventage_code::settings::ClaudeSettings::load(&folder);
    let _ = settings.apply_env();
    let model = ModelConfig::from_env(cli.model.clone().or(settings.model));
    let _ = eventage_code::secrets::capture_and_scrub();
    Ok((cli, model))
}

fn main() -> Result<()> {
    // Before anything parses arguments: a confined command re-executes this
    // binary's sibling trampoline, and the check has to come first.
    eventage_code::shell_sandbox::run_if_helper();

    let (cli, model) = prologue()?;
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run(cli, model))
}

async fn run(cli: Cli, model: ModelConfig) -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "cowork=info".into()))
        .with_writer(std::io::stderr)
        .init();

    let goal = cli.goal.join(" ");
    // A goal is required to start, but not to reopen: the point of reopening
    // is usually to look at what a previous run produced and keep or abandon
    // one of its results. Demanding a goal meant every resume immediately ran
    // something new and replaced what had just been reopened.
    if goal.trim().is_empty() && cli.resume.is_none() {
        anyhow::bail!("say what to work on, e.g. cowork \"summarise the Q3 notes\"");
    }

    let folder = match &cli.folder {
        Some(f) => std::path::PathBuf::from(f),
        None => std::env::current_dir()?,
    };

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
    let session = Arc::new(match &cli.resume {
        Some(id) => CoworkSession::resume(id.clone(), llm, config)
            .await
            .with_context(|| format!("could not reopen session {id}"))?,
        None => CoworkSession::open(uuid::Uuid::new_v4().to_string(), llm, config)
            .await
            .context("could not open the session")?,
    });
    println!("session:  {}", session.id);

    // What a reopened session already knows.
    let existing = session.workstreams().await;
    if !existing.is_empty() {
        println!("reopened with {} workstream(s) from before", existing.len());
    }

    // Goals arriving from anywhere but this command line — the HTTP channel,
    // an automation — are run by this. Without it those surfaces accepted
    // requests and dropped them.
    let requests = tokio::spawn(Arc::clone(&session).serve_requests());

    if let Some(port) = cli.port {
        let token = uuid::Uuid::new_v4().to_string();
        let addr = cowork::channels::http::serve(session.bus.clone(), token.clone(), port).await?;
        // The token goes to stderr with the address, never into the event log.
        eprintln!("steer from elsewhere: http://{addr}  token: {token}");
    }

    println!("folder:   {}", folder.display());
    println!("steering: {} — {}", steering.id(), steering.describe());

    let streams = match goal.trim().is_empty() {
        // Reopened with nothing new to do: show what is there and stop.
        true => {
            println!("goal:     (reopened; nothing new asked for)\n");
            existing
        }
        false => {
            println!("goal:     {goal}\n");
            session.run(&goal).await?
        }
    };

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
                    let outcome = session.adopt(&stream.id).await?;
                    match outcome.conflicts.is_empty() {
                        true => println!(
                            "\nAdopted '{}': {} file(s) written to your folder.",
                            stream.title,
                            outcome.applied.len()
                        ),
                        // Nothing was written. Naming the files is the whole
                        // point: the user has two versions and has to choose.
                        false => {
                            println!(
                                "\nNot adopted — '{}' and your folder both changed these \
                                 since the session started:",
                                stream.title
                            );
                            for conflict in &outcome.conflicts {
                                println!(
                                    "    {}  (workstream {}, yours {})",
                                    conflict.path,
                                    conflict.workstream.as_str(),
                                    conflict.live.as_str()
                                );
                            }
                            println!(
                                "\nYour folder is untouched. Move or keep your versions, \
                                 then adopt again."
                            );
                        }
                    }
                }
            }
        }
    }

    requests.abort();
    // The log is flushed before the process says it is done; without this the
    // last events — the ones a resume needs — are dropped with the task.
    let failures = session.close().await;
    if failures > 0 {
        eprintln!(
            "warning: {failures} event(s) never reached the log; this session will \
                   reopen incomplete"
        );
    }
    Ok(())
}
