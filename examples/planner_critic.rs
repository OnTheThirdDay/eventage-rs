//! Planner-Critic example — iterative refinement via a feedback loop.
//!
//! Demonstrates building non-ReAct patterns by composing core primitives.
//!
//! # Architecture
//! - Planner creates `plan.draft` events.
//! - Critic (`EventWorker`) scores drafts and emits `plan.feedback` or `plan.accepted`.
//! - Planner's context assembler injects prior drafts and feedback for refinement.
//! - Orchestration is driven by an external loop, limiting the planner to single-pass generations.
//!
//! # Run
//! Uses `MockLlmProvider` so no Ollama is required.
//! ```bash
//! cargo run -p example-planner-critic
//! ```

use async_trait::async_trait;
use eventage::{kinds, Event, EventBus};
use eventage::llm::{types::ChatMessage, MockLlmProvider};
use eventage::{
    AgentBuilder, AssemblyContext, ContextAssembler, EventWorker, ReactStrategy, WorkerError,
    WorkerSet,
};
use serde_json::json;

// ── Custom event kinds ────────────────────────────────────────────────────────

mod plan_kinds {
    /// Planner produced a draft; payload: `{ content, round }`.
    pub const DRAFT: &str = "plan.draft";
    /// Critic rejected the draft; payload: `{ score, feedback, round }`.
    pub const FEEDBACK: &str = "plan.feedback";
    /// Critic accepted the draft; payload: `{ score, round }`.
    pub const ACCEPTED: &str = "plan.accepted";
}

// ── PlannerAssembler ──────────────────────────────────────────────────────────

/// Assembles the planner's prompt, injecting prior draft+feedback pairs
/// so the LLM knows what to improve on each refinement round.
struct PlannerAssembler;

#[async_trait]
impl ContextAssembler for PlannerAssembler {
    async fn assemble(&self, ctx: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        // Find the task.
        let task = ctx
            .events
            .iter()
            .find(|e| e.kind == kinds::USER_MESSAGE)
            .and_then(|e| e.payload["text"].as_str())
            .unwrap_or_default();

        if task.is_empty() {
            return vec![];
        }

        let mut messages = vec![
            ChatMessage::system(
                "You are an expert software architect. When given a task, produce a clear, \
                 numbered plan covering: error handling, testing, authentication/security, \
                 and monitoring/logging. Be thorough but concise.",
            ),
            ChatMessage::user(task),
        ];

        // Pair up draft events with their feedback events and inject them as
        // an alternating assistant/user turn sequence. This gives the LLM
        // full context about what it previously wrote and why it was rejected.
        let drafts: Vec<&Event> = ctx
            .events
            .iter()
            .filter(|e| e.kind == plan_kinds::DRAFT)
            .collect();

        let feedbacks: Vec<&Event> = ctx
            .events
            .iter()
            .filter(|e| e.kind == plan_kinds::FEEDBACK)
            .collect();

        for (draft_event, fb_event) in drafts.iter().zip(feedbacks.iter()) {
            let draft_text = draft_event.payload["content"].as_str().unwrap_or("");
            let score = fb_event.payload["score"].as_f64().unwrap_or(0.0);
            let feedback = fb_event.payload["feedback"].as_str().unwrap_or("");

            messages.push(ChatMessage::assistant(draft_text));
            messages.push(ChatMessage::user(format!(
                "Critic score: {score:.0}/10. Issues found:\n  {feedback}\n\n\
                 Please revise the plan to address ALL of the above issues."
            )));
        }

        messages
    }
}

// ── CriticWorker ──────────────────────────────────────────────────────────────

/// Programmatic critic evaluating plans against quality criteria.
///
/// Publishes `plan.accepted` when passing, or `plan.feedback` with notes when failing.
/// In production, this can be seamlessly replaced with an LLM-based scorer.
struct CriticWorker {
    threshold: f64,
}

#[async_trait]
impl EventWorker for CriticWorker {
    fn subscribed_kinds(&self) -> Vec<String> {
        vec![plan_kinds::DRAFT.to_string()]
    }

    async fn handle(&self, event: &Event, bus: &EventBus) -> Result<(), WorkerError> {
        let content = event.payload["content"]
            .as_str()
            .unwrap_or("")
            .to_lowercase();
        let round = event.payload["round"].as_u64().unwrap_or(1);

        // Score on five criteria, 2 points each (max 10).
        let mut score = 0.0f64;
        let mut issues = Vec::new();

        if content.contains("error") || content.contains("exception") || content.contains("fail") {
            score += 2.0;
        } else {
            issues.push("missing error handling");
        }

        if content.contains("test") || content.contains("spec") || content.contains("coverage") {
            score += 2.0;
        } else {
            issues.push("missing testing strategy");
        }

        if content.contains("auth") || content.contains("security") || content.contains("token") {
            score += 2.0;
        } else {
            issues.push("missing authentication / security");
        }

        // Structured numbering: check for a line starting with a digit.
        let has_numbers = content
            .lines()
            .any(|l| l.trim_start().starts_with(|c: char| c.is_ascii_digit()));
        if has_numbers {
            score += 2.0;
        } else {
            issues.push("plan should use numbered steps");
        }

        if content.contains("monitor") || content.contains("log") || content.contains("observ") {
            score += 2.0;
        } else {
            issues.push("missing monitoring / logging");
        }

        let verdict = if issues.is_empty() {
            "all criteria met".to_string()
        } else {
            issues.join("; ")
        };

        println!("  [Critic]  Round {round}: {score:.0}/10 — {verdict}");

        if score >= self.threshold {
            bus.publish(Event::new(
                plan_kinds::ACCEPTED,
                json!({ "score": score, "round": round }),
            ))
            .await
            .map_err(WorkerError::Bus)?;
        } else {
            bus.publish(Event::new(
                plan_kinds::FEEDBACK,
                json!({ "score": score, "feedback": issues.join("; "), "round": round }),
            ))
            .await
            .map_err(WorkerError::Bus)?;
        }

        Ok(())
    }
}

// ── Mock LLM: simulates improving plans ──────────────────────────────────────

fn planner_llm() -> MockLlmProvider {
    MockLlmProvider::with_texts([
        // Round 1: bare-bones (will fail the critic: no auth, no tests, no monitoring).
        "Here is the plan:\n\
         1. Set up the project repository.\n\
         2. Implement the API endpoints.\n\
         3. Handle errors where they occur.\n\
         4. Deploy to the server.",
        // Round 2: improved after critic feedback (all five criteria met).
        "Revised plan addressing all critic feedback:\n\
         1. Set up project with structured logging and configuration management.\n\
         2. Implement API endpoints with full input validation and error handling.\n\
         3. Add authentication middleware using JWT tokens; enforce HTTPS.\n\
         4. Write unit tests and integration tests targeting ≥80% coverage.\n\
         5. Add monitoring with health-check endpoints, metrics, and alerting.\n\
         6. Deploy with a rollback strategy and post-deploy smoke tests.",
    ])
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("warn")
        .with_writer(std::io::stderr)
        .init();

    let bus = EventBus::new();

    // ── Start the critic worker ───────────────────────────────────────────────
    // The critic runs concurrently as an EventWorker. It scores every
    // `plan.draft` event and publishes `plan.accepted` or `plan.feedback`.
    let worker_bus = bus.clone();
    tokio::spawn(async move {
        WorkerSet::new()
            .add_worker(CriticWorker { threshold: 7.0 })
            .run_on(worker_bus)
            .await
            .ok();
    });

    // ── Build the planner agent ───────────────────────────────────────────────
    // `max_react_steps: 1` enforces a single-pass generation (no tool loop).
    // The refinement loop is driven externally.
    let agent = AgentBuilder::new()
        .agent_id("planner")
        .bus(bus.clone())
        .context(PlannerAssembler)
        .llm(planner_llm())
        .strategy(ReactStrategy {
            max_steps: 1,
            max_concurrent_tools: 4,
        })
        .build();

    // ── Publish the task ──────────────────────────────────────────────────────
    let task = "Design a plan for building a production-ready REST API service.";
    println!("Task: {task}\n");
    bus.publish(Event::new(kinds::USER_MESSAGE, json!({ "text": task })))
        .await?;

    // ── Refinement loop ───────────────────────────────────────────────────────
    // Drive the planner iteratively, waiting for the critic's verdict after each draft.
    let max_rounds = 5;
    for round in 1..=max_rounds {
        println!("--- Round {round} ---");

        // One LLM call: generate (or refine) the plan.
        agent.cycle().await?;

        // Extract the plan from the latest assistant.message on the bus.
        let log = bus.log().await;
        let draft = log
            .iter()
            .rev()
            .find(|e| e.kind == kinds::ASSISTANT_MESSAGE)
            .and_then(|e| e.payload["content"].as_str())
            .unwrap_or("")
            .to_string();

        if draft.is_empty() {
            println!("  [Planner] No plan produced.");
            break;
        }

        println!("  [Planner] Draft:\n{}\n", indent(&draft, "    "));

        // Signal the critic by publishing a plan.draft event.
        // The critic is an async EventWorker — use wait_for() to synchronise.
        bus.publish(Event::new(
            plan_kinds::DRAFT,
            json!({ "content": draft, "round": round }),
        ))
        .await?;

        // Block until the critic publishes its verdict.
        let verdict = bus
            .wait_for(|e| e.kind == plan_kinds::ACCEPTED || e.kind == plan_kinds::FEEDBACK)
            .await?;

        if verdict.kind == plan_kinds::ACCEPTED {
            let score = verdict.payload["score"].as_f64().unwrap_or(0.0);
            println!("\n✓ Plan accepted after {round} round(s) — score {score:.0}/10.");
            break;
        }

        if round == max_rounds {
            println!("\n✗ Max rounds reached. Using current plan.");
        }
        // Otherwise, PlannerAssembler injects feedback on the next cycle.
    }

    // ── Event log ─────────────────────────────────────────────────────────────
    let log = bus.log().await;
    println!("\n{}", "─".repeat(60));
    println!("Event log ({} events):", log.len());
    for event in &log {
        match event.kind.as_str() {
            kinds::USER_MESSAGE => {
                println!(
                    "  [user.message]   {}",
                    event.payload["text"].as_str().unwrap_or("")
                );
            }
            kinds::AGENT_CYCLE_START => println!("  [cycle.start]    planner"),
            kinds::AGENT_CYCLE_END => println!("  [cycle.end]      planner"),
            kinds::ASSISTANT_MESSAGE => {
                let first_line = event.payload["content"]
                    .as_str()
                    .unwrap_or("")
                    .lines()
                    .next()
                    .unwrap_or("")
                    .chars()
                    .take(55)
                    .collect::<String>();
                println!("  [assistant]      {first_line}…");
            }
            plan_kinds::DRAFT => {
                println!("  [plan.draft]     round={}", event.payload["round"]);
            }
            plan_kinds::FEEDBACK => {
                println!(
                    "  [plan.feedback]  score={} — {}",
                    event.payload["score"],
                    event.payload["feedback"].as_str().unwrap_or("")
                );
            }
            plan_kinds::ACCEPTED => {
                println!(
                    "  [plan.accepted]  score={} round={}",
                    event.payload["score"], event.payload["round"]
                );
            }
            _ => {}
        }
    }

    Ok(())
}

fn indent(s: &str, prefix: &str) -> String {
    s.lines()
        .map(|l| format!("{prefix}{l}"))
        .collect::<Vec<_>>()
        .join("\n")
}
