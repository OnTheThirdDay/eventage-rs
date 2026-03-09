//! Workflow example — iterative PRD writing with structured quality review cycles.
//!
//! Demonstrates a bounded `prd-reviewer -> prd-refiner -> prd-reviewer` loop.
//!
//! # Key design points
//! - **Cycle**: Each refinement round improves the PRD until the reviewer approves it or hits a max round limit.
//! - **Structured output**: The reviewer emits a JSON object (`score`, `approved`, `missing_requirements`, `feedback`). The `ReviewRouter` worker parses this to make routing decisions.
//! - **Data flow via events**: Handoffs use `agent.message` with `TO_AGENT_ID` metadata. Workers correlate outputs via `trace_id`.
//! - **Custom context assemblers**: Agents filter the bus log to only relevant messages, keeping context focused and noise-free.
//!
//! # Run
//! ```bash
//! cargo run -p example-workflow
//! ```

use async_trait::async_trait;
use eventage_core::{kinds, meta_keys, Event, EventBus};
use eventage_llm::{types::ChatMessage, LlmProvider, OpenAiProvider};
use eventage_provided_impl::{
    context::{AssemblyContext, ContextAssembler},
    strategy::SingleShotStrategy,
    worker::{EventWorker, WorkerError, WorkerSet},
    AgentBuilder, AgentSet,
};
use serde_json::json;
use std::sync::Arc;

// ── Agent IDs ─────────────────────────────────────────────────────────────────

const WRITER_ID: &str = "prd-writer";
const REVIEWER_ID: &str = "prd-reviewer";
const REFINER_ID: &str = "prd-refiner";
const SUMMARISER_ID: &str = "exec-summariser";

// ── Custom event kind ─────────────────────────────────────────────────────────

const WORKFLOW_DONE: &str = "workflow.done";

// ── Configuration ─────────────────────────────────────────────────────────────

/// Maximum review-refine rounds before the workflow proceeds regardless.
const MAX_REVIEW_ROUNDS: usize = 3;

// ── Context Assemblers ────────────────────────────────────────────────────────

/// Assembler for `prd-writer`.
///
/// Collects the initial `user.message` feature request and prior responses.
struct WriterAssembler;

#[async_trait]
impl ContextAssembler for WriterAssembler {
    async fn assemble(&self, ctx: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let mut task_messages: Vec<ChatMessage> = Vec::new();

        for event in ctx.events {
            if event.kind == kinds::USER_MESSAGE {
                let text = event
                    .payload
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                task_messages.push(ChatMessage::user(text));
            } else if event.kind == kinds::ASSISTANT_MESSAGE {
                let from = event
                    .metadata
                    .get(meta_keys::AGENT_ID)
                    .and_then(|v| v.as_str());
                if from == Some(WRITER_ID) {
                    let content = event
                        .payload
                        .get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    task_messages.push(ChatMessage::assistant(content));
                }
            }
        }

        if task_messages.is_empty() {
            return vec![];
        }

        let mut messages = vec![ChatMessage::system(
            "You are a senior product manager. Given a feature request, write a comprehensive \
             Product Requirements Document (PRD) with the following sections:\n\
             1. Overview\n\
             2. Goals\n\
             3. User Stories\n\
             4. Functional Requirements\n\
             5. Non-Functional Requirements\n\
             6. Success Metrics\n\
             7. Out of Scope\n\n\
             Be thorough and specific. Each section must have concrete, measurable details.",
        )];
        messages.extend(task_messages);
        messages
    }
}

/// Generic assembler for directed-message agents.
///
/// Only includes `agent.message` events routed to this agent and its own responses.
struct DirectedAssembler {
    agent_id: &'static str,
    system_prompt: String,
}

#[async_trait]
impl ContextAssembler for DirectedAssembler {
    async fn assemble(&self, ctx: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let mut task_messages: Vec<ChatMessage> = Vec::new();

        for event in ctx.events {
            let to = event
                .metadata
                .get(meta_keys::TO_AGENT_ID)
                .and_then(|v| v.as_str());
            let from = event
                .metadata
                .get(meta_keys::AGENT_ID)
                .and_then(|v| v.as_str());

            if event.kind == kinds::AGENT_MESSAGE && to == Some(self.agent_id) {
                let text = event
                    .payload
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                task_messages.push(ChatMessage::user(text));
            } else if event.kind == kinds::ASSISTANT_MESSAGE && from == Some(self.agent_id) {
                let content = event
                    .payload
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                task_messages.push(ChatMessage::assistant(content));
            }
        }

        if task_messages.is_empty() {
            return vec![];
        }

        let mut messages = vec![ChatMessage::system(&self.system_prompt)];
        messages.extend(task_messages);
        messages
    }
}

// ── Helper: find assistant.message for a specific cycle ──────────────────────

/// Return the text content of the `assistant.message` that belongs to the
/// given cycle, identified by `trace_id`.
fn find_cycle_output<'a>(log: &'a [Event], trace_id: &str) -> Option<&'a str> {
    log.iter()
        .rev()
        .find(|e| {
            e.kind == kinds::ASSISTANT_MESSAGE
                && e.metadata.get(meta_keys::TRACE_ID).and_then(|v| v.as_str()) == Some(trace_id)
        })
        .and_then(|e| e.payload.get("content").and_then(|v| v.as_str()))
}

// ── Workers ───────────────────────────────────────────────────────────────────

/// After `prd-writer` or `prd-refiner` completes a cycle, extract the PRD
/// and forward it to `prd-reviewer`.
struct PrdBridge;

#[async_trait]
impl EventWorker for PrdBridge {
    fn subscribed_kinds(&self) -> Vec<String> {
        vec![kinds::AGENT_CYCLE_END.to_string()]
    }

    async fn handle(&self, event: &Event, bus: &EventBus) -> Result<(), WorkerError> {
        let agent_id = event
            .metadata
            .get(meta_keys::AGENT_ID)
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if agent_id != WRITER_ID && agent_id != REFINER_ID {
            return Ok(());
        }

        let label = if agent_id == WRITER_ID {
            "initial"
        } else {
            "revised"
        };
        let trace_id = event
            .metadata
            .get(meta_keys::TRACE_ID)
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let log = bus.log().await;
        let prd = match find_cycle_output(&log, trace_id) {
            Some(p) if !p.is_empty() => p.to_string(),
            _ => return Ok(()),
        };

        eprintln!(
            "[prd-bridge] forwarding {label} PRD ({} chars) to reviewer",
            prd.len()
        );

        bus.publish(
            Event::new(
                kinds::AGENT_MESSAGE,
                json!({ "text": format!("Please review this {label} PRD:\n\n{prd}") }),
            )
            .with_meta(meta_keys::TO_AGENT_ID, json!(REVIEWER_ID)),
        )
        .await
        .map_err(WorkerError::Bus)
    }
}

/// After `prd-reviewer` completes a cycle, parses output to route to summariser
/// if approved (or max rounds), else routes to refiner.
struct ReviewRouter {
    max_rounds: usize,
}

impl ReviewRouter {
    fn new(max_rounds: usize) -> Self {
        Self { max_rounds }
    }
}

#[async_trait]
impl EventWorker for ReviewRouter {
    fn subscribed_kinds(&self) -> Vec<String> {
        vec![kinds::AGENT_CYCLE_END.to_string()]
    }

    async fn handle(&self, event: &Event, bus: &EventBus) -> Result<(), WorkerError> {
        let agent_id = event
            .metadata
            .get(meta_keys::AGENT_ID)
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if agent_id != REVIEWER_ID {
            return Ok(());
        }

        let trace_id = event
            .metadata
            .get(meta_keys::TRACE_ID)
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let log = bus.log().await;

        // ── Parse the reviewer's structured JSON output ────────────────────────
        let raw = find_cycle_output(&log, trace_id).unwrap_or("{}");

        // Strip optional markdown code fences that some models add.
        let clean = raw
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        let review: serde_json::Value = serde_json::from_str(clean).unwrap_or_else(|_| {
            json!({
                "score": 0,
                "approved": false,
                "missing_requirements": [],
                "feedback": raw
            })
        });

        let approved = review
            .get("approved")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let score = review.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let feedback = review
            .get("feedback")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let missing: Vec<String> = review
            .get("missing_requirements")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|s| s.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        // ── Count completed review cycles ──────────────────────────────────────
        let review_count = log
            .iter()
            .filter(|e| {
                e.kind == kinds::AGENT_CYCLE_END
                    && e.metadata.get(meta_keys::AGENT_ID).and_then(|v| v.as_str())
                        == Some(REVIEWER_ID)
            })
            .count();

        eprintln!("[review-router] round={review_count}, score={score:.0}/10, approved={approved}");

        // ── Recover the PRD that was submitted for review ──────────────────────
        // The most recent agent.message directed to prd-reviewer is the PRD.
        let latest_prd = log
            .iter()
            .rev()
            .find(|e| {
                e.kind == kinds::AGENT_MESSAGE
                    && e.metadata
                        .get(meta_keys::TO_AGENT_ID)
                        .and_then(|v| v.as_str())
                        == Some(REVIEWER_ID)
            })
            .and_then(|e| e.payload.get("text").and_then(|v| v.as_str()))
            // Strip the "Please review this ... PRD:\n\n" prefix to get the raw PRD text.
            .and_then(|s| s.split_once("\n\n").map(|(_, prd)| prd))
            .unwrap_or("")
            .to_string();

        if approved || review_count >= self.max_rounds {
            if !approved {
                eprintln!(
                    "[review-router] max rounds reached (score={score:.0}/10) — \
                     proceeding to executive summary"
                );
            } else {
                eprintln!("[review-router] PRD approved — proceeding to executive summary");
            }
            // Forward the approved (or best-effort) PRD to the executive summariser.
            bus.publish(
                Event::new(kinds::AGENT_MESSAGE, json!({ "text": latest_prd }))
                    .with_meta(meta_keys::TO_AGENT_ID, json!(SUMMARISER_ID)),
            )
            .await
            .map_err(WorkerError::Bus)
        } else {
            // Route back to prd-refiner with the PRD and structured feedback.
            let missing_list = if missing.is_empty() {
                "None listed".to_string()
            } else {
                missing
                    .iter()
                    .map(|s| format!("- {s}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            };

            let refine_msg = format!(
                "The following PRD needs improvement based on reviewer feedback.\n\n\
                 ## Current PRD\n\
                 {latest_prd}\n\n\
                 ## Reviewer Feedback (score: {score:.0}/10)\n\
                 {feedback}\n\n\
                 ## Missing Requirements\n\
                 {missing_list}\n\n\
                 Rewrite the complete PRD, addressing every point of feedback and \
                 filling in all missing requirements."
            );

            eprintln!(
                "[review-router] routing to refiner — feedback: {} chars, missing: {}",
                feedback.len(),
                missing.len()
            );

            bus.publish(
                Event::new(kinds::AGENT_MESSAGE, json!({ "text": refine_msg }))
                    .with_meta(meta_keys::TO_AGENT_ID, json!(REFINER_ID)),
            )
            .await
            .map_err(WorkerError::Bus)
        }
    }
}

/// After `exec-summariser` completes a cycle, publish `workflow.done` with
/// the executive summary, signalling the workflow is complete.
struct SummaryBridge;

#[async_trait]
impl EventWorker for SummaryBridge {
    fn subscribed_kinds(&self) -> Vec<String> {
        vec![kinds::AGENT_CYCLE_END.to_string()]
    }

    async fn handle(&self, event: &Event, bus: &EventBus) -> Result<(), WorkerError> {
        let agent_id = event
            .metadata
            .get(meta_keys::AGENT_ID)
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if agent_id != SUMMARISER_ID {
            return Ok(());
        }

        let trace_id = event
            .metadata
            .get(meta_keys::TRACE_ID)
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let log = bus.log().await;
        let summary = find_cycle_output(&log, trace_id).unwrap_or("[no summary generated]");

        eprintln!("[summary-bridge] workflow complete — publishing workflow.done");

        bus.publish(Event::new(WORKFLOW_DONE, json!({ "summary": summary })))
            .await
            .map_err(WorkerError::Bus)
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("eventage=info,workflow=info")
        .init();

    let bus = EventBus::new();
    let llm: Arc<dyn LlmProvider> = Arc::new(OpenAiProvider::ollama("qwen3:4b"));

    // ── Agents ────────────────────────────────────────────────────────────────

    let prd_writer = AgentBuilder::new()
        .agent_id(WRITER_ID)
        .bus(bus.clone())
        .llm_arc(llm.clone())
        .context(WriterAssembler)
        .strategy(SingleShotStrategy)
        .build();

    let prd_reviewer = AgentBuilder::new()
        .agent_id(REVIEWER_ID)
        .bus(bus.clone())
        .llm_arc(llm.clone())
        .context(DirectedAssembler {
            agent_id: REVIEWER_ID,
            system_prompt: "\
You are a strict PRD quality reviewer. Analyze the PRD provided and respond with ONLY \
a JSON object — no markdown fences, no explanation, just the raw JSON:

{
  \"score\": <integer 1-10>,
  \"approved\": <true if score >= 8, otherwise false>,
  \"missing_requirements\": [\"<specific missing item>\", ...],
  \"feedback\": \"<concise, actionable improvement guidance>\"
}

Scoring criteria:
- 1-4: Major sections missing or too vague to implement
- 5-7: Core sections present but lacking measurable detail
- 8-10: All sections present, concrete success metrics, clear scope

A PRD scoring 8 or above is approved for development."
                .to_string(),
        })
        .strategy(SingleShotStrategy)
        .build();

    let prd_refiner = AgentBuilder::new()
        .agent_id(REFINER_ID)
        .bus(bus.clone())
        .llm_arc(llm.clone())
        .context(DirectedAssembler {
            agent_id: REFINER_ID,
            system_prompt: "\
You are a product manager specializing in PRD refinement. You receive a PRD alongside \
structured reviewer feedback (score, missing requirements, and specific improvement notes). \
Your task: produce a complete, improved version of the PRD that:
- Addresses every point of feedback
- Fills in every missing requirement
- Retains all sections that were already good
- Makes all requirements concrete and measurable

Output the full revised PRD — do not summarise or omit sections."
                .to_string(),
        })
        .strategy(SingleShotStrategy)
        .build();

    let exec_summariser = AgentBuilder::new()
        .agent_id(SUMMARISER_ID)
        .bus(bus.clone())
        .llm_arc(llm.clone())
        .context(DirectedAssembler {
            agent_id: SUMMARISER_ID,
            system_prompt: "\
You are an executive communications specialist. Given a PRD, write a concise 2-3 paragraph \
executive summary suitable for C-level stakeholders. Focus on:
- Business value and strategic fit
- Key deliverables and timeline implications
- Expected outcomes and success criteria

Avoid technical jargon. Write in plain business language."
                .to_string(),
        })
        .strategy(SingleShotStrategy)
        .build();

    // ── Workers ───────────────────────────────────────────────────────────────

    let workers = WorkerSet::new()
        .add_worker(PrdBridge)
        .add_worker(ReviewRouter::new(MAX_REVIEW_ROUNDS))
        .add_worker(SummaryBridge);

    // ── Subscribe before kicking off so we don't miss workflow.done ────────────

    let mut done_rx = bus.subscribe();

    // ── Publish the initial feature request ────────────────────────────────────

    bus.publish(Event::new(
        kinds::USER_MESSAGE,
        json!({
            "text": "Feature request: Build a real-time collaborative code editor with \
                     AI-assisted code completion, inline comments, and integrated CI/CD \
                     pipeline visualization. The tool must support 50+ simultaneous users \
                     per editing session, maintain sub-100ms latency for keystroke sync, \
                     and integrate natively with GitHub, GitLab, and Bitbucket."
        }),
    ))
    .await?;

    // ── Run agents + workers concurrently ──────────────────────────────────────

    let workers_bus = bus.clone();
    let workers_handle = tokio::spawn(async move { workers.run_on(workers_bus).await });

    let agents_handle = tokio::spawn(
        AgentSet::new()
            .add_agent(prd_writer)
            .add_agent(prd_reviewer)
            .add_agent(prd_refiner)
            .add_agent(exec_summariser)
            .run_until_all_complete(),
    );

    // ── Wait for the final workflow.done event ─────────────────────────────────

    loop {
        match done_rx.recv().await {
            Some(event) if event.kind == WORKFLOW_DONE => {
                println!("\n╔══════════════════════════════════════════════════════╗");
                println!("║              Executive Summary                       ║");
                println!("╚══════════════════════════════════════════════════════╝\n");
                println!("{}", event.payload["summary"].as_str().unwrap_or(""));
                break;
            }
            Some(_) => continue,
            None => {
                eprintln!("Bus closed before workflow completed");
                break;
            }
        }
    }

    workers_handle.abort();
    agents_handle.abort();

    Ok(())
}
