//! Multi-agent example — orchestrator/worker pattern via event worker.
//!
//! Architecture:
//! - `[stdin]` -> `bus.publish(user.message)`
//! - `orchestrator.run()` handles user message, calls `delegate_to_summariser`
//! - `orchestrator` publishes `agent.message(to=summariser)`
//! - `summariser.run()` handles task, ends cycle
//! - `SummariserBridge (EventWorker)` detects `CYCLE_END`, routes output back
//!
//! Demonstrates native event-driven orchestration without manual `cycle()` calls.
//!
//! # Run
//! Uses `MockLlmProvider` so no Ollama is required.
//! ```bash
//! cargo run -p example-multi-agent
//! ```

use async_trait::async_trait;
use eventage::{kinds, meta_keys, Event, EventBus};
use eventage::llm::{
    types::{ChatMessage, FunctionCall, LlmResponse, ToolCall, ToolDefinition},
    MockLlmProvider,
};
use eventage::{
    AgentBuilder, AgentError, AgentSet, AssemblyContext, ContextAssembler, EventWorker,
    ReactStrategy, Tool, WorkerError, WorkerSet,
};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

// ── Agent IDs ─────────────────────────────────────────────────────────────────

const ORCHESTRATOR: &str = "orchestrator";
const SUMMARISER: &str = "summariser";

// ── Context assemblers ────────────────────────────────────────────────────────

/// The orchestrator responds to `user.message` and `agent.message` addressed
/// to it. Returns empty when there is nothing relevant to act on.
struct OrchestratorAssembler;

#[async_trait]
impl ContextAssembler for OrchestratorAssembler {
    async fn assemble(&self, ctx: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let has_trigger = ctx.events.iter().any(|e| {
            e.kind == kinds::USER_MESSAGE
                || (e.kind == kinds::AGENT_MESSAGE
                    && (e
                        .metadata
                        .get(meta_keys::TO_AGENT_ID)
                        .and_then(|v| v.as_str())
                        == Some(ORCHESTRATOR)))
        });
        if !has_trigger {
            return vec![]; // skip spurious wakes
        }

        let mut messages = vec![ChatMessage::system(
            "You are an orchestrator. For the user's question, delegate summarisation \
             to the summariser using the delegate_to_summariser tool, then present the \
             result as a final answer.",
        )];

        for event in ctx.events {
            match event.kind.as_str() {
                kinds::USER_MESSAGE => {
                    if let Some(text) = event.payload["text"].as_str() {
                        messages.push(ChatMessage::user(text));
                    }
                }
                kinds::ASSISTANT_MESSAGE => {
                    let aid = event
                        .metadata
                        .get(meta_keys::AGENT_ID)
                        .and_then(|v| v.as_str());
                    if aid == Some(ORCHESTRATOR) {
                        if let Some(content) = event.payload["content"].as_str() {
                            messages.push(ChatMessage::assistant(content));
                        }
                    }
                }
                kinds::TOOL_RESULT => {
                    let tc_id = event.payload["tool_call_id"].as_str().unwrap_or("tool");
                    let result = event.payload.get("result").cloned().unwrap_or_default();
                    messages.push(ChatMessage::tool_result(tc_id, result.to_string()));
                }
                kinds::AGENT_MESSAGE => {
                    let to = event
                        .metadata
                        .get(meta_keys::TO_AGENT_ID)
                        .and_then(|v| v.as_str());
                    if to == Some(ORCHESTRATOR) {
                        if let Some(text) = event.payload["text"].as_str() {
                            messages
                                .push(ChatMessage::user(format!("[Summariser result]: {text}")));
                        }
                    }
                }
                _ => {}
            }
        }
        messages
    }
}

/// The summariser only processes `agent.message` events addressed to it.
struct SummariserAssembler;

#[async_trait]
impl ContextAssembler for SummariserAssembler {
    async fn assemble(&self, ctx: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let task = ctx.events.iter().rev().find(|e| {
            e.kind == kinds::AGENT_MESSAGE
                && (e
                    .metadata
                    .get(meta_keys::TO_AGENT_ID)
                    .and_then(|v| v.as_str())
                    == Some(SUMMARISER))
        });

        let Some(task_event) = task else {
            return vec![]; // nothing to summarise
        };

        vec![
            ChatMessage::system(
                "You are a concise summariser. Summarise the given text in 1-2 sentences.",
            ),
            ChatMessage::user(task_event.payload["text"].as_str().unwrap_or("")),
        ]
    }
}

// ── delegate_to_summariser tool ───────────────────────────────────────────────

/// Published by the orchestrator to route a task to the summariser.
struct DelegateToSummariser {
    bus: EventBus,
}

#[async_trait]
impl Tool for DelegateToSummariser {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "delegate_to_summariser",
            "Send a summarisation task to the summariser agent.",
            json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "The text to summarise." }
                },
                "required": ["text"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let text = args["text"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'text'".into()))?;

        self.bus
            .publish(
                Event::new(kinds::AGENT_MESSAGE, json!({ "text": text }))
                    .with_meta(meta_keys::TO_AGENT_ID, json!(SUMMARISER)),
            )
            .await
            .map_err(|e| AgentError::Tool(e.to_string()))?;

        Ok(json!({ "dispatched": true, "to": SUMMARISER }))
    }
}

// ── SummariserBridge worker ───────────────────────────────────────────────────

/// Watches for the summariser's cycle completion and routes its output to the
/// orchestrator as an `agent.message`.
struct SummariserBridge;

#[async_trait]
impl EventWorker for SummariserBridge {
    fn subscribed_kinds(&self) -> Vec<String> {
        vec![kinds::AGENT_CYCLE_END.to_string()]
    }

    async fn handle(&self, event: &Event, bus: &EventBus) -> Result<(), WorkerError> {
        // Only react to the summariser's cycle end.
        let agent_id = event
            .metadata
            .get(meta_keys::AGENT_ID)
            .and_then(|v| v.as_str());
        if agent_id != Some(SUMMARISER) {
            return Ok(());
        }

        // Find the summariser's most recent assistant message.
        let log = bus.log().await;
        let summary = log
            .iter()
            .rev()
            .find(|e| {
                e.kind == kinds::ASSISTANT_MESSAGE
                    && e.metadata.get(meta_keys::AGENT_ID).and_then(|v| v.as_str())
                        == Some(SUMMARISER)
            })
            .and_then(|e| e.payload["content"].as_str())
            .map(|s| s.to_string());

        if let Some(text) = summary {
            // Route summariser output to orchestrator.
            bus.publish(
                Event::new(kinds::AGENT_MESSAGE, json!({ "text": text }))
                    .with_meta(meta_keys::TO_AGENT_ID, json!(ORCHESTRATOR)),
            )
            .await
            .map_err(WorkerError::Bus)?;
        }

        Ok(())
    }
}

// ── CycleLogger worker ────────────────────────────────────────────────────────

/// Logs statistics after every completed agent cycle.
struct CycleLogger {
    cycles: Arc<AtomicU32>,
}

#[async_trait]
impl EventWorker for CycleLogger {
    fn subscribed_kinds(&self) -> Vec<String> {
        vec![kinds::AGENT_CYCLE_END.to_string()]
    }

    async fn handle(&self, event: &Event, _bus: &EventBus) -> Result<(), WorkerError> {
        let n = self.cycles.fetch_add(1, Ordering::Relaxed) + 1;
        let agent = event
            .metadata
            .get(meta_keys::AGENT_ID)
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let ms = event
            .metadata
            .get(meta_keys::ELAPSED_MS)
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        println!("  [CycleLogger] cycle #{n} — agent={agent} elapsed={ms}ms");
        Ok(())
    }
}

// ── Mock LLM responses ────────────────────────────────────────────────────────

fn orchestrator_llm() -> MockLlmProvider {
    MockLlmProvider::new(vec![
        // Cycle 1: delegate the user's request to the summariser.
        LlmResponse {
            content: Some("I'll ask the summariser to handle this.".into()),
            tool_calls: vec![ToolCall {
                id: "tc_01".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "delegate_to_summariser".into(),
                    arguments: r#"{"text": "The Rust programming language was created by Graydon Hoare at Mozilla Research starting in 2006. It focuses on safety, speed, and concurrency. Rust uses a unique ownership model to guarantee memory safety without a garbage collector. It was first released publicly in 2010 and reached version 1.0 in May 2015."}"#.into(),
                },
            }],
            finish_reason: "tool_calls".into(),
        },
        // Cycle 2: synthesise the summariser's reply (arrives via agent.message).
        LlmResponse {
            content: Some(
                "Based on the summariser's result: Rust is a systems programming language \
                 developed at Mozilla from 2006, known for memory safety through ownership, \
                 speed, and concurrency support. It reached v1.0 in 2015."
                    .into(),
            ),
            tool_calls: vec![],
            finish_reason: "stop".into(),
        },
    ])
}

fn summariser_llm() -> MockLlmProvider {
    MockLlmProvider::with_texts([
        "Rust is a systems language by Mozilla (2006) emphasising memory safety via ownership, \
         speed, and concurrency — stable since 2015.",
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
    let cycles = Arc::new(AtomicU32::new(0));

    // ── Agents ────────────────────────────────────────────────────────────────
    let orchestrator = AgentBuilder::new()
        .agent_id(ORCHESTRATOR)
        .bus(bus.clone())
        .context(OrchestratorAssembler)
        .llm(orchestrator_llm())
        .tool(DelegateToSummariser { bus: bus.clone() })
        .strategy(ReactStrategy::default())
        .build();

    let summariser = AgentBuilder::new()
        .agent_id(SUMMARISER)
        .bus(bus.clone())
        .context(SummariserAssembler)
        .llm(summariser_llm())
        .strategy(ReactStrategy::default())
        .build();

    // ── Spawn agents (event-driven run loops) ─────────────────────────────────
    // Both agents act reactively. No manual .cycle() calls.
    tokio::spawn(
        AgentSet::new()
            .add_agent(orchestrator)
            .add_agent(summariser)
            .run_until_all_complete(),
    );

    // ── Spawn workers ─────────────────────────────────────────────────────────
    // Workers run concurrently alongside agents on the same bus.
    tokio::spawn(
        WorkerSet::new()
            .add_worker(CycleLogger {
                cycles: Arc::clone(&cycles),
            })
            .add_worker(SummariserBridge)
            .run_on(bus.clone()),
    );

    // ── Trigger the pipeline ──────────────────────────────────────────────────
    println!("User: Summarise the history of Rust programming language.\n");
    bus.publish(Event::new(
        kinds::USER_MESSAGE,
        json!({ "text": "Summarise the history of Rust programming language." }),
    ))
    .await?;

    // ── Wait for the final orchestrator answer ────────────────────────────────
    // Blocks until the orchestrator replies without tool calls.
    let final_event = bus
        .wait_for(|e| {
            e.kind == kinds::ASSISTANT_MESSAGE
                && e.metadata.get(meta_keys::AGENT_ID).and_then(|v| v.as_str())
                    == Some(ORCHESTRATOR)
                && e.payload["tool_calls"]
                    .as_array()
                    .is_none_or(|a| a.is_empty())
        })
        .await;

    let answer = final_event.payload["content"]
        .as_str()
        .unwrap_or("(no answer)");
    println!("\nFinal answer:\n  {answer}");

    // ── Show the event log ────────────────────────────────────────────────────
    let log = bus.log().await;
    println!("\n{}", "─".repeat(60));
    println!("Full event log ({} events):", log.len());
    for event in &log {
        let agent = event
            .metadata
            .get(meta_keys::AGENT_ID)
            .and_then(|v| v.as_str())
            .unwrap_or("system");
        match event.kind.as_str() {
            kinds::USER_MESSAGE => {
                println!(
                    "  [user]        {}",
                    event.payload["text"].as_str().unwrap_or("")
                )
            }
            kinds::AGENT_CYCLE_START => println!("  [cycle start] {agent}"),
            kinds::AGENT_CYCLE_END => println!("  [cycle end]   {agent}"),
            kinds::TOOL_CALL_PROPOSED => {
                println!(
                    "  [tool call]   {}",
                    event.payload["name"].as_str().unwrap_or("?")
                )
            }
            kinds::TOOL_RESULT => {
                println!(
                    "  [tool result] {}",
                    event.payload["name"].as_str().unwrap_or("?")
                )
            }
            kinds::ASSISTANT_MESSAGE => {
                let tc = event.payload["tool_calls"]
                    .as_array()
                    .map_or(0, |a| a.len());
                if tc > 0 {
                    println!("  [assistant]   {agent} → {tc} tool call(s)");
                } else if let Some(text) = event.payload["content"].as_str() {
                    let short: String = text.chars().take(70).collect();
                    println!("  [assistant]   {agent}: {short}…");
                }
            }
            kinds::AGENT_MESSAGE => {
                let to = event
                    .metadata
                    .get(meta_keys::TO_AGENT_ID)
                    .and_then(|v| v.as_str())
                    .unwrap_or("broadcast");
                let text = event.payload["text"].as_str().unwrap_or("");
                let short: String = text.chars().take(60).collect();
                println!("  [msg]         {agent} → {to}: {short}…");
            }
            _ => {}
        }
    }

    Ok(())
}
