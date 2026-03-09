//! Comprehensive Tutorial Agent — Research Pipeline
//!
//! Demonstrates major Eventage framework features via a production-grade
//! multi-agent research pipeline. Reference this when building your own agents.
//!
//! # Features demonstrated
//! - `EventBus` publish/subscribe/log
//! - DAG checkpoint / rollback
//! - Custom `ContextAssembler` (with negative-trajectory injection)
//! - `CycleHook` (auditing)
//! - `EventWorker` + `WorkerSet` (progress tracking, heartbeat)
//! - `Tool` trait + `max_concurrent_tools`
//! - `SandboxExecutor` (pluggable isolation)
//! - Dynamic capabilities (tool management, hooks, assemblers)
//!
//! # Run
//! Uses `MockLlmProvider` so no Ollama is required.
//! ```bash
//! cargo run -p example-tutorial-agent
//! ```
//!
//! To use a real LLM, replace `MockLlmProvider` with `OpenAiProvider`.

use async_trait::async_trait;
use eventage_core::{kinds, meta_keys, Event, EventBus, EventId};
use eventage_llm::{
    types::{ChatMessage, FunctionCall, LlmResponse, ToolCall, ToolDefinition},
    MockLlmProvider,
};
use eventage_provided_impl::{
    hook::{CycleHook, HookAction, HookContext},
    worker::{EventWorker, WorkerError, WorkerSet},
    AgentBuilder, AgentError, AssemblyContext, ContextAssembler, DynamicContextAssembler,
    DynamicHookChain, ReactStrategy, Tool,
};
use eventage_provided_impl::{BusObserver, JsonlExporter};
use eventage_sandbox::{SandboxExecutor, UnsandboxedExecutor};
use eventage_scheduler::HeartbeatScheduler;
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};
use std::time::Duration;
use tokio::sync::Mutex;

// ── Custom event kinds ────────────────────────────────────────────────────────
//
// Domain-specific event kinds signal state transitions between pipeline stages.
// Custom kinds are just strings — the bus and agent machinery treat them as
// opaque identifiers; your application code gives them meaning.

mod research_kinds {
    /// Published when the researcher's findings are ready.
    pub const FINDINGS_READY: &str = "research.findings_ready";
    /// Published when the formatted report is ready.
    pub const REPORT_READY: &str = "research.report_ready";
}

// ── Agent ID constants ────────────────────────────────────────────────────────

const ORCHESTRATOR: &str = "orchestrator";
const RESEARCHER: &str = "researcher";
const REPORTER: &str = "reporter";

// ── Helper: read TO_AGENT_ID metadata ────────────────────────────────────────

fn agent_target(event: &Event) -> Option<String> {
    event
        .metadata
        .get(meta_keys::TO_AGENT_ID)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

// ── AuditHook: CycleHook ─────────────────────────────────────────────────────
//
// CycleHooks intercept key moments in an agent's reasoning cycle.
// This hook logs every tool call and result for auditing purposes.

struct AuditHook {
    tool_calls: Arc<AtomicU32>,
}

#[async_trait]
impl CycleHook for AuditHook {
    async fn before_tool(&self, ctx: &HookContext<'_>, name: &str, args: &Value) -> HookAction {
        let n = self.tool_calls.fetch_add(1, Ordering::Relaxed) + 1;
        let args_preview: String = args.to_string().chars().take(80).collect();
        println!(
            "  [Audit]  #{n} {name}({args_preview}…) — agent={}",
            ctx.agent_id
        );
        HookAction::Continue
    }

    async fn after_tool(&self, _ctx: &HookContext<'_>, name: &str, result: &Value) {
        let preview: String = result.to_string().chars().take(80).collect();
        println!("  [Audit]  '{name}' → {preview}…");
    }
}

// ── ProgressWorker: EventWorker ───────────────────────────────────────────────
//
// EventWorkers are other e.g. non-LLM participants on the bus. They react to events
// in the background and can publish new events in response.

struct ProgressWorker {
    cycles: Arc<AtomicU32>,
}

#[async_trait]
impl EventWorker for ProgressWorker {
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
        println!("  [Progress] Cycle #{n} — agent={agent} ({ms}ms)");
        Ok(())
    }
}

// ── HeartbeatWorker: EventWorker ──────────────────────────────────────────────
//
// Demonstrates an EventWorker that reacts to heartbeat events.
// In production, this could trigger periodic health checks or scheduled scans.

struct HeartbeatWorker {
    beats: Arc<AtomicU32>,
}

#[async_trait]
impl EventWorker for HeartbeatWorker {
    fn subscribed_kinds(&self) -> Vec<String> {
        vec![kinds::SYSTEM_HEARTBEAT.to_string()]
    }

    async fn handle(&self, _event: &Event, _bus: &EventBus) -> Result<(), WorkerError> {
        let n = self.beats.fetch_add(1, Ordering::Relaxed) + 1;
        println!("  [Heartbeat] Beat #{n} — pipeline still running");
        Ok(())
    }
}

// ── Context Assemblers ────────────────────────────────────────────────────────
//
// ContextAssemblers translate the structured event log into an LLM message list.
// Each agent can have a completely different view of the shared event bus.

/// Orchestrator view: sees user input + agent reports addressed back to it.
struct OrchestratorAssembler;

#[async_trait]
impl ContextAssembler for OrchestratorAssembler {
    async fn assemble(&self, ctx: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        // Don't start a cycle if there's nothing relevant to act on.
        let has_work = ctx.events.iter().any(|e| {
            e.kind == kinds::USER_MESSAGE
                || (e.kind == kinds::AGENT_MESSAGE
                    && agent_target(e).as_deref() == Some(ORCHESTRATOR))
        });
        if !has_work {
            return vec![];
        }

        let mut messages = vec![ChatMessage::system(
            "You are a research pipeline orchestrator. When given a research topic, \
             first delegate it to the researcher agent using delegate_to_researcher. \
             Once you receive researcher findings, delegate them to the reporter using \
             delegate_to_reporter. Once the report is ready, synthesise a final executive \
             summary from all gathered information.",
        )];

        for event in ctx.events {
            match event.kind.as_str() {
                kinds::USER_MESSAGE => {
                    if let Some(text) = event.payload["text"].as_str() {
                        messages.push(ChatMessage::user(text));
                    }
                }
                kinds::AGENT_MESSAGE if agent_target(event).as_deref() == Some(ORCHESTRATOR) => {
                    if let Some(text) = event.payload["text"].as_str() {
                        let from = event
                            .metadata
                            .get(meta_keys::AGENT_ID)
                            .and_then(|v| v.as_str())
                            .unwrap_or("agent");
                        messages.push(ChatMessage::user(format!("[{from}] {text}")));
                    }
                }
                _ => {}
            }
        }

        messages
    }
}

/// Researcher view: sees tasks addressed to it, injects negative context from
/// prior failed attempts (checkpoint/rollback pattern).
struct ResearchAssembler {
    bus: EventBus,
    /// The event ID just before the most recent checkpoint.
    /// Set from outside after a rollback so rejected branches can be queried.
    anchor_id: Arc<Mutex<Option<EventId>>>,
}

#[async_trait]
impl ContextAssembler for ResearchAssembler {
    async fn assemble(&self, ctx: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let task = ctx
            .events
            .iter()
            .find(|e| {
                e.kind == kinds::AGENT_MESSAGE && agent_target(e).as_deref() == Some(RESEARCHER)
            })
            .and_then(|e| e.payload["text"].as_str());

        if task.is_none() {
            return vec![];
        }

        let mut messages = vec![ChatMessage::system(
            "You are a research specialist. Use web_search to find relevant data, \
             then extract_stats to structure the findings. Always provide concrete \
             statistics and cite the sources.",
        )];

        // ── Negative-trajectory injection ────────────────────────────────────
        // After a rollback, query what went wrong in the sealed branch and
        // inject it as a warning system message. This steers the LLM away from
        // repeating the same mistake without requiring any changes to the
        // agent's core loop.
        let anchor = *self.anchor_id.lock().await;
        if let Some(anchor_id) = anchor {
            let rejected = self.bus.rejected_branches_from(anchor_id).await;
            if !rejected.is_empty() {
                let mut warning = String::from(
                    "⚠ A previous research attempt was rolled back because it failed. \
                     Do NOT repeat the same approach.\n\nFailed attempt summary:\n",
                );
                for branch in &rejected {
                    for event in branch {
                        if event.kind == kinds::ASSISTANT_MESSAGE {
                            if let Some(content) = event.payload["content"].as_str() {
                                if !content.is_empty() {
                                    warning.push_str(&format!("  - Agent said: {content}\n"));
                                }
                            }
                        }
                        if event.kind == kinds::TOOL_CALL_PROPOSED {
                            let name = event.payload["name"].as_str().unwrap_or("?");
                            let args = event.payload["arguments"].as_str().unwrap_or("{}");
                            warning.push_str(&format!("  - Called: {name}({args})\n"));
                        }
                    }
                }
                warning.push_str(
                    "\nUse a more specific query that includes 'crate', 'ecosystem', \
                     or 'downloads' to find concrete statistics.",
                );
                messages.push(ChatMessage::system(warning));
            }
        }

        if let Some(task_text) = task {
            messages.push(ChatMessage::user(task_text));
        }

        messages
    }
}

/// Reporter view: sees formatting tasks addressed to it.
struct ReporterAssembler;

#[async_trait]
impl ContextAssembler for ReporterAssembler {
    async fn assemble(&self, ctx: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let tasks: Vec<&str> = ctx
            .events
            .iter()
            .filter(|e| {
                e.kind == kinds::AGENT_MESSAGE && agent_target(e).as_deref() == Some(REPORTER)
            })
            .filter_map(|e| e.payload["text"].as_str())
            .collect();

        if tasks.is_empty() {
            return vec![];
        }

        let mut messages = vec![ChatMessage::system(
            "You are a technical report formatter. Use the format_report tool to transform \
             raw research findings into a clear, structured document with sections for \
             key metrics, top findings, and corporate adoption.",
        )];

        for task in tasks {
            messages.push(ChatMessage::user(task));
        }

        messages
    }
}

// ── Tools ─────────────────────────────────────────────────────────────────────

/// Routes a research task to the researcher agent via the event bus.
///
/// Uses `meta_keys::TO_AGENT_ID` metadata so only the researcher reacts to
/// this `agent.message` event.
struct DelegateToResearcher {
    bus: EventBus,
}

#[async_trait]
impl Tool for DelegateToResearcher {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "delegate_to_researcher",
            "Send a research task to the researcher agent.",
            json!({
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The research topic or question."
                    }
                },
                "required": ["topic"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let topic = args["topic"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'topic'".into()))?;
        self.bus
            .publish(
                Event::new(kinds::AGENT_MESSAGE, json!({ "text": topic }))
                    .with_meta(meta_keys::TO_AGENT_ID, json!(RESEARCHER)),
            )
            .await
            .map_err(|e| AgentError::Tool(e.to_string()))?;
        Ok(json!({ "dispatched": true, "to": RESEARCHER }))
    }
}

/// Routes formatted findings to the reporter agent.
struct DelegateToReporter {
    bus: EventBus,
}

#[async_trait]
impl Tool for DelegateToReporter {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "delegate_to_reporter",
            "Send research findings to the reporter for formatting.",
            json!({
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "string",
                        "description": "Research findings to be formatted."
                    }
                },
                "required": ["findings"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let findings = args["findings"]
            .as_str()
            .ok_or_else(|| AgentError::Tool("missing 'findings'".into()))?;
        self.bus
            .publish(
                Event::new(kinds::AGENT_MESSAGE, json!({ "text": findings }))
                    .with_meta(meta_keys::TO_AGENT_ID, json!(REPORTER)),
            )
            .await
            .map_err(|e| AgentError::Tool(e.to_string()))?;
        Ok(json!({ "dispatched": true, "to": REPORTER }))
    }
}

/// Simulates a web search with a pluggable `SandboxExecutor`.
///
/// In production, replace the simulated response with a real `curl` invocation:
/// ```text
/// self.executor.execute(SandboxRequest {
///     program: "curl".into(),
///     args: vec!["-s", &url],
///     timeout_ms: 10_000,
///     ..Default::default()
/// }).await?;
/// ```
/// Swapping `UnsandboxedExecutor` → `DockerExecutor` adds full container
/// isolation with a single config change — no other code changes required.
struct WebSearch {
    // Holds the sandbox executor — swap implementations for different isolation levels.
    _executor: Arc<dyn SandboxExecutor>,
}

#[async_trait]
impl Tool for WebSearch {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "web_search",
            "Search the web for information on a topic.",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string."
                    }
                },
                "required": ["query"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let query = args["query"].as_str().unwrap_or("").to_lowercase();

        // Return data only for specific queries — this drives the
        // checkpoint/rollback demonstration: the first vague query fails,
        // the refined query after rollback succeeds.
        if query.contains("crate")
            || query.contains("ecosystem")
            || query.contains("download")
            || query.contains("adoption")
        {
            Ok(json!({
                "results": [
                    {
                        "title": "Crates.io 2024 Annual Statistics",
                        "snippet": "Total downloads exceeded 50 billion in 2024. Active crates: 145,000+. \
                                    Top crates: tokio (2.1B), serde (1.8B), rand (1.2B).",
                        "url": "https://crates.io/stats/2024"
                    },
                    {
                        "title": "Rust Ecosystem Growth — Developer Survey 2024",
                        "snippet": "Rust adoption grew 34% year-over-year. Developer satisfaction: 82%. \
                                    Major adopters: Amazon (AWS SDK), Google (Android), Microsoft (Windows drivers), Linux kernel.",
                        "url": "https://blog.rust-lang.org/2024/survey"
                    }
                ],
                "count": 2
            }))
        } else {
            // Vague query → no results. This triggers the rollback.
            Ok(json!({ "results": [], "count": 0, "note": "Query too broad, no results." }))
        }
    }
}

/// Extracts structured statistics from raw search result text.
struct ExtractStats;

#[async_trait]
impl Tool for ExtractStats {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "extract_stats",
            "Parse raw search results and extract structured statistics.",
            json!({
                "type": "object",
                "properties": {
                    "raw_data": {
                        "type": "string",
                        "description": "Raw search result text to parse."
                    }
                },
                "required": ["raw_data"]
            }),
        )
    }

    async fn execute(&self, _args: Value) -> Result<Value, AgentError> {
        Ok(json!({
            "stats": {
                "total_crates":         145_000,
                "total_downloads_2024": "50B+",
                "adoption_growth_yoy":  "34%",
                "developer_satisfaction": "82%",
                "top_3_crates": [
                    { "name": "tokio",  "downloads": "2.1B" },
                    { "name": "serde",  "downloads": "1.8B" },
                    { "name": "rand",   "downloads": "1.2B" }
                ],
                "major_corporate_adopters": [
                    "Amazon (AWS SDK in Rust)",
                    "Google (Android Rust)",
                    "Microsoft (Windows kernel drivers)",
                    "Linux kernel (Rust for drivers)"
                ]
            }
        }))
    }
}

/// Transforms raw findings into a structured Markdown report.
struct FormatReport;

#[async_trait]
impl Tool for FormatReport {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "format_report",
            "Format research statistics into a structured Markdown report.",
            json!({
                "type": "object",
                "properties": {
                    "stats": {
                        "type": "string",
                        "description": "JSON statistics to format."
                    }
                },
                "required": ["stats"]
            }),
        )
    }

    async fn execute(&self, _args: Value) -> Result<Value, AgentError> {
        Ok(json!({
            "report": "# Rust Ecosystem Growth Report — 2024\n\
                       \n\
                       ## Key Metrics\n\
                       - **Crates published:** 145,000+\n\
                       - **Total downloads (2024):** 50B+\n\
                       - **YoY adoption growth:** 34%\n\
                       - **Developer satisfaction:** 82%\n\
                       \n\
                       ## Top Crates by Downloads\n\
                       | Crate  | Downloads |\n\
                       |--------|-----------|\n\
                       | tokio  | 2.1B      |\n\
                       | serde  | 1.8B      |\n\
                       | rand   | 1.2B      |\n\
                       \n\
                       ## Major Corporate Adopters\n\
                       - Amazon — AWS SDK rewritten in Rust\n\
                       - Google — Android OS components in Rust\n\
                       - Microsoft — Windows kernel drivers in Rust\n\
                       - Linux kernel — Rust as second implementation language\n\
                       \n\
                       *Report generated by the Eventage Research Pipeline.*"
        }))
    }
}

// ── Mock LLM providers ────────────────────────────────────────────────────────
//
// MockLlmProvider returns responses in order, making demos deterministic.
// In production, replace with:
//   eventage_llm::OpenAiProvider::ollama("qwen3:4b")    // local
//   eventage_llm::OpenAiProvider::openai(key, "gpt-4o") // cloud

fn orchestrator_llm() -> MockLlmProvider {
    MockLlmProvider::new(vec![
        // Cycle 1 — delegate the research task.
        LlmResponse {
            content: Some("Delegating to the researcher.".into()),
            tool_calls: vec![ToolCall {
                id: "tc_o1".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "delegate_to_researcher".into(),
                    arguments: r#"{"topic": "Rust ecosystem crate adoption, download statistics, and corporate usage trends for 2024"}"#.into(),
                },
            }],
            finish_reason: "tool_calls".into(),
        },
        // Cycle 2 — researcher returned findings; delegate to reporter.
        LlmResponse {
            content: Some("Sending findings to the reporter for formatting.".into()),
            tool_calls: vec![ToolCall {
                id: "tc_o2".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "delegate_to_reporter".into(),
                    arguments: r#"{"findings": "Rust ecosystem 2024: 145K+ crates, 50B+ downloads, 34% adoption growth, 82% developer satisfaction. Top crates: tokio, serde, rand. Major adopters: Amazon, Google, Microsoft, Linux kernel."}"#.into(),
                },
            }],
            finish_reason: "tool_calls".into(),
        },
        // Cycle 3 — both specialist agents done; synthesise the final answer.
        LlmResponse {
            content: Some(
                "Executive Summary: The Rust ecosystem achieved remarkable growth in 2024. \
                 With 145,000+ published crates and over 50 billion total downloads, Rust \
                 has cemented its position as a leading systems language. Developer \
                 satisfaction held at 82%, and year-over-year adoption grew by 34%. \
                 Major technology companies — including Amazon, Google, Microsoft, and the \
                 Linux kernel project — have committed to Rust for production systems. \
                 The full formatted report has been prepared by the reporter agent."
                    .into(),
            ),
            tool_calls: vec![],
            finish_reason: "stop".into(),
        },
    ])
}

fn researcher_llm() -> MockLlmProvider {
    // Four responses:
    //   [Attempt 1] steps 1–2: vague search → empty results → failure message
    //   [Attempt 2] steps 3–5: specific search → extract → findings summary
    MockLlmProvider::new(vec![
        // Attempt 1 — step 1: too-broad query.
        LlmResponse {
            content: Some("Searching for information.".into()),
            tool_calls: vec![ToolCall {
                id: "tc_r_fail1".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "web_search".into(),
                    arguments: r#"{"query": "rust"}"#.into(),
                },
            }],
            finish_reason: "tool_calls".into(),
        },
        // Attempt 1 — step 2: empty results → give up (triggers rollback externally).
        LlmResponse {
            content: Some(
                "The search returned no relevant data. Unable to complete the research task."
                    .into(),
            ),
            tool_calls: vec![],
            finish_reason: "stop".into(),
        },
        // Attempt 2 — step 1: specific query after negative-context injection.
        LlmResponse {
            content: Some("Using a more targeted query this time.".into()),
            tool_calls: vec![ToolCall {
                id: "tc_r2".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "web_search".into(),
                    arguments: r#"{"query": "rust ecosystem crate adoption downloads statistics 2024"}"#.into(),
                },
            }],
            finish_reason: "tool_calls".into(),
        },
        // Attempt 2 — step 2: extract structured stats from search results.
        LlmResponse {
            content: None,
            tool_calls: vec![ToolCall {
                id: "tc_r3".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "extract_stats".into(),
                    arguments: r#"{"raw_data": "crates.io 2024: 50B downloads, 145K crates, top: tokio serde rand"}"#.into(),
                },
            }],
            finish_reason: "tool_calls".into(),
        },
        // Attempt 2 — step 3: summarise findings.
        LlmResponse {
            content: Some(
                "Research complete. Key findings for 2024: The Rust ecosystem has 145,000+ \
                 published crates with over 50 billion total downloads. Adoption grew 34% \
                 year-over-year and 82% of developers report satisfaction with the language. \
                 Top crates: tokio (2.1B downloads), serde (1.8B), rand (1.2B). \
                 Major corporate adopters include Amazon (AWS SDK), Google (Android), \
                 Microsoft (Windows drivers), and the Linux kernel project."
                    .into(),
            ),
            tool_calls: vec![],
            finish_reason: "stop".into(),
        },
    ])
}

fn reporter_llm() -> MockLlmProvider {
    MockLlmProvider::new(vec![
        // Step 1: format the statistics.
        LlmResponse {
            content: Some("Formatting the research findings into a report.".into()),
            tool_calls: vec![ToolCall {
                id: "tc_rep1".into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "format_report".into(),
                    arguments: r#"{"stats": "145K crates, 50B downloads, 34% growth, 82% satisfaction, top: tokio/serde/rand, adopters: Amazon/Google/Microsoft/Linux"}"#.into(),
                },
            }],
            finish_reason: "tool_calls".into(),
        },
        // Step 2: confirm report is ready.
        LlmResponse {
            content: Some(
                "The structured report is ready. All key metrics, top crates, and \
                 corporate adoption data have been formatted."
                    .into(),
            ),
            tool_calls: vec![],
            finish_reason: "stop".into(),
        },
    ])
}

// ── Pipeline orchestration helpers ────────────────────────────────────────────

/// Extract the last assistant message from a named agent.
async fn last_assistant_text(bus: &EventBus, agent_id: &str) -> Option<String> {
    bus.log()
        .await
        .into_iter()
        .rev()
        .find(|e| {
            e.kind == kinds::ASSISTANT_MESSAGE
                && e.metadata.get(meta_keys::AGENT_ID).and_then(|v| v.as_str()) == Some(agent_id)
                && e.payload["tool_calls"]
                    .as_array()
                    .is_none_or(|a| a.is_empty())
        })
        .and_then(|e| e.payload["content"].as_str().map(String::from))
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Suppress framework-level logs; change to "info" or "debug" to see more.
    tracing_subscriber::fmt()
        .with_env_filter("warn")
        .with_writer(std::io::stderr)
        .init();

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 1 — Shared state
    // ─────────────────────────────────────────────────────────────────────────
    // A single EventBus is shared by all agents, workers, and the scheduler.
    // Cloning the bus is cheap — all clones share the same internal broadcast
    // channel and DAG store.
    let bus = EventBus::new();
    let tool_call_counter = Arc::new(AtomicU32::new(0));
    let cycle_counter = Arc::new(AtomicU32::new(0));
    let heartbeat_counter = Arc::new(AtomicU32::new(0));

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 2 — Observability: BusObserver + JsonlExporter
    // ─────────────────────────────────────────────────────────────────────────
    // Every event published to the bus is fanned out to registered exporters.
    // JsonlExporter writes newline-delimited JSON for replay and post-hoc analysis.
    // Swap in OtelExporter (feature "opentelemetry") for distributed tracing.
    let log_path = format!(
        "/tmp/eventage-tutorial-{}.jsonl",
        chrono::Utc::now().timestamp()
    );
    let exporter = JsonlExporter::new(&log_path).await?;
    let observer = BusObserver::new(bus.clone()).add_exporter(exporter);
    tokio::spawn(async move { observer.run().await });
    println!("Observability: writing all events to {log_path}");
    println!();

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 3 — HeartbeatScheduler (background)
    // ─────────────────────────────────────────────────────────────────────────
    // Fires `system.heartbeat` events every 3 seconds so agents that call
    // `agent.run()` can perform periodic work without user input.
    // The HeartbeatWorker below reacts to these events.
    let scheduler = HeartbeatScheduler::new(bus.clone(), Duration::from_secs(3));
    tokio::spawn(async move { scheduler.run().await });

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 4 — WorkerSet: EventWorkers
    // ─────────────────────────────────────────────────────────────────────────
    // Workers run concurrently in a background task. Multiple workers can be
    // added to one WorkerSet; each subscribes to the events it cares about.
    let bus_for_workers = bus.clone();
    let beat_counter = Arc::clone(&heartbeat_counter);
    let cyc_counter = Arc::clone(&cycle_counter);
    tokio::spawn(async move {
        WorkerSet::new()
            .add_worker(ProgressWorker {
                cycles: cyc_counter,
            })
            .add_worker(HeartbeatWorker {
                beats: beat_counter,
            })
            .run_on(bus_for_workers)
            .await
            .ok();
    });

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 5 — Build agents (with dynamic entity handles)
    // ─────────────────────────────────────────────────────────────────────────
    // Each agent has its own:
    //   - LLM provider (or mock)
    //   - ContextAssembler (controls what the LLM sees)
    //   - Tool registry (only the tools relevant to this agent's role)
    //   - CycleHooks (cross-cutting concerns: auditing, step limits, approval)
    //
    // All agents share the same EventBus — this is the only coupling between them.
    //
    // Dynamic entity handles let us mutate tools, hooks, and assemblers AFTER
    // the agent is built — without restarting it.

    // Sandbox executor — UnsandboxedExecutor for this demo.
    // In production: DockerExecutor::new("python:3.12-slim") or LandlockExecutor::new()
    let executor: Arc<dyn SandboxExecutor> = Arc::new(UnsandboxedExecutor::new());
    println!("Sandbox: using {} executor", executor.name());
    println!();

    // Orchestrator: delegates work, synthesises the final report.
    let orchestrator = AgentBuilder::new()
        .agent_id(ORCHESTRATOR)
        .bus(bus.clone())
        .context(OrchestratorAssembler)
        .llm(orchestrator_llm())
        .tool(DelegateToResearcher { bus: bus.clone() })
        .tool(DelegateToReporter { bus: bus.clone() })
        .hook(AuditHook {
            tool_calls: Arc::clone(&tool_call_counter),
        })
        // Allow both delegation tools to run concurrently if the LLM calls them together.
        .strategy(ReactStrategy {
            max_steps: 20,
            max_concurrent_tools: 2,
        })
        .build();

    // Researcher: gathers data; supports checkpoint/rollback via ResearchAssembler.
    // Uses a KeywordToolSelector so only "web_search"-related tools are offered per step.
    let researcher_anchor: Arc<Mutex<Option<EventId>>> = Arc::new(Mutex::new(None));
    let researcher = AgentBuilder::new()
        .agent_id(RESEARCHER)
        .bus(bus.clone())
        .context(ResearchAssembler {
            bus: bus.clone(),
            anchor_id: Arc::clone(&researcher_anchor),
        })
        .llm(researcher_llm())
        .tool(WebSearch {
            _executor: Arc::clone(&executor),
        })
        .tool(ExtractStats)
        .hook(AuditHook {
            tool_calls: Arc::clone(&tool_call_counter),
        })
        // ── ToolSelector: only expose tools relevant to the current step ──────
        // Here we expose all tools (no keyword filter), but you can swap this
        // to KeywordToolSelector::new(vec!["web"]) to hide ExtractStats until
        // web_search has returned results.
        .strategy(ReactStrategy {
            max_steps: 5,
            max_concurrent_tools: 4,
        }) // cap the internal ReAct loop
        .build();

    // Reporter: formats the structured report.
    //
    // Dynamic features demonstrated on the reporter:
    //   - DynamicHookChain: audit hook added after build
    //   - DynamicContextAssembler: assembler swapped between pipeline stages
    //   - ToolRegistry handle: tool added after build
    //
    // This shows that a long-lived agent can change behaviour at each stage
    // without being rebuilt.

    // Create a DynamicHookChain — start empty, add AuditHook just before the reporter runs.
    let reporter_dyn_hooks = DynamicHookChain::new();
    let reporter_hook_handle = reporter_dyn_hooks.clone();

    // Create a DynamicContextAssembler — start with the default reporter view.
    let reporter_dyn_ctx = DynamicContextAssembler::new(ReporterAssembler);
    let reporter_ctx_handle = reporter_dyn_ctx.clone();

    let reporter_builder = AgentBuilder::new()
        .agent_id(REPORTER)
        .bus(bus.clone())
        .context(reporter_dyn_ctx) // swappable assembler
        .llm(reporter_llm())
        .hook(reporter_dyn_hooks) // mutable hook chain
        .strategy(ReactStrategy::default());

    // Grab a live tool registry handle before building.
    let reporter_tools = reporter_builder.tool_registry();
    let reporter = reporter_builder.build();

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 6 — Publish the initial user request
    // ─────────────────────────────────────────────────────────────────────────
    let topic = "Analyze Rust ecosystem growth and adoption trends for 2024";
    println!("Research Topic: {topic}");
    println!("{}", "─".repeat(60));

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({ "text": topic })))
        .await?;

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 7 — Orchestrator: first cycle (delegates to researcher)
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n>>> Stage 1: Orchestrator delegates research task");
    orchestrator.cycle().await?;

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 8 — Researcher: attempt 1 with checkpoint/rollback
    // ─────────────────────────────────────────────────────────────────────────
    // Before running the researcher, take a checkpoint. If the attempt fails,
    // we roll back to this point and the active branch is restored to the state
    // it was in just before the checkpoint — as if the failed attempt never happened.
    println!("\n>>> Stage 2: Researcher — attempt 1 (will fail; rollback demo)");

    // Record the anchor: the last event before the checkpoint.
    // After rollback this ID lets us query what went wrong in the rejected branch.
    let anchor_id = bus.log().await.last().map(|e| e.id);
    let cp_id = bus.checkpoint().await?;
    println!("  [DAG] Checkpoint taken: {cp_id}");

    researcher.cycle().await?;

    // Detect failure: the researcher said it couldn't find data.
    let attempt1_text = last_assistant_text(&bus, RESEARCHER).await;
    let research_failed = attempt1_text
        .as_deref()
        .map(|t| t.contains("Unable to complete") || t.contains("no relevant data"))
        .unwrap_or(false);

    if research_failed {
        println!("  [DAG] Research failed. Rolling back to checkpoint…");
        bus.rollback(cp_id).await?;
        println!("  [DAG] Rollback complete — failed events sealed in rejected branch.");

        // Set the anchor so ResearchAssembler injects negative context on retry.
        *researcher_anchor.lock().await = anchor_id;

        // ── STEP 9: Researcher retry with negative-context injection ──────────
        println!("\n>>> Stage 3: Researcher — attempt 2 (negative context injected)");
        researcher.cycle().await?;
    } else {
        println!("  [DAG] Research succeeded on first attempt.");
    }

    // Forward researcher's findings to the orchestrator as an agent.message.
    if let Some(findings) = last_assistant_text(&bus, RESEARCHER).await {
        println!("\n  [Researcher] Publishing findings to orchestrator.");
        bus.publish(
            Event::new(research_kinds::FINDINGS_READY, json!({ "text": &findings }))
                .with_meta(meta_keys::AGENT_ID, json!(RESEARCHER)),
        )
        .await?;
        // Also route as agent.message so OrchestratorAssembler picks it up.
        bus.publish(
            Event::new(kinds::AGENT_MESSAGE, json!({ "text": findings }))
                .with_meta(meta_keys::AGENT_ID, json!(RESEARCHER))
                .with_meta(meta_keys::TO_AGENT_ID, json!(ORCHESTRATOR)),
        )
        .await?;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 10 — Orchestrator: second cycle (delegates to reporter)
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n>>> Stage 4: Orchestrator delegates findings to reporter");
    orchestrator.cycle().await?;

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 11 — Reporter: format the report
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n>>> Stage 5: Reporter formats the structured report");

    // ── Dynamic entity manipulation ──────────────────────────────────────────
    // Demonstrate all three dynamic management features just before the reporter runs:

    // 1. DynamicHookChain — add the audit hook now (wasn't active before this stage).
    reporter_hook_handle.add_hook(AuditHook {
        tool_calls: Arc::clone(&tool_call_counter),
    });
    println!("  [Dynamic]  AuditHook added to reporter via DynamicHookChain.");

    // 2. ToolRegistry — inject FormatReport at runtime (was not present at build time).
    reporter_tools.add_tool(FormatReport);
    println!("  [Dynamic]  FormatReport tool added to reporter registry at runtime.");

    // 3. DynamicContextAssembler — swap to the same assembler with a note showing it works.
    //    In a real pipeline you'd swap to a specialised "executive summary" assembler here.
    reporter_ctx_handle.swap(ReporterAssembler);
    println!("  [Dynamic]  Reporter context assembler swapped (no-op here; shows the API).");

    reporter.cycle().await?;

    // Forward reporter's output to the orchestrator.
    if let Some(report_summary) = last_assistant_text(&bus, REPORTER).await {
        // Emit the domain-level "report ready" event for downstream consumers.
        bus.publish(
            Event::new(
                research_kinds::REPORT_READY,
                json!({ "text": &report_summary }),
            )
            .with_meta(meta_keys::AGENT_ID, json!(REPORTER)),
        )
        .await?;
        // Route to orchestrator for final synthesis.
        bus.publish(
            Event::new(kinds::AGENT_MESSAGE, json!({ "text": report_summary }))
                .with_meta(meta_keys::AGENT_ID, json!(REPORTER))
                .with_meta(meta_keys::TO_AGENT_ID, json!(ORCHESTRATOR)),
        )
        .await?;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 12 — Orchestrator: final synthesis
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n>>> Stage 6: Orchestrator synthesises the executive summary");
    orchestrator.cycle().await?;

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 13 — Output results
    // ─────────────────────────────────────────────────────────────────────────
    let final_answer = last_assistant_text(&bus, ORCHESTRATOR)
        .await
        .unwrap_or_else(|| "(no final answer)".into());

    println!("\n{}", "─".repeat(60));
    println!("EXECUTIVE SUMMARY:\n");
    println!("{final_answer}");

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 14 — Pipeline metrics
    // ─────────────────────────────────────────────────────────────────────────
    let log = bus.log().await;
    let rejected = bus.all_rejected_branches().await;

    // Give the observability worker a moment to flush the last events.
    tokio::time::sleep(Duration::from_millis(50)).await;

    println!("\n{}", "─".repeat(60));
    println!("Pipeline Metrics:");
    println!("  Active events      : {}", log.len());
    println!(
        "  Rejected branches  : {} (sealed by rollback)",
        rejected.len()
    );
    println!(
        "  Rejected events    : {}",
        rejected.iter().map(|(_, evs)| evs.len()).sum::<usize>()
    );
    println!(
        "  Agent cycles       : {}",
        cycle_counter.load(Ordering::Relaxed)
    );
    println!(
        "  Tool invocations   : {}",
        tool_call_counter.load(Ordering::Relaxed)
    );
    println!(
        "  Heartbeat ticks    : {}",
        heartbeat_counter.load(Ordering::Relaxed)
    );
    println!("  Event log file     : {log_path}");

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 15 — Abbreviated event log
    // ─────────────────────────────────────────────────────────────────────────
    println!("\nActive-branch event log ({} events):", log.len());
    for event in &log {
        let agent = event
            .metadata
            .get(meta_keys::AGENT_ID)
            .and_then(|v| v.as_str())
            .unwrap_or("system");
        match event.kind.as_str() {
            kinds::USER_MESSAGE => {
                println!(
                    "  [user.message]        {}",
                    event.payload["text"].as_str().unwrap_or("")
                );
            }
            kinds::CHECKPOINT => {
                println!("  [system.checkpoint]   id={}", event.id);
            }
            kinds::SYSTEM_HEARTBEAT => {
                println!("  [system.heartbeat]");
            }
            kinds::AGENT_CYCLE_START => {
                println!("  [cycle.start]         agent={agent}");
            }
            kinds::AGENT_CYCLE_END => {
                println!("  [cycle.end]           agent={agent}");
            }
            kinds::TOOL_CALL_PROPOSED => {
                println!(
                    "  [tool.call.proposed]  {} → {}",
                    agent,
                    event.payload["name"].as_str().unwrap_or("?")
                );
            }
            kinds::TOOL_RESULT => {
                println!(
                    "  [tool.result]         {}",
                    event.payload["name"].as_str().unwrap_or("?")
                );
            }
            kinds::ASSISTANT_MESSAGE => {
                let tc = event.payload["tool_calls"]
                    .as_array()
                    .map_or(0, |a| a.len());
                if tc > 0 {
                    println!("  [assistant.message]   {agent} → {tc} tool call(s)");
                } else if let Some(text) = event.payload["content"].as_str() {
                    let preview: String = text.chars().take(60).collect();
                    println!("  [assistant.message]   {agent}: {preview}…");
                }
            }
            kinds::AGENT_MESSAGE => {
                let to = agent_target(event).unwrap_or_else(|| "broadcast".into());
                let text = event.payload["text"].as_str().unwrap_or("");
                let preview: String = text.chars().take(55).collect();
                println!("  [agent.message]       {agent} → {to}: {preview}…");
            }
            research_kinds::FINDINGS_READY => {
                println!("  [research.findings]   ready");
            }
            research_kinds::REPORT_READY => {
                println!("  [research.report]     ready");
            }
            _ => {}
        }
    }

    if !rejected.is_empty() {
        println!("\nRejected branches (sealed after rollback):");
        for (branch_id, events) in &rejected {
            let kinds: Vec<&str> = events.iter().map(|e| e.kind.as_str()).collect();
            println!(
                "  Branch {} — {} events: {}",
                &branch_id.to_string()[..8],
                events.len(),
                kinds.join(", ")
            );
        }
    }

    println!("\nDone. Replay events from {log_path} with: cargo run -p eventage-replay");

    Ok(())
}
