# Building a Production Agent with Eventage

This tutorial walks you through building a **multi-agent research pipeline** from scratch
using every major feature of the Eventage framework. By the end you will have a system
that can be deployed as-is or extended into a real application.

The complete runnable code lives in `src/main.rs`. Every section below corresponds
directly to a labelled step in that file.

---

## What we are building

A research pipeline with three agents that coordinate over a shared event bus:

```
User prompt
    │
    ▼
[Orchestrator] ──delegate──► [Researcher] (with rollback on failure)
    │                               │ findings
    │◄──────────────────────────────┘
    │
    │──delegate──► [Reporter]
    │                     │ formatted report
    │◄────────────────────┘
    │
    ▼
Executive Summary
```

Supporting infrastructure running in the background:

- **HeartbeatScheduler** — fires periodic `system.heartbeat` events
- **ProgressWorker** / **HeartbeatWorker** — `EventWorker`s that react to bus events
- **BusObserver + JsonlExporter** — records every event to a JSONL file
- **AuditHook** — `CycleHook` that logs every tool call and result

---

## Prerequisites

No external services are required. The tutorial uses `MockLlmProvider` for deterministic
output. To switch to a real model, see the note at the end of each relevant section.

```bash
# Build and run
cargo run -p example-tutorial-agent
```

---

## Step 1 — The EventBus

The `EventBus` is the single shared medium through which every component
communicates. It is an async, append-only broadcast log that also supports
DAG branching via checkpoints and rollbacks.

```rust
use eventage_core::EventBus;

let bus = EventBus::new();
```

Cloning the bus is cheap — all clones share the same internal broadcast channel
and DAG store. Pass clones freely to agents, workers, tools, and schedulers.

### Key bus operations

| Operation | Description |
|-----------|-------------|
| `bus.publish(event)` | Append to active branch + broadcast |
| `bus.subscribe()` | Receive all future events |
| `bus.log()` | Full active-branch snapshot |
| `bus.log_since(n)` | Events after position `n` |
| `bus.checkpoint()` | Mark a safe rollback point |
| `bus.rollback(cp_id)` | Truncate to checkpoint; seal removed events |
| `bus.wait_for(predicate)` | Block until a matching event arrives |
| `bus.rejected_branches_from(anchor_id)` | Query sealed failed trajectories |

---

## Step 2 — Observability

Connect a `BusObserver` to the bus *before* anything else so every event is
captured from the first publish.

```rust
use eventage_observability::{BusObserver, JsonlExporter};

let exporter = JsonlExporter::new("/tmp/events.jsonl").await?;
let observer = BusObserver::new(bus.clone()).add_exporter(exporter);

// Spawn as a background task — it runs until the bus closes.
tokio::spawn(async move { observer.run().await });
```

`JsonlExporter` appends one JSON object per line. Each line is a complete `Event`
struct, so any tool that reads JSON can process the log.

### Replay

After the run, replay all events to a fresh exporter (e.g., for post-hoc analysis):

```rust
let observer = BusObserver::new(bus.clone()).add_exporter(another_exporter);
observer.export_snapshot().await?;
```

### OpenTelemetry

Enable the `opentelemetry` feature in `eventage-observability` and swap in
`OtelExporter`. It maps agent cycles to spans and tool calls to child spans:

```rust
use eventage_observability::OtelExporter;
let observer = BusObserver::new(bus.clone()).add_exporter(OtelExporter::new());
```

---

## Step 3 — HeartbeatScheduler

The scheduler fires a `system.heartbeat` event on a regular interval. Agents
that call `agent.run()` wake on heartbeats and can perform periodic maintenance:

```rust
use eventage_scheduler::HeartbeatScheduler;
use std::time::Duration;

let scheduler = HeartbeatScheduler::new(bus.clone(), Duration::from_secs(30));
tokio::spawn(async move { scheduler.run().await });
```

The pipeline in this tutorial is manually orchestrated via `agent.cycle()` calls,
so the heartbeat is consumed by `HeartbeatWorker` for demonstration. In a
fully autonomous pipeline, replace manual `cycle()` calls with `agent.run()`.

---

## Step 4 — EventWorkers

`EventWorker` is the interface for other e.g. non-LLM bus participants. Workers subscribe
to specific event kinds and react in the background:

```rust
use eventage_agent::worker::{EventWorker, WorkerError, WorkerSet};
use eventage_core::{Event, EventBus, kinds};
use async_trait::async_trait;

struct ProgressWorker { cycles: Arc<AtomicU32> }

#[async_trait]
impl EventWorker for ProgressWorker {
    fn subscribed_kinds(&self) -> Vec<String> {
        vec![kinds::AGENT_CYCLE_END.to_string()]
    }

    async fn handle(&self, event: &Event, _bus: &EventBus) -> Result<(), WorkerError> {
        let n = self.cycles.fetch_add(1, Ordering::Relaxed) + 1;
        let agent = event.metadata.get("agent_id")
            .and_then(|v| v.as_str()).unwrap_or("?");
        println!("Cycle #{n} complete — agent={agent}");
        Ok(())
    }
}
```

Group multiple workers into a `WorkerSet` and run them on the bus concurrently:

```rust
WorkerSet::new()
    .add_worker(ProgressWorker { cycles: Arc::new(AtomicU32::new(0)) })
    .add_worker(HeartbeatWorker { beats: Arc::new(AtomicU32::new(0)) })
    .run_on(bus.clone())
    .await?;
```

Workers can also *publish* events. A common pattern is a worker that acts as an
escalation trigger: if a certain error event is seen three times, it publishes a
`system.alert` event that other components react to.

---

## Step 5 — Tools and the SandboxExecutor

Tools are how agents interact with the world. Each tool implements the `Tool` trait:

```rust
use eventage_agent::{AgentError, Tool};
use eventage_llm::types::ToolDefinition;
use serde_json::Value;
use async_trait::async_trait;

struct WebSearch { executor: Arc<dyn SandboxExecutor> }

#[async_trait]
impl Tool for WebSearch {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "web_search",
            "Search the web for information.",
            serde_json::json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let query = args["query"].as_str().unwrap_or("");
        // In production: invoke executor to run curl in isolation
        // self.executor.execute(SandboxRequest { program: "curl".into(), ... }).await?;
        Ok(serde_json::json!({ "results": [] }))
    }
}
```

### SandboxExecutor

The `SandboxExecutor` trait decouples tools from the execution environment.
Three implementations are provided:

| Implementation | Isolation | When to use |
|----------------|-----------|-------------|
| `UnsandboxedExecutor` | None | Development, trusted code |
| `LandlockExecutor` | Filesystem (Linux 5.13+) | Linux production |
| `DockerExecutor` | Full container | Maximum isolation, all platforms |

```rust
use eventage_sandbox::{UnsandboxedExecutor, DockerExecutor};

// Development
let executor: Arc<dyn SandboxExecutor> = Arc::new(UnsandboxedExecutor::new());

// Production (swap with no other code changes required)
let executor: Arc<dyn SandboxExecutor> = Arc::new(DockerExecutor::new("python:3.12-slim"));
```

Passing `executor` through to the tool (as a constructor argument stored as a field)
means changing the execution environment requires zero changes to tool logic.

---

## Step 6 — CycleHooks

Hooks intercept key moments in the agent's reasoning cycle. Register any number
of hooks via `AgentBuilder::hook`; they run in order and short-circuit on the first
non-`Continue` action.

```rust
use eventage_agent::hook::{CycleHook, HookAction, HookContext};
use serde_json::Value;
use async_trait::async_trait;

struct AuditHook { tool_calls: Arc<AtomicU32> }

#[async_trait]
impl CycleHook for AuditHook {
    async fn before_tool(&self, ctx: &HookContext<'_>, name: &str, args: &Value) -> HookAction {
        println!("Tool '{}' called by agent '{}'", name, ctx.agent_id);
        HookAction::Continue
    }

    async fn after_tool(&self, _ctx: &HookContext<'_>, name: &str, result: &Value) {
        println!("Tool '{}' returned: {}", name, result);
    }
}
```

### Common hook patterns

**Step limiter** (built-in):
```rust
use eventage_agent::hook::MaxStepsHook;
.hook(MaxStepsHook::new(10))
```

**Human approval gate** (veto individual tool calls):
```rust
async fn before_tool(&self, _: &HookContext<'_>, name: &str, _: &Value) -> HookAction {
    eprint!("Allow '{}'? [y/N]: ", name);
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).ok();
    if line.trim().eq_ignore_ascii_case("y") { HookAction::Continue } else { HookAction::Skip }
}
```

**Context editor** (modify the LLM prompt before each call):
```rust
async fn before_llm(&self, _: &HookContext<'_>, messages: &mut Vec<ChatMessage>) -> HookAction {
    messages.push(ChatMessage::system("Always respond in bullet points."));
    HookAction::Continue
}
```

---

## Step 7 — ContextAssemblers

The `ContextAssembler` converts the event log into an LLM message list. Each
agent can have a completely different view of the shared bus.

```rust
use eventage_agent::{AssemblyContext, ContextAssembler};
use eventage_llm::types::ChatMessage;
use async_trait::async_trait;

struct OrchestratorAssembler;

#[async_trait]
impl ContextAssembler for OrchestratorAssembler {
    async fn assemble(&self, ctx: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let mut messages = vec![ChatMessage::system("You are the orchestrator.")];
        for event in ctx.events {
            if event.kind == "user.message" {
                if let Some(text) = event.payload["text"].as_str() {
                    messages.push(ChatMessage::user(text));
                }
            }
            // ... handle agent.message, etc.
        }
        messages
    }
}
```

Return an empty `Vec` to signal "nothing to act on yet" — the agent will skip
the cycle. This is how agents naturally idle until relevant events appear.

### Built-in assemblers

- **`DefaultContextAssembler`** — converts standard event kinds (`user.message`,
  `assistant.message`, `tool.result`) to OpenAI-style turns. Suitable for simple
  single-agent chatbots.
- **`NegativeAwareContextAssembler`** — wraps any assembler and injects a warning
  system message when `context.rejected_branches` is non-empty.

### Custom assembler with bus access

For checkpoint/rollback integration, the `ResearchAssembler` in this tutorial
holds an `EventBus` reference and queries rejected branches directly:

```rust
struct ResearchAssembler {
    bus: EventBus,
    anchor_id: Arc<Mutex<Option<EventId>>>,
}

#[async_trait]
impl ContextAssembler for ResearchAssembler {
    async fn assemble(&self, ctx: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let mut messages = vec![ChatMessage::system("You are a researcher.")];

        // Query what went wrong in the rejected branch.
        if let Some(anchor_id) = *self.anchor_id.lock().await {
            let rejected = self.bus.rejected_branches_from(anchor_id).await;
            if !rejected.is_empty() {
                messages.push(ChatMessage::system("Previous attempt failed: ..."));
            }
        }

        // ... add task from event log
        messages
    }
}
```

After a rollback, set `anchor_id` from outside. On the next `cycle()` call the
assembler automatically injects the negative context — no changes to the agent
core are needed.

---

## Step 8 — AgentBuilder

`AgentBuilder` is the fluent constructor for `Agent`. Every option has a
sensible default; only `llm` is required.

```rust
use eventage_agent::AgentBuilder;

let researcher = AgentBuilder::new()
    .agent_id("researcher")       // stable ID for routing + observability
    .bus(bus.clone())              // shared event bus
    .context(ResearchAssembler { bus: bus.clone(), anchor_id: ... })
    .llm(researcher_llm())        // LlmProvider implementation
    .tool(WebSearch { _executor: executor.clone() })
    .tool(ExtractStats)
    .hook(AuditHook { ... })      // hooks run in registration order
    .max_react_steps(5)           // guard against infinite tool loops
    .max_concurrent_tools(4)      // parallel tool execution (default: 4)
    .build();
```

### Real LLM providers

```rust
use eventage_llm::OpenAiProvider;

// Local Ollama
let llm = OpenAiProvider::ollama("qwen3:4b");

// OpenAI
let llm = OpenAiProvider::openai(
    std::env::var("OPENAI_API_KEY").unwrap(),
    "gpt-4o",
);

// Any OpenAI-compatible endpoint
let llm = OpenAiProvider::custom("https://api.example.com/v1", api_key, "model-name");
```

---

## Step 9 — Running the pipeline

Once agents are built, the pipeline is driven by publishing the initial event
and then manually calling `agent.cycle()` in the appropriate sequence:

```rust
// Kick off the pipeline.
bus.publish(Event::new(kinds::USER_MESSAGE, json!({ "text": topic }))).await?;

// Step through the pipeline manually.
orchestrator.cycle().await?;
researcher.cycle().await?;   // may be retried after rollback
orchestrator.cycle().await?; // sees researcher result
reporter.cycle().await?;
orchestrator.cycle().await?; // final synthesis
```

For fully autonomous operation, use `agent.run()` instead, which loops forever
and wakes on `user.message`, `system.heartbeat`, or `agent.message` events:

```rust
tokio::spawn(async move { orchestrator.run().await });
tokio::spawn(async move { researcher.run().await });
tokio::spawn(async move { reporter.run().await });
```

For concurrent multi-agent execution with bounded parallelism, use `AgentSet`:

```rust
use eventage_agent::AgentSet;

AgentSet::new()
    .add_agent(researcher)
    .add_agent(reporter)
    .max_concurrent(4)
    .run_until_all_complete()
    .await?;
```

---

## Step 10 — Checkpoint / Rollback

The DAG event store enables speculative execution: take a checkpoint before a
risky operation, attempt it, and roll back cleanly if it fails.

```rust
// Record the anchor: the last event before the checkpoint.
let anchor_id = bus.log().await.last().map(|e| e.id);

// Create the checkpoint marker in the active branch.
let cp_id = bus.checkpoint().await?;

// Attempt the risky operation.
researcher.cycle().await?;

// Inspect the result.
let failed = detect_failure(&bus, "researcher").await;

if failed {
    // Roll back: events after cp_id are removed from the active branch
    // and sealed into a RejectedBranch. The bus is restored to its state
    // at `anchor_id` — as if the failed attempt never happened.
    bus.rollback(cp_id).await?;

    // Let the assembler inject negative context on the next cycle.
    *researcher_anchor.lock().await = anchor_id;

    // Retry.
    researcher.cycle().await?;
}
```

### Querying rejected branches

The sealed events are never deleted. Query them for negative context, evaluation,
or debugging:

```rust
// Events from branches that diverged at `anchor_id`.
let rejected = bus.rejected_branches_from(anchor_id).await;
// → Vec<Vec<Event>>; each inner Vec is one rejected trajectory.

// All rejected branches across the entire session.
let all = bus.all_rejected_branches().await;
// → Vec<(BranchId, Vec<Event>)>
```

This is what `ResearchAssembler` does: it queries `rejected_branches_from(anchor_id)`,
extracts what the LLM said and which tools it called, and injects a concise
warning into the next prompt. The LLM learns from its own failure without any
external training loop.

---

## Step 11 — Custom event kinds

Standard event kinds (`user.message`, `agent.cycle.end`, etc.) handle the
framework's built-in events. For domain logic, define your own:

```rust
mod research_kinds {
    pub const FINDINGS_READY: &str = "research.findings_ready";
    pub const REPORT_READY:   &str = "research.report_ready";
}

// Publish when findings are ready.
bus.publish(
    Event::new(research_kinds::FINDINGS_READY, json!({ "text": findings }))
        .with_meta(meta_keys::AGENT_ID, json!("researcher")),
).await?;

// React in a worker.
fn subscribed_kinds(&self) -> Vec<String> {
    vec![research_kinds::FINDINGS_READY.to_string()]
}
```

Custom kinds integrate seamlessly with the full event system: the `BusObserver`
records them, `wait_for` can match them, and workers can publish downstream events
in response. They are the building blocks of domain-specific workflows.

---

## Step 12 — Multi-agent routing with `meta_keys`

Route messages between agents using the `TO_AGENT_ID` metadata key:

```rust
use eventage_core::{meta_keys, kinds};

// Addressed to a specific agent — only that agent reacts.
bus.publish(
    Event::new(kinds::AGENT_MESSAGE, json!({ "text": task }))
        .with_meta(meta_keys::TO_AGENT_ID, json!("researcher")),
).await?;

// No TO_AGENT_ID → broadcast; all agents calling agent.run() wake up.
bus.publish(Event::new(kinds::AGENT_MESSAGE, json!({ "text": task }))).await?;
```

The `AGENT_ID` metadata key is stamped automatically by the framework on every
event published inside an agent cycle. Use it to identify which agent produced
a given event without any extra bookkeeping:

```rust
let agent = event.metadata
    .get(meta_keys::AGENT_ID)
    .and_then(|v| v.as_str())
    .unwrap_or("system");
```

---

## Step 13 — Event log and audit trail

At any point you can inspect the full active-branch log:

```rust
let log = bus.log().await;
for event in &log {
    println!("{} {:?}", event.kind, event.payload);
}
```

Every `agent.cycle.end` event carries timing metadata:

```rust
let elapsed_ms = event.metadata
    .get(meta_keys::ELAPSED_MS)
    .and_then(|v| v.as_u64())
    .unwrap_or(0);
```

The JSONL file written by `JsonlExporter` is the canonical audit trail. Pipe it
into `jq`, import it into a database, or replay it through `eventage-replay` for
a visual timeline.

---

## Full run output

Running `cargo run -p example-tutorial-agent` produces output like this:

```
Observability: writing all events to /tmp/eventage-tutorial-1720000000.jsonl
Sandbox: using unsandboxed executor

Research Topic: Analyze Rust ecosystem growth and adoption trends for 2024
────────────────────────────────────────────────────

>>> Stage 1: Orchestrator delegates research task
  [Audit]  #1 delegate_to_researcher({...})…
  [Audit]  'delegate_to_researcher' → {"dispatched":true,"to":"researcher"}…
  [Progress] Cycle #1 — agent=orchestrator (4ms)

>>> Stage 2: Researcher — attempt 1 (will fail; rollback demo)
  [DAG] Checkpoint taken: 3fa4e91c-...
  [Audit]  #2 web_search({"query":"rust"})…
  [Audit]  'web_search' → {"results":[],"count":0,"note":"Query too broad"}…
  [Progress] Cycle #2 — agent=researcher (3ms)
  [DAG] Research failed. Rolling back to checkpoint…
  [DAG] Rollback complete — failed events sealed in rejected branch.

>>> Stage 3: Researcher — attempt 2 (negative context injected)
  [Audit]  #3 web_search({"query":"rust ecosystem crate adoption downloads…"})…
  [Audit]  #4 extract_stats({...})…
  [Progress] Cycle #3 — agent=researcher (5ms)

  [Researcher] Publishing findings to orchestrator.

>>> Stage 4: Orchestrator delegates findings to reporter
  [Audit]  #5 delegate_to_reporter({...})…
  [Progress] Cycle #4 — agent=orchestrator (2ms)

>>> Stage 5: Reporter formats the structured report
  [Audit]  #6 format_report({...})…
  [Progress] Cycle #5 — agent=reporter (3ms)

>>> Stage 6: Orchestrator synthesises the executive summary
  [Progress] Cycle #6 — agent=orchestrator (1ms)

────────────────────────────────────────────────────
EXECUTIVE SUMMARY:

Executive Summary: The Rust ecosystem achieved remarkable growth in 2024. ...

────────────────────────────────────────────────────
Pipeline Metrics:
  Active events      : 38
  Rejected branches  : 1 (sealed by rollback)
  Rejected events    : 7
  Agent cycles       : 6
  Tool invocations   : 6
  Heartbeat ticks    : 0
  Event log file     : /tmp/eventage-tutorial-1720000000.jsonl

Active-branch event log (38 events):
  [user.message]        Analyze Rust ecosystem growth ...
  [cycle.start]         agent=orchestrator
  [assistant.message]   orchestrator → 1 tool call(s)
  [tool.call.proposed]  orchestrator → delegate_to_researcher
  [tool.result]         delegate_to_researcher
  [cycle.end]           agent=orchestrator
  [agent.message]       system → researcher: Rust ecosystem crate adoption ...
  [system.checkpoint]   id=3fa4e91c-...
  [cycle.start]         agent=researcher
  ...                   (failed attempt elided — in rejected branch)
  [cycle.start]         agent=researcher
  [assistant.message]   researcher → 1 tool call(s)
  [tool.call.proposed]  researcher → web_search
  [tool.result]         web_search
  [tool.call.proposed]  researcher → extract_stats
  [tool.result]         extract_stats
  [assistant.message]   researcher: Research complete. Key findings ...
  [cycle.end]           agent=researcher
  ...

Rejected branches (sealed after rollback):
  Branch 3fa4e91c — 7 events: system.checkpoint, agent.cycle.start, ...
```

---

## Step 14 — Dynamic entity management

All three agent components — tools, hooks, and context assemblers — can be mutated
at runtime without rebuilding or restarting the agent. This is demonstrated in
Stage 5 of `src/main.rs`.

### Dynamic tool registry

`ToolRegistry` uses an internal `Arc<RwLock<...>>`, so every clone is a live
handle to the same tool list.

```rust
// Obtain a handle before building.
let tools = builder.tool_registry();
let agent = builder.build();

// Or obtain it after building.
let tools = agent.tools();

// Add at any time — the agent sees it on the next ReAct step.
tools.add_tool(NewTool);

// Remove to narrow the context window.
tools.remove("heavy_tool");

// Stage transition: swap the whole tool set.
tools.clear();
tools.add_tool(StageOneTool);
```

### ToolSelector — intelligent per-step filtering

A `ToolSelector` filters which tools the LLM *sees* on each ReAct step without
touching the registry. The full registry is still used for execution; the selector
only narrows the `tool_definitions` the LLM receives.

```rust
// Built-in: keyword filter.
.tool_selector(KeywordToolSelector::new(vec!["search", "fetch"]))

// Custom: implement ToolSelector for any logic — including LLM-driven routing.
struct PrioritySelector { top_k: usize }

#[async_trait]
impl ToolSelector for PrioritySelector {
    async fn select(&self, tools: &[Arc<dyn Tool>], messages: &[ChatMessage]) -> Vec<Arc<dyn Tool>> {
        // Score tools against the last user message, return the top-K.
        tools[..self.top_k.min(tools.len())].to_vec()
    }
}
```

### DynamicHookChain

Wrap hooks in a `DynamicHookChain` to add or remove them after build:

```rust
let dyn_hooks = DynamicHookChain::new();
let handle = dyn_hooks.clone();       // keep for runtime use

let agent = AgentBuilder::new()
    .hook(dyn_hooks)                  // registered as one hook
    .build();

// Enable auditing for a sensitive phase.
handle.add_hook(AuditHook::new());

// Disable it afterwards.
handle.remove_all();
```

Static hooks (registered via `.hook()`) are permanent and compose with the
`DynamicHookChain` — use both: static for safety invariants, dynamic for phases.

### DynamicContextAssembler

Swap the context assembler atomically between pipeline stages:

```rust
let dyn_ctx = DynamicContextAssembler::new(ResearchAssembler);
let handle  = dyn_ctx.clone();

let agent = AgentBuilder::new()
    .context(dyn_ctx)
    .build();

// Phase 1: research
agent.cycle().await?;

// Swap persona — no restart needed.
handle.swap(ReportWriterAssembler);

// Phase 2: write the report
agent.cycle().await?;
```

### Why this matters

These primitives let a **single long-lived agent** serve multiple pipeline stages,
each with different tools, safety hooks, and context strategies. Combined:

```
Stage 1 (research)
  tools: [web_search, extract_stats]
  hooks: [MaxStepsHook(10)]
  assembler: ResearchAssembler

  ── runtime swap ──

Stage 2 (reporting)
  tools: [format_report, send_email]
  hooks: [MaxStepsHook(5), TokenBudgetHook(2000)]
  assembler: ReportWriterAssembler
```

No agent restart. No new process. The bus accumulates the full history.

---

## Extension points

Once you have the basic pipeline working, these are the natural next steps:

### Add MCP tool servers

Use `eventage-mcp` to expose any MCP-compatible server as Eventage tools:

```rust
use eventage_mcp::McpToolset;

let mcp_tools = McpToolset::from_http("http://localhost:3000").await?;
let mut builder = AgentBuilder::new()...;
for tool in mcp_tools.tools() {
    builder = builder.tool_arc(tool);
}
```

### Window the context

Prevent token explosion in long-running sessions by windowing the event log
inside your custom assembler:

```rust
let recent: Vec<_> = ctx.events.iter().rev().take(20).collect();
// Build messages from `recent` instead of all events.
```

### Add memory

Inject external facts into `before_llm`:

```rust
async fn before_llm(&self, _ctx: &HookContext<'_>, messages: &mut Vec<ChatMessage>) -> HookAction {
    let facts = self.memory.recall(&messages).await;
    messages.insert(0, ChatMessage::system(facts));
    HookAction::Continue
}
```

### Evaluation pipeline

After a run, iterate all rejected branches and score them to build an evaluation
dataset — no code changes to the agent required:

```rust
let all_branches = bus.all_rejected_branches().await;
for (branch_id, events) in all_branches {
    let score = evaluate(&events);
    println!("Branch {}: score={}", branch_id, score);
}
```

### Multi-model routing

Swap the LLM provider at build time based on task complexity. Use a fast model
for classification and a powerful model for synthesis:

```rust
let researcher = AgentBuilder::new()
    .llm(OpenAiProvider::ollama("qwen3:4b"))   // fast + cheap
    ...

let orchestrator = AgentBuilder::new()
    .llm(OpenAiProvider::openai(key, "gpt-4o")) // powerful
    ...
```

---

## Summary of framework features used

| Feature | File location |
|---------|---------------|
| `EventBus` (publish / subscribe / log) | `main` — Step 1 |
| `JsonlExporter` + `BusObserver` | `main` — Step 2 |
| `HeartbeatScheduler` | `main` — Step 3 |
| `EventWorker` + `WorkerSet` | `ProgressWorker`, `HeartbeatWorker`, `main` — Step 4 |
| `Tool` trait + `SandboxExecutor` | `WebSearch`, `ExtractStats`, `FormatReport` |
| Delegation tools + `meta_keys::TO_AGENT_ID` | `DelegateToResearcher`, `DelegateToReporter` |
| `CycleHook` (`before_tool`, `after_tool`) | `AuditHook` |
| Custom `ContextAssembler` | `OrchestratorAssembler`, `ResearchAssembler`, `ReporterAssembler` |
| Checkpoint / rollback | `main` — Step 8 |
| Negative-trajectory injection | `ResearchAssembler::assemble` |
| `AgentBuilder` | All three agents |
| `max_react_steps` + `max_concurrent_tools` | Researcher builder |
| Custom event kinds | `research_kinds` |
| `meta_keys` (AGENT_ID, ELAPSED_MS, TO_AGENT_ID) | Throughout |
| `MockLlmProvider` (deterministic testing) | `orchestrator_llm`, `researcher_llm`, `reporter_llm` |
| **Dynamic `ToolRegistry`** | `reporter_tools.add_tool(FormatReport)` — Step 14 |
| **`ToolSelector`** | `KeywordToolSelector` — available on any builder |
| **`DynamicHookChain`** | `reporter_hook_handle.add_hook(AuditHook)` — Step 14 |
| **`DynamicContextAssembler`** | `reporter_ctx_handle.swap(...)` — Step 14 |
