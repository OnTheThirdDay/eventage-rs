# Eventage — Developer Guide

Eventage is an event-driven agent framework for Rust. It provides modularity and complete observability to build everything from simple chatbots to highly complex, multi-agent coding systems.

This guide explores every concept in the Eventage framework and relevant provided implementations.

---

## 1. Introduction & Architecture

The central philosophy of Eventage is simple: **everything that happens in the system is an event on a shared bus**. 

Human inputs, LLM generated contents, tool executions, and inter-agent routing are all emitted as events over an append-only, Directed Acyclic Graph (DAG) log called the `EventBus`.

This design inherently decouples the system. It enables precise reproducibility, fault tolerance, live observability, speculative branching (DAG rollback), and effortless multi-agent concurrency.

### The Agent Loop

Because the `EventBus` is the center of the universe, an "Agent" is simply a component that listens to the bus and responds. 

Here is exactly how an Eventage agent (using the default ReAct strategy) executes its reasoning loop over the bus:

```text
┌─────────────────────────────────────────────────────────────┐
│                          EventBus                           │
│          (append-only DAG log + broadcast channel)          │
└─┬─────────────────────────────▲───────────────────────────┬─┘
  │ subscribe                   │ publish                   │ subscribe
  │                             │                           │
  ▼                             │                           ▼
┌─┴─────────────────────────────┴─────────────────┐  ┌──────┴──────────────┐
│                     Agent                       │  │                     │
│               (LLM intelligence)                │  │    EventWorker      │
│ 1. Agent::run() wakes on `user.message`, etc.   │  │ (Automation Harness)│
│                                                 │  │ e.g. HeartbeatRunner│
│ 2. Agent::cycle()                               │  └─────────┬───────────┘
│ ├── Publish `agent.cycle.start`                 │            │ publish
│ │                                               │  ┌─────────┴───────────┐
│ └── 3. ExecutionStrategy (e.g. ReactStrategy)   │  │                     │
│     │                                           │  │    Observability    │
│     ├── loop:                                   │  │   (JSONL, Otel)     │
│     │   ├── a. CycleHook::before_step()         │  │ e.g. JsonlExporter  │
│     │   ├── b. ContextAssembler::assemble()     │  └─────────────────────┘
│     │   │      (e.g. DefaultContextAssembler)   │
│     │   ├── c. CycleHook::before_llm()          │
│     │   ├── d. ToolSelector::select()           │
│     │   ├── e. LlmProvider::complete()          │
│     │   │      (e.g. OpenAiProvider)            │
│     │   ├── f. Publish `assistant.message`      │
│     │   │                                       │
│     │   └── g. For each Tool Call:              │
│     │       ├── Publish `tool.call.proposed`    │
│     │       ├── CycleHook::before_tool()        │
│     │       │   (e.g. StdinApprovalGate for     │
│     │       │    human-in-the-loop)             │
│     │       ├── Tool::execute() (concurrent)    │
│     │       │   (e.g. McpTool, SandboxTool)     │
│     │       ├── CycleHook::after_tool()         │
│     │       └── Publish `tool.result`           │
│     │                                           │
│     └── Publish `agent.cycle.end`               │
└─────────────────────────────────────────────────┘
```

When an agent wakes up dynamically to process the bus (Step 1), it starts a reasoning cycle (Step 2) and hands execution to its **`ExecutionStrategy`** (Step 3). Note that every component inside the agent is just a trait implementation — you can swap them out or build your own.

Inside the strategy loop, the agent uses its **`ContextAssembler`** (such as the chronological `DefaultContextAssembler`) to parse the raw DAG of past events into the exact chat structure required by the LLM. It queries its **`LlmProvider`** (like `OpenAiProvider`), safely executes requested **`Tools`** (like `eventage-mcp` tools or shell sandboxes) concurrently, and intercepts execution along the way via **`CycleHooks`** (like a `StdinApprovalGate` that pauses execution until a human types "y" for a dangerous action).

Because everything happening in Step 3 is published back to the `EventBus` in real-time, independent entities like **`EventWorker`**s (such as the `HeartbeatScheduler` triggering async events) and **`Observability`** exporters (like the `JsonlExporter`) can instantly stream, save, or react to the agent's thought process without ever blocking the agent itself.

### The Crate Architecture

The framework is strictly separated into the core abstraction layer (`eventage-agent` and `eventage-core`) and the concrete tools to actually build with it (`eventage-provided-impl` and auxiliaries).

*   **`eventage-core`**: The absolute minimum. Defines `Event`, `EventBus`, and basic primitives.
*   **`eventage-agent`**: The core traits defining what an agent *is*. It provides traits like `Agent`, `Tool`, `CycleHook`, `ContextAssembler`, and `ExecutionStrategy`.
*   **`eventage-provided-impl`**: The "batteries included" facade. Provides concrete, ready-to-use implementations of the traits from `eventage-agent` (like `Session`, `AgentSet`, `WorkerSet`, strategies, and context assemblers).
*   **Auxiliary Crates**: Capabilities separated strictly by dependency domain (`eventage-llm`, `eventage-sandbox`, `eventage-mcp`, `eventage-sqlite`, `eventage-observability`).

---

## 2. Core Concepts: The Bus

At the bottom of the stack sits `eventage-core`. These concepts ensure the system remains perfectly auditable.

### 2.1 Event
An `Event` is an immutable, UUID-identified packet of data.
*   `kind`: A namespaced string defining the event (e.g., `"user.message"`, `"assistant.message"`, `"tool.call.proposed"`).
*   `payload`: A JSON payload capturing the contents (e.g., text, parameters, or tool outputs).
*   `metadata`: Contextual data like trace IDs, routing information, or token counts.
*   `parent_event_id`: A pointer to the previous event, forming the DAG.

### 2.2 EventBus
The `EventBus` is an append-only DAG log and a live broadcast channel. 
Every component in Eventage takes a `Clone` of the same bus. When they publish, the event is appended to the active branch. When they subscribe, they listen live.

### 2.3 DAG Checkpointing & Rollback
Because the bus is a DAG instead of a flat list, Eventage natively supports speculative execution.
You can create a checkpoint, try a risky operation (like executing uncertain LLM code), and if it fails, you `rollback()` to the checkpoint. The failed events are cleanly snipped off the active branch and stored as a "rejected branch," which can later be used to teach an agent what *not* to do.

---

## 3. Building an Agent

An agent in Eventage is a reactive loop that listens to the bus, thinks using an LLM, and takes actions.
The `eventage-agent` crate provides the builder; `eventage-provided-impl` provides high-level orchestrators.

### 3.1 The Agent Core
An `Agent` relies on an `EventBus`, an `LlmProvider`, and a collection of capabilities (tools, hooks, strategies).
When you call `agent.run()`, the agent drops into a reactive mode, waking automatically up upon relevant trigger events to begin a reasoning cycle.

### 3.2 The Session API (`eventage-provided-impl`)
For the vast majority of human-in-the-loop interactions, manually publishing to the bus is tedious. The `Session` API neatly wraps an `Agent` and its internal `EventBus`, exposing a clean `.chat("Question")` async endpoint.

**How they work together:**
Under the hood, `Session` publishes a `"user.message"` event, triggers the wrapped agent's cycle execution, and listens to the bus sequentially until it parses out the `"assistant.message"` event to return to the caller.

### 3.3 Multi-Agent orchestration (`eventage-provided-impl`)
When multiple agents share the same `EventBus`, they can communicate asynchronously.
*   `AgentSet` simplifies booting up and managing arbitrary squads of concurrent agents hooked to the same bus.
*   An agent can publish an `"agent.message"` addressed directly to `meta_keys::TO_AGENT_ID`. 
*   `AgentSet` limits concurrency (via `max_concurrent()`) so you don't blow out your LLM rate limits when operating swarm intelligence.

---

## 4. Context Assembly

How does an LLM see the `EventBus` log? Through a `ContextAssembler`.
The assembler's job is to read the raw DAG of events and format them into an array of `ChatMessage`s the specific LLM can understand.

### Provided Implementations (`eventage-provided-impl`):
1.  **`DefaultContextAssembler`**: The standard chronological parser. It maps `"user.message"` to User, `"assistant.message"` to Assistant, etc.
2.  **`NegativeAwareContextAssembler`**: Uses the DAG's rejected branches. If a rollback occurs, this assembler gathers the rejected trajectory and inserts it into the prompt with instructions like *"You previously attempted X, which failed with error Y."*
3.  **`DynamicContextAssembler`**: A Thread-safe Mutex-backed assembler that lets you hot-swap the internal context logic perfectly mid-flight without pausing the agent.
4.  **`ToolResultClearingAssembler`** *(context editing)*: The cheap first line of context management. Once the assembled context crosses a token trigger, the *content* of the oldest tool results is replaced with a short placeholder — zero LLM calls, and lossless because the full output still lives in the event log. Clearing is a monotonic ratchet, so the edited prefix stays stable for provider prompt caches.
5.  **`SummarizingContextAssembler`**: The heavyweight fallback. When the context still exceeds a token budget, the oldest conversation messages are folded into an LLM-generated summary (optionally archived to disk). Compose it *around* the clearing assembler so summarization only fires when clearing alone cannot reclaim enough:

```rust
let base = DefaultContextAssembler::new("You are helpful.");
let clearing = ToolResultClearingAssembler::new(Arc::new(base), 24_000);
let assembler = SummarizingContextAssembler::new(Arc::new(clearing), llm, 32_000, "session-id");
```

```rust
// Instantiating and attaching a Context Assembler
let assembler = DefaultContextAssembler::new(
    "You are a helpful assistant.",
    // Optional parameter to ignore events older than X to save tokens
    Some(100) 
);

let agent = AgentBuilder::new()
    .context_assembler(assembler)
    .build();
```

---

## 5. Execution Strategies

An `ExecutionStrategy` dictates the precise orchestration inside a reasoning cycle. Should the agent think step-by-step, plan upfront, or fire just once?

### Provided Implementations (`eventage-provided-impl`):
1.  **`ReactStrategy` (Reason + Act)**: The heavyweight champion. The LLM generates thought + tool calls, the tools execute, the results are pushed to the bus, and the LLM is queried again until it yields a final answer. 
2.  **`SingleShotStrategy`**: Highly optimized for simple text transformation or routing. It calls the LLM exactly once, preventing tool-loop spin-outs and dramatically reducing latency.

`ReactStrategy` ships with harness guardrails **on by default**:

| Field | Default | Behavior |
|---|---|---|
| `tool_timeout` | `300s` | A hung tool produces an error `tool.result` (visible to the model) instead of blocking the cycle forever. |
| `max_tool_result_chars` | `30_000` | Oversized tool outputs are middle-truncated (head + tail preserved) with an explanatory marker; the full output stays in the event log. |
| `finalize_on_max_steps` | `true` | On step-budget exhaustion the strategy makes one final *tool-free* LLM call, nudging the model to report progress, remaining work, and blockers — instead of erroring with `MaxStepsReached`. |

Malformed tool arguments (invalid JSON) never reach the tool: the parse error is returned to the model as a tool error so it can self-correct on the next step. Parsed arguments are additionally validated against the tool's JSON Schema (`type` / `required` / `properties` / `items` / `enum`) — violations are phrased for the model (`ToolExecOptions { validate_args: false, .. }` opts out).

**Streaming**: set `ReactStrategy { stream: true, .. }` to use `LlmProvider::complete_stream`. Incremental tokens are broadcast as **ephemeral** `assistant.delta` events (`EventBus::broadcast` — delivered to subscribers, never stored in the DAG or the LLM context); the durable `assistant.message` event is unchanged. All three native providers stream via SSE (including reasoning deltas and usage); providers without native support fall back to a single delta per response, so TUIs can subscribe uniformly.

### LLM providers

| Provider | API | Highlights |
|---|---|---|
| `AnthropicProvider` | Messages API (native) | Automatic prompt caching, extended thinking (`with_thinking(budget)`) with signature round-trip, betas, top_k/stop sequences |
| `OpenAiResponsesProvider` | Responses API | Encrypted reasoning items round-trip statelessly through the event log, `with_reasoning_effort("high")` |
| `OpenAiProvider` | Chat Completions | Works with any compatible server (Ollama, Groq, vLLM, OpenRouter…); typed sampling params, `with_json_schema` structured outputs, `with_body_param` escape hatch |

Wrap any provider with `RetryProvider` (transient-error backoff) and `RateLimitedProvider` (request pacing); both forward streaming. Provider-specific reasoning state (thinking blocks, reasoning items) is carried in `provider_extra`, persisted on `assistant.message` events, and restored automatically by the context assemblers — multi-step tool loops keep their chain of thought with zero configuration.

### Skills, project context, and plugins

- **Skills** (`eventage::agent::skills`): point `SkillsLibrary::discover` at a directory of Claude-compatible `SKILL.md` bundles, append `library.system_prompt_section()` to the prompt, and register `SkillTool::new(library)`. The model sees one line per skill and loads full instructions on demand.
- **Project context** (`eventage::agent::project`): `load_project_context(dir)` reads `AGENTS.md` (or `CLAUDE.md`); `load_project_context_walkup` collects nested files monorepo-style, nearest last.
- **Plugins** (`eventage::plugin`): `PluginHost::load` reads `eventage-plugin.toml` manifests (prompt fragment + skills dir + MCP servers); `host.install(&registry)` connects everything and returns the combined prompt fragment.

```rust
// Choosing a strategy for the agent
let strategy = SingleShotStrategy::new(); 
// let strategy = ReactStrategy::new().max_tool_iterations(5);

let agent = AgentBuilder::new()
    .strategy(strategy)
    .build();
```

---

## 6. Tools, Selection & Execution

Tools are the hands of the LLM. The `Tool` trait requires two methods:
*   `definition()`: Yields the JSON Schema given to the LLM.
*   `execute()`: The async runtime logic.

### 6.1 Tool Registration & Dynamic Tooling
You can bind static tools on boot using `AgentBuilder::tool()`. However, `eventage` provides a `ToolRegistry` that can be dynamically updated. Tools can be injected or pruned at any point in the lifecycle.

```rust
// 1. Static registration at boot
let agent = AgentBuilder::new()
    .tool(MyWeatherTool::new())
    .tool(MyCalculatorTool::new())
    .build();

// 2. Dynamic tooling via Registry
let registry = ToolRegistry::new();
registry.register(MyWeatherTool::new()).await;

// Later, another component can add or remove tools while the agent runs!
registry.remove("weather_tool").await;
```

### 6.2 Tool Selectors
Sometimes an agent has 1,000 tools but context limits demand we only send 10. A `ToolSelector` filters the `ToolRegistry` before the LLM prompt is assembled.
*   **Provided Impl**: `KeywordToolSelector` (filters based on prompt keywords).

### 6.3 Expanding your capability
*   **MCP (Model Context Protocol)**: Through `eventage-mcp`, any standard MCP server can be wrapped seamlessly as an Eventage `McpTool` and handed to the agent.
*   **Sandboxing**: Heavy tools (like executing compiled C code) need safety. `eventage-sandbox` provides pluggable execution environments, from local processes (`UnsandboxedExecutor`), to lightweight kernel limits (`LandlockExecutor`), to full isolation (`DockerExecutor`).

---

## 7. Lifecycle Hooks

A `CycleHook` lets you cleanly intercept, observe, or manipulate the agent processing loop without touching core code. Hooks attach to `before_step`, `before_llm`, `before_tool`, or `after_tool`.

Typical uses include:
*   Building manual approval gates (`before_tool` pausing).
*   Injecting runtime RAG context (`before_llm` mutating the `messages` array).

Hooks return a `HookAction`:
*   `Continue` — proceed normally.
*   `Skip` — veto the operation silently (the model sees a generic "vetoed by hook" result).
*   `Deny(reason)` — veto a tool call **and tell the model why**. The reason lands in the synthetic `tool.result`, so the model can pick another tool, adjust arguments, or ask the user — prefer this over `Skip` for permission gates.
*   `AbortCycle` — end the cycle immediately.

### Provided governance hooks

*   **`PermissionPolicyHook`** — glob-based `allow` / `deny(reason)` / `ask` rules evaluated first-match-wins, with `deny_by_default` / `ask_by_default` hardening. `ask` publishes a durable `permission.request` event and waits (with timeout) for a `permission.decision` published by any approver on the bus — approval is fully asynchronous and transport-agnostic.

    ```rust
    let policy = PermissionPolicyHook::new()
        .allow("read_*").allow("search_*")
        .ask("write_*")
        .deny("delete_*", "deletion is disabled")
        .deny_by_default("tool not allowlisted");
    ```

*   **`TokenBudgetHook`** — enforces a token ceiling computed from usage metadata in the event log (session-wide by default, `agent_scoped()` optional). At 80% it publishes `budget.warning` and injects a wrap-up note into the prompt; at 100% it publishes `budget.exhausted` and aborts the cycle. Accounting is derived from the bus, so it survives restore-from-SQLite restarts.

---

## 7.5 Speculative Best-of-N Execution

`eventage::agent::speculate::best_of_n` runs N candidate agents **in parallel forks of the bus**, scores their trajectories, splices the winner onto the main log, and seals every loser as a rejected branch (feeding `NegativeAwareContextAssembler`).

```rust
use eventage::agent::speculate::{best_of_n, SpeculationCandidate, LlmJudgeScorer};

let outcome = best_of_n(&bus, vec![
    SpeculationCandidate::new("temp-0.2", |fork| build_agent(fork, 0.2)),
    SpeculationCandidate::new("temp-0.9", |fork| build_agent(fork, 0.9)),
], &LlmJudgeScorer::new(judge, "correctness and brevity")).await?;
```

Scorers implement `BranchScorer`: use `FnScorer` for heuristics (tests pass? shortest diff?) or `LlmJudgeScorer` for LLM-as-judge on a 0–10 scale. The round is recorded as a durable `speculation.completed` event.

### Provided Implementation (`eventage-provided-impl`):
*   **`DynamicHookChain`**: Allows for hooks to be safely swapped, updated, or manipulated at runtime on a live agent system.

```rust
// Attaching multiple hooks to an agent
let agent = AgentBuilder::new()
    .hook(RequestLoggerHook) // Logs every incoming user message
    .hook(CostLimiterHook::new(5.00)) // Aborts cycle if OpenAI bill exceeds $5
    .hook(StdinApprovalGate) // Manually approve tools dropping DB tables
    .build();
```

---

## 8. Event Workers

Not everything requires an LLM. An `EventWorker` is a trait for deterministic code that statically subscribes to specific event kinds, processes data, and potentially emits new events in response.

Workers are perfect for workflows: step sequencers, async data pipelines, human-approval bridges, or webhook triggers.

### Provided Implementations (`eventage-provided-impl`):
*   **`WorkerSet`**: Bootstraps multiple independent workers to listen identically on the shared bus.
*   **`DynamicWorkerHandle`**: Lets applications seamlessly suspend, resume, or cleanly kill asynchronous background workers remotely at runtime.

```rust
// Defining and running a background EventWorker
#[async_trait]
impl EventWorker for AnalyticsWorker {
    async fn start(&self, mut bus: EventBus) -> anyhow::Result<()> {
        let mut sub = bus.subscribe();
        while let Ok(event) = sub.recv().await {
            if event.kind == "tool.call.proposed" {
                println!("Recording analytics: Tool {} was called", event.payload["name"]);
            }
        }
        Ok(())
    }
}

// Spin it up in the background
let worker = AnalyticsWorker::new();
tokio::spawn(async move { worker.start(bus).await });
```

---

## 9. Observability & Persistence

Because everything travels over the `EventBus`, observing or saving an agent’s state is trivial.
*   **`BusObserver`**: A lightweight trait to intercept and handle live streamed events continuously.

### Provided Observability (`eventage-provided-impl`):
*   **`JsonlExporter`**: Writes all events robustly to a JSON Lines file.
*   **`OtelExporter`**: Native integration with OpenTelemetry traces, pushing agent cycles directly to backends like Jaeger or DataDog.

### Persistence (`eventage-sqlite`):
*   **`SqliteEventStore`**: Plugs securely into the dag interface to durably capture the graph context.
*   **`SqliteExporter`**: Constantly streams the active volatile memory DAG straight into a robust SQLite replica to survive full power outages. When rebooting, you feed the SQLite replica back into an `EventBus` to achieve infinite memory persistence.

```rust
// 1. Live file logging
let json_exporter = JsonlExporter::new(File::create("agent_run.jsonl")?);
tokio::spawn(async move { json_exporter.observe(bus_for_json).await });

// 2. OpenTelemetry Tracing (e.g., Jaeger)
let otel_exporter = OtelExporter::new();
tokio::spawn(async move { otel_exporter.observe(bus_for_otel).await });

// 3. SQLite DB Persistence
let sqlite_exporter = SqliteExporter::new(db_connection);
tokio::spawn(async move { sqlite_exporter.observe(bus_for_db).await });
```

---

## 10. The Big Picture

When executed, an Eventage system follows this cadence:
1.  **Configure:** Load LLMs (`eventage-llm`), strategies, workers (`eventage-provided-impl`), and start persistence (`eventage-sqlite`).
2.  **Listen:** A `Session` or `Worker` publishes a trigger event (`"user.message"`) onto the shared `EventBus`.
3.  **Assemble:** An Agent wakes up. Its `ContextAssembler` filters the DAG into LLM chat arrays, potentially injecting `rejected_branches` for negative learning.
4.  **Execute:** The `ExecutionStrategy` (e.g. `ReactStrategy`) orchestrates the loop. Before talking to the LLM, `CycleHook`s run (enabling RAG). The LLM processes the data. If tools are requested, they might be securely trapped inside an `eventage-sandbox` before yielding the result event.
5.  **Broadcast:** Everything happening inside the execution cycle (heartbeats, LLM responses) emits live back into the bus. Observers instantly stream changes to UI, file, and tracing backends.

By embracing the event bus, Eventage brings complete predictability to highly unpredictable AI behavior.

---

## 11. Advanced Capabilities

### Multimodal content

`ChatMessage` carries either plain text or ordered `ContentPart`s (text +
images, from a URL or inline base64). Publish a multimodal turn by putting
parts in the event payload:

```rust
let parts = vec![
    ContentPart::text("What regressed in this chart?"),
    ContentPart::image_base64("image/png", &b64),
];
bus.publish(Event::new(kinds::USER_MESSAGE,
    json!({ "parts": serde_json::to_value(&parts)? }))).await?;
```

Each provider maps parts to its own wire format (Chat Completions
`image_url`, Anthropic `image` blocks, Responses `input_image`), and the
token estimator charges images so they cannot silently blow a budget.

### Token calibration

Estimating tokens without the model's tokenizer is guesswork, so the harness
*measures* its own error: every `assistant.message` records the pre-call
estimate alongside the provider's real `input_tokens`, and `TokenCalibration`
learns the ratio (EWMA, outlier-clamped). The summarizing and clearing
assemblers calibrate themselves from the events they already receive — no
wiring needed. Share one calibration across both with `with_calibration`.

### Structured output

`complete_structured` is object-safe and returns JSON; `complete_as::<T>()`
adds typing and validates the result against the schema before deserializing:

```rust
let verdict: Verdict = llm.complete_as(messages, "verdict", schema).await?;
```

Native paths: Chat Completions `response_format`, Anthropic forced tool use,
Responses `text.format`. Providers without native support fall back to
prompted JSON with fenced-block extraction, so this works everywhere —
including local models.

### Crash recovery for interrupted tools

A process that dies between `tool.call.proposed` and `tool.result` leaves an
orphaned call. Call `reconcile_interrupted_tools` after restoring a log:

```rust
let policy = ToolRecovery::new().replay("read_*").fail("transfer_*");
reconcile_interrupted_tools(&bus, &policy, Some(&registry)).await?;
```

True exactly-once is impossible without tool cooperation, so the policy is
explicit per tool: `ReportInterrupted` (default, **at-most-once** — tells the
model the outcome is unknown), `Replay` (**at-least-once**, idempotent tools
only), or `Fail`. Reconciliation also repairs the history so providers stop
rejecting the next request.

### Beam search (per-step speculation)

Where `best_of_n` speculates over whole cycles, `beam_search` speculates at
every ReAct step — exploring *actions*, not just wordings:

```rust
let config = BeamConfig { candidates_per_step: 3, beam_width: 1, ..Default::default() };
let outcome = beam_search(&bus, &config, &scorer, |fork, candidate| {
    build_agent(fork, temperature_for(candidate))
}).await?;
```

Every pruned trajectory is sealed as a rejected branch, feeding
`NegativeAwareContextAssembler`. Cost scales with
`beam_width × candidates_per_step` per step — use a cheap model to explore.

### Distributed bus

`DistributedBus` bridges a local `EventBus` onto a `BusTransport`; the
built-in `TcpTransport` speaks newline-delimited JSON with no broker and no
extra dependencies. Events carry an origin marker so nothing echoes, and
`filter_kinds` keeps bulky local traffic off the wire.

```rust
DistributedBus::new(bus.clone(), TcpTransport::listen("0.0.0.0:7700").await?)
    .with_node_id("orchestrator")
    .filter_kinds(vec![kinds::AGENT_MESSAGE])
    .spawn();
```

Ordering is per-connection, not global, and delivery is best-effort — treat
it as a coordination and observation fabric, and keep one writer per logical
conversation. Implement `BusTransport` for NATS/Redis/Kafka as needed.

### MCP elicitation

Attach a bus to an MCP client with `with_bus` and server-initiated
elicitation requests become `mcp.elicitation.request` events answered by
`mcp.elicitation.response` — the same asynchronous approval pattern as tool
permissions. `notifications/tools/list_changed` surfaces as
`mcp.tools.changed` to drive `McpToolset::reload`.

---

## 12. Roadmap

- **Global ordering for the distributed bus** — the current transport is
  best-effort and per-connection; consensus or a log-server would be needed
  for a single authoritative cross-host branch.
- **Schema generation** — `complete_as` takes an explicit JSON Schema;
  deriving it from `T` (via `schemars`) would remove the duplication.
- **Bounded active-path memory** — eviction currently applies to rejected
  branches only, so a 24/7 single-session agent grows without bound.
- **Richer multimodal input** — audio and document parts (images are
  supported today), plus provider-accurate image token accounting.
- **MCP resources, prompts, and sampling** — tools and elicitation are
  covered; the remaining server primitives are not.
