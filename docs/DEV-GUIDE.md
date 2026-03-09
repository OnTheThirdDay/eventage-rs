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
