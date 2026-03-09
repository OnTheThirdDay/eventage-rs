# Eventage Framework API Reference

The Eventage framework provides a suite of primitives to build, execute, and scale event-driven LLM agents. This document provides a comprehensive API reference covering all core abstractions, builders, interfaces, and provided implementations.

## Table of Contents
1. [Core Abstractions (`eventage-core`)](#1-core-abstractions-eventage-core)
   - [Event](#11-event)
   - [EventBus](#12-eventbus)
   - [BusConfig & Constants](#13-busconfig--constants)
2. [Agent Orchestration (`eventage-agent`)](#2-agent-orchestration-eventage-agent)
   - [AgentBuilder & Agent](#21-agentbuilder--agent)
   - [ExecutionStrategy](#22-executionstrategy)
   - [ContextAssembler](#23-contextassembler)
   - [CycleHook](#24-cyclehook)
   - [EventWorker](#25-eventworker)
3. [Tool Ecosystem (`eventage-agent::tool`)](#3-tool-ecosystem-eventage-agenttool)
   - [Tool Trait](#31-tool-trait)
   - [ToolRegistry & ToolSelector](#32-toolregistry--toolselector)
4. [Provided Implementations (`eventage-provided-impl`)](#4-provided-implementations-eventage-provided-impl)
   - [Session & AgentSet](#41-session--agentset)
   - [Strategies & Context](#42-strategies--context)
5. [LLM Providers (`eventage-llm`)](#5-llm-providers-eventage-llm)
6. [Sandboxed Tools (`eventage-sandbox`)](#6-sandboxed-tools-eventage-sandbox)
7. [Model Context Protocol (`eventage-mcp`)](#7-model-context-protocol-eventage-mcp)
8. [Persistence (`eventage-sqlite`)](#8-persistence-eventage-sqlite)
9. [Observability (`eventage-observability`)](#9-observability-eventage-observability)

---

## 1. Core Abstractions (`eventage-core`)

The `eventage-core` crate defines the underlying data structures for the event-driven architecture.

### 1.1 `Event`
A structured, immutable data point representing a message, thought, or system pulse. Events form a Directed Acyclic Graph (DAG) when published to the `EventBus`.

```rust
pub struct Event {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub kind: String,                      // e.g., "user.message"
    pub payload: serde_json::Value,        // Dynamic payload data
    pub parent_event_id: Option<Uuid>,     // Link to preceding event (set by EventBus)
    pub metadata: HashMap<String, Value>,  // Tracing and routing data
}
```

**Methods:**
*   `Event::new(kind: impl Into<String>, payload: Value) -> Self`
    Creates a new root event.
*   `Event::with_meta(self, key: impl Into<String>, value: Value) -> Self`
    Builder method to attach metadata (e.g., `agent_id`, `trace_id`).

### 1.2 `EventBus`
An asynchronous publish-subscribe engine backed by an internal DAG store. It guarantees causality and tracks the history of all participants.

**Creation:**
*   `EventBus::new()`
    Creates a bus with `BusConfig::default()`.
*   `EventBus::with_config(config: BusConfig)`
    Creates a bus with custom memory/queue limits.

**Core Methods:**
*   `async fn publish(&self, event: Event) -> Result<(), BusError>`
    Appends the event to the active branch (assigning `parent_event_id`) and broadcasts it.
*   `fn subscribe(&self) -> BusReceiver`
    Returns a channel receiver that streams copies of all subsequent events.
*   `async fn log(&self) -> Vec<Event>`
    Returns a consistent read of the current active branch from root to tip.
*   `async fn wait_for<F>(&self, predicate: F) -> Event where F: Fn(&Event) -> bool`
    Blocks until an event matching the predicate is published.

**Graph Control:**
*   `async fn checkpoint(&self) -> Result<Uuid, BusError>`
    Marks the current graph tip as a safe checkpoint, returning its `EventId`.
*   `async fn rollback(&self, to_checkpoint: Uuid) -> Result<(), BusError>`
    Prunes back the active branch to the checkpoint, sealing the rejected branch and emitting a `system.branch_sealed` event.

### 1.3 `BusConfig` & Constants
Configuration for memory limits and queue behavior.

```rust
pub struct BusConfig {
    pub max_retained_branches: usize, // Default: 10
    pub subscriber_capacity: usize,   // Default: 1024
    pub eviction_strategy: Arc<dyn BranchEvictionStrategy>,
}
```

**Standard Event Kinds (`eventage_core::kinds`):**
*   `USER_MESSAGE` (`"user.message"`)
*   `ASSISTANT_MESSAGE` (`"assistant.message"`)
*   `TOOL_CALL_PROPOSED` (`"tool.call.proposed"`)
*   `TOOL_RESULT` (`"tool.result"`)
*   `AGENT_CYCLE_START` / `AGENT_CYCLE_END`

**Metadata Keys (`eventage_core::meta_keys`):**
*   `AGENT_ID` (`"agent_id"`)
*   `TRACE_ID` (`"trace_id"`)
*   `TO_AGENT_ID` (`"to_agent_id"`)

---

## 2. Agent Orchestration (`eventage-agent`)

The `Agent` orchestrates LLM reasoning, tool execution, and bus I/O based on an injected `ExecutionStrategy`.

### 2.1 `AgentBuilder` & `Agent`
Fluent builder for constructing an `Agent`.

**Builder Methods:**
*   `fn new() -> Self`
*   `fn agent_id(self, id: impl Into<String>) -> Self`
*   `fn bus(self, bus: EventBus) -> Self`
*   `fn llm(self, llm: impl LlmProvider + 'static) -> Self` (Required)
*   `fn strategy(self, strategy: impl ExecutionStrategy + 'static) -> Self` (Required)
*   `fn system_prompt(self, prompt: impl Into<String>) -> Self` 
*   `fn context(self, context: impl ContextAssembler + 'static) -> Self`
*   `fn tool(self, tool: impl Tool + 'static) -> Self`
*   `fn tool_selector(self, selector: impl ToolSelector + 'static) -> Self`
*   `fn hook(self, hook: impl CycleHook + 'static) -> Self`
*   `fn tool_registry(&self) -> ToolRegistry`
*   `fn build(self) -> Agent`

**Agent Methods:**
*   `async fn cycle(&self) -> Result<(), AgentError>`
    Executes a single reasoning cycle.
*   `async fn run(&self) -> Result<(), AgentError>`
    Continuously listens. Blocks the thread, triggering `cycle()` on waking events.
*   `fn tools(&self) -> ToolRegistry`
    Live reference to the tool registry.

### 2.2 `ExecutionStrategy`
Defines the cognitive loop.

```rust
#[async_trait]
pub trait ExecutionStrategy: Send + Sync {
    async fn execute(&self, ctx: &AgentContext) -> Result<(), AgentError>;
}
```
*   `AgentContext`: Passed to strategy, containing `agent_id`, `bus`, `llm`, `tools`, etc.

### 2.3 `ContextAssembler`
Converts raw bus `Event`s into LLM `ChatMessage`s.

```rust
#[async_trait]
pub trait ContextAssembler: Send + Sync {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage>;
}
```

### 2.4 `CycleHook`
Intercept and mutate specific lifecycle segments.

```rust
#[async_trait]
pub trait CycleHook: Send + Sync {
    async fn before_step(&self, ctx: &HookContext<'_>) -> HookAction;
    async fn before_llm(&self, ctx: &HookContext<'_>, messages: &mut Vec<ChatMessage>) -> HookAction;
    async fn before_tool(&self, ctx: &HookContext<'_>, name: &str, args: &Value) -> HookAction;
    async fn after_tool(&self, ctx: &HookContext<'_>, name: &str, result: &Value);
}
```

### 2.5 `EventWorker`
Deterministic actors that trigger native code on specific events without LLMs.

```rust
#[async_trait]
pub trait EventWorker: Send + Sync {
    fn subscribed_kinds(&self) -> Vec<String> { Vec::new() } // empty = all events
    async fn handle(&self, event: &Event, bus: &EventBus) -> Result<(), WorkerError>;
}
```

---

## 3. Tool Ecosystem (`eventage-agent::tool`)

### 3.1 `Tool` Trait
An executable function exposed to the LLM.

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    async fn execute(&self, args: Value) -> Result<Value, AgentError>;
    fn is_terminal(&self) -> bool { false } // Aborts ReactStrategy on execution
}
```

### 3.2 `ToolRegistry` & `ToolSelector`
*   **`ToolRegistry`**: Dynamic, thread-safe map bound to an agent. Methods: `add_tool`, `remove`, `clear`.
*   **`ToolSelector`**: Filter definitions exposed to the LLM per-step. Methods: `async fn select(...)`

---

## 4. Provided Implementations (`eventage-provided-impl`)

### 4.1 Session & AgentSet
*   **`Session`**: Wraps a single agent and bus. 
    *   `session.chat("Hello")`: Synchronous blocking API.
    *   `session.run()`: Async loop for event-driven flows.
*   **`AgentSet`**: Runs multiple agents concurrently on a shared bus.
    *   `AgentSet::new().add_agent(a).add_agent(b).run_until_all_complete().await`

### 4.2 Strategies & Context
*   **`ReactStrategy`**: Loop: Context → LLM → Parallel Tools.
*   **`SingleShotStrategy`**: Call LLM once, publish tools without executing.
*   **`DefaultContextAssembler`**: Standard chronological message flattening.
*   **`NegativeAwareContextAssembler`**: Leverages `bus.rollback()` rejected branches to inject "lessons learned" into the agent's context.

---

## 5. LLM Providers (`eventage-llm`)

```rust
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(&self, messages: Vec<ChatMessage>, tools: Vec<ToolDefinition>) -> Result<LlmResponse, LlmError>;
    fn model(&self) -> &str;
}
```
*   `OpenAiProvider::openai(key, model)`: Hits standard OpenAI API.
*   `OpenAiProvider::ollama(model)`: Targets local Ollama (port 11434).
*   `MockLlmProvider`: Hardcoded outputs for unit testing.

---

## 6. Sandboxed Tools (`eventage-sandbox`)

Run tool execution processes safely.

```rust
#[async_trait]
pub trait SandboxExecutor: Send + Sync {
    async fn execute(&self, req: SandboxRequest) -> Result<SandboxOutput, SandboxError>;
}
```

**Implementations:**
*   `UnsandboxedExecutor`: Normal host OS process spawn.
*   `LandlockExecutor` (Linux only): Confinement of filesystem access paths.
*   `DockerExecutor`: Full container isolation.
*   `WasmExecutor`: Running WASM / WASI bytecode safely.

---

## 7. Model Context Protocol (`eventage-mcp`)

Native MCP client and tools.

```rust
// Connect to standard MCP server and load capabilities dynamically:
let toolset = McpToolset::from_http("http://localhost:3000/mcp").await?;

// Agent adopts all tools transparently via JSON-RPC:
toolset.add_to_registry(&agent.tools());
```

---

## 8. Persistence

Backed event stores for state-saving, memory, and checkpoints. One provided implementation is a bundled SQLite  (`eventage-sqlite`) to avoid system dependencies (`features = ["bundled"]`). The framework also supports implementation of any other persistence options.

### `SqliteEventStore`
Read/write historic events without blocking Tokio tasks.

*   `SqliteEventStore::new(path)`
*   `async fn append(&self, event: &Event) -> Result<()>`
*   `async fn load_all(&self) -> Result<Vec<Event>>`
    Useful to recreate state: `bus.restore_from(store.load_all().await?)`
*   `async fn load_since_idx(&self, after_idx: i64) -> Result<Vec<Event>>`

### `SqliteExporter`
An `ObservabilityExporter` to stream live events asynchronously exactly to a SQLite DB.

---

## 9. Observability (`eventage-observability`)

Record bus traffic transparently to various backends via the `BusObserver`.

```rust
pub struct BusObserver { ... }

impl BusObserver {
    pub fn new(bus: EventBus) -> Self;
    pub fn add_exporter(self, exporter: impl ObservabilityExporter) -> Self;
    pub async fn run(self);
}
```

**`ObservabilityExporter` Trait:**
```rust
#[async_trait]
pub trait ObservabilityExporter: Send + Sync {
    async fn export(&self, event: &Event) -> Result<(), ObsError>;
    async fn flush(&self) -> Result<(), ObsError> { Ok(()) }
}
```

**Provided Exporters (`eventage-provided-impl`):**
*   `JsonlExporter`: Writes all events to a `.jsonl` file. Compatible with `eventage-replay`.
*   `OtelExporter`: Emits OpenTelemetry traces matching cycle span IDs to trace graphs and metrics perfectly.
