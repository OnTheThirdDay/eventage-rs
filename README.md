# Eventage

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Eventage is an event-driven agent framework for Rust. It provides modularity and complete observability to build everything from simple chatbots to highly complex, multi-agent coding systems.

In Eventage, **everything that happens in the system is an event on a shared bus.**

Human inputs, LLM generated contents, tool executions, and inter-agent routing are all emitted as events over an append-only, Directed Acyclic Graph (DAG) log called the `EventBus`.

This design inherently decouples the system. It enables precise reproducibility, fault tolerance, live observability, speculative branching (DAG rollback), and effortless multi-agent concurrency.

---

## The Architecture at a Glance

In Eventage, an agent subscribes to the `EventBus` and makes decisions using an LLM. Because the bus is the central nervous system, independent entities — like Background Workers or Observability exporters — can easily live on the exact same bus without getting in the agent's way.

```text
┌─────────────────────────────────────────────────────────────┐
│                          EventBus                           │
│          (append-only DAG log + broadcast channel)          │
└─┬─────────────────────────────▲───────────────────────────┬─┘
  │ subscribe                   │ publish                   │ subscribe
  │                             │ (e.g. `system.heartbeat`) │
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

Because every internal stage is abstracted behind a trait in `eventage-agent` (Context Assemblers, Strategies, Hooks, LLM Providers, and Tools), the entire reasoning loop is perfectly pluggable.

---

## 💡 Key Capabilities

Because Eventage uses an append-only DAG instead of hidden mutable state arrays, you gain incredible capabilities simply by interacting with the `EventBus`.

### 1. Zero-Cost Speculative Execution (DAG Rollback)
If an agent makes a mistake, you don't need to try and manually delete messages from the end of an opaque array. You simply rollback the DAG to a known safe point. The failed trajectory is automatically sealed as a "rejected branch" that the agent can later learn from.

```rust
// Mark a safe point before telling the agent to do something risky
let safe_point_id = bus.checkpoint().await?;

// ...Agent attempts an unsafe external operation and fails...
bus.publish(Event::new("tool.result", json!({ "error": "compile failed" }))).await?;

// Roll back the graph! The active branch instantly reverts, and all
// failed events are cleanly snipped off into a "rejected" trajectory.
bus.rollback(safe_point_id).await?;

// We can now pass those rejected trajectories to the LLM context assembler
// so the agent inherently knows "I tried that before, and it didn't work."
```

### 2. Live, Invisible Observability & Interception
Want to watch exactly what an agent is doing in real-time, or pause it to inject a RAG document? You don't have to rewrite the agent. You just attach a subscriber to the bus, or a `CycleHook`.

```rust
// 1. Instantly stream every thought, heartbeat, and tool call to a file
let exporter = JsonlExporter::new(std::fs::File::create("agent_log.jsonl")?);
tokio::spawn(async move {
    exporter.observe(bus.clone()).await; 
});

// 2. Or attach a Hook to pause the agent mid-thought for human approval
AgentBuilder::new()
    .bus(bus)
    .llm(openai)
    .hook(StdinApprovalGate) // Wait for a human to type "y" in the terminal before running a tool
    .build();
```

![eventage-replay](assets/eventage-replay.png)


### 3. Effortless Multi-Agent Swarms
Boot up multiple independent agents on the same or separate `EventBus`. Because the bus is the single reliable source of truth, agents organically share a context graph and can route messages to each other asynchronously without writing complex peer-to-peer networking logic.

### 4. Sandboxing
When an agent decides to compile arbitrary C code or run a shell script, Eventage supports wrapping those tool executions in pluggable execution contexts. Be honest about what each buys you: **`DockerExecutor`** is the option for untrusted code (no network, memory/CPU/pid caps, `--cap-drop=ALL`, `no-new-privileges`, fresh container per call, killed on timeout); **`LandlockExecutor`** is filesystem-only defense-in-depth (no network or resource isolation — a seatbelt, not a boundary); **`WasmExecutor`** runs WASI modules with fuel, memory, and output caps. All executors run with a scrubbed environment so host API keys never reach sandboxed code.

### 5. Harness Guardrails, On By Default
The ReAct loop is production-hardened out of the box: per-tool wall-clock timeouts, middle-truncation of oversized tool outputs (the full data stays in the event log), model-visible feedback for malformed tool arguments and policy denials (`HookAction::Deny("reason")`), loop/stuck detection whose hints actually reach the model, and a graceful tool-free wrap-up turn when the step budget runs out. The LLM layer composes `RetryProvider` (exponential backoff on 429/5xx) with `RateLimitedProvider` (request pacing).

### 6. Layered Context Management: Edit the View, Not the History
Because the event log is the source of truth, context management is *assembly-time editing* — lossless by construction. `ToolResultClearingAssembler` reclaims budget for free by clearing stale tool outputs from the LLM's view (a monotonic ratchet that stays prompt-cache friendly), and `SummarizingContextAssembler` folds old conversation into an LLM-generated summary only when clearing isn't enough. Reasoning traces from thinking models and cached-token usage are captured on every `assistant.message` event for observability and replay.

### 7. Speculative Best-of-N Execution
The capability the DAG was built for: fork the bus once per candidate, run N agents on the same task in parallel (different temperatures, models, or strategies), score the trajectories (heuristic or LLM judge), splice the winner onto the main log, and seal the losers as rejected branches the agent can *learn from* on future turns.

```rust
let outcome = best_of_n(&bus, vec![
    SpeculationCandidate::new("fast", |fork| build_agent(fork, "gpt-5-mini")),
    SpeculationCandidate::new("deep", |fork| build_agent(fork, "gpt-5")),
], &LlmJudgeScorer::new(judge_llm, "correct, concise, complete")).await?;
```

### 8. Enterprise Governance & Operations
- **`PermissionPolicyHook`** — glob-based allow/deny/ask rules per tool. `ask` publishes a `permission.request` event and waits for a `permission.decision` from *any* approver on the bus (TUI, Slack bridge, dashboard, another agent), denying on timeout.
- **`TokenBudgetHook`** — session- or agent-scoped token budgets computed from the event log (they survive restarts): a warn threshold nudges the model to wrap up, the hard ceiling stops the loop and emits `budget.exhausted`.
- **Native streaming** — `LlmProvider::complete_stream` with SSE support in every native provider; `ReactStrategy { stream: true, .. }` broadcasts ephemeral `assistant.delta` events that never pollute the durable log.
- **Schema-validated tool calls** — arguments are checked against each tool's JSON Schema before execution; violations go back to the model as correctable errors.
- **Durable resume** — `Session::builder().bus(restored_bus)` continues a conversation from a SQLite-restored event log; rollbacks are replayed faithfully.

### 9. First-Class Provider Support
Three native providers, all streaming, all reasoning-aware:
- **`AnthropicProvider`** — native Messages API with **automatic prompt caching** (`cache_control` breakpoints on the system prompt and the moving conversation tip) and **extended thinking**: thinking blocks and signatures are persisted on the event bus and replayed across tool-loop steps as the API requires.
- **`OpenAiResponsesProvider`** — the Responses API for reasoning models: encrypted reasoning items round-trip through the event log (`store: false`, fully stateless), with `reasoning_effort` control.
- **`OpenAiProvider`** — Chat Completions for OpenAI and every compatible server (Ollama, Groq, Mistral, vLLM, OpenRouter…), with typed params (temperature, top_p, max_tokens, stop, seed, penalties, `reasoning_effort`, `parallel_tool_calls`), **structured outputs** (`with_json_schema`), and an `extra_body` escape hatch.

Compose freely: `RetryProvider::new(RateLimitedProvider::new(AnthropicProvider::new(...), 60))`.

### 10. Skills, AGENTS.md, and Plugins
- **Skills** — drop Claude-compatible `SKILL.md` bundles in a directory; `SkillsLibrary` discovers them and the `skill` tool loads them on demand (progressive disclosure keeps the context lean).
- **AGENTS.md / CLAUDE.md** — `load_project_context` (and its monorepo walk-up variant) injects project instructions into the system prompt.
- **Plugins** — distributable directories with an `eventage-plugin.toml` manifest bundling skills, MCP servers (stdio or HTTP, auto-prefixed), and prompt fragments; `PluginHost::install` wires everything into an agent in one call.
- **MCP 2025-06-18** — stdio + Streamable HTTP transports, session IDs, version negotiation with `MCP-Protocol-Version` echo, SSE responses, and `structuredContent` tool results.

---

## ⚡ Getting Started

The quickest way to get an agent off the ground is using the **Session API** provided by `eventage-provided-impl`, which wraps an internal `EventBus` and `Agent` away behind a clean chat interface.

```toml
[dependencies]
eventage-core = "0.1"
eventage-provided-impl = "0.1"
eventage-llm = "0.1"
tokio = { version = "1", features = ["full"] }
```

```rust
use eventage_provided_impl::Session;
use eventage_llm::OpenAiProvider;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Build a session with an Ollama provider
    let mut session = Session::builder()
        .llm(OpenAiProvider::ollama("qwen3:4b"))
        .system_prompt("You are a concise, helpful assistant.")
        // .tool(MyCustomTool)
        .build();

    // 2. Chat with the agent
    // Under the hood, this publishes a `user.message` to the EventBus,
    // runs the `ReactStrategy` loop, and awaits the final `assistant.message`.
    let reply = session.chat("What is the capital of France?").await?;
    println!("Agent: {reply}");

    Ok(())
}
```

## 📚 Documentation & Examples

For a deep dive into the framework's mechanics — from low-level bus routing, to SQLite DAG persistence, OpenTelemetry exports, and Docker-bound landlocked execution environments — refer to the exhaustive [Developer Guide](docs/DEV-GUIDE.md) and [API Reference](docs/API.md).

You can also explore the `crates/` directory to see production-grade examples built natively with Eventage:
*   `example-basic-chat` — Multi-turn standard chat using the Session API.
*   `example-workflow` — Sequential PRD-writing pipeline with human review.
*   `example-multi-agent` — Orchestrator-and-Worker routing inside an `AgentSet`.
*   `example-clang-agent` — Advanced C-programming showcase proving complex OS-level sandboxing interactions.

