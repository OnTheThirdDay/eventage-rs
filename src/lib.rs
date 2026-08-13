//! **eventage** — event-driven agent framework for Rust.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use eventage::{EventBus, agent::{Session, SessionBuilder}};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let bus = EventBus::new();
//! // ... configure session with an LLM provider and run
//! # Ok(())
//! # }
//! ```
//!
//! # Feature Flags
//!
//! | Feature | Description |
//! |---|---|
//! | `observability` | [`observability`] module — event exporters and bus observer |
//! | `scheduler` | [`scheduler`] module — heartbeat/tick scheduler |
//! | `sqlite` | [`sqlite`] module — SQLite event store and exporter |
//! | `sqlite-bundled` | Implies `sqlite`, bundles SQLite |
//! | `replay` | [`replay`] module — live replay HTTP server |
//! | `mcp` | [`mcp`] module — Model Context Protocol client |
//! | `sandbox` | [`sandbox`] module — sandboxed execution (Docker, Landlock, Unsandboxed) |
//! | `sandbox-wasm` | Implies `sandbox`, adds WASM executor |
//! | `opentelemetry` | OpenTelemetry exporter (requires `observability`) |
//! | `full` | Enables all of the above |

// ── Always-on modules ─────────────────────────────────────────────────────────

pub mod bus;
pub mod event;
pub mod error;
pub mod llm;
pub mod agent;
pub mod bridge;
pub mod eviction;
pub mod plugin;

// ── Feature-gated modules ─────────────────────────────────────────────────────

#[cfg(feature = "observability")]
pub mod observability;

#[cfg(feature = "scheduler")]
pub mod scheduler;

#[cfg(feature = "sqlite")]
pub mod sqlite;

#[cfg(feature = "replay")]
pub mod replay;

#[cfg(feature = "mcp")]
pub mod mcp;

#[cfg(feature = "sandbox")]
pub mod sandbox;

// ── Flat re-exports ───────────────────────────────────────────────────────────

pub use bus::{
    BranchData, BranchEvictionStrategy, BranchId, BusConfig, BusReceiver, EventBus, PruneStrategy,
    secrets_masking_transform,
};
pub use error::{BusError, CoreError};
pub use event::{Event, EventId, kinds, meta_keys};

pub use agent::{
    Agent, AgentBuilder, AgentContext, AgentError, AgentSet, AssemblyContext, BranchScorer,
    BudgetScope, ContextAssembler, CycleHook, DEFAULT_MAX_CONCURRENT_TOOLS,
    DEFAULT_MAX_REACT_STEPS, DEFAULT_MAX_TOOL_RESULT_CHARS, DEFAULT_TOOL_TIMEOUT_SECS,
    DefaultContextAssembler, DynamicContextAssembler, DynamicHookChain, DynamicWorkerHandle,
    EventWorker, ExecutionStrategy, FnScorer, HookAction, HookContext, KeywordToolSelector,
    LlmJudgeScorer, NegativeAwareContextAssembler, PermissionPolicyHook, PermissionVerdict,
    ReactStrategy, Session, SessionBuilder, SingleShotStrategy, SpeculationCandidate,
    SpeculationOutcome, SummarizingContextAssembler, TokenBudgetHook, Tool, ToolExecOptions,
    ToolRegistry, ToolResultClearingAssembler, ToolSelector, WorkerError, WorkerSet, best_of_n,
    detect_stuck, truncate_middle, StuckAnalysis, StuckKind,
};

pub use bridge::{BusBridge, BRIDGE_HOPS_KEY};
pub use eviction::{EpitaphStore, EpitaphStrategy};

#[cfg(feature = "observability")]
pub use observability::{BusObserver, JsonlExporter};

pub use llm::{
    AnthropicProvider, ChatMessage, LlmError, LlmProvider, LlmResponse, OpenAiProvider,
    OpenAiResponsesProvider, RateLimitedProvider, RetryProvider, ToolCall, ToolDefinition,
};
pub use plugin::{Plugin, PluginError, PluginHost};
