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
//! The modules named here appear in the sidebar when their feature is on.
//! Deliberately not intra-doc links: a link to `observability` is only
//! resolvable when that feature is enabled, so linking them made `cargo doc`
//! fail in every configuration except the one CI happened to run
//! (`--all-features`).
//!
//! | Feature | Description |
//! |---|---|
//! | `observability` | `observability` module — event exporters and bus observer |
//! | `scheduler` | `scheduler` module — heartbeat/tick scheduler |
//! | `sqlite` | `sqlite` module — SQLite event store and exporter |
//! | `sqlite-bundled` | Implies `sqlite`, bundles SQLite |
//! | `replay` | `replay` module — live replay HTTP server |
//! | `mcp` | `mcp` module — Model Context Protocol client |
//! | `sandbox` | `sandbox` module — sandboxed execution (Docker, Landlock, Unsandboxed) |
//! | `sandbox-wasm` | Implies `sandbox`, adds WASM executor |
//! | `opentelemetry` | OpenTelemetry exporter (requires `observability`) |
//! | `full` | Enables all of the above |

// ── Always-on modules ─────────────────────────────────────────────────────────

pub mod agent;
pub mod bridge;
pub mod bus;
pub mod component;
pub mod distributed;
pub mod error;
pub mod event;
pub mod eviction;
pub mod llm;
pub mod plugin;
pub mod schema;

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
    secrets_masking_transform, BranchData, BranchEvictionStrategy, BranchId, BusConfig,
    BusReceiver, EventBus, PruneStrategy,
};
pub use error::{BusError, CoreError};
pub use event::{kinds, meta_keys, Event, EventId};

pub use agent::web::{WebFetchTool, WebSearchTool};
pub use agent::{
    detect_stuck, truncate_middle, Agent, AgentBuilder, AgentContext, AgentError, AgentSet,
    AssemblyContext, BudgetScope, ContextAssembler, CycleHook, DefaultContextAssembler,
    DynamicContextAssembler, DynamicHookChain, DynamicWorkerHandle, EventWorker, ExecutionStrategy,
    HookAction, HookContext, KeywordToolSelector, NegativeAwareContextAssembler,
    PermissionPolicyHook, PermissionVerdict, ReactStrategy, Session, SessionBuilder,
    SingleShotStrategy, StuckAnalysis, StuckKind, SummarizingContextAssembler, TokenBudgetHook,
    Tool, ToolExecOptions, ToolRegistry, ToolResultClearingAssembler, ToolSelector, WorkerError,
    WorkerSet, DEFAULT_MAX_CONCURRENT_TOOLS, DEFAULT_MAX_REACT_STEPS,
    DEFAULT_MAX_TOOL_RESULT_CHARS, DEFAULT_TOOL_TIMEOUT_SECS,
};

pub use bridge::{BusBridge, BRIDGE_HOPS_KEY};
pub use component::{
    Component, ComponentContext, ComponentError, ComponentHost, ComponentState, ServiceRegistry,
};
pub use distributed::{BusTransport, DistributedBus, TcpTransport};
pub use eviction::{EpitaphStore, EpitaphStrategy};

#[cfg(feature = "observability")]
pub use observability::{BusObserver, JsonlExporter};

pub use llm::{
    AnthropicProvider, ChatMessage, ContentPart, ImageSource, LlmError, LlmProvider, LlmResponse,
    OpenAiProvider, OpenAiResponsesProvider, QwenProvider, RateLimitedProvider, RetryProvider,
    StructuredExt, ToolCall, ToolDefinition,
};
pub use plugin::{Plugin, PluginError, PluginHost};
