//! High-level `Session` API.
//!
//! A [`Session`] wraps a single agent and its bus, providing two modes:
//! - **Synchronous `chat()`**: Blocking request-response, ideal for REPLs.
//! - **Reactive `run()`**: Event-driven loop, ideal for concurrent tasks or streaming.
//!
//! See [`Session`] for detailed examples.
//!
//! ```rust,no_run
//! use eventage_provided_impl::Session;
//! use eventage_core::{kinds, Event, EventBus};
//! use eventage_llm::OpenAiProvider;
//! use serde_json::json;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let session = Session::builder()
//!     .llm(OpenAiProvider::ollama("qwen3:4b"))
//!     .system_prompt("You are a helpful assistant.")
//!     .build();
//!
//! let bus = session.bus().clone();
//!
//! // Subscribe to responses *before* spawning the agent so no event is missed.
//! let mut reply_rx = bus.subscribe();
//!
//! // Reactive agent — cycles automatically on every user.message.
//! tokio::spawn(async move { session.run().await });
//!
//! // Publish user input from any task as a plain bus event.
//! bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "Hello!"}))).await?;
//!
//! // Receive the assistant response from the bus.
//! while let Some(event) = reply_rx.recv().await {
//!     if event.kind == kinds::ASSISTANT_MESSAGE {
//!         println!("{}", event.payload["content"].as_str().unwrap_or(""));
//!         break;
//!     }
//! }
//! # Ok(()) }
//! ```

use crate::context::DefaultContextAssembler;
use crate::strategy_impl::ReactStrategy;
use eventage_agent::{
    agent::Agent,
    builder::AgentBuilder,
    error::AgentError,
    tool::{Tool, ToolRegistry},
};
use eventage_core::{kinds, Event, EventBus};
use eventage_llm::LlmProvider;
use serde_json::json;
use std::sync::Arc;

// ── SessionBuilder ────────────────────────────────────────────────────────────

/// Builder for [`Session`].
#[derive(Default)]
pub struct SessionBuilder {
    llm: Option<Arc<dyn LlmProvider>>,
    system_prompt: Option<String>,
    tools: Vec<Arc<dyn Tool>>,
    max_steps: Option<usize>,
    max_concurrent_tools: Option<usize>,
    context_max_events: Option<usize>,
}

impl SessionBuilder {
    /// Sets the LLM provider.
    pub fn llm(mut self, llm: impl LlmProvider + 'static) -> Self {
        self.llm = Some(Arc::new(llm));
        self
    }

    /// Sets a pre-boxed LLM provider.
    pub fn llm_arc(mut self, llm: Arc<dyn LlmProvider>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Sets the system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Registers a tool.
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    /// Sets the maximum ReAct loop iterations.
    pub fn max_steps(mut self, n: usize) -> Self {
        self.max_steps = Some(n);
        self
    }

    /// Sets the maximum tools executing parallelly per ReAct step.
    pub fn max_concurrent_tools(mut self, n: usize) -> Self {
        self.max_concurrent_tools = Some(n);
        self
    }

    /// Bounds the context window to the most recent `n` events.
    pub fn context_window(mut self, max_events: usize) -> Self {
        self.context_max_events = Some(max_events);
        self
    }

    /// Builds the [`Session`].
    ///
    /// # Panics
    /// Panics if no LLM provider was set.
    pub fn build(self) -> Session {
        let bus = EventBus::new();
        let llm = self
            .llm
            .expect("Session::builder: llm provider is required");

        let strategy = ReactStrategy {
            max_steps: self.max_steps.unwrap_or(20),
            max_concurrent_tools: self.max_concurrent_tools.unwrap_or(4),
        };

        // Build the context assembler using DefaultContextAssembler.
        let mut base_assembler = if let Some(prompt) = self.system_prompt {
            DefaultContextAssembler::new(prompt)
        } else {
            DefaultContextAssembler::without_system_prompt()
        };
        if let Some(max) = self.context_max_events {
            base_assembler = base_assembler.with_max_events(max);
        }

        let mut builder = AgentBuilder::new()
            .bus(bus.clone())
            .llm_arc(llm)
            .context(base_assembler)
            .strategy(strategy);

        for tool in self.tools {
            builder = builder.tool_arc(tool);
        }

        let agent = builder.build();
        Session { agent, bus }
    }
}

// ── Session ───────────────────────────────────────────────────────────────────

/// A stateful, single-agent conversation session.
///
/// Modes:
/// - **[`chat()`][Self::chat]**: Synchronous request-response for REPLs.
/// - **[`run()`][Self::run]**: Reactive event-driven loop for GUIs and streams.
///
/// Build via [`Session::builder`].
pub struct Session {
    agent: Agent,
    bus: EventBus,
}

impl Session {
    /// Return a new [`SessionBuilder`].
    pub fn builder() -> SessionBuilder {
        SessionBuilder::default()
    }

    /// Send a user message, run one full reasoning cycle, and return the
    /// assistant's final text response.
    ///
    /// This is the simplest way to use a session: the call blocks until the
    /// full ReAct loop completes (including all tool calls) and returns the
    /// assistant's reply. Conversation history is maintained across calls via
    /// the underlying event bus.
    ///
    /// For event-driven scenarios where input can arrive from multiple sources
    /// or concurrently, use [`run()`][Self::run] instead.
    pub async fn chat(&mut self, message: &str) -> Result<String, AgentError> {
        self.bus
            .publish(Event::new(kinds::USER_MESSAGE, json!({ "text": message })))
            .await?;

        self.agent.cycle().await?;

        let log = self.bus.log().await;
        let response = log
            .iter()
            .rev()
            .find(|e| e.kind == kinds::ASSISTANT_MESSAGE)
            .and_then(|e| e.payload.get("content").and_then(|v| v.as_str()))
            .unwrap_or("")
            .to_string();

        Ok(response)
    }

    /// Run the agent reactively, cycling on every `user.message` or
    /// `system.heartbeat` event published to the bus.
    ///
    /// Returns when the bus closes (all senders are dropped). Blocks the
    /// current task; spawn it with `tokio::spawn` to run concurrently.
    ///
    /// User input is published to the bus from any task via
    /// `session.bus().publish(Event::new(kinds::USER_MESSAGE, ...))`.
    /// Subscribe to `assistant.message` events (or `agent.cycle.end`) to
    /// receive responses asynchronously.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use eventage_provided_impl::Session;
    /// use eventage_core::{kinds, Event};
    /// use eventage_llm::OpenAiProvider;
    /// use serde_json::json;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let session = Session::builder()
    ///     .llm(OpenAiProvider::ollama("qwen3:4b"))
    ///     .build();
    ///
    /// let bus = session.bus().clone();
    /// let mut reply_rx = bus.subscribe();
    ///
    /// tokio::spawn(async move { session.run().await });
    ///
    /// // Input from any source — no `chat()` call needed.
    /// bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "Hi!"}))).await?;
    ///
    /// while let Some(ev) = reply_rx.recv().await {
    ///     if ev.kind == kinds::ASSISTANT_MESSAGE {
    ///         println!("{}", ev.payload["content"].as_str().unwrap_or(""));
    ///         break;
    ///     }
    /// }
    /// # Ok(()) }
    /// ```
    pub async fn run(self) -> Result<(), AgentError> {
        self.agent.run().await
    }

    /// Register a new tool at runtime.
    pub fn add_tool(&self, tool: impl Tool + 'static) {
        self.agent.tools().register(Arc::new(tool));
    }

    /// Remove a registered tool by name.
    pub fn remove_tool(&self, name: &str) {
        self.agent.tools().remove(name);
    }

    /// Live handle to the agent's tool registry.
    pub fn tools(&self) -> ToolRegistry {
        self.agent.tools()
    }

    /// Access the underlying [`EventBus`].
    pub fn bus(&self) -> &EventBus {
        &self.bus
    }

    /// Access the underlying [`Agent`].
    pub fn agent(&self) -> &Agent {
        &self.agent
    }

    /// Return the conversation history as `(role, content)` pairs.
    pub async fn history(&self) -> Vec<(String, String)> {
        let log = self.bus.log().await;
        let mut turns = Vec::new();
        for event in &log {
            match event.kind.as_str() {
                kinds::USER_MESSAGE => {
                    if let Some(text) = event.payload.get("text").and_then(|v| v.as_str()) {
                        turns.push(("user".to_string(), text.to_string()));
                    }
                }
                kinds::ASSISTANT_MESSAGE => {
                    if let Some(text) = event.payload.get("content").and_then(|v| v.as_str()) {
                        if !text.is_empty() {
                            turns.push(("assistant".to_string(), text.to_string()));
                        }
                    }
                }
                kinds::TOOL_RESULT => {
                    let name = event
                        .payload
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("tool");
                    let result = event
                        .payload
                        .get("result")
                        .map(|v| v.to_string())
                        .unwrap_or_default();
                    turns.push((format!("tool:{name}"), result));
                }
                _ => {}
            }
        }
        turns
    }
}
