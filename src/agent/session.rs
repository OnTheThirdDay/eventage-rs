//! High-level `Session` API.
//!
//! A [`Session`] wraps a single agent and its bus, providing two modes:
//! - **Synchronous `chat()`**: Blocking request-response, ideal for REPLs.
//! - **Reactive `run()`**: Event-driven loop, ideal for concurrent tasks or streaming.
//!
//! See [`Session`] for detailed examples.

use super::core::Agent;
use super::builder::AgentBuilder;
use super::context::DefaultContextAssembler;
use super::error::AgentError;
use super::strategy::ReactStrategy;
use super::tool::{Tool, ToolRegistry};
use crate::bus::EventBus;
use crate::event::{kinds, Event};
use crate::llm::LlmProvider;
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
