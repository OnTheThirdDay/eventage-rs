//! Tool registry and selection mechanisms.
//!
//! Add/remove tools at runtime via [`ToolRegistry`]. Filter what the LLM
//! sees on each step via a [`ToolSelector`].

use crate::error::AgentError;
use async_trait::async_trait;
use eventage_llm::types::{ChatMessage, ToolDefinition};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// ── Tool trait ────────────────────────────────────────────────────────────────

/// A callable tool that the LLM can request.
#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;
    async fn execute(&self, args: Value) -> Result<Value, AgentError>;
    /// Marks the tool as terminal, exiting the react loop immediately after use.
    fn is_terminal(&self) -> bool {
        false
    }
}

// ── ToolRegistry ──────────────────────────────────────────────────────────────

/// A thread-safe, dynamically mutable registry of tools.
///
/// Changes reflect immediately across all clones.
#[derive(Clone)]
pub struct ToolRegistry {
    inner: Arc<RwLock<HashMap<String, Arc<dyn Tool>>>>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    // ── Write operations ──────────────────────────────────────────────────────

    /// Registers a pre-boxed tool, overwriting if existing.
    pub fn register(&self, tool: Arc<dyn Tool>) {
        self.inner
            .write()
            .unwrap()
            .insert(tool.definition().function.name.clone(), tool);
    }

    /// Adds a tool dynamically (preferred method).
    pub fn add_tool(&self, tool: impl Tool + 'static) {
        self.register(Arc::new(tool));
    }

    /// Remove a tool by name. Returns `true` if the tool existed.
    pub fn remove(&self, name: &str) -> bool {
        self.inner.write().unwrap().remove(name).is_some()
    }

    /// Remove all registered tools.
    pub fn clear(&self) {
        self.inner.write().unwrap().clear();
    }

    // ── Read operations ───────────────────────────────────────────────────────

    /// Snapshot of all registered tool names.
    pub fn names(&self) -> Vec<String> {
        self.inner.read().unwrap().keys().cloned().collect()
    }

    /// `true` if no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.inner.read().unwrap().is_empty()
    }

    /// Snapshot of all tool definitions (for passing to the LLM).
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.inner
            .read()
            .unwrap()
            .values()
            .map(|t| t.definition())
            .collect()
    }

    /// Snapshot of all tools as `Arc` handles (for passing to a [`ToolSelector`]).
    pub fn all_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.inner.read().unwrap().values().cloned().collect()
    }

    /// Look up a single tool by name. Returns `None` if not registered.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.inner.read().unwrap().get(name).cloned()
    }
}

// ── ToolSelector trait ────────────────────────────────────────────────────────

/// Filters the tool definitions exposed to the LLM per cycle.
///
/// Allows dynamic, context-aware tool routing without altering the main registry.
#[async_trait]
pub trait ToolSelector: Send + Sync {
    /// Returns the subset of `tools` to expose to the LLM.
    async fn select(&self, tools: &[Arc<dyn Tool>], messages: &[ChatMessage])
        -> Vec<Arc<dyn Tool>>;
}
