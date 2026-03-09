//! Built-in [`ToolSelector`] implementations.

use std::sync::Arc;

use async_trait::async_trait;
use eventage_agent::{Tool, ToolSelector};
use eventage_llm::ChatMessage;

/// Selects tools whose name contains any of the provided keywords.
///
/// Useful for domain-prefixed tool selection (e.g., `"search_web"` vs `"write_file"`).
///
/// # Example
///
/// ```rust,no_run
/// use eventage_agent::AgentBuilder;
/// use eventage_provided_impl::KeywordToolSelector;
/// use eventage_llm::MockLlmProvider;
///
/// // Only expose tools whose name contains "search" or "fetch".
/// let agent = AgentBuilder::new()
///     .llm(MockLlmProvider::with_texts(Vec::<&str>::new()))
///     .tool_selector(KeywordToolSelector::new(vec!["search", "fetch"]))
///     .strategy(eventage_provided_impl::ReactStrategy::default())
///     .build();
/// ```
pub struct KeywordToolSelector {
    keywords: Vec<String>,
}

impl KeywordToolSelector {
    pub fn new(keywords: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            keywords: keywords.into_iter().map(|k| k.into()).collect(),
        }
    }
}

#[async_trait]
impl ToolSelector for KeywordToolSelector {
    async fn select(
        &self,
        tools: &[Arc<dyn Tool>],
        _messages: &[ChatMessage],
    ) -> Vec<Arc<dyn Tool>> {
        tools
            .iter()
            .filter(|t| {
                let name = t.definition().function.name;
                self.keywords.iter().any(|kw| name.contains(kw.as_str()))
            })
            .cloned()
            .collect()
    }
}
