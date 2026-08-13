//! Shared token-count heuristics for context-management components.

use crate::llm::types::ChatMessage;

/// Approximate token count for a string.
///
/// Uses the higher of two heuristics so both prose (word-heavy) and
/// JSON/code (character-heavy with short tokens) are reasonably bounded:
/// - words × 1.3  (good for natural language)
/// - chars ÷ 4    (good for dense JSON / code)
pub fn estimate_tokens(s: &str) -> usize {
    let by_words = ((s.split_whitespace().count() as f64) * 1.3).ceil() as usize + 4;
    let by_chars = s.len() / 4 + 4;
    by_words.max(by_chars)
}

/// Approximate token count for a full message list.
pub fn messages_token_count(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .map(|m| {
            let content = m.content.as_deref().map(estimate_tokens).unwrap_or(0);
            let tool_calls: usize = m
                .tool_calls
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .map(|tc| {
                    estimate_tokens(&tc.function.arguments) + estimate_tokens(&tc.function.name) + 4
                })
                .sum();
            content + tool_calls + 4
        })
        .sum()
}
