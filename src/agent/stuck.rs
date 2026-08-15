//! Stuck / loop detection for the agent reasoning cycle.
//!
//! Before each cycle, calling [`detect_stuck`] on the recent event log lets
//! the agent detect when it has fallen into a repetitive pattern. Callers
//! should publish an [`AGENT_STUCK`](crate::event::kinds::AGENT_STUCK) hint
//! event so the LLM can self-correct on the next cycle.

use crate::event::{kinds, Event};

/// The category of loop or stuck pattern that was detected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StuckKind {
    /// The same tool name and arguments were proposed ≥ 3 times in a row.
    RepeatingAction,
    /// The same error message was returned by consecutive tool results ≥ 3 times.
    RepeatingError,
    /// ≥ 4 consecutive assistant messages contained no tool calls.
    Monologue,
    /// ≥ 2 context-window / token-limit errors appeared in the recent event window.
    ContextWindowLoop,
}

/// Details of a detected stuck pattern.
#[derive(Debug, Clone)]
pub struct StuckAnalysis {
    pub kind: StuckKind,
    /// How many times the pattern repeated.
    pub repeat_count: usize,
}

/// Inspect the last `window` events and return a [`StuckAnalysis`] if the
/// agent appears to be looping or stuck. Returns `None` when the cycle looks healthy.
pub fn detect_stuck(events: &[Event], window: usize) -> Option<StuckAnalysis> {
    let tail: &[Event] = if events.len() > window {
        &events[events.len() - window..]
    } else {
        events
    };

    // ── ContextWindowLoop ────────────────────────────────────────────────────
    // Two or more tool results with a context / token-limit error message.
    let context_errors = tail
        .iter()
        .filter(|e| {
            e.kind == kinds::TOOL_RESULT
                && e.payload
                    .get("error")
                    .and_then(|v| v.as_str())
                    .map(|msg| {
                        let lower = msg.to_lowercase();
                        lower.contains("context length")
                            || lower.contains("context window")
                            || lower.contains("token limit")
                            || lower.contains("maximum context")
                    })
                    .unwrap_or(false)
        })
        .count();
    if context_errors >= 2 {
        return Some(StuckAnalysis {
            kind: StuckKind::ContextWindowLoop,
            repeat_count: context_errors,
        });
    }

    // ── RepeatingAction ──────────────────────────────────────────────────────
    // Last ≥ 3 tool-call proposals share the same name and arguments string.
    let proposals: Vec<(&str, &str)> = tail
        .iter()
        .filter(|e| e.kind == kinds::TOOL_CALL_PROPOSED)
        .filter_map(|e| {
            let name = e.payload.get("name").and_then(|v| v.as_str())?;
            let args = e.payload.get("arguments").and_then(|v| v.as_str())?;
            Some((name, args))
        })
        .collect();
    if proposals.len() >= 3 {
        let last = *proposals.last().unwrap();
        let run = proposals.iter().rev().take_while(|&&p| p == last).count();
        if run >= 3 {
            return Some(StuckAnalysis {
                kind: StuckKind::RepeatingAction,
                repeat_count: run,
            });
        }
    }

    // ── RepeatingError ───────────────────────────────────────────────────────
    // Last ≥ 3 tool results all carry the same error message.
    let result_errors: Vec<&str> = tail
        .iter()
        .filter(|e| e.kind == kinds::TOOL_RESULT)
        .filter_map(|e| e.payload.get("error").and_then(|v| v.as_str()))
        .collect();
    if result_errors.len() >= 3 {
        let last = *result_errors.last().unwrap();
        let run = result_errors
            .iter()
            .rev()
            .take_while(|&&e| e == last)
            .count();
        if run >= 3 {
            return Some(StuckAnalysis {
                kind: StuckKind::RepeatingError,
                repeat_count: run,
            });
        }
    }

    // ── Monologue ────────────────────────────────────────────────────────────
    // Last ≥ 4 events are all assistant messages with no tool calls.
    if tail.len() >= 4
        && tail.iter().rev().take(4).all(|e| {
            e.kind == kinds::ASSISTANT_MESSAGE
                && e.payload
                    .get("tool_calls")
                    .and_then(|v| v.as_array())
                    .map(|a| a.is_empty())
                    .unwrap_or(true)
        })
    {
        return Some(StuckAnalysis {
            kind: StuckKind::Monologue,
            repeat_count: 4,
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::Event;
    use serde_json::json;

    fn tool_call(name: &str, args: &str) -> Event {
        Event::new(
            kinds::TOOL_CALL_PROPOSED,
            json!({ "name": name, "arguments": args }),
        )
    }

    fn tool_error(msg: &str) -> Event {
        Event::new(kinds::TOOL_RESULT, json!({ "error": msg }))
    }

    fn assistant_msg() -> Event {
        Event::new(
            kinds::ASSISTANT_MESSAGE,
            json!({ "content": "thinking..." }),
        )
    }

    #[test]
    fn detects_repeating_action() {
        let events: Vec<Event> = (0..4).map(|_| tool_call("search", "{}")).collect();
        let analysis = detect_stuck(&events, 10).unwrap();
        assert_eq!(analysis.kind, StuckKind::RepeatingAction);
        assert_eq!(analysis.repeat_count, 4);
    }

    #[test]
    fn detects_repeating_error() {
        let events: Vec<Event> = (0..3).map(|_| tool_error("permission denied")).collect();
        let analysis = detect_stuck(&events, 10).unwrap();
        assert_eq!(analysis.kind, StuckKind::RepeatingError);
    }

    #[test]
    fn detects_monologue() {
        let events: Vec<Event> = (0..4).map(|_| assistant_msg()).collect();
        let analysis = detect_stuck(&events, 10).unwrap();
        assert_eq!(analysis.kind, StuckKind::Monologue);
    }

    #[test]
    fn detects_context_window_loop() {
        let events = vec![
            tool_error("context length exceeded — please shorten"),
            tool_error("context length exceeded — please shorten"),
        ];
        let analysis = detect_stuck(&events, 10).unwrap();
        assert_eq!(analysis.kind, StuckKind::ContextWindowLoop);
    }

    #[test]
    fn no_false_positives_on_healthy_cycle() {
        let events = vec![
            assistant_msg(),
            tool_call("read_file", r#"{"path":"a.txt"}"#),
            Event::new(kinds::TOOL_RESULT, json!({ "result": "ok" })),
            assistant_msg(),
            tool_call("write_file", r#"{"path":"b.txt"}"#),
            Event::new(kinds::TOOL_RESULT, json!({ "result": "ok" })),
        ];
        assert!(detect_stuck(&events, 10).is_none());
    }
}
