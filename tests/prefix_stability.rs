//! Does our context stay a stable prefix as a turn grows?
//!
//! Prompt caches key on the longest identical leading run of tokens. A ReAct
//! turn is the ideal shape for that — every step appends a tool call and its
//! result and changes nothing before them — so a long turn should cache
//! almost everything after the first step.
//!
//! Observed behaviour was worse: about four thousand tokens cached and then
//! frozen while the context grew past twenty-six thousand. That could be the
//! provider, or it could be us rewriting something early in the message list.
//! These tests answer the half we own, and they are worth keeping regardless:
//! any change that breaks prefix stability silently multiplies the cost of
//! every session, and nothing else would catch it.

use eventage::agent::context::{AssemblyContext, ContextAssembler, DefaultContextAssembler};
use eventage::agent::{SummarizingContextAssembler, ToolResultClearingAssembler};
use eventage::event::{kinds, Event};
use eventage::llm::mock::MockLlmProvider;
use eventage::llm::types::ChatMessage;
use serde_json::json;
use std::sync::Arc;

/// One ReAct step: the model calls a tool, the tool answers.
fn step(index: usize, result_bytes: usize) -> Vec<Event> {
    vec![
        Event::new(
            kinds::ASSISTANT_MESSAGE,
            json!({
                "content": null,
                "tool_calls": [{
                    "id": format!("call{index}"),
                    "type": "function",
                    "function": { "name": "read_file", "arguments": format!("{{\"path\":\"f{index}.rs\"}}") }
                }]
            }),
        ),
        Event::new(
            kinds::TOOL_RESULT,
            json!({
                "tool_call_id": format!("call{index}"),
                "name": "read_file",
                "result": { "content": "x".repeat(result_bytes) }
            }),
        ),
    ]
}

/// How many leading messages two assemblies agree on.
fn shared_prefix(a: &[ChatMessage], b: &[ChatMessage]) -> usize {
    a.iter()
        .zip(b.iter())
        // Compared as serialised, which is what the provider actually sees
        // and therefore what a cache keys on.
        .take_while(|(x, y)| serde_json::to_string(x).ok() == serde_json::to_string(y).ok())
        .count()
}

#[tokio::test]
async fn a_growing_turn_only_ever_appends() {
    // The base case. If this drifts, nothing above it can cache.
    let assembler = DefaultContextAssembler::new("You are a coding agent.");
    let mut events = vec![Event::new(kinds::USER_MESSAGE, json!({ "text": "go" }))];

    let mut previous = assembler.assemble(&AssemblyContext::new(&events)).await;
    for i in 0..12 {
        events.extend(step(i, 400));
        let current = assembler.assemble(&AssemblyContext::new(&events)).await;

        assert_eq!(
            shared_prefix(&previous, &current),
            previous.len(),
            "step {i} rewrote part of the context instead of appending to it"
        );
        assert!(current.len() > previous.len(), "step {i} added nothing");
        previous = current;
    }
}

#[tokio::test]
async fn clearing_rewrites_the_prefix_exactly_once_per_result() {
    // Clearing is a deliberate exception: reclaiming budget means editing
    // messages the model has already seen, which invalidates the cache from
    // that point. The ratchet is what keeps it from happening repeatedly —
    // once cleared, a result stays cleared, so the prefix settles again.
    let inner = Arc::new(DefaultContextAssembler::new("You are a coding agent."));
    let assembler = ToolResultClearingAssembler::new(inner, 2_000)
        .with_keep_recent(1)
        .with_min_clear_bytes(100);

    let mut events = vec![Event::new(kinds::USER_MESSAGE, json!({ "text": "go" }))];
    let mut previous = assembler.assemble(&AssemblyContext::new(&events)).await;
    let mut rewrites = 0;

    for i in 0..10 {
        events.extend(step(i, 3_000));
        let current = assembler.assemble(&AssemblyContext::new(&events)).await;
        if shared_prefix(&previous, &current) != previous.len() {
            rewrites += 1;
        }
        previous = current;
    }

    // Bounded by the number of results, not unbounded churn.
    assert!(
        rewrites <= 10,
        "clearing rewrote the prefix {rewrites} times; it should settle"
    );

    // And once it has cleared, re-assembling the same events changes nothing.
    let again = assembler.assemble(&AssemblyContext::new(&events)).await;
    assert_eq!(
        shared_prefix(&previous, &again),
        previous.len(),
        "a repeat assembly of identical events must be identical"
    );
}

#[tokio::test]
async fn the_full_stack_is_stable_while_under_budget() {
    // What a real session runs: default → clearing → summarizing. Below the
    // compaction threshold this must be pure append, or every step of every
    // turn pays full price.
    let base = Arc::new(DefaultContextAssembler::new("You are a coding agent."));
    let clearing = Arc::new(ToolResultClearingAssembler::new(base, 200_000));
    let assembler = SummarizingContextAssembler::new(
        clearing,
        Arc::new(MockLlmProvider::with_texts(["a summary"])),
        200_000,
        "prefix-test",
    );

    let mut events = vec![Event::new(kinds::USER_MESSAGE, json!({ "text": "go" }))];
    let mut previous = assembler.assemble(&AssemblyContext::new(&events)).await;

    for i in 0..15 {
        events.extend(step(i, 500));
        let current = assembler.assemble(&AssemblyContext::new(&events)).await;
        assert_eq!(
            shared_prefix(&previous, &current),
            previous.len(),
            "step {i} broke prefix stability in the assembled stack"
        );
        previous = current;
    }
}

#[tokio::test]
async fn the_system_prefix_is_byte_identical_across_steps() {
    // The cacheable head is the system block. If anything in it varies —
    // a timestamp, a set iterated in a different order, a regenerated map —
    // then nothing caches at all, however stable the conversation is.
    let assembler = DefaultContextAssembler::new("You are a coding agent.");
    let mut events = vec![Event::new(kinds::USER_MESSAGE, json!({ "text": "go" }))];

    let first = assembler.assemble(&AssemblyContext::new(&events)).await;
    let head = first[0].content.clone();

    for i in 0..5 {
        events.extend(step(i, 200));
        let current = assembler.assemble(&AssemblyContext::new(&events)).await;
        assert_eq!(
            current[0].content, head,
            "the system block changed at step {i}"
        );
    }
}
