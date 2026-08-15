//! What a tool result leaves behind.
//!
//! The README promised that oversized tool output is shortened for the model
//! while "the full data stays in the event log", and the context-clearing
//! assembler is justified as lossless on exactly that basis. It was not true:
//! the value was capped before the event was built, so the log held the short
//! version and the original was gone.

use async_trait::async_trait;
use eventage::agent::error::AgentError;
use eventage::agent::tool::Tool;
use eventage::event::kinds;
use eventage::llm::types::ToolDefinition;
use eventage::llm::types::{FunctionCall, LlmResponse, ToolCall};
use eventage::{AgentBuilder, Event, EventBus, ReactStrategy};
use serde_json::{json, Value};

/// A tool whose output is far past any sane context cap.
struct Firehose;

#[async_trait]
impl Tool for Firehose {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function("firehose", "returns a lot", json!({ "type": "object" }))
    }
    async fn execute(&self, _args: Value) -> Result<Value, AgentError> {
        Ok(json!({ "content": format!("{}END", "x".repeat(80_000)) }))
    }
}

#[tokio::test]
async fn the_event_log_keeps_the_whole_result_even_when_the_model_sees_less() {
    let bus = EventBus::new();
    let agent = AgentBuilder::new()
        .agent_id("fidelity")
        .bus(bus.clone())
        .llm(eventage::llm::mock::MockLlmProvider::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "c1".into(),
                    kind: "function".into(),
                    function: FunctionCall {
                        name: "firehose".into(),
                        arguments: "{}".into(),
                    },
                    extra_content: None,
                }],
                finish_reason: "tool_calls".into(),
                ..Default::default()
            },
        ]))
        .strategy(ReactStrategy {
            max_steps: 1,
            max_tool_result_chars: Some(2_000),
            finalize_on_max_steps: false,
            ..Default::default()
        })
        .tool(Firehose)
        .build();

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({ "text": "go" })))
        .await
        .unwrap();
    let _ = agent.cycle().await;

    let log = bus.log().await;
    let result = log
        .iter()
        .find(|e| e.kind == kinds::TOOL_RESULT)
        .expect("the tool should have produced a result");

    let full = result.payload["result"].to_string();
    assert!(
        full.len() > 70_000,
        "the log kept only {} bytes — the original is gone",
        full.len()
    );
    assert!(full.contains("END"), "the tail was cut from the record");

    // And the model's copy is still bounded.
    let for_context = result.payload["result_for_context"].to_string();
    assert!(
        for_context.len() < 5_000,
        "the context copy was not capped: {} bytes",
        for_context.len()
    );
}
