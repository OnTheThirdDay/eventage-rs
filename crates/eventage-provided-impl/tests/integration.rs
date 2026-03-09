use async_trait::async_trait;
use eventage_core::{kinds, meta_keys, Event, EventBus};
use eventage_llm::{
    types::{ChatMessage, FunctionCall, LlmResponse, ToolCall, ToolDefinition},
    MockLlmProvider,
};
use eventage_provided_impl::{
    hook::{CycleHook, HookAction, HookContext},
    AgentBuilder, AgentError, AssemblyContext, ContextAssembler, DynamicContextAssembler,
    DynamicHookChain, KeywordToolSelector, ReactStrategy, Session, Tool,
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

// ── helpers ──────────────────────────────────────────────────────────────────

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("eventage=debug")
        .with_test_writer()
        .try_init();
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn basic_text_response() {
    init_tracing();
    let bus = EventBus::new();

    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::with_texts(["Hello from the agent!"]))
        .system_prompt("You are a helpful assistant.")
        .strategy(ReactStrategy::default())
        .build();

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "Hi!"})))
        .await
        .unwrap();

    agent.cycle().await.unwrap();

    let log = bus.log().await;
    let assistant_msg = log
        .iter()
        .find(|e| e.kind == kinds::ASSISTANT_MESSAGE)
        .expect("no assistant.message in log");

    assert_eq!(
        assistant_msg.payload["content"].as_str().unwrap(),
        "Hello from the agent!"
    );
}

#[tokio::test]
async fn tool_call_executes_and_loops() {
    init_tracing();

    // A simple tool that echoes its input.
    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "echo",
                "Echoes the input text",
                json!({
                    "type": "object",
                    "properties": {
                        "text": { "type": "string" }
                    },
                    "required": ["text"]
                }),
            )
        }

        async fn execute(&self, args: Value) -> Result<Value, AgentError> {
            let text = args["text"].as_str().unwrap_or("").to_string();
            Ok(json!({ "echoed": text }))
        }
    }

    // First LLM call requests the tool; second returns the final answer.
    let tool_call_response = LlmResponse {
        content: None,
        tool_calls: vec![ToolCall {
            id: "call_001".to_string(),
            kind: "function".to_string(),
            function: FunctionCall {
                name: "echo".to_string(),
                arguments: r#"{"text":"ping"}"#.to_string(),
            },
        }],
        finish_reason: "tool_calls".to_string(),
    };
    let final_response = LlmResponse {
        content: Some("The echo returned: ping".to_string()),
        tool_calls: vec![],
        finish_reason: "stop".to_string(),
    };

    let bus = EventBus::new();
    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::new(vec![
            tool_call_response,
            final_response,
        ]))
        .tool(EchoTool)
        .strategy(ReactStrategy::default())
        .build();

    bus.publish(Event::new(
        kinds::USER_MESSAGE,
        json!({"text": "Please echo ping"}),
    ))
    .await
    .unwrap();

    agent.cycle().await.unwrap();

    let log = bus.log().await;

    // tool.call.proposed must be present
    assert!(
        log.iter().any(|e| e.kind == kinds::TOOL_CALL_PROPOSED),
        "missing tool.call.proposed"
    );

    // tool.result must be present with the echoed value
    let tool_result = log
        .iter()
        .find(|e| e.kind == kinds::TOOL_RESULT)
        .expect("no tool.result in log");
    assert_eq!(tool_result.payload["result"]["echoed"], "ping");

    // Final assistant message
    let final_msg = log
        .iter()
        .filter(|e| e.kind == kinds::ASSISTANT_MESSAGE)
        .next_back()
        .expect("no final assistant.message");
    assert_eq!(
        final_msg.payload["content"].as_str().unwrap(),
        "The echo returned: ping"
    );
}

#[tokio::test]
async fn event_log_is_ordered() {
    init_tracing();
    let bus = EventBus::new();
    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::with_texts(["ok"]))
        .strategy(ReactStrategy::default())
        .build();

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "test"})))
        .await
        .unwrap();
    agent.cycle().await.unwrap();

    let log = bus.log().await;
    let kinds_vec: Vec<&str> = log.iter().map(|e| e.kind.as_str()).collect();

    // Verify ordering invariants
    let user_pos = kinds_vec.iter().position(|&k| k == "user.message").unwrap();
    let cycle_start = kinds_vec
        .iter()
        .position(|&k| k == "agent.cycle.start")
        .unwrap();
    let assistant_pos = kinds_vec
        .iter()
        .position(|&k| k == "assistant.message")
        .unwrap();
    let cycle_end = kinds_vec
        .iter()
        .position(|&k| k == "agent.cycle.end")
        .unwrap();

    assert!(user_pos < cycle_start);
    assert!(cycle_start < assistant_pos);
    assert!(assistant_pos < cycle_end);
}

// ── agent_id / trace / metadata tests ────────────────────────────────────────

#[tokio::test]
async fn cycle_events_carry_agent_id_and_trace_id() {
    init_tracing();
    let bus = EventBus::new();
    let agent = AgentBuilder::new()
        .agent_id("test-agent")
        .bus(bus.clone())
        .llm(MockLlmProvider::with_texts(["done"]))
        .strategy(ReactStrategy::default())
        .build();

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "go"})))
        .await
        .unwrap();
    agent.cycle().await.unwrap();

    let log = bus.log().await;

    let cycle_start = log
        .iter()
        .find(|e| e.kind == kinds::AGENT_CYCLE_START)
        .expect("missing agent.cycle.start");
    let cycle_end = log
        .iter()
        .find(|e| e.kind == kinds::AGENT_CYCLE_END)
        .expect("missing agent.cycle.end");

    // Both events should carry agent_id.
    assert_eq!(
        cycle_start.metadata[meta_keys::AGENT_ID].as_str().unwrap(),
        "test-agent"
    );
    assert_eq!(
        cycle_end.metadata[meta_keys::AGENT_ID].as_str().unwrap(),
        "test-agent"
    );

    // Both events should share the same trace_id.
    let trace_id = cycle_start.metadata[meta_keys::TRACE_ID]
        .as_str()
        .expect("missing trace_id on cycle_start");
    assert_eq!(
        cycle_end.metadata[meta_keys::TRACE_ID].as_str().unwrap(),
        trace_id
    );

    // cycle_end should carry elapsed_ms.
    assert!(
        cycle_end.metadata.contains_key(meta_keys::ELAPSED_MS),
        "missing elapsed_ms on cycle_end"
    );
}

#[tokio::test]
async fn agent_message_carries_agent_id_metadata() {
    // Verify that AGENT_MESSAGE events tagged with TO_AGENT_ID carry the metadata
    // correctly, and that an agent's cycle events are stamped with its agent_id.
    init_tracing();
    let bus = EventBus::new();

    // Publish an AGENT_MESSAGE addressed to "agent-a".
    let msg = Event::new(kinds::AGENT_MESSAGE, json!({"text": "hello a"}))
        .with_meta(meta_keys::TO_AGENT_ID, json!("agent-a"));
    bus.publish(msg).await.unwrap();

    // Agent A manually cycles: it should see the AGENT_MESSAGE and start a cycle.
    // Custom assembler: returns a prompt only when an AGENT_MESSAGE is present.
    struct AgentMsgAssembler;
    #[async_trait]
    impl ContextAssembler for AgentMsgAssembler {
        async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
            if context
                .events
                .iter()
                .any(|e| e.kind == kinds::AGENT_MESSAGE)
            {
                vec![ChatMessage::user("process this")]
            } else {
                vec![]
            }
        }
    }

    let agent_a = AgentBuilder::new()
        .agent_id("agent-a")
        .bus(bus.clone())
        .context(AgentMsgAssembler)
        .llm(MockLlmProvider::with_texts(["routing works"]))
        .strategy(ReactStrategy::default())
        .build();

    agent_a.cycle().await.unwrap();

    let log = bus.log().await;

    // The AGENT_MESSAGE must carry TO_AGENT_ID.
    let agent_msg = log.iter().find(|e| e.kind == kinds::AGENT_MESSAGE).unwrap();
    assert_eq!(
        agent_msg.metadata[meta_keys::TO_AGENT_ID].as_str().unwrap(),
        "agent-a"
    );

    // The cycle start emitted by agent-a must carry its agent_id.
    let cycle_start = log
        .iter()
        .find(|e| e.kind == kinds::AGENT_CYCLE_START)
        .expect("missing cycle.start");
    assert_eq!(
        cycle_start.metadata[meta_keys::AGENT_ID].as_str().unwrap(),
        "agent-a"
    );
}

#[tokio::test]
async fn multiple_agents_cycle_concurrently_on_shared_bus() {
    // Verify that two agents can run concurrent cycles on the same bus and that
    // each cycle event is stamped with the correct agent_id.
    init_tracing();
    let bus = EventBus::new();

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "go"})))
        .await
        .unwrap();

    let agent_a = std::sync::Arc::new(
        AgentBuilder::new()
            .agent_id("agent-a")
            .bus(bus.clone())
            .llm(MockLlmProvider::with_texts(["done-a"]))
            .strategy(ReactStrategy::default())
            .build(),
    );
    let agent_b = std::sync::Arc::new(
        AgentBuilder::new()
            .agent_id("agent-b")
            .bus(bus.clone())
            .llm(MockLlmProvider::with_texts(["done-b"]))
            .strategy(ReactStrategy::default())
            .build(),
    );

    let a = agent_a.clone();
    let b = agent_b.clone();
    let (r_a, r_b) = tokio::join!(
        tokio::spawn(async move { a.cycle().await }),
        tokio::spawn(async move { b.cycle().await }),
    );
    r_a.unwrap().unwrap();
    r_b.unwrap().unwrap();

    let log = bus.log().await;
    let cycle_starts: Vec<_> = log
        .iter()
        .filter(|e| e.kind == kinds::AGENT_CYCLE_START)
        .collect();

    // Both agents should have produced a cycle start event.
    assert_eq!(cycle_starts.len(), 2, "expected 2 cycle.start events");

    // Each with a distinct agent_id.
    let ids: std::collections::HashSet<_> = cycle_starts
        .iter()
        .map(|e| e.metadata[meta_keys::AGENT_ID].as_str().unwrap())
        .collect();
    assert_eq!(ids.len(), 2, "expected 2 distinct agent_ids");
    assert!(ids.contains("agent-a"));
    assert!(ids.contains("agent-b"));
}

// ── DAG / checkpoint tests ────────────────────────────────────────────────────

#[tokio::test]
async fn checkpoint_and_rollback_truncates_active_log() {
    init_tracing();
    let bus = EventBus::new();

    // Publish a user message and checkpoint it.
    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "first"})))
        .await
        .unwrap();
    let cp_id = bus.checkpoint().await.unwrap();

    // Simulate a "bad" assistant response after the checkpoint.
    bus.publish(Event::new(
        kinds::ASSISTANT_MESSAGE,
        json!({"content": "wrong", "tool_calls": []}),
    ))
    .await
    .unwrap();
    assert_eq!(bus.log_len().await, 3);

    // Roll back to the checkpoint: active log should shrink to 1 (just user.message).
    let branch_id = bus.rollback(cp_id).await.unwrap();
    assert_eq!(bus.log_len().await, 1);

    // The rejected branch should contain checkpoint + assistant.message.
    let all = bus.all_rejected_branches().await;
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].0, branch_id);
    assert_eq!(all[0].1.len(), 2);
}

#[tokio::test]
async fn rejected_branches_from_returns_correct_branch() {
    init_tracing();
    let bus = EventBus::new();

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "q"})))
        .await
        .unwrap();
    let user_event_id = bus.log().await[0].id;

    let cp_id = bus.checkpoint().await.unwrap();
    bus.publish(Event::new(
        kinds::ASSISTANT_MESSAGE,
        json!({"content": "bad", "tool_calls": []}),
    ))
    .await
    .unwrap();
    bus.rollback(cp_id).await.unwrap();

    // After rollback, the anchor is the user.message (the event before the checkpoint).
    let rejected = bus.rejected_branches_from(user_event_id).await;
    assert_eq!(rejected.len(), 1);
    assert_eq!(rejected[0][0].kind, kinds::CHECKPOINT);
    assert_eq!(rejected[0][1].kind, kinds::ASSISTANT_MESSAGE);
}

#[tokio::test]
async fn negative_aware_assembler_uses_rejected_branches() {
    init_tracing();

    struct RecordingAssembler {
        saw_negatives: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    #[async_trait]
    impl ContextAssembler for RecordingAssembler {
        async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
            if !context.rejected_branches.is_empty() {
                self.saw_negatives
                    .store(true, std::sync::atomic::Ordering::SeqCst);
            }
            // Return a minimal valid prompt so the LLM gets called.
            vec![ChatMessage::user("retry")]
        }
    }

    let saw = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let assembler = RecordingAssembler {
        saw_negatives: saw.clone(),
    };

    let bus = EventBus::new();
    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "q"})))
        .await
        .unwrap();
    let anchor_id = bus.log().await[0].id;
    let cp_id = bus.checkpoint().await.unwrap();
    bus.publish(Event::new(
        kinds::ASSISTANT_MESSAGE,
        json!({"content": "bad", "tool_calls": []}),
    ))
    .await
    .unwrap();
    bus.rollback(cp_id).await.unwrap();

    // Manually call the assembler with negative context, as an orchestration
    // layer would do after detecting the failure.
    let active = bus.log().await;
    let rejected = bus.rejected_branches_from(anchor_id).await;
    let ctx = AssemblyContext::new(&active).with_rejected_branches(rejected);
    assembler.assemble(&ctx).await;

    assert!(
        saw.load(std::sync::atomic::Ordering::SeqCst),
        "assembler should have received non-empty rejected_branches"
    );
}

#[tokio::test]
async fn agent_cycle_continues_normally_after_rollback() {
    init_tracing();
    let bus = EventBus::new();

    // Simulate: user message → checkpoint → bad assistant → rollback → retry.
    bus.publish(Event::new(
        kinds::USER_MESSAGE,
        json!({"text": "write a hello-world"}),
    ))
    .await
    .unwrap();
    let cp_id = bus.checkpoint().await.unwrap();
    bus.publish(Event::new(
        kinds::ASSISTANT_MESSAGE,
        json!({"content": "no tool calls", "tool_calls": []}),
    ))
    .await
    .unwrap();
    bus.rollback(cp_id).await.unwrap();

    // After rollback, the agent should still be able to run a fresh cycle.
    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::with_texts(["Retried successfully!"]))
        .system_prompt("You are helpful.")
        .strategy(ReactStrategy::default())
        .build();

    agent.cycle().await.unwrap();

    let log = bus.log().await;
    let final_msg = log
        .iter()
        .filter(|e| e.kind == kinds::ASSISTANT_MESSAGE)
        .next_back()
        .expect("no final assistant.message");
    assert_eq!(
        final_msg.payload["content"].as_str().unwrap(),
        "Retried successfully!"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Dynamic entity tests
// ─────────────────────────────────────────────────────────────────────────────

// Helper tool for tests
struct CounterTool {
    name: String,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for CounterTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            &self.name,
            "A counter tool",
            json!({ "type": "object", "properties": {} }),
        )
    }
    async fn execute(&self, _args: Value) -> Result<Value, AgentError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(json!({ "count": self.calls.load(Ordering::Relaxed) }))
    }
}

#[tokio::test]
async fn dynamic_tool_add_visible_in_cycle() {
    // Tools added after build() are seen by the agent on the next cycle.
    let bus = EventBus::new();
    let calls = Arc::new(AtomicUsize::new(0));
    let calls2 = Arc::clone(&calls);

    let builder = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "t1".into(),
                    kind: "function".into(),
                    function: FunctionCall {
                        name: "late_tool".into(),
                        arguments: "{}".into(),
                    },
                }],
                finish_reason: "tool_calls".into(),
            },
            LlmResponse {
                content: Some("done".into()),
                tool_calls: vec![],
                finish_reason: "stop".into(),
            },
        ]))
        .strategy(ReactStrategy::default());

    // Get handle BEFORE build.
    let registry = builder.tool_registry();
    let agent = builder.build();

    // Add the tool AFTER building.
    registry.add_tool(CounterTool {
        name: "late_tool".into(),
        calls: calls2,
    });

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "go"})))
        .await
        .unwrap();
    agent.cycle().await.unwrap();

    assert_eq!(
        calls.load(Ordering::Relaxed),
        1,
        "late_tool should have been called once"
    );
}

#[tokio::test]
async fn dynamic_tool_remove_hides_from_llm() {
    // After remove(), the tool definition is gone from the next LLM call.
    // The LLM will not be offered it and hence will not call it.
    let bus = EventBus::new();

    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::with_texts(["no tools needed"]))
        .strategy(ReactStrategy::default())
        .build();

    // The agent has no tools to begin with — nothing to remove, but the API should work.
    let tools = agent.tools();
    let calls = Arc::new(AtomicUsize::new(0));
    tools.add_tool(CounterTool {
        name: "removable".into(),
        calls: Arc::clone(&calls),
    });
    assert!(tools.names().contains(&"removable".to_string()));

    tools.remove("removable");
    assert!(!tools.names().contains(&"removable".to_string()));

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "hi"})))
        .await
        .unwrap();
    agent.cycle().await.unwrap();

    assert_eq!(
        calls.load(Ordering::Relaxed),
        0,
        "removed tool should never be called"
    );
}

#[tokio::test]
async fn tool_registry_clone_shares_state() {
    // Two clones of the same registry share the underlying tool list.
    let bus = EventBus::new();
    let builder = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::with_texts(Vec::<&str>::new()))
        .strategy(ReactStrategy::default());

    let r1 = builder.tool_registry();
    let r2 = r1.clone(); // second handle
    let _agent = builder.build();

    r1.add_tool(CounterTool {
        name: "shared".into(),
        calls: Arc::new(AtomicUsize::new(0)),
    });
    assert!(
        r2.names().contains(&"shared".to_string()),
        "r2 should see the tool added via r1"
    );

    r2.remove("shared");
    assert!(
        !r1.names().contains(&"shared".to_string()),
        "r1 should see the removal done via r2"
    );
}

#[tokio::test]
async fn keyword_tool_selector_filters_definitions() {
    // KeywordToolSelector only exposes tools whose names match a keyword.
    // The LLM only receives the matching definitions, so it can only call those.
    let bus = EventBus::new();
    let search_calls = Arc::new(AtomicUsize::new(0));
    let write_calls = Arc::new(AtomicUsize::new(0));

    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::new(vec![
            // LLM is only offered "search_web" — calls it.
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "t1".into(),
                    kind: "function".into(),
                    function: FunctionCall {
                        name: "search_web".into(),
                        arguments: "{}".into(),
                    },
                }],
                finish_reason: "tool_calls".into(),
            },
            LlmResponse {
                content: Some("done".into()),
                tool_calls: vec![],
                finish_reason: "stop".into(),
            },
        ]))
        .tool(CounterTool {
            name: "search_web".into(),
            calls: Arc::clone(&search_calls),
        })
        .tool(CounterTool {
            name: "write_file".into(),
            calls: Arc::clone(&write_calls),
        })
        .tool_selector(KeywordToolSelector::new(vec!["search"]))
        .strategy(ReactStrategy::default())
        .build();

    bus.publish(Event::new(
        kinds::USER_MESSAGE,
        json!({"text": "search something"}),
    ))
    .await
    .unwrap();
    agent.cycle().await.unwrap();

    assert_eq!(
        search_calls.load(Ordering::Relaxed),
        1,
        "search_web should be called"
    );
    assert_eq!(
        write_calls.load(Ordering::Relaxed),
        0,
        "write_file should not be called"
    );
}

#[tokio::test]
async fn dynamic_hook_chain_add_remove() {
    // DynamicHookChain: adding a hook after build affects the next cycle.
    let bus = EventBus::new();
    let step_count = Arc::new(AtomicUsize::new(0));
    let step_count2 = Arc::clone(&step_count);

    struct CountingHook {
        counter: Arc<AtomicUsize>,
    }
    #[async_trait]
    impl CycleHook for CountingHook {
        async fn before_step(&self, _ctx: &HookContext<'_>) -> HookAction {
            self.counter.fetch_add(1, Ordering::Relaxed);
            HookAction::Continue
        }
    }

    let dyn_hooks = DynamicHookChain::new();
    let handle = dyn_hooks.clone();

    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::with_texts(["first", "second"]))
        .hook(dyn_hooks)
        .strategy(ReactStrategy::default())
        .build();

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "hi"})))
        .await
        .unwrap();

    // First cycle — no hooks added yet.
    agent.cycle().await.unwrap();
    assert_eq!(step_count.load(Ordering::Relaxed), 0);

    // Add a counting hook — it fires on the next cycle.
    handle.add_hook(CountingHook {
        counter: step_count2,
    });
    agent.cycle().await.unwrap();
    assert!(
        step_count.load(Ordering::Relaxed) > 0,
        "hook should fire after being added"
    );

    // Remove all — the hook should not fire again.
    handle.remove_all();
    let before = step_count.load(Ordering::Relaxed);
    agent.cycle().await.unwrap();
    assert_eq!(
        step_count.load(Ordering::Relaxed),
        before,
        "hook should not fire after remove_all"
    );
}

#[tokio::test]
async fn dynamic_context_assembler_swap() {
    // DynamicContextAssembler: swapping the inner assembler changes the
    // messages the LLM sees on the next cycle.
    let bus = EventBus::new();

    struct FixedAssembler(String);
    #[async_trait]
    impl ContextAssembler for FixedAssembler {
        async fn assemble(&self, _ctx: &AssemblyContext<'_>) -> Vec<ChatMessage> {
            vec![ChatMessage::user(self.0.clone())]
        }
    }

    let dyn_ctx = DynamicContextAssembler::new(FixedAssembler("phase-one".into()));
    let handle = dyn_ctx.clone();

    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::with_texts(["reply-one", "reply-two"]))
        .context(dyn_ctx)
        .strategy(ReactStrategy::default())
        .build();

    // First cycle — uses the original assembler.
    agent.cycle().await.unwrap();

    // Swap to a different assembler.
    handle.swap(FixedAssembler("phase-two".into()));

    // Second cycle — uses the swapped assembler.
    agent.cycle().await.unwrap();

    // Both cycles produced responses (we mostly check it runs without panic).
    let log = bus.log().await;
    let replies: Vec<_> = log
        .iter()
        .filter(|e| e.kind == kinds::ASSISTANT_MESSAGE)
        .collect();
    assert_eq!(
        replies.len(),
        2,
        "each cycle should produce one assistant.message"
    );
}

// ── Session::run() — reactive mode tests ─────────────────────────────────────

#[tokio::test]
async fn session_run_reacts_to_bus_event() {
    // Session::run() should cycle automatically when a user.message is published
    // to the bus from an external task — no chat() call required.
    init_tracing();

    let session = Session::builder()
        .llm(MockLlmProvider::with_texts(["reactive reply"]))
        .system_prompt("You are helpful.")
        .build();

    let bus = session.bus().clone();
    let mut reply_rx = bus.subscribe();

    // Spawn the reactive agent loop.
    tokio::spawn(async move { session.run().await });

    // Yield once so the spawned task has a chance to subscribe to the bus
    // before we publish — agent.run() subscribes synchronously on entry.
    tokio::task::yield_now().await;

    // Publish a user message directly as a bus event from "outside" the session.
    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "hello"})))
        .await
        .unwrap();

    // Wait for the assistant response on the bus.
    let mut got_reply = false;
    while let Some(event) = reply_rx.recv().await {
        if event.kind == kinds::ASSISTANT_MESSAGE {
            assert_eq!(event.payload["content"].as_str().unwrap(), "reactive reply");
            got_reply = true;
            break;
        }
    }
    assert!(got_reply, "expected assistant.message via reactive run()");
}

#[tokio::test]
async fn session_run_handles_multiple_turns() {
    // Session::run() should handle multiple sequential user messages,
    // producing one assistant response per user message.
    init_tracing();

    let session = Session::builder()
        .llm(MockLlmProvider::with_texts(["first", "second"]))
        .system_prompt("You are helpful.")
        .build();

    let bus = session.bus().clone();
    let mut reply_rx = bus.subscribe();

    tokio::spawn(async move { session.run().await });

    // Yield so the spawned agent task subscribes before we publish.
    tokio::task::yield_now().await;

    // First turn.
    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "turn 1"})))
        .await
        .unwrap();

    let mut replies = 0usize;
    let mut last_content = String::new();

    // Collect responses until we have two non-empty assistant messages.
    while let Some(event) = reply_rx.recv().await {
        if event.kind == kinds::ASSISTANT_MESSAGE {
            let content = event.payload["content"].as_str().unwrap_or("").to_string();
            if !content.is_empty() {
                last_content = content;
                replies += 1;
                if replies == 1 {
                    // Publish second turn after first reply arrives.
                    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "turn 2"})))
                        .await
                        .unwrap();
                } else {
                    break;
                }
            }
        }
    }

    assert_eq!(replies, 2);
    assert_eq!(last_content, "second");
}
