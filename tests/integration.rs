use async_trait::async_trait;
use eventage::agent::{
    AgentBuilder, AgentError, AssemblyContext, ContextAssembler, CycleHook,
    DynamicContextAssembler, DynamicHookChain, HookAction, HookContext, KeywordToolSelector,
    ReactStrategy, Session, Tool,
};
use eventage::llm::{
    ChatMessage, FunctionCall, LlmResponse, MockLlmProvider, ToolCall, ToolDefinition,
};
use eventage::{kinds, meta_keys, Event, EventBus};
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

    let tool_call_response = LlmResponse {
        content: None,
        tool_calls: vec![ToolCall {
            id: "call_001".to_string(),
            kind: "function".to_string(),
            function: FunctionCall {
                name: "echo".to_string(),
                arguments: r#"{"text":"ping"}"#.to_string(),
            },
            extra_content: None,
        }],
        finish_reason: "tool_calls".to_string(),
        ..Default::default()
    };
    let final_response = LlmResponse {
        content: Some("The echo returned: ping".to_string()),
        tool_calls: vec![],
        finish_reason: "stop".to_string(),
        ..Default::default()
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

    assert!(
        log.iter().any(|e| e.kind == kinds::TOOL_CALL_PROPOSED),
        "missing tool.call.proposed"
    );

    let tool_result = log
        .iter()
        .find(|e| e.kind == kinds::TOOL_RESULT)
        .expect("no tool.result in log");
    assert_eq!(tool_result.payload["result"]["echoed"], "ping");

    let final_msg = log
        .iter().rfind(|e| e.kind == kinds::ASSISTANT_MESSAGE)
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

    assert_eq!(
        cycle_start.metadata[meta_keys::AGENT_ID].as_str().unwrap(),
        "test-agent"
    );
    assert_eq!(
        cycle_end.metadata[meta_keys::AGENT_ID].as_str().unwrap(),
        "test-agent"
    );

    let trace_id = cycle_start.metadata[meta_keys::TRACE_ID]
        .as_str()
        .expect("missing trace_id on cycle_start");
    assert_eq!(
        cycle_end.metadata[meta_keys::TRACE_ID].as_str().unwrap(),
        trace_id
    );

    assert!(
        cycle_end.metadata.contains_key(meta_keys::ELAPSED_MS),
        "missing elapsed_ms on cycle_end"
    );
}

#[tokio::test]
async fn agent_message_carries_agent_id_metadata() {
    init_tracing();
    let bus = EventBus::new();

    let msg = Event::new(kinds::AGENT_MESSAGE, json!({"text": "hello a"}))
        .with_meta(meta_keys::TO_AGENT_ID, json!("agent-a"));
    bus.publish(msg).await.unwrap();

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

    let agent_msg = log.iter().find(|e| e.kind == kinds::AGENT_MESSAGE).unwrap();
    assert_eq!(
        agent_msg.metadata[meta_keys::TO_AGENT_ID].as_str().unwrap(),
        "agent-a"
    );

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

    assert_eq!(cycle_starts.len(), 2, "expected 2 cycle.start events");

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

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "first"})))
        .await
        .unwrap();
    let cp_id = bus.checkpoint().await.unwrap();

    bus.publish(Event::new(
        kinds::ASSISTANT_MESSAGE,
        json!({"content": "wrong", "tool_calls": []}),
    ))
    .await
    .unwrap();
    assert_eq!(bus.log_len().await, 3);

    let branch_id = bus.rollback(cp_id).await.unwrap();
    // user message + durable rollback tombstone
    assert_eq!(bus.log_len().await, 2);

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

    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::with_texts(["Retried successfully!"]))
        .system_prompt("You are helpful.")
        .strategy(ReactStrategy::default())
        .build();

    agent.cycle().await.unwrap();

    let log = bus.log().await;
    let final_msg = log
        .iter().rfind(|e| e.kind == kinds::ASSISTANT_MESSAGE)
        .expect("no final assistant.message");
    assert_eq!(
        final_msg.payload["content"].as_str().unwrap(),
        "Retried successfully!"
    );
}

// ── Dynamic entity tests ──────────────────────────────────────────────────────

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
                    extra_content: None,
                }],
                finish_reason: "tool_calls".into(),
                ..Default::default()
            },
            LlmResponse {
                content: Some("done".into()),
                tool_calls: vec![],
                finish_reason: "stop".into(),
                ..Default::default()
            },
        ]))
        .strategy(ReactStrategy::default());

    let registry = builder.tool_registry();
    let agent = builder.build();

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
    let bus = EventBus::new();

    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::with_texts(["no tools needed"]))
        .strategy(ReactStrategy::default())
        .build();

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
    let bus = EventBus::new();
    let builder = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::with_texts(Vec::<&str>::new()))
        .strategy(ReactStrategy::default());

    let r1 = builder.tool_registry();
    let r2 = r1.clone();
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
    let bus = EventBus::new();
    let search_calls = Arc::new(AtomicUsize::new(0));
    let write_calls = Arc::new(AtomicUsize::new(0));

    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::new(vec![
            LlmResponse {
                content: None,
                tool_calls: vec![ToolCall {
                    id: "t1".into(),
                    kind: "function".into(),
                    function: FunctionCall {
                        name: "search_web".into(),
                        arguments: "{}".into(),
                    },
                    extra_content: None,
                }],
                finish_reason: "tool_calls".into(),
                ..Default::default()
            },
            LlmResponse {
                content: Some("done".into()),
                tool_calls: vec![],
                finish_reason: "stop".into(),
                ..Default::default()
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

    agent.cycle().await.unwrap();
    assert_eq!(step_count.load(Ordering::Relaxed), 0);

    handle.add_hook(CountingHook {
        counter: step_count2,
    });
    agent.cycle().await.unwrap();
    assert!(
        step_count.load(Ordering::Relaxed) > 0,
        "hook should fire after being added"
    );

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

    agent.cycle().await.unwrap();

    handle.swap(FixedAssembler("phase-two".into()));

    agent.cycle().await.unwrap();

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
    init_tracing();

    let session = Session::builder()
        .llm(MockLlmProvider::with_texts(["reactive reply"]))
        .system_prompt("You are helpful.")
        .build();

    let bus = session.bus().clone();
    let mut reply_rx = bus.subscribe();

    tokio::spawn(async move { session.run().await });

    tokio::task::yield_now().await;

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "hello"})))
        .await
        .unwrap();

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
    init_tracing();

    let session = Session::builder()
        .llm(MockLlmProvider::with_texts(["first", "second"]))
        .system_prompt("You are helpful.")
        .build();

    let bus = session.bus().clone();
    let mut reply_rx = bus.subscribe();

    tokio::spawn(async move { session.run().await });

    tokio::task::yield_now().await;

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "turn 1"})))
        .await
        .unwrap();

    let mut replies = 0usize;
    let mut last_content = String::new();

    while let Some(event) = reply_rx.recv().await {
        if event.kind == kinds::ASSISTANT_MESSAGE {
            let content = event.payload["content"].as_str().unwrap_or("").to_string();
            if !content.is_empty() {
                last_content = content;
                replies += 1;
                if replies == 1 {
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

// ── Harness guardrail tests (2026 upgrades) ──────────────────────────────────

/// A tool whose execution sleeps, for timeout testing.
struct SlowTool;

#[async_trait]
impl Tool for SlowTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "slow",
            "sleeps",
            json!({ "type": "object", "properties": {} }),
        )
    }
    async fn execute(&self, _args: Value) -> Result<Value, AgentError> {
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        Ok(json!({ "done": true }))
    }
}

fn tool_call_response(name: &str, args: &str) -> LlmResponse {
    LlmResponse {
        content: None,
        tool_calls: vec![ToolCall {
            id: "t1".into(),
            kind: "function".into(),
            function: FunctionCall {
                name: name.into(),
                arguments: args.into(),
            },
            extra_content: None,
        }],
        finish_reason: "tool_calls".into(),
        ..Default::default()
    }
}

fn text_response(text: &str) -> LlmResponse {
    LlmResponse {
        content: Some(text.into()),
        tool_calls: vec![],
        finish_reason: "stop".into(),
        ..Default::default()
    }
}

#[tokio::test]
async fn tool_timeout_produces_error_result() {
    init_tracing();
    let bus = EventBus::new();
    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::new(vec![
            tool_call_response("slow", "{}"),
            text_response("gave up"),
        ]))
        .tool(SlowTool)
        .strategy(ReactStrategy {
            tool_timeout: Some(std::time::Duration::from_millis(50)),
            ..Default::default()
        })
        .build();

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "go"})))
        .await
        .unwrap();
    agent.cycle().await.unwrap();

    let log = bus.log().await;
    let result = log
        .iter()
        .find(|e| e.kind == kinds::TOOL_RESULT)
        .expect("no tool.result");
    let err = result.payload["error"]
        .as_str()
        .expect("expected error field");
    assert!(err.contains("timed out"), "unexpected error: {err}");
}

#[tokio::test]
async fn invalid_json_args_are_reported_to_model_without_executing() {
    init_tracing();
    let calls = Arc::new(AtomicUsize::new(0));
    let bus = EventBus::new();
    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::new(vec![
            tool_call_response("counter", "{not json"),
            text_response("ok"),
        ]))
        .tool(CounterTool {
            name: "counter".into(),
            calls: Arc::clone(&calls),
        })
        .strategy(ReactStrategy::default())
        .build();

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "go"})))
        .await
        .unwrap();
    agent.cycle().await.unwrap();

    assert_eq!(calls.load(Ordering::Relaxed), 0, "tool must not execute");
    let log = bus.log().await;
    let result = log.iter().find(|e| e.kind == kinds::TOOL_RESULT).unwrap();
    let err = result.payload["error"].as_str().unwrap();
    assert!(err.contains("invalid JSON"), "unexpected error: {err}");
}

#[tokio::test]
async fn deny_hook_surfaces_reason_to_model() {
    init_tracing();

    struct DenyHook;
    #[async_trait]
    impl CycleHook for DenyHook {
        async fn before_tool(
            &self,
            _ctx: &HookContext<'_>,
            _name: &str,
            _args: &Value,
        ) -> HookAction {
            HookAction::Deny("writes are not allowed in read-only mode".into())
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let bus = EventBus::new();
    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::new(vec![
            tool_call_response("counter", "{}"),
            text_response("understood"),
        ]))
        .tool(CounterTool {
            name: "counter".into(),
            calls: Arc::clone(&calls),
        })
        .hook(DenyHook)
        .strategy(ReactStrategy::default())
        .build();

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "go"})))
        .await
        .unwrap();
    agent.cycle().await.unwrap();

    assert_eq!(calls.load(Ordering::Relaxed), 0, "denied tool must not run");
    let log = bus.log().await;
    let result = log.iter().find(|e| e.kind == kinds::TOOL_RESULT).unwrap();
    assert_eq!(result.payload["result"]["denied"], true);
    assert_eq!(
        result.payload["result"]["reason"],
        "writes are not allowed in read-only mode"
    );
}

#[tokio::test]
async fn oversized_tool_results_are_middle_truncated() {
    init_tracing();

    struct BigTool;
    #[async_trait]
    impl Tool for BigTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "big",
                "returns a lot",
                json!({ "type": "object", "properties": {} }),
            )
        }
        async fn execute(&self, _args: Value) -> Result<Value, AgentError> {
            Ok(json!(format!("HEAD{}TAIL", "x".repeat(10_000))))
        }
    }

    let bus = EventBus::new();
    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::new(vec![
            tool_call_response("big", "{}"),
            text_response("done"),
        ]))
        .tool(BigTool)
        .strategy(ReactStrategy {
            max_tool_result_chars: Some(1_000),
            ..Default::default()
        })
        .build();

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "go"})))
        .await
        .unwrap();
    agent.cycle().await.unwrap();

    let log = bus.log().await;
    let result = log.iter().find(|e| e.kind == kinds::TOOL_RESULT).unwrap();
    let text = result.payload["result"].as_str().unwrap();
    assert!(
        text.len() < 2_000,
        "result should be capped, got {}",
        text.len()
    );
    assert!(text.contains("HEAD"), "head must be preserved");
    assert!(text.contains("TAIL"), "tail must be preserved");
    assert!(text.contains("elided by the harness"), "marker missing");
}

#[tokio::test]
async fn max_steps_finalizes_with_wrapup_instead_of_error() {
    init_tracing();
    let calls = Arc::new(AtomicUsize::new(0));
    let bus = EventBus::new();
    // The mock always proposes another tool call, so the loop only ends via
    // the step budget.
    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::new(vec![tool_call_response(
            "counter", "{}",
        )]))
        .tool(CounterTool {
            name: "counter".into(),
            calls: Arc::clone(&calls),
        })
        .strategy(ReactStrategy {
            max_steps: 2,
            ..Default::default()
        })
        .build();

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "go"})))
        .await
        .unwrap();

    agent.cycle().await.expect("finalization should not error");

    let log = bus.log().await;
    let final_msg = log
        .iter().rfind(|e| e.kind == kinds::ASSISTANT_MESSAGE)
        .unwrap();
    assert_eq!(final_msg.payload["finalized_due_to"], "max_steps");
    assert_eq!(
        calls.load(Ordering::Relaxed),
        2,
        "budget of 2 steps ran 2 tools"
    );
}

#[tokio::test]
async fn stuck_hint_reaches_llm_context() {
    use eventage::agent::events_to_messages;

    let events = vec![
        Event::new(kinds::USER_MESSAGE, json!({"text": "go"})),
        Event::new(
            kinds::AGENT_STUCK,
            json!({
                "kind": "RepeatingAction",
                "repeat_count": 3,
                "hint": "Try a different approach."
            }),
        ),
    ];
    let messages = events_to_messages(&events);
    assert_eq!(messages.len(), 2);
    let hint = messages[1].content.as_deref().unwrap();
    assert!(hint.contains("Loop detected"), "got: {hint}");
    assert!(hint.contains("Try a different approach."), "got: {hint}");
}

#[tokio::test]
async fn reasoning_content_is_preserved_on_bus() {
    init_tracing();
    let bus = EventBus::new();
    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::new(vec![LlmResponse {
            content: Some("42".into()),
            reasoning_content: Some("6 times 7 is 42".into()),
            finish_reason: "stop".into(),
            ..Default::default()
        }]))
        .strategy(ReactStrategy::default())
        .build();

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "6*7?"})))
        .await
        .unwrap();
    agent.cycle().await.unwrap();

    let log = bus.log().await;
    let msg = log
        .iter()
        .find(|e| e.kind == kinds::ASSISTANT_MESSAGE)
        .unwrap();
    assert_eq!(msg.payload["reasoning_content"], "6 times 7 is 42");

    // But reasoning must NOT be replayed into subsequent LLM requests.
    let messages = eventage::agent::events_to_messages(&log);
    let assistant = messages
        .iter()
        .find(|m| m.role == eventage::llm::Role::Assistant)
        .unwrap();
    assert_eq!(assistant.content.as_deref(), Some("42"));
}

// ── Enterprise-harness tests (streaming, governance, speculation) ────────────

#[tokio::test]
async fn streaming_broadcasts_ephemeral_deltas() {
    init_tracing();
    let bus = EventBus::new();
    let mut rx = bus.subscribe();

    // MockLlmProvider has no native streaming — the default complete_stream
    // emits the whole answer as one delta, exercising the fallback path.
    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::with_texts(["streamed answer"]))
        .strategy(ReactStrategy {
            stream: true,
            ..Default::default()
        })
        .build();

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "go"})))
        .await
        .unwrap();
    agent.cycle().await.unwrap();

    // Deltas are broadcast to subscribers…
    let mut saw_delta = false;
    while let Ok(Some(event)) =
        tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv()).await
    {
        if event.kind == kinds::ASSISTANT_DELTA {
            assert_eq!(event.payload["content"], "streamed answer");
            saw_delta = true;
        }
    }
    assert!(saw_delta, "subscriber should receive assistant.delta");

    // …but never stored in the durable log.
    let log = bus.log().await;
    assert!(
        log.iter().all(|e| e.kind != kinds::ASSISTANT_DELTA),
        "deltas must not pollute the DAG"
    );
    assert!(log.iter().any(|e| e.kind == kinds::ASSISTANT_MESSAGE));
}

#[tokio::test]
async fn permission_ask_flow_approves_via_bus() {
    use eventage::agent::PermissionPolicyHook;
    init_tracing();

    let calls = Arc::new(AtomicUsize::new(0));
    let bus = EventBus::new();

    // Approver: watches for permission.request and approves it.
    // Subscribe before spawning so no request can be missed.
    let mut rx = bus.subscribe();
    let approver_bus = bus.clone();
    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            if event.kind == kinds::PERMISSION_REQUEST {
                let request_id = event.payload["request_id"].as_str().unwrap().to_string();
                approver_bus
                    .publish(Event::new(
                        kinds::PERMISSION_DECISION,
                        json!({ "request_id": request_id, "approve": true }),
                    ))
                    .await
                    .unwrap();
            }
        }
    });

    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::new(vec![
            tool_call_response("counter", "{}"),
            text_response("done"),
        ]))
        .tool(CounterTool {
            name: "counter".into(),
            calls: Arc::clone(&calls),
        })
        .hook(
            PermissionPolicyHook::new()
                .ask("counter")
                .deny_by_default("not allowlisted"),
        )
        .strategy(ReactStrategy::default())
        .build();

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "go"})))
        .await
        .unwrap();
    agent.cycle().await.unwrap();

    assert_eq!(calls.load(Ordering::Relaxed), 1, "approved tool should run");
}

#[tokio::test]
async fn permission_deny_by_default_blocks_unlisted_tools() {
    use eventage::agent::PermissionPolicyHook;
    init_tracing();

    let calls = Arc::new(AtomicUsize::new(0));
    let bus = EventBus::new();
    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::new(vec![
            tool_call_response("counter", "{}"),
            text_response("ok"),
        ]))
        .tool(CounterTool {
            name: "counter".into(),
            calls: Arc::clone(&calls),
        })
        .hook(
            PermissionPolicyHook::new()
                .allow("read_*")
                .deny_by_default("tool not in the deployment allowlist"),
        )
        .strategy(ReactStrategy::default())
        .build();

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "go"})))
        .await
        .unwrap();
    agent.cycle().await.unwrap();

    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let log = bus.log().await;
    let result = log.iter().find(|e| e.kind == kinds::TOOL_RESULT).unwrap();
    assert_eq!(result.payload["result"]["denied"], true);
    assert_eq!(
        result.payload["result"]["reason"],
        "tool not in the deployment allowlist"
    );
}

#[tokio::test]
async fn schema_violations_are_fed_back_to_model() {
    init_tracing();

    struct StrictTool;
    #[async_trait]
    impl Tool for StrictTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::function(
                "strict",
                "needs a path",
                json!({
                    "type": "object",
                    "properties": { "path": { "type": "string" } },
                    "required": ["path"]
                }),
            )
        }
        async fn execute(&self, _args: Value) -> Result<Value, AgentError> {
            panic!("must not execute with invalid args");
        }
    }

    let bus = EventBus::new();
    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::new(vec![
            tool_call_response("strict", r#"{"path": 42}"#),
            text_response("corrected"),
        ]))
        .tool(StrictTool)
        .strategy(ReactStrategy::default())
        .build();

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "go"})))
        .await
        .unwrap();
    agent.cycle().await.unwrap();

    let log = bus.log().await;
    let result = log.iter().find(|e| e.kind == kinds::TOOL_RESULT).unwrap();
    let err = result.payload["error"].as_str().unwrap();
    assert!(err.contains("arguments.path"), "got: {err}");
    assert!(err.contains("'string'"), "got: {err}");
}

#[tokio::test]
async fn session_resumes_from_prior_bus() {
    init_tracing();

    // First session: build history.
    let bus = EventBus::new();
    bus.publish(Event::new(
        kinds::USER_MESSAGE,
        json!({"text": "remember 42"}),
    ))
    .await
    .unwrap();
    bus.publish(Event::new(
        kinds::ASSISTANT_MESSAGE,
        json!({"content": "noted: 42", "tool_calls": []}),
    ))
    .await
    .unwrap();

    // Second session: attach the restored bus and continue.
    let mut session = Session::builder()
        .llm(MockLlmProvider::with_texts(["it was 42"]))
        .system_prompt("You are helpful.")
        .bus(bus.clone())
        .build();

    let reply = session.chat("what number?").await.unwrap();
    assert_eq!(reply, "it was 42");
    assert!(
        bus.log().await.len() >= 4,
        "resumed bus must retain prior history"
    );
}

#[tokio::test]
async fn provider_extra_round_trips_through_event_log() {
    init_tracing();

    // A response carrying provider-opaque state (e.g. Anthropic thinking
    // blocks or OpenAI Responses reasoning items).
    let with_state = LlmResponse {
        content: None,
        tool_calls: vec![ToolCall {
            id: "c1".into(),
            kind: "function".into(),
            function: FunctionCall {
                name: "counter".into(),
                arguments: "{}".into(),
            },
            extra_content: None,
        }],
        finish_reason: "tool_calls".into(),
        provider_extra: Some(json!({
            "anthropic_blocks": [{ "type": "thinking", "thinking": "hmm", "signature": "sig123" }]
        })),
        ..Default::default()
    };

    // A provider that asserts the opaque state comes back on the next call.
    struct AssertingProvider {
        first: std::sync::Mutex<Option<LlmResponse>>,
    }
    #[async_trait]
    impl eventage::llm::LlmProvider for AssertingProvider {
        async fn complete(
            &self,
            messages: Vec<ChatMessage>,
            _tools: Vec<ToolDefinition>,
        ) -> Result<LlmResponse, eventage::llm::LlmError> {
            if let Some(first) = self.first.lock().unwrap().take() {
                return Ok(first);
            }
            // Second step: the assistant turn must carry the state back.
            let assistant = messages
                .iter()
                .find(|m| m.role == eventage::llm::Role::Assistant)
                .expect("assistant turn in history");
            let extra = assistant.provider_extra.as_ref().expect("state restored");
            assert_eq!(extra["anthropic_blocks"][0]["signature"], "sig123");
            Ok(LlmResponse {
                content: Some("done".into()),
                finish_reason: "stop".into(),
                ..Default::default()
            })
        }
        fn model(&self) -> &str {
            "asserting"
        }
    }

    let calls = Arc::new(AtomicUsize::new(0));
    let bus = EventBus::new();
    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(AssertingProvider {
            first: std::sync::Mutex::new(Some(with_state)),
        })
        .tool(CounterTool {
            name: "counter".into(),
            calls: Arc::clone(&calls),
        })
        .strategy(ReactStrategy::default())
        .build();

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "go"})))
        .await
        .unwrap();
    agent.cycle().await.unwrap();

    // The opaque blocks are also durable on the event itself.
    let log = bus.log().await;
    let msg = log
        .iter()
        .find(|e| e.kind == kinds::ASSISTANT_MESSAGE)
        .unwrap();
    assert_eq!(
        msg.payload["provider_extra"]["anthropic_blocks"][0]["signature"],
        "sig123"
    );
}

// ── Roadmap-closure tests ────────────────────────────────────────────────────

#[tokio::test]
async fn multimodal_user_message_reaches_the_provider() {
    use eventage::llm::{ContentPart, Role};
    init_tracing();

    // A provider that asserts the image survived assembly.
    struct VisionProvider;
    #[async_trait]
    impl eventage::llm::LlmProvider for VisionProvider {
        async fn complete(
            &self,
            messages: Vec<ChatMessage>,
            _tools: Vec<ToolDefinition>,
        ) -> Result<LlmResponse, eventage::llm::LlmError> {
            let user = messages.iter().find(|m| m.role == Role::User).unwrap();
            assert!(
                user.is_multimodal(),
                "image part must survive the event log"
            );
            assert_eq!(user.parts.len(), 2);
            assert_eq!(user.parts[0].as_text(), Some("what is this?"));

            // And it must serialize to the OpenAI content-array wire form.
            let wire = serde_json::to_value(user).unwrap();
            assert_eq!(wire["content"][1]["type"], "image_url");
            assert_eq!(
                wire["content"][1]["image_url"]["url"],
                "data:image/png;base64,QUJD"
            );

            Ok(LlmResponse {
                content: Some("a screenshot".into()),
                finish_reason: "stop".into(),
                ..Default::default()
            })
        }
        fn model(&self) -> &str {
            "vision"
        }
    }

    let bus = EventBus::new();
    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(VisionProvider)
        .strategy(ReactStrategy::default())
        .build();

    let parts = vec![
        ContentPart::text("what is this?"),
        ContentPart::image_base64("image/png", "QUJD"),
    ];
    bus.publish(Event::new(
        kinds::USER_MESSAGE,
        json!({ "parts": serde_json::to_value(&parts).unwrap() }),
    ))
    .await
    .unwrap();

    agent.cycle().await.unwrap();
    let log = bus.log().await;
    assert!(log
        .iter()
        .any(|e| e.payload.get("content").and_then(|c| c.as_str()) == Some("a screenshot")));
}

#[tokio::test]
async fn token_estimates_are_recorded_for_calibration() {
    use eventage::agent::TokenCalibration;
    init_tracing();

    let bus = EventBus::new();
    let agent = AgentBuilder::new()
        .bus(bus.clone())
        .llm(MockLlmProvider::new(vec![LlmResponse {
            content: Some("hi".into()),
            finish_reason: "stop".into(),
            // Provider says the prompt really cost 900 tokens.
            input_tokens: Some(900),
            ..Default::default()
        }]))
        .system_prompt("You are helpful.")
        .strategy(ReactStrategy::default())
        .build();

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "hello"})))
        .await
        .unwrap();
    agent.cycle().await.unwrap();

    let log = bus.log().await;
    let msg = log
        .iter()
        .find(|e| e.kind == kinds::ASSISTANT_MESSAGE)
        .unwrap();
    let estimated = msg.metadata["llm_estimated_input_tokens"].as_u64().unwrap();
    assert!(estimated > 0, "estimate must be recorded on the event");

    // A calibration fed this log learns the estimator ran low.
    let cal = TokenCalibration::new();
    cal.observe_events(&log);
    assert_eq!(cal.samples(), 1);
    assert!(
        cal.ratio() > 1.0,
        "calibration should learn the estimate was too low, got {}",
        cal.ratio()
    );
}

#[tokio::test]
async fn interrupted_tool_call_is_reconciled_before_the_next_cycle() {
    use eventage::agent::recovery::{reconcile_interrupted_tools, ToolRecovery};
    init_tracing();

    // Simulate a restored log that crashed mid-tool-call.
    let bus = EventBus::new();
    bus.publish(Event::new(
        kinds::USER_MESSAGE,
        json!({"text": "deploy it"}),
    ))
    .await
    .unwrap();
    bus.publish(Event::new(
        kinds::ASSISTANT_MESSAGE,
        json!({
            "content": null,
            "tool_calls": [{
                "id": "c1", "type": "function",
                "function": { "name": "deploy", "arguments": "{}" }
            }]
        }),
    ))
    .await
    .unwrap();
    bus.publish(Event::new(
        kinds::TOOL_CALL_PROPOSED,
        json!({ "tool_call_id": "c1", "name": "deploy", "arguments": "{}" }),
    ))
    .await
    .unwrap();
    // …crash here: no tool.result was ever written.

    let report = reconcile_interrupted_tools(&bus, &ToolRecovery::new(), None)
        .await
        .unwrap();
    assert_eq!(report.reported, 1);

    // The history is now valid: the model sees a result for every call.
    let log = bus.log().await;
    let messages = eventage::agent::events_to_messages(&log);
    let tool_msgs: Vec<_> = messages
        .iter()
        .filter(|m| m.role == eventage::llm::Role::Tool)
        .collect();
    assert_eq!(tool_msgs.len(), 1);
    assert!(tool_msgs[0]
        .content
        .as_deref()
        .unwrap()
        .contains("UNKNOWN whether it took effect"));
}
