//! An evicted branch leaves its lesson on the log, not only in memory.

use eventage::event::kinds;
use eventage::llm::MockLlmProvider;
use eventage::{BusConfig, EpitaphStrategy, Event, EventBus};
use serde_json::json;
use std::sync::Arc;

#[tokio::test]
async fn an_evicted_branch_publishes_its_epitaph() {
    // The branch's events are deleted when it is evicted. If the epitaph
    // lives only in an in-memory store it dies with the process, and the one
    // thing left of the attempt is lost on reopening.
    let strategy = Arc::new(EpitaphStrategy::new(Arc::new(MockLlmProvider::with_texts(
        ["Rewriting the lexer broke the date parser tests."],
    ))));
    let bus = EventBus::with_config(BusConfig {
        // One retained branch, so the second rollback evicts the first.
        max_retained_branches: 1,
        eviction_strategy: Arc::clone(&strategy) as Arc<dyn eventage::BranchEvictionStrategy>,
        ..Default::default()
    });
    strategy.publish_to(bus.clone());

    let mut seen = bus.subscribe();

    for attempt in 0..2 {
        bus.publish(Event::new(kinds::USER_MESSAGE, json!({ "text": "go" })))
            .await
            .unwrap();
        let anchor = bus.checkpoint().await.unwrap();
        bus.publish(Event::new(
            kinds::ASSISTANT_MESSAGE,
            json!({ "content": format!("attempt {attempt}") }),
        ))
        .await
        .unwrap();
        bus.rollback(anchor).await.unwrap();
    }

    let epitaph = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            let event = seen.recv().await.expect("the bus stayed open");
            if event.kind == kinds::SYSTEM_EPITAPH {
                return event;
            }
        }
    })
    .await
    .expect("the epitaph was published");

    assert!(epitaph.payload["epitaph"]
        .as_str()
        .unwrap()
        .contains("date parser"));
    assert!(epitaph.payload["events_lost"].as_u64().unwrap() >= 1);

    // Durable, so it survives reopening — unlike the eviction notice, which
    // is ephemeral and carries only counts.
    assert!(bus
        .log()
        .await
        .iter()
        .any(|e| e.kind == kinds::SYSTEM_EPITAPH));
}
