#![cfg(feature = "observability")]
use eventage::observability::{BusObserver, JsonlExporter};
use eventage::{kinds, Event, EventBus};
use serde_json::json;
use tokio::io::AsyncBufReadExt;

#[tokio::test]
async fn jsonl_exporter_writes_events_to_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("events.jsonl");

    let bus = EventBus::new();
    let exporter = JsonlExporter::new(&path).await.unwrap();
    let observer = BusObserver::new(bus.clone()).add_exporter(exporter);

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "hello"})))
        .await
        .unwrap();
    bus.publish(Event::new(
        kinds::ASSISTANT_MESSAGE,
        json!({"content": "hi", "tool_calls": []}),
    ))
    .await
    .unwrap();

    observer.export_snapshot().await.unwrap();

    let file = tokio::fs::File::open(&path).await.unwrap();
    let mut lines = tokio::io::BufReader::new(file).lines();

    let line1 = lines.next_line().await.unwrap().expect("expected line 1");
    let line2 = lines.next_line().await.unwrap().expect("expected line 2");

    let ev1: serde_json::Value = serde_json::from_str(&line1).unwrap();
    let ev2: serde_json::Value = serde_json::from_str(&line2).unwrap();

    assert_eq!(ev1["kind"].as_str().unwrap(), kinds::USER_MESSAGE);
    assert_eq!(ev1["payload"]["text"].as_str().unwrap(), "hello");
    assert_eq!(ev2["kind"].as_str().unwrap(), kinds::ASSISTANT_MESSAGE);
    assert_eq!(ev2["payload"]["content"].as_str().unwrap(), "hi");
}

#[tokio::test]
async fn bus_observer_background_task_captures_live_events() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("live.jsonl");

    let bus = EventBus::new();
    let exporter = JsonlExporter::new(&path).await.unwrap();
    let observer = BusObserver::new(bus.clone()).add_exporter(exporter);

    let handle = tokio::spawn(async move { observer.run().await });

    tokio::time::sleep(std::time::Duration::from_millis(5)).await;

    bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "live"})))
        .await
        .unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    handle.abort();
    let _ = handle.await;

    let content = tokio::fs::read_to_string(&path).await.unwrap();
    let ev: serde_json::Value = serde_json::from_str(content.trim()).unwrap();
    assert_eq!(ev["kind"].as_str().unwrap(), kinds::USER_MESSAGE);
}
