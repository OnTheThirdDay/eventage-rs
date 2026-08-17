//! What survives reopening a session.
//!
//! There are two different histories and they are easy to conflate. The
//! **conversation** is what goes back onto the bus: user messages, assistant
//! messages, tool results — the things that become LLM context. The **record**
//! is everything that was ever written down, including events that were only
//! ever fanned out to observers.
//!
//! `restore_from` deliberately rebuilds only the first, because replaying a
//! streaming delta onto the active branch produces a branch made of message
//! fragments. But a trace wants the second, and seeding it from the bus meant
//! a reopened session came back with an empty trace and, in particular, an
//! empty context panel.
//!
//! Its own test binary because it redirects the state directory,
//! which is process-wide.

use eventage::event::{kinds, meta_keys};
use eventage::sqlite::SqliteEventStore;
use eventage::Event;
use eventage_code::agent::CodingSession;
use eventage_code::config::{ModelConfig, SessionConfig};
use serde_json::json;

#[tokio::test]
async fn a_reopened_session_still_knows_what_it_sent_the_model() {
    let state = tempfile::tempdir().unwrap();
    // SAFETY: this binary holds exactly one test, so nothing else is reading
    // the environment concurrently.
    unsafe { std::env::set_var("EVENTAGE_STATE_DIR", state.path()) };

    let workspace = tempfile::tempdir().unwrap();
    let config = SessionConfig::new(
        workspace.path().to_str().unwrap(),
        ModelConfig::from_env(None),
    );

    // Write a log by hand: a real conversation event, and an assembly record
    // marked ephemeral exactly as `broadcast` marks it.
    // A UUID, because ids become file names and are validated as such.
    let id = "3f2b9c10-5d41-4a2e-9b77-0c1d2e3f4a5b";
    let db = config.state_dir().join(format!("{id}.db"));
    tokio::fs::create_dir_all(config.state_dir()).await.unwrap();

    let mut assembly = Event::new(
        "agent.context.assembled",
        json!({
            "messages": 2,
            "total_tokens": 4034,
            "manifest": [{ "index": 0, "role": "system", "tokens": 4034,
                           "source": "system", "text": "the system prefix" }],
        }),
    );
    assembly
        .metadata
        .insert(meta_keys::EPHEMERAL.to_string(), json!(true));

    {
        let store = SqliteEventStore::new(&db).await.unwrap();
        store
            .append(&Event::new(kinds::USER_MESSAGE, json!({ "text": "go" })))
            .await
            .unwrap();
        store.append(&assembly).await.unwrap();
    }

    let session = CodingSession::resume(id, config, None).await.unwrap();

    // The conversation excludes it, and must keep excluding it — that is the
    // fix for deltas being resurrected as orphan history.
    let conversation = session.bus.log().await;
    assert!(
        !conversation
            .iter()
            .any(|e| e.kind == "agent.context.assembled"),
        "a broadcast must not come back onto the branch"
    );

    // The record includes it, which is what the trace and the context panel
    // are seeded from.
    let recorded = session.history();
    assert!(
        recorded.iter().any(|e| e.kind == "agent.context.assembled"),
        "the assembly should survive reopening: {:?}",
        recorded.iter().map(|e| &e.kind).collect::<Vec<_>>()
    );
    assert!(
        recorded.iter().any(|e| e.kind == kinds::USER_MESSAGE),
        "and so should the conversation"
    );
}
