//! A cowork session survives the process that started it.
//!
//! Nothing is kept only in memory: the plan, each stream's identity and
//! outcome, what it changed, why one was abandoned, and the base snapshot
//! they all branch from are published as events because a surface needs them.
//! Reopening reads the same events back. There is no second copy of the state
//! to drift.

use cowork::session::{CoworkConfig, CoworkSession, Status};
use cowork::steering::Steering;
use eventage::llm::MockLlmProvider;
use std::sync::Arc;

const ID: &str = "6f1d0a52-5a3e-4c77-b0a1-2f9e8d4c3b71";

fn config(folder: &std::path::Path, state: &std::path::Path) -> CoworkConfig {
    let mut config = CoworkConfig::new(folder);
    config.state_dir = state.to_path_buf();
    config.max_parallel = 1;
    config.steering = Steering::Auto;
    config
}

#[tokio::test]
async fn a_session_reopens_with_its_workstreams_intact() {
    let folder = tempfile::tempdir().unwrap();
    std::fs::write(folder.path().join("notes.md"), "the original notes\n").unwrap();
    let state = tempfile::tempdir().unwrap();

    let plan = r#"[{"title":"summary","brief":"write the summary"},
                   {"title":"index","brief":"build the index"}]"#;

    let sealed_id = {
        let Ok(session) = CoworkSession::open(
            ID,
            Arc::new(MockLlmProvider::with_texts(vec![
                plan,
                "wrote it",
                "indexed it",
            ])),
            config(folder.path(), state.path()),
        )
        .await
        else {
            return; // no git on this machine
        };

        session.steer(Steering::Skip).await;
        let streams = session.run("tidy the notes").await.unwrap();
        assert_eq!(streams.len(), 2);

        let victim = streams[0].id.clone();
        session
            .seal(&victim, "the summary lost the citations")
            .await
            .unwrap();

        // Flushed before the process is said to be finished; without this the
        // last events — the ones a resume needs — are still in the queue.
        assert_eq!(session.close().await, 0, "events failed to reach the log");
        victim
    };

    // A different session object, as a different process would build.
    let reopened = CoworkSession::resume(
        ID,
        Arc::new(MockLlmProvider::with_texts(vec!["unused"])),
        config(folder.path(), state.path()),
    )
    .await
    .unwrap();

    let streams = reopened.workstreams().await;
    assert_eq!(streams.len(), 2, "{streams:?}");

    // Planned order, not the order they happened to start in.
    assert_eq!(streams[0].title, "summary");
    assert_eq!(streams[1].title, "index");

    // The sealed one came back sealed, with its reason.
    let sealed = streams.iter().find(|s| s.id == sealed_id).unwrap();
    assert_eq!(sealed.status, Status::Sealed);
    assert_eq!(
        sealed.epitaph.as_deref(),
        Some("the summary lost the citations")
    );

    // And the lesson is available to the next plan, which is the whole point
    // of having kept it.
    assert_eq!(reopened.lessons().await.len(), 1);

    // The other finished, with the snapshot it produced still resolvable —
    // referenced snapshots are why `git gc` cannot take them between runs.
    let done = streams.iter().find(|s| s.status == Status::Done).unwrap();
    assert!(done.commit.is_some(), "{done:?}");

    // The steering mode it was left in, not the configured default.
    assert_eq!(reopened.steering(), Steering::Skip);

    reopened.close().await;
}

#[tokio::test]
async fn a_stream_interrupted_mid_turn_does_not_come_back_as_finished() {
    // A crash between `started` and `finished` leaves a stream that never
    // produced anything. Reporting it as done would offer the user a result
    // that is not there.
    use eventage::event::Event;
    use serde_json::json;

    let folder = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();

    {
        let Ok(session) = CoworkSession::open(
            ID,
            Arc::new(MockLlmProvider::with_texts(vec!["[]"])),
            config(folder.path(), state.path()),
        )
        .await
        else {
            return;
        };
        // Exactly what a turn writes before it starts working, and nothing
        // after it — the process died here.
        session
            .bus
            .publish(Event::new(
                cowork::kinds::PLAN_PROPOSED,
                json!({ "goal": "g", "base": "deadbeef",
                        "workstreams": [{ "title": "half", "brief": "b" }] }),
            ))
            .await
            .unwrap();
        session
            .bus
            .publish(Event::new(
                cowork::kinds::WORKSTREAM_STARTED,
                json!({ "id": "aa11", "title": "half", "brief": "b" }),
            ))
            .await
            .unwrap();
        session.close().await;
    }

    let reopened = CoworkSession::resume(
        ID,
        Arc::new(MockLlmProvider::with_texts(vec!["unused"])),
        config(folder.path(), state.path()),
    )
    .await
    .unwrap();

    let streams = reopened.workstreams().await;
    assert_eq!(streams.len(), 1);
    assert_eq!(streams[0].status, Status::Failed);
    assert!(streams[0].commit.is_none());
    assert!(streams[0]
        .report
        .as_deref()
        .unwrap_or_default()
        .contains("interrupted"));
    reopened.close().await;
}

#[tokio::test]
async fn a_session_id_cannot_be_a_path() {
    // The id becomes a directory name directly beneath the state directory.
    let folder = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let opened = CoworkSession::open(
        "../../etc/passwd",
        Arc::new(MockLlmProvider::with_texts(vec!["[]"])),
        config(folder.path(), state.path()),
    )
    .await;
    let Err(err) = opened else {
        panic!("a path was accepted as a session id");
    };
    assert!(err.to_string().contains("not a valid session id"), "{err}");
}

#[tokio::test]
async fn a_lesson_survives_into_the_next_round() {
    // `run` replaced the workstream list wholesale, so a stream sealed in one
    // round was gone by the next and `lessons()` came back empty — the
    // feedback worked within a round and silently stopped at the boundary.
    let folder = tempfile::tempdir().unwrap();
    std::fs::write(folder.path().join("notes.md"), "notes\n").unwrap();
    let state = tempfile::tempdir().unwrap();

    let one = r#"[{"title":"first attempt","brief":"try it this way"}]"#;
    let two = r#"[{"title":"second attempt","brief":"try it another way"}]"#;
    let Ok(session) = CoworkSession::open(
        ID,
        Arc::new(MockLlmProvider::with_texts(vec![
            one,
            "did it",
            two,
            "did it again",
        ])),
        config(folder.path(), state.path()),
    )
    .await
    else {
        return;
    };

    let first = session.run("tidy the notes").await.unwrap();
    session
        .seal(&first[0].id, "that approach lost the citations")
        .await
        .unwrap();
    assert_eq!(session.lessons().await.len(), 1);

    // A second round, in the same session.
    session.run("tidy the notes differently").await.unwrap();

    let lessons = session.lessons().await;
    assert_eq!(lessons.len(), 1, "the lesson was dropped between rounds");
    assert!(lessons[0].contains("citations"), "{lessons:?}");

    // The sealed one is still listed; the finished one from round one is not,
    // because round two took a fresh base and its diff no longer applies.
    let streams = session.workstreams().await;
    assert!(streams.iter().any(|s| s.status == Status::Sealed));
    assert!(streams.iter().any(|s| s.title == "second attempt"));
    assert!(!streams
        .iter()
        .any(|s| s.title == "first attempt" && s.status == Status::Done));

    session.close().await;
}
