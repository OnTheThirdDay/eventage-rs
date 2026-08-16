//! The claim cowork is built on, exercised end to end.
//!
//! A goal is split, the parts run against independent copies of the folder,
//! and **nothing reaches the user's folder until a result is adopted**. That
//! last property is what makes running several worth doing: if they wrote
//! straight through, the second one to finish would overwrite the first and
//! there would be nothing to compare.

use cowork::session::{CoworkConfig, CoworkSession, Status};
use cowork::steering::Steering;
use eventage::llm::MockLlmProvider;
use std::sync::Arc;

/// A folder with something in it, and a session over it.
async fn session(
    replies: Vec<&str>,
) -> Option<(tempfile::TempDir, tempfile::TempDir, CoworkSession)> {
    let folder = tempfile::tempdir().unwrap();
    std::fs::write(folder.path().join("notes.md"), "the original notes\n").unwrap();

    let state = tempfile::tempdir().unwrap();
    let mut config = CoworkConfig::new(folder.path());
    config.state_dir = state.path().to_path_buf();
    // One at a time, so the mock's fixed replies are handed out predictably.
    config.max_parallel = 1;
    config.steering = Steering::Skip;

    let session = CoworkSession::open(
        "fan-out-test",
        Arc::new(MockLlmProvider::with_texts(replies)),
        config,
    )
    .await
    .ok()?;
    Some((folder, state, session))
}

#[tokio::test]
async fn a_goal_fans_into_workstreams_that_cannot_see_each_other() {
    let plan = r#"[{"title":"summary","brief":"write the summary"},
                   {"title":"index","brief":"write the index"}]"#;
    let Some((folder, _state, session)) = session(vec![plan, "summary done", "index done"]).await
    else {
        return; // no git on this machine
    };

    let streams = session.run("tidy up the notes").await.unwrap();
    assert_eq!(streams.len(), 2, "{streams:?}");
    assert!(
        streams.iter().all(|s| s.status == Status::Done),
        "{streams:?}"
    );

    // Each ran in its own copy, and the user's folder is exactly as it was.
    assert_eq!(
        std::fs::read_to_string(folder.path().join("notes.md")).unwrap(),
        "the original notes\n",
        "a workstream wrote straight into the user's folder"
    );

    // The whole run is on the bus, so a surface that joined late can still
    // render it — that is what makes a session resumable rather than a
    // property of the process that started it.
    let log = session.bus.log().await;
    let kinds: Vec<&str> = log.iter().map(|e| e.kind.as_str()).collect();
    assert!(kinds.contains(&cowork::kinds::GOAL_SET));
    assert!(kinds.contains(&cowork::kinds::PLAN_PROPOSED));
    assert_eq!(
        kinds
            .iter()
            .filter(|k| **k == cowork::kinds::WORKSTREAM_STARTED)
            .count(),
        2
    );
}

#[tokio::test]
async fn an_abandoned_workstream_stays_as_something_already_tried() {
    // The difference from rejecting a diff in Cowork or the Codex app: there,
    // rejecting discards the reasoning with the result.
    let plan = r#"[{"title":"rewrite","brief":"rewrite it"}]"#;
    let Some((_folder, _state, session)) = session(vec![plan, "rewrote it"]).await else {
        return;
    };

    let streams = session.run("rewrite the notes").await.unwrap();
    let id = streams[0].id.clone();

    session
        .seal(&id, "rewriting lost the citations, which are the point")
        .await
        .unwrap();

    let lessons = session.lessons().await;
    assert_eq!(lessons.len(), 1);
    assert!(lessons[0].contains("citations"), "{lessons:?}");

    let sealed = session.workstreams().await;
    assert_eq!(sealed[0].status, Status::Sealed);
    assert!(
        sealed[0].report.is_some(),
        "the account of it survives sealing"
    );
}

#[tokio::test]
async fn reverting_puts_the_folder_back_after_an_adoption() {
    // Adopting is not a one-way door: the base snapshot is still there, so a
    // session that went the wrong way is recoverable in one call.
    let plan = r#"[{"title":"only","brief":"do the thing"}]"#;
    let Some((folder, _state, session)) = session(vec![plan, "done"]).await else {
        return;
    };
    session.run("do the thing").await.unwrap();

    // Stand in for what a workstream would have written, then put it back.
    std::fs::write(folder.path().join("notes.md"), "changed by someone\n").unwrap();
    std::fs::write(folder.path().join("extra.md"), "and this appeared\n").unwrap();

    let restored = session.revert().await.unwrap();
    assert_eq!(restored.len(), 2, "{restored:?}");
    assert_eq!(
        std::fs::read_to_string(folder.path().join("notes.md")).unwrap(),
        "the original notes\n"
    );
    assert!(!folder.path().join("extra.md").exists());
}
