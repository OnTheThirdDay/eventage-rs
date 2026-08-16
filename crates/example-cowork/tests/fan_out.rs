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

#[tokio::test]
#[cfg(unix)]
async fn adopting_cannot_write_through_a_planted_symlink() {
    // The escape: a tracked file in the live folder is replaced by a symlink
    // after the base snapshot is taken. An ambient `folder.join(path)` write
    // follows it, so adopting a workstream — or reverting a session — puts
    // content wherever the link points. Both paths now go through the
    // workspace capability handle, which replaces a link rather than writing
    // through it.
    use cowork::shadow::Shadow;

    let dir = tempfile::tempdir().unwrap();
    let folder = dir.path().join("folder");
    std::fs::create_dir_all(&folder).unwrap();
    std::fs::write(folder.join("notes.md"), "original\n").unwrap();

    let outside = dir.path().join("outside");
    std::fs::create_dir_all(&outside).unwrap();
    let victim = outside.join("victim.txt");
    std::fs::write(&victim, "do not touch\n").unwrap();

    let Ok((shadow, _)) = Shadow::open(dir.path().join("shadow.git"), &folder).await else {
        return; // no git on this machine
    };
    let base = shadow.snapshot("base").await.unwrap();

    // A workstream edits the file in its own copy.
    let ws = dir.path().join("ws");
    shadow.worktree(&ws, &base).await.unwrap();
    std::fs::write(ws.join("notes.md"), "the workstream's version\n").unwrap();
    let result = shadow.snapshot_tree(&ws, "ws").await.unwrap();

    // Meanwhile the live file becomes a link pointing out of the folder.
    std::fs::remove_file(folder.join("notes.md")).unwrap();
    std::os::unix::fs::symlink(&victim, folder.join("notes.md")).unwrap();

    let _ = shadow.adopt(&result, &base).await;

    assert_eq!(
        std::fs::read_to_string(&victim).unwrap(),
        "do not touch\n",
        "adoption wrote through a symlink and clobbered a file outside the folder"
    );
}

#[tokio::test]
async fn a_goal_from_the_channel_actually_runs() {
    // The HTTP endpoint published a `user.message` that nothing consumed, so
    // it answered "accepted" and dropped the goal; the scheduler did the same
    // with its firing. Both now publish one request kind, and one consumer
    // runs it — so a goal from a phone, a cron entry, or the command line all
    // take the same path.
    use eventage::Event;
    use serde_json::json;
    use std::sync::Arc;

    let plan = r#"[{"title":"only","brief":"do the thing"}]"#;
    let Some((_folder, _state, session)) = session(vec![plan, "did the thing"]).await else {
        return;
    };
    let session = Arc::new(session);

    let mut watching = session.bus.subscribe();
    let requests = tokio::spawn(Arc::clone(&session).serve_requests());

    session
        .bus
        .publish(Event::new(
            cowork::kinds::GOAL_REQUESTED,
            json!({ "goal": "tidy the notes", "source": "http" }),
        ))
        .await
        .unwrap();

    // The run is what proves it: a plan was proposed for the goal we sent.
    let proposed = tokio::time::timeout(std::time::Duration::from_secs(20), async {
        loop {
            let event = watching.recv().await.expect("the bus stayed open");
            if event.kind == cowork::kinds::PLAN_PROPOSED {
                return event;
            }
        }
    })
    .await
    .expect("the requested goal was never run");

    assert_eq!(proposed.payload["goal"], "tidy the notes");
    assert_eq!(session.workstreams().await.len(), 1);
    requests.abort();
}

#[tokio::test]
async fn a_request_with_no_goal_is_ignored_rather_than_run() {
    use eventage::Event;
    use serde_json::json;
    use std::sync::Arc;

    let Some((_folder, _state, session)) = session(vec!["[]"]).await else {
        return;
    };
    let session = Arc::new(session);
    let requests = tokio::spawn(Arc::clone(&session).serve_requests());

    session
        .bus
        .publish(Event::new(
            cowork::kinds::GOAL_REQUESTED,
            json!({ "goal": "   ", "source": "http" }),
        ))
        .await
        .unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    assert!(
        session.workstreams().await.is_empty(),
        "an empty goal started a session's worth of work"
    );
    requests.abort();
}
