//! What a rewind undoes, and what it leaves behind.
//!
//! Rewinding is a graph operation on the conversation. It used to leave the
//! files the rewound turns wrote exactly where they were, and anyone watching
//! a turn vanish from the transcript reasonably reads that as an undo. In a
//! git repository the tree is now put back as well; anywhere else the session
//! says so rather than letting the silence imply otherwise.

use eventage::event::{kinds, Event};
use eventage_code::acp::wire::ContentBlock;
use eventage_code::agent::{CodingSession, WORKING_TREE_RESTORED, WORKING_TREE_UNCHANGED};
use eventage_code::config::{ModelConfig, Provider, SessionConfig};
use serde_json::json;

async fn session(state: &tempfile::TempDir, workspace: &tempfile::TempDir) -> CodingSession {
    // SAFETY: set before the session is built; this test binary runs alone
    // against the environment.
    unsafe {
        std::env::set_var("EVENTAGE_STATE_DIR", state.path());
        std::env::set_var("OPENAI_API_KEY", "test");
    }
    let mut config = SessionConfig::new(
        workspace.path().to_str().unwrap(),
        ModelConfig::from_env(Some("test-model".into())),
    );
    config.model.provider = Provider::OpenAiChat;
    CodingSession::create("3f2b91cc-0d54-42a7-9a1e-77c5b0e4d318".into(), config, None)
        .await
        .unwrap()
}

#[tokio::test]
async fn a_rewind_names_the_files_it_did_not_revert() {
    let state = tempfile::tempdir().unwrap();
    let workspace = tempfile::tempdir().unwrap();
    let session = session(&state, &workspace).await;

    let mut notices = session.bus.subscribe();

    session
        .submit_prompt(&[ContentBlock::text("edit something")])
        .await
        .unwrap();
    // The shape every editing tool reports, and what the editor already reads
    // to draw its diff card.
    session
        .bus
        .publish(Event::new(
            kinds::TOOL_RESULT,
            json!({
                "tool_call_id": "c1",
                "name": "write_file",
                "result": {
                    "path": "src/lib.rs",
                    "_diff": { "path": "/repo/src/lib.rs", "old_text": "", "new_text": "x" }
                }
            }),
        ))
        .await
        .unwrap();

    assert_eq!(session.rewind(1).await.unwrap(), 0);

    let notice = loop {
        let event = notices.recv().await.expect("the bus stayed open");
        if event.kind == WORKING_TREE_UNCHANGED {
            break event;
        }
    };
    assert_eq!(
        notice.payload["paths"],
        json!(["/repo/src/lib.rs"]),
        "the rewind said nothing about the file it left changed"
    );

    // A note to the person watching, not a fact for the model's next context.
    assert!(session
        .bus
        .log()
        .await
        .iter()
        .all(|e| e.kind != WORKING_TREE_UNCHANGED));
}

#[tokio::test]
async fn a_rewind_that_changed_no_files_says_nothing() {
    let state = tempfile::tempdir().unwrap();
    let workspace = tempfile::tempdir().unwrap();
    let session = session(&state, &workspace).await;

    session
        .submit_prompt(&[ContentBlock::text("just talk")])
        .await
        .unwrap();
    session
        .bus
        .publish(Event::new(
            kinds::ASSISTANT_MESSAGE,
            json!({ "content": "hello" }),
        ))
        .await
        .unwrap();

    let mut notices = session.bus.subscribe();
    session.rewind(1).await.unwrap();

    // Nothing was written, so there is nothing to warn about — a notice on
    // every rewind would train people to ignore it. A sentinel published
    // afterwards bounds how long to look: the rewind's own events are all
    // ahead of it on the same channel.
    session
        .bus
        .publish(Event::new(kinds::SYSTEM_MESSAGE, json!({ "content": "x" })))
        .await
        .unwrap();
    loop {
        let event = notices.recv().await.unwrap();
        assert_ne!(
            event.kind, WORKING_TREE_UNCHANGED,
            "warned about a rewind that changed nothing"
        );
        if event.kind == kinds::SYSTEM_MESSAGE {
            break;
        }
    }
}

async fn git(root: &std::path::Path, args: &[&str]) -> bool {
    tokio::process::Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .await
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[tokio::test]
async fn a_rewind_puts_the_working_tree_back() {
    let state = tempfile::tempdir().unwrap();
    let workspace = tempfile::tempdir().unwrap();
    let root = workspace.path().to_path_buf();

    if !git(&root, &["init", "-q"]).await {
        return; // No git on this machine.
    }
    git(&root, &["config", "user.email", "t@example.invalid"]).await;
    git(&root, &["config", "user.name", "Test"]).await;
    std::fs::write(root.join("lib.rs"), "fn a() {}\n").unwrap();
    git(&root, &["add", "-A"]).await;
    git(&root, &["commit", "-qm", "base"]).await;

    // An uncommitted edit the user made before the turn. A rewind has to put
    // this back, not `HEAD` — which is why the snapshot is of the working
    // tree rather than the last commit.
    std::fs::write(root.join("lib.rs"), "fn a() { mine }\n").unwrap();

    let session = session(&state, &workspace).await;
    let mut notices = session.bus.subscribe();
    session
        .submit_prompt(&[ContentBlock::text("edit it")])
        .await
        .unwrap();

    // What the turn did to the tree.
    std::fs::write(root.join("lib.rs"), "fn a() { the agent }\n").unwrap();
    std::fs::write(root.join("added.rs"), "fn b() {}\n").unwrap();

    session.rewind(1).await.unwrap();

    assert_eq!(
        std::fs::read_to_string(root.join("lib.rs")).unwrap(),
        "fn a() { mine }\n",
        "the rewind did not restore the file the turn changed"
    );
    assert!(
        !root.join("added.rs").exists(),
        "a file the turn created survived the rewind"
    );

    let notice = loop {
        let event = notices.recv().await.expect("the bus stayed open");
        if event.kind == WORKING_TREE_RESTORED {
            break event;
        }
        assert_ne!(
            event.kind, WORKING_TREE_UNCHANGED,
            "reported the tree as untouched after restoring it"
        );
    };
    // The undo is itself undoable, and the id to do it with is reported.
    assert!(notice.payload["undo"].is_string());
    let paths: Vec<&str> = notice.payload["paths"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|p| p.as_str())
        .collect();
    assert!(paths.contains(&"lib.rs"), "{paths:?}");
    assert!(paths.contains(&"added.rs"), "{paths:?}");
}
