//! Two edits to one file, at the same time.
//!
//! The ReAct loop runs up to four tools concurrently, which is right for
//! searches and wrong for edits. Each `edit_file` read the file, applied its
//! replacement and wrote the result back with nothing in between, so two of
//! them against the same path both read the same original and the second
//! write discarded the first — no error, no warning, the change simply gone.

use eventage::agent::Tool;
use eventage_code::lsp::LspPool;
use eventage_code::tools;
use eventage_code::workspace::Workspace;
use serde_json::json;
use std::sync::Arc;

fn editor(ws: &Arc<Workspace>) -> tools::EditFile {
    tools::EditFile {
        ws: Arc::clone(ws),
        client: None,
        lsp: Arc::new(LspPool::new(ws.root())),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_edits_to_one_file_all_survive() {
    // Repeated, because the loss depends on how the two tasks interleave:
    // one run proves nothing, twenty runs of a broken version lose an edit.
    for attempt in 0..20 {
        let dir = tempfile::tempdir().unwrap();
        let ws = Arc::new(Workspace::open(dir.path()).unwrap());
        std::fs::write(dir.path().join("lib.rs"), "alpha\nbeta\ngamma\n").unwrap();

        let (e1, e2, e3) = (editor(&ws), editor(&ws), editor(&ws));
        let (a, b, c) = tokio::join!(
            e1.execute(json!({
                "path": "lib.rs", "old_string": "alpha", "new_string": "ALPHA"
            })),
            e2.execute(json!({
                "path": "lib.rs", "old_string": "beta", "new_string": "BETA"
            })),
            e3.execute(json!({
                "path": "lib.rs", "old_string": "gamma", "new_string": "GAMMA"
            })),
        );
        assert!(a.is_ok() && b.is_ok() && c.is_ok(), "{a:?} {b:?} {c:?}");

        let final_text = std::fs::read_to_string(dir.path().join("lib.rs")).unwrap();
        assert_eq!(
            final_text, "ALPHA\nBETA\nGAMMA\n",
            "attempt {attempt}: an edit was lost — got {final_text:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_write_cannot_land_in_the_middle_of_an_edit() {
    for _ in 0..20 {
        let dir = tempfile::tempdir().unwrap();
        let ws = Arc::new(Workspace::open(dir.path()).unwrap());
        std::fs::write(dir.path().join("a.txt"), "keep\n").unwrap();

        let write = tools::WriteFile {
            ws: Arc::clone(&ws),
            client: None,
            lsp: Arc::new(LspPool::new(ws.root())),
        };
        let edit = editor(&ws);
        let (_, _) = tokio::join!(
            edit.execute(json!({
                "path": "a.txt", "old_string": "keep", "new_string": "EDITED"
            })),
            write.execute(json!({ "path": "a.txt", "content": "REPLACED\n" })),
        );

        // Either order is legitimate; what must not happen is a blend, or a
        // result that reflects neither operation.
        let text = std::fs::read_to_string(dir.path().join("a.txt")).unwrap();
        assert!(
            text == "REPLACED\n" || text == "EDITED\n",
            "the file ended up in a state neither tool produced: {text:?}"
        );
    }
}

/// Different files have nothing to wait for.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn edits_to_different_files_still_run_together() {
    let dir = tempfile::tempdir().unwrap();
    let ws = Arc::new(Workspace::open(dir.path()).unwrap());
    for name in ["a.rs", "b.rs"] {
        std::fs::write(dir.path().join(name), "old\n").unwrap();
    }

    let (e1, e2) = (editor(&ws), editor(&ws));
    let (a, b) = tokio::join!(
        e1.execute(json!({ "path": "a.rs", "old_string": "old", "new_string": "new" })),
        e2.execute(json!({ "path": "b.rs", "old_string": "old", "new_string": "new" })),
    );
    assert!(a.is_ok() && b.is_ok());
    for name in ["a.rs", "b.rs"] {
        assert_eq!(
            std::fs::read_to_string(dir.path().join(name)).unwrap(),
            "new\n"
        );
    }
}

/// Two multi-file operations naming the same pair in opposite orders.
///
/// The classic ABBA deadlock: one takes `x` then wants `y`, the other holds
/// `y` and wants `x`, and neither can go on. `lock_paths` sorts, so both ask
/// in the same order however the caller listed them. Run as separate tasks,
/// because each has to be able to finish and release.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn multi_file_tools_locking_the_same_pair_do_not_deadlock() {
    let dir = tempfile::tempdir().unwrap();
    let ws = Arc::new(Workspace::open(dir.path()).unwrap());

    let mut runs = Vec::new();
    for order in [["x.rs", "y.rs"], ["y.rs", "x.rs"]] {
        let ws = Arc::clone(&ws);
        runs.push(tokio::spawn(async move {
            for _ in 0..50 {
                let guard = ws.lock_paths(&order).await;
                tokio::task::yield_now().await;
                drop(guard);
            }
        }));
    }

    let finished = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        for run in runs {
            run.await.unwrap();
        }
    })
    .await;
    assert!(
        finished.is_ok(),
        "locking the same paths in opposite orders deadlocked"
    );
}

// ── writes against a file something else changed ──────────────────────────────

fn writer(ws: &Arc<Workspace>) -> tools::WriteFile {
    tools::WriteFile {
        ws: Arc::clone(ws),
        client: None,
        lsp: Arc::new(LspPool::new(ws.root())),
    }
}

#[tokio::test]
async fn overwriting_a_file_that_changed_since_it_was_read_is_refused() {
    // The sequence that loses work: read it, something else edits it, write
    // the whole thing back from what you read. Per-path locks cannot help —
    // the other writer is a shell command or the user's editor.
    let dir = tempfile::tempdir().unwrap();
    let ws = Arc::new(Workspace::open(dir.path()).unwrap());
    std::fs::write(dir.path().join("notes.md"), "original\n").unwrap();

    // The agent reads it…
    let read = tools::ReadFile {
        ws: Arc::clone(&ws),
        client: None,
    };
    read.execute(json!({ "path": "notes.md" })).await.unwrap();

    // …someone else changes it…
    std::fs::write(dir.path().join("notes.md"), "someone else's work\n").unwrap();

    // …and the agent writes back what it computed from the old contents.
    let err = writer(&ws)
        .execute(json!({ "path": "notes.md", "content": "agent's version\n" }))
        .await
        .unwrap_err()
        .to_string();

    assert!(err.contains("changed since you last read it"), "{err}");
    assert!(
        err.contains("Read it again"),
        "it must say what to do: {err}"
    );
    assert_eq!(
        std::fs::read_to_string(dir.path().join("notes.md")).unwrap(),
        "someone else's work\n",
        "the other change was discarded"
    );

    // Re-reading clears it: the agent has now seen what is really there.
    read.execute(json!({ "path": "notes.md" })).await.unwrap();
    assert!(writer(&ws)
        .execute(json!({ "path": "notes.md", "content": "agent's version\n" }))
        .await
        .is_ok());
}

#[tokio::test]
async fn ordinary_writes_are_not_obstructed() {
    let dir = tempfile::tempdir().unwrap();
    let ws = Arc::new(Workspace::open(dir.path()).unwrap());

    // A file we never read: overwriting one is an ordinary thing to do.
    std::fs::write(dir.path().join("generated.rs"), "stale\n").unwrap();
    assert!(writer(&ws)
        .execute(json!({ "path": "generated.rs", "content": "fresh\n" }))
        .await
        .is_ok());

    // A new file.
    assert!(writer(&ws)
        .execute(json!({ "path": "new.rs", "content": "x\n" }))
        .await
        .is_ok());

    // And writing twice in a row, which is our own change both times.
    assert!(writer(&ws)
        .execute(json!({ "path": "new.rs", "content": "y\n" }))
        .await
        .is_ok());
}

#[tokio::test]
async fn an_edit_is_not_blocked_by_an_unrelated_change_to_the_same_file() {
    // Why `edit_file` deliberately does not use the whole-file check: a
    // formatter touching another part of the file must not stop an edit whose
    // own anchor still matches.
    let dir = tempfile::tempdir().unwrap();
    let ws = Arc::new(Workspace::open(dir.path()).unwrap());
    std::fs::write(dir.path().join("lib.rs"), "fn a(){}\nfn target(){}\n").unwrap();

    tools::ReadFile {
        ws: Arc::clone(&ws),
        client: None,
    }
    .execute(json!({ "path": "lib.rs" }))
    .await
    .unwrap();

    // Something reformats a different function.
    std::fs::write(dir.path().join("lib.rs"), "fn a() {}\nfn target(){}\n").unwrap();

    assert!(
        editor(&ws)
            .execute(json!({
                "path": "lib.rs", "old_string": "fn target(){}", "new_string": "fn renamed(){}"
            }))
            .await
            .is_ok(),
        "a still-valid edit was refused because of an unrelated change"
    );
    let text = std::fs::read_to_string(dir.path().join("lib.rs")).unwrap();
    assert!(
        text.contains("fn a() {}"),
        "the other change was lost: {text}"
    );
    assert!(text.contains("fn renamed(){}"), "{text}");
}
