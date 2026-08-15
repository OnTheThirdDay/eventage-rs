//! A session refuses to reopen in a workspace it was not recorded in.
//!
//! Its history is full of paths, diffs and tool results from one checkout.
//! Replaying that against another produces an agent reasoning confidently
//! about files that were never there.
//!
//! Its own binary, like `resume_history`, because it points `XDG_DATA_HOME`
//! at a temp directory and that is process-wide.

use eventage_code::agent::CodingSession;
use eventage_code::config::{ModelConfig, SessionConfig};

#[tokio::test]
async fn a_session_will_not_resume_into_a_different_workspace() {
    let state = tempfile::tempdir().unwrap();
    // SAFETY: one test per binary, so nothing else reads the environment.
    unsafe { std::env::set_var("XDG_DATA_HOME", state.path()) };

    // Two different checkouts that happen to share a name. State is normally
    // namespaced by a digest of the path, which keeps them apart on its own —
    // but a directory left by the older name-only scheme is still honoured so
    // nobody loses their history, and inside one of those the two collide.
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    std::fs::create_dir(first.path().join("repo")).unwrap();
    std::fs::create_dir(second.path().join("repo")).unwrap();
    std::fs::create_dir_all(state.path().join("eventage-code/repo")).unwrap();

    let config_for = |dir: &tempfile::TempDir| {
        SessionConfig::new(
            dir.path().join("repo").to_str().unwrap(),
            ModelConfig::from_env(None),
        )
    };
    let id = "7c4e1a90-2b83-4d55-8e16-9f0a1b2c3d4e";

    // Recorded in the first checkout…
    CodingSession::create(id.to_string(), config_for(&first), None)
        .await
        .unwrap();

    // …and reopened in the other one, whose state directory it shares.
    let message = match CodingSession::resume(id, config_for(&second), None).await {
        Ok(_) => panic!("a session resumed into a workspace it was not recorded in"),
        Err(e) => format!("{e:#}"),
    };
    assert!(message.contains("recorded in"), "{message}");
    assert!(message.contains("refusing to resume"), "{message}");

    // The one it belongs to still opens.
    assert!(CodingSession::resume(id, config_for(&first), None)
        .await
        .is_ok());
}
