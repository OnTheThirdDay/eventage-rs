//! What an implementation subagent sees of the repository.
//!
//! It used to be `HEAD`: the last commit, and nothing the user had done
//! since. A subagent would read an older version of the code, implement
//! against interfaces that had already changed, and return a patch that
//! conflicted with the tree it was written for.

#![cfg(unix)]

use std::path::Path;
use std::process::Command;

fn git(repo: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .expect("git");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// A repository with a commit, then work the user has not committed.
fn repo_with_uncommitted_work() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path();
    git(path, &["init", "-q"]);
    git(path, &["config", "user.email", "t@example.com"]);
    git(path, &["config", "user.name", "t"]);
    std::fs::write(path.join(".gitignore"), "ignored/\n").unwrap();
    std::fs::write(path.join("committed.rs"), "fn old_name() {}\n").unwrap();
    git(path, &["add", "-A"]);
    git(path, &["commit", "-qm", "first"]);

    // Everything below is invisible to a checkout of HEAD.
    std::fs::write(path.join("committed.rs"), "fn new_name() {}\n").unwrap();
    std::fs::write(path.join("staged.rs"), "fn staged() {}\n").unwrap();
    git(path, &["add", "staged.rs"]);
    std::fs::write(path.join("untracked.rs"), "fn untracked() {}\n").unwrap();
    std::fs::create_dir(path.join("ignored")).unwrap();
    std::fs::write(path.join("ignored/junk"), "should not travel\n").unwrap();
    dir
}

#[tokio::test]
async fn a_subagent_sees_the_working_tree_not_the_last_commit() {
    let repo = repo_with_uncommitted_work();
    let before = git(repo.path(), &["status", "--porcelain"]);

    let out = eventage_code::tools::task::snapshot_for_test(repo.path())
        .await
        .expect("snapshot");

    let unstaged = std::fs::read_to_string(out.join("committed.rs")).unwrap();
    assert_eq!(
        unstaged, "fn new_name() {}\n",
        "an unstaged edit was missed"
    );
    assert!(out.join("staged.rs").exists(), "a staged file was missed");
    assert!(
        out.join("untracked.rs").exists(),
        "an untracked file was missed"
    );

    // `.gitignore` still applies: build output and secrets do not travel.
    assert!(
        !out.join("ignored/junk").exists(),
        "an ignored file was copied into the subagent's checkout"
    );

    // And the user's own staging area is exactly as they left it.
    assert_eq!(
        git(repo.path(), &["status", "--porcelain"]),
        before,
        "the snapshot disturbed the user's index"
    );
}

#[tokio::test]
async fn a_diff_from_the_snapshot_shows_only_the_subagents_work() {
    // If the diff were taken against HEAD it would also contain the user's
    // uncommitted work, and the parent would review changes it did not ask
    // for as though the subagent had made them.
    let repo = repo_with_uncommitted_work();
    let out = eventage_code::tools::task::snapshot_for_test(repo.path())
        .await
        .unwrap();

    std::fs::write(out.join("added_by_subagent.rs"), "fn added() {}\n").unwrap();
    let diff = eventage_code::tools::task::diff_for_test(&out).await;

    assert!(diff.contains("added_by_subagent.rs"), "{diff}");
    assert!(
        !diff.contains("untracked.rs"),
        "the user's work leaked into the diff:\n{diff}"
    );
    assert!(
        !diff.contains("new_name"),
        "the user's edit leaked into the diff:\n{diff}"
    );
}

#[tokio::test]
async fn a_repository_with_no_commits_yet_still_works() {
    let dir = tempfile::tempdir().unwrap();
    git(dir.path(), &["init", "-q"]);
    git(dir.path(), &["config", "user.email", "t@example.com"]);
    git(dir.path(), &["config", "user.name", "t"]);
    std::fs::write(dir.path().join("first.rs"), "fn first() {}\n").unwrap();

    let out = eventage_code::tools::task::snapshot_for_test(dir.path())
        .await
        .expect("a fresh repository should still snapshot");
    assert!(out.join("first.rs").exists());
}
