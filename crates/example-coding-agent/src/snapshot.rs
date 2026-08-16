//! Content-addressed snapshots of the working tree, and restoring from one.
//!
//! Rewinding used to undo the conversation and leave the files exactly as the
//! agent had written them. Anyone watching a turn disappear from the
//! transcript reads that as an undo, so the gap between the two was a trap
//! rather than a limitation.
//!
//! The mechanism was already here, built for subagent worktrees: stage the
//! working tree through a **separate index**, write it out as a tree, and
//! commit that tree. The user's staging area is never touched, untracked
//! files are picked up, `.gitignore` is still honoured, and the result is a
//! commit id that is not on any branch. This module is that code, extracted,
//! plus the restore half.
//!
//! Two properties worth stating because they bound what this promises:
//!
//! * **The undo is itself undoable.** A rewind snapshots the current tree
//!   before overwriting it and reports the commit id, so work done since the
//!   checkpoint is recoverable with `git checkout <id> -- .` rather than
//!   gone.
//! * **The snapshots are unreferenced**, so `git gc` will eventually collect
//!   them. They are a session-lifetime safety net, not an archive.
//!
//! A workspace that is not a git repository gets no snapshots and no restore.
//! There is no second implementation for that case: reinventing content
//! addressing beside a tool that already does it would be the wrong trade.

use anyhow::{bail, Context, Result};
use std::path::Path;

/// Run a git command in `repo`, returning stdout **verbatim**.
///
/// Bytes rather than a trimmed `String`, because one caller is `git show`
/// reading a file's contents back. Trimming there would silently drop a
/// trailing newline from every restored file, and decoding as UTF-8 would
/// corrupt every binary one — a rewind that quietly rewrites a PNG is worse
/// than a rewind that does nothing.
/// How long any one snapshot command may take.
///
/// A snapshot runs on **every prompt**, so its cost is paid constantly while
/// its benefit — a rewind — is claimed rarely. `git add -A` walks the whole
/// working tree and can invoke repository-configured clean filters and
/// fsmonitor commands, so on a large or unusual repository it is not
/// obviously fast. Without a bound, a turn could simply never start.
const SNAPSHOT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

async fn git_bytes(repo: &Path, index: Option<&Path>, args: &[&str]) -> Result<Vec<u8>> {
    let mut cmd = tokio::process::Command::new("git");
    cmd.args(args)
        .current_dir(repo)
        // The same reasoning as the `git` tool: this runs against a
        // repository somebody else wrote, and it must not hand that
        // repository's hooks this process's environment.
        .env_clear()
        .envs(crate::tools::scrubbed_env())
        .kill_on_drop(true)
        // Neither system nor user configuration applies. Both can define
        // `core.fsmonitor` and clean filters — programs git runs on our
        // behalf, chosen by someone who is not the operator and not us.
        // `.git/config` still applies, and that one is the user's own.
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_OPTIONAL_LOCKS", "0");
    if let Some(index) = index {
        cmd.env("GIT_INDEX_FILE", index);
    }
    let out = tokio::time::timeout(SNAPSHOT_TIMEOUT, cmd.output())
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "git {} did not finish within {}s",
                args.join(" "),
                SNAPSHOT_TIMEOUT.as_secs()
            )
        })?
        .with_context(|| format!("could not run git {}", args.join(" ")))?;
    if !out.status.success() {
        bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(out.stdout)
}

/// As [`git_bytes`], for the commands whose output is an identifier or a list.
async fn git(repo: &Path, index: Option<&Path>, args: &[&str]) -> Result<String> {
    let out = git_bytes(repo, index, args).await?;
    Ok(String::from_utf8_lossy(&out).trim().to_string())
}

/// Is this directory inside a git repository?
pub async fn is_repo(repo: &Path) -> bool {
    git(repo, None, &["rev-parse", "--git-dir"]).await.is_ok()
}

/// Record the working tree as it is now, returning a commit id.
///
/// Built through a scratch index so the user's staging area is untouched:
/// read `HEAD`'s tree into it, stage everything on top — which picks up
/// untracked files while still honouring `.gitignore` — write it out, and
/// commit the result with `HEAD` as its parent. The commit is never on a
/// branch and never pushed.
///
/// One thing it cannot see: a buffer open and unsaved in the user's editor.
/// That would have to come from the client over ACP.
pub async fn capture(repo: &Path) -> Result<String> {
    let id = uuid::Uuid::new_v4().to_string();
    let index = std::env::temp_dir().join(format!("eventage-snap-{}", &id[..8]));
    let scratch = Some(index.as_path());

    // A repository with no commits yet has no HEAD to read or to parent.
    let head = git(repo, None, &["rev-parse", "HEAD"]).await.ok();
    if head.is_some() {
        git(repo, scratch, &["read-tree", "HEAD"]).await?;
    }
    git(repo, scratch, &["add", "-A"]).await?;
    let tree = git(repo, scratch, &["write-tree"]).await?;
    let _ = tokio::fs::remove_file(&index).await;

    let mut args = vec!["commit-tree", &tree, "-m", "eventage snapshot"];
    if let Some(parent) = &head {
        args.push("-p");
        args.push(parent);
    }
    git(repo, None, &args).await
}

/// What a restore did.
#[derive(Debug, Default, Clone)]
pub struct Restored {
    /// Paths brought back to the snapshot's contents.
    pub paths: Vec<String>,
    /// A snapshot of the tree *before* the restore, so it can be undone.
    pub undo: Option<String>,
}

/// Put the working tree back to `target`, reporting what changed.
///
/// Deliberately not `git checkout <target> -- .` or `read-tree -u`: both
/// rewrite the index, and the whole point of building snapshots through a
/// scratch index was to leave the user's staging area alone. Instead the two
/// trees are diffed and only the differing paths are touched — written back
/// from the snapshot, or removed if the snapshot does not have them.
///
/// Ignored files are never involved, because `git add -A` never staged them.
pub async fn restore(repo: &Path, target: &str) -> Result<Restored> {
    // Every write goes through the workspace handle rather than
    // `repo.join(path)` and ambient `tokio::fs`. A tracked file can be
    // replaced by a symlink after the checkpoint was taken, and an ambient
    // write then follows it and puts the old contents somewhere outside the
    // repository entirely. The handle resolves beneath its root and replaces
    // a link rather than writing through it — the same discipline every tool
    // already uses, which restore had simply never been held to.
    let ws = crate::workspace::Workspace::open(repo)
        .with_context(|| format!("cannot open '{}' to restore into", repo.display()))?;

    // Before anything is overwritten. Work done since the checkpoint is then
    // recoverable rather than destroyed, which is the difference between an
    // undo and a loss.
    let undo = capture(repo).await.ok();

    let status = git(
        repo,
        None,
        &[
            "diff",
            "--name-status",
            "-z",
            // Renames would need pairing up; treated as a delete and an add,
            // the restore is identical and the parsing is one field wide.
            "--no-renames",
            target,
            undo.as_deref().unwrap_or("HEAD"),
        ],
    )
    .await?;

    let mut fields = status.split('\0').filter(|f| !f.is_empty());
    let mut paths = Vec::new();
    while let (Some(kind), Some(path)) = (fields.next(), fields.next()) {
        match kind.chars().next() {
            // Present now, absent in the snapshot: the turn created it.
            Some('A') => {
                let _ = ws.remove_file(path).await;
            }
            // Changed or deleted since the snapshot: write the old blob back,
            // byte for byte.
            Some('M' | 'D' | 'T') => {
                let blob = git_bytes(repo, None, &["show", &format!("{target}:{path}")]).await?;
                ws.write(path, blob).await?;
            }
            _ => continue,
        }
        paths.push(path.to_string());
    }

    paths.sort();
    Ok(Restored { paths, undo })
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn repo() -> Option<tempfile::TempDir> {
        let dir = tempfile::tempdir().unwrap();
        git(dir.path(), None, &["init", "-q"]).await.ok()?;
        git(dir.path(), None, &["config", "user.email", "t@e.invalid"])
            .await
            .ok()?;
        git(dir.path(), None, &["config", "user.name", "T"])
            .await
            .ok()?;
        Some(dir)
    }

    #[tokio::test]
    async fn a_restore_puts_back_what_a_turn_changed_and_removes_what_it_added() {
        let Some(dir) = repo().await else { return };
        let root = dir.path();
        std::fs::write(root.join("kept.rs"), "fn a() {}\n").unwrap();
        git(root, None, &["add", "-A"]).await.unwrap();
        git(root, None, &["commit", "-qm", "base"]).await.unwrap();

        // The state a turn starts from, including an uncommitted edit — the
        // reason this snapshots the working tree rather than HEAD.
        std::fs::write(root.join("kept.rs"), "fn a() { original }\n").unwrap();
        let before = capture(root).await.unwrap();

        // What the turn did.
        std::fs::write(root.join("kept.rs"), "fn a() { agent wrote this }\n").unwrap();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/new.rs"), "fn b() {}\n").unwrap();
        std::fs::remove_file(root.join("nothing")).ok();

        let restored = restore(root, &before).await.unwrap();

        assert_eq!(
            std::fs::read_to_string(root.join("kept.rs")).unwrap(),
            "fn a() { original }\n",
            "the uncommitted edit the turn started from was not restored"
        );
        assert!(
            !root.join("src/new.rs").exists(),
            "a file the turn created survived the rewind"
        );
        assert!(restored.paths.contains(&"kept.rs".to_string()));
        assert!(restored.paths.contains(&"src/new.rs".to_string()));

        // The undo is itself undoable.
        let undo = restored.undo.expect("a pre-restore snapshot was recorded");
        let text = git(root, None, &["show", &format!("{undo}:src/new.rs")])
            .await
            .unwrap();
        assert_eq!(text, "fn b() {}");
    }

    #[tokio::test]
    async fn a_restore_that_changes_nothing_reports_nothing() {
        let Some(dir) = repo().await else { return };
        let root = dir.path();
        std::fs::write(root.join("a.rs"), "x\n").unwrap();
        git(root, None, &["add", "-A"]).await.unwrap();
        git(root, None, &["commit", "-qm", "base"]).await.unwrap();

        let before = capture(root).await.unwrap();
        assert!(restore(root, &before).await.unwrap().paths.is_empty());
    }

    #[tokio::test]
    async fn an_ignored_file_is_never_touched() {
        // `git add -A` honours `.gitignore`, so build output is not in the
        // snapshot — and a rewind must not delete someone's `target/`.
        let Some(dir) = repo().await else { return };
        let root = dir.path();
        std::fs::write(root.join(".gitignore"), "build/\n").unwrap();
        git(root, None, &["add", "-A"]).await.unwrap();
        git(root, None, &["commit", "-qm", "base"]).await.unwrap();

        let before = capture(root).await.unwrap();
        std::fs::create_dir_all(root.join("build")).unwrap();
        std::fs::write(root.join("build/out.o"), "binary").unwrap();

        let restored = restore(root, &before).await.unwrap();
        assert!(restored.paths.is_empty(), "{:?}", restored.paths);
        assert!(
            root.join("build/out.o").exists(),
            "ignored output was deleted"
        );
    }

    #[tokio::test]
    async fn a_restore_is_byte_for_byte() {
        // The restore reads blobs back through `git show`. Trimming that
        // output would drop a trailing newline from every text file, and
        // decoding it as UTF-8 would corrupt every binary one.
        let Some(dir) = repo().await else { return };
        let root = dir.path();
        let png: Vec<u8> = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0xff, 0x00];
        std::fs::write(root.join("logo.png"), &png).unwrap();
        std::fs::write(root.join("no-newline.txt"), "last line").unwrap();
        std::fs::write(root.join("blank-lines.txt"), "text\n\n\n").unwrap();
        git(root, None, &["add", "-A"]).await.unwrap();
        git(root, None, &["commit", "-qm", "base"]).await.unwrap();

        let before = capture(root).await.unwrap();
        std::fs::write(root.join("logo.png"), b"clobbered").unwrap();
        std::fs::write(root.join("no-newline.txt"), "changed").unwrap();
        std::fs::write(root.join("blank-lines.txt"), "changed").unwrap();

        restore(root, &before).await.unwrap();

        assert_eq!(std::fs::read(root.join("logo.png")).unwrap(), png);
        assert_eq!(
            std::fs::read_to_string(root.join("no-newline.txt")).unwrap(),
            "last line"
        );
        assert_eq!(
            std::fs::read_to_string(root.join("blank-lines.txt")).unwrap(),
            "text\n\n\n"
        );
    }

    #[tokio::test]
    async fn a_directory_that_is_not_a_repository_is_recognised_as_such() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!is_repo(dir.path()).await);
        assert!(capture(dir.path()).await.is_err());
    }
}
