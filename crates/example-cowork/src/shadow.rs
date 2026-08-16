//! A git repository *for* a folder, kept outside it.
//!
//! Cowork's differentiator is that a working session is a graph you can
//! navigate: fan a goal into parallel workstreams, compare what each produced,
//! keep one, and leave the others in the history as evidence. Codex gets that
//! for code by using git worktrees, which works because the thing being worked
//! on is already a repository.
//!
//! A folder of documents is not. So one is made for it, and kept somewhere
//! else: the git directory lives under the session's state directory and the
//! folder is passed as the work tree. Nothing is written into the user's
//! folder — no `.git`, not even the file that `--separate-git-dir` leaves —
//! and content addressing, cheap snapshots, worktrees and diffs all follow.
//!
//! Three things this had to get right, each found by trying it:
//!
//! * **A nested repository breaks `add -A` outright** — "does not have a
//!   commit checked out", and the whole snapshot fails. A documents folder
//!   with a checkout in it is completely ordinary, so nested repositories are
//!   discovered and excluded, and [`Shadow::open`] reports which. That is also
//!   the right answer on the merits: a repository inside the folder has its
//!   own history and its own undo, and is not cowork's to snapshot.
//! * **Snapshots must be referenced.** The coding agent's are not, so `git gc`
//!   may collect them. Here each one gets a ref under `refs/cowork/`, which
//!   keeps it alive and makes the session's branches enumerable with
//!   `for-each-ref`.
//! * **Blobs are read as bytes.** A folder of real work is full of `.xlsx` and
//!   `.png`; reading them back through a `String` would corrupt every one.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// How deep to look for nested repositories, and how many to tolerate.
///
/// A bound rather than an unbounded walk: `open` runs on every session start,
/// and a folder pointed at `$HOME` should degrade rather than hang.
const MAX_SCAN_DEPTH: usize = 6;
const MAX_SCAN_ENTRIES: usize = 50_000;

/// Directories never worth snapshotting, whatever else is in the folder.
///
/// Build output and dependency trees are large, regenerable, and not the work.
const ALWAYS_EXCLUDED: &[&str] = &[
    "node_modules/",
    "target/",
    ".venv/",
    "venv/",
    "__pycache__/",
    ".DS_Store",
    ".cowork/",
];

/// A git repository that tracks a folder from outside it.
pub struct Shadow {
    git_dir: PathBuf,
    folder: PathBuf,
}

/// One file's fate between two snapshots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileChange {
    pub path: String,
    pub status: ChangeStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeStatus {
    Added,
    Modified,
    Deleted,
}

impl ChangeStatus {
    fn parse(field: &str) -> Option<Self> {
        match field.chars().next()? {
            'A' => Some(Self::Added),
            'M' | 'T' => Some(Self::Modified),
            'D' => Some(Self::Deleted),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Added => "added",
            Self::Modified => "modified",
            Self::Deleted => "deleted",
        }
    }
}

/// What [`Shadow::open`] decided not to track.
#[derive(Debug, Clone, Default)]
pub struct Excluded {
    /// Nested git repositories, relative to the folder.
    pub repositories: Vec<String>,
    /// `true` when the scan hit its bound and may have missed some.
    pub scan_truncated: bool,
}

impl Shadow {
    /// Open — creating if absent — the shadow repository for `folder`.
    ///
    /// Returns what it decided not to track, so the caller can say so rather
    /// than leave the user to discover that a rewind skipped part of their
    /// folder.
    pub async fn open(
        git_dir: impl Into<PathBuf>,
        folder: impl Into<PathBuf>,
    ) -> Result<(Self, Excluded)> {
        let git_dir = git_dir.into();
        let folder = folder
            .into()
            .canonicalize()
            .context("the folder to work in does not exist")?;

        if !git_dir.join("HEAD").exists() {
            tokio::fs::create_dir_all(&git_dir).await.ok();
            let out = tokio::process::Command::new("git")
                .args(["init", "-q", "--bare"])
                .arg(&git_dir)
                .env_clear()
                .envs(eventage_code::tools::scrubbed_env())
                .output()
                .await
                .context("git is required for cowork sessions but could not be run")?;
            if !out.status.success() {
                bail!(
                    "could not create the shadow repository: {}",
                    String::from_utf8_lossy(&out.stderr).trim()
                );
            }
        }

        let shadow = Self { git_dir, folder };
        let excluded = shadow.write_excludes().await?;
        Ok((shadow, excluded))
    }

    /// The folder this shadow tracks.
    pub fn folder(&self) -> &Path {
        &self.folder
    }

    /// Run git against the shadow, returning stdout verbatim.
    ///
    /// Bytes, because one caller reads file contents back — see the module
    /// docs. `work_tree` selects which checkout the command applies to, so the
    /// same shadow serves the folder and every workstream's worktree.
    async fn git_bytes(
        &self,
        work_tree: Option<&Path>,
        index: Option<&Path>,
        args: &[&str],
    ) -> Result<Vec<u8>> {
        let mut cmd = tokio::process::Command::new("git");
        cmd.arg("--git-dir")
            .arg(&self.git_dir)
            .args(args)
            .env_clear()
            .envs(eventage_code::tools::scrubbed_env())
            // The folder may contain a repository whose config would
            // otherwise be consulted, and it is not ours.
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_AUTHOR_NAME", "cowork")
            .env("GIT_AUTHOR_EMAIL", "cowork@localhost")
            .env("GIT_COMMITTER_NAME", "cowork")
            .env("GIT_COMMITTER_EMAIL", "cowork@localhost");
        if let Some(tree) = work_tree {
            cmd.env("GIT_WORK_TREE", tree);
        }
        if let Some(index) = index {
            cmd.env("GIT_INDEX_FILE", index);
        }
        let out = cmd
            .output()
            .await
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

    /// As [`git_bytes`](Self::git_bytes), for output that is an id or a list.
    async fn git(
        &self,
        work_tree: Option<&Path>,
        index: Option<&Path>,
        args: &[&str],
    ) -> Result<String> {
        let out = self.git_bytes(work_tree, index, args).await?;
        Ok(String::from_utf8_lossy(&out).trim().to_string())
    }

    /// Find nested repositories and write the exclude file.
    ///
    /// `info/exclude` rather than a `.gitignore`, because a `.gitignore` would
    /// be a file written into the user's folder — the one thing this whole
    /// arrangement exists to avoid.
    async fn write_excludes(&self) -> Result<Excluded> {
        let mut found = Excluded::default();
        let mut lines: Vec<String> = ALWAYS_EXCLUDED.iter().map(|s| s.to_string()).collect();

        let folder = self.folder.clone();
        let scan = tokio::task::spawn_blocking(move || scan_for_repositories(&folder))
            .await
            .context("scanning the folder panicked")?;
        found.repositories = scan.0;
        found.scan_truncated = scan.1;

        for repo in &found.repositories {
            // Anchored and directory-only, so a file of the same name
            // elsewhere is unaffected.
            lines.push(format!("/{repo}/"));
        }

        let info = self.git_dir.join("info");
        tokio::fs::create_dir_all(&info).await.ok();
        tokio::fs::write(info.join("exclude"), lines.join("\n") + "\n")
            .await
            .context("could not write the shadow exclude file")?;
        Ok(found)
    }

    /// Record the folder as it is now, under `refs/cowork/<label>`.
    ///
    /// Referenced deliberately: an unreferenced commit is collectable, and a
    /// session that offers to go back to a snapshot has to still have it.
    pub async fn snapshot(&self, label: &str) -> Result<String> {
        self.snapshot_tree(&self.folder.clone(), label).await
    }

    /// Record an arbitrary checkout — a workstream's worktree — as a commit.
    pub async fn snapshot_tree(&self, work_tree: &Path, label: &str) -> Result<String> {
        let index = std::env::temp_dir().join(format!("cowork-idx-{}", uuid::Uuid::new_v4()));
        let scratch = Some(index.as_path());

        self.git(Some(work_tree), scratch, &["add", "-A"]).await?;
        let tree = self.git(Some(work_tree), scratch, &["write-tree"]).await?;
        let _ = tokio::fs::remove_file(&index).await;

        let commit = self
            .git(None, None, &["commit-tree", &tree, "-m", label])
            .await?;
        self.git(
            None,
            None,
            &[
                "update-ref",
                &format!("refs/cowork/{}", ref_safe(label)),
                &commit,
            ],
        )
        .await?;
        Ok(commit)
    }

    /// Check out `from` into its own directory, for a workstream to work in.
    ///
    /// Detached, so no branch is created and none can be orphaned.
    pub async fn worktree(&self, path: &Path, from: &str) -> Result<()> {
        self.git(
            None,
            None,
            &[
                "worktree",
                "add",
                "-q",
                "--detach",
                &path.display().to_string(),
                from,
            ],
        )
        .await?;
        Ok(())
    }

    /// Remove a workstream's checkout.
    pub async fn remove_worktree(&self, path: &Path) -> Result<()> {
        self.git(
            None,
            None,
            &["worktree", "remove", "--force", &path.display().to_string()],
        )
        .await?;
        Ok(())
    }

    /// What changed between two snapshots.
    pub async fn diff(&self, from: &str, to: &str) -> Result<Vec<FileChange>> {
        let out = self
            .git(
                None,
                None,
                &["diff", "--name-status", "-z", "--no-renames", from, to],
            )
            .await?;
        let mut fields = out.split('\0').filter(|f| !f.is_empty());
        let mut changes = Vec::new();
        while let (Some(status), Some(path)) = (fields.next(), fields.next()) {
            if let Some(status) = ChangeStatus::parse(status) {
                changes.push(FileChange {
                    path: path.to_string(),
                    status,
                });
            }
        }
        changes.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(changes)
    }

    /// A file's contents at a snapshot, byte for byte.
    pub async fn show(&self, commit: &str, path: &str) -> Result<Vec<u8>> {
        self.git_bytes(None, None, &["show", &format!("{commit}:{path}")])
            .await
    }

    /// Put the folder back to `to`, returning what was touched.
    ///
    /// Only the differing paths are written, and the index is never involved —
    /// there is no index to speak of here, but the same discipline keeps the
    /// operation minimal and auditable.
    pub async fn restore(&self, to: &str) -> Result<Vec<FileChange>> {
        let now = self.snapshot("restore-point").await?;
        let changes = self.diff(to, &now).await?;
        for change in &changes {
            let full = self.folder.join(&change.path);
            match change.status {
                // Present now, absent in the target: created since.
                ChangeStatus::Added => {
                    let _ = tokio::fs::remove_file(&full).await;
                }
                ChangeStatus::Modified | ChangeStatus::Deleted => {
                    let blob = self.show(to, &change.path).await?;
                    if let Some(parent) = full.parent() {
                        tokio::fs::create_dir_all(parent).await.ok();
                    }
                    tokio::fs::write(&full, blob).await?;
                }
            }
        }
        Ok(changes)
    }

    /// Apply one workstream's result to the folder.
    ///
    /// The workstream worked in its own checkout; this brings its changes
    /// across, and reports them so the caller can show what landed.
    pub async fn adopt(&self, workstream_commit: &str, base: &str) -> Result<Vec<FileChange>> {
        let changes = self.diff(base, workstream_commit).await?;
        for change in &changes {
            let full = self.folder.join(&change.path);
            match change.status {
                ChangeStatus::Deleted => {
                    let _ = tokio::fs::remove_file(&full).await;
                }
                ChangeStatus::Added | ChangeStatus::Modified => {
                    let blob = self.show(workstream_commit, &change.path).await?;
                    if let Some(parent) = full.parent() {
                        tokio::fs::create_dir_all(parent).await.ok();
                    }
                    tokio::fs::write(&full, blob).await?;
                }
            }
        }
        Ok(changes)
    }
}

/// Paths under `root` that are themselves git repositories.
///
/// Bounded in depth and entries, because this runs at session start and the
/// folder is whatever the user pointed at. Returns the paths and whether the
/// walk gave up early.
fn scan_for_repositories(root: &Path) -> (Vec<String>, bool) {
    let mut found = Vec::new();
    let mut seen = 0usize;
    let mut truncated = false;
    let mut queue: Vec<(PathBuf, usize)> = vec![(root.to_path_buf(), 0)];

    while let Some((dir, depth)) = queue.pop() {
        if depth > MAX_SCAN_DEPTH {
            truncated = true;
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            seen += 1;
            if seen > MAX_SCAN_ENTRIES {
                return (found, true);
            }
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == ".git" {
                // The repository is the parent, recorded relative to the root.
                if let Ok(rel) = dir.strip_prefix(root) {
                    let rel = rel.display().to_string();
                    if !rel.is_empty() {
                        found.push(rel);
                    }
                }
                // No point descending into a repository we are excluding.
                queue.retain(|(queued, _)| !queued.starts_with(&dir));
                break;
            }
            if ALWAYS_EXCLUDED
                .iter()
                .any(|x| x.trim_end_matches('/') == name)
            {
                continue;
            }
            queue.push((path, depth + 1));
        }
    }
    found.sort();
    (found, truncated)
}

/// Make `label` usable as a git ref component.
fn ref_safe(label: &str) -> String {
    let cleaned: String = label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let cleaned = cleaned.trim_matches('-').to_string();
    match cleaned.is_empty() {
        true => "snapshot".into(),
        false => cleaned,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A folder with some work in it, and a place to keep its shadow.
    async fn folder() -> Option<(tempfile::TempDir, Shadow, Excluded)> {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path().join("folder");
        std::fs::create_dir_all(work.join("sub")).unwrap();
        std::fs::write(work.join("report.txt"), "version one\n").unwrap();
        std::fs::write(work.join("sub/data.csv"), "a,b\n1,2\n").unwrap();

        let (shadow, excluded) = Shadow::open(dir.path().join("shadow.git"), &work)
            .await
            .ok()?;
        Some((dir, shadow, excluded))
    }

    #[tokio::test]
    async fn the_users_folder_is_left_completely_unmarked() {
        // The whole reason the git directory lives elsewhere. Even the file
        // that `--separate-git-dir` leaves behind would be a change to a
        // folder cowork was only given permission to read and write within.
        let Some((_dir, shadow, _)) = folder().await else {
            return;
        };
        shadow.snapshot("base").await.unwrap();
        let entries: Vec<String> = std::fs::read_dir(shadow.folder())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert!(
            !entries.iter().any(|e| e.starts_with(".git")),
            "{entries:?}"
        );
    }

    #[tokio::test]
    async fn a_nested_repository_is_excluded_rather_than_fatal() {
        // Found by trying it: `git add -A` fails outright on a nested
        // repository with no commit — "does not have a commit checked out" —
        // so a documents folder containing a checkout could not be snapshotted
        // at all.
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path().join("folder");
        std::fs::create_dir_all(work.join("cloned")).unwrap();
        std::fs::write(work.join("notes.md"), "mine\n").unwrap();
        std::fs::create_dir_all(work.join("cloned/.git")).unwrap();
        std::fs::write(work.join("cloned/README"), "theirs\n").unwrap();

        let Ok((shadow, excluded)) = Shadow::open(dir.path().join("shadow.git"), &work).await
        else {
            return; // no git on this machine
        };
        assert_eq!(excluded.repositories, vec!["cloned".to_string()]);

        let commit = shadow.snapshot("base").await.unwrap();
        let tracked = shadow
            .git(None, None, &["ls-tree", "-r", "--name-only", &commit])
            .await
            .unwrap();
        assert!(tracked.contains("notes.md"));
        assert!(
            !tracked.contains("cloned"),
            "the nested repository was tracked: {tracked}"
        );
    }

    #[tokio::test]
    async fn a_snapshot_survives_garbage_collection() {
        // The coding agent's snapshots are unreferenced and `git gc` may take
        // them. A session that offers to return to a snapshot has to still
        // have it when the user asks.
        let Some((_dir, shadow, _)) = folder().await else {
            return;
        };
        let commit = shadow.snapshot("base").await.unwrap();
        shadow
            .git(None, None, &["gc", "--prune=now", "--quiet"])
            .await
            .unwrap();
        let still = shadow.git(None, None, &["cat-file", "-t", &commit]).await;
        assert_eq!(still.unwrap(), "commit");
    }

    #[tokio::test]
    async fn two_workstreams_get_independent_copies() {
        // The property the whole design rests on: parallel work must not
        // collide, and each result must be comparable against the same base.
        let Some((dir, shadow, _)) = folder().await else {
            return;
        };
        let base = shadow.snapshot("base").await.unwrap();

        let a = dir.path().join("ws-a");
        let b = dir.path().join("ws-b");
        shadow.worktree(&a, &base).await.unwrap();
        shadow.worktree(&b, &base).await.unwrap();

        std::fs::write(a.join("report.txt"), "A rewrote this\n").unwrap();
        std::fs::write(b.join("report.txt"), "B rewrote this\n").unwrap();
        std::fs::write(b.join("extra.md"), "B added this\n").unwrap();

        // Neither has disturbed the other, nor the folder.
        assert_eq!(
            std::fs::read_to_string(shadow.folder().join("report.txt")).unwrap(),
            "version one\n"
        );

        let commit_a = shadow.snapshot_tree(&a, "ws-a").await.unwrap();
        let commit_b = shadow.snapshot_tree(&b, "ws-b").await.unwrap();

        let changes_a = shadow.diff(&base, &commit_a).await.unwrap();
        assert_eq!(
            changes_a,
            vec![FileChange {
                path: "report.txt".into(),
                status: ChangeStatus::Modified
            }]
        );
        let changes_b = shadow.diff(&base, &commit_b).await.unwrap();
        assert_eq!(changes_b.len(), 2, "{changes_b:?}");

        // Keeping B's result brings exactly its changes into the folder.
        let landed = shadow.adopt(&commit_b, &base).await.unwrap();
        assert_eq!(landed.len(), 2);
        assert_eq!(
            std::fs::read_to_string(shadow.folder().join("report.txt")).unwrap(),
            "B rewrote this\n"
        );
        assert!(shadow.folder().join("extra.md").exists());

        shadow.remove_worktree(&a).await.unwrap();
        shadow.remove_worktree(&b).await.unwrap();
    }

    #[tokio::test]
    async fn restoring_is_byte_for_byte() {
        // A folder of real work is full of spreadsheets and images. Reading a
        // blob back through a `String` would corrupt every one of them.
        let Some((_dir, shadow, _)) = folder().await else {
            return;
        };
        let png: Vec<u8> = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0xff, 0x00];
        std::fs::write(shadow.folder().join("chart.png"), &png).unwrap();
        std::fs::write(shadow.folder().join("no-newline.txt"), "ends abruptly").unwrap();
        let base = shadow.snapshot("base").await.unwrap();

        std::fs::write(shadow.folder().join("chart.png"), b"clobbered").unwrap();
        std::fs::write(shadow.folder().join("no-newline.txt"), "changed").unwrap();
        std::fs::write(shadow.folder().join("new.txt"), "created since").unwrap();

        let restored = shadow.restore(&base).await.unwrap();
        assert_eq!(restored.len(), 3, "{restored:?}");
        assert_eq!(
            std::fs::read(shadow.folder().join("chart.png")).unwrap(),
            png
        );
        assert_eq!(
            std::fs::read_to_string(shadow.folder().join("no-newline.txt")).unwrap(),
            "ends abruptly"
        );
        assert!(!shadow.folder().join("new.txt").exists());
    }

    #[tokio::test]
    async fn build_output_is_never_snapshotted() {
        let dir = tempfile::tempdir().unwrap();
        let work = dir.path().join("folder");
        std::fs::create_dir_all(work.join("node_modules/pkg")).unwrap();
        std::fs::write(work.join("node_modules/pkg/index.js"), "huge\n").unwrap();
        std::fs::write(work.join("notes.md"), "mine\n").unwrap();

        let Ok((shadow, _)) = Shadow::open(dir.path().join("shadow.git"), &work).await else {
            return;
        };
        let commit = shadow.snapshot("base").await.unwrap();
        let tracked = shadow
            .git(None, None, &["ls-tree", "-r", "--name-only", &commit])
            .await
            .unwrap();
        assert!(tracked.contains("notes.md"));
        assert!(!tracked.contains("node_modules"), "{tracked}");
    }

    #[test]
    fn a_label_becomes_a_usable_ref() {
        assert_eq!(ref_safe("ws a/b"), "ws-a-b");
        assert_eq!(ref_safe("--"), "snapshot");
        assert_eq!(ref_safe("base"), "base");
    }
}
