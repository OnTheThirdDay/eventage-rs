//! Workspace — the directory the agent is confined to.
//!
//! Confinement is enforced with a **capability handle**, not by inspecting
//! path strings. The workspace holds an open directory descriptor for its
//! root and every file operation is performed relative to it, so the kernel
//! refuses anything that would leave — on Linux via `openat2` with
//! `RESOLVE_BENEATH`, and through the equivalent careful emulation elsewhere.
//! That is what `cap-std` is: the Bytecode Alliance's capability-oriented
//! filesystem layer, the same one WASI is built on.
//!
//! It used to be a string check: normalise the path lexically, confirm it
//! starts with the root. That cannot work, and the reason is worth stating
//! plainly because the code looked right. Lexical normalisation does not
//! resolve symlinks, but every subsequent `read`/`write` does. So a
//! repository containing
//!
//! ```text
//! repo/
//!   notes -> /home/you
//! ```
//!
//! turned `read_file("notes/.ssh/id_rsa")` into a path that passes the
//! prefix test and reads your private key — through `read_file`, which is
//! classified read-only and allowed in every permission mode, including Plan.
//! An adversarial repository could walk credentials straight into the model's
//! context. Canonicalising the string instead would have closed that case and
//! left the race: a symlink can be created between the check and the open.
//! Only holding the directory open removes both.
//!
//! Symlinks *within* the workspace still work — they resolve beneath the
//! root, which is exactly what `RESOLVE_BENEATH` permits.

use anyhow::{bail, Context, Result};
use cap_std::ambient_authority;
use cap_std::fs::Dir;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};

/// The working directory the agent is confined to.
#[derive(Debug, Clone)]
pub struct Workspace {
    root: PathBuf,
    /// The capability. Every read and write goes through this, never through
    /// an absolute path handed to the ambient filesystem.
    dir: Arc<Dir>,
    /// One lock per path, for read-modify-write sequences.
    ///
    /// The ReAct loop runs up to four tools at once, which is right for
    /// searches and reads and wrong for edits: two edits to one file both
    /// read the same original, and the second write silently discards the
    /// first. Whoever holds the path holds it across the whole read, modify
    /// and write.
    edit_locks: Arc<StdMutex<HashMap<PathBuf, Arc<tokio::sync::Mutex<()>>>>>,
    /// What each file looked like the last time *we* touched it.
    ///
    /// The agent should not blindly overwrite a file that changed underneath
    /// it. The obvious way to check is to make the model carry a hash from
    /// its read to its write, and that does not work: it means asking an LLM
    /// to transcribe 64 hex characters across a conversation that may be
    /// summarised in between, and a mistyped one reports a *conflict* — so
    /// the diagnosis points at concurrent modification when the real cause
    /// was a typo. Recording it here instead asks the model for nothing and
    /// cannot be got wrong.
    observed: Arc<StdMutex<HashMap<PathBuf, u64>>>,
}

/// A cheap content fingerprint. Not cryptographic — this detects an honest
/// change, it does not defend against one crafted to collide.
fn fingerprint(bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

/// Exclusive access to a set of paths, released when dropped.
pub struct EditGuard {
    _held: Vec<tokio::sync::OwnedMutexGuard<()>>,
}

impl Workspace {
    /// Open (or create) a workspace at `root`.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;
        // Canonicalised once, here, so `root()` is a real path — the handle
        // is what enforces confinement, so this is only for display.
        let root = std::fs::canonicalize(&root).unwrap_or(root);
        let dir = Dir::open_ambient_dir(&root, ambient_authority())
            .with_context(|| format!("cannot open workspace '{}'", root.display()))?;
        Ok(Self {
            root,
            dir: Arc::new(dir),
            edit_locks: Arc::new(StdMutex::new(HashMap::new())),
            observed: Arc::new(StdMutex::new(HashMap::new())),
        })
    }

    /// The absolute path of the workspace root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Check a relative path and return it, ready for the capability handle.
    ///
    /// Rejects what a capability cannot express: absolute paths, and `..` in
    /// any position. `..` is refused rather than collapsed because collapsing
    /// it is precisely the lexical reasoning that used to be wrong — `a/../b`
    /// is not `b` when `a` is a symlink.
    fn checked(&self, rel: &str) -> Result<PathBuf> {
        let path = Path::new(rel);
        if path.is_absolute() || rel.starts_with('\\') {
            bail!("absolute paths are not allowed: '{rel}'");
        }
        for component in path.components() {
            match component {
                std::path::Component::ParentDir => {
                    bail!("'..' is not allowed in a workspace path: '{rel}'")
                }
                std::path::Component::Prefix(_) | std::path::Component::RootDir => {
                    bail!("absolute paths are not allowed: '{rel}'")
                }
                _ => {}
            }
        }
        // An empty path means the root itself, which no file operation wants.
        if path.as_os_str().is_empty() {
            bail!("a path is required");
        }
        Ok(path.to_path_buf())
    }

    /// The absolute path a workspace-relative path *names*.
    ///
    /// For display, for `_locations`, for handing to a language server as a
    /// URI, for a subprocess's working directory. **This is not the security
    /// boundary** — it is a string, and a string cannot describe what a
    /// symlink will do at open time. Anything that reads or writes must go
    /// through the methods below instead.
    pub fn resolve(&self, rel: &str) -> Result<PathBuf> {
        Ok(self.root.join(self.checked(rel)?))
    }

    /// Run a blocking filesystem operation against the capability.
    async fn with_dir<T, F>(&self, work: F) -> Result<T>
    where
        F: FnOnce(&Dir) -> std::io::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let dir = Arc::clone(&self.dir);
        tokio::task::spawn_blocking(move || work(&dir))
            .await
            .context("filesystem task panicked")?
            .map_err(Into::into)
    }

    /// Read a file as text, confined to the workspace.
    pub async fn read_to_string(&self, rel: &str) -> Result<String> {
        let path = self.checked(rel)?;
        let text = self
            .with_dir({
                let path = path.clone();
                move |dir| dir.read_to_string(&path)
            })
            .await
            .with_context(|| format!("cannot read '{rel}'"))?;
        self.remember(&path, text.as_bytes());
        Ok(text)
    }

    fn remember(&self, path: &Path, bytes: &[u8]) {
        self.observed
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(path.to_path_buf(), fingerprint(bytes));
    }

    /// Record what was last seen at `rel`, whatever route the text arrived by.
    ///
    /// A read served out of the editor's buffer never touched this, so
    /// `observed` stayed empty and [`ensure_unchanged`](Self::ensure_unchanged)
    /// was a no-op for every file the editor had open — which is precisely
    /// the set of files a human is most likely to be editing at the same
    /// time. The guard was present headless and absent where it mattered.
    pub fn remember_source(&self, rel: &str, text: &str) {
        if let Ok(path) = self.checked(rel) {
            self.remember(&path, text.as_bytes());
        }
    }

    /// Fail if `current` differs from what was last seen at `rel`.
    ///
    /// Takes the current contents rather than reading them, because the
    /// comparison has to be made against whatever source the write will go
    /// to. Reading from disk when the editor holds an unsaved buffer
    /// compares two different things and refuses honest work.
    pub fn ensure_matches(&self, rel: &str, current: &[u8]) -> Result<()> {
        let path = self.checked(rel)?;
        let seen = self
            .observed
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&path)
            .copied();
        if let Some(seen) = seen {
            if fingerprint(current) != seen {
                bail!(
                    "'{rel}' has changed since you last read it — something else \
                     edited it. Read it again and reapply your change; writing now \
                     would discard theirs."
                );
            }
        }
        Ok(())
    }

    /// Fail if the file changed since we last saw it.
    ///
    /// Separate from the write, and called *before* anything else reads the
    /// file, because a read refreshes the record — `write_file` reads the old
    /// contents to build its diff, and doing that first silently re-observed
    /// the very change this is meant to catch.
    ///
    /// For blind overwrites — `write_file` replaces a whole file without
    /// looking at what was there. `edit_file` and `apply_patch` do not need
    /// this and deliberately do not use it: they match on surrounding text,
    /// which is both a finer check and a safer one. A whole-file comparison
    /// would refuse an edit that is still perfectly valid merely because
    /// something reformatted an unrelated part of the file.
    ///
    /// A file we have never read is not protected, because overwriting one is
    /// an ordinary thing to do. What this catches is the dangerous sequence:
    /// read it, something else changes it, write it back.
    pub async fn ensure_unchanged(&self, rel: &str) -> Result<()> {
        let path = self.checked(rel)?;
        let probe = path.clone();
        // A file that cannot be read now — deleted, or never there — is not
        // a stale write; the write itself will report whatever is wrong.
        let Ok(current) = self.with_dir(move |dir| dir.read(&probe)).await else {
            return Ok(());
        };
        self.ensure_matches(rel, &current)
    }

    /// Read a file as bytes, confined to the workspace.
    pub async fn read(&self, rel: &str) -> Result<Vec<u8>> {
        let path = self.checked(rel)?;
        self.with_dir(move |dir| dir.read(&path))
            .await
            .with_context(|| format!("cannot read '{rel}'"))
    }

    /// Write a file, creating parent directories, confined to the workspace.
    ///
    /// Written to a neighbouring temporary file, flushed, then renamed over
    /// the target. A plain write truncates first and fills after, so a crash
    /// or a full disk in between leaves a source file that is half its old
    /// contents and half nothing — the tool reports success and the work is
    /// gone. `rename` within a directory is atomic, so a reader sees either
    /// the old file or the new one.
    ///
    /// The temporary file is a sibling because rename is only atomic within a
    /// filesystem, and it inherits the target's permissions so a rename does
    /// not quietly make an executable script unexecutable.
    pub async fn write(&self, rel: &str, contents: impl AsRef<[u8]>) -> Result<()> {
        let path = self.checked(rel)?;
        let contents_ref = contents.as_ref();
        let bytes = contents_ref.to_vec();
        let contents_ref = contents_ref.to_vec();
        let contents_ref = &contents_ref[..];
        self.with_dir(move |dir| {
            use std::io::Write;

            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    dir.create_dir_all(parent)?;
                }
            }

            // Writing *through* a symlink is a deliberate act — the link
            // stays a link, and the file it points at is updated. Renaming
            // over it would replace the link instead, which is a different
            // operation than the caller asked for.
            if dir.symlink_metadata(&path).is_ok_and(|m| m.is_symlink()) {
                return dir.write(&path, &bytes);
            }

            let permissions = dir.metadata(&path).ok().map(|m| m.permissions());
            let temporary = path.with_file_name(format!(
                ".{}.eventage-{}",
                path.file_name().and_then(|n| n.to_str()).unwrap_or("tmp"),
                uuid::Uuid::new_v4().simple()
            ));

            let outcome = (|| -> std::io::Result<()> {
                let mut file = dir.create(&temporary)?;
                file.write_all(&bytes)?;
                file.sync_all()?;
                if let Some(mode) = permissions {
                    file.set_permissions(mode)?;
                }
                drop(file);
                dir.rename(&temporary, dir, &path)
            })();

            if outcome.is_err() {
                let _ = dir.remove_file(&temporary);
            }
            outcome
        })
        .await
        .with_context(|| format!("cannot write '{rel}'"))?;

        self.remember(&self.checked(rel)?, contents_ref);
        Ok(())
    }

    /// File metadata, confined to the workspace.
    pub async fn metadata(&self, rel: &str) -> Result<cap_std::fs::Metadata> {
        let path = self.checked(rel)?;
        self.with_dir(move |dir| dir.metadata(&path))
            .await
            .with_context(|| format!("cannot stat '{rel}'"))
    }

    /// Whether a path exists inside the workspace.
    ///
    /// A path that escapes reads as absent, which is the honest answer: there
    /// is nothing at that path *in this workspace*.
    pub async fn exists(&self, rel: &str) -> bool {
        self.metadata(rel).await.is_ok()
    }

    /// Delete a file, confined to the workspace.
    pub async fn remove_file(&self, rel: &str) -> Result<()> {
        let path = self.checked(rel)?;
        self.with_dir(move |dir| dir.remove_file(&path))
            .await
            .with_context(|| format!("cannot delete '{rel}'"))
    }

    /// Take exclusive access to these paths for a read-modify-write.
    ///
    /// Acquired in sorted order, always. Two calls that lock the same pair in
    /// opposite orders deadlock, and a multi-file tool — `apply_patch`,
    /// `lsp_rename` — has no natural order of its own, so a total one is
    /// imposed here rather than left to each caller to remember.
    ///
    /// This serialises *this process*. It says nothing about the user's
    /// editor or a shell command writing the same file; `edit_file` catches
    /// that separately by requiring its `old_string` to still match.
    pub async fn lock_paths<S: AsRef<str>>(&self, rels: &[S]) -> EditGuard {
        let mut keys: Vec<PathBuf> = rels.iter().map(|r| PathBuf::from(r.as_ref())).collect();
        keys.sort();
        keys.dedup();

        let mut held = Vec::with_capacity(keys.len());
        for key in keys {
            let lock = {
                let mut locks = self.edit_locks.lock().unwrap_or_else(|e| e.into_inner());
                Arc::clone(locks.entry(key).or_default())
            };
            held.push(lock.lock_owned().await);
        }
        EditGuard { _held: held }
    }

    /// Exclusive access to one path.
    pub async fn lock_path(&self, rel: &str) -> EditGuard {
        self.lock_paths(&[rel]).await
    }

    /// Verify that `rel` names a directory beneath the root, and return its
    /// absolute path.
    ///
    /// For the recursive walkers, which need a real path to hand to the
    /// `ignore` crate. Opening it through the handle first is what makes the
    /// starting point safe: `Walk::new` follows a symlinked root, so
    /// `grep(path: "notes")` where `notes -> /home/you` would otherwise have
    /// walked a home directory. Inside the tree the walker does not follow
    /// links, so checking the root is sufficient.
    ///
    /// This is still a path, so it inherits a path's weakness: nothing stops
    /// the directory being replaced between this check and the walk. That is
    /// a far narrower window than the original one, and closing it entirely
    /// means a walker that works from a directory handle.
    pub async fn confined_dir(&self, rel: &str) -> Result<PathBuf> {
        let checked = match rel {
            "" | "." | "./" => None,
            other => Some(self.checked(other)?),
        };
        let Some(path) = checked else {
            return Ok(self.root.clone());
        };
        let probe = path.clone();
        let is_dir = self
            .with_dir(move |dir| dir.metadata(&probe).map(|m| m.is_dir()))
            .await
            .with_context(|| format!("cannot open '{rel}'"))?;
        if !is_dir {
            bail!("'{rel}' is not a directory");
        }
        Ok(self.root.join(path))
    }

    /// Directory entries as `(name, is_dir)`, confined to the workspace.
    pub async fn read_dir(&self, rel: &str) -> Result<Vec<(String, bool)>> {
        // The root is the one path `checked` refuses, and listing it is
        // ordinary, so it is spelled out here.
        let path = match rel {
            "" | "." | "./" => PathBuf::from("."),
            other => self.checked(other)?,
        };
        self.with_dir(move |dir| {
            let mut out = Vec::new();
            for entry in dir.read_dir(&path)? {
                let entry = entry?;
                let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
                out.push((entry.file_name().to_string_lossy().to_string(), is_dir));
            }
            out.sort();
            Ok(out)
        })
        .await
        .with_context(|| format!("cannot list '{rel}'"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The attack this module exists to stop.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_symlink_out_of_the_workspace_cannot_be_read_through() {
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("id_rsa"), "PRIVATE KEY").unwrap();

        let repo = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), repo.path().join("notes")).unwrap();

        let ws = Workspace::open(repo.path()).unwrap();

        // The lexical check this replaced passed this path, and the read then
        // followed the link.
        let err = ws.read_to_string("notes/id_rsa").await.unwrap_err();
        assert!(
            !format!("{err:#}").contains("PRIVATE KEY"),
            "the key must not come back"
        );
        assert!(!ws.exists("notes/id_rsa").await);

        // Writing through it is refused too, and leaves nothing behind.
        assert!(ws.write("notes/planted", "x").await.is_err());
        assert!(!outside.path().join("planted").exists());
    }

    /// The symlink itself is no more readable than what it points at.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_symlink_to_a_file_outside_is_refused() {
        let outside = tempfile::tempdir().unwrap();
        let secret = outside.path().join("secret");
        std::fs::write(&secret, "s3cret").unwrap();

        let repo = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(&secret, repo.path().join("innocent.txt")).unwrap();

        let ws = Workspace::open(repo.path()).unwrap();
        assert!(ws.read_to_string("innocent.txt").await.is_err());
    }

    /// Confinement is not a ban on symlinks — one that stays inside is fine.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_symlink_that_stays_inside_still_works() {
        let repo = tempfile::tempdir().unwrap();
        std::fs::create_dir(repo.path().join("real")).unwrap();
        std::fs::write(repo.path().join("real/data.txt"), "inside").unwrap();
        std::os::unix::fs::symlink("real", repo.path().join("link")).unwrap();

        let ws = Workspace::open(repo.path()).unwrap();
        assert_eq!(ws.read_to_string("link/data.txt").await.unwrap(), "inside");
    }

    #[tokio::test]
    async fn ordinary_reads_and_writes_work() {
        let repo = tempfile::tempdir().unwrap();
        let ws = Workspace::open(repo.path()).unwrap();

        ws.write("src/nested/main.rs", "fn main() {}")
            .await
            .unwrap();
        assert_eq!(
            ws.read_to_string("src/nested/main.rs").await.unwrap(),
            "fn main() {}"
        );
        assert!(ws.exists("src/nested/main.rs").await);
        assert_eq!(ws.metadata("src/nested/main.rs").await.unwrap().len(), 12);

        let entries = ws.read_dir("src").await.unwrap();
        assert_eq!(entries, vec![("nested".to_string(), true)]);

        ws.remove_file("src/nested/main.rs").await.unwrap();
        assert!(!ws.exists("src/nested/main.rs").await);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_write_replaces_the_file_rather_than_truncating_it() {
        use std::os::unix::fs::PermissionsExt;
        let repo = tempfile::tempdir().unwrap();
        let ws = Workspace::open(repo.path()).unwrap();

        // An executable script, as a build or a hook would leave it.
        ws.write("run.sh", "#!/bin/sh\necho old\n").await.unwrap();
        let target = repo.path().join("run.sh");
        std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o755)).unwrap();

        ws.write("run.sh", "#!/bin/sh\necho new\n").await.unwrap();

        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            "#!/bin/sh\necho new\n"
        );
        // The rename must not have quietly made it unexecutable.
        assert_eq!(
            std::fs::metadata(&target).unwrap().permissions().mode() & 0o777,
            0o755,
        );
        // And no temporary file is left behind.
        let leftovers: Vec<_> = std::fs::read_dir(repo.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains("eventage-"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    /// Writing through a link inside the workspace updates its target.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_internal_symlink_is_written_through_not_replaced() {
        let repo = tempfile::tempdir().unwrap();
        std::fs::write(repo.path().join("real.txt"), "before").unwrap();
        std::os::unix::fs::symlink("real.txt", repo.path().join("link.txt")).unwrap();

        let ws = Workspace::open(repo.path()).unwrap();
        ws.write("link.txt", "after").await.unwrap();

        assert_eq!(
            std::fs::read_to_string(repo.path().join("real.txt")).unwrap(),
            "after",
            "the link was replaced instead of written through"
        );
        assert!(std::fs::symlink_metadata(repo.path().join("link.txt"))
            .unwrap()
            .is_symlink());
    }

    #[tokio::test]
    async fn absolute_and_parent_paths_are_refused() {
        let repo = tempfile::tempdir().unwrap();
        let ws = Workspace::open(repo.path()).unwrap();

        for path in ["/etc/passwd", "../outside", "a/../../outside", "a/../b"] {
            assert!(
                ws.read_to_string(path).await.is_err(),
                "'{path}' should be refused"
            );
            assert!(ws.resolve(path).is_err(), "'{path}' should be refused");
        }
        // `..` is refused even where it would collapse harmlessly: deciding
        // that requires knowing whether `a` is a symlink, which a string
        // cannot tell you.
        assert!(ws.resolve("a/b/../c").is_err());
    }

    #[test]
    fn resolve_names_a_path_without_touching_the_disk() {
        let repo = tempfile::tempdir().unwrap();
        let ws = Workspace::open(repo.path()).unwrap();
        let path = ws.resolve("does/not/exist.rs").unwrap();
        assert!(path.starts_with(ws.root()));
        assert!(path.ends_with("does/not/exist.rs"));
    }
}
