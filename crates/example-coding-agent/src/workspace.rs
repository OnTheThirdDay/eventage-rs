//! Workspace — a sandboxed working directory for the coding agent.
//!
//! The workspace is a directory on the local filesystem. All tool operations
//! (read, write, execute) are confined to this directory. Paths supplied by the
//! LLM are resolved against the workspace root and validated to prevent escapes.

use anyhow::{bail, Result};
use std::path::{Path, PathBuf};

/// An entry returned by [`Workspace::list_files`].
#[derive(Debug, Clone)]
pub struct FileEntry {
    /// Workspace-relative path (e.g. `"src/main.py"`).
    pub path: String,
    /// File size in bytes.
    pub size_bytes: u64,
}

/// A sandboxed working directory for the coding agent.
#[derive(Debug)]
pub struct Workspace {
    root: PathBuf,
}

impl Workspace {
    /// Open (or create) a workspace at `root`.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// The absolute path of the workspace root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a workspace-relative path to an absolute path, rejecting any
    /// attempt to escape the root via `..` or absolute paths.
    pub fn resolve(&self, rel: &str) -> Result<PathBuf> {
        // Reject absolute paths outright.
        if rel.starts_with('/') || rel.starts_with('\\') {
            bail!("absolute paths are not allowed: '{rel}'");
        }

        let candidate = self.root.join(rel);
        // Canonicalize the candidate without resolving symlinks (the file may
        // not exist yet). We normalise manually instead.
        let normalised = normalize_path(&candidate);

        if !normalised.starts_with(&self.root) {
            bail!("path escape detected: '{rel}' resolves outside workspace");
        }

        Ok(normalised)
    }

    /// List all regular files in the workspace recursively.
    pub fn list_files(&self) -> Result<Vec<FileEntry>> {
        let mut entries = Vec::new();
        collect_files(&self.root, &self.root, &mut entries)?;
        entries.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(entries)
    }

    /// Ensure `bin/` directory exists (used by the execute tool).
    pub fn ensure_bin_dir(&self) -> Result<()> {
        #![allow(dead_code)]
        std::fs::create_dir_all(self.root.join("bin"))?;
        Ok(())
    }
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<FileEntry>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let meta = entry.metadata()?;
        if meta.is_dir() {
            collect_files(root, &path, out)?;
        } else if meta.is_file() {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();
            out.push(FileEntry {
                path: rel,
                size_bytes: meta.len(),
            });
        }
    }
    Ok(())
}

/// Normalise a path without requiring it to exist (no `canonicalize`).
fn normalize_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            c => out.push(c),
        }
    }
    out
}
