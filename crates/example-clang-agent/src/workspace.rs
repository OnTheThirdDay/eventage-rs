use std::path::{Component, Path, PathBuf};

/// A path-sandboxed workspace directory for a single agent session.
///
/// All file operations performed by tools are resolved through this struct,
/// which prevents path traversal outside the workspace root.
#[derive(Clone)]
pub struct Workspace {
    root: PathBuf,
}

#[derive(Debug)]
pub struct FileEntry {
    /// Path relative to the workspace root, using '/' as separator.
    pub path: String,
    pub size_bytes: u64,
}

impl Workspace {
    /// Open (and create if necessary) the workspace for a session.
    pub fn open(path: PathBuf) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&path)?;
        // Canonicalise only the root (it now exists).
        let root = path.canonicalize()?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a relative path safely.
    ///
    /// Returns `Err` if:
    /// - the path is absolute
    /// - the normalised result escapes the workspace root
    pub fn resolve(&self, relative: &str) -> anyhow::Result<PathBuf> {
        if Path::new(relative).is_absolute() {
            anyhow::bail!("absolute paths are not allowed: '{relative}'");
        }
        let joined = self.root.join(relative);
        let normalised = normalise_path(&joined);
        if normalised.starts_with(&self.root) {
            Ok(normalised)
        } else {
            anyhow::bail!("path '{relative}' escapes the workspace boundary")
        }
    }

    /// Ensure the `bin/` subdirectory exists.
    pub fn ensure_bin_dir(&self) -> anyhow::Result<()> {
        std::fs::create_dir_all(self.root.join("bin"))?;
        Ok(())
    }

    /// Recursively list all files in the workspace with metadata.
    pub fn list_files(&self) -> anyhow::Result<Vec<FileEntry>> {
        let mut entries = Vec::new();
        collect_files(&self.root, &self.root, &mut entries)?;
        entries.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(entries)
    }
}

/// Normalise a path (resolve `.` and `..`) without requiring it to exist.
fn normalise_path(path: &Path) -> PathBuf {
    let mut parts: Vec<Component> = Vec::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                // Only pop if we have something to pop (don't pop the root prefix).
                if matches!(parts.last(), Some(Component::Normal(_))) {
                    parts.pop();
                }
            }
            Component::CurDir => {}
            c => parts.push(c),
        }
    }
    parts.iter().collect()
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<FileEntry>) -> anyhow::Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(root, &path, out)?;
        } else {
            let meta = entry.metadata()?;
            let rel = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            out.push(FileEntry {
                path: rel,
                size_bytes: meta.len(),
            });
        }
    }
    Ok(())
}
