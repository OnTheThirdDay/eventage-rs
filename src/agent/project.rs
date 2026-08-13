//! Project-context files: `AGENTS.md` / `CLAUDE.md` loading.
//!
//! The de-facto standard for giving agents project-specific instructions is
//! a markdown file at the workspace root — `AGENTS.md` (the open convention)
//! or `CLAUDE.md`. [`load_project_context`] finds and reads them so a harness
//! can inject the content into the system prompt:
//!
//! ```no_run
//! use eventage::agent::project::load_project_context;
//!
//! let mut system_prompt = String::from("You are a coding agent.");
//! if let Some(ctx) = load_project_context(".") {
//!     system_prompt.push_str(&format!(
//!         "\n\n## Project instructions (from {})\n{}",
//!         ctx.source.display(),
//!         ctx.content
//!     ));
//! }
//! ```

use std::path::{Path, PathBuf};

/// Files recognized as project context, in priority order.
const CONTEXT_FILES: &[&str] = &["AGENTS.md", "CLAUDE.md"];

/// Project instructions loaded from a context file.
#[derive(Debug, Clone)]
pub struct ProjectContext {
    /// The file the content came from.
    pub source: PathBuf,
    /// Full markdown content.
    pub content: String,
}

impl ProjectContext {
    /// Render as a system-prompt section.
    pub fn system_prompt_section(&self) -> String {
        format!(
            "## Project instructions (from {})\n{}",
            self.source.display(),
            self.content.trim()
        )
    }
}

/// Load `AGENTS.md` (preferred) or `CLAUDE.md` from `dir`.
///
/// Returns `None` when neither exists or the file is empty.
pub fn load_project_context(dir: impl AsRef<Path>) -> Option<ProjectContext> {
    let dir = dir.as_ref();
    for name in CONTEXT_FILES {
        let path = dir.join(name);
        if let Ok(content) = std::fs::read_to_string(&path) {
            if !content.trim().is_empty() {
                return Some(ProjectContext {
                    source: path,
                    content,
                });
            }
        }
    }
    None
}

/// Like [`load_project_context`], but walks up from `dir` toward the
/// filesystem root, collecting **every** context file found — nearest last,
/// so the most specific instructions carry the most recency weight in the
/// prompt. Mirrors how monorepo agents pick up nested `AGENTS.md` files.
pub fn load_project_context_walkup(dir: impl AsRef<Path>) -> Vec<ProjectContext> {
    let mut found = Vec::new();
    let mut current = Some(dir.as_ref().to_path_buf());
    while let Some(d) = current {
        if let Some(ctx) = load_project_context(&d) {
            found.push(ctx);
        }
        current = d.parent().map(Path::to_path_buf);
    }
    found.reverse(); // outermost first, nearest last
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_agents_md_over_claude_md() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("CLAUDE.md"), "claude instructions").unwrap();
        std::fs::write(tmp.path().join("AGENTS.md"), "agents instructions").unwrap();

        let ctx = load_project_context(tmp.path()).unwrap();
        assert_eq!(ctx.content, "agents instructions");
        assert!(ctx.source.ends_with("AGENTS.md"));
        assert!(ctx.system_prompt_section().contains("Project instructions"));
    }

    #[test]
    fn falls_back_to_claude_md() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("CLAUDE.md"), "claude instructions").unwrap();
        let ctx = load_project_context(tmp.path()).unwrap();
        assert_eq!(ctx.content, "claude instructions");
    }

    #[test]
    fn empty_and_missing_yield_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(load_project_context(tmp.path()).is_none());
        std::fs::write(tmp.path().join("AGENTS.md"), "  \n").unwrap();
        assert!(load_project_context(tmp.path()).is_none());
    }

    #[test]
    fn walkup_collects_nested_contexts_nearest_last() {
        let tmp = tempfile::tempdir().unwrap();
        let nested = tmp.path().join("packages").join("web");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(tmp.path().join("AGENTS.md"), "root rules").unwrap();
        std::fs::write(nested.join("AGENTS.md"), "web rules").unwrap();

        let contexts = load_project_context_walkup(&nested);
        assert!(contexts.len() >= 2);
        let last_two: Vec<&str> = contexts
            .iter()
            .rev()
            .take(2)
            .map(|c| c.content.as_str())
            .collect();
        assert_eq!(last_two, vec!["web rules", "root rules"]);
    }
}
