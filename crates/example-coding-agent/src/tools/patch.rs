//! A patch envelope that edits several files in one call.
//!
//! The format is the one Codex uses, deliberately. Two reasons, and the
//! second is the stronger: frontier models have seen it, so they emit it more
//! reliably than a format invented here; and it expresses create, delete and
//! rename, which `edit_file` and `multi_edit` cannot.
//!
//! ```text
//! *** Begin Patch
//! *** Update File: src/bus.rs
//! @@ impl EventBus
//!      pub fn new() -> Self {
//! -        Self { count: 0 }
//! +        Self { count: 1 }
//!      }
//! *** Add File: src/new.rs
//! +pub fn hello() {}
//! *** Delete File: src/old.rs
//! *** End Patch
//! ```
//!
//! Hunks match on *context*, not line numbers: three lines either side, with
//! optional `@@ selector` lines to jump to the right region when three lines
//! are not unique. That is what makes a patch survive a file having drifted
//! since it was read — the failure mode that makes line-numbered diffs so
//! frustrating for an agent working from a stale view.
//!
//! Parsing and applying are separate from any file I/O so both can be tested
//! against strings, which is most of what makes a format like this safe.

use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq)]
pub enum FileOp {
    Add {
        path: String,
        contents: String,
    },
    Delete {
        path: String,
    },
    Update {
        path: String,
        move_to: Option<String>,
        hunks: Vec<Hunk>,
    },
}

impl FileOp {
    pub fn path(&self) -> &str {
        match self {
            FileOp::Add { path, .. } | FileOp::Delete { path } | FileOp::Update { path, .. } => {
                path
            }
        }
    }
}

/// One change within a file.
#[derive(Debug, PartialEq, Eq)]
pub struct Hunk {
    /// `@@ selector` lines, outermost first. Narrow where to look before the
    /// context is matched, for when the context alone repeats.
    pub selectors: Vec<String>,
    /// Lines to find: context and removals, in order.
    pub before: Vec<String>,
    /// Lines to put in their place: context and additions, in order.
    pub after: Vec<String>,
}

#[derive(Debug)]
pub struct PatchError(pub String);

impl std::fmt::Display for PatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn err<T>(message: impl Into<String>) -> Result<T, PatchError> {
    Err(PatchError(message.into()))
}

const BEGIN: &str = "*** Begin Patch";
const END: &str = "*** End Patch";
const ADD: &str = "*** Add File: ";
const DELETE: &str = "*** Delete File: ";
const UPDATE: &str = "*** Update File: ";
const MOVE: &str = "*** Move to: ";
const EOF_MARK: &str = "*** End of File";

/// Parse a patch envelope into file operations.
pub fn parse(patch: &str) -> Result<Vec<FileOp>, PatchError> {
    let lines: Vec<&str> = patch.lines().collect();
    let start = match lines.iter().position(|l| l.trim_end() == BEGIN) {
        Some(i) => i + 1,
        None => return err("patch must start with '*** Begin Patch'"),
    };
    let end = match lines.iter().position(|l| l.trim_end() == END) {
        Some(i) => i,
        None => return err("patch must end with '*** End Patch'"),
    };
    if end < start {
        return err("'*** End Patch' came before '*** Begin Patch'");
    }

    let mut ops = Vec::new();
    let mut i = start;
    while i < end {
        let line = lines[i];
        if let Some(path) = line.strip_prefix(ADD) {
            let (contents, next) = parse_added_lines(&lines, i + 1, end)?;
            ops.push(FileOp::Add {
                path: path.trim().to_string(),
                contents,
            });
            i = next;
        } else if let Some(path) = line.strip_prefix(DELETE) {
            ops.push(FileOp::Delete {
                path: path.trim().to_string(),
            });
            i += 1;
        } else if let Some(path) = line.strip_prefix(UPDATE) {
            let mut cursor = i + 1;
            let mut move_to = None;
            if cursor < end {
                if let Some(dest) = lines[cursor].strip_prefix(MOVE) {
                    move_to = Some(dest.trim().to_string());
                    cursor += 1;
                }
            }
            let (hunks, next) = parse_hunks(&lines, cursor, end)?;
            if hunks.is_empty() {
                return err(format!("'{}' has no hunks; use *** Add File to create a file", path.trim()));
            }
            ops.push(FileOp::Update {
                path: path.trim().to_string(),
                move_to,
                hunks,
            });
            i = next;
        } else if line.trim().is_empty() {
            i += 1;
        } else {
            return err(format!(
                "expected a file header (Add/Delete/Update File), found: {line}"
            ));
        }
    }

    if ops.is_empty() {
        return err("the patch contains no file operations");
    }
    Ok(ops)
}

fn parse_added_lines(
    lines: &[&str],
    mut i: usize,
    end: usize,
) -> Result<(String, usize), PatchError> {
    let mut body = Vec::new();
    while i < end && !lines[i].starts_with("*** ") {
        match lines[i].strip_prefix('+') {
            Some(text) => body.push(text.to_string()),
            None if lines[i].trim().is_empty() => body.push(String::new()),
            None => {
                return err(format!(
                    "every line of an added file must start with '+', found: {}",
                    lines[i]
                ))
            }
        }
        i += 1;
    }
    let mut contents = body.join("\n");
    if !contents.is_empty() {
        contents.push('\n');
    }
    Ok((contents, i))
}

fn parse_hunks(lines: &[&str], mut i: usize, end: usize) -> Result<(Vec<Hunk>, usize), PatchError> {
    let mut hunks = Vec::new();
    while i < end && !lines[i].starts_with("*** ") {
        if !lines[i].starts_with("@@") {
            if lines[i].trim().is_empty() {
                i += 1;
                continue;
            }
            return err(format!("expected a hunk starting with '@@', found: {}", lines[i]));
        }

        let mut selectors = Vec::new();
        while i < end && lines[i].starts_with("@@") {
            let selector = lines[i].trim_start_matches('@').trim();
            if !selector.is_empty() {
                selectors.push(selector.to_string());
            }
            i += 1;
        }

        let mut before = Vec::new();
        let mut after = Vec::new();
        while i < end && !lines[i].starts_with("@@") && !lines[i].starts_with("*** ") {
            let line = lines[i];
            if line == EOF_MARK {
                i += 1;
                continue;
            }
            match line.chars().next() {
                Some('-') => before.push(line[1..].to_string()),
                Some('+') => after.push(line[1..].to_string()),
                Some(' ') => {
                    before.push(line[1..].to_string());
                    after.push(line[1..].to_string());
                }
                // A bare empty line is context for a blank line: models emit
                // it constantly, and rejecting it would make the format
                // needlessly brittle.
                None => {
                    before.push(String::new());
                    after.push(String::new());
                }
                Some(_) => {
                    return err(format!(
                        "hunk lines must start with ' ', '-' or '+', found: {line}"
                    ))
                }
            }
            i += 1;
        }

        if before.is_empty() && after.is_empty() {
            return err("a hunk changed nothing");
        }
        hunks.push(Hunk {
            selectors,
            before,
            after,
        });
    }
    Ok((hunks, i))
}

// ── Applying ──────────────────────────────────────────────────────────────────

/// Apply every hunk to `original`, or explain which one did not fit.
pub fn apply_hunks(original: &str, hunks: &[Hunk]) -> Result<String, PatchError> {
    let mut lines: Vec<String> = original.lines().map(str::to_string).collect();
    let trailing_newline = original.ends_with('\n') || original.is_empty();
    // Later hunks are located after earlier ones, which keeps two hunks from
    // matching the same region.
    let mut search_from = 0usize;

    for (index, hunk) in hunks.iter().enumerate() {
        let from = locate_selectors(&lines, &hunk.selectors, search_from).ok_or_else(|| {
            PatchError(format!(
                "hunk {} — could not find the section named by '@@ {}'",
                index + 1,
                hunk.selectors.join(" / ")
            ))
        })?;

        let at = find_run(&lines, &hunk.before, from).ok_or_else(|| {
            PatchError(format!(
                "hunk {} — the context did not match the file. \
                 Re-read the file and copy the surrounding lines exactly.",
                index + 1
            ))
        })?;

        if find_run(&lines, &hunk.before, at + 1).is_some() && hunk.selectors.is_empty() {
            return err(format!(
                "hunk {} — that context appears more than once. \
                 Add an '@@ <enclosing function or type>' line, or include more context.",
                index + 1
            ));
        }

        lines.splice(at..at + hunk.before.len(), hunk.after.iter().cloned());
        search_from = at + hunk.after.len();
    }

    let mut out = lines.join("\n");
    if trailing_newline && !out.is_empty() {
        out.push('\n');
    }
    Ok(out)
}

/// Where to start searching, after honouring any `@@` selectors.
fn locate_selectors(lines: &[String], selectors: &[String], from: usize) -> Option<usize> {
    let mut at = from;
    for selector in selectors {
        at = lines
            .iter()
            .enumerate()
            .skip(at)
            .find(|(_, line)| line.contains(selector.as_str()))
            .map(|(i, _)| i + 1)?;
    }
    Some(at)
}

/// The first index at or after `from` where `needle` appears in `haystack`.
fn find_run(haystack: &[String], needle: &[String], from: usize) -> Option<usize> {
    if needle.is_empty() {
        return Some(from.min(haystack.len()));
    }
    if needle.len() > haystack.len() {
        return None;
    }
    (from..=haystack.len().saturating_sub(needle.len()))
        .find(|&i| haystack[i..i + needle.len()] == *needle)
}

/// Files an update touches, so a caller can read them before applying.
pub fn touched_paths(ops: &[FileOp]) -> HashMap<String, ()> {
    ops.iter().map(|op| (op.path().to_string(), ())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_four_operations_in_one_envelope() {
        let ops = parse(
            "*** Begin Patch\n\
             *** Update File: src/a.rs\n\
             @@\n\
             -old\n\
             +new\n\
             *** Add File: src/b.rs\n\
             +fn hello() {}\n\
             *** Delete File: src/c.rs\n\
             *** End Patch\n",
        )
        .unwrap();

        assert_eq!(ops.len(), 3);
        assert_eq!(ops[0].path(), "src/a.rs");
        assert!(matches!(ops[1], FileOp::Add { .. }));
        assert!(matches!(ops[2], FileOp::Delete { .. }));
    }

    #[test]
    fn a_rename_rides_along_with_an_update() {
        let ops = parse(
            "*** Begin Patch\n\
             *** Update File: old.rs\n\
             *** Move to: new.rs\n\
             @@\n\
             -a\n\
             +b\n\
             *** End Patch\n",
        )
        .unwrap();
        match &ops[0] {
            FileOp::Update { move_to, .. } => assert_eq!(move_to.as_deref(), Some("new.rs")),
            other => panic!("expected an update, got {other:?}"),
        }
    }

    #[test]
    fn context_lines_belong_to_both_sides() {
        let ops = parse(
            "*** Begin Patch\n\
             *** Update File: a.rs\n\
             @@\n\
             \x20keep\n\
             -drop\n\
             +add\n\
             \x20keep2\n\
             *** End Patch\n",
        )
        .unwrap();
        match &ops[0] {
            FileOp::Update { hunks, .. } => {
                assert_eq!(hunks[0].before, vec!["keep", "drop", "keep2"]);
                assert_eq!(hunks[0].after, vec!["keep", "add", "keep2"]);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn applies_a_change_by_matching_context() {
        let original = "fn main() {\n    let x = 1;\n    println!(\"{x}\");\n}\n";
        let hunks = vec![Hunk {
            selectors: vec![],
            before: vec!["    let x = 1;".into()],
            after: vec!["    let x = 2;".into()],
        }];
        assert_eq!(
            apply_hunks(original, &hunks).unwrap(),
            "fn main() {\n    let x = 2;\n    println!(\"{x}\");\n}\n"
        );
    }

    #[test]
    fn a_selector_disambiguates_repeated_context() {
        // The case the format exists for: identical bodies in two functions.
        let original = "fn a() {\n    work();\n}\n\nfn b() {\n    work();\n}\n";
        let hunks = vec![Hunk {
            selectors: vec!["fn b()".into()],
            before: vec!["    work();".into()],
            after: vec!["    work_faster();".into()],
        }];
        assert_eq!(
            apply_hunks(original, &hunks).unwrap(),
            "fn a() {\n    work();\n}\n\nfn b() {\n    work_faster();\n}\n"
        );
    }

    #[test]
    fn ambiguous_context_without_a_selector_is_refused() {
        let original = "fn a() {\n    work();\n}\n\nfn b() {\n    work();\n}\n";
        let hunks = vec![Hunk {
            selectors: vec![],
            before: vec!["    work();".into()],
            after: vec!["    changed();".into()],
        }];
        let message = apply_hunks(original, &hunks).unwrap_err().to_string();
        assert!(message.contains("more than once"), "{message}");
        assert!(message.contains("@@"), "the error should say how to fix it");
    }

    #[test]
    fn context_that_does_not_match_says_so_usefully() {
        let hunks = vec![Hunk {
            selectors: vec![],
            before: vec!["nonexistent".into()],
            after: vec!["x".into()],
        }];
        let message = apply_hunks("actual contents\n", &hunks)
            .unwrap_err()
            .to_string();
        assert!(message.contains("Re-read the file"), "{message}");
    }

    #[test]
    fn several_hunks_apply_in_order_without_colliding() {
        let original = "one\ntwo\nthree\nfour\n";
        let hunks = vec![
            Hunk {
                selectors: vec![],
                before: vec!["one".into()],
                after: vec!["ONE".into()],
            },
            Hunk {
                selectors: vec![],
                before: vec!["three".into()],
                after: vec!["THREE".into()],
            },
        ];
        assert_eq!(
            apply_hunks(original, &hunks).unwrap(),
            "ONE\ntwo\nTHREE\nfour\n"
        );
    }

    #[test]
    fn an_insertion_needs_no_removal() {
        let hunks = vec![Hunk {
            selectors: vec![],
            before: vec!["use a;".into()],
            after: vec!["use a;".into(), "use b;".into()],
        }];
        assert_eq!(
            apply_hunks("use a;\n\nfn main() {}\n", &hunks).unwrap(),
            "use a;\nuse b;\n\nfn main() {}\n"
        );
    }

    #[test]
    fn a_file_without_a_trailing_newline_keeps_it_that_way() {
        let hunks = vec![Hunk {
            selectors: vec![],
            before: vec!["a".into()],
            after: vec!["b".into()],
        }];
        assert_eq!(apply_hunks("a", &hunks).unwrap(), "b");
    }

    #[test]
    fn malformed_envelopes_are_rejected_with_the_reason() {
        for (patch, expected) in [
            ("no envelope at all", "Begin Patch"),
            ("*** Begin Patch\n", "End Patch"),
            ("*** Begin Patch\n*** End Patch\n", "no file operations"),
            (
                "*** Begin Patch\nrandom text\n*** End Patch\n",
                "expected a file header",
            ),
            (
                "*** Begin Patch\n*** Update File: a.rs\n*** End Patch\n",
                "no hunks",
            ),
            (
                "*** Begin Patch\n*** Add File: a.rs\nmissing plus\n*** End Patch\n",
                "must start with '+'",
            ),
        ] {
            let message = parse(patch).unwrap_err().to_string();
            assert!(
                message.contains(expected),
                "for {patch:?} expected {expected:?}, got {message:?}"
            );
        }
    }

    #[test]
    fn an_added_file_keeps_its_blank_lines() {
        let ops = parse(
            "*** Begin Patch\n\
             *** Add File: m.rs\n\
             +fn a() {}\n\
             +\n\
             +fn b() {}\n\
             *** End Patch\n",
        )
        .unwrap();
        match &ops[0] {
            FileOp::Add { contents, .. } => assert_eq!(contents, "fn a() {}\n\nfn b() {}\n"),
            other => panic!("{other:?}"),
        }
    }
}

// ── The tool ──────────────────────────────────────────────────────────────────

use crate::acp::ClientFs;
use crate::lsp::LspPool;
use crate::workspace::Workspace;
use async_trait::async_trait;
use eventage::agent::error::AgentError;
use eventage::agent::tool::Tool;
use eventage::llm::types::ToolDefinition;
use serde_json::{json, Value};
use std::sync::Arc;

/// Applies a whole patch across several files, or none of it.
pub struct ApplyPatch {
    pub ws: Arc<Workspace>,
    /// When present, file I/O is routed through the editor.
    pub client: Option<ClientFs>,
    pub lsp: Arc<LspPool>,
}

/// A file's contents before and after, held until the whole patch validates.
struct Pending {
    path: String,
    absolute: std::path::PathBuf,
    before: Option<String>,
    after: Option<String>,
    move_to: Option<std::path::PathBuf>,
}

impl ApplyPatch {
    async fn read(&self, path: &std::path::Path) -> Option<String> {
        if let Some(client) = &self.client {
            if let Some(text) = client.read(&path.display().to_string()).await {
                return Some(text);
            }
        }
        tokio::fs::read_to_string(path).await.ok()
    }

    async fn write(&self, path: &std::path::Path, contents: &str) -> Result<(), AgentError> {
        if let Some(client) = &self.client {
            if client.write(&path.display().to_string(), contents).await {
                return Ok(());
            }
        }
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.ok();
        }
        tokio::fs::write(path, contents)
            .await
            .map_err(|e| AgentError::Tool(format!("cannot write {}: {e}", path.display())))
    }
}

#[async_trait]
impl Tool for ApplyPatch {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "apply_patch",
            "Apply one patch across several files at once — the right tool for a \
             change that spans files, and the only one that can create, delete or \
             rename them. Hunks match on surrounding context rather than line \
             numbers. Either the whole patch applies or nothing is written.\n\n\
             *** Begin Patch\n\
             *** Update File: path/to/file.rs\n\
             @@ optional selector, e.g. an enclosing fn or impl\n\
             \x20context line\n\
             -removed line\n\
             +added line\n\
             \x20context line\n\
             *** Add File: path/to/new.rs\n\
             +every line prefixed with +\n\
             *** Delete File: path/to/old.rs\n\
             *** End Patch\n\n\
             Give three lines of context either side of a change. If that is not \
             unique, add '@@ <enclosing function or type>' to say where to look.",
            json!({
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": "The full patch envelope, from *** Begin Patch to *** End Patch."
                    }
                },
                "required": ["patch"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let patch = super::arg_str(&args, "patch")?;
        let ops = parse(&patch).map_err(|e| AgentError::Tool(e.to_string()))?;

        // Work everything out in memory first. A patch that fails halfway
        // would leave the tree in a state neither the user nor the model
        // asked for, and finding out which half applied is worse than the
        // original failure.
        let mut pending: Vec<Pending> = Vec::new();
        for op in &ops {
            let absolute = self
                .ws
                .resolve(op.path())
                .map_err(|e| AgentError::Tool(e.to_string()))?;

            match op {
                FileOp::Add { path, contents } => {
                    if self.read(&absolute).await.is_some() {
                        return Err(AgentError::Tool(format!(
                            "'{path}' already exists — use *** Update File to change it"
                        )));
                    }
                    pending.push(Pending {
                        path: path.clone(),
                        absolute,
                        before: None,
                        after: Some(contents.clone()),
                        move_to: None,
                    });
                }
                FileOp::Delete { path } => {
                    let before = self.read(&absolute).await.ok_or_else(|| {
                        AgentError::Tool(format!("cannot delete '{path}': it does not exist"))
                    })?;
                    pending.push(Pending {
                        path: path.clone(),
                        absolute,
                        before: Some(before),
                        after: None,
                        move_to: None,
                    });
                }
                FileOp::Update {
                    path,
                    move_to,
                    hunks,
                } => {
                    let before = self.read(&absolute).await.ok_or_else(|| {
                        AgentError::Tool(format!(
                            "cannot update '{path}': it does not exist. \
                             Use *** Add File to create it."
                        ))
                    })?;
                    let after = apply_hunks(&before, hunks).map_err(|e| {
                        AgentError::Tool(format!("{path}: {e}"))
                    })?;
                    let destination = match move_to {
                        Some(dest) => Some(
                            self.ws
                                .resolve(dest)
                                .map_err(|e| AgentError::Tool(e.to_string()))?,
                        ),
                        None => None,
                    };
                    pending.push(Pending {
                        path: path.clone(),
                        absolute,
                        before: Some(before),
                        after: Some(after),
                        move_to: destination,
                    });
                }
            }
        }

        // Everything validated; now commit.
        let mut changed = Vec::new();
        let mut diffs = Vec::new();
        for item in &pending {
            match (&item.before, &item.after) {
                (_, Some(after)) => {
                    let target = item.move_to.as_ref().unwrap_or(&item.absolute);
                    self.write(target, after).await?;
                    if let Some(dest) = &item.move_to {
                        tokio::fs::remove_file(&item.absolute).await.ok();
                        self.lsp.notify_changed(dest).await;
                    }
                    self.lsp.notify_changed(&item.absolute).await;
                    diffs.push(json!({
                        "path": target.display().to_string(),
                        "old_text": item.before,
                        "new_text": after,
                    }));
                }
                (Some(_), None) => {
                    tokio::fs::remove_file(&item.absolute).await.map_err(|e| {
                        AgentError::Tool(format!("cannot delete {}: {e}", item.path))
                    })?;
                    self.lsp.notify_changed(&item.absolute).await;
                }
                (None, None) => {}
            }
            changed.push(item.path.clone());
        }

        Ok(json!({
            "files_changed": changed.len(),
            "paths": changed,
            // The first diff drives the review card; the rest are listed so
            // the editor can show every file the patch touched.
            "_diff": diffs.first().cloned(),
            "_diffs": diffs,
            "_locations": pending
                .iter()
                .map(|p| json!({ "path": p.absolute.display().to_string() }))
                .collect::<Vec<_>>(),
        }))
    }
}

#[cfg(test)]
mod tool_tests {
    use super::*;

    fn agent() -> (tempfile::TempDir, ApplyPatch) {
        let dir = tempfile::tempdir().unwrap();
        let ws = Arc::new(Workspace::open(dir.path().to_str().unwrap()).unwrap());
        let lsp = Arc::new(LspPool::new(dir.path()));
        (dir, ApplyPatch { ws, client: None, lsp })
    }

    #[tokio::test]
    async fn one_call_changes_several_files() {
        let (dir, tool) = agent();
        std::fs::write(dir.path().join("a.rs"), "fn a() {\n    old();\n}\n").unwrap();
        std::fs::write(dir.path().join("b.rs"), "fn b() {\n    old();\n}\n").unwrap();

        let out = tool
            .execute(json!({ "patch":
                "*** Begin Patch\n\
                 *** Update File: a.rs\n\
                 @@\n\
                 -    old();\n\
                 +    new();\n\
                 *** Update File: b.rs\n\
                 @@\n\
                 -    old();\n\
                 +    new();\n\
                 *** Add File: c.rs\n\
                 +fn c() {}\n\
                 *** End Patch\n" }))
            .await
            .unwrap();

        assert_eq!(out["files_changed"], 3);
        assert!(std::fs::read_to_string(dir.path().join("a.rs")).unwrap().contains("new()"));
        assert!(std::fs::read_to_string(dir.path().join("b.rs")).unwrap().contains("new()"));
        assert_eq!(std::fs::read_to_string(dir.path().join("c.rs")).unwrap(), "fn c() {}\n");
    }

    #[tokio::test]
    async fn nothing_is_written_when_any_hunk_fails() {
        // The reason for validating up front: a half-applied refactor is
        // worse than a rejected one, because nobody knows which half.
        let (dir, tool) = agent();
        std::fs::write(dir.path().join("a.rs"), "fn a() {\n    old();\n}\n").unwrap();
        std::fs::write(dir.path().join("b.rs"), "fn b() {\n    different();\n}\n").unwrap();

        let err = tool
            .execute(json!({ "patch":
                "*** Begin Patch\n\
                 *** Update File: a.rs\n\
                 @@\n\
                 -    old();\n\
                 +    new();\n\
                 *** Update File: b.rs\n\
                 @@\n\
                 -    old();\n\
                 +    new();\n\
                 *** End Patch\n" }))
            .await
            .unwrap_err()
            .to_string();

        assert!(err.contains("b.rs"), "the error should name the file: {err}");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a.rs")).unwrap(),
            "fn a() {\n    old();\n}\n",
            "the file that would have applied must be untouched"
        );
    }

    #[tokio::test]
    async fn it_can_delete_and_rename() {
        let (dir, tool) = agent();
        std::fs::write(dir.path().join("gone.rs"), "obsolete\n").unwrap();
        std::fs::write(dir.path().join("old_name.rs"), "fn thing() {}\n").unwrap();

        tool.execute(json!({ "patch":
            "*** Begin Patch\n\
             *** Delete File: gone.rs\n\
             *** Update File: old_name.rs\n\
             *** Move to: new_name.rs\n\
             @@\n\
             -fn thing() {}\n\
             +fn renamed_thing() {}\n\
             *** End Patch\n" }))
            .await
            .unwrap();

        assert!(!dir.path().join("gone.rs").exists());
        assert!(!dir.path().join("old_name.rs").exists());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("new_name.rs")).unwrap(),
            "fn renamed_thing() {}\n"
        );
    }

    #[tokio::test]
    async fn adding_over_an_existing_file_is_refused() {
        let (dir, tool) = agent();
        std::fs::write(dir.path().join("a.rs"), "existing\n").unwrap();
        let err = tool
            .execute(json!({ "patch":
                "*** Begin Patch\n*** Add File: a.rs\n+new\n*** End Patch\n" }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("already exists"), "{err}");
        assert_eq!(std::fs::read_to_string(dir.path().join("a.rs")).unwrap(), "existing\n");
    }

    #[tokio::test]
    async fn it_cannot_escape_the_workspace() {
        let (_dir, tool) = agent();
        assert!(tool
            .execute(json!({ "patch":
                "*** Begin Patch\n*** Add File: ../../escaped.rs\n+bad\n*** End Patch\n" }))
            .await
            .is_err());
    }

    #[tokio::test]
    async fn every_touched_file_is_reported_for_review() {
        let (dir, tool) = agent();
        std::fs::write(dir.path().join("a.rs"), "one\n").unwrap();
        std::fs::write(dir.path().join("b.rs"), "two\n").unwrap();

        let out = tool
            .execute(json!({ "patch":
                "*** Begin Patch\n\
                 *** Update File: a.rs\n@@\n-one\n+ONE\n\
                 *** Update File: b.rs\n@@\n-two\n+TWO\n\
                 *** End Patch\n" }))
            .await
            .unwrap();

        assert_eq!(out["_diffs"].as_array().unwrap().len(), 2);
        assert!(out["_diff"].is_object(), "the review card needs one diff");
    }
}
