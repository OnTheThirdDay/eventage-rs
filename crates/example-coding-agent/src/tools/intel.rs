//! Code-intelligence tools backed by real language servers, plus planning.

use super::{read_source, unified_diff, write_source};
use crate::acp::ClientFs;
use crate::lsp::{format_diagnostic, format_location, path_to_uri, uri_to_path, LspPool};
use crate::workspace::Workspace;
use async_trait::async_trait;
use eventage::agent::{AgentError, Tool};
use eventage::llm::ToolDefinition;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Shared setup for the position-based LSP tools.
async fn position_request(
    ws: &Workspace,
    lsp: &LspPool,
    args: &Value,
    method: &str,
    extra: Option<Value>,
) -> Result<Value, AgentError> {
    let path = args
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AgentError::Tool("missing 'path'".into()))?;
    let line = args
        .get("line")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| AgentError::Tool("missing 'line' (1-based)".into()))?;
    let character = args.get("character").and_then(|v| v.as_u64()).unwrap_or(1);

    let abs = ws
        .resolve(path)
        .map_err(|e| AgentError::Tool(e.to_string()))?;
    let Some(client) = lsp.for_path(&abs).await else {
        return Err(AgentError::Tool(format!(
            "no language server available for {path}; fall back to grep"
        )));
    };
    client
        .open_file(&abs)
        .await
        .map_err(|e| AgentError::Tool(e.to_string()))?;

    let mut params = json!({
        "textDocument": { "uri": path_to_uri(&abs) },
        // LSP positions are 0-based; our tools take 1-based like editors show.
        "position": { "line": line.saturating_sub(1), "character": character.saturating_sub(1) },
    });
    if let (Some(extra), Some(obj)) = (extra, params.as_object_mut()) {
        if let Some(extra_obj) = extra.as_object() {
            for (k, v) in extra_obj {
                obj.insert(k.clone(), v.clone());
            }
        }
    }

    client
        .request(method, params)
        .await
        .map_err(|e| AgentError::Tool(e.to_string()))
}

fn locations_from(result: &Value) -> Vec<String> {
    match result {
        Value::Array(items) => items.iter().filter_map(format_location).collect(),
        Value::Null => Vec::new(),
        single => format_location(single).into_iter().collect(),
    }
}

// ── lsp_diagnostics ───────────────────────────────────────────────────────────

pub struct LspDiagnostics {
    pub ws: Arc<Workspace>,
    pub lsp: Arc<LspPool>,
}

#[async_trait]
impl Tool for LspDiagnostics {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "lsp_diagnostics",
            "Compiler/linter diagnostics from the language server. Call this after \
             editing code to confirm you did not break the build — it is far faster \
             than a full compile.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File to check; omit for all known files" }
                }
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        match args.get("path").and_then(|v| v.as_str()) {
            Some(path) => {
                let abs = self
                    .ws
                    .resolve(path)
                    .map_err(|e| AgentError::Tool(e.to_string()))?;
                let Some(client) = self.lsp.for_path(&abs).await else {
                    return Ok(json!({
                        "path": path,
                        "available": false,
                        "note": "no language server for this file type",
                    }));
                };
                client
                    .open_file(&abs)
                    .await
                    .map_err(|e| AgentError::Tool(e.to_string()))?;
                // Diagnostics arrive asynchronously after didOpen/didChange.
                tokio::time::sleep(std::time::Duration::from_millis(600)).await;
                let diags: Vec<String> = client
                    .diagnostics_for(&abs)
                    .await
                    .iter()
                    .map(format_diagnostic)
                    .collect();
                Ok(json!({
                    "path": path,
                    "available": true,
                    "count": diags.len(),
                    "diagnostics": diags,
                }))
            }
            None => {
                let all = self.lsp.all_diagnostics().await;
                let rendered: Vec<Value> = all
                    .iter()
                    .map(|(path, diags)| {
                        json!({
                            "path": path,
                            "diagnostics": diags.iter().map(format_diagnostic).collect::<Vec<_>>(),
                        })
                    })
                    .collect();
                Ok(json!({ "files": rendered }))
            }
        }
    }
}

// ── navigation ────────────────────────────────────────────────────────────────

pub struct LspDefinition {
    pub ws: Arc<Workspace>,
    pub lsp: Arc<LspPool>,
}

#[async_trait]
impl Tool for LspDefinition {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "lsp_definition",
            "Jump to where a symbol is defined. Give the position of a use of the symbol.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "line": { "type": "integer", "description": "1-based" },
                    "character": { "type": "integer", "description": "1-based column" }
                },
                "required": ["path", "line"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let result =
            position_request(&self.ws, &self.lsp, &args, "textDocument/definition", None).await?;
        let locations = locations_from(&result);
        Ok(json!({ "count": locations.len(), "definitions": locations }))
    }
}

pub struct LspReferences {
    pub ws: Arc<Workspace>,
    pub lsp: Arc<LspPool>,
}

#[async_trait]
impl Tool for LspReferences {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "lsp_references",
            "Find every real usage of a symbol. Use this before renaming or changing a \
             signature — unlike grep it resolves symbols, so it neither misses call \
             sites nor reports unrelated text matches.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "line": { "type": "integer", "description": "1-based" },
                    "character": { "type": "integer", "description": "1-based column" },
                    "include_declaration": { "type": "boolean" }
                },
                "required": ["path", "line"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let include = args
            .get("include_declaration")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let result = position_request(
            &self.ws,
            &self.lsp,
            &args,
            "textDocument/references",
            Some(json!({ "context": { "includeDeclaration": include } })),
        )
        .await?;
        let locations = locations_from(&result);
        Ok(json!({ "count": locations.len(), "references": locations }))
    }
}

pub struct LspHover {
    pub ws: Arc<Workspace>,
    pub lsp: Arc<LspPool>,
}

#[async_trait]
impl Tool for LspHover {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "lsp_hover",
            "Type signature and documentation for the symbol at a position.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "line": { "type": "integer", "description": "1-based" },
                    "character": { "type": "integer", "description": "1-based column" }
                },
                "required": ["path", "line"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let result =
            position_request(&self.ws, &self.lsp, &args, "textDocument/hover", None).await?;
        // `contents` is a string, a {value}, or an array of either.
        let text = match result.get("contents") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Object(o)) => o
                .get("value")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            Some(Value::Array(items)) => items
                .iter()
                .map(|i| match i {
                    Value::String(s) => s.clone(),
                    other => other
                        .get("value")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                })
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        Ok(json!({ "hover": text }))
    }
}

pub struct LspSymbols {
    pub ws: Arc<Workspace>,
    pub lsp: Arc<LspPool>,
}

#[async_trait]
impl Tool for LspSymbols {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "lsp_symbols",
            "Search symbols by name across the project, or list the symbols defined in \
             one file. The fastest way to find where something lives.",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Symbol name to search for" },
                    "path": { "type": "string", "description": "List symbols in this file instead" }
                }
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        // Document symbols for one file…
        if let Some(path) = args.get("path").and_then(|v| v.as_str()) {
            let abs = self
                .ws
                .resolve(path)
                .map_err(|e| AgentError::Tool(e.to_string()))?;
            let Some(client) = self.lsp.for_path(&abs).await else {
                return Err(AgentError::Tool(format!(
                    "no language server for {path}; use grep"
                )));
            };
            client
                .open_file(&abs)
                .await
                .map_err(|e| AgentError::Tool(e.to_string()))?;
            let result = client
                .request(
                    "textDocument/documentSymbol",
                    json!({ "textDocument": { "uri": path_to_uri(&abs) } }),
                )
                .await
                .map_err(|e| AgentError::Tool(e.to_string()))?;
            return Ok(json!({ "symbols": summarize_symbols(&result) }));
        }

        // …or a workspace-wide search.
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::Tool("provide 'query' or 'path'".into()))?;
        // Any running server can answer; pick one by probing a source file.
        let probe = self.ws.root().join("src/main.rs");
        let client = match self.lsp.for_path(&probe).await {
            Some(c) => c,
            None => {
                return Err(AgentError::Tool(
                    "no language server running; use grep for text search".into(),
                ))
            }
        };
        let result = client
            .request("workspace/symbol", json!({ "query": query }))
            .await
            .map_err(|e| AgentError::Tool(e.to_string()))?;
        Ok(json!({ "symbols": summarize_symbols(&result) }))
    }
}

/// Flatten LSP symbol results (both flat and hierarchical shapes).
fn summarize_symbols(result: &Value) -> Vec<Value> {
    fn walk(node: &Value, out: &mut Vec<Value>) {
        let name = node.get("name").and_then(|n| n.as_str()).unwrap_or("");
        if !name.is_empty() {
            let line = node
                .pointer("/location/range/start/line")
                .or_else(|| node.pointer("/range/start/line"))
                .and_then(|l| l.as_u64())
                .map(|l| l + 1);
            let path = node
                .pointer("/location/uri")
                .and_then(|u| u.as_str())
                .map(crate::lsp::uri_to_path);
            out.push(json!({ "name": name, "path": path, "line": line }));
        }
        if let Some(children) = node.get("children").and_then(|c| c.as_array()) {
            for child in children {
                walk(child, out);
            }
        }
    }

    let mut out = Vec::new();
    if let Some(items) = result.as_array() {
        for item in items {
            walk(item, &mut out);
        }
    }
    out.truncate(100);
    out
}

// ── plan ──────────────────────────────────────────────────────────────────────

/// The live plan, mirrored to the editor as an ACP task checklist.
#[derive(Default)]
pub struct PlanState {
    pub entries: Mutex<Vec<Value>>,
}

pub struct Plan {
    pub state: Arc<PlanState>,
}

#[async_trait]
impl Tool for Plan {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "plan",
            "Record or update your task plan. The editor renders it as a live \
             checklist. Use it for any task with more than two steps, and keep exactly \
             one entry 'in_progress'.",
            json!({
                "type": "object",
                "properties": {
                    "entries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": { "type": "string" },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"]
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"]
                                }
                            },
                            "required": ["content", "status"]
                        }
                    }
                },
                "required": ["entries"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let entries = args
            .get("entries")
            .and_then(|v| v.as_array())
            .ok_or_else(|| AgentError::Tool("entries must be an array".into()))?
            .clone();

        let in_progress = entries
            .iter()
            .filter(|e| e.get("status").and_then(|s| s.as_str()) == Some("in_progress"))
            .count();
        if in_progress > 1 {
            return Err(AgentError::Tool(format!(
                "{in_progress} entries are in_progress; exactly one may be active"
            )));
        }

        *self.state.entries.lock().await = entries.clone();
        let done = entries
            .iter()
            .filter(|e| e.get("status").and_then(|s| s.as_str()) == Some("completed"))
            .count();

        Ok(json!({
            "total": entries.len(),
            "completed": done,
            "_plan": entries,
        }))
    }
}

// ── lsp_rename ────────────────────────────────────────────────────────────────

/// Byte offset of an LSP position within `text`.
///
/// LSP counts columns in UTF-16 code units — not bytes, not characters. On an
/// ASCII line all three agree, which is why getting this wrong survives most
/// testing and then corrupts the one file with an accented name in it.
fn byte_offset(text: &str, line: u64, character: u64) -> Option<usize> {
    let mut offset = 0usize;
    let mut seen = 0u64;
    for current in text.split_inclusive('\n') {
        if seen == line {
            let mut units = 0u64;
            for (index, ch) in current.char_indices() {
                if units >= character {
                    return Some(offset + index);
                }
                units += ch.len_utf16() as u64;
            }
            // Past the end of the line: clamp to its content, not its newline.
            return Some(offset + current.trim_end_matches(['\n', '\r']).len());
        }
        offset += current.len();
        seen += 1;
    }
    // One line past the last is how servers spell "end of file".
    (line == seen).then_some(text.len())
}

/// Apply one file's `TextEdit`s.
///
/// Every range is stated against the *original* text, so they are resolved to
/// byte offsets up front and then applied back to front — otherwise the first
/// edit invalidates the offsets of all the others.
fn apply_text_edits(text: &str, edits: &[Value]) -> Result<String, AgentError> {
    let mut resolved = Vec::with_capacity(edits.len());
    for edit in edits {
        let range = edit
            .get("range")
            .ok_or_else(|| AgentError::Tool("the server sent an edit with no range".into()))?;
        let at = |which: &str| -> Option<usize> {
            let point = range.get(which)?;
            byte_offset(
                text,
                point.get("line")?.as_u64()?,
                point.get("character")?.as_u64()?,
            )
        };
        let start = at("start")
            .ok_or_else(|| AgentError::Tool("an edit starts outside the file".into()))?;
        let end =
            at("end").ok_or_else(|| AgentError::Tool("an edit ends outside the file".into()))?;
        if start > end {
            return Err(AgentError::Tool("an edit range runs backwards".into()));
        }
        resolved.push((
            start,
            end,
            edit.get("newText")
                .and_then(|t| t.as_str())
                .unwrap_or_default()
                .to_string(),
        ));
    }

    resolved.sort_by_key(|(start, _, _)| std::cmp::Reverse(*start));
    let mut out = text.to_string();
    let mut lowest = out.len();
    for (start, end, new_text) in resolved {
        if end > lowest {
            return Err(AgentError::Tool(
                "the server returned overlapping edits; nothing was written".into(),
            ));
        }
        out.replace_range(start..end, &new_text);
        lowest = start;
    }
    Ok(out)
}

/// Per-file edits from a `WorkspaceEdit`, in either shape servers use.
fn edits_by_uri(workspace_edit: &Value) -> Result<Vec<(String, Vec<Value>)>, AgentError> {
    if let Some(changes) = workspace_edit
        .get("documentChanges")
        .and_then(|v| v.as_array())
    {
        let mut out = Vec::new();
        for change in changes {
            // Resource operations carry a `kind`; plain text edits do not.
            // Renaming a Rust module asks for the file to move too, and
            // applying only the text half leaves a workspace that does not
            // build — so refuse the whole edit and say what is missing.
            if let Some(kind) = change.get("kind").and_then(|k| k.as_str()) {
                return Err(AgentError::Tool(format!(
                    "this rename also wants to {kind} a file, which this tool will not do \
                     halfway. Move the file yourself (git mv), then rename the symbol."
                )));
            }
            let uri = change
                .get("textDocument")
                .and_then(|d| d.get("uri"))
                .and_then(|u| u.as_str())
                .ok_or_else(|| AgentError::Tool("a change arrived with no document".into()))?;
            let edits = change
                .get("edits")
                .and_then(|e| e.as_array())
                .cloned()
                .unwrap_or_default();
            out.push((uri.to_string(), edits));
        }
        return Ok(out);
    }

    match workspace_edit.get("changes") {
        Some(Value::Object(map)) => Ok(map
            .iter()
            .map(|(uri, edits)| (uri.clone(), edits.as_array().cloned().unwrap_or_default()))
            .collect()),
        _ => Ok(Vec::new()),
    }
}

/// Rename a symbol everywhere, using the language server's own understanding.
///
/// The difference from a search-and-replace is not convenience, it is
/// correctness: the server knows which `next` is the iterator method and which
/// is a local binding, so shadowed names, same-named fields on other types and
/// the word appearing in a comment are all left alone.
pub struct LspRename {
    pub ws: Arc<Workspace>,
    pub lsp: Arc<LspPool>,
    /// When present, file I/O is routed through the editor.
    pub client: Option<ClientFs>,
}

#[async_trait]
impl Tool for LspRename {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "lsp_rename",
            "Rename a symbol across the whole workspace, using the language server's \
             resolution rather than text matching — so it renames every real reference \
             and nothing that merely shares the name. Prefer this over edit_file for \
             any rename. Give the position of the symbol; all files are updated \
             together or none are.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "line": { "type": "integer", "description": "1-based" },
                    "character": {
                        "type": "integer",
                        "description": "1-based column, inside the symbol's name"
                    },
                    "new_name": { "type": "string", "description": "The new identifier" }
                },
                "required": ["path", "line", "new_name"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let new_name = args
            .get("new_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::Tool("missing 'new_name'".into()))?
            .to_string();
        if new_name.trim().is_empty() {
            return Err(AgentError::Tool("'new_name' must not be empty".into()));
        }

        let result = position_request(
            &self.ws,
            &self.lsp,
            &args,
            "textDocument/rename",
            Some(json!({ "newName": new_name })),
        )
        .await?;

        if result.is_null() {
            return Err(AgentError::Tool(
                "the language server declined to rename at that position — point at the \
                 symbol's own name, and check it is not from a dependency"
                    .into(),
            ));
        }

        // Everything is computed before anything is written, so a failure on
        // the fourth file cannot leave the first three renamed.
        //
        // The paths are locked for the whole read-compute-write, because a
        // rename is exactly the case where a concurrent edit to one of the
        // files would be overwritten without trace.
        let touched: Vec<String> = edits_by_uri(&result)?
            .iter()
            .filter(|(_, edits)| !edits.is_empty())
            .map(|(uri, _)| uri_to_path(uri))
            .filter_map(|abs| {
                std::path::Path::new(&abs)
                    .strip_prefix(self.ws.root())
                    .ok()
                    .map(|p| p.display().to_string())
            })
            .collect();
        let _guard = self.ws.lock_paths(&touched).await;

        let mut pending: Vec<(String, std::path::PathBuf, String, String)> = Vec::new();
        let mut edit_count = 0usize;
        for (uri, edits) in edits_by_uri(&result)? {
            if edits.is_empty() {
                continue;
            }
            let target = std::path::PathBuf::from(uri_to_path(&uri));
            let relative = target
                .strip_prefix(self.ws.root())
                .map_err(|_| {
                    AgentError::Tool(format!(
                        "the rename reaches {}, outside the workspace — declined",
                        target.display()
                    ))
                })?
                .display()
                .to_string();
            // Re-resolve so the same escape check guards this as every other write.
            let abs = self
                .ws
                .resolve(&relative)
                .map_err(|e| AgentError::Tool(e.to_string()))?;

            let original = read_source(&self.ws, &self.client, &relative).await?;
            let updated = apply_text_edits(&original, &edits)?;
            if updated != original {
                edit_count += edits.len();
                pending.push((relative, abs, original, updated));
            }
        }

        if pending.is_empty() {
            return Err(AgentError::Tool(
                "the rename produced no changes — the symbol may already have that name".into(),
            ));
        }
        pending.sort_by(|a, b| a.0.cmp(&b.0));

        let mut changes = Vec::new();
        let mut diffs = Vec::new();
        let mut locations = Vec::new();
        for (relative, abs, original, updated) in &pending {
            write_source(&self.ws, &self.client, relative, updated).await?;
            self.lsp.notify_changed(abs).await;

            let abs_str = abs.display().to_string();
            changes.push(json!({
                "path": relative,
                "diff": unified_diff(relative, original, updated),
            }));
            diffs.push(json!({
                "path": abs_str,
                "old_text": original,
                "new_text": updated,
            }));
            locations.push(json!({ "path": abs_str }));
        }

        Ok(json!({
            "renamed_to": new_name,
            "files": pending.len(),
            "references_updated": edit_count,
            "changes": changes,
            "_diff": diffs.first().cloned(),
            "_diffs": diffs,
            "_locations": locations,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_are_counted_in_utf16_like_the_protocol_says() {
        // "é" is two bytes but one UTF-16 unit; "𝄞" is four bytes and two
        // units. A byte- or char-based reading of the same column lands
        // somewhere else entirely.
        let text = "let é𝄞x = 1;\nsecond\n";
        assert_eq!(byte_offset(text, 0, 0), Some(0));
        // Column 4 is the "é" itself: four units in, but four *bytes* only by
        // coincidence of the ASCII prefix.
        assert_eq!(byte_offset(text, 0, 4), Some("let ".len()));
        // Column 5 is the "𝄞" — one unit past "é", two bytes past it.
        assert_eq!(byte_offset(text, 0, 5), Some("let é".len()));
        // And "𝄞" spends two units, so "x" is at column 7, not 6.
        assert_eq!(byte_offset(text, 0, 7), Some("let é𝄞".len()));
        assert_eq!(byte_offset(text, 1, 0), Some("let é𝄞x = 1;\n".len()));
        // End of file, and past it.
        assert_eq!(byte_offset(text, 2, 0), Some(text.len()));
        assert_eq!(byte_offset(text, 9, 0), None);
    }

    #[test]
    fn a_positions_offsets_stay_valid_when_the_replacement_changes_length() {
        // Two edits on one line, applied back to front. A forward application
        // would shift the second one by the length difference of the first.
        let text = "foo(foo, foo)\n";
        let edit = |line, from, to| {
            json!({
                "range": { "start": { "line": line, "character": from },
                           "end": { "line": line, "character": to } },
                "newText": "renamed_at_length"
            })
        };
        let out = apply_text_edits(text, &[edit(0, 0, 3), edit(0, 4, 7), edit(0, 9, 12)]).unwrap();
        assert_eq!(
            out,
            "renamed_at_length(renamed_at_length, renamed_at_length)\n"
        );
    }

    #[test]
    fn overlapping_edits_are_refused_rather_than_applied() {
        let text = "abcdef\n";
        let edits = [
            json!({ "range": { "start": { "line": 0, "character": 0 },
                               "end": { "line": 0, "character": 4 } }, "newText": "X" }),
            json!({ "range": { "start": { "line": 0, "character": 2 },
                               "end": { "line": 0, "character": 6 } }, "newText": "Y" }),
        ];
        let err = apply_text_edits(text, &edits).unwrap_err().to_string();
        assert!(err.contains("overlapping"), "{err}");
    }

    #[test]
    fn both_workspace_edit_shapes_are_understood() {
        let edit = json!({ "range": { "start": { "line": 0, "character": 0 },
                                      "end": { "line": 0, "character": 1 } }, "newText": "z" });

        let modern = json!({ "documentChanges": [
            { "textDocument": { "uri": "file:///a.rs", "version": 1 }, "edits": [edit] },
        ]});
        assert_eq!(edits_by_uri(&modern).unwrap().len(), 1);

        let legacy = json!({ "changes": { "file:///a.rs": [edit] } });
        let grouped = edits_by_uri(&legacy).unwrap();
        assert_eq!(grouped[0].0, "file:///a.rs");
        assert_eq!(grouped[0].1.len(), 1);
    }

    #[test]
    fn a_rename_that_needs_to_move_a_file_is_refused_whole() {
        // Half a module rename builds worse than none of it.
        let with_file_op = json!({ "documentChanges": [
            { "kind": "rename", "oldUri": "file:///old.rs", "newUri": "file:///new.rs" },
        ]});
        let err = edits_by_uri(&with_file_op).unwrap_err().to_string();
        assert!(err.contains("git mv"), "{err}");
    }

    #[tokio::test]
    async fn renaming_needs_a_new_name() {
        let dir = tempfile::tempdir().unwrap();
        let ws = Arc::new(Workspace::open(dir.path()).unwrap());
        let tool = LspRename {
            lsp: Arc::new(LspPool::new(dir.path())),
            ws,
            client: None,
        };
        let err = tool
            .execute(json!({ "path": "a.rs", "line": 1, "new_name": "  " }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("must not be empty"), "{err}");
    }

    #[tokio::test]
    async fn plan_rejects_multiple_active_entries() {
        let tool = Plan {
            state: Arc::new(PlanState::default()),
        };
        let err = tool
            .execute(json!({ "entries": [
                { "content": "a", "status": "in_progress" },
                { "content": "b", "status": "in_progress" }
            ]}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("exactly one"), "{err}");
    }

    #[tokio::test]
    async fn plan_records_and_reports_progress() {
        let state = Arc::new(PlanState::default());
        let tool = Plan {
            state: state.clone(),
        };
        let out = tool
            .execute(json!({ "entries": [
                { "content": "a", "status": "completed" },
                { "content": "b", "status": "in_progress" },
                { "content": "c", "status": "pending" }
            ]}))
            .await
            .unwrap();
        assert_eq!(out["total"], 3);
        assert_eq!(out["completed"], 1);
        assert_eq!(state.entries.lock().await.len(), 3);
        // The bridge picks this up to drive the editor checklist.
        assert!(out["_plan"].is_array());
    }

    #[test]
    fn flattens_hierarchical_symbols() {
        let result = json!([
            {
                "name": "Foo",
                "range": { "start": { "line": 4 } },
                "children": [ { "name": "bar", "range": { "start": { "line": 9 } } } ]
            }
        ]);
        let symbols = summarize_symbols(&result);
        assert_eq!(symbols.len(), 2);
        assert_eq!(symbols[0]["name"], "Foo");
        assert_eq!(symbols[0]["line"], 5);
        assert_eq!(symbols[1]["name"], "bar");
        assert_eq!(symbols[1]["line"], 10);
    }

    #[test]
    fn locations_handle_single_and_array_results() {
        let single = json!({
            "uri": "file:///a.rs", "range": { "start": { "line": 0, "character": 0 } }
        });
        assert_eq!(locations_from(&single), vec!["/a.rs:1:1"]);
        assert!(locations_from(&Value::Null).is_empty());
        assert_eq!(locations_from(&json!([single])).len(), 1);
    }
}
