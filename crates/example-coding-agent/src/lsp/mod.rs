//! Language Server Protocol client — real code intelligence for the agent.
//!
//! Grep and embeddings guess at code structure; a language server *knows* it.
//! This client spawns the project's language servers and exposes
//! go-to-definition, find-references, hover types, document/workspace symbols,
//! and live diagnostics, so the agent can navigate and verify code the way a
//! developer's editor does.
//!
//! Servers are started lazily per language and reused for the session.
//! A server that is not installed simply yields no results — the agent falls
//! back to text search rather than failing.

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::{oneshot, Mutex};
use tokio::time::{timeout, Duration};
use tracing::{debug, warn};

/// How long to wait for a language-server response before giving up.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);

/// A language server we know how to launch.
#[derive(Debug, Clone)]
pub struct ServerSpec {
    pub language_id: &'static str,
    pub command: &'static str,
    pub args: &'static [&'static str],
    pub extensions: &'static [&'static str],
}

/// Built-in server definitions, matched by file extension.
pub const SERVERS: &[ServerSpec] = &[
    ServerSpec {
        language_id: "rust",
        command: "rust-analyzer",
        args: &[],
        extensions: &["rs"],
    },
    ServerSpec {
        language_id: "typescript",
        command: "typescript-language-server",
        args: &["--stdio"],
        extensions: &["ts", "tsx", "js", "jsx", "mjs", "cjs"],
    },
    ServerSpec {
        language_id: "python",
        command: "pyright-langserver",
        args: &["--stdio"],
        extensions: &["py", "pyi"],
    },
    ServerSpec {
        language_id: "go",
        command: "gopls",
        args: &[],
        extensions: &["go"],
    },
];

/// Find the server spec responsible for `path`, if any.
pub fn server_for(path: &Path) -> Option<&'static ServerSpec> {
    let ext = path.extension()?.to_str()?;
    SERVERS
        .iter()
        .find(|s| s.extensions.iter().any(|e| e.eq_ignore_ascii_case(ext)))
}

/// Convert a filesystem path to a `file://` URI.
pub fn path_to_uri(path: &Path) -> String {
    format!("file://{}", path.display())
}

/// Convert a `file://` URI back to a path string.
pub fn uri_to_path(uri: &str) -> String {
    uri.strip_prefix("file://").unwrap_or(uri).to_string()
}

// ── Client ────────────────────────────────────────────────────────────────────

type Pending = Arc<Mutex<HashMap<i64, oneshot::Sender<Result<Value, String>>>>>;
/// Diagnostics keyed by file URI, refreshed by `publishDiagnostics`.
type Diagnostics = Arc<Mutex<HashMap<String, Vec<Value>>>>;

/// A running language server.
pub struct LspClient {
    #[allow(dead_code)]
    child: Child,
    stdin: Mutex<ChildStdin>,
    next_id: AtomicI64,
    pending: Pending,
    diagnostics: Diagnostics,
    /// Files we have already sent `didOpen` for.
    opened: Mutex<HashMap<String, ()>>,
    pub language_id: String,
}

impl LspClient {
    /// Spawn `spec`'s server rooted at `root` and run the initialize handshake.
    pub async fn start(spec: &ServerSpec, root: &Path) -> Result<Self> {
        let mut child = tokio::process::Command::new(spec.command)
            .args(spec.args)
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("language server '{}' not available", spec.command))?;

        let stdin = child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;

        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let diagnostics: Diagnostics = Arc::new(Mutex::new(HashMap::new()));
        tokio::spawn(read_loop(
            BufReader::new(stdout),
            Arc::clone(&pending),
            Arc::clone(&diagnostics),
        ));

        let client = Self {
            child,
            stdin: Mutex::new(stdin),
            next_id: AtomicI64::new(1),
            pending,
            diagnostics,
            opened: Mutex::new(HashMap::new()),
            language_id: spec.language_id.to_string(),
        };

        client
            .request(
                "initialize",
                json!({
                    "processId": std::process::id(),
                    "rootUri": path_to_uri(root),
                    "workspaceFolders": [{
                        "uri": path_to_uri(root),
                        "name": root.file_name().and_then(|n| n.to_str()).unwrap_or("workspace")
                    }],
                    "capabilities": {
                        "textDocument": {
                            "hover": { "contentFormat": ["plaintext", "markdown"] },
                            "definition": { "linkSupport": false },
                            "references": {},
                            "documentSymbol": { "hierarchicalDocumentSymbolSupport": true },
                            "publishDiagnostics": {},
                            "rename": { "prepareSupport": true },
                        },
                        "workspace": {
                            "symbol": {},
                            "workspaceFolders": true,
                            // `documentChanges` is what makes a rename legible:
                            // edits arrive grouped per file, and file-level
                            // operations (renaming a module's file) are
                            // declared rather than left implicit.
                            "workspaceEdit": {
                                "documentChanges": true,
                                "resourceOperations": ["rename", "create", "delete"],
                            },
                        },
                    },
                }),
            )
            .await?;
        client.notify("initialized", json!({})).await?;
        debug!(language = spec.language_id, "language server ready");
        Ok(client)
    }

    async fn send(&self, message: &Value) -> Result<()> {
        let body = serde_json::to_string(message)?;
        let framed = format!("Content-Length: {}\r\n\r\n{}", body.len(), body);
        let mut stdin = self.stdin.lock().await;
        stdin.write_all(framed.as_bytes()).await?;
        stdin.flush().await?;
        Ok(())
    }

    pub async fn notify(&self, method: &str, params: Value) -> Result<()> {
        self.send(&json!({ "jsonrpc": "2.0", "method": method, "params": params }))
            .await
    }

    pub async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);

        self.send(&json!({
            "jsonrpc": "2.0", "id": id, "method": method, "params": params
        }))
        .await?;

        match timeout(REQUEST_TIMEOUT, rx).await {
            Ok(Ok(Ok(value))) => Ok(value),
            Ok(Ok(Err(e))) => Err(anyhow!("language server error: {e}")),
            Ok(Err(_)) => Err(anyhow!("language server closed the connection")),
            Err(_) => {
                self.pending.lock().await.remove(&id);
                Err(anyhow!("language server timed out on '{method}'"))
            }
        }
    }

    /// Tell the server about a file (idempotent), which is required before
    /// position-based requests and triggers diagnostics.
    pub async fn open_file(&self, path: &Path) -> Result<()> {
        let uri = path_to_uri(path);
        {
            let opened = self.opened.lock().await;
            if opened.contains_key(&uri) {
                return Ok(());
            }
        }
        let text = tokio::fs::read_to_string(path).await.unwrap_or_default();
        self.notify(
            "textDocument/didOpen",
            json!({
                "textDocument": {
                    "uri": uri,
                    "languageId": self.language_id,
                    "version": 1,
                    "text": text,
                }
            }),
        )
        .await?;
        self.opened.lock().await.insert(uri, ());
        Ok(())
    }

    /// Notify the server that a file changed on disk so diagnostics refresh.
    pub async fn file_changed(&self, path: &Path) -> Result<()> {
        let uri = path_to_uri(path);
        let is_open = self.opened.lock().await.contains_key(&uri);
        if !is_open {
            return self.open_file(path).await;
        }
        let text = tokio::fs::read_to_string(path).await.unwrap_or_default();
        self.notify(
            "textDocument/didChange",
            json!({
                "textDocument": { "uri": uri, "version": 2 },
                "contentChanges": [{ "text": text }],
            }),
        )
        .await
    }

    /// Diagnostics the server has published for `path` (possibly empty).
    pub async fn diagnostics_for(&self, path: &Path) -> Vec<Value> {
        self.diagnostics
            .lock()
            .await
            .get(&path_to_uri(path))
            .cloned()
            .unwrap_or_default()
    }

    /// All diagnostics currently known, as `(path, diagnostics)`.
    pub async fn all_diagnostics(&self) -> Vec<(String, Vec<Value>)> {
        self.diagnostics
            .lock()
            .await
            .iter()
            .filter(|(_, d)| !d.is_empty())
            .map(|(uri, d)| (uri_to_path(uri), d.clone()))
            .collect()
    }
}

/// Read framed LSP messages, routing responses to waiters and recording
/// diagnostics from notifications.
async fn read_loop(mut reader: BufReader<ChildStdout>, pending: Pending, diagnostics: Diagnostics) {
    loop {
        // Headers, terminated by a blank line.
        let mut content_length: Option<usize> = None;
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line).await {
                Ok(0) => return, // server exited
                Ok(_) => {}
                Err(e) => {
                    warn!("lsp read error: {e}");
                    return;
                }
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                break;
            }
            if let Some(value) = trimmed.strip_prefix("Content-Length:") {
                content_length = value.trim().parse().ok();
            }
        }

        let Some(len) = content_length else { continue };
        let mut buf = vec![0u8; len];
        if reader.read_exact(&mut buf).await.is_err() {
            return;
        }
        let Ok(message) = serde_json::from_slice::<Value>(&buf) else {
            continue;
        };

        // Response to one of our requests?
        if let Some(id) = message.get("id").and_then(|v| v.as_i64()) {
            if let Some(tx) = pending.lock().await.remove(&id) {
                let outcome = if let Some(error) = message.get("error") {
                    Err(error
                        .get("message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("unknown")
                        .to_string())
                } else {
                    Ok(message.get("result").cloned().unwrap_or(Value::Null))
                };
                let _ = tx.send(outcome);
                continue;
            }
        }

        // Server-initiated notification.
        if message.get("method").and_then(|m| m.as_str()) == Some("textDocument/publishDiagnostics")
        {
            let params = message.get("params").cloned().unwrap_or(Value::Null);
            if let Some(uri) = params.get("uri").and_then(|u| u.as_str()) {
                let items = params
                    .get("diagnostics")
                    .and_then(|d| d.as_array())
                    .cloned()
                    .unwrap_or_default();
                diagnostics.lock().await.insert(uri.to_string(), items);
            }
        }
    }
}

// ── Pool ──────────────────────────────────────────────────────────────────────

/// Lazily starts and reuses one language server per language.
///
/// Servers that are not installed are remembered as unavailable so we do not
/// retry spawning them on every call.
pub struct LspPool {
    root: PathBuf,
    clients: Mutex<HashMap<String, Option<Arc<LspClient>>>>,
    /// Whether this pool may start servers at all. See [`LspPool::disabled`].
    servers: bool,
}

impl LspPool {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            clients: Mutex::new(HashMap::new()),
            servers: true,
        }
    }

    /// A pool that never starts anything, for callers that only need a pool to
    /// exist.
    ///
    /// The editing tools each hold a pool and notify it after a write. A test
    /// that builds a tool per operation therefore builds a pool per operation,
    /// and every one of them spawned a real `rust-analyzer` and waited on its
    /// `initialize` — up to this module's 20-second request timeout. On a developer
    /// machine each answers in milliseconds and this is invisible. On a
    /// two-core CI runner it did not answer at all: a test doing three writes
    /// took just over sixty seconds — three timeouts, one per pool — and one
    /// doing twenty iterations of three concurrent edits was still running
    /// when the job hit its step limit and was cancelled.
    ///
    /// `rename_live.rs` is `#[ignore]`d for exactly this reason. That covers a
    /// test whose *subject* is the language server; it could not cover a
    /// server started as a side effect of writing a file.
    ///
    /// So a test whose subject is not the language server says so, and gets a
    /// pool that cannot start one. Tests that *are* about the LSP keep
    /// [`LspPool::new`].
    pub fn disabled(root: impl Into<PathBuf>) -> Self {
        Self {
            servers: false,
            ..Self::new(root)
        }
    }

    /// Get (or start) the server responsible for `path`.
    ///
    /// Returns `None` when the language is unsupported or its server is not
    /// installed — callers should degrade to text search.
    pub async fn for_path(&self, path: &Path) -> Option<Arc<LspClient>> {
        if !self.servers {
            return None;
        }
        let spec = server_for(path)?;
        let mut clients = self.clients.lock().await;
        if let Some(entry) = clients.get(spec.language_id) {
            return entry.clone();
        }
        let started = match LspClient::start(spec, &self.root).await {
            Ok(client) => Some(Arc::new(client)),
            Err(e) => {
                debug!(
                    language = spec.language_id,
                    "language server unavailable: {e}"
                );
                None
            }
        };
        clients.insert(spec.language_id.to_string(), started.clone());
        started
    }

    /// Notify every running server that a file changed.
    pub async fn notify_changed(&self, path: &Path) {
        if let Some(client) = self.for_path(path).await {
            let _ = client.file_changed(path).await;
        }
    }

    /// Diagnostics across all running servers.
    pub async fn all_diagnostics(&self) -> Vec<(String, Vec<Value>)> {
        let clients: Vec<Arc<LspClient>> = self
            .clients
            .lock()
            .await
            .values()
            .filter_map(|c| c.clone())
            .collect();
        let mut out = Vec::new();
        for client in clients {
            out.extend(client.all_diagnostics().await);
        }
        out
    }
}

/// Render an LSP diagnostic as a compact `severity line:col message` line.
pub fn format_diagnostic(diag: &Value) -> String {
    let severity = match diag.get("severity").and_then(|s| s.as_u64()) {
        Some(1) => "error",
        Some(2) => "warning",
        Some(3) => "info",
        Some(4) => "hint",
        _ => "note",
    };
    let line = diag
        .pointer("/range/start/line")
        .and_then(|l| l.as_u64())
        .map(|l| l + 1)
        .unwrap_or(0);
    let col = diag
        .pointer("/range/start/character")
        .and_then(|c| c.as_u64())
        .map(|c| c + 1)
        .unwrap_or(0);
    let message = diag
        .get("message")
        .and_then(|m| m.as_str())
        .unwrap_or("")
        .replace('\n', " ");
    format!("{severity} [{line}:{col}] {message}")
}

/// Render an LSP `Location`/`LocationLink` as `path:line:col`.
pub fn format_location(location: &Value) -> Option<String> {
    let uri = location
        .get("uri")
        .or_else(|| location.get("targetUri"))
        .and_then(|u| u.as_str())?;
    let range = location
        .get("range")
        .or_else(|| location.get("targetSelectionRange"))
        .or_else(|| location.get("targetRange"))?;
    let line = range.pointer("/start/line").and_then(|l| l.as_u64())? + 1;
    let col = range.pointer("/start/character").and_then(|c| c.as_u64())? + 1;
    Some(format!("{}:{}:{}", uri_to_path(uri), line, col))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_servers_by_extension() {
        assert_eq!(
            server_for(Path::new("/a/b.rs")).unwrap().language_id,
            "rust"
        );
        assert_eq!(
            server_for(Path::new("/a/b.tsx")).unwrap().language_id,
            "typescript"
        );
        assert_eq!(
            server_for(Path::new("/a/b.py")).unwrap().language_id,
            "python"
        );
        assert!(server_for(Path::new("/a/README.md")).is_none());
        assert!(server_for(Path::new("/a/noext")).is_none());
    }

    #[test]
    fn uri_round_trip() {
        let path = Path::new("/repo/src/main.rs");
        let uri = path_to_uri(path);
        assert_eq!(uri, "file:///repo/src/main.rs");
        assert_eq!(uri_to_path(&uri), "/repo/src/main.rs");
    }

    #[test]
    fn formats_diagnostics_one_based() {
        let diag = json!({
            "severity": 1,
            "range": { "start": { "line": 9, "character": 4 } },
            "message": "cannot find value `x`"
        });
        assert_eq!(
            format_diagnostic(&diag),
            "error [10:5] cannot find value `x`"
        );
    }

    #[test]
    fn formats_locations_and_links() {
        let location = json!({
            "uri": "file:///repo/src/lib.rs",
            "range": { "start": { "line": 41, "character": 7 } }
        });
        assert_eq!(format_location(&location).unwrap(), "/repo/src/lib.rs:42:8");

        let link = json!({
            "targetUri": "file:///repo/src/lib.rs",
            "targetSelectionRange": { "start": { "line": 0, "character": 0 } }
        });
        assert_eq!(format_location(&link).unwrap(), "/repo/src/lib.rs:1:1");
    }

    #[tokio::test]
    async fn missing_server_degrades_to_none() {
        let pool = LspPool::new("/tmp");
        // .md has no configured server at all.
        assert!(pool.for_path(Path::new("/tmp/x.md")).await.is_none());
    }

    #[tokio::test]
    async fn a_disabled_pool_starts_nothing_even_where_a_server_exists() {
        // `.rs` is the case that matters: a configured server, which on a
        // machine that has rust-analyzer installed would really be spawned and
        // really be waited on. Asserting against `.md` would pass whether the
        // switch worked or not.
        assert!(server_for(Path::new("/tmp/x.rs")).is_some(), "premise");

        let pool = LspPool::disabled("/tmp");
        assert!(pool.for_path(Path::new("/tmp/x.rs")).await.is_none());
        assert!(
            pool.clients.lock().await.is_empty(),
            "a disabled pool must not even record an attempt"
        );
    }
}
