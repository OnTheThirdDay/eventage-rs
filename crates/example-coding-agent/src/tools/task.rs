//! Subagent delegation.
//!
//! A subagent gets its **own bus, own context, and own tool set**, so a wide
//! investigation ("find every call site of X and summarise the patterns")
//! costs the parent only the final report instead of hundreds of tool
//! results. That is the whole point: context is the scarce resource.
//!
//! Implementation subagents can run in an isolated **git worktree**, so
//! several of them may edit the same repository concurrently without
//! colliding; the parent reviews and integrates their diffs afterwards.

use crate::config::PermissionMode;
use crate::lsp::LspPool;
use crate::tools;
use crate::tools::intel;
use crate::workspace::Workspace;
use async_trait::async_trait;
use eventage::agent::{AgentError, DefaultContextAssembler, Tool};
use eventage::event::kinds;
use eventage::llm::{LlmProvider, ToolDefinition};
use eventage::{AgentBuilder, Event, EventBus, ReactStrategy};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::{info, warn};

/// Deepest nesting allowed, so a runaway agent cannot fork forever.
pub const MAX_DEPTH: usize = 3;

/// What kind of helper to spawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubagentKind {
    /// Read-only, search-optimised sweep of the codebase.
    Explore,
    /// Read-only design work that returns a plan.
    Plan,
    /// Full tool access; may edit (optionally inside a worktree).
    General,
}

impl SubagentKind {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "explore" => Some(Self::Explore),
            "plan" => Some(Self::Plan),
            "general" => Some(Self::General),
            _ => None,
        }
    }

    fn can_edit(&self) -> bool {
        matches!(self, Self::General)
    }

    fn prompt(&self) -> &'static str {
        match self {
            Self::Explore => {
                "You are an exploration subagent. Search the codebase and report what \
                 you find. You cannot edit anything. Be exhaustive in searching but \
                 concise in reporting: give file:line references and the shape of the \
                 code, not long excerpts. Your report is the only thing your caller \
                 sees, so make it self-contained."
            }
            Self::Plan => {
                "You are a planning subagent. Investigate, then return a concrete, \
                 ordered implementation plan with the specific files and functions to \
                 change. You cannot edit anything. Call out risks and unknowns \
                 explicitly rather than glossing over them."
            }
            Self::General => {
                "You are an implementation subagent working in an isolated copy of the \
                 repository. Complete the assigned task, verify it (build/tests where \
                 possible), and report exactly what you changed and what you verified. \
                 Report failures honestly — your caller will review your diff."
            }
        }
    }
}

/// A checked-out git worktree that is removed when dropped.
struct Worktree {
    path: std::path::PathBuf,
    repo: std::path::PathBuf,
    branch: String,
}

impl Worktree {
    /// Create a detached worktree off HEAD.
    async fn create(repo: &std::path::Path) -> Result<Self, AgentError> {
        let id = uuid::Uuid::new_v4().to_string();
        let branch = format!("eventage-subagent/{}", &id[..8]);
        let path = std::env::temp_dir().join(format!("eventage-wt-{}", &id[..8]));

        let output = tokio::process::Command::new("git")
            .args([
                "worktree",
                "add",
                "-b",
                &branch,
                &path.display().to_string(),
                "HEAD",
            ])
            .current_dir(repo)
            .output()
            .await
            .map_err(|e| AgentError::Tool(format!("git worktree unavailable: {e}")))?;

        if !output.status.success() {
            return Err(AgentError::Tool(format!(
                "could not create worktree: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        Ok(Self {
            path,
            repo: repo.to_path_buf(),
            branch,
        })
    }

    /// Diff of everything the subagent changed in the worktree.
    async fn diff(&self) -> String {
        let mut out = String::new();
        for args in [
            vec!["add", "-A"],
            vec!["diff", "--cached", "--no-color"],
        ] {
            if let Ok(o) = tokio::process::Command::new("git")
                .args(&args)
                .current_dir(&self.path)
                .output()
                .await
            {
                out.push_str(&String::from_utf8_lossy(&o.stdout));
            }
        }
        out
    }

}

impl Drop for Worktree {
    /// Now that a subagent outlives its first call, the checkout has to
    /// survive with it — and be removed however the session ends, including
    /// a crash. Blocking here is acceptable: it is two short git invocations
    /// at teardown.
    fn drop(&mut self) {
        let _ = std::process::Command::new("git")
            .args([
                "worktree",
                "remove",
                "--force",
                &self.path.display().to_string(),
            ])
            .current_dir(&self.repo)
            .output();
        let _ = std::process::Command::new("git")
            .args(["branch", "-D", &self.branch])
            .current_dir(&self.repo)
            .output();
    }
}

// ── the live subagents of one session ─────────────────────────────────────────

/// How many subagents may stay resident at once.
///
/// Each holds an agent, a bus and possibly a git checkout, so they are not
/// free; the oldest idle one is retired when the cap is reached.
const MAX_LIVE: usize = 8;

/// A subagent that is still running, between calls.
struct Live {
    agent: Arc<eventage::agent::Agent>,
    bus: EventBus,
    kind: SubagentKind,
    /// Kept alive with the subagent so a follow-up sees its own earlier work.
    worktree: Option<Worktree>,
    turns: usize,
    /// Where the transcript had reached last time we reported on it, so each
    /// turn reports what *it* did rather than the whole history again.
    reported_upto: usize,
}

/// The subagents belonging to one session.
///
/// Owned by the session, so they die with it — which is what makes it safe to
/// keep them alive at all. A dropped registry drops each agent, its bus, and
/// its worktree.
#[derive(Default)]
pub struct SubagentRegistry {
    live: tokio::sync::Mutex<std::collections::HashMap<String, Live>>,
    next: AtomicUsize,
}

impl SubagentRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ids people can actually read back to you: `explore-1`, `general-2`.
    fn mint(&self, kind: SubagentKind) -> String {
        let n = self.next.fetch_add(1, Ordering::Relaxed) + 1;
        format!("{}-{n}", format!("{kind:?}").to_lowercase())
    }

    async fn insert(&self, id: String, live: Live) {
        let mut map = self.live.lock().await;
        if map.len() >= MAX_LIVE {
            // Retire the least advanced one; it has the least to lose.
            if let Some(oldest) = map
                .iter()
                .min_by_key(|(_, l)| l.turns)
                .map(|(k, _)| k.clone())
            {
                map.remove(&oldest);
            }
        }
        map.insert(id, live);
    }

    /// How many are resident, for the tool description and for tests.
    pub async fn count(&self) -> usize {
        self.live.lock().await.len()
    }
}

/// The `task` tool.
pub struct Task {
    pub ws: Arc<Workspace>,
    pub llm: Arc<dyn LlmProvider>,
    /// How deep the *current* agent is; children run at `depth + 1`.
    pub depth: usize,
    /// Concurrent subagents currently running, for observability.
    pub active: Arc<AtomicUsize>,
    /// The permission mode of the session this subagent belongs to.
    pub mode: PermissionMode,
    /// The parent's bus, so what a subagent did lands in the session's trace.
    pub bus: EventBus,
    /// The subagents of this session, kept alive between calls.
    pub registry: Arc<SubagentRegistry>,
}

/// What a subagent did, read back off its own event log.
///
/// The report is the subagent's account of itself, which is worth having but
/// is not evidence. This is the record: the tools it actually called, the
/// files it actually touched, what failed. It goes back to the caller in the
/// tool result and onto the parent's bus, so it survives in the trace.
fn digest(log: &[Event], from: usize) -> Value {
    let mut tools: std::collections::BTreeMap<String, usize> = Default::default();
    let mut files: Vec<String> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    let mut denials: Vec<String> = Vec::new();

    for event in log.iter().skip(from) {
        let name = |key: &str| {
            event
                .payload
                .get(key)
                .and_then(|v| v.as_str())
                .unwrap_or("?")
                .to_string()
        };
        match event.kind.as_str() {
            kinds::TOOL_CALL_PROPOSED => *tools.entry(name("name")).or_default() += 1,
            kinds::TOOL_RESULT => {
                if let Some(error) = event.payload.get("error").and_then(|e| e.as_str()) {
                    failures.push(format!("{}: {error}", name("name")));
                }
                // `_locations` is the same convention the editor reads, so a
                // tool that already reports what it touched needs no changes.
                if let Some(locations) = event
                    .payload
                    .get("result")
                    .and_then(|r| r.get("_locations"))
                    .and_then(|l| l.as_array())
                {
                    for location in locations {
                        if let Some(path) = location.get("path").and_then(|p| p.as_str()) {
                            if !files.iter().any(|f| f == path) {
                                files.push(path.to_string());
                            }
                        }
                    }
                }
            }
            kinds::PERMISSION_DECISION
                if event.payload.get("approved").and_then(|a| a.as_bool()) == Some(false) =>
            {
                denials.push(name("tool"));
            }
            _ => {}
        }
    }

    json!({
        "tools_used": tools,
        "files_touched": files,
        "failures": failures,
        "refused": denials,
    })
}

#[async_trait]
impl Tool for Task {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "task",
            "Delegate a self-contained piece of work to a subagent with its own \
             context. Use it for wide investigation (so the search results never enter \
             your context) or for parallel implementation in an isolated worktree. \
             Give a complete, standalone brief — the subagent cannot see your \
             conversation.\n\n\
             The subagent stays alive afterwards. Its reply comes back with a \
             `subagent_id`; pass that id with a new `prompt` to ask a follow-up, and \
             it answers with its full memory of what it already did — cheaper and far \
             more accurate than briefing a fresh one. Push back on a thin report \
             rather than accepting it.",
            json!({
                "type": "object",
                "properties": {
                    "description": { "type": "string", "description": "Short label for the UI" },
                    "prompt": {
                        "type": "string",
                        "description": "The brief, or the follow-up when continuing"
                    },
                    "subagent_type": {
                        "type": "string",
                        "enum": ["explore", "plan", "general"],
                        "description": "explore/plan are read-only; general may edit. Omit when continuing."
                    },
                    "subagent_id": {
                        "type": "string",
                        "description": "Continue this subagent instead of starting a new one"
                    },
                    "isolated": {
                        "type": "boolean",
                        "description": "Run in a throwaway git worktree (general only) and return a diff"
                    }
                },
                "required": ["prompt"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        if self.depth >= MAX_DEPTH {
            return Err(AgentError::Tool(format!(
                "subagent nesting limit ({MAX_DEPTH}) reached; do this work yourself"
            )));
        }

        let brief = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::Tool("missing 'prompt'".into()))?
            .to_string();

        // A follow-up goes to the subagent that already did the work, with
        // everything it learned still in its own context.
        if let Some(id) = args.get("subagent_id").and_then(|v| v.as_str()) {
            return self.continue_with(id, &brief).await;
        }

        let kind = args
            .get("subagent_type")
            .and_then(|v| v.as_str())
            .and_then(SubagentKind::from_str)
            .ok_or_else(|| {
                AgentError::Tool(
                    "subagent_type must be explore, plan, or general when starting a \
                     new subagent (pass subagent_id to continue an existing one)"
                        .into(),
                )
            })?;
        let isolated = args.get("isolated").and_then(|v| v.as_bool()).unwrap_or(false)
            && kind.can_edit();

        // Isolated runs get their own checkout so concurrent edits cannot collide.
        let worktree = if isolated {
            Some(Worktree::create(self.ws.root()).await?)
        } else {
            None
        };
        let root = worktree
            .as_ref()
            .map(|w| w.path.clone())
            .unwrap_or_else(|| self.ws.root().to_path_buf());

        let ws = Arc::new(
            Workspace::open(&root).map_err(|e| AgentError::Tool(e.to_string()))?,
        );
        let lsp = Arc::new(LspPool::new(&root));
        let bus = EventBus::new();

        // A subagent used to run with no policy at all. It inherits the
        // session's now, minus anything that would need a human to answer —
        // there is no UI on the other end of this bus.
        let mut builder = AgentBuilder::new()
            .agent_id(format!("subagent-{:?}", kind).to_lowercase())
            .bus(bus.clone())
            .llm_arc(Arc::clone(&self.llm))
            .hook(self.mode.subagent_policy(isolated))
            .context(DefaultContextAssembler::new(kind.prompt()))
            .strategy(ReactStrategy {
                max_steps: 40,
                max_concurrent_tools: 4,
                ..Default::default()
            })
            // Subagents work in their own checkout, which the editor does not
            // have open — so file I/O goes straight to disk.
            .tool(tools::ReadFile {
                ws: ws.clone(),
                client: None,
            })
            .tool(tools::Glob { ws: ws.clone() })
            .tool(tools::Grep { ws: ws.clone() })
            .tool(tools::ListDirectory { ws: ws.clone() })
            .tool(intel::LspDefinition {
                ws: ws.clone(),
                lsp: lsp.clone(),
            })
            .tool(intel::LspReferences {
                ws: ws.clone(),
                lsp: lsp.clone(),
            })
            .tool(intel::LspSymbols {
                ws: ws.clone(),
                lsp: lsp.clone(),
            });

        if kind.can_edit() {
            builder = builder
                .tool(tools::WriteFile {
                    ws: ws.clone(),
                    client: None,
                    lsp: lsp.clone(),
                })
                .tool(tools::EditFile {
                    ws: ws.clone(),
                    client: None,
                    lsp: lsp.clone(),
                })
                .tool(tools::MultiEdit {
                    ws: ws.clone(),
                    client: None,
                    lsp: lsp.clone(),
                })
                .tool(tools::Bash {
                    ws: ws.clone(),
                    jobs: Arc::new(tools::BackgroundJobs::default()),
                })
                .tool(intel::LspDiagnostics {
                    ws: ws.clone(),
                    lsp: lsp.clone(),
                })
                .tool(intel::LspRename {
                    ws: ws.clone(),
                    lsp: lsp.clone(),
                    client: None,
                });
        }

        let mut live = Live {
            agent: Arc::new(builder.build()),
            bus,
            kind,
            worktree,
            turns: 0,
            reported_upto: 0,
        };

        let id = self.registry.mint(kind);
        info!(%id, ?kind, isolated, "subagent starting");
        let result = self.run_turn(&id, &mut live, &brief).await?;
        self.registry.insert(id, live).await;
        Ok(result)
    }
}

impl Task {
    /// Push a message into a subagent, run it, and report what it did.
    ///
    /// Shared by the first brief and every follow-up, so a continuation is
    /// reported in exactly the same shape as the original call.
    async fn run_turn(
        &self,
        id: &str,
        live: &mut Live,
        message: &str,
    ) -> Result<Value, AgentError> {
        live.bus
            .publish(Event::new(kinds::USER_MESSAGE, json!({ "text": message })))
            .await
            .map_err(|e| AgentError::Tool(e.to_string()))?;

        self.active.fetch_add(1, Ordering::Relaxed);
        let outcome = live.agent.cycle().await;
        self.active.fetch_sub(1, Ordering::Relaxed);
        live.turns += 1;

        let log = live.bus.log().await;
        let report = log
            .iter()
            .rev()
            .find(|e| e.kind == kinds::ASSISTANT_MESSAGE)
            .and_then(|e| e.payload.get("content").and_then(|c| c.as_str()))
            .unwrap_or("(the subagent produced no report)")
            .to_string();

        // What it did *this* turn, not since the beginning.
        let state = digest(&log, live.reported_upto);
        live.reported_upto = log.len();

        let mut result = json!({
            "subagent_id": id,
            "subagent_type": format!("{:?}", live.kind).to_lowercase(),
            "turn": live.turns,
            "report": report,
            "state": state,
            "still_available": true,
            "note": format!(
                "This subagent is still alive with its full context. To ask it \
                 anything else — including to justify or correct the above — call \
                 task again with subagent_id \"{id}\" and a new prompt."
            ),
        });

        if let Some(wt) = &live.worktree {
            let diff = wt.diff().await;
            result["diff"] = json!(diff.chars().take(60_000).collect::<String>());
            result["worktree_branch"] = json!(wt.branch);
        }

        if let Err(e) = outcome {
            warn!(%id, "subagent failed: {e}");
            result["error"] = json!(e.to_string());
        }

        // The parent's log gets the record too, so the trace shows what the
        // subagent did rather than only what it claimed.
        self.bus.broadcast(Event::new(
            "subagent.turn",
            json!({
                "subagent_id": id,
                "subagent_type": format!("{:?}", live.kind).to_lowercase(),
                "turn": live.turns,
                "prompt": message,
                "report": result["report"].clone(),
                "state": result["state"].clone(),
            }),
        ));

        Ok(result)
    }

    /// Ask a subagent that is already running a follow-up question.
    async fn continue_with(&self, id: &str, message: &str) -> Result<Value, AgentError> {
        // Taken out of the map for the duration of the turn: cycling holds it
        // for a long time, and nothing else should be able to interleave a
        // second message into the same conversation.
        let mut live = {
            let mut map = self.registry.live.lock().await;
            map.remove(id).ok_or_else(|| {
                let known: Vec<&str> = map.keys().map(String::as_str).collect();
                AgentError::Tool(if known.is_empty() {
                    format!("no subagent '{id}' is running; start one with subagent_type")
                } else {
                    format!(
                        "no subagent '{id}' is running. Still available: {}",
                        known.join(", ")
                    )
                })
            })?
        };

        info!(%id, turn = live.turns + 1, "subagent continuing");
        let result = self.run_turn(id, &mut live, message).await;
        self.registry.insert(id.to_string(), live).await;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eventage::llm::MockLlmProvider;

    /// A tool whose subagents answer with `replies`, one per turn.
    fn task_tool_with(depth: usize, replies: Vec<&str>) -> (tempfile::TempDir, Task) {
        let dir = tempfile::tempdir().unwrap();
        let tool = Task {
            ws: Arc::new(Workspace::open(dir.path()).unwrap()),
            llm: Arc::new(MockLlmProvider::with_texts(replies)),
            depth,
            active: Arc::new(AtomicUsize::new(0)),
            mode: PermissionMode::Auto,
            bus: EventBus::new(),
            registry: Arc::new(SubagentRegistry::new()),
        };
        (dir, tool)
    }

    fn task_tool(depth: usize) -> Task {
        task_tool_with(depth, vec!["done"]).1
    }

    #[tokio::test]
    async fn a_subagent_stays_alive_and_remembers_its_own_work() {
        // The report used to be all that survived: bus and agent were locals
        // inside `execute`, so a follow-up meant briefing a fresh copy that
        // had never seen the codebase.
        let (_dir, tool) = task_tool_with(0, vec!["I found the parser", "It is in src/lex.rs"]);

        let first = tool
            .execute(json!({ "prompt": "find the parser", "subagent_type": "explore" }))
            .await
            .unwrap();
        let id = first["subagent_id"].as_str().unwrap().to_string();
        assert_eq!(first["turn"], 1);
        assert_eq!(first["still_available"], true);
        assert_eq!(tool.registry.count().await, 1);

        let second = tool
            .execute(json!({ "prompt": "where exactly?", "subagent_id": id.clone() }))
            .await
            .unwrap();
        assert_eq!(second["turn"], 2, "the same subagent, one turn later");
        assert_eq!(second["subagent_id"], json!(id));
        assert_eq!(second["report"], "It is in src/lex.rs");
        // Still one: continuing must not spawn a second.
        assert_eq!(tool.registry.count().await, 1);
    }

    #[tokio::test]
    async fn continuing_a_subagent_that_is_gone_says_which_ones_are_left() {
        let (_dir, tool) = task_tool_with(0, vec!["done"]);
        tool.execute(json!({ "prompt": "go", "subagent_type": "explore" }))
            .await
            .unwrap();

        let err = tool
            .execute(json!({ "prompt": "more", "subagent_id": "explore-99" }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("explore-1"), "should name what is available: {err}");
    }

    #[tokio::test]
    async fn the_caller_is_told_what_the_subagent_did_not_only_what_it_says() {
        // A report is the subagent's account of itself. The digest is read
        // off its event log, so it cannot flatter itself in it.
        let (_dir, tool) = task_tool_with(0, vec!["all done"]);
        let out = tool
            .execute(json!({ "prompt": "go", "subagent_type": "explore" }))
            .await
            .unwrap();

        let state = &out["state"];
        assert!(state["tools_used"].is_object(), "{state}");
        assert!(state["files_touched"].is_array(), "{state}");
        assert!(state["failures"].is_array(), "{state}");
        assert!(state["refused"].is_array(), "{state}");
    }

    #[tokio::test]
    async fn what_a_subagent_did_reaches_the_parent_trace() {
        let (_dir, tool) = task_tool_with(0, vec!["found it"]);
        let mut seen = tool.bus.subscribe();

        tool.execute(json!({ "prompt": "go", "subagent_type": "explore" }))
            .await
            .unwrap();

        let event = tokio::time::timeout(std::time::Duration::from_secs(2), seen.recv())
            .await
            .expect("nothing reached the parent bus")
            .unwrap();
        assert_eq!(event.kind, "subagent.turn");
        assert_eq!(event.payload["report"], "found it");
        assert!(event.payload["state"].is_object());
    }

    #[tokio::test]
    async fn subagents_are_capped_so_a_session_cannot_hoard_them() {
        let (_dir, tool) = task_tool_with(0, vec!["ok"; MAX_LIVE + 3]);
        for _ in 0..MAX_LIVE + 3 {
            tool.execute(json!({ "prompt": "go", "subagent_type": "explore" }))
                .await
                .unwrap();
        }
        assert_eq!(tool.registry.count().await, MAX_LIVE);
    }

    #[test]
    fn kinds_control_edit_access() {
        assert!(!SubagentKind::Explore.can_edit());
        assert!(!SubagentKind::Plan.can_edit());
        assert!(SubagentKind::General.can_edit());
        assert_eq!(SubagentKind::from_str("explore"), Some(SubagentKind::Explore));
        assert_eq!(SubagentKind::from_str("nope"), None);
    }

    #[tokio::test]
    async fn nesting_is_bounded() {
        let tool = task_tool(MAX_DEPTH);
        let err = tool
            .execute(json!({ "prompt": "go", "subagent_type": "explore" }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("nesting limit"), "{err}");
    }

    #[tokio::test]
    async fn rejects_unknown_subagent_type() {
        let tool = task_tool(0);
        let err = tool
            .execute(json!({ "prompt": "go", "subagent_type": "wizard" }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("explore, plan, or general"), "{err}");
        assert!(err.to_string().contains("subagent_id"), "and how to continue: {err}");
    }

    #[tokio::test]
    async fn explore_subagent_returns_only_its_report() {
        let tool = task_tool(0);
        let out = tool
            .execute(json!({ "prompt": "find the parser", "subagent_type": "explore" }))
            .await
            .unwrap();
        // The caller sees the report, not the subagent's intermediate steps.
        assert_eq!(out["report"], "done");
        assert_eq!(out["subagent_type"], "explore");
        assert!(out.get("diff").is_none(), "no worktree for read-only kinds");
    }
}
