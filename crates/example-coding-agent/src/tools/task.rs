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
                 repository. Complete the assigned task, verify it, and report exactly \
                 what you changed and what you verified. Use `verify` to run the \
                 project's build and tests — it needs no approval. `bash` is available \
                 for anything else, but it asks the user, so prefer `verify` and do not \
                 reach for the shell out of habit. Report failures honestly: your \
                 caller will review your diff."
            }
        }
    }
}

/// A checked-out git worktree that is removed when dropped.
struct Worktree {
    path: std::path::PathBuf,
    repo: std::path::PathBuf,
    /// The snapshot commit the checkout was made from, so a diff against it
    /// shows what the subagent changed and nothing else.
    snapshot: String,
}

impl Worktree {
    /// Check out a copy of the repository *as it is now*.
    ///
    /// Not `HEAD`. A subagent branched from the last commit sees none of the
    /// user's uncommitted work — not their staged changes, not their
    /// unstaged edits, not the file they just created — so it reads an older
    /// version of the code, implements against interfaces that have already
    /// changed, and hands back a patch that conflicts with the tree it was
    /// supposedly written for. Tests pass or fail for the wrong reasons.
    ///
    /// The snapshot is content-addressed and built through a **separate
    /// index**, so the user's staging area is never touched: read HEAD's tree
    /// into a scratch index, stage everything on top (which picks up
    /// untracked files while still honouring `.gitignore`), write it out as a
    /// tree, and commit that tree with HEAD as its parent. The commit is
    /// never on a branch and never pushed; it exists so a worktree can be
    /// checked out from it.
    ///
    /// One thing it still cannot see: a buffer the user has open and unsaved
    /// in their editor. That would have to come from the client over ACP.
    async fn create(repo: &std::path::Path) -> Result<Self, AgentError> {
        let id = uuid::Uuid::new_v4().to_string();
        let path = std::env::temp_dir().join(format!("eventage-wt-{}", &id[..8]));
        let index = std::env::temp_dir().join(format!("eventage-idx-{}", &id[..8]));

        let git = |args: Vec<String>, with_index: bool| {
            let mut cmd = tokio::process::Command::new("git");
            cmd.args(args).current_dir(repo);
            if with_index {
                cmd.env("GIT_INDEX_FILE", &index);
            }
            cmd
        };
        let run = |mut cmd: tokio::process::Command, what: &'static str| async move {
            let out = cmd
                .output()
                .await
                .map_err(|e| AgentError::Tool(format!("git unavailable: {e}")))?;
            if !out.status.success() {
                return Err(AgentError::Tool(format!(
                    "could not {what}: {}",
                    String::from_utf8_lossy(&out.stderr).trim()
                )));
            }
            Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
        };

        let args = |list: &[&str]| list.iter().map(|s| s.to_string()).collect::<Vec<_>>();

        // A repository with no commits yet has no HEAD to read or parent.
        let head = run(git(args(&["rev-parse", "HEAD"]), false), "read HEAD")
            .await
            .ok();

        if head.is_some() {
            run(
                git(args(&["read-tree", "HEAD"]), true),
                "read the current tree",
            )
            .await?;
        }
        run(git(args(&["add", "-A"]), true), "stage the working tree").await?;
        let tree = run(git(args(&["write-tree"]), true), "record the working tree").await?;
        // The scratch index has done its job.
        let _ = tokio::fs::remove_file(&index).await;

        let mut commit_args = args(&["commit-tree", &tree, "-m", "eventage subagent snapshot"]);
        if let Some(parent) = &head {
            commit_args.push("-p".into());
            commit_args.push(parent.clone());
        }
        let commit = run(git(commit_args, false), "record a snapshot commit").await?;

        // Detached: no branch is created, so none can be orphaned.
        run(
            git(
                args(&[
                    "worktree",
                    "add",
                    "--detach",
                    &path.display().to_string(),
                    &commit,
                ]),
                false,
            ),
            "create the worktree",
        )
        .await?;

        Ok(Self {
            path,
            repo: repo.to_path_buf(),
            snapshot: commit,
        })
    }

    /// Diff of everything the subagent changed in the worktree.
    async fn diff(&self) -> String {
        let mut out = String::new();
        for args in [vec!["add", "-A"], vec!["diff", "--cached", "--no-color"]] {
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
    /// Removed when the subagent that owns it is dropped, which is when the
    /// session ends.
    ///
    /// Not crash-proof, and the comment here used to claim otherwise: `Drop`
    /// runs on a normal teardown and on unwinding, but not after a SIGKILL or
    /// an abort. [`sweep_orphans`] handles what that leaves behind. Blocking
    /// is acceptable — two short git invocations at teardown.
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
    }
}

/// Remove subagent worktrees left behind by a process that died abruptly.
///
/// `Drop` cannot run after a SIGKILL, so without this the branches and
/// checkouts accumulate silently — one per subagent, forever. Called at
/// startup, when nothing of ours can legitimately own one.
pub async fn sweep_orphans(repo: &std::path::Path) {
    let listed = tokio::process::Command::new("git")
        .args(["worktree", "list", "--porcelain"])
        .current_dir(repo)
        .output()
        .await;
    let Ok(output) = listed else { return };

    let mut removed = 0;
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Some(path) = line.strip_prefix("worktree ") else {
            continue;
        };
        // Ours are named distinctively and live in the temp directory, so
        // this cannot reach a worktree the user made.
        if !std::path::Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("eventage-wt-"))
        {
            continue;
        }
        let _ = tokio::process::Command::new("git")
            .args(["worktree", "remove", "--force", path])
            .current_dir(repo)
            .output()
            .await;
        removed += 1;
    }
    let _ = tokio::process::Command::new("git")
        .args(["worktree", "prune"])
        .current_dir(repo)
        .output()
        .await;
    if removed > 0 {
        info!(removed, "cleaned up subagent worktrees from a previous run");
    }
}

/// Build a snapshot worktree and hand back its path, for tests.
///
/// The worktree is leaked deliberately: the caller inspects the checkout, and
/// the temp directory it lives in goes with the test's own cleanup.
#[doc(hidden)]
pub async fn snapshot_for_test(repo: &std::path::Path) -> Result<std::path::PathBuf, AgentError> {
    let wt = Worktree::create(repo).await?;
    let path = wt.path.clone();
    std::mem::forget(wt);
    Ok(path)
}

/// The diff a subagent working in `path` would return.
#[doc(hidden)]
pub async fn diff_for_test(path: &std::path::Path) -> String {
    let wt = Worktree {
        path: path.to_path_buf(),
        repo: path.to_path_buf(),
        snapshot: String::new(),
    };
    let diff = wt.diff().await;
    std::mem::forget(wt);
    diff
}

// ── the live subagents of one session ─────────────────────────────────────────

/// How many subagents may stay resident at once.
///
/// Each holds an agent, a bus and possibly a git checkout, so they are not
/// free; the oldest idle one is retired when the cap is reached.
const MAX_LIVE: usize = 8;

/// Carry a subagent's permission requests to the user, and the answer back.
///
/// A subagent runs on its own bus so its tool results never enter the
/// parent's context. That isolation also meant its permission requests went
/// to a bus with no UI on the other end, so anything needing approval could
/// only be denied — which is why subagents had no shell, and why their own
/// instructions asked them to verify work they had no way to verify.
///
/// The isolation worth keeping is of *context*, not of the user. A request is
/// republished on the parent's bus, where the editor or Studio is already
/// listening, and the decision is relayed back. The `request_id` is carried
/// through unchanged, so the waiting hook matches its own answer; the
/// subagent's id rides along so the prompt can say who is asking.
///
/// Returns handles that stop the relay when dropped — which happens when the
/// subagent is dropped, at the end of the session.
fn bridge_permissions(
    child: &EventBus,
    parent: &EventBus,
    id: &str,
) -> Vec<tokio::task::JoinHandle<()>> {
    // Requests we have forwarded and are waiting on. Without this the relay
    // would push every decision on the parent's bus into every subagent,
    // including answers to the parent's own prompts.
    let pending: Arc<tokio::sync::Mutex<std::collections::HashSet<String>>> =
        Arc::new(tokio::sync::Mutex::new(std::collections::HashSet::new()));

    let outbound = {
        let mut rx = child.subscribe();
        let parent = parent.clone();
        let pending = Arc::clone(&pending);
        let id = id.to_string();
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                if event.kind != kinds::PERMISSION_REQUEST {
                    continue;
                }
                let Some(request_id) = event
                    .payload
                    .get("request_id")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
                else {
                    continue;
                };
                pending.lock().await.insert(request_id);

                let mut payload = event.payload.clone();
                if let Some(object) = payload.as_object_mut() {
                    object.insert("subagent_id".into(), json!(id));
                }
                let _ = parent
                    .publish(Event::new(kinds::PERMISSION_REQUEST, payload))
                    .await;
            }
        })
    };

    let inbound = {
        let mut rx = parent.subscribe();
        let child = child.clone();
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                if event.kind != kinds::PERMISSION_DECISION {
                    continue;
                }
                let Some(request_id) = event.payload.get("request_id").and_then(|v| v.as_str())
                else {
                    continue;
                };
                if !pending.lock().await.remove(request_id) {
                    continue;
                }
                let _ = child
                    .publish(Event::new(
                        kinds::PERMISSION_DECISION,
                        event.payload.clone(),
                    ))
                    .await;
            }
        })
    };

    vec![outbound, inbound]
}

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
    /// Relays permission prompts to the user and answers back. Dropped with
    /// the subagent, which is what stops it.
    _permissions: Vec<tokio::task::JoinHandle<()>>,
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
    /// The session's permission mode, read at call time — a copy taken when
    /// the tool was built went stale the moment the user switched modes.
    pub mode: Arc<crate::config::SharedMode>,
    /// The parent's bus, so what a subagent did lands in the session's trace.
    pub bus: EventBus,
    /// The subagents of this session, kept alive between calls.
    pub registry: Arc<SubagentRegistry>,
    /// Registered as `task_explore` rather than `task`: read-only kinds only,
    /// never a worktree.
    ///
    /// Two tools rather than one flag on a single tool, because the
    /// permission system gates on the tool's *name*. As one tool it had to be
    /// classified either read-only — which let an isolated run create a git
    /// branch and a worktree under a Plan-mode session — or as an edit,
    /// which made ordinary exploration need write permission and put it out
    /// of reach in Plan mode entirely. Neither is the truth; there are two
    /// different capabilities here and they are named separately.
    pub read_only: bool,
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
            // Nothing else records a refusal. A policy `Deny` never reaches
            // `permission.decision` at all — it is answered by the hook, and
            // for a subagent that is *every* refusal, since a subagent has
            // nobody to ask. The evidence is in the result.
            kinds::TOOL_RESULT
                if event
                    .payload
                    .get("result")
                    .and_then(|r| r.get("denied"))
                    .and_then(|d| d.as_bool())
                    == Some(true) =>
            {
                let reason = event
                    .payload
                    .get("result")
                    .and_then(|r| r.get("reason"))
                    .and_then(|r| r.as_str())
                    .unwrap_or("no reason given");
                denials.push(format!("{}: {reason}", name("name")));
            }
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
            if self.read_only {
                "task_explore"
            } else {
                "task"
            },
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
        if self.read_only && kind.can_edit() {
            return Err(AgentError::Tool(
                "task_explore only runs read-only subagents (explore, plan). Use `task` \
                 for one that may edit."
                    .into(),
            ));
        }
        let isolated = args
            .get("isolated")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
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

        let ws = Arc::new(Workspace::open(&root).map_err(|e| AgentError::Tool(e.to_string()))?);
        let lsp = Arc::new(LspPool::new(&root));
        let bus = EventBus::new();
        // Minted before the agent is built so the relay can name it.
        let id = self.registry.mint(kind);
        let permissions = bridge_permissions(&bus, &self.bus, &id);

        // A subagent used to run with no policy at all. It inherits the
        // session's now, minus anything that would need a human to answer —
        // there is no UI on the other end of this bus.
        let mut builder = AgentBuilder::new()
            .agent_id(format!("subagent-{:?}", kind).to_lowercase())
            .bus(bus.clone())
            .llm_arc(Arc::clone(&self.llm))
            .hook(self.mode.load().subagent_policy(isolated))
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
                // `verify` first, because it is what a subagent should reach
                // for and needs no approval at all. `bash` is here too now
                // that a prompt reaches the user rather than a bus nobody is
                // listening to.
                .tool(tools::Verify {
                    ws: ws.clone(),
                    containment: tools::ShellContainment::Confined,
                })
                .tool(tools::Bash {
                    ws: ws.clone(),
                    jobs: Arc::new(tools::BackgroundJobs::default()),
                    containment: tools::ShellContainment::Confined,
                    container_image: tools::DEFAULT_CONTAINER_IMAGE.into(),
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
            _permissions: permissions,
        };

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
            result["snapshot_commit"] = json!(wt.snapshot);
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
    use crate::config::PermissionMode;
    use eventage::llm::MockLlmProvider;

    /// A tool whose subagents answer with `replies`, one per turn.
    fn task_tool_with(depth: usize, replies: Vec<&str>) -> (tempfile::TempDir, Task) {
        let dir = tempfile::tempdir().unwrap();
        let tool = Task {
            ws: Arc::new(Workspace::open(dir.path()).unwrap()),
            llm: Arc::new(MockLlmProvider::with_texts(replies)),
            depth,
            active: Arc::new(AtomicUsize::new(0)),
            mode: Arc::new(crate::config::SharedMode::new(PermissionMode::Auto)),
            bus: EventBus::new(),
            registry: Arc::new(SubagentRegistry::new()),
            read_only: false,
        };
        (dir, tool)
    }

    fn task_tool(depth: usize) -> Task {
        task_tool_with(depth, vec!["done"]).1
    }

    #[tokio::test]
    async fn exploring_and_delegating_work_are_separate_tools() {
        // The permission system gates on a tool's name, so one tool could
        // only be classified as read-only — which let an isolated run create
        // a worktree under a Plan-mode session — or as an edit, which put
        // ordinary exploration out of reach in Plan mode.
        let (_dir, mut tool) = task_tool_with(0, vec!["ok"]);
        tool.read_only = true;
        assert_eq!(tool.definition().function.name, "task_explore");

        // A read-only delegate cannot ask for an editing subagent.
        let err = tool
            .execute(json!({ "prompt": "go", "subagent_type": "general" }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("read-only"), "{err}");
        assert!(
            err.contains("`task`"),
            "it should say which tool to use: {err}"
        );

        // And the read-only kinds still work.
        assert!(tool
            .execute(json!({ "prompt": "go", "subagent_type": "explore" }))
            .await
            .is_ok());
    }

    #[test]
    fn every_gated_tool_name_actually_exists() {
        // `task_explore` was named in the permission lists before any tool
        // answered to it, so the list gated a tool nobody could call while
        // the real one fell through to the wrong classification.
        let names = [
            PermissionMode::READ_ONLY_TOOLS,
            PermissionMode::EDIT_TOOLS,
            PermissionMode::RISKY_TOOLS,
        ];
        for name in names.iter().flat_map(|list| list.iter()) {
            assert!(
                !name.contains('*'),
                "wildcards hide exactly this mistake: '{name}'"
            );
        }
        assert!(PermissionMode::READ_ONLY_TOOLS.contains(&"task_explore"));
        assert!(PermissionMode::EDIT_TOOLS.contains(&"task"));
    }

    #[tokio::test]
    async fn a_subagents_permission_request_reaches_the_user_and_the_answer_comes_back() {
        // The whole reason a subagent could not have a shell: its bus had no
        // UI on the other end, so anything needing approval could only be
        // denied. The request goes to the parent's bus now, where the editor
        // is already listening.
        let (_dir, tool) = task_tool_with(0, vec!["done"]);
        let child = EventBus::new();
        let handles = bridge_permissions(&child, &tool.bus, "general-7");

        let mut on_parent = tool.bus.subscribe();

        // A subagent's hook asks for something.
        child
            .publish(Event::new(
                kinds::PERMISSION_REQUEST,
                json!({ "request_id": "r1", "tool": "bash", "arguments": {} }),
            ))
            .await
            .unwrap();

        // It arrives where a person can see it, named.
        let asked = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while let Some(event) = on_parent.recv().await {
                if event.kind == kinds::PERMISSION_REQUEST {
                    return event;
                }
            }
            panic!("the request never reached the parent");
        })
        .await
        .expect("the request should be forwarded promptly");
        assert_eq!(asked.payload["request_id"], "r1");
        assert_eq!(asked.payload["tool"], "bash");
        assert_eq!(
            asked.payload["subagent_id"], "general-7",
            "the prompt has to say who is asking"
        );

        // The user answers on the parent's bus…
        let mut on_child = child.subscribe();
        tool.bus
            .publish(Event::new(
                kinds::PERMISSION_DECISION,
                json!({ "request_id": "r1", "approve": true }),
            ))
            .await
            .unwrap();

        // …and the waiting subagent hears it, with its own request id intact.
        let answered = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while let Some(event) = on_child.recv().await {
                if event.kind == kinds::PERMISSION_DECISION {
                    return event;
                }
            }
            panic!("the decision never came back");
        })
        .await
        .expect("the decision should be relayed promptly");
        assert_eq!(answered.payload["request_id"], "r1");
        assert_eq!(answered.payload["approve"], true);

        drop(handles);
    }

    #[tokio::test]
    async fn a_decision_meant_for_the_parent_is_not_pushed_into_a_subagent() {
        // Both buses carry decisions. Relaying every one of them would answer
        // a subagent's prompt with the reply to somebody else's question.
        let (_dir, tool) = task_tool_with(0, vec!["done"]);
        let child = EventBus::new();
        let _handles = bridge_permissions(&child, &tool.bus, "general-1");
        let mut on_child = child.subscribe();

        tool.bus
            .publish(Event::new(
                kinds::PERMISSION_DECISION,
                json!({ "request_id": "the-parent's-own", "approve": true }),
            ))
            .await
            .unwrap();

        let leaked = tokio::time::timeout(std::time::Duration::from_millis(300), async {
            while let Some(event) = on_child.recv().await {
                if event.kind == kinds::PERMISSION_DECISION {
                    return true;
                }
            }
            false
        })
        .await;
        assert!(
            leaked.is_err() || !leaked.unwrap(),
            "an unrelated decision was pushed into the subagent"
        );
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
        assert!(
            err.contains("explore-1"),
            "should name what is available: {err}"
        );
    }

    #[test]
    fn a_refusal_is_reported_as_refused_not_as_a_failure() {
        // A policy `Deny` never produces a `permission.decision` — the hook
        // answers it — and for a subagent that is every refusal, since there
        // is nobody to ask. Reading the decision was reading nothing.
        let log = vec![
            Event::new(
                kinds::TOOL_RESULT,
                json!({
                    "tool_call_id": "c1",
                    "name": "bash",
                    "result": { "denied": true, "reason": "nobody to ask" },
                }),
            ),
            Event::new(
                kinds::TOOL_RESULT,
                json!({ "tool_call_id": "c2", "name": "read_file", "error": "no such file" }),
            ),
        ];

        let state = digest(&log, 0);
        let refused = state["refused"].as_array().unwrap();
        assert_eq!(refused.len(), 1, "{state}");
        assert!(refused[0].as_str().unwrap().contains("bash"), "{state}");
        assert!(
            refused[0].as_str().unwrap().contains("nobody to ask"),
            "the caller needs to know why: {state}"
        );
        // A refusal is not a failure; they mean different things to a caller.
        assert_eq!(state["failures"].as_array().unwrap().len(), 1, "{state}");
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
        assert_eq!(
            SubagentKind::from_str("explore"),
            Some(SubagentKind::Explore)
        );
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
        assert!(
            err.to_string().contains("explore, plan, or general"),
            "{err}"
        );
        assert!(
            err.to_string().contains("subagent_id"),
            "and how to continue: {err}"
        );
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
