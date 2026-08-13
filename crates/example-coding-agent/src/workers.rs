//! Background workers for the coding agent.
//!
//! - [`SubAgentWorker`] — listens for `subagent.task.launch` events and runs
//!   isolated sub-agents in background tokio tasks.
//! - [`TurnDiffWorker`] — snapshots the workspace at cycle start and publishes
//!   unified diffs at cycle end.

use crate::kinds::{CODING_TURN_DIFF, SUBAGENT_TASK_COMPLETE, SUBAGENT_TASK_ERROR, SUBAGENT_TASK_LAUNCH};
use crate::tools::{build_sub_agent_tools, SubAgentSpec};
use crate::workspace::Workspace;
use async_trait::async_trait;
use eventage::{
    agent::worker::{EventWorker, WorkerError},
    event::{kinds, Event},
    llm::LlmProvider,
    AgentBuilder, EventBus, ReactStrategy,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use similar::{ChangeTag, TextDiff};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, error};

// ── SubAgentWorker ────────────────────────────────────────────────────────────

/// Listens for `subagent.task.launch` events and runs the requested sub-agent
/// in a background tokio task. Results are published back to the shared bus.
pub struct SubAgentWorker {
    pub llm: Arc<dyn LlmProvider>,
    pub specs: Vec<SubAgentSpec>,
    pub base_system_prompt: String,
    pub max_steps: usize,
    pub work_dir: PathBuf,
}

#[async_trait]
impl EventWorker for SubAgentWorker {
    fn subscribed_kinds(&self) -> Vec<String> {
        vec![SUBAGENT_TASK_LAUNCH.to_string()]
    }

    async fn handle(&self, event: &Event, bus: &EventBus) -> Result<(), WorkerError> {
        let job_id = event
            .payload
            .get("job_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let subagent_type = event
            .payload
            .get("subagent_type")
            .and_then(|v| v.as_str())
            .unwrap_or("general-purpose")
            .to_string();

        let description = event
            .payload
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if job_id.is_empty() || description.is_empty() {
            return Ok(()); // malformed event, skip
        }

        let spec = self
            .specs
            .iter()
            .find(|s| s.name == subagent_type)
            .cloned()
            .or_else(|| self.specs.first().cloned());

        let spec = match spec {
            Some(s) => s,
            None => return Ok(()),
        };

        let llm = self.llm.clone();
        let work_dir = self.work_dir.clone();
        let max_steps = self.max_steps;
        let base_prompt = self.base_system_prompt.clone();
        let bus_clone = bus.clone();

        debug!("launching async sub-agent job={job_id} type={subagent_type}");

        tokio::spawn(async move {
            let result =
                run_sub_agent(llm, &spec, &base_prompt, &description, &work_dir, max_steps).await;

            let publish_result = match result {
                Ok(response) => {
                    bus_clone
                        .publish(Event::new(
                            SUBAGENT_TASK_COMPLETE,
                            json!({ "job_id": job_id, "result": response }),
                        ))
                        .await
                }
                Err(e) => {
                    error!("async sub-agent job={job_id} failed: {e}");
                    bus_clone
                        .publish(Event::new(
                            SUBAGENT_TASK_ERROR,
                            json!({ "job_id": job_id, "error": e.to_string() }),
                        ))
                        .await
                }
            };

            if let Err(e) = publish_result {
                error!("failed to publish sub-agent result for job={job_id}: {e}");
            }
        });

        Ok(())
    }
}

async fn run_sub_agent(
    llm: Arc<dyn LlmProvider>,
    spec: &SubAgentSpec,
    base_prompt: &str,
    description: &str,
    work_dir: &Path,
    max_steps: usize,
) -> Result<String, eventage::AgentError> {
    let sub_bus = EventBus::new();

    let system_prompt = if spec.system_prompt.trim().is_empty() {
        base_prompt.to_string()
    } else {
        format!("{}\n\n{}", spec.system_prompt, base_prompt)
    };

    let tools = build_sub_agent_tools(work_dir);
    let mut builder = AgentBuilder::new()
        .bus(sub_bus.clone())
        .llm_arc(llm)
        .system_prompt(system_prompt)
        .strategy(ReactStrategy {
            max_steps,
            max_concurrent_tools: 4,
            ..Default::default()
        });

    for tool in tools {
        builder = builder.tool_arc(tool);
    }

    let agent = builder.build();

    sub_bus
        .publish(Event::new(kinds::USER_MESSAGE, json!({ "text": description })))
        .await
        .map_err(|e| eventage::AgentError::Tool(format!("bus: {e}")))?;

    agent.cycle().await?;

    let log = sub_bus.log().await;
    let result = log
        .iter()
        .rev()
        .find(|e| e.kind == kinds::ASSISTANT_MESSAGE)
        .and_then(|e| e.payload["content"].as_str().map(|s| s.to_string()))
        .unwrap_or_else(|| "(no response)".to_string());

    Ok(result)
}

// ── TurnDiffWorker ────────────────────────────────────────────────────────────

#[derive(Clone)]
struct FileSnap {
    sha256: String,
    content: String,
}

/// An [`EventWorker`] that snapshots workspace files at the start of each
/// agent cycle and publishes unified diffs at the end.
pub struct TurnDiffWorker {
    workspace: Arc<Workspace>,
    baseline: Mutex<HashMap<String, FileSnap>>,
}

impl TurnDiffWorker {
    pub fn new(workspace: Arc<Workspace>) -> Self {
        Self {
            workspace,
            baseline: Mutex::new(HashMap::new()),
        }
    }

    async fn snapshot(&self) -> HashMap<String, FileSnap> {
        let mut snaps = HashMap::new();
        let Ok(files) = self.workspace.list_files() else {
            return snaps;
        };
        for entry in &files {
            let abs = match self.workspace.resolve(&entry.path) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let Ok(content) = std::fs::read_to_string(&abs) else {
                continue; // skip binary files
            };
            let sha256 = hex::encode(Sha256::digest(content.as_bytes()));
            snaps.insert(entry.path.clone(), FileSnap { sha256, content });
        }
        snaps
    }
}

#[async_trait]
impl EventWorker for TurnDiffWorker {
    fn subscribed_kinds(&self) -> Vec<String> {
        vec![
            kinds::AGENT_CYCLE_START.to_string(),
            kinds::AGENT_CYCLE_END.to_string(),
        ]
    }

    async fn handle(&self, event: &Event, bus: &EventBus) -> Result<(), WorkerError> {
        match event.kind.as_str() {
            k if k == kinds::AGENT_CYCLE_START => {
                let snap = self.snapshot().await;
                debug!(files = snap.len(), "TurnDiffWorker: baseline snapshot taken");
                *self.baseline.lock().await = snap;
            }

            k if k == kinds::AGENT_CYCLE_END => {
                let baseline = self.baseline.lock().await.clone();
                let current = self.snapshot().await;

                let mut diffs: HashMap<String, String> = HashMap::new();
                let mut changed = 0usize;
                let mut new_files = 0usize;
                let mut deleted = 0usize;

                for (path, cur_snap) in &current {
                    match baseline.get(path) {
                        Some(old) if old.sha256 == cur_snap.sha256 => {}
                        Some(old) => {
                            changed += 1;
                            let diff = compute_unified_diff(
                                &format!("a/{path}"),
                                &format!("b/{path}"),
                                &old.content,
                                &cur_snap.content,
                            );
                            diffs.insert(path.clone(), diff);
                        }
                        None => {
                            new_files += 1;
                            let diff = compute_unified_diff(
                                "/dev/null",
                                &format!("b/{path}"),
                                "",
                                &cur_snap.content,
                            );
                            diffs.insert(path.clone(), diff);
                        }
                    }
                }

                for path in baseline.keys() {
                    if !current.contains_key(path) {
                        deleted += 1;
                        let old = &baseline[path];
                        let diff = compute_unified_diff(
                            &format!("a/{path}"),
                            "/dev/null",
                            &old.content,
                            "",
                        );
                        diffs.insert(path.clone(), diff);
                    }
                }

                if !diffs.is_empty() {
                    debug!(changed, new_files, deleted, "TurnDiffWorker: publishing diff");
                    bus.publish(Event::new(
                        CODING_TURN_DIFF,
                        json!({
                            "changed_files": changed,
                            "new_files": new_files,
                            "deleted_files": deleted,
                            "diffs": diffs,
                        }),
                    ))
                    .await
                    .map_err(WorkerError::Bus)?;
                }
            }

            _ => {}
        }
        Ok(())
    }
}

fn compute_unified_diff(old_label: &str, new_label: &str, old: &str, new: &str) -> String {
    let diff = TextDiff::from_lines(old, new);
    let mut out = format!("--- {old_label}\n+++ {new_label}\n");

    for group in diff.grouped_ops(3) {
        let first = group.first().unwrap();
        let old_start = first.old_range().start + 1;
        let old_len: usize = group.iter().map(|op| op.old_range().len()).sum();
        let new_start = first.new_range().start + 1;
        let new_len: usize = group.iter().map(|op| op.new_range().len()).sum();

        out.push_str(&format!(
            "@@ -{},{} +{},{} @@\n",
            old_start, old_len, new_start, new_len
        ));

        for op in &group {
            for change in diff.iter_changes(op) {
                let tag = match change.tag() {
                    ChangeTag::Delete => '-',
                    ChangeTag::Insert => '+',
                    ChangeTag::Equal => ' ',
                };
                out.push(tag);
                out.push_str(change.value());
                if change.missing_newline() {
                    out.push('\n');
                }
            }
        }
    }

    out
}
