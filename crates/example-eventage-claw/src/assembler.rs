//! Context assemblers for eventage-claw.
//!
//! Assembler chain per group:
//!   DefaultContextAssembler(system_prompt)
//!     → SummarizingAssembler (token budget management)
//!     → GroupMemoryAssembler (global + per-group AGENT.md)
//!     → SkillsAssembler (SKILL.md files from skills dir)
//!
//! Adapted from example-coding-agent/src/assembler.rs.

use async_trait::async_trait;
use eventage::{
    agent::{AssemblyContext, ContextAssembler},
    llm::{ChatMessage, LlmProvider, Role},
};
use serde::Deserialize;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::Mutex;
use tracing::{debug, warn};

// ── Token estimation ──────────────────────────────────────────────────────────

fn estimate_tokens(s: &str) -> usize {
    let words = s.split_whitespace().count();
    ((words as f64) * 1.3).ceil() as usize + 4
}

fn messages_token_count(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .map(|m| {
            let content_tokens = m.content.as_deref().map(estimate_tokens).unwrap_or(0);
            let tool_call_tokens: usize = m
                .tool_calls
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .map(|tc| {
                    estimate_tokens(&tc.function.arguments) + estimate_tokens(&tc.function.name) + 4
                })
                .sum();
            content_tokens + tool_call_tokens + 4
        })
        .sum()
}

fn partition_messages(messages: &[ChatMessage]) -> (Vec<ChatMessage>, Vec<ChatMessage>) {
    let (sys, conv): (Vec<_>, Vec<_>) = messages
        .iter()
        .cloned()
        .partition(|m| m.role == Role::System);
    (sys, conv)
}

// ── SummarizingAssembler ──────────────────────────────────────────────────────

struct SummaryState {
    summary: String,
    summarized_conv_count: usize,
}

/// Wraps any `ContextAssembler` and automatically summarizes old conversation
/// when the assembled context exceeds a token budget.
pub struct SummarizingAssembler {
    inner: Arc<dyn ContextAssembler>,
    llm: Arc<dyn LlmProvider>,
    pub max_tokens: usize,
    pub threshold: f64,
    pub keep_fraction: f64,
    pub session_id: String,
    state: Mutex<Option<SummaryState>>,
}

impl SummarizingAssembler {
    pub fn new(
        inner: Arc<dyn ContextAssembler>,
        llm: Arc<dyn LlmProvider>,
        max_tokens: usize,
        session_id: impl Into<String>,
    ) -> Self {
        Self {
            inner,
            llm,
            max_tokens,
            threshold: 0.85,
            keep_fraction: 0.10,
            session_id: session_id.into(),
            state: Mutex::new(None),
        }
    }

    async fn do_summarize(&self, to_summarize: &[ChatMessage]) -> String {
        let text: String = to_summarize
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::User => "User",
                    Role::Assistant => "Assistant",
                    Role::Tool => "Tool",
                    Role::System => "System",
                };
                format!(
                    "{role}: {}\n",
                    m.content.as_deref().unwrap_or("(tool calls)")
                )
            })
            .collect();

        let prompt = format!(
            "Summarize the following conversation history. \
             Your summary MUST start with a \"User instructions\" bullet list that quotes, \
             verbatim, every short instruction or correction the user gave. \
             After the bullet list, write a concise narrative preserving all important \
             context, decisions, and results.\n\n{text}"
        );

        match self
            .llm
            .complete(vec![ChatMessage::user(prompt)], vec![])
            .await
        {
            Ok(resp) => resp.content.unwrap_or_else(|| "(empty summary)".into()),
            Err(e) => {
                warn!("summarization failed: {e}");
                format!("[Summary unavailable: {e}]")
            }
        }
    }

    async fn offload_to_file(&self, messages: &[ChatMessage]) {
        let dir = PathBuf::from("/tmp/claw-history");
        if tokio::fs::create_dir_all(&dir).await.is_err() {
            return;
        }
        let path = dir.join(format!("{}.md", self.session_id));
        let text: String = messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::User => "User",
                    Role::Assistant => "Assistant",
                    Role::Tool => "Tool",
                    Role::System => "System",
                };
                format!(
                    "## {role}\n{}\n\n",
                    m.content.as_deref().unwrap_or("(tool calls)")
                )
            })
            .collect();
        let _ = tokio::fs::write(&path, text).await;
        debug!("offloaded history to {}", path.display());
    }
}

#[async_trait]
impl ContextAssembler for SummarizingAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let mut messages = self.inner.assemble(context).await;
        let (sys_msgs, conv_msgs) = partition_messages(&messages);
        let budget = (self.max_tokens as f64 * self.threshold) as usize;

        {
            let state = self.state.lock().await;
            if let Some(ref s) = *state {
                if conv_msgs.len() > s.summarized_conv_count {
                    let summary_msg = ChatMessage::system(format!(
                        "<conversation_summary>\n\
                         The following is a summary of the earlier conversation:\n\n\
                         {}\n\
                         </conversation_summary>",
                        s.summary
                    ));
                    messages.clear();
                    messages.extend(sys_msgs.clone());
                    messages.push(summary_msg);
                    messages.extend_from_slice(&conv_msgs[s.summarized_conv_count..]);
                }
                return messages;
            }
        }

        let total_tokens = messages_token_count(&messages);
        if total_tokens < budget {
            return messages;
        }

        let keep = ((conv_msgs.len() as f64 * self.keep_fraction).ceil() as usize).max(4);
        let cutoff = conv_msgs.len().saturating_sub(keep);

        if cutoff == 0 {
            return messages;
        }

        let to_summarize = &conv_msgs[..cutoff];
        let to_keep = &conv_msgs[cutoff..];

        self.offload_to_file(to_summarize).await;
        let summary = self.do_summarize(to_summarize).await;

        {
            let mut state = self.state.lock().await;
            *state = Some(SummaryState {
                summary: summary.clone(),
                summarized_conv_count: cutoff,
            });
        }

        let summary_msg = ChatMessage::system(format!(
            "<conversation_summary>\n\
             The following is a summary of the earlier conversation:\n\n\
             {summary}\n\
             </conversation_summary>"
        ));

        messages.clear();
        messages.extend(sys_msgs);
        messages.push(summary_msg);
        messages.extend_from_slice(to_keep);
        messages
    }
}

// ── GroupMemoryAssembler ──────────────────────────────────────────────────────

// ── Mtime helpers ─────────────────────────────────────────────────────────────

fn file_mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

// ── GroupMemoryAssembler ──────────────────────────────────────────────────────

struct MemoryCache {
    content: String,
    global_mtime: Option<SystemTime>,
    group_mtime: Option<SystemTime>,
}

/// Loads global AGENT.md (shared across all groups) and per-group AGENT.md,
/// injecting both as system context.
///
/// Uses mtime-based caching: a `stat()` is issued each cycle; the files are
/// re-read only when a modification timestamp has changed.  This means a skill
/// or memory update written by the agent via `WriteFileTool` takes effect in
/// the very next cycle without any restart, while idle cycles cost only two
/// cheap syscalls.
pub struct GroupMemoryAssembler {
    inner: Arc<dyn ContextAssembler>,
    global_memory_path: PathBuf,
    group_memory_path: PathBuf,
    cache: Mutex<Option<MemoryCache>>,
}

impl GroupMemoryAssembler {
    pub fn new(
        inner: Arc<dyn ContextAssembler>,
        global_memory_path: &Path,
        group_memory_path: &Path,
    ) -> Self {
        Self {
            inner,
            global_memory_path: global_memory_path.to_owned(),
            group_memory_path: group_memory_path.to_owned(),
            cache: Mutex::new(None),
        }
    }
}

#[async_trait]
impl ContextAssembler for GroupMemoryAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let mut messages = self.inner.assemble(context).await;

        let current_global_mtime = file_mtime(&self.global_memory_path);
        let current_group_mtime = file_mtime(&self.group_memory_path);

        // Check the cache without holding the lock during I/O.
        let cache_hit = self.cache.lock().await.as_ref().is_some_and(|c| {
            c.global_mtime == current_global_mtime && c.group_mtime == current_group_mtime
        });

        let content = if cache_hit {
            self.cache
                .lock()
                .await
                .as_ref()
                .map(|c| c.content.clone())
                .unwrap_or_default()
        } else {
            // Mtime changed or first call — re-read files with no lock held.
            let mut parts = vec![];
            if let Ok(text) = std::fs::read_to_string(&self.global_memory_path) {
                if !text.trim().is_empty() {
                    parts.push(format!("# Global Memory\n{text}"));
                }
            }
            if let Ok(text) = std::fs::read_to_string(&self.group_memory_path) {
                if !text.trim().is_empty() {
                    parts.push(format!("# Group Memory\n{text}"));
                }
            }
            let new_content = parts.join("\n\n");
            debug!(
                global = %self.global_memory_path.display(),
                group  = %self.group_memory_path.display(),
                "GroupMemoryAssembler: reloaded memory files"
            );
            *self.cache.lock().await = Some(MemoryCache {
                content: new_content.clone(),
                global_mtime: current_global_mtime,
                group_mtime: current_group_mtime,
            });
            new_content
        };

        if content.is_empty() {
            return messages;
        }

        let memory_msg = ChatMessage::system(format!("<agent_memory>\n{content}\n</agent_memory>"));

        let insert_pos = messages
            .iter()
            .rposition(|m| m.role == Role::System)
            .map(|i| i + 1)
            .unwrap_or(0);

        messages.insert(insert_pos, memory_msg);
        messages
    }
}

// ── SkillsAssembler ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SkillMetadata {
    pub name: String,
    pub description: String,
    pub instructions: String,
}

#[derive(Debug, Deserialize)]
struct SkillFrontmatter {
    name: Option<String>,
    description: Option<String>,
}

fn load_skill(path: &Path) -> Option<SkillMetadata> {
    let content = std::fs::read_to_string(path).ok()?;

    if !content.starts_with("---") {
        return None;
    }

    let rest = &content[3..];
    let end = rest.find("---")?;
    let yaml_str = &rest[..end];
    let instructions = rest.get(end + 3..).unwrap_or("").trim().to_string();

    let front: SkillFrontmatter = serde_yaml::from_str(yaml_str).ok()?;

    Some(SkillMetadata {
        name: front.name.unwrap_or_else(|| {
            path.parent()
                .and_then(|p| p.file_name())
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "unnamed".into())
        }),
        description: front
            .description
            .unwrap_or_else(|| "No description.".into()),
        instructions,
    })
}

type SkillsCache = Option<(HashMap<PathBuf, Option<SystemTime>>, Vec<SkillMetadata>)>;

/// Loads SKILL.md files from a skills directory and injects them into context.
///
/// Uses mtime-based caching: each cycle a `stat()` is issued per skill file;
/// skills are re-parsed only when any file's modification timestamp changes or
/// new files appear.  New skills written by the agent via `WriteFileTool` are
/// therefore available in the very next cycle (self-evolution without restart)
/// while unchanged sessions pay only cheap `stat` syscalls — not file reads.
pub struct SkillsAssembler {
    inner: Arc<dyn ContextAssembler>,
    skills_dir: PathBuf,
    cache: Mutex<SkillsCache>,
}

impl SkillsAssembler {
    pub fn new(inner: Arc<dyn ContextAssembler>, skills_dir: &Path) -> Self {
        Self {
            inner,
            skills_dir: skills_dir.to_owned(),
            cache: Mutex::new(None),
        }
    }
}

/// Walk the skills dir and build a `path → mtime` signature map.
/// Returns an empty map if the directory doesn't exist yet.
fn skills_signature(skills_dir: &Path) -> HashMap<PathBuf, Option<SystemTime>> {
    walkdir::WalkDir::new(skills_dir)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name() == "SKILL.md")
        .map(|e| {
            let mtime = e.metadata().ok().and_then(|m| m.modified().ok());
            (e.path().to_owned(), mtime)
        })
        .collect()
}

#[async_trait]
impl ContextAssembler for SkillsAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let mut messages = self.inner.assemble(context).await;

        // Build the current signature (stat only — no file reads yet).
        let sig = skills_signature(&self.skills_dir);

        // Check cache without holding the lock during file I/O.
        let cache_hit = self
            .cache
            .lock()
            .await
            .as_ref()
            .is_some_and(|(cached_sig, _)| *cached_sig == sig);

        let skills = if cache_hit {
            self.cache
                .lock()
                .await
                .as_ref()
                .map(|(_, s)| s.clone())
                .unwrap_or_default()
        } else {
            // Signature changed — re-parse all skill files with no lock held.
            let mut loaded = vec![];
            for path in sig.keys() {
                if let Some(skill) = load_skill(path) {
                    debug!("loaded skill '{}' from {}", skill.name, path.display());
                    loaded.push(skill);
                }
            }
            debug!(count = loaded.len(), "SkillsAssembler: reloaded skills");
            *self.cache.lock().await = Some((sig, loaded.clone()));
            loaded
        };

        if skills.is_empty() {
            return messages;
        }

        let skills_content: String = skills
            .iter()
            .map(|s| {
                format!(
                    "## Skill: {} — {}\n\n{}",
                    s.name, s.description, s.instructions
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");

        let skills_msg = ChatMessage::system(format!(
            "<available_skills>\n{skills_content}\n</available_skills>"
        ));

        let insert_pos = messages
            .iter()
            .rposition(|m| m.role == Role::System)
            .map(|i| i + 1)
            .unwrap_or(0);

        messages.insert(insert_pos, skills_msg);
        messages
    }
}

// ── UserCorrectionsAssembler ──────────────────────────────────────────────────

/// Classifies user follow-up messages as behavioral instructions and injects
/// them as a sticky system message that survives summarization.
pub struct UserCorrectionsAssembler {
    inner: Arc<dyn ContextAssembler>,
    llm: Arc<dyn LlmProvider>,
    cache: Mutex<Option<(u64, Vec<String>)>>,
}

impl UserCorrectionsAssembler {
    pub fn new(inner: Arc<dyn ContextAssembler>, llm: Arc<dyn LlmProvider>) -> Self {
        Self {
            inner,
            llm,
            cache: Mutex::new(None),
        }
    }

    async fn extract_instructions(&self, user_msgs: &[&str]) -> Vec<String> {
        let numbered: String = user_msgs
            .iter()
            .enumerate()
            .map(|(i, msg)| format!("{}. {:?}", i + 1, msg))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Below are follow-up messages a user sent to an AI personal assistant.\n\
             Classify each as INSTRUCTION (tells the agent HOW to behave) or NOT.\n\
             Return a JSON array with verbatim INSTRUCTION messages only, or [].\n\
             Messages:\n{numbered}\n\
             Return ONLY the JSON array."
        );

        match self
            .llm
            .complete(vec![ChatMessage::user(prompt)], vec![])
            .await
        {
            Ok(resp) => {
                let content = resp.content.unwrap_or_default();
                let s = content.trim();
                let s = s
                    .find('[')
                    .and_then(|start| s.rfind(']').map(|end| &s[start..=end]))
                    .unwrap_or("[]");
                serde_json::from_str::<Vec<String>>(s).unwrap_or_default()
            }
            Err(e) => {
                warn!("UserCorrectionsAssembler: {e}");
                vec![]
            }
        }
    }
}

#[async_trait]
impl ContextAssembler for UserCorrectionsAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        use eventage::event::kinds as core_kinds;

        let mut messages = self.inner.assemble(context).await;

        let mut user_msgs: Vec<&str> = vec![];
        let mut first_seen = false;
        for event in context.events {
            if event.kind != core_kinds::USER_MESSAGE {
                continue;
            }
            let text = event
                .payload
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            if !first_seen {
                first_seen = true;
                continue;
            }
            if !text.is_empty() {
                user_msgs.push(text);
            }
        }

        if user_msgs.is_empty() {
            return messages;
        }

        let mut hasher = DefaultHasher::new();
        user_msgs.hash(&mut hasher);
        let current_hash = hasher.finish();

        let cached = {
            let guard = self.cache.lock().await;
            guard
                .as_ref()
                .filter(|(h, _)| *h == current_hash)
                .map(|(_, v)| v.clone())
        };

        let instructions = if let Some(v) = cached {
            v
        } else {
            let extracted = self.extract_instructions(&user_msgs).await;
            *self.cache.lock().await = Some((current_hash, extracted.clone()));
            extracted
        };

        if instructions.is_empty() {
            return messages;
        }

        let bullet_list: String = instructions
            .iter()
            .map(|s| format!("- {s}"))
            .collect::<Vec<_>>()
            .join("\n");
        let instruction_msg = ChatMessage::system(format!(
            "<user_instructions>\n\
             The user gave the following behavioral instructions. Follow ALL of them:\n\
             {bullet_list}\n\
             </user_instructions>"
        ));

        let insert_pos = messages
            .iter()
            .rposition(|m| m.role == Role::System)
            .map(|i| i + 1)
            .unwrap_or(0);

        messages.insert(insert_pos, instruction_msg);
        messages
    }
}
