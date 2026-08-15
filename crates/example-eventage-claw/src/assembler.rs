//! Context assemblers for eventage-claw.
//!
//! Assembler chain per group:
//!   DefaultContextAssembler(system_prompt)
//!     → SummarizingContextAssembler (token budget management)
//!     → GroupMemoryAssembler (global + per-group AGENT.md)
//!     → SkillsAssembler (SKILL.md files from skills dir)

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

// Re-export for use in agent.rs and other claw modules.
pub use eventage::SummarizingContextAssembler as SummarizingAssembler;

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
    /// Keywords that gate injection. Empty means always inject.
    pub triggers: Vec<String>,
}

/// Accepts either a YAML sequence (`[pdf, document]`) or a comma-separated
/// string (`"pdf, document"`) for the `triggers` frontmatter field.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StringOrVec {
    Str(String),
    Vec(Vec<String>),
}

impl StringOrVec {
    fn into_trigger_vec(self) -> Vec<String> {
        match self {
            StringOrVec::Str(s) => s
                .split(',')
                .map(|t| t.trim().to_lowercase())
                .filter(|t| !t.is_empty())
                .collect(),
            StringOrVec::Vec(v) => v.into_iter().map(|t| t.trim().to_lowercase()).collect(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct SkillFrontmatter {
    name: Option<String>,
    description: Option<String>,
    triggers: Option<StringOrVec>,
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
        triggers: front
            .triggers
            .map(StringOrVec::into_trigger_vec)
            .unwrap_or_default(),
    })
}

/// Maximum number of trigger-gated skills to inject when many match.
const MAX_TRIGGERED_SKILLS: usize = 5;

type SkillsCache = Option<(HashMap<PathBuf, Option<SystemTime>>, Vec<SkillMetadata>)>;

/// Loads SKILL.md files from a skills directory and injects them into context.
///
/// Uses mtime-based caching: each cycle a `stat()` is issued per skill file;
/// skills are re-parsed only when any file's modification timestamp changes or
/// new files appear.  New skills written by the agent via `WriteFileTool` are
/// therefore available in the very next cycle (self-evolution without restart)
/// while unchanged sessions pay only cheap `stat` syscalls — not file reads.
///
/// When an LLM is provided via [`with_llm`](SkillsAssembler::with_llm), triggered skills
/// go through a two-phase gradual disclosure: the LLM first receives a compact manifest
/// (names and descriptions only) and selects which skills are relevant, then only the
/// selected skills are expanded to full instructions. This call is completely isolated —
/// it uses `LlmProvider::complete` directly with no EventBus involvement.
pub struct SkillsAssembler {
    inner: Arc<dyn ContextAssembler>,
    skills_dir: PathBuf,
    cache: Mutex<SkillsCache>,
    llm: Option<Arc<dyn LlmProvider>>,
    /// Cache for LLM skill selection: (hash of recent user text, selected skill names).
    selection_cache: Mutex<Option<(u64, Vec<String>)>>,
}

impl SkillsAssembler {
    pub fn new(inner: Arc<dyn ContextAssembler>, skills_dir: &Path) -> Self {
        Self {
            inner,
            skills_dir: skills_dir.to_owned(),
            cache: Mutex::new(None),
            llm: None,
            selection_cache: Mutex::new(None),
        }
    }

    /// Enable LLM-based gradual skill disclosure.
    ///
    /// When set, triggered skills are resolved by showing the LLM a compact manifest
    /// (name + description only) and asking it to select relevant skills by name.
    /// The call is isolated: no events are published to any bus.
    pub fn with_llm(mut self, llm: Arc<dyn LlmProvider>) -> Self {
        self.llm = Some(llm);
        self
    }
}

/// Ask the LLM to select relevant skills from a compact manifest.
///
/// Sends only skill names and descriptions (no instructions) to keep the prompt small.
/// Returns skill names chosen by the LLM. On any error returns an empty vec, which
/// causes the caller to fall back to injecting only always-on skills for that cycle.
async fn llm_select_skills(
    llm: &Arc<dyn LlmProvider>,
    candidates: &[&SkillMetadata],
    recent_text: &str,
) -> Vec<String> {
    let manifest: String = candidates
        .iter()
        .map(|s| format!("- {}: {}", s.name, s.description))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = format!(
        "Select the skills relevant to the user's recent request. \
         Return a JSON array of skill names. Return [] if none are relevant.\n\n\
         User's recent messages:\n\"{recent_text}\"\n\n\
         Available skills:\n{manifest}\n\n\
         Return ONLY a JSON array, e.g. [\"skill-name\"] or []."
    );

    match llm.complete(vec![ChatMessage::user(prompt)], vec![]).await {
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
            warn!("SkillsAssembler: LLM skill selection failed: {e}");
            vec![]
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
        use eventage::event::kinds as core_kinds;

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

        // ── Hybrid trigger filtering ──────────────────────────────────────────
        //
        // Skills without triggers are always injected (backward-compatible).
        // Skills with triggers go through one of two resolution paths:
        //
        //   LLM path (when an LLM is configured): the LLM receives a compact
        //   manifest (name + description only) and selects relevant skills by
        //   name. The call is fully isolated — no events touch the main bus.
        //
        //   Keyword path (fallback): existing trigger keyword density scoring,
        //   top-MAX_TRIGGERED_SKILLS by match count.

        // Collect the last 3 user messages as a single lowercase search text.
        let user_texts: Vec<&str> = context
            .events
            .iter()
            .filter(|e| e.kind == core_kinds::USER_MESSAGE)
            .filter_map(|e| e.payload.get("text").and_then(|v| v.as_str()))
            .collect();
        let recent_text = user_texts[user_texts.len().saturating_sub(3)..]
            .join(" ")
            .to_lowercase();

        let (always_on, triggered): (Vec<&SkillMetadata>, Vec<&SkillMetadata>) =
            skills.iter().partition(|s| s.triggers.is_empty());

        let resolved_triggered: Vec<&SkillMetadata> = if triggered.is_empty() {
            vec![]
        } else if let Some(ref llm) = self.llm {
            // ── LLM-based gradual disclosure ──────────────────────────────────
            // Hash the recent text to detect when user input has changed.
            let mut hasher = DefaultHasher::new();
            recent_text.hash(&mut hasher);
            let text_hash = hasher.finish();

            // Check the selection cache first.
            let cached_names: Option<Vec<String>> = {
                let guard = self.selection_cache.lock().await;
                guard
                    .as_ref()
                    .filter(|(h, _)| *h == text_hash)
                    .map(|(_, names)| names.clone())
            };

            let selected_names = if let Some(names) = cached_names {
                names
            } else {
                let names = llm_select_skills(llm, &triggered, &recent_text).await;
                *self.selection_cache.lock().await = Some((text_hash, names.clone()));
                names
            };

            triggered
                .into_iter()
                .filter(|s| {
                    selected_names
                        .iter()
                        .any(|n| n.eq_ignore_ascii_case(&s.name))
                })
                .collect()
        } else {
            // ── Keyword density path (fallback) ───────────────────────────────
            let mut matched: Vec<(&SkillMetadata, usize)> = triggered
                .into_iter()
                .filter_map(|skill| {
                    let score: usize = skill
                        .triggers
                        .iter()
                        .map(|t| recent_text.matches(t.as_str()).count())
                        .sum();
                    if score > 0 {
                        Some((skill, score))
                    } else {
                        None
                    }
                })
                .collect();
            if matched.len() > MAX_TRIGGERED_SKILLS {
                // Highest score first.
                matched.sort_by_key(|(_, score)| std::cmp::Reverse(*score));
                matched.truncate(MAX_TRIGGERED_SKILLS);
            }
            matched.into_iter().map(|(s, _)| s).collect()
        };

        let selected_skills: Vec<&SkillMetadata> =
            always_on.into_iter().chain(resolved_triggered).collect();

        if selected_skills.is_empty() {
            return messages;
        }

        let skills_content: String = selected_skills
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
