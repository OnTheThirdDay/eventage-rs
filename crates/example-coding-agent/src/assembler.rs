use async_trait::async_trait;
use eventage::{
    agent::{AssemblyContext, ContextAssembler},
    event::kinds as core_kinds,
    llm::{ChatMessage, LlmProvider, Role},
};
use serde::Deserialize;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
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
            content_tokens + tool_call_tokens + 4 // 4 overhead per message
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

/// Wraps any ContextAssembler and automatically summarizes old conversation
/// when the assembled context exceeds a token budget.
///
/// Summarization is performed by calling the LLM directly and storing the
/// result. Subsequent assembly calls inject the summary in place of the
/// compacted messages.
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
                let content = m.content.as_deref().unwrap_or("(tool calls)");
                format!("{role}: {content}\n")
            })
            .collect();

        let prompt = format!(
            "Summarize the following conversation history. \
             Your summary MUST start with a \"User instructions\" bullet list that quotes, \
             verbatim, every short instruction or correction the user gave (e.g. \"use tool \
             call not in the chat\", \"don't use write_file\"). \
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
        let dir = PathBuf::from("/tmp/coding-agent-history");
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
        debug!("offloaded conversation history to {}", path.display());
    }
}

#[async_trait]
impl ContextAssembler for SummarizingAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let mut messages = self.inner.assemble(context).await;
        let (sys_msgs, conv_msgs) = partition_messages(&messages);
        let budget = (self.max_tokens as f64 * self.threshold) as usize;

        // Apply existing summary if present
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

        // Over budget — compact
        let keep = ((conv_msgs.len() as f64 * self.keep_fraction).ceil() as usize).max(4);
        let cutoff = conv_msgs.len().saturating_sub(keep);

        if cutoff == 0 {
            return messages;
        }

        let to_summarize = &conv_msgs[..cutoff];
        let to_keep = &conv_msgs[cutoff..];

        // Offload and summarize (drop lock before awaiting)
        self.offload_to_file(to_summarize).await;
        let summary = self.do_summarize(to_summarize).await;

        // Update state
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

// ── MemoryAssembler ───────────────────────────────────────────────────────────

/// Loads AGENTS.md files at construction time and injects their contents
/// into every assembled context as a system message.
pub struct MemoryAssembler {
    inner: Arc<dyn ContextAssembler>,
    memory_content: String,
}

impl MemoryAssembler {
    /// Load memory from the given file paths. Missing files are silently skipped.
    pub fn new(inner: Arc<dyn ContextAssembler>, sources: &[PathBuf]) -> Self {
        let mut parts = vec![];
        for path in sources {
            match std::fs::read_to_string(path) {
                Ok(content) => {
                    parts.push(format!("# From: {}\n{}", path.display(), content));
                }
                Err(e) => warn!("could not load memory file {}: {e}", path.display()),
            }
        }
        Self {
            inner,
            memory_content: parts.join("\n\n"),
        }
    }
}

#[async_trait]
impl ContextAssembler for MemoryAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let mut messages = self.inner.assemble(context).await;

        if self.memory_content.is_empty() {
            return messages;
        }

        let memory_msg = ChatMessage::system(format!(
            "<agent_memory>\n{}\n</agent_memory>",
            self.memory_content
        ));

        // Insert after the last system message (so it's closest to the conversation)
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

    // Parse YAML frontmatter between --- markers
    if !content.starts_with("---") {
        return None;
    }

    let rest = &content[3..];
    let end = rest.find("---")?;
    let yaml_str = &rest[..end];
    let instructions = rest[end + 3..].trim().to_string();

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

/// Loads SKILL.md files from skill directories and injects them into context.
pub struct SkillsAssembler {
    inner: Arc<dyn ContextAssembler>,
    skills: Vec<SkillMetadata>,
}

impl SkillsAssembler {
    /// Walk each directory recursively for SKILL.md files.
    pub fn new(inner: Arc<dyn ContextAssembler>, skill_dirs: &[PathBuf]) -> Self {
        let mut skills = vec![];
        for dir in skill_dirs {
            for entry in walkdir::WalkDir::new(dir)
                .follow_links(false)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_name() == "SKILL.md")
            {
                if let Some(skill) = load_skill(entry.path()) {
                    debug!(
                        "loaded skill '{}' from {}",
                        skill.name,
                        entry.path().display()
                    );
                    skills.push(skill);
                }
            }
        }
        Self { inner, skills }
    }
}

#[async_trait]
impl ContextAssembler for SkillsAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let mut messages = self.inner.assemble(context).await;

        if self.skills.is_empty() {
            return messages;
        }

        let skills_content: String = self
            .skills
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
            "<available_skills>\n{}\n</available_skills>",
            skills_content
        ));

        // Append after all system messages
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

/// Uses the LLM to classify user follow-up messages as behavioral
/// instructions/corrections vs. regular tasks, questions, or confirmations.
/// Extracted instructions are injected as a sticky `<user_instructions>` system
/// message that survives `SummarizingAssembler` compaction.
///
/// **Cost**: one extra LLM call per user turn. During the multi-step ReAct loop
/// the result is cached (keyed on the hash of all user messages seen), so the
/// LLM is not called again until the user sends a new message.
pub struct UserCorrectionsAssembler {
    inner: Arc<dyn ContextAssembler>,
    llm: Arc<dyn LlmProvider>,
    /// Cache: hash of all non-first user message texts → extracted instructions.
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

    /// Ask the LLM which messages are behavioral instructions.
    async fn extract_instructions(&self, user_msgs: &[&str]) -> Vec<String> {
        let numbered: String = user_msgs
            .iter()
            .enumerate()
            .map(|(i, msg)| format!("{}. {:?}", i + 1, msg))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "Below are follow-up messages a user sent to an AI coding agent during a session.\n\
             Classify each message as either an INSTRUCTION or NOT.\n\
             \n\
             An INSTRUCTION tells the agent HOW to behave going forward — it corrects or constrains\n\
             the agent's behavior (e.g. \"use tool calls, not text\", \"don't use write_file\",\n\
             \"always read the file before editing\", \"keep responses concise\").\n\
             \n\
             NOT an instruction: a task request (\"fix the auth bug\"), a question\n\
             (\"what does X do?\"), a short acknowledgement (\"ok\", \"thanks\", \"looks good\",\n\
             \"continue\"), or a statement of fact.\n\
             \n\
             Messages:\n\
             {numbered}\n\
             \n\
             Return a JSON array containing verbatim only the INSTRUCTION messages, or [] if none.\n\
             Return ONLY the JSON array — no markdown fences, no explanation."
        );

        let response = self
            .llm
            .complete(vec![ChatMessage::user(prompt)], vec![])
            .await;

        match response {
            Ok(resp) => {
                let content = resp.content.unwrap_or_default();
                // Extract the JSON array even if the LLM wrapped it in prose/fences.
                let s = content.trim();
                let s = s
                    .find('[')
                    .and_then(|start| s.rfind(']').map(|end| &s[start..=end]))
                    .unwrap_or("[]");
                serde_json::from_str::<Vec<String>>(s).unwrap_or_default()
            }
            Err(e) => {
                warn!("UserCorrectionsAssembler: LLM extraction failed: {e}");
                vec![]
            }
        }
    }
}

#[async_trait]
impl ContextAssembler for UserCorrectionsAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let mut messages = self.inner.assemble(context).await;

        // Collect all non-first user messages from the event log.
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
                continue; // skip the initial task description
            }
            if !text.is_empty() {
                user_msgs.push(text);
            }
        }

        if user_msgs.is_empty() {
            return messages;
        }

        // Hash the current set of user messages to detect changes.
        let mut hasher = DefaultHasher::new();
        user_msgs.hash(&mut hasher);
        let current_hash = hasher.finish();

        // Return cached instructions if the user message set hasn't changed
        // (i.e. we're in the same ReAct loop, no new user turn).
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
            debug!(
                count = extracted.len(),
                "UserCorrectionsAssembler: extracted {} instructions via LLM",
                extracted.len()
            );
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
             The user gave the following behavioral instructions during this session.\n\
             STRICTLY follow ALL of them on every response:\n\
             {bullet_list}\n\
             </user_instructions>"
        ));

        // Insert after all existing system messages (closest to the conversation).
        let insert_pos = messages
            .iter()
            .rposition(|m| m.role == Role::System)
            .map(|i| i + 1)
            .unwrap_or(0);

        messages.insert(insert_pos, instruction_msg);
        messages
    }
}
