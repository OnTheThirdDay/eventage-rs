use async_trait::async_trait;
use eventage_agent::{context::events_to_messages, AssemblyContext, ContextAssembler};
use eventage_llm::types::ChatMessage;
use std::sync::Arc;

use crate::workspace::Workspace;

/// A context assembler tuned for C development.
///
/// Produces a prompt with three sections:
/// 1. System prompt — agent role and tool documentation (always present).
/// 2. Workspace status — current file list (re-computed each call so the LLM
///    always sees the up-to-date state of the workspace).
/// 3. Conversation window — the last `max_messages` conversation messages
///    (user turns, assistant turns with optional tool_calls, tool results).
///    Older messages are summarised as a brief omission notice.
pub struct CAgentContextAssembler {
    pub system_prompt: String,
    /// Maximum number of conversation messages to send to the LLM.
    pub max_messages: usize,
    pub workspace: Arc<Workspace>,
}

#[async_trait]
impl ContextAssembler for CAgentContextAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        // ── 1. System prompt ──────────────────────────────────────────────
        let mut messages = vec![ChatMessage::system(&self.system_prompt)];

        // ── 2. Workspace status ───────────────────────────────────────────
        let status_line = match self.workspace.list_files() {
            Err(_) => "[workspace: could not read file list]".to_string(),
            Ok(files) if files.is_empty() => "[workspace: empty — no files yet]".to_string(),
            Ok(files) => {
                let entries: Vec<String> = files
                    .iter()
                    .map(|f| format!("{} ({}B)", f.path, f.size_bytes))
                    .collect();
                format!("[workspace files: {}]", entries.join(", "))
            }
        };
        messages.push(ChatMessage::system(status_line));

        // ── 3. Sliding conversation window ────────────────────────────────
        let conv = events_to_messages(context.events);
        if conv.is_empty() {
            return messages;
        }

        if conv.len() <= self.max_messages {
            messages.extend(conv);
        } else {
            let omitted = conv.len() - self.max_messages;
            messages.push(ChatMessage::system(format!(
                "[{omitted} earlier messages omitted — context window applied]"
            )));
            messages.extend(conv[omitted..].to_vec());
        }

        messages
    }
}
