use crate::event::{kinds, Event, EventId};
use crate::llm::types::{ChatMessage, FunctionCall, Role, ToolCall};
use crate::llm::ContentPart;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

// ── AssemblyContext ───────────────────────────────────────────────────────────

/// Context data provided to a [`ContextAssembler`].
///
/// Contains active branch `events` and optionally `rejected_branches`
/// for injecting negative constraints ("what not to do").
///
/// # Example
/// ```no_run
/// # use eventage::agent::AssemblyContext;
/// # let active_events: Vec<eventage::Event> = vec![];
/// # let rejected: Vec<Vec<eventage::Event>> = vec![];
/// // Basic usage (no negative context):
/// let ctx = AssemblyContext::new(&active_events);
///
/// // With negative context from a rollback point:
/// let ctx = AssemblyContext::new(&active_events).with_rejected_branches(rejected);
/// ```
pub struct AssemblyContext<'a> {
    /// Events on the current active branch, in chronological order.
    pub events: &'a [Event],
    /// Events from sealed (rejected) branches that share an ancestor checkpoint
    /// with the current active branch.  Each inner `Vec<Event>` is one rejected branch.
    pub rejected_branches: Vec<Vec<Event>>,
}

impl<'a> AssemblyContext<'a> {
    pub fn new(events: &'a [Event]) -> Self {
        Self {
            events,
            rejected_branches: vec![],
        }
    }

    pub fn with_rejected_branches(mut self, branches: Vec<Vec<Event>>) -> Self {
        self.rejected_branches = branches;
        self
    }
}

// ── ContextAssembler trait ────────────────────────────────────────────────────

/// Transforms the event log into a list of LLM `ChatMessage`s.
///
/// Allows filtering, windowing, summarization, or injecting negative feedback
/// prior to LLM reasoning.
#[async_trait]
pub trait ContextAssembler: Send + Sync {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage>;
}

// ── Internal minimal assemblers (used by AgentBuilder defaults) ───────────────

/// A basic assembler that prepends a fixed system prompt.
pub(crate) struct SystemPromptAssembler {
    pub(crate) system_prompt: String,
}

#[async_trait]
impl ContextAssembler for SystemPromptAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let mut messages = vec![ChatMessage::system(&self.system_prompt)];
        messages.extend(events_to_messages(context.events));
        messages
    }
}

/// A transparent assembler with no system prompt.
pub(crate) struct RawContextAssembler;

#[async_trait]
impl ContextAssembler for RawContextAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        events_to_messages(context.events)
    }
}

// ── Shared event→message conversion ──────────────────────────────────────────

/// If the event payload contains a `"name"` field, attach it to the message.
fn apply_name(msg: ChatMessage, payload: &serde_json::Value) -> ChatMessage {
    match payload.get("name").and_then(|v| v.as_str()) {
        Some(name) if !name.is_empty() => msg.with_name(name),
        _ => msg,
    }
}

/// Build a user message from a payload, honoring multimodal `parts`.
///
/// A payload may carry `{"text": "..."}` (text-only) and/or
/// `{"parts": [ ... ]}` for images and interleaved content. When both are
/// present, `parts` wins — publishers should include the text as a text part.
fn user_message_from_payload(payload: &serde_json::Value) -> Option<ChatMessage> {
    if let Some(parts) = payload.get("parts") {
        if let Ok(parts) = serde_json::from_value::<Vec<ContentPart>>(parts.clone()) {
            if !parts.is_empty() {
                return Some(apply_name(ChatMessage::user_with_parts(parts), payload));
            }
        }
    }
    let text = payload.get("text").and_then(|v| v.as_str())?;
    Some(apply_name(ChatMessage::user(text), payload))
}

/// Converts raw events to `ChatMessage`s.
///
/// Includes a repair pass: if an `assistant` message declares `tool_calls` but
/// some of their results are missing (e.g. due to a tool panic), synthetic
/// The image a tool result is carrying, if any.
///
/// Tools advertise it the same way they advertise diffs and locations: an
/// underscore-prefixed key the harness understands and the model never has to
/// parse.
fn image_part(payload: &serde_json::Value) -> Option<ContentPart> {
    let image = payload.get("result")?.get("_image")?;
    Some(ContentPart::image_base64(
        image.get("media_type")?.as_str()?,
        image.get("data")?.as_str()?,
    ))
}

/// `tool` messages with an error are inserted so the history stays valid for
/// any LLM provider that enforces the "all tool_call_ids must be answered" rule.
pub fn events_to_messages(events: &[Event]) -> Vec<ChatMessage> {
    let raw = events_to_messages_raw(events);
    repair_tool_results(raw)
}

/// Raw conversion without the repair pass (used internally and in tests).
fn events_to_messages_raw(events: &[Event]) -> Vec<ChatMessage> {
    let mut messages = Vec::new();

    for event in events {
        match event.kind.as_str() {
            kinds::USER_MESSAGE => {
                if let Some(msg) = user_message_from_payload(&event.payload) {
                    messages.push(msg);
                }
            }
            kinds::ASSISTANT_MESSAGE => {
                let content = event
                    .payload
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let tool_calls: Vec<ToolCall> = event
                    .payload
                    .get("tool_calls")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|tc| {
                                Some(ToolCall {
                                    id: tc.get("id")?.as_str()?.to_string(),
                                    kind: tc
                                        .get("type")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("function")
                                        .to_string(),
                                    function: FunctionCall {
                                        name: tc
                                            .get("function")?
                                            .get("name")?
                                            .as_str()?
                                            .to_string(),
                                        arguments: tc
                                            .get("function")?
                                            .get("arguments")?
                                            .as_str()?
                                            .to_string(),
                                    },
                                    extra_content: tc.get("extra_content").cloned(),
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                let provider_extra = event.payload.get("provider_extra").cloned();

                let mut msg = if !tool_calls.is_empty() {
                    Some(ChatMessage::assistant_with_tool_calls(content, tool_calls))
                } else if let Some(text) = content {
                    Some(ChatMessage::assistant(text))
                } else if provider_extra.is_some() {
                    // Reasoning-only turn (e.g. thinking followed by tool use
                    // that got vetoed): keep the message so provider state
                    // still round-trips. `tool_calls` stays None so the
                    // OpenAI wire format remains valid.
                    Some(ChatMessage::assistant(String::new()))
                } else {
                    None
                };
                if let (Some(m), Some(extra)) = (msg.as_mut(), provider_extra) {
                    m.provider_extra = Some(extra);
                }
                if let Some(m) = msg {
                    messages.push(m);
                }
            }
            kinds::TOOL_RESULT => {
                if let Some(id) = event.payload.get("tool_call_id").and_then(|v| v.as_str()) {
                    let result = event
                        .payload
                        .get("result")
                        .map(|v| v.to_string())
                        .or_else(|| event.payload.get("error").map(|e| format!("error: {}", e)))
                        .unwrap_or_default();
                    messages.push(ChatMessage::tool_result(id, result));

                    // A tool that produced an image gets it in front of the
                    // model as an image rather than as base64 in a string.
                    //
                    // It arrives as a follow-up user message because a tool
                    // message is text-only on most providers; only Anthropic
                    // accepts image blocks inside a tool result, and routing
                    // it through a user turn works everywhere.
                    if let Some(part) = image_part(&event.payload) {
                        messages.push(ChatMessage::user_with_parts(vec![
                            ContentPart::text("(image returned by the previous tool call)"),
                            part,
                        ]));
                    }
                }
            }
            kinds::AGENT_MESSAGE => {
                if let Some(msg) = user_message_from_payload(&event.payload) {
                    messages.push(msg);
                }
            }
            kinds::SYSTEM_MESSAGE => {
                if let Some(msg) = user_message_from_payload(&event.payload) {
                    messages.push(msg);
                }
            }
            kinds::AGENT_STUCK => {
                // Surface the loop-detection hint to the model. Without this the
                // agent.stuck event exists only for observers and the LLM never
                // learns it is looping.
                if let Some(hint) = event.payload.get("hint").and_then(|v| v.as_str()) {
                    let pattern = event
                        .payload
                        .get("kind")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Loop");
                    messages.push(
                        ChatMessage::user(format!("[harness] Loop detected ({pattern}): {hint}"))
                            .with_name("loop_detector"),
                    );
                }
            }
            _ => {}
        }
    }

    messages
}

/// Scan a message list and insert synthetic `tool` error messages for any
/// `tool_call_id` that appears in an `assistant` message but has no matching
/// `tool` result later in the list.
///
/// This guards against corrupt histories caused by tool panics or other
/// unexpected interruptions that prevented a `tool.result` event from being
/// published.  Without this repair the LLM provider returns a 400 on every
/// subsequent call, permanently deadlocking the agent.
fn repair_tool_results(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
    use std::collections::HashSet;

    // Fast path: no repair needed if there are no assistant messages with tool calls.
    let needs_repair = messages.iter().any(|m| {
        m.role == Role::Assistant && m.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty())
    });
    if !needs_repair {
        return messages;
    }

    // Collect all tool_call_ids that have a tool result.
    let answered: HashSet<&str> = messages
        .iter()
        .filter(|m| m.role == Role::Tool)
        .filter_map(|m| m.tool_call_id.as_deref())
        .collect();

    // Rebuild, injecting synthetic results immediately after any assistant message
    // that references an unanswered tool_call_id.
    let mut out = Vec::with_capacity(messages.len());
    for msg in &messages {
        if msg.role == Role::Assistant {
            if let Some(tcs) = &msg.tool_calls {
                out.push(msg.clone());
                for tc in tcs {
                    if !answered.contains(tc.id.as_str()) {
                        out.push(ChatMessage::tool_result(
                            &tc.id,
                            format!(
                                "error: tool '{}' execution was interrupted and produced no result",
                                tc.function.name
                            ),
                        ));
                    }
                }
                continue;
            }
        }
        out.push(msg.clone());
    }
    out
}

// ── DefaultContextAssembler ───────────────────────────────────────────────────

/// Incremental cache for [`DefaultContextAssembler`].
struct MessageCache {
    last_event_id: EventId,
    event_count: usize,
    messages: Vec<ChatMessage>,
}

/// Converts events sequentially into OpenAI-style chat messages.
///
/// Handles `user.message`, `assistant.message`, and `tool.result`. Other events
/// are skipped. Context boundaries use `NegativeAwareContextAssembler`.
///
/// # Performance
/// Features incremental caching (O(new_events)) and an opt-in sliding window
/// `with_max_events` bounding work to O(N).
pub struct DefaultContextAssembler {
    pub system_prompt: Option<String>,
    max_events: Option<usize>,
    cache: Mutex<Option<MessageCache>>,
}

impl DefaultContextAssembler {
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: Some(system_prompt.into()),
            max_events: None,
            cache: Mutex::new(None),
        }
    }

    pub fn without_system_prompt() -> Self {
        Self {
            system_prompt: None,
            max_events: None,
            cache: Mutex::new(None),
        }
    }

    /// Caps the context window to the most recent `n` events.
    ///
    /// Keeps the LLM prompt bounded. Older events are excluded but remain in the bus.
    /// Bypasses incremental caching.
    pub fn with_max_events(mut self, n: usize) -> Self {
        self.max_events = Some(n);
        self
    }
}

#[async_trait]
impl ContextAssembler for DefaultContextAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let mut messages = Vec::new();

        if let Some(prompt) = &self.system_prompt {
            messages.push(ChatMessage::system(prompt));
        }

        // Sliding window path — bypass cache.
        if let Some(max) = self.max_events {
            let start = context.events.len().saturating_sub(max);
            messages.extend(events_to_messages(&context.events[start..]));
            return messages;
        }

        // Incremental cache path.
        let mut cache = self.cache.lock().unwrap();
        messages.extend(build_with_cache(&mut cache, context.events));
        messages
    }
}

/// Build (or extend) a `Vec<ChatMessage>` using the incremental cache.
fn build_with_cache(cache: &mut Option<MessageCache>, events: &[Event]) -> Vec<ChatMessage> {
    if events.is_empty() {
        *cache = None;
        return vec![];
    }

    let current_last_id = events.last().unwrap().id;

    match cache.as_mut() {
        Some(c) if c.last_event_id == current_last_id => c.messages.clone(),

        Some(c)
            if c.event_count > 0
                && events.len() >= c.event_count
                && events[c.event_count - 1].id == c.last_event_id =>
        {
            let new_msgs = events_to_messages(&events[c.event_count..]);
            c.messages.extend(new_msgs);
            c.event_count = events.len();
            c.last_event_id = current_last_id;
            c.messages.clone()
        }

        _ => {
            let messages = events_to_messages(events);
            *cache = Some(MessageCache {
                last_event_id: current_last_id,
                event_count: events.len(),
                messages: messages.clone(),
            });
            messages
        }
    }
}

// ── NegativeAwareContextAssembler ─────────────────────────────────────────────

/// Type alias for the negative context formatter function.
type NegativeContextFormatter = dyn Fn(&[Vec<Event>]) -> String + Send + Sync;

/// Wraps any [`ContextAssembler`] to inject negative-trajectory summaries
/// for failed branches. Steering the LLM away from repeated mistakes.
///
/// # Customizing the failure summary
///
/// By default the summary uses [`default_negative_context_format`] which
/// produces plain-language bullet points. Override it with [`with_formatter`]
/// to produce JSON blocks, XML tags, or any other format your model responds
/// to best.
pub struct NegativeAwareContextAssembler {
    inner: Box<dyn ContextAssembler>,
    formatter: Box<NegativeContextFormatter>,
}

impl NegativeAwareContextAssembler {
    pub fn new(inner: impl ContextAssembler + 'static) -> Self {
        Self {
            inner: Box::new(inner),
            formatter: Box::new(default_negative_context_format),
        }
    }

    /// Replace the default formatter with a custom one.
    pub fn with_formatter<F>(mut self, formatter: F) -> Self
    where
        F: Fn(&[Vec<Event>]) -> String + Send + Sync + 'static,
    {
        self.formatter = Box::new(formatter);
        self
    }
}

#[async_trait]
impl ContextAssembler for NegativeAwareContextAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let mut messages = self.inner.assemble(context).await;

        if !context.rejected_branches.is_empty() {
            let summary = (self.formatter)(&context.rejected_branches);
            let insert_pos = messages
                .iter()
                .position(|m| m.role != Role::System)
                .unwrap_or(messages.len());
            messages.insert(insert_pos, ChatMessage::system(summary));
        }

        messages
    }
}

/// Default formatter for rejected-trajectory context injection.
///
/// Produces a plain-English summary of each failed branch listing the assistant
/// messages and tool errors observed.
pub fn default_negative_context_format(branches: &[Vec<Event>]) -> String {
    let mut parts = vec![
        "⚠ Previous attempt(s) on this task failed. Do NOT repeat these approaches:".to_string(),
    ];

    for (i, branch) in branches.iter().enumerate() {
        let label = format!("Attempt {}:", i + 1);
        let mut details = vec![label];

        for event in branch {
            match event.kind.as_str() {
                kinds::ASSISTANT_MESSAGE => {
                    if let Some(text) = event.payload.get("content").and_then(|v| v.as_str()) {
                        if !text.is_empty() {
                            details.push(format!("  Assistant said: {}", truncate(text, 200)));
                        }
                    }
                    if let Some(calls) = event.payload.get("tool_calls").and_then(|v| v.as_array())
                    {
                        for tc in calls {
                            let name = tc
                                .get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|n| n.as_str())
                                .unwrap_or("?");
                            let args = tc
                                .get("function")
                                .and_then(|f| f.get("arguments"))
                                .and_then(|a| a.as_str())
                                .unwrap_or("{}");
                            details.push(format!(
                                "  Called tool: {}({})",
                                name,
                                truncate(args, 150)
                            ));
                        }
                    }
                }
                kinds::TOOL_RESULT => {
                    if let Some(err) = event.payload.get("error").and_then(|v| v.as_str()) {
                        details.push(format!("  Tool error: {}", truncate(err, 200)));
                    }
                }
                _ => {}
            }
        }

        parts.push(details.join("\n"));
    }

    parts.join("\n")
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        // Walk back from `max` to a valid UTF-8 char boundary to avoid panics.
        let mut end = max.min(s.len());
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

// ── DynamicContextAssembler ───────────────────────────────────────────────────

/// A [`ContextAssembler`] whose inner assembler can be swapped at runtime.
///
/// All clones of a `DynamicContextAssembler` share the same inner state. Call
/// [`swap`] to atomically replace the assembler — the agent uses the new one
/// on its next ReAct step.
#[derive(Clone)]
pub struct DynamicContextAssembler {
    inner: Arc<Mutex<Arc<dyn ContextAssembler>>>,
}

impl DynamicContextAssembler {
    /// Create a new dynamic assembler with the given initial implementation.
    pub fn new(assembler: impl ContextAssembler + 'static) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Arc::new(assembler))),
        }
    }

    /// Atomically replace the active assembler.
    pub fn swap(&self, assembler: impl ContextAssembler + 'static) {
        *self.inner.lock().unwrap_or_else(|e| e.into_inner()) = Arc::new(assembler);
    }

    /// Replace with a pre-boxed assembler.
    pub fn swap_arc(&self, assembler: Arc<dyn ContextAssembler>) {
        *self.inner.lock().unwrap_or_else(|e| e.into_inner()) = assembler;
    }
}

#[async_trait]
impl ContextAssembler for DynamicContextAssembler {
    async fn assemble(&self, context: &AssemblyContext<'_>) -> Vec<ChatMessage> {
        let assembler = self.inner.lock().unwrap_or_else(|e| e.into_inner()).clone();
        assembler.assemble(context).await
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::kinds;
    use serde_json::json;

    #[test]
    fn an_image_from_a_tool_reaches_the_model_as_an_image() {
        // Tool messages are text on most providers, so an image a tool
        // produced has to arrive as a following user turn or the model only
        // ever sees a base64 string.
        let events = vec![
            Event::new(
                kinds::ASSISTANT_MESSAGE,
                json!({ "content": null, "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": { "name": "view_image", "arguments": "{}" }
                }]}),
            ),
            Event::new(
                kinds::TOOL_RESULT,
                json!({
                    "tool_call_id": "c1",
                    "result": {
                        "path": "mockup.png",
                        "_image": { "media_type": "image/png", "data": "QUJD" }
                    }
                }),
            ),
        ];
        let messages = events_to_messages(&events);
        let carried = messages
            .iter()
            .find(|m| m.is_multimodal())
            .expect("an image part should have been added");
        assert!(carried
            .content_parts()
            .iter()
            .any(|p| matches!(p, crate::llm::content::ContentPart::Image { .. })));
    }

    #[test]
    fn a_tool_result_without_an_image_is_left_alone() {
        let events = vec![
            Event::new(
                kinds::ASSISTANT_MESSAGE,
                json!({ "content": null, "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": { "name": "read_file", "arguments": "{}" }
                }]}),
            ),
            Event::new(
                kinds::TOOL_RESULT,
                json!({ "tool_call_id": "c1", "result": { "content": "hello" } }),
            ),
        ];
        assert!(!events_to_messages(&events).iter().any(|m| m.is_multimodal()));
    }

    #[tokio::test]
    async fn assembles_user_message() {
        let assembler = RawContextAssembler;
        let events = vec![Event::new(kinds::USER_MESSAGE, json!({"text": "hello"}))];
        let ctx = AssemblyContext::new(&events);
        let messages = assembler.assemble(&ctx).await;
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content.as_deref(), Some("hello"));
    }

    #[tokio::test]
    async fn skips_observability_events() {
        let assembler = RawContextAssembler;
        let events = vec![
            Event::new(kinds::AGENT_CYCLE_START, json!({})),
            Event::new(kinds::USER_MESSAGE, json!({"text": "hi"})),
            Event::new(kinds::AGENT_CYCLE_END, json!({})),
        ];
        let ctx = AssemblyContext::new(&events);
        let messages = assembler.assemble(&ctx).await;
        assert_eq!(messages.len(), 1);
    }

    #[tokio::test]
    async fn system_prompt_assembler_prepends_system() {
        let assembler = SystemPromptAssembler {
            system_prompt: "Be helpful.".to_string(),
        };
        let events = vec![Event::new(kinds::USER_MESSAGE, json!({"text": "hi"}))];
        let ctx = AssemblyContext::new(&events);
        let messages = assembler.assemble(&ctx).await;
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(messages[0].content.as_deref(), Some("Be helpful."));
    }

    #[tokio::test]
    async fn default_assembler_assembles_user_message() {
        let assembler = DefaultContextAssembler::without_system_prompt();
        let events = vec![Event::new(kinds::USER_MESSAGE, json!({"text": "hello"}))];
        let ctx = AssemblyContext::new(&events);
        let messages = assembler.assemble(&ctx).await;
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content.as_deref(), Some("hello"));
    }

    #[tokio::test]
    async fn sliding_window_limits_context() {
        let assembler = DefaultContextAssembler::without_system_prompt().with_max_events(2);
        let events = vec![
            Event::new(kinds::USER_MESSAGE, json!({"text": "a"})),
            Event::new(kinds::USER_MESSAGE, json!({"text": "b"})),
            Event::new(kinds::USER_MESSAGE, json!({"text": "c"})),
        ];
        let ctx = AssemblyContext::new(&events);
        let messages = assembler.assemble(&ctx).await;
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content.as_deref(), Some("b"));
        assert_eq!(messages[1].content.as_deref(), Some("c"));
    }

    #[tokio::test]
    async fn incremental_cache_extends_on_new_events() {
        let assembler = DefaultContextAssembler::without_system_prompt();

        let events1 = vec![Event::new(kinds::USER_MESSAGE, json!({"text": "first"}))];
        let ctx1 = AssemblyContext::new(&events1);
        let msgs1 = assembler.assemble(&ctx1).await;
        assert_eq!(msgs1.len(), 1);

        let mut events2 = events1.clone();
        events2.push(Event::new(kinds::USER_MESSAGE, json!({"text": "second"})));
        let ctx2 = AssemblyContext::new(&events2);
        let msgs2 = assembler.assemble(&ctx2).await;
        assert_eq!(msgs2.len(), 2);
        assert_eq!(msgs2[1].content.as_deref(), Some("second"));
    }

    #[tokio::test]
    async fn negative_aware_assembler_injects_warning() {
        let base = DefaultContextAssembler::without_system_prompt();
        let assembler = NegativeAwareContextAssembler::new(base);

        let active = vec![Event::new(
            kinds::USER_MESSAGE,
            json!({"text": "try again"}),
        )];
        let rejected_event = Event::new(
            kinds::ASSISTANT_MESSAGE,
            json!({
                "content": "I tried approach X",
                "tool_calls": []
            }),
        );
        let rejected_branches = vec![vec![rejected_event]];

        let ctx = AssemblyContext::new(&active).with_rejected_branches(rejected_branches);
        let messages = assembler.assemble(&ctx).await;

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert!(messages[0]
            .content
            .as_deref()
            .unwrap_or("")
            .contains("failed"));
    }
}
