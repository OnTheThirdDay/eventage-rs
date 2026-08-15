//! Agent Client Protocol wire types (JSON-RPC 2.0 over stdio).
//!
//! These mirror the ACP schema so editors (Zed, Kiro, Cline, JetBrains, …)
//! can drive this agent as a subprocess. Everything the protocol exchanges
//! lives in this module, so tracking a spec revision means editing one file.
//!
//! Targets **protocol version 1** and negotiates down gracefully; the v2
//! draft (July 2026) changes diff and permission shapes, so
//! [`negotiate_version`] pins what we actually speak.
//!
//! All paths crossing the wire are absolute, and line numbers are 1-based.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Protocol version this agent implements.
pub const PROTOCOL_VERSION: u32 = 1;

/// Agree on a version with the client: the highest we both understand.
pub fn negotiate_version(client_version: Option<u32>) -> u32 {
    client_version
        .unwrap_or(PROTOCOL_VERSION)
        .min(PROTOCOL_VERSION)
}

// ── JSON-RPC envelope ─────────────────────────────────────────────────────────

/// An incoming JSON-RPC message: request (has `id`) or notification.
#[derive(Debug, Deserialize)]
pub struct RpcRequest {
    #[allow(dead_code)]
    pub jsonrpc: Option<String>,
    #[serde(default)]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Serialize)]
pub struct RpcResponse {
    pub jsonrpc: &'static str,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

impl RpcResponse {
    pub fn ok(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    pub fn err(id: Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(RpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// Standard JSON-RPC error codes used by the server.
pub mod codes {
    pub const PARSE_ERROR: i64 = -32700;
    pub const INVALID_REQUEST: i64 = -32600;
    pub const METHOD_NOT_FOUND: i64 = -32601;
    pub const INVALID_PARAMS: i64 = -32602;
    pub const INTERNAL_ERROR: i64 = -32603;
}

/// An outgoing notification (no `id`, no response expected).
#[derive(Debug, Serialize)]
pub struct RpcNotification {
    pub jsonrpc: &'static str,
    pub method: &'static str,
    pub params: Value,
}

impl RpcNotification {
    pub fn new(method: &'static str, params: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            method,
            params,
        }
    }
}

// ── initialize ────────────────────────────────────────────────────────────────

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeRequest {
    #[serde(default)]
    pub protocol_version: Option<u32>,
    #[serde(default)]
    pub client_capabilities: ClientCapabilities,
    #[serde(default)]
    pub client_info: Option<Implementation>,
}

/// What the *client* can do for us — notably whether it will perform file
/// I/O and host terminals on our behalf.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientCapabilities {
    #[serde(default)]
    pub fs: FsCapabilities,
    #[serde(default)]
    pub terminal: bool,
}

#[derive(Debug, Default, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FsCapabilities {
    #[serde(default)]
    pub read_text_file: bool,
    #[serde(default)]
    pub write_text_file: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Implementation {
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResponse {
    pub protocol_version: u32,
    pub agent_capabilities: AgentCapabilities,
    pub auth_methods: Vec<Value>,
    pub agent_info: Implementation,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCapabilities {
    pub load_session: bool,
    pub prompt_capabilities: PromptCapabilities,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptCapabilities {
    pub image: bool,
    pub audio: bool,
    pub embedded_context: bool,
}

// ── sessions ──────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewSessionRequest {
    /// Absolute path the session works in.
    pub cwd: String,
    #[serde(default)]
    pub mcp_servers: Vec<McpServerSpec>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServerSpec {
    #[serde(default)]
    pub name: Option<String>,
    /// stdio servers.
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: Vec<EnvVar>,
    /// HTTP/SSE servers.
    #[serde(default)]
    pub url: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EnvVar {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NewSessionResponse {
    pub session_id: String,
    pub modes: SessionModeState,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadSessionRequest {
    pub session_id: String,
    pub cwd: String,
    #[serde(default)]
    pub mcp_servers: Vec<McpServerSpec>,
}

/// Permission/operating modes, surfaced to the editor as a mode picker.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionModeState {
    pub available_modes: Vec<SessionMode>,
    pub current_mode_id: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionMode {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetModeRequest {
    pub session_id: String,
    pub mode_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptRequest {
    pub session_id: String,
    pub prompt: Vec<ContentBlock>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelNotification {
    pub session_id: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptResponse {
    pub stop_reason: StopReason,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    MaxTurnRequests,
    Refusal,
    Cancelled,
}

// ── content blocks ────────────────────────────────────────────────────────────

/// A piece of prompt or message content. Mirrors MCP's content model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    Audio {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    ResourceLink {
        uri: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Resource {
        resource: EmbeddedResource,
    },
}

impl ContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        ContentBlock::Text { text: text.into() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedResource {
    pub uri: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
    #[serde(default, rename = "mimeType", skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

// ── session/update notifications ──────────────────────────────────────────────

/// One `session/update` notification payload.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionNotification {
    pub session_id: String,
    pub update: SessionUpdate,
}

/// The editor renders each variant differently: message chunks stream into
/// the transcript, tool calls become collapsible cards with diffs, and the
/// plan drives a task checklist.
#[derive(Debug, Serialize)]
#[serde(tag = "sessionUpdate", rename_all = "snake_case")]
pub enum SessionUpdate {
    AgentMessageChunk {
        content: ContentBlock,
    },
    AgentThoughtChunk {
        content: ContentBlock,
    },
    ToolCall(ToolCallUpdate),
    ToolCallUpdate(ToolCallUpdate),
    Plan {
        entries: Vec<PlanEntry>,
    },
    #[serde(rename_all = "camelCase")]
    CurrentModeUpdate {
        current_mode_id: String,
    },
}

#[derive(Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallUpdate {
    pub tool_call_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<ToolKind>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<ToolCallStatus>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub content: Vec<ToolCallContent>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub locations: Vec<ToolCallLocation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_input: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_output: Option<Value>,
}

/// Editors use this to pick an icon for the tool card.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    Read,
    Edit,
    Delete,
    Move,
    Search,
    Execute,
    Think,
    Fetch,
    Other,
}

impl ToolKind {
    /// Classify one of our tools for editor rendering.
    pub fn for_tool(name: &str) -> Self {
        match name {
            "read_file" | "list_directory" => ToolKind::Read,
            "write_file" | "edit_file" | "multi_edit" | "apply_patch" => ToolKind::Edit,
            "delete_file" => ToolKind::Delete,
            "move_file" => ToolKind::Move,
            "grep" | "glob" | "lsp_references" | "lsp_symbols" | "lsp_definition" => {
                ToolKind::Search
            }
            "bash" | "run_tests" | "git" => ToolKind::Execute,
            "plan" | "think" => ToolKind::Think,
            "web_fetch" | "web_search" => ToolKind::Fetch,
            _ => ToolKind::Other,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

/// Content attached to a tool call. `Diff` is what makes an editor render a
/// real side-by-side review of a proposed edit.
///
/// A `terminal` variant exists in the protocol for editor-hosted terminals;
/// we run commands ourselves, so it is added when that is implemented.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolCallContent {
    Content {
        content: ContentBlock,
    },
    #[serde(rename_all = "camelCase")]
    Diff {
        path: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        old_text: Option<String>,
        new_text: String,
    },
}

/// Where a tool acted — editors use this to follow along in the file tree.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallLocation {
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
}

/// One entry of the agent's plan (rendered as a task checklist).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanEntry {
    pub content: String,
    pub priority: PlanPriority,
    pub status: PlanStatus,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanPriority {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    Pending,
    InProgress,
    Completed,
}

// ── session/request_permission (agent → client) ───────────────────────────────

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestPermissionParams {
    pub session_id: String,
    pub tool_call: ToolCallUpdate,
    pub options: Vec<PermissionOption>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionOption {
    pub option_id: String,
    pub name: String,
    pub kind: PermissionOptionKind,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionOptionKind {
    AllowOnce,
    AllowAlways,
    RejectOnce,
    RejectAlways,
}

impl PermissionOption {
    /// The standard four choices an editor shows for a gated tool call.
    pub fn standard_set() -> Vec<Self> {
        vec![
            Self {
                option_id: "allow_once".into(),
                name: "Allow once".into(),
                kind: PermissionOptionKind::AllowOnce,
            },
            Self {
                option_id: "allow_always".into(),
                name: "Always allow this tool".into(),
                kind: PermissionOptionKind::AllowAlways,
            },
            Self {
                option_id: "reject_once".into(),
                name: "Reject".into(),
                kind: PermissionOptionKind::RejectOnce,
            },
            Self {
                option_id: "reject_always".into(),
                name: "Always reject this tool".into(),
                kind: PermissionOptionKind::RejectAlways,
            },
        ]
    }
}

/// The client's answer to a permission request.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestPermissionResult {
    pub outcome: PermissionOutcome,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum PermissionOutcome {
    #[serde(rename_all = "camelCase")]
    Selected {
        option_id: String,
    },
    Cancelled,
}

// ── fs/* and terminal/* (agent → client) ──────────────────────────────────────

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadTextFileParams {
    pub session_id: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct ReadTextFileResult {
    pub content: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WriteTextFileParams {
    pub session_id: String,
    pub path: String,
    pub content: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_negotiation_never_exceeds_ours() {
        assert_eq!(negotiate_version(Some(99)), PROTOCOL_VERSION);
        assert_eq!(negotiate_version(Some(1)), 1);
        assert_eq!(negotiate_version(None), PROTOCOL_VERSION);
    }

    #[test]
    fn session_update_uses_discriminator() {
        let update = SessionUpdate::AgentMessageChunk {
            content: ContentBlock::text("hello"),
        };
        let json = serde_json::to_value(&update).unwrap();
        assert_eq!(json["sessionUpdate"], "agent_message_chunk");
        assert_eq!(json["content"]["type"], "text");
        assert_eq!(json["content"]["text"], "hello");
    }

    #[test]
    fn tool_call_serializes_diff_content() {
        let update = SessionUpdate::ToolCall(ToolCallUpdate {
            tool_call_id: "t1".into(),
            title: Some("Edit src/main.rs".into()),
            kind: Some(ToolKind::Edit),
            status: Some(ToolCallStatus::Completed),
            content: vec![ToolCallContent::Diff {
                path: "/repo/src/main.rs".into(),
                old_text: Some("a".into()),
                new_text: "b".into(),
            }],
            locations: vec![ToolCallLocation {
                path: "/repo/src/main.rs".into(),
                line: Some(12),
            }],
            ..Default::default()
        });
        let json = serde_json::to_value(&update).unwrap();
        assert_eq!(json["sessionUpdate"], "tool_call");
        assert_eq!(json["kind"], "edit");
        assert_eq!(json["status"], "completed");
        assert_eq!(json["content"][0]["type"], "diff");
        assert_eq!(json["content"][0]["newText"], "b");
        assert_eq!(json["locations"][0]["line"], 12);
        // Empty optionals stay off the wire.
        assert!(json.get("rawInput").is_none());
    }

    #[test]
    fn tool_kind_classification() {
        assert!(matches!(ToolKind::for_tool("read_file"), ToolKind::Read));
        assert!(matches!(ToolKind::for_tool("edit_file"), ToolKind::Edit));
        assert!(matches!(ToolKind::for_tool("bash"), ToolKind::Execute));
        assert!(matches!(ToolKind::for_tool("grep"), ToolKind::Search));
        assert!(matches!(ToolKind::for_tool("mystery"), ToolKind::Other));
    }

    #[test]
    fn permission_outcome_parses_both_shapes() {
        let selected: RequestPermissionResult = serde_json::from_value(serde_json::json!({
            "outcome": { "outcome": "selected", "optionId": "allow_once" }
        }))
        .unwrap();
        assert!(matches!(
            selected.outcome,
            PermissionOutcome::Selected { ref option_id } if option_id == "allow_once"
        ));

        let cancelled: RequestPermissionResult = serde_json::from_value(serde_json::json!({
            "outcome": { "outcome": "cancelled" }
        }))
        .unwrap();
        assert!(matches!(cancelled.outcome, PermissionOutcome::Cancelled));
    }

    #[test]
    fn prompt_request_accepts_multimodal_content() {
        let req: PromptRequest = serde_json::from_value(serde_json::json!({
            "sessionId": "s1",
            "prompt": [
                { "type": "text", "text": "what is this?" },
                { "type": "image", "data": "QUJD", "mimeType": "image/png" }
            ]
        }))
        .unwrap();
        assert_eq!(req.session_id, "s1");
        assert_eq!(req.prompt.len(), 2);
        assert!(matches!(req.prompt[1], ContentBlock::Image { .. }));
    }
}
