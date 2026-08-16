//! The wire model the UI consumes.
//!
//! Everything the app shows — the transcript, the tool cards, the trace panel
//! — is derived from one ordered stream of [`StudioEvent`]. That mirrors how
//! the framework itself works: the event log is the source of truth and every
//! view is re-derived from it, so the chat and the trace can never disagree
//! about what happened.
//!
//! Both backends normalise onto this shape. The local backend forwards the
//! session's own bus events verbatim; the ACP backend synthesises the same
//! `kind`s from protocol notifications. The UI therefore needs one reducer,
//! not two, and the only difference between the modes is how much detail the
//! trace panel has to show.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// One thing that happened, as the UI sees it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudioEvent {
    /// Position in this session's stream, starting at 1.
    ///
    /// The UI passes the last one it saw back as `?after=` when a dropped
    /// connection is re-established, so a reconnect resumes rather than
    /// replaying the session from the beginning.
    pub seq: u64,
    /// The originating event's id where there is one, else a generated id.
    pub id: String,
    pub ts: String,
    /// Dot-separated kind, e.g. `assistant.delta`, `tool.result`.
    pub kind: String,
    pub payload: Value,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub meta: HashMap<String, Value>,
    /// Parent in the event DAG. Present only in local mode, where the UI can
    /// show branch structure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
}

impl StudioEvent {
    /// Build an event that Studio itself originated rather than observed.
    ///
    /// These carry the `studio.` prefix so the trace panel can tell the
    /// app's own bookkeeping apart from the agent's activity.
    pub fn studio(kind: &str, payload: Value) -> Self {
        Self {
            seq: 0,
            id: uuid::Uuid::new_v4().to_string(),
            ts: chrono::Utc::now().to_rfc3339(),
            kind: kind.to_string(),
            payload,
            meta: HashMap::new(),
            parent: None,
        }
    }
}

impl From<&eventage::Event> for StudioEvent {
    fn from(event: &eventage::Event) -> Self {
        Self {
            seq: 0, // assigned by the feed on push
            id: event.id.to_string(),
            ts: event.timestamp.to_rfc3339(),
            kind: event.kind.clone(),
            payload: event.payload.clone(),
            meta: event.metadata.clone(),
            parent: event.parent_event_id.map(|id| id.to_string()),
        }
    }
}

// ── Studio-originated event kinds ─────────────────────────────────────────────

/// Kinds Studio publishes itself, for things the agent's own log does not
/// record because they happen outside a turn.
pub mod studio_kinds {
    /// A turn was interrupted by the user pressing stop.
    pub const TURN_INTERRUPTED: &str = "studio.turn.interrupted";
    /// A turn ended, with the reason. Closes out the UI's "thinking" state.
    pub const TURN_ENDED: &str = "studio.turn.ended";
    /// The session failed to complete a turn.
    pub const TURN_FAILED: &str = "studio.turn.failed";
    /// The permission mode changed.
    pub const MODE_CHANGED: &str = "studio.mode.changed";
    /// The conversation was rewound by N turns.
    pub const REWOUND: &str = "studio.rewound";
    /// The backend connection dropped (ACP mode: the child process exited).
    pub const BACKEND_LOST: &str = "studio.backend.lost";
    /// The connected agent asked to read or write outside the workspace.
    pub const FS_REFUSED: &str = "studio.fs.refused";
}

// ── Requests ──────────────────────────────────────────────────────────────────

/// A block of user input. Structurally identical to the ACP content block, so
/// it passes through to an ACP agent unchanged and converts cleanly to the
/// multimodal parts the local agent's bus expects.
pub type PromptBlock = eventage_code::acp::wire::ContentBlock;

#[derive(Debug, Clone, Deserialize)]
pub struct NewSessionRequest {
    /// Workspace root. Defaults to the directory Studio was launched in.
    #[serde(default)]
    pub cwd: Option<String>,
    /// Permission mode id: `plan` | `ask` | `auto` | `yolo`.
    #[serde(default)]
    pub mode: Option<String>,
    /// Reopen a persisted session with this id instead of starting fresh.
    #[serde(default)]
    pub resume: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PromptRequest {
    pub blocks: Vec<PromptBlock>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModeRequest {
    pub mode: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RewindRequest {
    #[serde(default = "one")]
    pub turns: usize,
    /// Rewind to this checkpoint instead of counting turns backwards.
    #[serde(default)]
    pub to: Option<String>,
}

fn one() -> usize {
    1
}

/// Start a new session from this one's history up to a point.
#[derive(Debug, Clone, Deserialize)]
pub struct BranchRequest {
    /// Keep events up to and including this sequence number.
    pub from_seq: u64,
}

/// Replace the summary compaction produced.
#[derive(Debug, Clone, Deserialize)]
pub struct SummaryOverride {
    pub summary: String,
    /// How many conversation messages the replacement covers.
    pub summarized_count: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PermissionResponse {
    pub request_id: String,
    pub approve: bool,
    #[serde(default)]
    pub reason: Option<String>,
    /// Remember this answer for the rest of the session.
    #[serde(default)]
    pub always: bool,
}

// ── Responses ─────────────────────────────────────────────────────────────────

/// What the UI needs to render its title bar and mode menu.
#[derive(Debug, Clone, Serialize)]
pub struct AppInfo {
    pub backend: &'static str,
    pub backend_detail: String,
    pub model: String,
    pub provider: String,
    pub default_cwd: String,
    pub modes: Vec<ModeInfo>,
    pub version: &'static str,
    /// False in ACP mode, where the protocol carries no event DAG. The UI
    /// uses this to explain a thinner trace rather than looking broken.
    pub full_trace: bool,
    /// Set when no credentials were found, so the app can say so up front
    /// rather than letting the first message fail with a connection error
    /// against a default endpoint the user never chose.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials_hint: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModeInfo {
    pub id: String,
    pub label: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionInfo {
    pub id: String,
    pub cwd: String,
    pub mode: String,
    pub title: String,
    pub created_at: String,
    pub running: bool,
    /// Number of completed turns — what `rewind` counts in.
    pub turns: usize,
}

/// A session on disk that is not currently open.
#[derive(Debug, Clone, Serialize)]
pub struct StoredSession {
    pub id: String,
    pub cwd: String,
    pub title: String,
    pub updated_at: String,
    pub size_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub error: String,
}
