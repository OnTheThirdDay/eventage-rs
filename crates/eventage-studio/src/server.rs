//! The HTTP surface the app front-end talks to.
//!
//! Everything is JSON except one Server-Sent Events stream per session, which
//! is what makes the transcript and the trace update as the agent works.
//!
//! # Access
//!
//! The server binds to loopback and requires a token minted at startup. That
//! matters more than it sounds: a session's event stream carries prompts,
//! file contents and command output, and loopback alone would leave all of it
//! readable by any other process on the machine. The token travels in the URL
//! Studio opens, is exchanged for a cookie, and is never logged.

use crate::assets;
use crate::backend::{Backend, Session};
use crate::protocol::*;
use anyhow::Result;
use axum::{
    extract::{Path, Query, Request, State},
    http::{header, HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{
        sse::{Event as SseEvent, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{delete, get, post},
    Json, Router,
};
use futures_util::stream::{self, Stream, StreamExt};
use serde::Deserialize;
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::{broadcast::error::RecvError, RwLock};
use tracing::info;

/// The cookie the browser holds after the first load.
const TOKEN_COOKIE: &str = "eventage_studio_token";

#[derive(Clone)]
pub struct AppState {
    pub backend: Arc<dyn Backend>,
    pub sessions: Arc<RwLock<HashMap<String, Arc<dyn Session>>>>,
    pub token: Arc<String>,
}

impl AppState {
    pub fn new(backend: Arc<dyn Backend>, token: String) -> Self {
        Self {
            backend,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            token: Arc::new(token),
        }
    }

    async fn session(&self, id: &str) -> Result<Arc<dyn Session>, ApiError> {
        self.sessions
            .read()
            .await
            .get(id)
            .cloned()
            .ok_or_else(|| ApiError::not_found("no such session"))
    }

    /// Close every session. Called on shutdown so no child process, LSP
    /// server or background turn outlives the app.
    pub async fn shutdown(&self) {
        let sessions: Vec<_> = self
            .sessions
            .write()
            .await
            .drain()
            .map(|(_, s)| s)
            .collect();
        for session in sessions {
            session.shutdown().await;
        }
    }
}

// ── Errors ────────────────────────────────────────────────────────────────────

pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn not_found(message: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
        }
    }
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        // Everything the backends reject is the caller's fault in practice —
        // a busy session, an unknown mode, a workspace that does not exist —
        // and the message is written to be shown to a person.
        Self::bad_request(e.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorBody {
                error: self.message,
            }),
        )
            .into_response()
    }
}

// ── Router ────────────────────────────────────────────────────────────────────

pub fn router(state: AppState) -> Router {
    let api = Router::new()
        .route("/app", get(app_info))
        .route("/sessions", get(list_sessions).post(open_session))
        .route("/sessions/:id", delete(close_session))
        .route("/sessions/:id/events", get(events))
        .route("/sessions/:id/stream", get(stream_events))
        .route("/sessions/:id/prompt", post(prompt))
        .route("/sessions/:id/interrupt", post(interrupt))
        .route("/sessions/:id/mode", post(set_mode))
        .route("/sessions/:id/rewind", post(rewind))
        .route("/sessions/:id/permission", post(permission))
        .route("/sessions/:id/adopt", post(adopt))
        .route("/sessions/:id/seal", post(seal))
        .route("/sessions/:id/summary", post(override_summary))
        .route("/sessions/:id/branch", post(branch))
        .route("/stored/:id", delete(forget_session))
        .route("/model", get(model_settings).post(set_model_settings))
        .route("/fs/list", get(list_dir))
        .with_state(state.clone());

    Router::new()
        .nest("/api", api)
        .route("/", get(index))
        .route("/*path", get(static_asset))
        .layer(middleware::from_fn_with_state(state.clone(), require_token))
        .with_state(state)
}

/// Reject anything that does not carry the startup token.
///
/// The token may arrive as `?t=` — which is how the browser first arrives,
/// and how `EventSource` connects, since it cannot set headers — or as the
/// cookie handed out on that first load.
async fn require_token(State(state): State<AppState>, request: Request, next: Next) -> Response {
    let from_query = request
        .uri()
        .query()
        .and_then(|q| {
            q.split('&')
                .filter_map(|pair| pair.split_once('='))
                .find(|(k, _)| *k == "t")
                .map(|(_, v)| v.to_string())
        })
        .unwrap_or_default();

    let from_cookie = request
        .headers()
        .get(header::COOKIE)
        .and_then(|v| v.to_str().ok())
        .and_then(|cookies| {
            cookies
                .split(';')
                .filter_map(|c| c.trim().split_once('='))
                .find(|(k, _)| *k == TOKEN_COOKIE)
                .map(|(_, v)| v.to_string())
        })
        .unwrap_or_default();

    if from_query == *state.token || from_cookie == *state.token {
        return next.run(request).await;
    }

    (
        StatusCode::UNAUTHORIZED,
        "Eventage Studio: open the URL printed by the app.",
    )
        .into_response()
}

// ── Shell ─────────────────────────────────────────────────────────────────────

/// Serve the app, converting the URL token into a cookie so that subsequent
/// requests — including the event stream — carry it automatically.
async fn index(State(state): State<AppState>) -> Response {
    shell(&state)
}

async fn static_asset(State(state): State<AppState>, Path(path): Path<String>) -> Response {
    if assets::exists(&path) {
        return assets::serve(&path).into_response();
    }
    // Unknown paths are client-side routes, so serve the shell rather than
    // redirecting to `/`: a redirect would drop the `?t=` token before the
    // browser had been given the cookie, and the app would land on a 401.
    shell(&state)
}

fn shell(state: &AppState) -> Response {
    let mut headers = HeaderMap::new();
    if let Ok(cookie) = format!(
        "{TOKEN_COOKIE}={}; Path=/; SameSite=Strict; HttpOnly",
        state.token
    )
    .parse()
    {
        headers.insert(header::SET_COOKIE, cookie);
    }
    (headers, assets::serve("index.html")).into_response()
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn app_info(State(state): State<AppState>) -> Json<AppInfo> {
    Json(state.backend.info())
}

#[derive(serde::Serialize)]
struct SessionList {
    open: Vec<SessionInfo>,
    stored: Vec<StoredSession>,
}

async fn list_sessions(State(state): State<AppState>) -> Json<SessionList> {
    let mut open: Vec<SessionInfo> = state
        .sessions
        .read()
        .await
        .values()
        .map(|s| s.info())
        .collect();
    // Sessions live in a `HashMap`, whose iteration order is deliberately not
    // stable — so without this the sidebar reshuffled on every refresh, and
    // the client's "open the first one" on startup landed somewhere different
    // each launch. Oldest first, which is the order they were opened in.
    open.sort_by(|a, b| a.created_at.cmp(&b.created_at).then(a.id.cmp(&b.id)));
    let open_ids: Vec<&str> = open.iter().map(|s| s.id.as_str()).collect();
    let stored = state
        .backend
        .stored()
        .await
        .into_iter()
        .filter(|s| !open_ids.contains(&s.id.as_str()))
        .collect();
    Json(SessionList { open, stored })
}

async fn open_session(
    State(state): State<AppState>,
    Json(req): Json<NewSessionRequest>,
) -> Result<Json<SessionInfo>, ApiError> {
    // A resumed session keeps its own id, so opening one that is already open
    // would insert a second handle under the same key — dropping the first
    // from the map with nothing left to shut it down, and leaving two agents
    // writing the same log. The one already open is what the caller wants.
    if let Some(id) = req.resume.as_deref() {
        let existing = state.sessions.read().await.get(id).cloned();
        if let Some(session) = existing {
            return Ok(Json(session.info()));
        }
    }

    let session = state.backend.open(req).await?;
    let info = session.info();
    state
        .sessions
        .write()
        .await
        .insert(info.id.clone(), session);
    info!(session = %info.id, cwd = %info.cwd, "session opened");
    Ok(Json(info))
}

async fn close_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let session = state.sessions.write().await.remove(&id);
    match session {
        Some(session) => {
            session.shutdown().await;
            Ok(StatusCode::NO_CONTENT)
        }
        None => Err(ApiError::not_found("no such session")),
    }
}

async fn forget_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    if let Some(session) = state.sessions.write().await.remove(&id) {
        session.shutdown().await;
    }
    state.backend.forget(&id).await?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
struct After {
    #[serde(default)]
    after: u64,
}

async fn events(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(q): Query<After>,
) -> Result<Json<Vec<Arc<StudioEvent>>>, ApiError> {
    let session = state.session(&id).await?;
    Ok(Json(session.feed().since(q.after)))
}

/// Live event stream, resumable from a sequence number.
async fn stream_events(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(q): Query<After>,
) -> Result<Sse<impl Stream<Item = Result<SseEvent, Infallible>>>, ApiError> {
    let session = state.session(&id).await?;
    let feed = session.feed();

    // Subscribe before snapshotting, so nothing published in between is lost;
    // the overlap is then filtered by sequence number.
    let rx = feed.subscribe();
    let backlog = feed.since(q.after);
    let last_sent = backlog.last().map(|e| e.seq).unwrap_or(q.after);

    let live = stream::unfold(rx, |mut rx| async move {
        match rx.recv().await {
            Ok(event) => Some((Some(event), rx)),
            // The client fell too far behind to be served from the live
            // buffer. Tell it to resync rather than silently skipping events
            // it will never otherwise see.
            Err(RecvError::Lagged(n)) => {
                let notice = Arc::new(StudioEvent::studio(
                    "studio.stream.lagged",
                    serde_json::json!({ "missed": n }),
                ));
                Some((Some(notice), rx))
            }
            Err(RecvError::Closed) => None,
        }
    })
    .filter_map(move |event| async move { event.filter(|e| e.seq == 0 || e.seq > last_sent) });

    // Announced before anything else, so a client resuming into a feed that
    // was rebuilt since it last connected can tell — its sequence numbers
    // refer to a numbering that no longer exists, and the backlog it just
    // asked for is the wrong slice.
    let hello = Arc::new(StudioEvent::studio(
        "studio.stream.hello",
        serde_json::json!({ "generation": feed.generation() }),
    ));

    let stream = stream::iter(std::iter::once(hello))
        .chain(stream::iter(backlog))
        .chain(live)
        .map(|event| Ok(SseEvent::default().json_data(&*event).unwrap_or_default()));

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

async fn prompt(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<PromptRequest>,
) -> Result<StatusCode, ApiError> {
    if req.blocks.is_empty() {
        return Err(ApiError::bad_request("nothing to send"));
    }
    state.session(&id).await?.prompt(req.blocks).await?;
    Ok(StatusCode::ACCEPTED)
}

async fn interrupt(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    state.session(&id).await?.interrupt().await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn set_mode(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<ModeRequest>,
) -> Result<StatusCode, ApiError> {
    state.session(&id).await?.set_mode(&req.mode).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// What the settings screen shows. Never includes the credential.
async fn model_settings(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let settings = state.backend.model_settings().ok_or_else(|| {
        ApiError::from(anyhow::anyhow!(
            "this backend does not own the model — the connected agent does"
        ))
    })?;
    Ok(Json(serde_json::json!(settings.view())))
}

/// Apply a change. Takes effect for sessions opened afterwards.
async fn set_model_settings(
    State(state): State<AppState>,
    Json(update): Json<crate::model_settings::ModelUpdate>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let settings = state.backend.model_settings().ok_or_else(|| {
        ApiError::from(anyhow::anyhow!(
            "this backend does not own the model — the connected agent does"
        ))
    })?;
    // The request body carries a credential, so nothing about it is logged —
    // not the body, not a "changed model to …" line that happens to include
    // it. The view that comes back has no key in it.
    let view = settings.set(update).await?;
    tracing::info!(provider = %view.provider, model = %view.model, "model settings changed");
    Ok(Json(serde_json::json!(view)))
}

#[derive(serde::Deserialize)]
struct WorkstreamRequest {
    workstream_id: String,
    /// Why it is being abandoned. Required to seal, ignored to adopt.
    #[serde(default)]
    reason: Option<String>,
    /// Adopt even where the folder has changed under the workstream.
    ///
    /// Defaults to false, so the first attempt always reports conflicts
    /// rather than resolving them by overwriting.
    #[serde(default)]
    force: bool,
}

/// Apply one workstream's result to the folder.
async fn adopt(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<WorkstreamRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let outcome = state
        .session(&id)
        .await?
        .adopt(&req.workstream_id, req.force)
        .await?;
    Ok(Json(serde_json::json!(outcome)))
}

/// Abandon one workstream, recording why.
async fn seal(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<WorkstreamRequest>,
) -> Result<StatusCode, ApiError> {
    // The reason is the point: an epitaph with nothing in it teaches a later
    // attempt nothing, and this is the one field the UI must not skip.
    let why = req.reason.unwrap_or_default();
    if why.trim().is_empty() {
        return Err(ApiError::from(anyhow::anyhow!(
            "say why this workstream is being abandoned — the reason is what a later \
             attempt reads back"
        )));
    }
    state
        .session(&id)
        .await?
        .seal(&req.workstream_id, &why)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(serde::Serialize)]
struct RewindResult {
    remaining: usize,
}

async fn rewind(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<RewindRequest>,
) -> Result<Json<RewindResult>, ApiError> {
    let remaining = state
        .session(&id)
        .await?
        .rewind(req.turns, req.to.as_deref())
        .await?;
    Ok(Json(RewindResult { remaining }))
}

async fn branch(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<BranchRequest>,
) -> Result<Json<SessionInfo>, ApiError> {
    let source = state.session(&id).await?;
    let branched = state.backend.branch(source.as_ref(), req.from_seq).await?;
    let info = branched.info();
    state
        .sessions
        .write()
        .await
        .insert(info.id.clone(), branched);
    info!(from = %id, to = %info.id, at = req.from_seq, "session branched");
    Ok(Json(info))
}

async fn override_summary(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<SummaryOverride>,
) -> Result<StatusCode, ApiError> {
    if req.summary.trim().is_empty() {
        return Err(ApiError::bad_request(
            "an empty summary would silently drop the compacted history",
        ));
    }
    state.session(&id).await?.override_summary(req).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn permission(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(req): Json<PermissionResponse>,
) -> Result<StatusCode, ApiError> {
    state.session(&id).await?.permission(req).await?;
    Ok(StatusCode::NO_CONTENT)
}

// ── Workspace picker ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ListDir {
    path: String,
}

#[derive(serde::Serialize)]
struct DirEntry {
    name: String,
    path: String,
}

#[derive(serde::Serialize)]
struct DirListing {
    path: String,
    parent: Option<String>,
    dirs: Vec<DirEntry>,
}

/// Directories under `path`, for choosing a workspace.
///
/// Only directories are listed: the picker chooses a workspace root, and
/// listing file names would leak more of the filesystem than it needs to.
async fn list_dir(Query(q): Query<ListDir>) -> Result<Json<DirListing>, ApiError> {
    let root = std::fs::canonicalize(&q.path)
        .map_err(|e| ApiError::bad_request(format!("cannot open '{}': {e}", q.path)))?;
    let mut dirs = Vec::new();
    let mut entries = tokio::fs::read_dir(&root)
        .await
        .map_err(|e| ApiError::bad_request(e.to_string()))?;
    while let Ok(Some(entry)) = entries.next_entry().await {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        if entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false) {
            dirs.push(DirEntry {
                path: entry.path().display().to_string(),
                name,
            });
        }
    }
    dirs.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(Json(DirListing {
        parent: root.parent().map(|p| p.display().to_string()),
        path: root.display().to_string(),
        dirs,
    }))
}
