//! The HTTP surface, exercised against a stand-in backend.
//!
//! No model, no API key, no network beyond loopback: the point is the parts
//! Studio owns — the token gate, session routing, event resumption and the
//! live stream — none of which should need an LLM to verify.

use async_trait::async_trait;
use eventage_studio::backend::{Backend, Session};
use eventage_studio::feed::EventFeed;
use eventage_studio::protocol::*;
use eventage_studio::server::{router, AppState};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;

// ── A backend that does nothing but record ────────────────────────────────────

struct FakeBackend {
    session: Arc<FakeSession>,
}

struct FakeSession {
    feed: Arc<EventFeed>,
    prompts: Mutex<Vec<String>>,
    interrupted: Mutex<bool>,
    mode: Mutex<String>,
    decisions: Mutex<Vec<(String, bool)>>,
    summaries: Mutex<Vec<String>>,
}

impl FakeSession {
    fn new() -> Self {
        Self {
            feed: Arc::new(EventFeed::new()),
            prompts: Mutex::new(Vec::new()),
            interrupted: Mutex::new(false),
            mode: Mutex::new("ask".into()),
            decisions: Mutex::new(Vec::new()),
            summaries: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Backend for FakeBackend {
    fn info(&self) -> AppInfo {
        AppInfo {
            backend: "local",
            backend_detail: "test double".into(),
            model: "test-model".into(),
            provider: "test".into(),
            default_cwd: "/tmp".into(),
            modes: vec![ModeInfo {
                id: "ask".into(),
                label: "Ask".into(),
                description: "test".into(),
            }],
            version: "0.0.0",
            full_trace: true,
            credentials_hint: None,
        }
    }

    async fn open(&self, _req: NewSessionRequest) -> anyhow::Result<Arc<dyn Session>> {
        Ok(self.session.clone())
    }

    async fn branch(
        &self,
        _source: &dyn Session,
        from_seq: u64,
    ) -> anyhow::Result<Arc<dyn Session>> {
        if from_seq == 0 {
            anyhow::bail!("there is nothing before that point to branch from");
        }
        Ok(self.session.clone())
    }

    async fn stored(&self) -> Vec<StoredSession> {
        Vec::new()
    }

    async fn forget(&self, _id: &str) -> anyhow::Result<()> {
        Ok(())
    }
}

#[async_trait]
impl Session for FakeSession {
    fn feed(&self) -> Arc<EventFeed> {
        Arc::clone(&self.feed)
    }

    fn info(&self) -> SessionInfo {
        SessionInfo {
            id: "s1".into(),
            cwd: "/tmp".into(),
            mode: self
                .mode
                .try_lock()
                .map(|m| m.clone())
                .unwrap_or_else(|_| "ask".into()),
            title: "test".into(),
            created_at: "2026-08-15T00:00:00Z".into(),
            running: false,
            turns: self.feed.count_turns(),
        }
    }

    async fn prompt(&self, blocks: Vec<PromptBlock>) -> anyhow::Result<()> {
        for block in &blocks {
            if let PromptBlock::Text { text } = block {
                self.prompts.lock().await.push(text.clone());
            }
        }
        self.feed.push(StudioEvent::studio(
            "user.message",
            json!({ "text": "recorded" }),
        ));
        Ok(())
    }

    async fn interrupt(&self) -> anyhow::Result<()> {
        *self.interrupted.lock().await = true;
        Ok(())
    }

    async fn set_mode(&self, mode: &str) -> anyhow::Result<()> {
        if mode == "nonsense" {
            anyhow::bail!("unknown permission mode 'nonsense'");
        }
        *self.mode.lock().await = mode.to_string();
        Ok(())
    }

    async fn rewind(&self, turns: usize, to: Option<&str>) -> anyhow::Result<usize> {
        Ok(if to.is_some() { 0 } else { turns })
    }

    async fn override_summary(&self, replacement: SummaryOverride) -> anyhow::Result<()> {
        self.summaries.lock().await.push(replacement.summary);
        Ok(())
    }

    async fn permission(&self, response: PermissionResponse) -> anyhow::Result<()> {
        self.decisions
            .lock()
            .await
            .push((response.request_id, response.approve));
        Ok(())
    }

    async fn shutdown(&self) {}
}

// ── Harness ───────────────────────────────────────────────────────────────────

struct Studio {
    base: String,
    token: String,
    session: Arc<FakeSession>,
    client: reqwest::Client,
}

impl Studio {
    async fn start() -> Self {
        let session = Arc::new(FakeSession::new());
        let backend = Arc::new(FakeBackend {
            session: session.clone(),
        });
        let token = "test-token".to_string();
        let state = AppState::new(backend, token.clone());
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router(state)).await.ok();
        });
        Self {
            base: format!("http://{addr}"),
            token,
            session,
            client: reqwest::Client::new(),
        }
    }

    fn url(&self, path: &str) -> String {
        let join = if path.contains('?') { '&' } else { '?' };
        format!("{}{path}{join}t={}", self.base, self.token)
    }

    async fn get(&self, path: &str) -> reqwest::Response {
        self.client.get(self.url(path)).send().await.unwrap()
    }

    async fn post(&self, path: &str, body: Value) -> reqwest::Response {
        self.client
            .post(self.url(path))
            .json(&body)
            .send()
            .await
            .unwrap()
    }

    /// Read the opening frames of an SSE stream, which never ends on its own.
    async fn stream_text(&self, path: &str) -> String {
        let mut response = self.get(path).await;
        let mut body = String::new();
        while let Ok(Some(chunk)) =
            tokio::time::timeout(std::time::Duration::from_millis(500), response.chunk())
                .await
                .unwrap_or(Ok(None))
        {
            body.push_str(&String::from_utf8_lossy(&chunk));
            if body.contains("studio.stream.hello") {
                break;
            }
        }
        body
    }

    /// Open a session and return its id.
    async fn open_session(&self) -> String {
        let info: SessionInfoOwned = self.post("/api/sessions", json!({})).await.json().await.unwrap();
        info.id
    }
}

#[derive(serde::Deserialize)]
struct SessionInfoOwned {
    id: String,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn the_api_is_closed_without_the_startup_token() {
    let studio = Studio::start().await;
    let response = studio
        .client
        .get(format!("{}/api/app", studio.base))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 401, "loopback alone must not be enough");

    let wrong = studio
        .client
        .get(format!("{}/api/app?t=guessed", studio.base))
        .send()
        .await
        .unwrap();
    assert_eq!(wrong.status(), 401);
}

#[tokio::test]
async fn the_first_load_exchanges_the_token_for_a_cookie() {
    let studio = Studio::start().await;
    let response = studio.get("/").await;
    assert!(response.status().is_success());
    let cookie = response
        .headers()
        .get("set-cookie")
        .expect("the shell must hand out a cookie")
        .to_str()
        .unwrap()
        .to_string();
    assert!(cookie.contains("test-token"));
    // Without it, an event stream opened by EventSource could not authenticate.
    assert!(cookie.contains("HttpOnly"));
    assert!(cookie.contains("SameSite=Strict"));
}

#[tokio::test]
async fn a_cookie_alone_authenticates_later_requests() {
    let studio = Studio::start().await;
    let response = studio
        .client
        .get(format!("{}/api/app", studio.base))
        .header("Cookie", format!("eventage_studio_token={}", studio.token))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
}

#[tokio::test]
async fn prompts_reach_the_session() {
    let studio = Studio::start().await;
    let id = studio.open_session().await;

    let response = studio
        .post(
            &format!("/api/sessions/{id}/prompt"),
            json!({ "blocks": [{ "type": "text", "text": "do the thing" }] }),
        )
        .await;
    assert_eq!(response.status(), 202);
    assert_eq!(
        studio.session.prompts.lock().await.as_slice(),
        &["do the thing".to_string()]
    );
}

#[tokio::test]
async fn an_empty_prompt_is_refused_rather_than_sent() {
    let studio = Studio::start().await;
    let id = studio.open_session().await;
    let response = studio
        .post(&format!("/api/sessions/{id}/prompt"), json!({ "blocks": [] }))
        .await;
    assert_eq!(response.status(), 400);
    assert!(studio.session.prompts.lock().await.is_empty());
}

#[tokio::test]
async fn acting_on_an_unknown_session_is_a_clean_404() {
    let studio = Studio::start().await;
    for path in ["/api/sessions/nope/events", "/api/sessions/nope/stream"] {
        assert_eq!(studio.get(path).await.status(), 404, "{path}");
    }
    assert_eq!(
        studio
            .post("/api/sessions/nope/interrupt", json!({}))
            .await
            .status(),
        404
    );
}

#[tokio::test]
async fn a_rejected_mode_surfaces_the_reason_to_the_user() {
    let studio = Studio::start().await;
    let id = studio.open_session().await;

    assert_eq!(
        studio
            .post(&format!("/api/sessions/{id}/mode"), json!({ "mode": "ask" }))
            .await
            .status(),
        204
    );

    let response = studio
        .post(
            &format!("/api/sessions/{id}/mode"),
            json!({ "mode": "nonsense" }),
        )
        .await;
    assert_eq!(response.status(), 400);
    let body: Value = response.json().await.unwrap();
    assert!(
        body["error"].as_str().unwrap().contains("nonsense"),
        "the message must name what was wrong: {body}"
    );
}

#[tokio::test]
async fn permission_answers_are_routed_to_the_session() {
    let studio = Studio::start().await;
    let id = studio.open_session().await;
    let response = studio
        .post(
            &format!("/api/sessions/{id}/permission"),
            json!({ "request_id": "r1", "approve": false, "reason": "no" }),
        )
        .await;
    assert_eq!(response.status(), 204);
    assert_eq!(
        studio.session.decisions.lock().await.as_slice(),
        &[("r1".to_string(), false)]
    );
}

#[tokio::test]
async fn the_stream_states_which_numbering_it_is_serving() {
    // Sequence numbers are per feed, and a feed is rebuilt on restart. A
    // client that resumes with `?after=` from an older numbering is asking
    // for a different slice than it thinks; this is how it finds out.
    let studio = Studio::start().await;
    let id = studio.open_session().await;
    let body = studio.stream_text(&format!("/api/sessions/{id}/stream?after=0")).await;

    let hello = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|json| serde_json::from_str::<Value>(json).ok())
        .find(|e| e["kind"] == "studio.stream.hello")
        .expect("the stream should open with its generation");
    assert!(
        hello["payload"]["generation"].as_str().is_some_and(|g| !g.is_empty()),
        "{hello}"
    );
}

#[tokio::test]
async fn a_session_can_be_branched() {
    let studio = Studio::start().await;
    let id = studio.open_session().await;
    let response = studio
        .post(&format!("/api/sessions/{id}/branch"), json!({ "from_seq": 3 }))
        .await;
    assert!(response.status().is_success(), "got {}", response.status());
    let info: Value = response.json().await.unwrap();
    assert!(info["id"].is_string());
}

#[tokio::test]
async fn branching_before_anything_happened_is_refused() {
    let studio = Studio::start().await;
    let id = studio.open_session().await;
    let response = studio
        .post(&format!("/api/sessions/{id}/branch"), json!({ "from_seq": 0 }))
        .await;
    assert_eq!(response.status(), 400);
}

#[tokio::test]
async fn a_summary_can_be_replaced_through_the_api() {
    let studio = Studio::start().await;
    let id = studio.open_session().await;
    let response = studio
        .post(
            &format!("/api/sessions/{id}/summary"),
            json!({ "summary": "the corrected version", "summarized_count": 30 }),
        )
        .await;
    assert_eq!(response.status(), 204);
    assert_eq!(
        studio.session.summaries.lock().await.as_slice(),
        &["the corrected version".to_string()]
    );
}

#[tokio::test]
async fn an_empty_replacement_is_refused() {
    // Saving a blank summary would drop the compacted history silently, which
    // is the opposite of what this feature is for.
    let studio = Studio::start().await;
    let id = studio.open_session().await;
    let response = studio
        .post(
            &format!("/api/sessions/{id}/summary"),
            json!({ "summary": "   ", "summarized_count": 30 }),
        )
        .await;
    assert_eq!(response.status(), 400);
    assert!(studio.session.summaries.lock().await.is_empty());
}

#[tokio::test]
async fn interrupting_reaches_the_session() {
    let studio = Studio::start().await;
    let id = studio.open_session().await;
    studio
        .post(&format!("/api/sessions/{id}/interrupt"), json!({}))
        .await;
    assert!(*studio.session.interrupted.lock().await);
}

#[tokio::test]
async fn events_can_be_fetched_from_a_sequence_number() {
    let studio = Studio::start().await;
    let id = studio.open_session().await;
    for i in 0..5 {
        studio
            .session
            .feed
            .push(StudioEvent::studio("test.kind", json!({ "n": i })));
    }

    let all: Vec<StudioEvent> = studio
        .get(&format!("/api/sessions/{id}/events"))
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(all.len(), 5);
    assert_eq!(all[0].seq, 1);

    // What a reconnecting client asks for.
    let rest: Vec<StudioEvent> = studio
        .get(&format!("/api/sessions/{id}/events?after=3"))
        .await
        .json()
        .await
        .unwrap();
    assert_eq!(rest.len(), 2);
    assert_eq!(rest[0].seq, 4);
}

#[tokio::test]
async fn the_stream_delivers_backlog_then_live_events_exactly_once() {
    use futures_util::StreamExt;

    let studio = Studio::start().await;
    let id = studio.open_session().await;

    // Two events exist before anyone connects.
    for i in 0..2 {
        studio
            .session
            .feed
            .push(StudioEvent::studio("before", json!({ "n": i })));
    }

    let response = studio.get(&format!("/api/sessions/{id}/stream")).await;
    assert!(response.status().is_success());
    let mut body = response.bytes_stream();

    // And two more arrive afterwards.
    let feed = Arc::clone(&studio.session.feed);
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
        for i in 0..2 {
            feed.push(StudioEvent::studio("after", json!({ "n": i })));
        }
    });

    let mut seqs = Vec::new();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut buffer = String::new();
    while seqs.len() < 4 && tokio::time::Instant::now() < deadline {
        let Ok(Some(Ok(chunk))) =
            tokio::time::timeout(std::time::Duration::from_secs(2), body.next()).await
        else {
            break;
        };
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(at) = buffer.find("\n\n") {
            let frame: String = buffer.drain(..at + 2).collect();
            for line in frame.lines() {
                if let Some(data) = line.strip_prefix("data:") {
                    if let Ok(event) = serde_json::from_str::<StudioEvent>(data.trim()) {
                        // The opening frame names the numbering rather than
                        // occupying a place in it, so it carries seq 0.
                        if event.seq > 0 {
                            seqs.push(event.seq);
                        }
                    }
                }
            }
        }
    }

    assert_eq!(
        seqs,
        vec![1, 2, 3, 4],
        "a subscriber must see the backlog and then live events, in order and without duplicates"
    );
}

#[tokio::test]
async fn a_resumed_stream_does_not_replay_what_the_client_already_has() {
    use futures_util::StreamExt;

    let studio = Studio::start().await;
    let id = studio.open_session().await;
    for i in 0..3 {
        studio
            .session
            .feed
            .push(StudioEvent::studio("old", json!({ "n": i })));
    }

    let response = studio
        .get(&format!("/api/sessions/{id}/stream?after=2"))
        .await;
    let mut body = response.bytes_stream();

    // The stream opens by naming its numbering, then serves the backlog.
    let mut text = String::new();
    while !text.contains("\"seq\":3") {
        let chunk = tokio::time::timeout(std::time::Duration::from_secs(3), body.next())
            .await
            .expect("the backlog should arrive promptly")
            .unwrap()
            .unwrap();
        text.push_str(&String::from_utf8_lossy(&chunk));
    }
    assert!(text.contains("studio.stream.hello"), "got: {text}");
    assert!(!text.contains("\"seq\":1"), "got: {text}");
    assert!(!text.contains("\"seq\":2"), "got: {text}");
}

#[tokio::test]
async fn a_deep_link_serves_the_shell_without_losing_the_token() {
    let studio = Studio::start().await;
    // A hard refresh on a client-side route must serve the app, not redirect:
    // a redirect to "/" would strip the ?t= token and land on a 401.
    let response = studio
        .client
        .get(studio.url("/some/client/route"))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success(), "got {}", response.status());
    assert!(
        response.headers().contains_key("set-cookie"),
        "the shell must still hand out the cookie on a deep link"
    );
    assert!(response.text().await.unwrap().contains("<html"));
}

#[tokio::test]
async fn the_workspace_picker_lists_directories_only() {
    let studio = Studio::start().await;
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("src")).unwrap();
    std::fs::write(dir.path().join("a-file.txt"), "x").unwrap();
    std::fs::create_dir(dir.path().join(".hidden")).unwrap();

    let listing: Value = studio
        .get(&format!(
            "/api/fs/list?path={}",
            urlencode(&dir.path().display().to_string())
        ))
        .await
        .json()
        .await
        .unwrap();

    let names: Vec<&str> = listing["dirs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["src"], "files and dotfiles must not be listed");
    assert!(listing["parent"].is_string());
}

#[tokio::test]
async fn a_missing_directory_is_reported_not_panicked_on() {
    let studio = Studio::start().await;
    let response = studio.get("/api/fs/list?path=/definitely/not/here").await;
    assert_eq!(response.status(), 400);
}

fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                (b as char).to_string()
            }
            other => format!("%{other:02X}"),
        })
        .collect()
}
