//! Cancelling a turn, and running one at a time.
//!
//! Both were reported as "the status is right, the behaviour is not":
//! cancellation set a flag and waited for the cycle to finish, and two
//! prompts arriving together ran concurrently against one agent.

use eventage_code::acp::wire::ContentBlock;
use eventage_code::agent::CodingSession;
use eventage_code::config::{ModelConfig, Provider, SessionConfig};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A session pointed at an endpoint that accepts the connection and then
/// never answers — so the turn is genuinely in flight, not merely slow.
async fn stalled_session(state: &tempfile::TempDir) -> (Arc<CodingSession>, tempfile::TempDir) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            // Accept and hold: a request that will never be answered.
            let Ok((socket, _)) = listener.accept().await else {
                return;
            };
            std::mem::forget(socket);
        }
    });

    // SAFETY: set before the session is built; these tests are in their own
    // binary and run one at a time against the environment.
    unsafe {
        std::env::set_var("XDG_DATA_HOME", state.path());
        std::env::set_var("OPENAI_BASE_URL", format!("http://{addr}/v1"));
        std::env::set_var("OPENAI_API_KEY", "test");
    }

    let workspace = tempfile::tempdir().unwrap();
    let mut config = SessionConfig::new(
        workspace.path().to_str().unwrap(),
        ModelConfig::from_env(Some("test-model".into())),
    );
    config.model.provider = Provider::OpenAiChat;
    let session =
        CodingSession::create("8a1c7d20-6e52-4b3f-ac88-1d2e3f4a5b6c".into(), config, None)
            .await
            .unwrap();
    (Arc::new(session), workspace)
}

#[tokio::test(flavor = "multi_thread")]
async fn cancelling_a_turn_stops_it_rather_than_labelling_it() {
    let state = tempfile::tempdir().unwrap();
    let (session, _ws) = stalled_session(&state).await;

    session
        .submit_prompt(&[ContentBlock::text("go")])
        .await
        .unwrap();

    let running = {
        let session = Arc::clone(&session);
        tokio::spawn(async move { session.run_cycle().await })
    };

    // Let the request actually leave.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let at = Instant::now();
    session.cancel();

    let outcome = tokio::time::timeout(Duration::from_secs(5), running)
        .await
        .expect("cancel must return promptly, not wait for the model")
        .unwrap();
    assert!(outcome.is_ok(), "{outcome:?}");
    assert!(
        at.elapsed() < Duration::from_secs(3),
        "took {:?} — the request was waited on rather than dropped",
        at.elapsed()
    );
    assert!(session.was_cancelled());
}

#[tokio::test(flavor = "multi_thread")]
async fn two_prompts_do_not_run_against_one_session_at_once() {
    let state = tempfile::tempdir().unwrap();
    let (session, _ws) = stalled_session(&state).await;
    session
        .submit_prompt(&[ContentBlock::text("first")])
        .await
        .unwrap();

    let first = {
        let session = Arc::clone(&session);
        tokio::spawn(async move { session.run_cycle().await })
    };
    tokio::time::sleep(Duration::from_millis(200)).await;

    // A second turn must wait for the gate rather than interleave.
    let second = {
        let session = Arc::clone(&session);
        tokio::spawn(async move { session.run_cycle().await })
    };
    let overlapped = tokio::time::timeout(Duration::from_millis(500), second).await;
    assert!(
        overlapped.is_err(),
        "the second turn ran while the first was still going"
    );

    session.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), first).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_second_prompt_cannot_join_the_running_turn() {
    // `submit_prompt` is ungated: it publishes the user message and clears
    // the cancellation flag, and only `run_cycle` takes the gate. A caller
    // that did the two separately — the ACP server did — left a window where
    // a pipelined `session/prompt` published a second user message into the
    // conversation already in flight and reset the cancellation of the turn
    // the user was in the middle of stopping.
    let state = tempfile::tempdir().unwrap();
    let (session, _ws) = stalled_session(&state).await;

    let first = {
        let session = Arc::clone(&session);
        tokio::spawn(async move { session.prompt_turn(&[ContentBlock::text("first")]).await })
    };
    tokio::time::sleep(Duration::from_millis(200)).await;

    assert!(session.is_busy());

    let refused = session.prompt_turn(&[ContentBlock::text("second")]).await;
    assert!(
        refused.is_err(),
        "the second prompt joined the running turn"
    );

    // And it published nothing: one user message, not two.
    let users = session
        .bus
        .log()
        .await
        .iter()
        .filter(|e| e.kind == eventage::event::kinds::USER_MESSAGE)
        .count();
    assert_eq!(users, 1);

    // The refusal must not have cleared the running turn's cancellation.
    session.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), first).await;
    assert!(session.was_cancelled());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_rewind_is_refused_while_a_turn_is_running() {
    // Rewinding rolls back the event DAG *and* rewrites the working tree.
    // Underneath a live turn that means the model's next tool call lands on
    // files that moved and its events attach to a branch sealed while it was
    // thinking. ACP exposes `session/rewind` directly, so nothing but this
    // stopped it.
    let state = tempfile::tempdir().unwrap();
    let (session, _ws) = stalled_session(&state).await;

    let running = {
        let session = Arc::clone(&session);
        tokio::spawn(async move { session.prompt_turn(&[ContentBlock::text("go")]).await })
    };
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(session.is_busy());

    let refused = session.rewind(1).await;
    assert!(refused.is_err(), "a rewind ran underneath a live turn");
    assert!(
        refused
            .unwrap_err()
            .to_string()
            .contains("stop the current turn"),
        "the refusal should say what to do about it"
    );

    session.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), running).await;
    // And it works again once the turn is over.
    assert!(session.rewind(1).await.is_ok());
}
