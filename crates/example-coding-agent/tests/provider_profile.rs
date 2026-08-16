//! The model configuration is a profile, resolved once.
//!
//! Endpoint, credential, model, authentication mode and headers are captured
//! together and never re-derived. They used not to be: `base_url()` re-read
//! `OPENAI_BASE_URL` at provider-construction time, which happens *after*
//! startup takes credential-shaped variables out of the environment — and
//! `OPENAI_*` is credential-shaped by design, because it names where
//! credentials go.
//!
//! The result was the exact failure the scrub existed to prevent. A session
//! pointed at a local vLLM or a private gateway silently fell back to a
//! hard-coded default, and for the Responses provider that default is
//! `api.openai.com`.

use eventage_code::config::ModelConfig;

/// Set on the child so the variables are inherited rather than added at
/// runtime, matching how an operator actually configures one of these.
const MARKER: &str = "EVENTAGE_TEST_PROFILE_CHILD";

#[test]
fn a_configured_endpoint_survives_the_credential_scrub() {
    if std::env::var_os(MARKER).is_some() {
        return child();
    }
    let exe = std::env::current_exe().unwrap();
    let output = std::process::Command::new(exe)
        .arg("--exact")
        .arg("a_configured_endpoint_survives_the_credential_scrub")
        .arg("--nocapture")
        .env(MARKER, "1")
        .env("OPENAI_API_KEY", "sk-local-only")
        .env("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("ANTHROPIC_AUTH_TOKEN")
        .env_remove("QWEN_API_KEY")
        .output()
        .expect("could not re-run the test binary");
    assert!(
        output.status.success(),
        "child failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn child() {
    let model = ModelConfig::from_env(Some("local-model".into()));
    let configured = model.base_url();
    assert_eq!(configured, "http://127.0.0.1:8000/v1");

    // What startup does before any provider is built.
    let _ = eventage_code::secrets::capture_and_scrub();

    assert_eq!(
        model.base_url(),
        configured,
        "the endpoint was re-derived from an environment that no longer has it"
    );
    assert!(
        !model.base_url().contains("api.openai.com"),
        "a local endpoint fell back to OpenAI's, sending the key and the prompt \
         somewhere the operator never chose"
    );
    // The credential is held, not lost — the profile is still usable.
    assert_eq!(model.api_key, "sk-local-only");
}
