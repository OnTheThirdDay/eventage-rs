//! What the app says about credentials must survive the credential scrub.
//!
//! Reported from a real session: with `QWEN_API_KEY` and `OPENAI_BASE_URL`
//! both exported, Studio's banner said *"No API key found"* and named the
//! gateway as though it were an unconfigured local fallback. The cause is the
//! same one that erased the endpoint — something asking the environment a
//! question after startup has deliberately erased the answer.

use eventage_code::config::{ModelConfig, Provider};

const MARKER: &str = "EVENTAGE_TEST_CREDENTIALS_CHILD";

#[test]
fn a_configured_key_is_still_reported_after_the_scrub() {
    if std::env::var_os(MARKER).is_some() {
        return child();
    }
    let exe = std::env::current_exe().unwrap();
    let output = std::process::Command::new(exe)
        .arg("--exact")
        .arg("a_configured_key_is_still_reported_after_the_scrub")
        .arg("--nocapture")
        .env(MARKER, "1")
        // Exactly the invocation that was reported.
        .env("QWEN_API_KEY", "sk-qwen-real")
        .env(
            "OPENAI_BASE_URL",
            "https://ws-example.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        )
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("ANTHROPIC_AUTH_TOKEN")
        .env_remove("OPENAI_API_KEY")
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
    let model = ModelConfig::from_env(Some("qwen3.7-max-2026-05-20".into()));
    assert_eq!(model.provider, Provider::Qwen);
    assert!(
        model.credentialed,
        "the key was not recognised at resolution"
    );

    let _ = eventage_code::secrets::capture_and_scrub();

    // The two facts the banner is derived from, after the scrub.
    assert!(
        model.credentialed,
        "the app would tell a user with a working key that they have none"
    );
    assert_eq!(
        model.base_url(),
        "https://ws-example.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        "the configured gateway was replaced by a default"
    );

    // And a genuinely unconfigured process still says so, or the banner would
    // be useless to the person it exists for.
    unsafe { std::env::remove_var("QWEN_API_KEY") };
    let bare = ModelConfig::from_env(None);
    assert!(!bare.credentialed);
}
