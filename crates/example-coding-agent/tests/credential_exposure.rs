//! The API key must not be readable out of this process's environment.
//!
//! `scrubbed_env` keeps it out of every command the agent runs, which was
//! never the whole problem: a process's environment is a file
//! (`/proc/<pid>/environ`) that any process of the same user can open. A
//! confined shell command, denied the credential in its own environment,
//! could read the parent's and have it anyway — and `/proc` has to stay
//! readable under Landlock for ordinary tooling to work.
//!
//! The distinction this test exists to hold: `remove_var` is *not* sufficient.
//! It unlinks the pointer from the `environ` array while the bytes stay in
//! the region the kernel serves from `/proc`. The check below only means
//! anything for a variable that was inherited at `exec`, so the test
//! re-executes itself with one set.

use eventage_code::secrets;

/// Set on the child, so the credential is in the block the process was
/// started with rather than one added at runtime.
const MARKER: &str = "EVENTAGE_TEST_CREDENTIAL_CHILD";
const SECRET: &str = "sk-ant-should-not-survive";

#[test]
fn the_scrub_takes_credentials_out_of_the_process_environment() {
    if std::env::var_os(MARKER).is_some() {
        return child();
    }

    let exe = std::env::current_exe().expect("the test binary knows where it is");
    let output = std::process::Command::new(exe)
        .arg("--exact")
        .arg("the_scrub_takes_credentials_out_of_the_process_environment")
        .arg("--nocapture")
        .env(MARKER, "1")
        .env("ANTHROPIC_API_KEY", SECRET)
        .env("SOME_VENDOR_SECRET", "also-gone")
        .env("EVENTAGE_ORDINARY_VAR", "kept")
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
    let held = secrets::capture_and_scrub();
    assert!(held.iter().any(|n| n == "ANTHROPIC_API_KEY"), "{held:?}");
    assert!(held.iter().any(|n| n == "SOME_VENDOR_SECRET"), "{held:?}");

    // Gone from `getenv`, so nothing inherits it by accident.
    assert!(std::env::var_os("ANTHROPIC_API_KEY").is_none());
    assert!(std::env::var_os("SOME_VENDOR_SECRET").is_none());

    // Anything not credential-shaped is left alone: the scrub must not
    // quietly break `PATH`, `HOME`, or a build's own configuration.
    assert_eq!(
        std::env::var("EVENTAGE_ORDINARY_VAR").as_deref(),
        Ok("kept")
    );
    assert!(std::env::var_os("PATH").is_some());

    // Still reachable in memory for the callers that genuinely need one.
    assert_eq!(secrets::get("ANTHROPIC_API_KEY").as_deref(), Some(SECRET));

    // The file a confined command would actually open. This is the assertion
    // that `remove_var` alone does not satisfy.
    #[cfg(target_os = "linux")]
    {
        let environ = std::fs::read("/proc/self/environ").expect("procfs is mounted");
        let environ = String::from_utf8_lossy(&environ);
        assert!(
            !environ.contains(SECRET),
            "the key is still in /proc/self/environ — unsetting a variable does not \
             erase the bytes the kernel reads"
        );
        assert!(
            !environ.contains("also-gone"),
            "a second credential survived in /proc/self/environ"
        );
        assert!(
            environ.contains("EVENTAGE_ORDINARY_VAR=kept"),
            "the environment block was destroyed rather than selectively wiped"
        );
    }
}
