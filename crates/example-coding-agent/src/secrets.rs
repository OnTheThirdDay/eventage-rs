//! Credentials, moved out of the process environment and into memory.
//!
//! `scrubbed_env` keeps the API key out of every command the agent runs, and
//! that is worth doing — but it never closed the hole it was aimed at. The
//! key was still in *this* process's environment, and a process's environment
//! is a file: `/proc/<pid>/environ`, readable by anything running as the same
//! user. A confined shell command, denied the credential in its own
//! environment, could read the parent's and have it anyway. Landlock does not
//! help, because `/proc` has to stay readable for ordinary tooling to work,
//! and a container is a heavy answer to a problem this direct.
//!
//! So the credentials are read once at startup, kept here, and **removed from
//! the environment**. What remains in memory is reachable only through
//! `/proc/<pid>/mem` or `ptrace`, which the kernel gates on `ptrace_scope`
//! and which is a materially higher bar than opening a file.
//!
//! Deliberately not a vault. It does not encrypt, lock pages against swap, or
//! zero on drop. It moves a secret from something any process can read to
//! something that needs a debugger, which is the whole of the improvement
//! being claimed.

use std::collections::BTreeMap;
use std::sync::OnceLock;

static CAPTURED: OnceLock<BTreeMap<String, String>> = OnceLock::new();

/// Take every credential-shaped variable out of the environment.
///
/// Call once at startup, **after** everything that reads configuration from
/// the environment has run — `ClaudeSettings::apply_env` and
/// `ModelConfig::from_env` both do, and both copy what they need into their
/// own structures first.
///
/// Returns the names it captured, for the startup log. Names only: the whole
/// point is that the values stop travelling.
pub fn capture_and_scrub() -> Vec<String> {
    let captured: BTreeMap<String, String> = std::env::vars()
        .filter(|(name, _)| crate::tools::is_credential(name))
        .collect();

    for name in captured.keys() {
        wipe(name);
        // SAFETY: called once during startup, before any thread that reads
        // the environment has been spawned.
        unsafe { std::env::remove_var(name) };
    }

    let names: Vec<String> = captured.keys().cloned().collect();
    // A second call would otherwise silently discard what it found, leaving
    // the values nowhere — worse than not calling it at all.
    let _ = CAPTURED.set(captured);
    names
}

/// Overwrite a variable's value where the kernel actually reads it.
///
/// `unsetenv` alone is not enough, and this is the part that is easy to get
/// wrong. It unlinks the pointer from the `environ` array, so `getenv` stops
/// finding the variable and children stop inheriting it — but the bytes are
/// untouched. `/proc/<pid>/environ` is not the `environ` array: the kernel
/// reads the raw region between `mm->env_start` and `mm->env_end`, which is
/// the block the process was `exec`d with. The string is still sitting there.
///
/// Verified rather than assumed. A probe that set a variable, called
/// `remove_var`, and then read `/proc/self/environ` still found the value;
/// with this wipe first, it does not.
///
/// So the value is zeroed in place before the variable is unset. `getenv`
/// returns a pointer into that same block — the classic `setproctitle`
/// territory, and the region is writable for the same reason. The name is
/// left behind, which is deliberate: names are not secrets, and truncating
/// the block would move everything after it.
fn wipe(name: &str) {
    #[cfg(unix)]
    {
        let Ok(c_name) = std::ffi::CString::new(name) else {
            return;
        };
        // SAFETY: `getenv` returns a pointer into the process's own
        // environment block, valid until the variable is changed. Called at
        // startup, and the variable is unset immediately afterwards, so
        // nothing reads through this pointer again.
        unsafe {
            let value = libc::getenv(c_name.as_ptr());
            if !value.is_null() {
                let len = libc::strlen(value);
                std::ptr::write_bytes(value, 0, len);
            }
        }
    }
    #[cfg(not(unix))]
    let _ = name;
}

/// A credential taken out of the environment at startup.
///
/// For the few places that genuinely need one: `gh` authenticates with
/// `GH_TOKEN`, and taking it away would break pull-request creation rather
/// than secure anything. Everything else should not be asking.
pub fn get(name: &str) -> Option<String> {
    match CAPTURED.get() {
        // Before the scrub — a test, or a library embedding — the environment
        // is still the source of truth.
        None => std::env::var(name).ok(),
        Some(captured) => captured.get(name).cloned(),
    }
}

/// Every captured credential, for a child process the operator chose to run.
///
/// Studio spawns its ACP agent from `--acp <command>`; that child is the
/// thing that talks to the model, so starting it with the credentials
/// stripped would simply break it. The operator named the command, which is
/// the same standard applied everywhere else here — and if the child is our
/// own binary it scrubs its own environment on the way up.
///
/// **Not** for MCP servers or tools. Those inherit the scrubbed environment
/// and take credentials through their own configuration (`env` in the server
/// spec), so a server the agent talks to never sees the model provider's key
/// by accident.
pub fn all() -> Vec<(String, String)> {
    match CAPTURED.get() {
        None => std::env::vars()
            .filter(|(name, _)| crate::tools::is_credential(name))
            .collect(),
        Some(captured) => captured
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_credential_is_readable_before_the_scrub_has_run() {
        // The library is usable without `capture_and_scrub` ever being
        // called; tests and embedders never call it.
        // SAFETY: this test binary's threads do not read this variable.
        unsafe { std::env::set_var("EVENTAGE_TEST_LOOKUP_TOKEN", "v") };
        assert_eq!(get("EVENTAGE_TEST_LOOKUP_TOKEN").as_deref(), Some("v"));
        unsafe { std::env::remove_var("EVENTAGE_TEST_LOOKUP_TOKEN") };
    }

    #[test]
    fn something_that_is_not_a_credential_is_left_where_it_is() {
        // The scrub is driven by `is_credential`, which is shape-based: the
        // next provider's variable is in no list anyone maintains.
        assert!(crate::tools::is_credential("ANTHROPIC_API_KEY"));
        assert!(crate::tools::is_credential("SOME_VENDOR_SECRET"));
        assert!(!crate::tools::is_credential("PATH"));
        assert!(!crate::tools::is_credential("HOME"));
    }
}
