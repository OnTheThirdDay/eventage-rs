//! The sandbox trampoline, as a binary of its own.
//!
//! A Landlock ruleset cannot be built between `fork` and `exec` — that code
//! runs in the forked child of a multi-threaded process, where allocating is
//! not allowed — so it is built in a fresh process instead. That process used
//! to be the agent's own binary re-executed with a marker argument, which
//! worked for the two binaries that remember to call `run_if_helper` first
//! and failed silently for anything else. A *test* binary re-executed that
//! way hands the marker to libtest, which treats it as a filter, runs no
//! tests, and exits 0 — so the caller sees a successful command that produced
//! no output. That is the worst possible failure for a sandbox: it does not
//! look like one, and it is why `git` could not be confined until now.
//!
//! Being a separate binary removes the question. It does nothing but confine
//! and exec, so there is no argument parsing to collide with and nothing for
//! an embedder to remember.

fn main() {
    eventage_code::shell_sandbox::run_if_helper();
    eprintln!(
        "{}: this program is the sandbox trampoline and is not run directly",
        eventage_code::shell_sandbox::HELPER_ARG
    );
    std::process::exit(64);
}
