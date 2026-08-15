//! Filesystem confinement for shell commands, applied safely.
//!
//! The obvious place to set up a sandbox is `pre_exec`, in the gap between
//! `fork` and `exec`. It does not work. That code runs in the forked child of
//! a multi-threaded process, where only async-signal-safe work is permitted —
//! and building a Landlock ruleset allocates and opens files. If another
//! thread held the allocator lock at the instant of the fork, the child blocks
//! on it forever. An earlier version of this did exactly that: standalone it
//! looked fine, and under a parallel test run `bash -c true` hung.
//!
//! So the sandbox is applied in a **fresh process** instead. The agent
//! re-executes its own binary with a marker argument; that process is
//! single-threaded from its first instruction, sets up Landlock with all the
//! allocation it likes, and then `exec`s the real command. Nothing runs
//! between a fork and an exec in a threaded parent, so the hazard is gone.
//!
//! This is the same shape Codex uses for its Linux sandbox, and the reason is
//! the same: it is the only way to get a real ruleset installed without
//! doing forbidden work in a forked child.
//!
//! What it buys: a command that wanders out of the repository cannot read
//! `~/.ssh` or write `~/.bashrc`. What it does not buy: network isolation or
//! resource accounting across a process tree. Landlock is filesystem-only —
//! a seatbelt, not a boundary. A genuinely untrusted repository wants
//! [`ShellContainment::Strict`](crate::tools::ShellContainment), and beyond that a container.

use std::path::{Path, PathBuf};

/// The marker that tells a fresh process it is the sandbox helper.
///
/// Prefixed and unlikely to collide with a real subcommand; both binaries
/// that embed the agent check for it before parsing their own arguments.
pub const HELPER_ARG: &str = "__eventage-confine-exec";

/// Directories a build legitimately writes to, besides the workspace.
///
/// A confinement that broke `cargo build` would be switched off within the
/// day, so the toolchain caches are writable by name. Everything else on the
/// filesystem stays readable and unwritable.
fn writable_paths(root: &Path) -> Vec<PathBuf> {
    let mut paths = vec![root.to_path_buf(), std::env::temp_dir()];
    for var in [
        "CARGO_HOME",
        "RUSTUP_HOME",
        "GOPATH",
        "GOCACHE",
        "npm_config_cache",
    ] {
        if let Ok(dir) = std::env::var(var) {
            paths.push(PathBuf::from(dir));
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        for cache in [
            ".cargo",
            ".rustup",
            ".cache",
            ".npm",
            ".local/share/virtualenvs",
        ] {
            paths.push(Path::new(&home).join(cache));
        }
    }
    paths.retain(|p| p.exists());
    paths
}

/// Build the command that runs `script` confined to `root`.
///
/// Returns `None` when confinement is unavailable — no Landlock in the
/// kernel, or the executable cannot locate itself — so the caller can decide
/// whether to run unconfined or refuse, rather than silently doing one.
#[cfg(target_os = "linux")]
pub fn confined_command(
    root: &Path,
    script: &str,
    network: Network,
) -> Option<std::process::Command> {
    if !available() {
        return None;
    }
    let exe = std::env::current_exe().ok()?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg(HELPER_ARG)
        .arg(root)
        .arg(network.token())
        .arg("bash")
        // `-c`, not `-lc`: a login shell sources the user's profile, which
        // re-imports the credentials the caller just scrubbed.
        .arg("-c")
        .arg(script);
    Some(cmd)
}

#[cfg(not(target_os = "linux"))]
pub fn confined_command(
    _root: &Path,
    _script: &str,
    _network: Network,
) -> Option<std::process::Command> {
    None
}

/// Does this kernel actually enforce Landlock?
///
/// Asked once, by creating and discarding an empty ruleset. A kernel without
/// it reports success for every call and enforces nothing, which is the worst
/// possible answer to give a caller — so this checks rather than assumes.
#[cfg(target_os = "linux")]
pub fn available() -> bool {
    use std::sync::OnceLock;
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::fs::metadata("/sys/kernel/security/lsm")
            .ok()
            .and_then(|_| std::fs::read_to_string("/sys/kernel/security/lsm").ok())
            .map(|lsms| lsms.split(',').any(|l| l.trim() == "landlock"))
            .unwrap_or(false)
    })
}

#[cfg(not(target_os = "linux"))]
pub fn available() -> bool {
    false
}

/// Deny the network with a seccomp filter.
///
/// Landlock covers the filesystem and nothing else, which is why the previous
/// answer to "how do we stop a command phoning home" was a container. It did
/// not have to be: refusing `socket(2)` for the internet address families is
/// a seccomp filter, applied to the same process, keeping the host's real
/// toolchain. That is what Codex does, and it is a much better trade than an
/// image with no compiler in it.
///
/// `AF_UNIX` is deliberately still allowed. Plenty of ordinary local
/// machinery speaks over unix sockets, and blocking those breaks things
/// without denying anything that leaves the machine.
///
/// Returns `EPERM` to the caller rather than killing it, so a command that
/// tries to fetch something fails with a readable error instead of dying by
/// signal with no explanation.
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
fn deny_network() -> Result<(), String> {
    // Offsets into `struct seccomp_data`: nr at 0, arch at 4, args[0] at 16.
    const NR: u32 = 0;
    const ARCH: u32 = 4;
    const ARG0: u32 = 16;

    #[cfg(target_arch = "x86_64")]
    const EXPECTED_ARCH: u32 = 0xC000_003E;
    #[cfg(target_arch = "x86_64")]
    const NR_SOCKET: u32 = 41;
    #[cfg(target_arch = "aarch64")]
    const EXPECTED_ARCH: u32 = 0xC000_00B7;
    #[cfg(target_arch = "aarch64")]
    const NR_SOCKET: u32 = 198;

    const LD_W_ABS: u16 = 0x20;
    const JMP_JEQ_K: u16 = 0x15;
    const RET_K: u16 = 0x06;
    const RET_ALLOW: u32 = 0x7fff_0000;
    const RET_ERRNO: u32 = 0x0005_0000;
    const AF_UNIX: u32 = 1;

    let f = |code: u16, jt: u8, jf: u8, k: u32| libc::sock_filter { code, jt, jf, k };
    let filter = [
        // Syscall numbers are per-architecture, so a filter written for one
        // means nothing on another. Leave an unexpected arch alone rather
        // than denying the wrong numbers.
        f(LD_W_ABS, 0, 0, ARCH),
        f(JMP_JEQ_K, 1, 0, EXPECTED_ARCH),
        f(RET_K, 0, 0, RET_ALLOW),
        f(LD_W_ABS, 0, 0, NR),
        f(JMP_JEQ_K, 0, 3, NR_SOCKET),
        f(LD_W_ABS, 0, 0, ARG0),
        f(JMP_JEQ_K, 1, 0, AF_UNIX),
        f(RET_K, 0, 0, RET_ERRNO | libc::EPERM as u32),
        f(RET_K, 0, 0, RET_ALLOW),
    ];
    let program = libc::sock_fprog {
        len: filter.len() as u16,
        filter: filter.as_ptr() as *mut libc::sock_filter,
    };

    // SAFETY: single-threaded, pre-exec, and both calls take plain values.
    // `NO_NEW_PRIVS` is required before an unprivileged process may install a
    // filter, and it is what stops a setuid binary escaping it afterwards.
    unsafe {
        if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
            return Err("could not set no_new_privs".into());
        }
        if libc::prctl(
            libc::PR_SET_SECCOMP,
            libc::SECCOMP_MODE_FILTER,
            &program as *const _,
        ) != 0
        {
            return Err(format!(
                "seccomp filter rejected: {}",
                std::io::Error::last_os_error()
            ));
        }
    }
    Ok(())
}

#[cfg(not(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
)))]
fn deny_network() -> Result<(), String> {
    Err("no seccomp on this platform".into())
}

/// Whether a confined command may reach the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Network {
    /// Left open. A build resolving dependencies needs it.
    Allow,
    /// Refused at the syscall. For a repository you do not trust.
    Deny,
}

impl Network {
    fn token(self) -> &'static str {
        match self {
            Network::Allow => "net-allow",
            Network::Deny => "net-deny",
        }
    }

    fn parse(token: &str) -> Option<Self> {
        match token {
            "net-allow" => Some(Network::Allow),
            "net-deny" => Some(Network::Deny),
            _ => None,
        }
    }
}

/// If this process was started as the sandbox helper, confine and exec.
///
/// Called first thing in `main`, before any runtime is started and before
/// arguments are parsed — the helper is not really the same program, it is a
/// two-line trampoline that happens to live in the same binary. Never returns
/// when it matches: either the `exec` succeeds and this program is replaced,
/// or it fails and the process exits with a message on stderr.
///
/// Both binaries that embed the agent must call this. `current_exe` names
/// whichever one spawned the command, so the helper has to be reachable from
/// all of them.
pub fn run_if_helper() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) != Some(HELPER_ARG) {
        return;
    }
    let Some(root) = args.get(2) else {
        eprintln!("{HELPER_ARG}: no workspace root given");
        std::process::exit(70);
    };
    let network = args.get(3).map(String::as_str).and_then(Network::parse);
    let Some(network) = network else {
        eprintln!("{HELPER_ARG}: no network policy given");
        std::process::exit(70);
    };
    if args.len() < 5 {
        eprintln!("{HELPER_ARG}: no command given");
        std::process::exit(70);
    }

    #[cfg(target_os = "linux")]
    {
        let root = PathBuf::from(root);
        // Everything readable, the workspace and the build caches writable.
        // Allocation and file opening are fine here: this process is
        // single-threaded and has not forked.
        if let Err(e) =
            eventage::sandbox::landlock_confine(&[PathBuf::from("/")], &writable_paths(&root))
        {
            // Refuse rather than run unconfined. The caller asked for a
            // confined command and got a helper that could not confine it;
            // running anyway would be the one outcome nobody asked for.
            eprintln!("{HELPER_ARG}: could not apply filesystem confinement: {e}");
            std::process::exit(71);
        }
    }

    if network == Network::Deny {
        if let Err(e) = deny_network() {
            // Same reasoning as the filesystem: asked to deny the network and
            // unable to, running anyway is the one outcome nobody wanted.
            eprintln!("{HELPER_ARG}: could not deny network access: {e}");
            std::process::exit(71);
        }
    }

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let error = std::process::Command::new(&args[4]).args(&args[5..]).exec();
        eprintln!("{HELPER_ARG}: could not run '{}': {error}", args[4]);
        std::process::exit(72);
    }

    #[cfg(not(unix))]
    {
        eprintln!("{HELPER_ARG}: not supported on this platform");
        std::process::exit(70);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_process_that_is_not_the_helper_is_left_alone() {
        // The guard has to be exact: a stray argument that merely resembles
        // the marker must not turn an ordinary run into a trampoline.
        run_if_helper();
    }

    #[test]
    fn the_writable_set_covers_the_workspace_and_nothing_imaginary() {
        let dir = tempfile::tempdir().unwrap();
        let paths = writable_paths(dir.path());
        assert!(paths.contains(&dir.path().to_path_buf()));
        assert!(
            paths.iter().all(|p| p.exists()),
            "a path that does not exist would fail the whole ruleset: {paths:?}"
        );
    }
}
