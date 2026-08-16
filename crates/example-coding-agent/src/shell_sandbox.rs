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
//! What it buys depends on the [`Reads`] policy. There are two, and the line
//! between them is deliberately stark:
//!
//! * [`Reads::Everywhere`] — **writes** are confined to the workspace and the
//!   toolchain caches; reads are not confined at all. A command can still read
//!   `~/.ssh`. This is the trade Codex's Linux sandbox makes, and the reason
//!   is that toolchains install themselves in unpredictable places under
//!   `$HOME`: a read policy that guesses wrong makes `node` or `cargo` vanish.
//! * [`Reads::Workspace`] — reads narrowed to the system directories, the
//!   toolchain locations we can name, and the workspace. Your other
//!   repositories go too. This is what makes
//!   [`ShellContainment::Strict`](crate::tools::ShellContainment) worth
//!   choosing, and it is allowed to break an exotic toolchain — a repository
//!   you do not trust is one you would rather fail loudly on.
//!
//! There used to be a third, in between: read everything *except* a named list
//! of credential stores, built by granting every sibling of every secret since
//! Landlock has no deny rules. It is gone, and the reasoning is worth keeping
//! because the same idea will look attractive again.
//!
//! It was not a boundary. It enumerated `$HOME` at startup, so a directory
//! created there afterwards became unreadable for no reason a user could
//! predict; and `git push` needs `~/.ssh`, so it grew a carve-out that let
//! git read exactly the secrets the policy existed to hide. A rule shaped by
//! what happened to break is not a security property, and a middle mode
//! invites the confidence of one without the substance. Two modes that each
//! mean something are worth more than three where one is a story.
//!
//! Neither buys network isolation or resource accounting across a process
//! tree; the network is a separate seccomp filter ([`Network`]), and nothing
//! here bounds a process tree. Landlock is filesystem-only — a seatbelt, not
//! a boundary. Beyond `Strict` lies a container.
//!
//! `/proc` stays readable under both, because too much ordinary machinery
//! reads it. That used to mean a command could open `/proc/<agent pid>/environ`
//! and recover the API key that `scrubbed_env` had kept out of its own
//! environment; [`secrets::capture_and_scrub`](crate::secrets) closes that by
//! wiping the value out of the block the kernel serves. What remains is
//! `/proc` for *other* processes of the same user, which no filesystem
//! sandbox can fix — that needs a PID namespace, which means a container.

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
    // The device files every shell expects to be able to write to. Landlock
    // governs these like any other path, so without them `cmd > /dev/null`
    // fails with EACCES — which reads as the tool being broken rather than as
    // a sandbox doing its job.
    for dev in [
        "/dev/null",
        "/dev/zero",
        "/dev/full",
        "/dev/random",
        "/dev/urandom",
        "/dev/tty",
        "/dev/shm",
        "/dev/ptmx",
        "/dev/pts",
        "/dev/stdout",
        "/dev/stderr",
    ] {
        paths.push(PathBuf::from(dev));
    }
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

/// What a [`Reads::Workspace`] command may read.
///
/// Everything a compiler, linker and test runner touches that is not the
/// repository, and nothing that belongs to the person running it. The
/// omissions are the point: `$HOME` is absent except for the toolchain
/// locations named here, so `~/.ssh`, `~/.aws`, `~/.netrc`,
/// `~/.git-credentials`, the shell history and every other checkout on the
/// machine are unreadable.
///
/// `/run` is omitted too, because `/run/user/$UID` holds the ssh-agent and
/// keyring sockets. `/proc` is *not* omitted — too much ordinary machinery
/// reads it — and the module docs say what that costs.
fn readable_paths(root: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = [
        "/bin", "/sbin", "/usr", "/lib", "/lib32", "/lib64", "/libx32", "/etc", "/opt", "/proc",
        "/sys", "/dev", "/var", "/nix", "/snap",
    ]
    .iter()
    .map(PathBuf::from)
    .collect();

    if let Ok(home) = std::env::var("HOME") {
        let home = Path::new(&home);
        for entry in [
            // Git needs its config or `git commit` cannot name an author.
            // `~/.git-credentials` is deliberately not here.
            ".gitconfig",
            ".config/git",
            // Language toolchains that install under $HOME by convention.
            ".rustup",
            ".cargo",
            ".nvm",
            ".pyenv",
            ".rbenv",
            ".asdf",
            ".volta",
            ".sdkman",
            ".bun",
            ".deno",
            ".local/bin",
            ".local/lib",
            ".local/share/virtualenvs",
            "go",
        ] {
            paths.push(home.join(entry));
        }
    }

    // Anything writable is necessarily readable.
    paths.extend(writable_paths(root));
    paths.retain(|p| p.exists());
    paths
}

/// How much of the filesystem a confined command may read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reads {
    /// Unrestricted. Only writes are confined — see the module docs for why
    /// the middle option was removed rather than narrowed.
    Everywhere,
    /// Reads narrowed to the system, the named toolchains and the workspace.
    Workspace,
}

impl Reads {
    fn token(self) -> &'static str {
        match self {
            Reads::Everywhere => "read-all",
            Reads::Workspace => "read-workspace",
        }
    }

    fn parse(token: &str) -> Option<Self> {
        match token {
            "read-all" => Some(Reads::Everywhere),
            "read-workspace" => Some(Reads::Workspace),
            _ => None,
        }
    }
}

/// The confinement a command is to run under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Policy {
    pub network: Network,
    pub reads: Reads,
}

impl Policy {
    /// Writes confined, reads open, network open. The default for a
    /// repository you are working on rather than inspecting.
    pub const fn permissive() -> Self {
        Self {
            network: Network::Allow,
            reads: Reads::Everywhere,
        }
    }

    /// Reads narrowed and the network refused. For a repository you do not
    /// trust.
    pub const fn strict() -> Self {
        Self {
            network: Network::Deny,
            reads: Reads::Workspace,
        }
    }
}

/// The trampoline binary that installs the ruleset, if it can be found.
///
/// Looked for beside the running executable and one level up, which covers
/// both an installed layout (`bin/eventage-code`, `bin/eventage-confine`) and
/// cargo's (`target/debug/deps/some-test`, `target/debug/eventage-confine`).
///
/// Falling back to `current_exe` with the marker argument — which is what
/// this used to do unconditionally — is not safe in general: a binary that
/// does not call [`run_if_helper`] first will hand the marker to its own
/// argument parser. A test binary does exactly that, and libtest reads it as
/// a filter, runs nothing, and exits 0. The caller sees a command that
/// succeeded and produced no output, which is a sandbox failing in the one
/// way you cannot notice. So: no helper, no confinement, and the caller is
/// told rather than misled.
#[cfg(target_os = "linux")]
fn helper_binary() -> Option<PathBuf> {
    use std::sync::OnceLock;
    static HELPER: OnceLock<Option<PathBuf>> = OnceLock::new();
    HELPER
        .get_or_init(|| {
            let exe = std::env::current_exe().ok()?;
            let mut dir = exe.parent()?;
            for _ in 0..2 {
                let candidate = dir.join("eventage-confine");
                if candidate.is_file() {
                    return Some(candidate);
                }
                dir = dir.parent()?;
            }
            None
        })
        .clone()
}

/// Build the command that runs `program` with `args`, confined to `root`.
///
/// Returns `None` when confinement is unavailable — no Landlock in the
/// kernel, or no trampoline binary to install the ruleset — so the caller can
/// decide whether to run unconfined or refuse, rather than silently doing
/// one.
#[cfg(target_os = "linux")]
pub fn confined_argv(
    root: &Path,
    policy: Policy,
    program: &str,
    args: &[String],
) -> Option<std::process::Command> {
    if !available() {
        return None;
    }
    let mut cmd = std::process::Command::new(helper_binary()?);
    cmd.arg(HELPER_ARG)
        .arg(root)
        .arg(policy.network.token())
        .arg(policy.reads.token())
        .arg(program)
        .args(args);
    Some(cmd)
}

#[cfg(not(target_os = "linux"))]
pub fn confined_argv(
    _root: &Path,
    _policy: Policy,
    _program: &str,
    _args: &[String],
) -> Option<std::process::Command> {
    None
}

/// Build the command that runs a shell `script` confined to `root`.
pub fn confined_command(
    root: &Path,
    script: &str,
    policy: Policy,
) -> Option<std::process::Command> {
    // `-c`, not `-lc`: a login shell sources the user's profile, which
    // re-imports the credentials the caller just scrubbed.
    confined_argv(
        root,
        policy,
        "bash",
        &["-c".to_string(), script.to_string()],
    )
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
    // Only Landlock reads it, but the argument is required everywhere so a
    // wrong invocation fails the same way on every platform.
    #[cfg(not(target_os = "linux"))]
    let _ = root;
    let network = args.get(3).map(String::as_str).and_then(Network::parse);
    let Some(network) = network else {
        eprintln!("{HELPER_ARG}: no network policy given");
        std::process::exit(70);
    };
    // Validated on every platform even though only Landlock acts on it: a
    // malformed argument list should fail here, not silently on Linux only.
    let reads = args.get(4).map(String::as_str).and_then(Reads::parse);
    let Some(reads) = reads else {
        eprintln!("{HELPER_ARG}: no read policy given");
        std::process::exit(70);
    };
    #[cfg(not(target_os = "linux"))]
    let _ = reads;
    if args.len() < 6 {
        eprintln!("{HELPER_ARG}: no command given");
        std::process::exit(70);
    }

    #[cfg(target_os = "linux")]
    {
        let root = PathBuf::from(root);
        // Allocation and file opening are fine here: this process is
        // single-threaded and has not forked.
        let readable = match reads {
            // `/` really does mean unrestricted reads: a deliberate choice,
            // documented as such, not an oversight.
            Reads::Everywhere => vec![PathBuf::from("/")],
            Reads::Workspace => readable_paths(&root),
        };
        if let Err(e) = eventage::sandbox::landlock_confine(&readable, &writable_paths(&root)) {
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
        let error = std::process::Command::new(&args[5]).args(&args[6..]).exec();
        eprintln!("{HELPER_ARG}: could not run '{}': {error}", args[5]);
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
        // `cmd > /dev/null` is not an exotic thing to write.
        assert!(paths.contains(&PathBuf::from("/dev/null")));
    }

    #[test]
    fn the_narrow_read_set_excludes_the_home_directory_itself() {
        // The claim the module docs make about `Reads::Workspace` is that
        // `~/.ssh` is unreadable. Landlock grants a whole hierarchy, so that
        // claim holds only if no ancestor of it is in the set.
        let dir = tempfile::tempdir().unwrap();
        let paths = readable_paths(dir.path());
        assert!(paths.contains(&dir.path().to_path_buf()));

        let home = std::env::var("HOME").expect("HOME is set in the test environment");
        let home = Path::new(&home);
        for secret in [".ssh", ".aws", ".git-credentials", ".config/gh", ".netrc"] {
            let secret = home.join(secret);
            assert!(
                !paths.iter().any(|granted| secret.starts_with(granted)),
                "{} is reachable through a granted path in {paths:?}",
                secret.display()
            );
        }
        assert!(
            !paths.iter().any(|p| p == home || p == Path::new("/")),
            "granting $HOME or / defeats the whole set: {paths:?}"
        );
    }

    #[test]
    fn the_helper_argument_order_round_trips() {
        // The helper parses positionally, so a change to the builder that is
        // not matched in the parser silently runs the wrong command.
        let dir = tempfile::tempdir().unwrap();
        let Some(cmd) = confined_argv(dir.path(), Policy::strict(), "echo", &["hello".to_string()])
        else {
            return; // No Landlock on this kernel; nothing to check.
        };
        let args: Vec<_> = cmd.get_args().map(|a| a.to_string_lossy()).collect();
        assert_eq!(args[0], HELPER_ARG);
        assert_eq!(args[1], dir.path().to_string_lossy());
        assert_eq!(Network::parse(&args[2]), Some(Network::Deny));
        assert_eq!(Reads::parse(&args[3]), Some(Reads::Workspace));
        assert_eq!(args[4], "echo");
        assert_eq!(args[5], "hello");
    }
}
