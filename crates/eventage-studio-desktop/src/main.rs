//! Eventage Studio in a native window.
//!
//! # Why a separate crate, outside the workspace
//!
//! Tauri renders through the *system* webview. On macOS that is WKWebView and
//! on Windows WebView2, both part of the OS — but on Linux it is WebKitGTK, a
//! package the user must install. The plain `eventage-studio` binary needs
//! nothing at all: download one file, run it. Folding Tauri in would trade
//! that away on every Linux machine to gain a window frame.
//!
//! So this is an *additional* artifact, and it is excluded from
//! `[workspace.members]` so `cargo build --workspace` and CI keep working on a
//! machine with no WebKitGTK — which is most of them.
//!
//! # It is deliberately thin
//!
//! Everything below `launch::serve` is shared with the plain binary: the same
//! server, the same HTTP API, the same front-end bytes. This starts that
//! server on port 0 and points a webview at the URL it reports. There is no
//! second implementation of anything here to drift out of step, and the
//! interesting behaviour is covered by `launch`'s own test — which binds port
//! 0, checks the reported URL answers, and shuts down.
//!
//! # Under WSL
//!
//! This works, because WSLg gives the distro a Wayland and an X display. It is
//! *not* how VS Code does it — VS Code runs a headless `vscode-server` in WSL
//! and its Electron UI on Windows, which is the split the plain binary uses
//! when it opens a Windows browser in app mode. Both are reasonable; this one
//! gives you a single process and an icon of its own.
//!
//! # Building
//!
//! ```sh
//! # Linux/WSL only: the system webview.
//! sudo apt-get install -y libwebkit2gtk-4.1-dev build-essential libssl-dev \
//!      libayatana-appindicator3-dev librsvg2-dev libdbus-1-dev pkg-config
//!
//! # The front-end is embedded by the server crate, so it must exist first.
//! (cd crates/eventage-studio/ui && npm ci && npm run build)
//!
//! cargo build --release --manifest-path crates/eventage-studio-desktop/Cargo.toml
//! ```

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tauri::{WebviewUrl, WebviewWindowBuilder};

/// Which folder to open as the workspace.
///
/// A program started from a shell inherits the directory you were standing in,
/// and that is the right answer — for a shell. Started from an icon it is not:
/// a `.app` opened from Finder gets `/`, a shortcut gets whatever its "start
/// in" says, which is usually the install directory, and an AppImage gets the
/// inside of its own mount, because linuxdeploy's `AppRun` `chdir`s there. The
/// packaged build was observed opening a workspace called `usr` — the AppDir's
/// `usr` — which is not anybody's project.
///
/// So the working directory is used when it can plausibly be one, and `$HOME`
/// otherwise, from where the workspace picker can take over. Pure, and taking
/// what it needs as arguments, so the cases can be tested without a process to
/// put them in.
///
/// `$HOME` is a fallback and not a good workspace: opening one builds a
/// repository map of it, which took a couple of seconds and 220KB here and will
/// take more on a fuller home directory. It is the least surprising place to
/// land when nothing else is known, and `--cwd` is there for a launcher that
/// knows better.
///
/// `bundle_root` is `$APPDIR` — set by an AppImage's `AppRun` to the root of
/// the mounted bundle. Checked and confirmed by running the packaged build:
/// `APPDIR=/…/squashfs-root` with the working directory at `…/squashfs-root/usr`.
/// It is the bundle root and not the executable's own directory that answers
/// this, because the directory `AppRun` leaves us in is *above* `usr/bin` —
/// which is why "is the working directory inside the executable's folder" does
/// not catch it. The narrower rule stays for the Windows case, where a
/// shortcut's start-in directory is exactly where the program lives, and it
/// deliberately does not look the other way round: a `cargo run` from a project
/// root has the executable under `target/release` *inside* the workspace, and
/// that is the one case that must keep working.
fn choose_workspace(
    cwd: &Path,
    bundle_root: Option<&Path>,
    exe_dir: Option<&Path>,
    home: Option<&Path>,
) -> PathBuf {
    let inside = |dir: Option<&Path>| dir.is_some_and(|dir| cwd.starts_with(dir));
    let unusable = cwd == Path::new("/") || inside(bundle_root) || inside(exe_dir);

    match (unusable, home) {
        (true, Some(home)) => home.to_path_buf(),
        // No home to fall back to is unusual enough that the honest move is to
        // proceed with what we have rather than invent somewhere.
        _ => cwd.to_path_buf(),
    }
}

/// The workspace named on the command line, if one was.
///
/// `--cwd <dir>` names it explicitly, as it does for the plain binary, and
/// overrides every guess in [`choose_workspace`]. It is what a `.desktop`
/// entry or a Windows shortcut can pass, which is the only way a launcher gets
/// to say where your work is.
///
/// Both spellings are accepted, because the plain binary parses its arguments
/// with clap and clap takes both — a flag that works in one of two programs
/// that document it identically is worse than no flag.
fn explicit_workspace(args: impl Iterator<Item = String>) -> Option<PathBuf> {
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        if let Some(dir) = arg.strip_prefix("--cwd=") {
            return Some(PathBuf::from(dir));
        }
        if arg == "--cwd" {
            return args.next().map(PathBuf::from);
        }
    }
    None
}

fn main() -> Result<()> {
    // Before anything parses arguments: a process started with the sandbox
    // marker is not this program, it is a trampoline that confines itself and
    // execs. Calling this also declares that this binary handles the marker,
    // which is what lets a confined command re-execute it.
    eventage_code::shell_sandbox::run_if_helper();

    init_tracing();

    // Studio's server is async and Tauri's event loop is not, so the runtime
    // is built explicitly, the server started on it, and the loop then owns
    // the main thread.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let cwd = match explicit_workspace(std::env::args().skip(1)) {
        Some(dir) => std::fs::canonicalize(&dir)
            .map_err(|e| anyhow::anyhow!("cannot open workspace '{}': {e}", dir.display()))?,
        None => {
            let exe = std::env::current_exe().ok();
            let bundle = std::env::var_os("APPDIR").map(PathBuf::from);
            choose_workspace(
                &std::env::current_dir()?,
                bundle.as_deref(),
                exe.as_deref().and_then(Path::parent),
                dirs::home_dir().as_deref(),
            )
        }
    };
    let cwd = cwd.display().to_string();
    tracing::info!(workspace = %cwd, "opening");

    // The same resolution the plain binary does, from the same code: workspace
    // `.claude/settings.json`, then the environment, then the credentials out of
    // the environment. Done here, before the runtime does anything, because it
    // mutates the process environment.
    let model = eventage_studio::launch::resolve_model(&cwd, None);

    let running = runtime.block_on(async {
        let state_dir =
            eventage_code::config::SessionConfig::new(cwd.clone(), model.clone()).state_dir();
        tokio::fs::create_dir_all(&state_dir).await.ok();

        let settings =
            Arc::new(eventage_studio::model_settings::ModelSettings::load(model, &state_dir).await);
        let backend = Arc::new(
            eventage_studio::backend::local::LocalBackend::new(settings, cwd.clone()).await,
        );
        eventage_studio::launch::serve(backend, 0).await
    })?;

    tracing::info!(url = %running.url, "studio server started");
    let url: tauri::Url = running.url.parse()?;

    // Held so the exit handler can shut it down. Dropping `Running` stops
    // nothing: sessions own language servers and ACP child processes, and
    // without an explicit shutdown they outlive the window that started them.
    // An app that leaves processes behind when you close it is worse than one
    // that never started them.
    let running = std::sync::Mutex::new(Some(running));
    let handle = runtime.handle().clone();

    tauri::Builder::default()
        .setup(move |app| {
            // Pointed at the loopback server rather than at bundled assets, so
            // the page is served by the same code that serves a browser —
            // including the cookie handshake that carries the session token.
            WebviewWindowBuilder::new(app, "studio", WebviewUrl::External(url.clone()))
                .title("Eventage Studio")
                .inner_size(1280.0, 860.0)
                // The floor is set by the layout, not by taste: the sidebar is
                // 248px and the trace pane 520px, both fixed, so a 1000px window
                // leaves the transcript about 240px and `.body` hides the
                // overflow rather than scrolling it. Measured at ~1010px with
                // the trace pane open, the composer's contents spilled past its
                // own edge. A narrower window is reasonable only once the trace
                // pane is collapsed, which is the user's choice to make, not the
                // shell's to assume.
                .min_inner_size(1100.0, 640.0)
                .build()?;
            Ok(())
        })
        .build(tauri::generate_context!())?
        .run(move |_app, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                if let Some(running) = running.lock().unwrap_or_else(|e| e.into_inner()).take() {
                    tracing::info!("shutting down sessions");
                    handle.block_on(running.shutdown());
                }
            }
        });

    Ok(())
}

fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("eventage_studio=info")),
        )
        .with_writer(std::io::stderr)
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> impl Iterator<Item = String> {
        list.iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[test]
    fn the_workspace_flag_takes_both_spellings() {
        // clap accepts both for the plain binary, so a hand-rolled parser that
        // took only one would make the same documented flag work in one program
        // and fail in the other.
        assert_eq!(
            explicit_workspace(args(&["--cwd", "/home/you/project"])),
            Some(PathBuf::from("/home/you/project"))
        );
        assert_eq!(
            explicit_workspace(args(&["--cwd=/home/you/project"])),
            Some(PathBuf::from("/home/you/project"))
        );
    }

    #[test]
    fn no_workspace_flag_means_no_opinion() {
        assert_eq!(explicit_workspace(args(&[])), None);
        assert_eq!(explicit_workspace(args(&["--verbose"])), None);
        // A trailing `--cwd` with nothing after it is not a path.
        assert_eq!(explicit_workspace(args(&["--cwd"])), None);
    }

    #[test]
    fn a_shell_launch_keeps_the_directory_you_were_standing_in() {
        // The developer case, and the one that must not regress: run it in a
        // project and that project is the workspace.
        let chosen = choose_workspace(
            Path::new("/home/you/project"),
            None,
            Some(Path::new("/usr/bin")),
            Some(Path::new("/home/you")),
        );
        assert_eq!(chosen, Path::new("/home/you/project"));
    }

    #[test]
    fn an_appimage_does_not_open_its_own_mount_as_the_workspace() {
        // What the packaged build actually did: linuxdeploy's AppRun chdirs
        // into the AppDir, so `current_dir` was the mount's `usr` and Studio
        // offered to work on a folder called "usr".
        // The paths are the shape the packaged build really reported:
        // APPDIR at the bundle root, the working directory one level into it.
        let chosen = choose_workspace(
            Path::new("/tmp/.mount_Eventa1B2c3/usr"),
            Some(Path::new("/tmp/.mount_Eventa1B2c3")),
            Some(Path::new("/tmp/.mount_Eventa1B2c3/usr/bin")),
            Some(Path::new("/home/you")),
        );
        assert_eq!(chosen, Path::new("/home/you"));
    }

    #[test]
    fn an_installed_shortcut_does_not_open_the_install_directory() {
        // A Windows shortcut's "start in" is normally where the program was
        // installed, which is the program, not your work.
        let chosen = choose_workspace(
            Path::new(r"C:\Program Files\Eventage Studio"),
            None,
            Some(Path::new(r"C:\Program Files\Eventage Studio")),
            Some(Path::new(r"C:\Users\you")),
        );
        assert_eq!(chosen, Path::new(r"C:\Users\you"));
    }

    #[test]
    fn a_finder_launch_lands_at_home_rather_than_the_root_of_the_disk() {
        let chosen = choose_workspace(
            Path::new("/"),
            None,
            Some(Path::new(
                "/Applications/Eventage Studio.app/Contents/MacOS",
            )),
            Some(Path::new("/Users/you")),
        );
        assert_eq!(chosen, Path::new("/Users/you"));
    }

    #[test]
    fn with_no_home_to_fall_back_to_it_proceeds_rather_than_inventing_one() {
        let chosen = choose_workspace(Path::new("/"), None, Some(Path::new("/usr/bin")), None);
        assert_eq!(chosen, Path::new("/"), "it must not guess a path");
    }
}
