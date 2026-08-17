//! What an adversarial repository can reach.
//!
//! The threat is not exotic. Clone a repository, point an agent at it, and the
//! repository decides what is on disk — including symlinks. Every tool here is
//! reachable in a permission mode the user would consider safe: `read_file`,
//! `grep`, `glob`, `list_directory` and `view_image` are all classified
//! read-only and allowed even in Plan mode, so nothing prompts.
//!
//! These drive the real tools rather than the workspace API, because that is
//! where the boundary has to hold. The old lexical check passed every one of
//! these paths.

#![cfg(unix)]

use eventage::agent::Tool;
use eventage_code::tools;
use eventage_code::workspace::Workspace;
use serde_json::json;
use std::sync::Arc;

const SECRET: &str = "BEGIN OPENSSH PRIVATE KEY";

/// A repository containing links out, next to a directory holding a secret.
fn hostile() -> (tempfile::TempDir, tempfile::TempDir, Arc<Workspace>) {
    let outside = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(outside.path().join(".ssh")).unwrap();
    std::fs::write(outside.path().join(".ssh/id_rsa"), SECRET).unwrap();
    std::fs::write(outside.path().join("secret.png"), b"\x89PNG\r\n\x1a\n").unwrap();

    let repo = tempfile::tempdir().unwrap();
    std::fs::write(repo.path().join("README.md"), "an ordinary file\n").unwrap();
    // A directory link — the classic escape.
    std::os::unix::fs::symlink(outside.path(), repo.path().join("vendor")).unwrap();
    // And a link that looks like a source file.
    std::os::unix::fs::symlink(
        outside.path().join(".ssh/id_rsa"),
        repo.path().join("config.rs"),
    )
    .unwrap();

    let ws = Arc::new(Workspace::open(repo.path()).unwrap());
    (outside, repo, ws)
}

#[tokio::test]
async fn read_file_cannot_be_walked_out_of_the_repository() {
    let (_outside, _repo, ws) = hostile();
    let tool = tools::ReadFile {
        ws: ws.clone(),
        client: None,
    };

    for path in ["vendor/.ssh/id_rsa", "config.rs"] {
        let result = tool.execute(json!({ "path": path })).await;
        let rendered = format!("{result:?}");
        assert!(
            !rendered.contains(SECRET),
            "'{path}' leaked the key: {rendered}"
        );
    }

    // The repository's own files still read.
    let ok = tool.execute(json!({ "path": "README.md" })).await.unwrap();
    assert!(ok["content"].as_str().unwrap().contains("an ordinary file"));
}

#[tokio::test]
async fn grep_cannot_be_rooted_outside_the_repository() {
    // `Walk::new` follows a symlinked *root*, so the starting directory is
    // its own escape even though the walker does not follow links inside.
    let (_outside, _repo, ws) = hostile();
    let tool = tools::Grep { ws: ws.clone() };

    let result = tool
        .execute(json!({ "pattern": "PRIVATE KEY", "path": "vendor" }))
        .await;
    assert!(
        !format!("{result:?}").contains(SECRET),
        "grep read outside the workspace: {result:?}"
    );

    // A search over the repository itself still works.
    let hits = tool
        .execute(json!({ "pattern": "ordinary" }))
        .await
        .unwrap();
    assert_eq!(hits["count"], 1, "{hits}");
}

#[tokio::test]
async fn a_whole_repository_search_does_not_follow_links_out() {
    let (_outside, _repo, ws) = hostile();
    let result = (tools::Grep { ws: ws.clone() })
        .execute(json!({ "pattern": "PRIVATE KEY" }))
        .await
        .unwrap();
    assert_eq!(result["count"], 0, "{result}");

    let files = (tools::Glob { ws })
        .execute(json!({ "pattern": "**/*" }))
        .await
        .unwrap();
    assert!(
        !format!("{files}").contains("id_rsa"),
        "glob reached outside: {files}"
    );
}

#[tokio::test]
async fn listing_a_link_out_shows_nothing() {
    let (_outside, _repo, ws) = hostile();
    let result = (tools::ListDirectory { ws })
        .execute(json!({ "path": "vendor" }))
        .await;
    assert!(
        !format!("{result:?}").contains(".ssh"),
        "listed outside the workspace: {result:?}"
    );
}

#[tokio::test]
async fn an_image_outside_the_repository_cannot_be_viewed() {
    let (_outside, repo, ws) = hostile();
    std::os::unix::fs::symlink(
        repo.path().join("vendor/secret.png"),
        repo.path().join("mockup.png"),
    )
    .unwrap();

    let result = (tools::vision::ViewImage { ws })
        .execute(json!({ "path": "mockup.png" }))
        .await;
    assert!(result.is_err(), "an image outside was read: {result:?}");
}

#[tokio::test]
async fn writing_through_a_link_cannot_land_outside() {
    let (outside, _repo, ws) = hostile();

    let write = (tools::WriteFile {
        ws: ws.clone(),
        client: None,
        lsp: Arc::new(eventage_code::lsp::LspPool::disabled(ws.root())),
    })
    .execute(json!({ "path": "vendor/planted.rs", "content": "pwned" }))
    .await;
    assert!(write.is_err(), "{write:?}");
    assert!(
        !outside.path().join("planted.rs").exists(),
        "a file was planted outside the workspace"
    );

    // And overwriting the key through the file-shaped link is refused too.
    let overwrite = (tools::EditFile {
        ws: ws.clone(),
        client: None,
        lsp: Arc::new(eventage_code::lsp::LspPool::disabled(ws.root())),
    })
    .execute(json!({
        "path": "config.rs",
        "old_string": "BEGIN",
        "new_string": "GONE",
    }))
    .await;
    assert!(overwrite.is_err(), "{overwrite:?}");
    assert_eq!(
        std::fs::read_to_string(outside.path().join(".ssh/id_rsa")).unwrap(),
        SECRET
    );
}

#[tokio::test]
async fn a_patch_cannot_write_outside_the_repository() {
    let (outside, _repo, ws) = hostile();
    let patch = "*** Begin Patch\n\
                 *** Add File: vendor/planted.rs\n\
                 +pwned\n\
                 *** End Patch\n";

    let result = (tools::patch::ApplyPatch {
        ws: ws.clone(),
        client: None,
        lsp: Arc::new(eventage_code::lsp::LspPool::disabled(ws.root())),
    })
    .execute(json!({ "patch": patch }))
    .await;

    assert!(result.is_err(), "{result:?}");
    assert!(!outside.path().join("planted.rs").exists());
}

// ── shell containment ─────────────────────────────────────────────────────────

/// The credential the agent itself is authenticating with must not be visible
/// to a command the model was talked into running.
#[tokio::test]
async fn a_shell_command_cannot_read_the_agents_own_api_key() {
    let repo = tempfile::tempdir().unwrap();
    let ws = Arc::new(Workspace::open(repo.path()).unwrap());

    // SAFETY: set before the command is spawned; this test reads it back
    // through a child process rather than from another thread.
    unsafe {
        std::env::set_var("ANTHROPIC_API_KEY", "sk-ant-must-not-leak");
        std::env::set_var("MY_DEPLOY_TOKEN", "tok-must-not-leak");
        std::env::set_var("ORDINARY_SETTING", "keep-me");
    }

    let out = (tools::Bash {
        ws,
        jobs: Arc::new(tools::BackgroundJobs::default()),
        containment: tools::ShellContainment::Confined,
        container_image: tools::DEFAULT_CONTAINER_IMAGE.into(),
    })
    .execute(json!({ "command": "env" }))
    .await
    .unwrap();

    let env = out["stdout"].as_str().unwrap();
    assert!(
        !env.contains("must-not-leak"),
        "a credential reached the shell:\n{env}"
    );
    // But an ordinary variable survives: a build needs its environment, and a
    // scrub that broke every toolchain would just be turned off.
    assert!(
        env.contains("keep-me"),
        "the environment was over-scrubbed:\n{env}"
    );
    // PATH in particular, or nothing runs at all.
    assert!(env.contains("PATH="), "{env}");
}

#[tokio::test]
async fn a_confined_command_runs_in_its_own_process_group() {
    // So a timeout or a cancellation can signal the whole tree rather than
    // the shell alone, leaving its children behind.
    let repo = tempfile::tempdir().unwrap();
    let ws = Arc::new(Workspace::open(repo.path()).unwrap());

    let out = (tools::Bash {
        ws,
        jobs: Arc::new(tools::BackgroundJobs::default()),
        containment: tools::ShellContainment::Confined,
        container_image: tools::DEFAULT_CONTAINER_IMAGE.into(),
    })
    .execute(
        json!({ "command": "test \"$$\" = \"$(ps -o pgid= -p $$ | tr -d ' ')\" && echo leader" }),
    )
    .await
    .unwrap();

    assert_eq!(out["stdout"].as_str().unwrap().trim(), "leader", "{out}");
}

#[tokio::test]
async fn the_result_states_how_the_command_was_contained() {
    let repo = tempfile::tempdir().unwrap();
    let ws = Arc::new(Workspace::open(repo.path()).unwrap());
    let out = (tools::Bash {
        ws,
        jobs: Arc::new(tools::BackgroundJobs::default()),
        containment: tools::ShellContainment::Confined,
        container_image: tools::DEFAULT_CONTAINER_IMAGE.into(),
    })
    .execute(json!({ "command": "true" }))
    .await
    .unwrap();
    assert!(
        out["containment"].as_str().unwrap().contains("scrubbed"),
        "{out}"
    );
}

#[tokio::test]
async fn a_runaway_command_is_stopped_by_the_kernel_not_by_the_machine_dying() {
    // A wall-clock timeout catches a command that hangs. It does nothing
    // about one that allocates until the machine swaps.
    let repo = tempfile::tempdir().unwrap();
    let ws = Arc::new(Workspace::open(repo.path()).unwrap());

    let out = (tools::Bash {
        ws,
        jobs: Arc::new(tools::BackgroundJobs::default()),
        containment: tools::ShellContainment::Confined,
        container_image: tools::DEFAULT_CONTAINER_IMAGE.into(),
    })
    // Far past the address-space limit, and fast.
    .execute(json!({ "command": "ulimit -v" }))
    .await
    .unwrap();

    let limit: u64 = out["stdout"].as_str().unwrap().trim().parse().unwrap_or(0);
    assert!(limit > 0, "no address-space limit was applied: {out}");
    assert!(
        limit < 16 * 1024 * 1024,
        "the limit is not a limit: {limit} KB"
    );
}

#[tokio::test]
async fn a_background_job_can_be_listed_and_stopped() {
    // They used to be a pid and a log path, recorded and then forgotten.
    let repo = tempfile::tempdir().unwrap();
    let ws = Arc::new(Workspace::open(repo.path()).unwrap());
    let jobs = Arc::new(tools::BackgroundJobs::default());

    let started = (tools::Bash {
        ws,
        jobs: Arc::clone(&jobs),
        containment: tools::ShellContainment::Confined,
        container_image: tools::DEFAULT_CONTAINER_IMAGE.into(),
    })
    .execute(json!({ "command": "sleep 60", "background": true }))
    .await
    .unwrap();
    let pid = started["pid"].as_i64().unwrap();
    assert!(pid > 0, "{started}");

    let tool = tools::Jobs {
        jobs: Arc::clone(&jobs),
    };
    let listed = tool.execute(json!({})).await.unwrap();
    assert_eq!(listed["jobs"][0]["running"], true, "{listed}");
    assert_eq!(listed["jobs"][0]["command"], "sleep 60");

    tool.execute(json!({ "stop": pid })).await.unwrap();
    let after = tool.execute(json!({})).await.unwrap();
    assert_eq!(after["jobs"][0]["running"], false, "{after}");

    // And an unknown pid says so rather than silently doing nothing.
    assert!(tool.execute(json!({ "stop": 999_999 })).await.is_err());
}

#[tokio::test]
async fn confined_mode_says_plainly_that_reads_are_not_confined() {
    // This test used to assert the opposite — that `Confined` blocked a read
    // outside the workspace — and it was right when it was written. The read
    // policies were later collapsed to two, and `Confined` became
    // writes-confined/reads-open, which is the trade Codex's Linux sandbox
    // makes and is documented as such. The assertion was not updated, so it
    // failed on CI and passed here, because this kernel has no Landlock and
    // the whole path is skipped.
    //
    // What is worth testing is not the old guarantee but the *agreement*
    // between what happens and what the tool tells the model. A containment
    // string that claims more than it delivers is the failure mode that
    // matters, and it is the one a reader of the result cannot detect.
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("id_rsa"), SECRET).unwrap();

    let repo = tempfile::tempdir().unwrap();
    let ws = Arc::new(Workspace::open(repo.path()).unwrap());

    let out = (tools::Bash {
        ws,
        jobs: Arc::new(tools::BackgroundJobs::default()),
        containment: tools::ShellContainment::Confined,
        container_image: tools::DEFAULT_CONTAINER_IMAGE.into(),
    })
    .execute(json!({
        "command": format!("cat {}/id_rsa", outside.path().display())
    }))
    .await
    .unwrap();

    let stated = out["containment"].as_str().unwrap_or_default();
    let read_it = format!("{out}").contains(SECRET);

    // Two truthful strings can describe an unconfined read: the deliberate
    // "reads are open" of this mode, and the "this kernel has no Landlock"
    // fallback. Either is honest; claiming confinement is not. Written this
    // way the test runs on both kinds of kernel rather than skipping on one —
    // which is how the stale assertion survived: it only ever ran on CI.
    if read_it {
        assert!(
            stated.contains("NOT confined"),
            "the read succeeded while the result claimed the filesystem was confined: {out}"
        );
    } else {
        // If a future change does narrow reads in this mode, the string has to
        // stop disclaiming it — otherwise the model is told it has less
        // authority than it does, which is the same defect pointing the other
        // way.
        assert!(
            !stated.contains("Reads are NOT confined"),
            "the read was blocked while the result still disclaimed confinement: {out}"
        );
    }

    // Whatever the read policy, writes outside the workspace are the actual
    // guarantee of this mode, and `Strict` is what narrows reads — covered by
    // `shell_sandbox`'s own tests, which can check the path set directly
    // instead of needing a kernel that enforces it.
    //
    // Note for anyone tempted to test read confinement this way: it cannot be
    // done with a `tempfile::tempdir`. `writable_paths` grants
    // `std::env::temp_dir()` so that builds work, so `/tmp` is readable under
    // *both* modes and a secret placed there proves nothing.
}

#[tokio::test]
async fn a_confined_command_can_still_do_its_job_in_the_workspace() {
    // Confinement that breaks `cargo test` gets switched off, so this is as
    // important as the test above.
    let repo = tempfile::tempdir().unwrap();
    std::fs::write(repo.path().join("input.txt"), "hello\n").unwrap();
    let ws = Arc::new(Workspace::open(repo.path()).unwrap());

    let out = (tools::Bash {
        ws,
        jobs: Arc::new(tools::BackgroundJobs::default()),
        containment: tools::ShellContainment::Confined,
        container_image: tools::DEFAULT_CONTAINER_IMAGE.into(),
    })
    .execute(json!({ "command": "cat input.txt && echo written > output.txt && ls" }))
    .await
    .unwrap();

    assert_eq!(out["exit_code"], 0, "{out}");
    assert!(out["stdout"].as_str().unwrap().contains("hello"), "{out}");
    assert!(
        repo.path().join("output.txt").exists(),
        "the write failed: {out}"
    );
}

// ── the verify capability ─────────────────────────────────────────────────────

#[tokio::test]
async fn container_containment_says_what_it_needs_when_docker_is_absent() {
    // The one containment that is a boundary rather than a seatbelt — and the
    // one with a dependency outside the process. When that dependency is
    // missing the tool has to say so in a way somebody can act on, rather
    // than failing obscurely or, worse, quietly running on the host.
    let repo = tempfile::tempdir().unwrap();
    let ws = Arc::new(Workspace::open(repo.path()).unwrap());
    let tool = tools::Bash {
        ws,
        jobs: Arc::new(tools::BackgroundJobs::default()),
        containment: tools::ShellContainment::Container,
        container_image: tools::DEFAULT_CONTAINER_IMAGE.into(),
    };

    match tool.execute(json!({ "command": "echo hi" })).await {
        Ok(result) => {
            // Docker is here: it ran, and said how.
            let stated = result["containment"].as_str().unwrap_or_default();
            assert!(stated.starts_with("container ("), "{result}");
            assert!(stated.contains("no network"), "{result}");
        }
        Err(e) => {
            let message = e.to_string();
            assert!(message.contains("docker pull"), "unactionable: {message}");
            assert!(
                message.contains(tools::DEFAULT_CONTAINER_IMAGE),
                "{message}"
            );
        }
    }

    // Background work has nowhere to live when the container is torn down
    // with the command, and saying so beats leaving a job that never appears.
    let err = tools::Bash {
        ws: Arc::new(Workspace::open(repo.path()).unwrap()),
        jobs: Arc::new(tools::BackgroundJobs::default()),
        containment: tools::ShellContainment::Container,
        container_image: tools::DEFAULT_CONTAINER_IMAGE.into(),
    }
    .execute(json!({ "command": "sleep 60", "background": true }))
    .await
    .unwrap_err()
    .to_string();
    assert!(err.contains("torn down"), "{err}");
}

#[tokio::test]
async fn strict_containment_denies_the_network() {
    // Landlock covers the filesystem and nothing else, which is why the first
    // answer to "how do we stop a command phoning home" was a container. It
    // did not have to be: refusing `socket(2)` for the internet families is a
    // seccomp filter on the same process, keeping the host's real toolchain.
    let repo = tempfile::tempdir().unwrap();
    let ws = Arc::new(Workspace::open(repo.path()).unwrap());

    let out = (tools::Bash {
        ws: Arc::clone(&ws),
        jobs: Arc::new(tools::BackgroundJobs::default()),
        containment: tools::ShellContainment::Strict,
        container_image: tools::DEFAULT_CONTAINER_IMAGE.into(),
    })
    .execute(json!({ "command": "getent hosts example.com || echo BLOCKED" }))
    .await;

    if !eventage_code::shell_sandbox::available() {
        // Strict refuses outright when it cannot confine the filesystem, so
        // there is nothing to observe on a kernel without Landlock.
        assert!(out.unwrap_err().to_string().contains("refusing to run"));
        return;
    }
    let result = out.unwrap();
    assert!(
        format!("{result}").contains("BLOCKED"),
        "the network was reachable under strict containment: {result}"
    );
}

#[tokio::test]
async fn confined_containment_leaves_the_network_alone() {
    // A build resolving dependencies needs it, and a mode that breaks
    // `cargo build` is a mode nobody keeps switched on.
    let repo = tempfile::tempdir().unwrap();
    let ws = Arc::new(Workspace::open(repo.path()).unwrap());
    let out = (tools::Bash {
        ws,
        jobs: Arc::new(tools::BackgroundJobs::default()),
        containment: tools::ShellContainment::Confined,
        container_image: tools::DEFAULT_CONTAINER_IMAGE.into(),
    })
    .execute(json!({ "command": "echo fine" }))
    .await
    .unwrap();
    assert_eq!(out["exit_code"], 0, "{out}");
}

#[test]
fn every_containment_level_has_a_name_a_user_can_type() {
    for id in ["host", "confined", "strict", "container"] {
        assert!(
            tools::ShellContainment::from_id(id).is_some(),
            "'{id}' should be selectable"
        );
    }
    assert!(tools::ShellContainment::from_id("sandbox").is_none());
}

#[tokio::test]
async fn an_invalid_glob_is_an_error_not_a_different_search() {
    // It used to fall back to a substring match, which answers a question the
    // caller did not ask and says nothing about having done so.
    let repo = tempfile::tempdir().unwrap();
    std::fs::write(repo.path().join("keep.rs"), "x").unwrap();
    let ws = Arc::new(Workspace::open(repo.path()).unwrap());

    let err = (tools::Glob {
        ws: Arc::clone(&ws),
    })
    .execute(json!({ "pattern": "[unclosed" }))
    .await
    .unwrap_err()
    .to_string();
    assert!(err.contains("not a valid glob"), "{err}");

    // A valid one still works.
    let ok = (tools::Glob { ws })
        .execute(json!({ "pattern": "**/*.rs" }))
        .await
        .unwrap();
    assert!(format!("{ok}").contains("keep.rs"), "{ok}");
}

#[tokio::test]
async fn an_invalid_grep_filter_is_reported_rather_than_ignored() {
    let repo = tempfile::tempdir().unwrap();
    std::fs::write(repo.path().join("a.rs"), "needle\n").unwrap();
    let ws = Arc::new(Workspace::open(repo.path()).unwrap());

    let err = (tools::Grep { ws })
        .execute(json!({ "pattern": "needle", "glob": "[unclosed" }))
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("invalid 'glob' filter"), "{err}");
}

#[test]
fn plan_mode_does_not_run_the_repositorys_code() {
    // Plan mode's own description says "no edits or commands". There used to
    // be a second execution path — `verify` — on the read-only list, so it was
    // automatically permitted here, and `cargo test` runs `build.rs`,
    // `npm run` runs whatever the package names, `pytest` imports
    // `conftest.py`. A mode that promises to run nothing and then runs the
    // repository's test suite is lying to whoever chose it.
    use eventage_code::config::PermissionMode;

    // There is one execution path now, and it is not read-only.
    assert!(!PermissionMode::READ_ONLY_TOOLS.contains(&"bash"));
    assert!(PermissionMode::RISKY_TOOLS.contains(&"bash"));
    for tool in PermissionMode::READ_ONLY_TOOLS {
        assert!(
            !["verify", "shell", "run", "exec"].contains(tool),
            "'{tool}' is automatically permitted and can execute code"
        );
    }

    // The promise the label makes, checked against the list it is built from.
    assert!(PermissionMode::Plan
        .description()
        .contains("no edits or commands"));
}
