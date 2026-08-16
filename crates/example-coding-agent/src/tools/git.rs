//! Git and pull-request tooling.
//!
//! Version control is where a coding agent's work becomes reviewable, so the
//! agent gets a first-class tool rather than raw `bash` calls: status and
//! diffs are structured, commits are checked for conventional-commit form,
//! and PR creation shells out to `gh` when it is available.

use crate::workspace::Workspace;
use async_trait::async_trait;
use eventage::agent::{AgentError, Tool};
use eventage::llm::ToolDefinition;
use serde_json::{json, Value};
use std::sync::Arc;

/// Run a git command in the workspace, returning stdout on success.
///
/// Two things this does that a bare `Command::new("git")` does not, both
/// because git is not an inert program — it runs scripts out of the
/// repository it is pointed at.
///
/// **Credentials are scrubbed.** `git commit` executes `.git/hooks/pre-commit`
/// and `commit-msg`, `git checkout` executes `post-checkout`; every one of
/// them is a shell script the repository supplies. Inheriting this process's
/// environment handed each of them `ANTHROPIC_API_KEY`, in a tool whose whole
/// job is to run on a repository somebody else wrote.
///
/// **Hooks are disabled.** `core.hooksPath` is pointed at a directory that
/// does not exist, so git finds no hooks to run. This is a real cost — a
/// `pre-commit` that runs `cargo fmt` will not — and it is worth paying,
/// because executing a repository's scripts is incidental to recording a
/// commit rather than the point of it. Contrast `verify`, which exists to run
/// the project's own code and so is confined instead of neutered. A user who
/// wants their hooks can run `git commit` through `bash`, where the shell
/// containment applies.
///
/// **And it runs under the session's containment**, like every other tool
/// that starts a process. `git` is on the risky list precisely because it
/// reaches outside the workspace — `push` talks to a remote, `checkout`
/// rewrites the tree — and it was the last thing here still running with the
/// host's full filesystem. Belt and braces with the hook suppression above:
/// hooks are the known execution path, and containment covers the ones nobody
/// has thought of.
pub async fn git(
    ws: &Workspace,
    containment: crate::tools::ShellContainment,
    args: &[&str],
) -> Result<String, AgentError> {
    // Before the subcommand: `git -c ... commit` is configuration,
    // `git commit -c ...` is a different flag entirely.
    let mut argv: Vec<String> = vec![
        "-c".into(),
        "core.hooksPath=/nonexistent/eventage-hooks-disabled".into(),
    ];
    argv.extend(args.iter().map(|a| a.to_string()));

    let mut cmd = match containment.confined_command("git", &argv, ws.root())? {
        Some(helper) => helper,
        None => {
            let mut plain = tokio::process::Command::new("git");
            plain.args(&argv);
            plain
        }
    };

    let output = cmd
        .current_dir(ws.root())
        .env_clear()
        .envs(crate::tools::scrubbed_env())
        .output()
        .await
        .map_err(|e| AgentError::Tool(format!("git not available: {e}")))?;
    if !output.status.success() {
        return Err(AgentError::Tool(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Conventional-commit prefixes we accept without complaint.
const CONVENTIONAL_PREFIXES: &[&str] = &[
    "feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore", "revert",
];

/// `true` when `subject` looks like `type(scope)!: description`.
pub fn is_conventional(subject: &str) -> bool {
    let Some((head, rest)) = subject.split_once(':') else {
        return false;
    };
    if rest.trim().is_empty() {
        return false;
    }
    let kind = head
        .split_once('(')
        .map(|(k, _)| k)
        .unwrap_or(head)
        .trim_end_matches('!');
    CONVENTIONAL_PREFIXES.contains(&kind)
}

pub struct Git {
    pub ws: Arc<Workspace>,
    /// The session's shell containment, applied to git like any other
    /// process the agent starts.
    pub containment: crate::tools::ShellContainment,
}

#[async_trait]
impl Tool for Git {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "git",
            "Inspect and record version-control state: status, diff, log, branch, \
             add, commit, and push. Commit messages should follow Conventional \
             Commits (e.g. 'fix(parser): handle empty input').",
            json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["status", "diff", "log", "branch", "add", "commit", "push"]
                    },
                    "message": {
                        "type": "string",
                        "description": "Commit message (action=commit). First line is the subject."
                    },
                    "paths": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Paths for add/diff; defaults to everything"
                    },
                    "name": { "type": "string", "description": "Branch name (action=branch)" },
                    "staged": { "type": "boolean", "description": "Diff the index instead of the worktree" }
                },
                "required": ["action"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::Tool("missing 'action'".into()))?;
        let paths: Vec<String> = args
            .get("paths")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|p| p.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();

        match action {
            "status" => {
                let porcelain = git(
                    &self.ws,
                    self.containment,
                    &["status", "--porcelain=v1", "-b"],
                )
                .await?;
                let files: Vec<Value> = porcelain
                    .lines()
                    .skip_while(|l| l.starts_with("##"))
                    .filter(|l| l.len() > 3)
                    .map(|l| json!({ "status": l[..2].trim(), "path": l[3..].to_string() }))
                    .collect();
                let branch = porcelain
                    .lines()
                    .find(|l| l.starts_with("##"))
                    .map(|l| l.trim_start_matches("## ").to_string())
                    .unwrap_or_default();
                Ok(json!({ "branch": branch, "changed": files.len(), "files": files }))
            }

            "diff" => {
                let mut argv = vec!["diff"];
                if args.get("staged").and_then(|v| v.as_bool()) == Some(true) {
                    argv.push("--cached");
                }
                if !paths.is_empty() {
                    argv.push("--");
                }
                let mut owned: Vec<&str> = argv;
                owned.extend(paths.iter().map(String::as_str));
                let diff = git(&self.ws, self.containment, &owned).await?;
                Ok(json!({
                    "diff": diff.chars().take(60_000).collect::<String>(),
                    "truncated": diff.len() > 60_000,
                }))
            }

            "log" => {
                let log = git(
                    &self.ws,
                    self.containment,
                    &["log", "--oneline", "--decorate", "-20", "--no-color"],
                )
                .await?;
                Ok(json!({ "log": log }))
            }

            "branch" => match args.get("name").and_then(|v| v.as_str()) {
                Some(name) => {
                    git(&self.ws, self.containment, &["checkout", "-b", name]).await?;
                    Ok(json!({ "created": name }))
                }
                None => Ok(
                    json!({ "current": git(&self.ws, self.containment, &["branch", "--show-current"]).await?.trim() }),
                ),
            },

            "add" => {
                let mut argv = vec!["add"];
                if paths.is_empty() {
                    argv.push("-A");
                }
                let mut owned: Vec<&str> = argv;
                owned.extend(paths.iter().map(String::as_str));
                git(&self.ws, self.containment, &owned).await?;
                Ok(
                    json!({ "staged": if paths.is_empty() { vec!["<all>".to_string()] } else { paths } }),
                )
            }

            "commit" => {
                let message = args
                    .get("message")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AgentError::Tool("commit requires 'message'".into()))?;
                let subject = message.lines().next().unwrap_or_default();
                if !is_conventional(subject) {
                    return Err(AgentError::Tool(format!(
                        "commit subject '{subject}' is not a Conventional Commit; use \
                         '<type>(<scope>): <description>' with type one of {}",
                        CONVENTIONAL_PREFIXES.join(", ")
                    )));
                }
                // Refuse an empty commit rather than producing a confusing error.
                let staged = git(
                    &self.ws,
                    self.containment,
                    &["diff", "--cached", "--name-only"],
                )
                .await?;
                if staged.trim().is_empty() {
                    return Err(AgentError::Tool("nothing staged; run git add first".into()));
                }
                git(&self.ws, self.containment, &["commit", "-m", message]).await?;
                let sha = git(
                    &self.ws,
                    self.containment,
                    &["rev-parse", "--short", "HEAD"],
                )
                .await?;
                Ok(json!({
                    "commit": sha.trim(),
                    "subject": subject,
                    "files": staged.lines().count(),
                }))
            }

            "push" => {
                let branch = git(&self.ws, self.containment, &["branch", "--show-current"]).await?;
                let branch = branch.trim();
                git(
                    &self.ws,
                    self.containment,
                    &["push", "-u", "origin", branch],
                )
                .await?;
                Ok(json!({ "pushed": branch }))
            }

            other => Err(AgentError::Tool(format!("unknown git action '{other}'"))),
        }
    }
}

/// Open a pull request with the GitHub CLI.
pub struct CreatePullRequest {
    pub ws: Arc<Workspace>,
}

#[async_trait]
impl Tool for CreatePullRequest {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::function(
            "create_pull_request",
            "Open a pull request for the current branch using the GitHub CLI. Commit \
             and push first.",
            json!({
                "type": "object",
                "properties": {
                    "title": { "type": "string" },
                    "body": { "type": "string", "description": "Markdown summary and test plan" },
                    "draft": { "type": "boolean" }
                },
                "required": ["title", "body"]
            }),
        )
    }

    async fn execute(&self, args: Value) -> Result<Value, AgentError> {
        let title = args
            .get("title")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AgentError::Tool("missing 'title'".into()))?;
        let body = args.get("body").and_then(|v| v.as_str()).unwrap_or("");

        let mut argv = vec!["pr", "create", "--title", title, "--body", body];
        if args.get("draft").and_then(|v| v.as_bool()) == Some(true) {
            argv.push("--draft");
        }

        let output = tokio::process::Command::new("gh")
            .args(&argv)
            .current_dir(self.ws.root())
            // `gh` authenticates with its own token; every other credential
            // in this process — the model provider's above all — is nothing
            // to do with opening a pull request. The token is fetched from
            // `secrets` rather than inherited, because startup takes it out
            // of the environment entirely.
            .env_clear()
            .envs(crate::tools::scrubbed_env())
            .envs(
                ["GH_TOKEN", "GITHUB_TOKEN"]
                    .into_iter()
                    .filter_map(|name| crate::secrets::get(name).map(|v| (name.to_string(), v))),
            )
            .output()
            .await
            .map_err(|e| {
                AgentError::Tool(format!(
                    "GitHub CLI not available ({e}); push the branch and open the PR manually"
                ))
            })?;

        if !output.status.success() {
            return Err(AgentError::Tool(format!(
                "gh pr create failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        Ok(json!({ "url": String::from_utf8_lossy(&output.stdout).trim() }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ShellContainment;

    #[test]
    fn recognises_conventional_subjects() {
        assert!(is_conventional("feat: add thing"));
        assert!(is_conventional("fix(parser): handle empty input"));
        assert!(is_conventional("refactor!: drop old API"));
        assert!(is_conventional("chore(deps)!: bump serde"));
    }

    #[test]
    fn rejects_non_conventional_subjects() {
        assert!(!is_conventional("added a thing"));
        assert!(!is_conventional("wip"));
        assert!(!is_conventional("feat:"), "description is required");
        assert!(!is_conventional("nope: something"), "unknown type");
    }

    #[tokio::test]
    async fn a_repository_hook_does_not_run_and_does_not_see_the_api_key() {
        // `git commit` executes `.git/hooks/pre-commit`, a shell script the
        // repository supplies. It used to run with this process's whole
        // environment, so a cloned repository got the model provider's
        // credential handed to it the first time the agent committed
        // anything.
        let dir = tempfile::tempdir().unwrap();
        let ws = Arc::new(Workspace::open(dir.path()).unwrap());

        if git(&ws, ShellContainment::Host, &["init", "-q"])
            .await
            .is_err()
        {
            return; // No git on this machine; nothing to assert.
        }
        let _ = git(
            &ws,
            ShellContainment::Host,
            &["config", "user.email", "t@example.invalid"],
        )
        .await;
        let _ = git(
            &ws,
            ShellContainment::Host,
            &["config", "user.name", "Test"],
        )
        .await;

        let hooks = dir.path().join(".git/hooks");
        std::fs::create_dir_all(&hooks).unwrap();
        let hook = hooks.join("pre-commit");
        std::fs::write(
            &hook,
            "#!/bin/sh\nprintf '%s' \"${EVENTAGE_TEST_FAKE_KEY-unset}\" > \"$(dirname \"$0\")/../../ran\"\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&hook, std::fs::Permissions::from_mode(0o755)).unwrap();
        }

        // SAFETY: set before the child is spawned; no other thread in this
        // test reads it.
        unsafe { std::env::set_var("EVENTAGE_TEST_FAKE_KEY", "sk-should-not-leak") };

        std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
        git(&ws, ShellContainment::Host, &["add", "a.txt"])
            .await
            .unwrap();
        let committed = git(
            &ws,
            ShellContainment::Host,
            &["commit", "-m", "feat: add a"],
        )
        .await;

        unsafe { std::env::remove_var("EVENTAGE_TEST_FAKE_KEY") };
        committed.unwrap();

        let marker = dir.path().join("ran");
        assert!(
            !marker.exists(),
            "the repository's pre-commit hook ran: {:?}",
            std::fs::read_to_string(&marker)
        );
    }

    #[tokio::test]
    async fn commit_rejects_bad_subject_before_touching_git() {
        let dir = tempfile::tempdir().unwrap();
        let ws = Arc::new(Workspace::open(dir.path()).unwrap());
        let tool = Git {
            ws,
            containment: ShellContainment::Host,
        };
        let err = tool
            .execute(json!({ "action": "commit", "message": "just some changes" }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("Conventional Commit"), "{err}");
    }
}
