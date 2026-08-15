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
pub async fn git(ws: &Workspace, args: &[&str]) -> Result<String, AgentError> {
    let output = tokio::process::Command::new("git")
        .args(args)
        .current_dir(ws.root())
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
    "feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore",
    "revert",
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
                let porcelain = git(&self.ws, &["status", "--porcelain=v1", "-b"]).await?;
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
                let diff = git(&self.ws, &owned).await?;
                Ok(json!({
                    "diff": diff.chars().take(60_000).collect::<String>(),
                    "truncated": diff.len() > 60_000,
                }))
            }

            "log" => {
                let log = git(
                    &self.ws,
                    &["log", "--oneline", "--decorate", "-20", "--no-color"],
                )
                .await?;
                Ok(json!({ "log": log }))
            }

            "branch" => match args.get("name").and_then(|v| v.as_str()) {
                Some(name) => {
                    git(&self.ws, &["checkout", "-b", name]).await?;
                    Ok(json!({ "created": name }))
                }
                None => Ok(json!({ "current": git(&self.ws, &["branch", "--show-current"]).await?.trim() })),
            },

            "add" => {
                let mut argv = vec!["add"];
                if paths.is_empty() {
                    argv.push("-A");
                }
                let mut owned: Vec<&str> = argv;
                owned.extend(paths.iter().map(String::as_str));
                git(&self.ws, &owned).await?;
                Ok(json!({ "staged": if paths.is_empty() { vec!["<all>".to_string()] } else { paths } }))
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
                let staged = git(&self.ws, &["diff", "--cached", "--name-only"]).await?;
                if staged.trim().is_empty() {
                    return Err(AgentError::Tool(
                        "nothing staged; run git add first".into(),
                    ));
                }
                git(&self.ws, &["commit", "-m", message]).await?;
                let sha = git(&self.ws, &["rev-parse", "--short", "HEAD"]).await?;
                Ok(json!({
                    "commit": sha.trim(),
                    "subject": subject,
                    "files": staged.lines().count(),
                }))
            }

            "push" => {
                let branch = git(&self.ws, &["branch", "--show-current"]).await?;
                let branch = branch.trim();
                git(&self.ws, &["push", "-u", "origin", branch]).await?;
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
    async fn commit_rejects_bad_subject_before_touching_git() {
        let dir = tempfile::tempdir().unwrap();
        let ws = Arc::new(Workspace::open(dir.path()).unwrap());
        let tool = Git { ws };
        let err = tool
            .execute(json!({ "action": "commit", "message": "just some changes" }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("Conventional Commit"), "{err}");
    }
}
