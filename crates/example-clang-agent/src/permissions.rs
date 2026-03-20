use async_trait::async_trait;
use eventage::agent::AgentError;

// ── PermissionGate trait ──────────────────────────────────────────────────────

/// Controls whether the agent is allowed to perform a sensitive operation.
///
/// Two built-in implementations:
/// - [`AutoApproveGate`]: always approves (use when running inside a sandbox).
/// - [`StdinPermissionGate`]: prompts the human operator on stdout/stdin.
#[async_trait]
pub trait PermissionGate: Send + Sync {
    /// Request permission to perform `action` on `target`.
    ///
    /// Returns `Ok(())` if approved, or `Err(AgentError::Tool(...))` if denied.
    async fn request(&self, action: &str, target: &str) -> Result<(), AgentError>;
}

// ── AutoApproveGate ───────────────────────────────────────────────────────────

/// Always grants permission.  Used when execution is isolated inside a real
/// sandbox (Landlock or Docker), so no human approval is required.
pub struct AutoApproveGate;

#[async_trait]
impl PermissionGate for AutoApproveGate {
    async fn request(&self, _action: &str, _target: &str) -> Result<(), AgentError> {
        Ok(())
    }
}

// ── StdinPermissionGate ───────────────────────────────────────────────────────

/// Prompts the human operator and waits for an explicit "y" answer.
///
/// Any answer other than "y" (case-insensitive) is treated as denial.
/// Blocking stdin reads are wrapped in [`tokio::task::block_in_place`] so
/// they do not stall the async runtime.
pub struct StdinPermissionGate;

#[async_trait]
impl PermissionGate for StdinPermissionGate {
    async fn request(&self, action: &str, target: &str) -> Result<(), AgentError> {
        let action = action.to_string();
        let target = target.to_string();
        let deny_msg = format!("operation denied by user: {action} \"{target}\"");

        let approved = tokio::task::block_in_place(move || {
            use std::io::{self, BufRead, Write};

            let stdout = io::stdout();
            let stdin = io::stdin();

            print!(
                "\n\x1b[33m[Permission]\x1b[0m Agent wants to \x1b[1m{action}\x1b[0m \"{target}\". Allow? [y/N] "
            );
            stdout.lock().flush().ok();

            let mut line = String::new();
            stdin.lock().read_line(&mut line).ok();
            line.trim().eq_ignore_ascii_case("y")
        });

        if approved {
            Ok(())
        } else {
            Err(AgentError::Tool(deny_msg))
        }
    }
}
