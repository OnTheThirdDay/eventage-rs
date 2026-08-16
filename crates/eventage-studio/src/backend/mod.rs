//! What Studio needs from whatever is doing the work.
//!
//! Two implementations sit behind these traits. [`local`] hosts a
//! `CodingSession` in this process and forwards its bus, which is what lets
//! the trace panel show hook decisions, token accounting and the branch
//! structure of the event DAG. [`acp`] drives a separate agent over the Agent
//! Client Protocol, which works with any ACP-capable agent but can only show
//! what the protocol carries.

pub mod acp;
pub mod cowork;
pub mod local;

use crate::feed::EventFeed;
use crate::protocol::{
    AppInfo, NewSessionRequest, PermissionResponse, PromptBlock, SessionInfo, StoredSession,
    SummaryOverride,
};
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

#[async_trait]
pub trait Backend: Send + Sync + 'static {
    /// Title-bar facts: which agent, which model, what the trace can show.
    fn info(&self) -> AppInfo;

    /// Open a session — fresh, or reopened from disk when `resume` is set.
    async fn open(&self, req: NewSessionRequest) -> Result<Arc<dyn Session>>;

    /// Fork `source` at `from_seq` into a new session.
    ///
    /// The point is to try a different direction without losing the one you
    /// have: rewind edits a session in place, branching leaves it untouched
    /// and starts a sibling from the same history.
    async fn branch(&self, source: &dyn Session, from_seq: u64) -> Result<Arc<dyn Session>>;

    /// Sessions on disk that are not currently open.
    async fn stored(&self) -> Vec<StoredSession>;

    /// Forget a stored session permanently.
    async fn forget(&self, id: &str) -> Result<()>;
}

/// What adopting a workstream did, or why it did not.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Adopted {
    /// Paths written into the folder.
    pub changed: Vec<String>,
    /// Paths changed by both the workstream and the folder since the base.
    /// Non-empty means nothing was written.
    pub conflicts: Vec<AdoptConflict>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AdoptConflict {
    pub path: String,
    pub workstream: String,
    pub live: String,
}

#[async_trait]
pub trait Session: Send + Sync + 'static {
    /// The event stream this session's views are derived from.
    fn feed(&self) -> Arc<EventFeed>;

    fn info(&self) -> SessionInfo;

    /// Start a turn. Returns an error if one is already running, so the UI
    /// can keep the composer honest rather than silently queueing.
    async fn prompt(&self, blocks: Vec<PromptBlock>) -> Result<()>;

    /// Stop the running turn.
    async fn interrupt(&self) -> Result<()>;

    async fn set_mode(&self, mode: &str) -> Result<()>;

    /// Undo history, returning how many turns remain.
    ///
    /// `to` names a checkpoint to return to; without one, `turns` are counted
    /// back from the newest.
    async fn rewind(&self, turns: usize, to: Option<&str>) -> Result<usize>;

    /// Replace the summary the next request will carry.
    ///
    /// For when compaction dropped something that mattered: the correction is
    /// appended to the log, so the original stays visible in the trace.
    async fn override_summary(&self, replacement: SummaryOverride) -> Result<()>;

    /// Answer a pending permission request.
    async fn permission(&self, response: PermissionResponse) -> Result<()>;

    /// Apply one workstream's result to the folder.
    ///
    /// Cowork's alone, and defaulted so the coding and ACP backends need to
    /// know nothing about it. A cowork turn deliberately leaves its results
    /// unmerged — several were run precisely because they are not equally
    /// good — so choosing between them is an action the session has to offer
    /// rather than something the turn does on its way out.
    /// Returns the paths written, or the conflicts that stopped it.
    ///
    /// `override_conflicts` is the user having seen them and chosen the
    /// workstream's version anyway.
    async fn adopt(&self, _workstream_id: &str, _override_conflicts: bool) -> Result<Adopted> {
        anyhow::bail!("this backend has no workstreams to adopt")
    }

    /// Abandon a workstream, recording why.
    ///
    /// Not a delete. The reasoning stays in the graph, and a later attempt is
    /// told about it — which is the difference between this and rejecting a
    /// diff in the products cowork is answering.
    async fn seal(&self, _workstream_id: &str, _why: &str) -> Result<()> {
        anyhow::bail!("this backend has no workstreams to seal")
    }

    /// Release everything the session holds. Called when it is closed and at
    /// shutdown, so child processes and LSP servers do not outlive the app.
    async fn shutdown(&self);
}
