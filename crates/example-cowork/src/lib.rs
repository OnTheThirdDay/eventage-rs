//! Cowork — a working session you can navigate, not just steer.
//!
//! The shape of the product category is settled: you describe a task, the
//! agent plans and executes it against files you have granted, and you watch
//! and redirect. Claude Cowork runs that on Anthropic's servers with three
//! steering modes; OpenAI's Codex app runs it locally with git worktrees so
//! parallel agents do not collide. Both are good, and both are **forward
//! only** — you can steer, and you can reject a result, but when a line of
//! work goes wrong the recovery is to ask again.
//!
//! This framework is event-sourced: an append-only DAG with checkpoints,
//! rollback, and abandoned trajectories sealed as rejected branches rather
//! than deleted. That makes a different thing possible, and it is what cowork
//! is for:
//!
//! > Fan a goal into parallel workstreams, each on its own branch and its own
//! > copy of the folder. Compare what they produced. Keep one. The others stay
//! > in the graph as evidence, and the agent reads them back as "you tried
//! > that and it did not work."
//!
//! [`shadow`] is what lets that apply to a folder of documents rather than
//! only to a repository. [`steering`] is the contract with the user about how
//! much happens without them.

pub mod channels;
pub mod kinds;
pub mod session;
pub mod shadow;
pub mod steering;
pub mod tools;
pub mod workers;
