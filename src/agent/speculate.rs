//! Speculative best-of-N execution over the event DAG.
//!
//! This is the capability the DAG architecture was built for: because the
//! event log is the single source of truth and buses can be forked cheaply,
//! an agent can explore **N candidate trajectories in parallel**, keep the
//! best one, and preserve every losing trajectory as a sealed rejected branch
//! that future context assembly can learn from
//! (via [`NegativeAwareContextAssembler`](crate::NegativeAwareContextAssembler)).
//!
//! # How it works
//!
//! 1. The main bus is [`fork`](crate::EventBus::fork)ed once per candidate —
//!    each fork is an isolated copy of the active branch with shared event IDs.
//! 2. Every candidate factory builds an [`Agent`] on its fork (same task,
//!    different temperature, model, strategy, or prompt) and runs one cycle.
//! 3. A [`BranchScorer`] rates each candidate's *new* events.
//! 4. The winner's events are spliced onto the main bus (IDs and parent links
//!    are already consistent); losers are grafted on as rejected branches via
//!    [`adopt_rejected_branch`](crate::EventBus::adopt_rejected_branch).
//! 5. A durable `speculation.completed` event records names, scores, and the
//!    winner for observability and replay.
//!
//! # Example
//!
//! ```no_run
//! # use eventage::{EventBus, AgentBuilder, ReactStrategy};
//! # use eventage::agent::speculate::{best_of_n, SpeculationCandidate, FnScorer};
//! # use eventage::llm::MockLlmProvider;
//! # async fn example(bus: EventBus) -> anyhow::Result<()> {
//! let candidates = vec![
//!     SpeculationCandidate::new("conservative", |fork| {
//!         AgentBuilder::new().bus(fork)
//!             .llm(MockLlmProvider::with_texts(["safe answer"]))
//!             .strategy(ReactStrategy::default()).build()
//!     }),
//!     SpeculationCandidate::new("creative", |fork| {
//!         AgentBuilder::new().bus(fork)
//!             .llm(MockLlmProvider::with_texts(["bold answer"]))
//!             .strategy(ReactStrategy::default()).build()
//!     }),
//! ];
//!
//! // Score: longest final assistant message wins (use an LLM judge in practice).
//! let scorer = FnScorer::new(|events| {
//!     events.iter().rev()
//!         .find_map(|e| e.payload.get("content").and_then(|c| c.as_str()))
//!         .map(|c| c.len() as f64)
//!         .unwrap_or(0.0)
//! });
//!
//! let outcome = best_of_n(&bus, candidates, &scorer).await?;
//! println!("winner: {} ({:?})", outcome.winner_name, outcome.scores);
//! # Ok(())
//! # }
//! ```

use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use tokio::task::JoinSet;
use tracing::{info, warn};

use super::core::Agent;
use super::error::AgentError;
use crate::bus::{BranchId, EventBus};
use crate::event::{kinds, Event};
use crate::llm::{ChatMessage, LlmProvider};

// ── Candidates ────────────────────────────────────────────────────────────────

type AgentFactory = Box<dyn FnOnce(EventBus) -> Agent + Send>;

/// One speculative trajectory: a name plus a factory that builds the agent
/// on a forked bus.
pub struct SpeculationCandidate {
    pub name: String,
    factory: AgentFactory,
}

impl SpeculationCandidate {
    pub fn new(
        name: impl Into<String>,
        factory: impl FnOnce(EventBus) -> Agent + Send + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            factory: Box::new(factory),
        }
    }
}

// ── Scoring ───────────────────────────────────────────────────────────────────

/// Rates a candidate trajectory. Higher is better.
#[async_trait]
pub trait BranchScorer: Send + Sync {
    /// Score the *new* events a candidate produced (its trajectory only —
    /// the shared history is not included).
    async fn score(&self, candidate_events: &[Event]) -> f64;
}

/// Wrap a plain function as a [`BranchScorer`].
pub struct FnScorer<F>(F);

impl<F> FnScorer<F>
where
    F: Fn(&[Event]) -> f64 + Send + Sync,
{
    pub fn new(f: F) -> Self {
        Self(f)
    }
}

#[async_trait]
impl<F> BranchScorer for FnScorer<F>
where
    F: Fn(&[Event]) -> f64 + Send + Sync,
{
    async fn score(&self, candidate_events: &[Event]) -> f64 {
        (self.0)(candidate_events)
    }
}

/// Scores trajectories with an LLM judge on a 0–10 scale.
pub struct LlmJudgeScorer {
    llm: Arc<dyn LlmProvider>,
    /// What "good" means for this task, e.g. "correct, concise, cites sources".
    pub criteria: String,
}

impl LlmJudgeScorer {
    pub fn new(llm: Arc<dyn LlmProvider>, criteria: impl Into<String>) -> Self {
        Self {
            llm,
            criteria: criteria.into(),
        }
    }

    fn render(events: &[Event]) -> String {
        let mut out = String::new();
        for e in events {
            match e.kind.as_str() {
                kinds::ASSISTANT_MESSAGE => {
                    if let Some(text) = e.payload.get("content").and_then(|v| v.as_str()) {
                        if !text.is_empty() {
                            out.push_str("Assistant: ");
                            out.push_str(text);
                            out.push('\n');
                        }
                    }
                }
                kinds::TOOL_CALL_PROPOSED => {
                    if let Some(name) = e.payload.get("name").and_then(|v| v.as_str()) {
                        out.push_str(&format!("Tool call: {name}\n"));
                    }
                }
                kinds::TOOL_RESULT => {
                    if let Some(err) = e.payload.get("error").and_then(|v| v.as_str()) {
                        out.push_str(&format!("Tool error: {err}\n"));
                    }
                }
                _ => {}
            }
        }
        out
    }
}

#[async_trait]
impl BranchScorer for LlmJudgeScorer {
    async fn score(&self, candidate_events: &[Event]) -> f64 {
        let transcript = Self::render(candidate_events);
        let prompt = format!(
            "You are grading one candidate solution produced by an AI agent.\n\
             Criteria: {}\n\nCandidate trajectory:\n{}\n\n\
             Reply with ONLY a number from 0 to 10.",
            self.criteria, transcript
        );
        match self
            .llm
            .complete(vec![ChatMessage::user(prompt)], vec![])
            .await
        {
            Ok(resp) => resp
                .content
                .as_deref()
                .and_then(first_number)
                .unwrap_or(0.0),
            Err(e) => {
                warn!("judge LLM failed: {e}");
                0.0
            }
        }
    }
}

fn first_number(text: &str) -> Option<f64> {
    let start = text.find(|c: char| c.is_ascii_digit())?;
    let tail = &text[start..];
    let end = tail
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(tail.len());
    tail[..end].parse().ok()
}

// ── Best-of-N orchestration ───────────────────────────────────────────────────

/// Result of a [`best_of_n`] round.
#[derive(Debug)]
pub struct SpeculationOutcome {
    /// Index of the winning candidate (in submission order).
    pub winner: usize,
    /// Name of the winning candidate.
    pub winner_name: String,
    /// Scores per candidate, in submission order. Candidates whose cycle
    /// errored score `f64::NEG_INFINITY`.
    pub scores: Vec<f64>,
    /// Number of events the winner contributed to the main bus.
    pub winner_event_count: usize,
    /// Branch IDs of the losing trajectories sealed onto the main bus.
    pub rejected_branch_ids: Vec<BranchId>,
}

/// Run every candidate concurrently on a forked bus, score the trajectories,
/// splice the winner onto `bus`, and seal the losers as rejected branches.
pub async fn best_of_n(
    bus: &EventBus,
    candidates: Vec<SpeculationCandidate>,
    scorer: &dyn BranchScorer,
) -> Result<SpeculationOutcome, AgentError> {
    if candidates.is_empty() {
        return Err(AgentError::Tool("best_of_n: no candidates supplied".into()));
    }

    let anchor = bus.log().await.last().map(|e| e.id);

    /// One candidate's finished trajectory: name, new events, cycle outcome.
    type CandidateRun = (String, Vec<Event>, Result<(), AgentError>);

    // ── Run all candidates concurrently on isolated forks ────────────────────
    let mut join_set: JoinSet<(usize, CandidateRun)> = JoinSet::new();
    for (i, candidate) in candidates.into_iter().enumerate() {
        let bus = bus.clone();
        join_set.spawn(async move {
            let fork = bus.fork().await;
            let baseline = fork.log_len().await;
            let agent = (candidate.factory)(fork.clone());
            let result = agent.cycle().await;
            let new_events = fork.log_since(baseline).await;
            (i, (candidate.name, new_events, result))
        });
    }

    let mut runs: Vec<CandidateRun> = Vec::new();
    let mut collected: Vec<(usize, CandidateRun)> = Vec::new();
    while let Some(joined) = join_set.join_next().await {
        match joined {
            Ok(entry) => collected.push(entry),
            Err(e) => warn!("speculation candidate task panicked: {e}"),
        }
    }
    collected.sort_by_key(|(i, _)| *i);
    for (_, run) in collected {
        runs.push(run);
    }
    if runs.is_empty() {
        return Err(AgentError::Tool(
            "best_of_n: all candidate tasks panicked".into(),
        ));
    }

    // ── Score ─────────────────────────────────────────────────────────────────
    let mut scores = Vec::with_capacity(runs.len());
    for (name, events, result) in &runs {
        let score = match result {
            Ok(()) => scorer.score(events).await,
            Err(e) => {
                warn!(candidate = %name, error = %e, "candidate cycle errored");
                f64::NEG_INFINITY
            }
        };
        scores.push(score);
    }

    let winner = scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    // ── Splice winner, seal losers ────────────────────────────────────────────
    let mut rejected_branch_ids = Vec::new();
    let mut winner_event_count = 0usize;
    let mut winner_name = String::new();

    for (i, (name, events, _)) in runs.into_iter().enumerate() {
        if i == winner {
            winner_event_count = events.len();
            winner_name = name;
            for event in events {
                // parent_event_id is already consistent with the main bus
                // (forks share event IDs), so linkage is preserved.
                bus.publish(event).await?;
            }
        } else if !events.is_empty() {
            rejected_branch_ids.push(bus.adopt_rejected_branch(anchor, events).await);
        }
    }

    info!(
        winner = %winner_name,
        ?scores,
        rejected = rejected_branch_ids.len(),
        "speculation round complete"
    );

    bus.publish(Event::new(
        kinds::SPECULATION_COMPLETED,
        json!({
            "winner_index": winner,
            "winner_name": winner_name,
            "scores": scores,
            "rejected_branches": rejected_branch_ids
                .iter()
                .map(|b| b.to_string())
                .collect::<Vec<_>>(),
        }),
    ))
    .await?;

    Ok(SpeculationOutcome {
        winner,
        winner_name,
        scores,
        winner_event_count,
        rejected_branch_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::builder::AgentBuilder;
    use crate::agent::strategy::ReactStrategy;
    use crate::llm::MockLlmProvider;

    fn candidate(name: &'static str, reply: &'static str) -> SpeculationCandidate {
        SpeculationCandidate::new(name, move |fork| {
            AgentBuilder::new()
                .agent_id(name)
                .bus(fork)
                .llm(MockLlmProvider::with_texts([reply]))
                .strategy(ReactStrategy::default())
                .build()
        })
    }

    #[tokio::test]
    async fn winner_is_spliced_and_losers_are_sealed() {
        let bus = EventBus::new();
        bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "solve it"})))
            .await
            .unwrap();
        let anchor = bus.log().await.last().unwrap().id;

        let scorer = FnScorer::new(|events: &[Event]| {
            events
                .iter()
                .rev()
                .find_map(|e| e.payload.get("content").and_then(|c| c.as_str()))
                .map(|c| c.len() as f64)
                .unwrap_or(0.0)
        });

        let outcome = best_of_n(
            &bus,
            vec![
                candidate("short", "ok"),
                candidate("long", "a much longer, more detailed answer"),
            ],
            &scorer,
        )
        .await
        .unwrap();

        assert_eq!(outcome.winner, 1);
        assert_eq!(outcome.winner_name, "long");
        assert_eq!(outcome.rejected_branch_ids.len(), 1);

        // Winner trajectory is on the active branch.
        let log = bus.log().await;
        let final_msg = log
            .iter()
            .filter(|e| e.kind == kinds::ASSISTANT_MESSAGE)
            .next_back()
            .expect("winner assistant message missing");
        assert_eq!(
            final_msg.payload["content"].as_str().unwrap(),
            "a much longer, more detailed answer"
        );

        // Loser trajectory is available as negative context at the anchor.
        let rejected = bus.rejected_branches_from(anchor).await;
        assert_eq!(rejected.len(), 1);
        assert!(rejected[0]
            .iter()
            .any(|e| e.payload.get("content").and_then(|c| c.as_str()) == Some("ok")));

        // The round is recorded durably.
        assert!(log.iter().any(|e| e.kind == kinds::SPECULATION_COMPLETED));
    }

    #[tokio::test]
    async fn errored_candidates_lose() {
        let bus = EventBus::new();
        bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "go"})))
            .await
            .unwrap();

        // This candidate's LLM always fails.
        struct FailingProvider;
        #[async_trait]
        impl LlmProvider for FailingProvider {
            async fn complete(
                &self,
                _m: Vec<ChatMessage>,
                _t: Vec<crate::llm::ToolDefinition>,
            ) -> Result<crate::llm::LlmResponse, crate::llm::LlmError> {
                Err(crate::llm::LlmError::EmptyResponse)
            }
            fn model(&self) -> &str {
                "failing"
            }
        }

        let failing = SpeculationCandidate::new("failing", |fork| {
            AgentBuilder::new()
                .bus(fork)
                .llm(FailingProvider)
                .strategy(ReactStrategy::default())
                .build()
        });

        let outcome = best_of_n(
            &bus,
            vec![failing, candidate("works", "answer")],
            &FnScorer::new(|_| 1.0),
        )
        .await
        .unwrap();

        assert_eq!(outcome.winner_name, "works");
        assert_eq!(outcome.scores[0], f64::NEG_INFINITY);
    }

    #[test]
    fn parses_judge_scores() {
        assert_eq!(first_number("8"), Some(8.0));
        assert_eq!(first_number("Score: 7.5/10"), Some(7.5));
        assert_eq!(first_number("I'd rate this 9 out of 10"), Some(9.0));
        assert_eq!(first_number("no digits"), None);
    }
}
