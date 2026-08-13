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
use tracing::{debug, info, warn};

use super::core::Agent;
use super::error::AgentError;
use super::strategy::{StepOutcome, ToolExecOptions};
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

// ── Beam search (per-step speculation) ────────────────────────────────────────

/// Configuration for [`beam_search`].
#[derive(Debug, Clone)]
pub struct BeamConfig {
    /// Candidate continuations sampled per surviving branch, per step.
    pub candidates_per_step: usize,
    /// How many branches survive each step (`1` = greedy best-of-N).
    pub beam_width: usize,
    /// Hard cap on steps before the best branch so far is accepted.
    pub max_steps: usize,
    /// Guardrails applied to tool execution inside each explored step.
    pub tool_opts: ToolExecOptions,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self {
            candidates_per_step: 3,
            beam_width: 1,
            max_steps: super::strategy::DEFAULT_MAX_REACT_STEPS,
            tool_opts: ToolExecOptions::default(),
        }
    }
}

/// One live branch of the search.
struct BeamMember {
    bus: EventBus,
    /// Events this branch added on top of the shared history.
    events: Vec<Event>,
    score: f64,
    done: bool,
}

/// Explore the reasoning tree **one step at a time**, keeping the highest
/// scoring branches and pruning the rest.
///
/// Where [`best_of_n`] speculates over whole cycles, `beam_search` speculates
/// at every ReAct step: each surviving branch is forked
/// `candidates_per_step` ways, one step is run on each fork, all resulting
/// trajectories are scored, and only `beam_width` survive. Because a step
/// includes tool execution, this explores *actions*, not just wordings —
/// letting the agent try several tool strategies and keep whichever the
/// scorer prefers.
///
/// The winner's events are spliced onto `bus`; every pruned trajectory is
/// sealed as a rejected branch, so
/// [`NegativeAwareContextAssembler`](crate::NegativeAwareContextAssembler)
/// can feed the failures back into later turns.
///
/// **Cost warning**: this multiplies LLM calls by roughly
/// `beam_width × candidates_per_step` per step. Use a cheap model or a small
/// budget for exploration, and prefer `beam_width: 1` unless the scorer is
/// genuinely discriminating.
///
/// `agent_factory` must build an agent context bound to the bus it is given;
/// use it to vary temperature, model, or prompt across candidates (it
/// receives the candidate index).
pub async fn beam_search<F>(
    bus: &EventBus,
    config: &BeamConfig,
    scorer: &dyn BranchScorer,
    agent_factory: F,
) -> Result<SpeculationOutcome, AgentError>
where
    F: Fn(EventBus, usize) -> Agent + Send + Sync + Clone + 'static,
{
    if config.candidates_per_step == 0 || config.beam_width == 0 {
        return Err(AgentError::Tool(
            "beam_search: candidates_per_step and beam_width must be >= 1".into(),
        ));
    }

    let anchor = bus.log().await.last().map(|e| e.id);
    let baseline = bus.log_len().await;

    // The beam starts as a single branch: the current conversation.
    let mut beam: Vec<BeamMember> = vec![BeamMember {
        bus: bus.fork().await,
        events: Vec::new(),
        score: 0.0,
        done: false,
    }];
    let mut pruned: Vec<Vec<Event>> = Vec::new();

    for step in 1..=config.max_steps {
        if beam.iter().all(|m| m.done) {
            break;
        }

        let mut expanded: Vec<BeamMember> = Vec::new();
        for member in &beam {
            if member.done {
                // Finished branches stay in the running unchanged.
                expanded.push(BeamMember {
                    bus: member.bus.clone(),
                    events: member.events.clone(),
                    score: member.score,
                    done: true,
                });
                continue;
            }

            // Fan out: one fork per candidate continuation.
            let mut join_set: JoinSet<(EventBus, Vec<Event>, bool)> = JoinSet::new();
            for candidate in 0..config.candidates_per_step {
                let parent_bus = member.bus.clone();
                let factory = agent_factory.clone();
                let opts = config.tool_opts.clone();
                join_set.spawn(async move {
                    let fork = parent_bus.fork().await;
                    let from = fork.log_len().await;
                    let agent = factory(fork.clone(), candidate);
                    let outcome = agent.step(step, &opts).await;
                    let new_events = fork.log_since(from).await;
                    let done = !matches!(outcome, Ok(StepOutcome::Continue));
                    (fork, new_events, done)
                });
            }

            while let Some(joined) = join_set.join_next().await {
                match joined {
                    Ok((fork, new_events, done)) => {
                        let mut events = member.events.clone();
                        events.extend(new_events);
                        expanded.push(BeamMember {
                            bus: fork,
                            events,
                            score: 0.0,
                            done,
                        });
                    }
                    Err(e) => warn!("beam candidate panicked: {e}"),
                }
            }
        }

        if expanded.is_empty() {
            break;
        }

        // Score every expanded branch on its full trajectory so far.
        for member in &mut expanded {
            member.score = scorer.score(&member.events).await;
        }
        expanded.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Prune: everything past the beam width is sealed as rejected.
        let survivors: Vec<BeamMember> = expanded
            .drain(..config.beam_width.min(expanded.len()))
            .collect();
        for loser in expanded {
            if !loser.events.is_empty() {
                pruned.push(loser.events);
            }
        }
        debug!(
            step,
            survivors = survivors.len(),
            pruned = pruned.len(),
            best = survivors.first().map(|m| m.score).unwrap_or(0.0),
            "beam step complete"
        );
        beam = survivors;
    }

    let winner = beam
        .into_iter()
        .max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| AgentError::Tool("beam_search: no surviving branch".into()))?;

    // Splice the winner onto the main bus; seal the pruned trajectories.
    let winner_event_count = winner.events.len();
    let winner_score = winner.score;
    for event in winner.events {
        bus.publish(event).await?;
    }
    let mut rejected_branch_ids = Vec::new();
    for events in pruned {
        rejected_branch_ids.push(bus.adopt_rejected_branch(anchor, events).await);
    }

    info!(
        score = winner_score,
        events = winner_event_count,
        pruned = rejected_branch_ids.len(),
        "beam search complete"
    );

    bus.publish(Event::new(
        kinds::SPECULATION_COMPLETED,
        json!({
            "mode": "beam_search",
            "winner_name": "beam",
            "winner_index": 0,
            "scores": [winner_score],
            "beam_width": config.beam_width,
            "candidates_per_step": config.candidates_per_step,
            "baseline_events": baseline,
            "rejected_branches": rejected_branch_ids
                .iter()
                .map(|b| b.to_string())
                .collect::<Vec<_>>(),
        }),
    ))
    .await?;

    Ok(SpeculationOutcome {
        winner: 0,
        winner_name: "beam".to_string(),
        scores: vec![winner_score],
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

    #[tokio::test]
    async fn beam_search_keeps_best_branch_and_seals_the_rest() {
        use crate::agent::strategy::ToolExecOptions;

        let bus = EventBus::new();
        bus.publish(Event::new(kinds::USER_MESSAGE, json!({"text": "solve"})))
            .await
            .unwrap();
        let anchor = bus.log().await.last().unwrap().id;

        // Candidate index decides the reply length, so scoring is deterministic.
        let factory = |fork: EventBus, candidate: usize| {
            let reply = match candidate {
                0 => "short",
                1 => "a considerably longer answer",
                _ => "mid length",
            };
            AgentBuilder::new()
                .agent_id(format!("cand-{candidate}"))
                .bus(fork)
                .llm(MockLlmProvider::with_texts([reply]))
                .strategy(ReactStrategy::default())
                .build()
        };

        let scorer = FnScorer::new(|events: &[Event]| {
            events
                .iter()
                .rev()
                .find_map(|e| e.payload.get("content").and_then(|c| c.as_str()))
                .map(|c| c.len() as f64)
                .unwrap_or(0.0)
        });

        let config = BeamConfig {
            candidates_per_step: 3,
            beam_width: 1,
            max_steps: 3,
            tool_opts: ToolExecOptions::default(),
        };
        let outcome = beam_search(&bus, &config, &scorer, factory).await.unwrap();

        // Longest reply wins and lands on the active branch.
        let log = bus.log().await;
        let final_msg = log
            .iter()
            .filter(|e| e.kind == kinds::ASSISTANT_MESSAGE)
            .next_back()
            .expect("winner message");
        assert_eq!(
            final_msg.payload["content"].as_str().unwrap(),
            "a considerably longer answer"
        );
        assert!(outcome.scores[0] > 0.0);

        // The two pruned candidates are available as negative context.
        let rejected = bus.rejected_branches_from(anchor).await;
        assert_eq!(rejected.len(), 2, "losing branches must be sealed");
        assert!(log.iter().any(|e| e.kind == kinds::SPECULATION_COMPLETED));
    }

    #[tokio::test]
    async fn beam_search_rejects_zero_width() {
        use crate::agent::strategy::ToolExecOptions;
        let bus = EventBus::new();
        let config = BeamConfig {
            candidates_per_step: 0,
            beam_width: 1,
            max_steps: 1,
            tool_opts: ToolExecOptions::default(),
        };
        let err = beam_search(&bus, &config, &FnScorer::new(|_| 1.0), |fork, _| {
            AgentBuilder::new()
                .bus(fork)
                .llm(MockLlmProvider::with_texts(["x"]))
                .strategy(ReactStrategy::default())
                .build()
        })
        .await
        .unwrap_err();
        assert!(err.to_string().contains("must be >= 1"));
    }

    #[test]
    fn parses_judge_scores() {
        assert_eq!(first_number("8"), Some(8.0));
        assert_eq!(first_number("Score: 7.5/10"), Some(7.5));
        assert_eq!(first_number("I'd rate this 9 out of 10"), Some(9.0));
        assert_eq!(first_number("no digits"), None);
    }
}
