//! [`AgentSet`] — run multiple agents concurrently.

use eventage_agent::agent::Agent;
use eventage_agent::error::AgentError;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::warn;

/// Runs multiple agents concurrently on a shared event bus.
///
/// All agents share the same [`eventage_core::EventBus`] and receive every
/// event published to it. Each agent independently decides whether to act
/// based on its own `agent_id` routing and event kind filters.
///
/// # Example
/// ```no_run
/// use eventage_agent::AgentBuilder;
/// use eventage_provided_impl::AgentSet;
/// use eventage_core::EventBus;
/// use eventage_llm::MockLlmProvider;
/// # async fn example() {
/// let bus = EventBus::default();
/// let orchestrator = AgentBuilder::new()
///     .agent_id("orchestrator")
///     .bus(bus.clone())
///     .llm(MockLlmProvider::with_texts(Vec::<&str>::new()))
///     .strategy(eventage_provided_impl::ReactStrategy::default())
///     .build();
/// let worker = AgentBuilder::new()
///     .agent_id("worker")
///     .bus(bus.clone())
///     .llm(MockLlmProvider::with_texts(Vec::<&str>::new()))
///     .strategy(eventage_provided_impl::ReactStrategy::default())
///     .build();
///
/// AgentSet::new()
///     .add_agent(orchestrator)
///     .add_agent(worker)
///     .run_until_all_complete()
///     .await
///     .unwrap();
/// # }
/// ```
pub struct AgentSet {
    agents: Vec<Arc<Agent>>,
    /// If set, at most this many agents may be executing a cycle simultaneously.
    max_concurrent: Option<usize>,
}

impl AgentSet {
    pub fn new() -> Self {
        Self {
            agents: vec![],
            max_concurrent: None,
        }
    }

    pub fn add_agent(mut self, agent: Agent) -> Self {
        self.agents.push(Arc::new(agent));
        self
    }

    pub fn add_agent_arc(mut self, agent: Arc<Agent>) -> Self {
        self.agents.push(agent);
        self
    }

    /// Limit how many agents may execute a reasoning cycle at the same time.
    pub fn max_concurrent(mut self, n: usize) -> Self {
        self.max_concurrent = Some(n.max(1));
        self
    }

    /// Spawn all agents concurrently and wait until every agent's `run()` loop exits.
    pub async fn run_until_all_complete(self) -> Result<(), AgentError> {
        let sem: Option<Arc<Semaphore>> = self.max_concurrent.map(|n| Arc::new(Semaphore::new(n)));

        let mut set: JoinSet<Result<(), AgentError>> = JoinSet::new();

        for agent in self.agents {
            let sem = sem.clone();
            set.spawn(async move {
                match sem {
                    Some(s) => {
                        let mut rx = agent.bus().subscribe();
                        while let Some(event) = rx.recv().await {
                            use eventage_core::{kinds, meta_keys};
                            let wake = match event.kind.as_str() {
                                kinds::USER_MESSAGE | kinds::SYSTEM_HEARTBEAT => true,
                                kinds::AGENT_MESSAGE => event
                                    .metadata
                                    .get(meta_keys::TO_AGENT_ID)
                                    .and_then(|v| v.as_str())
                                    .is_none_or(|to| to == agent.agent_id),
                                _ => false,
                            };
                            if wake {
                                let _permit = s.acquire().await.unwrap();
                                agent.cycle().await?;
                            }
                        }
                        Ok(())
                    }
                    None => agent.run().await,
                }
            });
        }

        let mut first_err: Option<AgentError> = None;
        while let Some(result) = set.join_next().await {
            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    warn!("agent exited with error: {e}");
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                    set.abort_all();
                }
                Err(join_err) => {
                    warn!("agent task panicked: {join_err}");
                }
            }
        }

        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

impl Default for AgentSet {
    fn default() -> Self {
        Self::new()
    }
}
