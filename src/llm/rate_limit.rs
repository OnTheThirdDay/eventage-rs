use super::error::LlmError;
use super::provider::LlmProvider;
use super::types::{ChatMessage, LlmResponse, ToolDefinition};
use async_trait::async_trait;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio::time::sleep;
use tracing::debug;

/// Wraps any [`LlmProvider`] and throttles requests to a configurable rate.
///
/// Uses a **slot-reservation** strategy: each call reserves the next available
/// time slot and sleeps until that slot arrives.  No requests are ever dropped;
/// they are merely queued and dispatched at the configured pace.
///
/// # Example
///
/// ```rust,no_run
/// use eventage::llm::{OpenAiProvider, RateLimitedProvider};
///
/// // Throttle to 15 requests per minute (matches Gemini free-tier Flash limit)
/// let provider = RateLimitedProvider::new(
///     OpenAiProvider::openai("sk-...", "gpt-4o-mini"),
///     15,
/// );
/// ```
pub struct RateLimitedProvider {
    inner: Arc<dyn LlmProvider>,
    /// Minimum gap between successive requests.
    interval: Duration,
    /// Earliest instant at which the next request may be dispatched.
    next_slot: Mutex<Instant>,
}

impl RateLimitedProvider {
    /// Create a new throttled provider.
    ///
    /// `requests_per_minute` is the maximum sustained call rate.  A value of
    /// `0` is treated as unlimited (no throttling applied).
    pub fn new(inner: impl LlmProvider + 'static, requests_per_minute: u32) -> Self {
        let interval = if requests_per_minute == 0 {
            Duration::ZERO
        } else {
            Duration::from_millis(60_000 / requests_per_minute as u64)
        };
        Self {
            inner: Arc::new(inner),
            interval,
            next_slot: Mutex::new(Instant::now()),
        }
    }

    /// Build from an already-`Arc`-ed provider.
    pub fn from_arc(inner: Arc<dyn LlmProvider>, requests_per_minute: u32) -> Self {
        let interval = if requests_per_minute == 0 {
            Duration::ZERO
        } else {
            Duration::from_millis(60_000 / requests_per_minute as u64)
        };
        Self {
            inner,
            interval,
            next_slot: Mutex::new(Instant::now()),
        }
    }

    /// Reserve a slot and return the duration to sleep before the request fires.
    async fn reserve_slot(&self) -> Duration {
        if self.interval.is_zero() {
            return Duration::ZERO;
        }
        let mut slot = self.next_slot.lock().await;
        let now = Instant::now();
        let sleep_for = slot.saturating_duration_since(now);
        // Advance the slot by one interval from whichever is later: now or current slot.
        *slot = now.max(*slot) + self.interval;
        sleep_for
    }
}

#[async_trait]
impl LlmProvider for RateLimitedProvider {
    async fn complete(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
    ) -> Result<LlmResponse, LlmError> {
        let wait = self.reserve_slot().await;
        if !wait.is_zero() {
            debug!(
                wait_ms = wait.as_millis(),
                model = self.inner.model(),
                "rate limiter: waiting before LLM request"
            );
            sleep(wait).await;
        }
        self.inner.complete(messages, tools).await
    }

    fn model(&self) -> &str {
        self.inner.model()
    }
}
