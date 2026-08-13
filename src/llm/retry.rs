//! Automatic retry with exponential backoff for transient LLM failures.

use super::error::LlmError;
use super::provider::LlmProvider;
use super::types::{ChatMessage, LlmResponse, ToolDefinition};
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tracing::warn;

/// Wraps any [`LlmProvider`] and retries transient failures with exponential
/// backoff and jitter.
///
/// Retried errors:
/// - HTTP transport failures (timeouts, connection resets)
/// - API status `408`, `429`, `5xx`, and `529` (provider overloaded)
///
/// Non-transient errors (`400` schema errors, auth failures, ...) are returned
/// immediately so the agent loop can surface them.
///
/// Composes with [`RateLimitedProvider`](super::RateLimitedProvider) — put the
/// rate limiter *inside* the retrier so retries are also paced:
///
/// ```rust,no_run
/// use eventage::llm::{OpenAiProvider, RateLimitedProvider, RetryProvider};
///
/// let provider = RetryProvider::new(
///     RateLimitedProvider::new(OpenAiProvider::openai("sk-...", "gpt-5-mini"), 60),
/// );
/// ```
pub struct RetryProvider {
    inner: Arc<dyn LlmProvider>,
    /// Maximum retry attempts after the initial call (default 4).
    max_retries: u32,
    /// Backoff for the first retry; doubles per attempt (default 1s).
    base_delay: Duration,
    /// Upper bound for a single backoff sleep (default 30s).
    max_delay: Duration,
}

impl RetryProvider {
    pub fn new(inner: impl LlmProvider + 'static) -> Self {
        Self::from_arc(Arc::new(inner))
    }

    pub fn from_arc(inner: Arc<dyn LlmProvider>) -> Self {
        Self {
            inner,
            max_retries: 4,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
        }
    }

    /// Sets the maximum number of retries after the initial attempt.
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Sets the base backoff delay (doubles per retry, capped at `max_delay`).
    pub fn with_base_delay(mut self, delay: Duration) -> Self {
        self.base_delay = delay;
        self
    }

    /// Sets the upper bound for a single backoff sleep.
    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    fn is_transient(error: &LlmError) -> bool {
        match error {
            LlmError::Http(e) => e.is_timeout() || e.is_connect() || e.is_request(),
            LlmError::Api { status, .. } => {
                matches!(status, 408 | 429 | 529) || (500..600).contains(status)
            }
            _ => false,
        }
    }

    fn backoff(&self, attempt: u32) -> Duration {
        let exp = self
            .base_delay
            .saturating_mul(2u32.saturating_pow(attempt))
            .min(self.max_delay);
        // Add up to +25% jitter (derived from the clock — no rand dependency)
        // so a fleet of agents doesn't retry in lockstep.
        let jitter_pct = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
            % 250) as u64;
        exp + exp.mul_f64(jitter_pct as f64 / 1000.0)
    }
}

#[async_trait]
impl LlmProvider for RetryProvider {
    async fn complete(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
    ) -> Result<LlmResponse, LlmError> {
        let mut attempt: u32 = 0;
        loop {
            match self.inner.complete(messages.clone(), tools.clone()).await {
                Ok(resp) => return Ok(resp),
                Err(e) if attempt < self.max_retries && Self::is_transient(&e) => {
                    let delay = self.backoff(attempt);
                    attempt += 1;
                    warn!(
                        attempt,
                        max_retries = self.max_retries,
                        delay_ms = delay.as_millis() as u64,
                        error = %e,
                        "transient LLM error — retrying"
                    );
                    sleep(delay).await;
                }
                Err(e) => return Err(e),
            }
        }
    }

    async fn complete_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
        on_delta: super::types::DeltaHandler,
    ) -> Result<LlmResponse, LlmError> {
        let mut attempt: u32 = 0;
        loop {
            match self
                .inner
                .complete_stream(messages.clone(), tools.clone(), on_delta.clone())
                .await
            {
                Ok(resp) => return Ok(resp),
                Err(e) if attempt < self.max_retries && Self::is_transient(&e) => {
                    let delay = self.backoff(attempt);
                    attempt += 1;
                    warn!(
                        attempt,
                        max_retries = self.max_retries,
                        delay_ms = delay.as_millis() as u64,
                        error = %e,
                        "transient LLM error — retrying stream"
                    );
                    sleep(delay).await;
                }
                Err(e) => return Err(e),
            }
        }
    }

    fn model(&self) -> &str {
        self.inner.model()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FlakyProvider {
        calls: AtomicUsize,
        fail_first: usize,
        status: u16,
    }

    #[async_trait]
    impl LlmProvider for FlakyProvider {
        async fn complete(
            &self,
            _messages: Vec<ChatMessage>,
            _tools: Vec<ToolDefinition>,
        ) -> Result<LlmResponse, LlmError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_first {
                Err(LlmError::Api {
                    status: self.status,
                    body: "overloaded".into(),
                })
            } else {
                Ok(LlmResponse {
                    content: Some("ok".into()),
                    finish_reason: "stop".into(),
                    ..Default::default()
                })
            }
        }

        fn model(&self) -> &str {
            "flaky"
        }
    }

    #[tokio::test]
    async fn retries_transient_errors_then_succeeds() {
        let provider = RetryProvider::new(FlakyProvider {
            calls: AtomicUsize::new(0),
            fail_first: 2,
            status: 429,
        })
        .with_base_delay(Duration::from_millis(1));

        let resp = provider.complete(vec![], vec![]).await.unwrap();
        assert_eq!(resp.content.as_deref(), Some("ok"));
    }

    #[tokio::test]
    async fn does_not_retry_client_errors() {
        let provider = RetryProvider::new(FlakyProvider {
            calls: AtomicUsize::new(0),
            fail_first: 10,
            status: 400,
        })
        .with_base_delay(Duration::from_millis(1));

        let err = provider.complete(vec![], vec![]).await.unwrap_err();
        assert!(matches!(err, LlmError::Api { status: 400, .. }));
    }

    #[tokio::test]
    async fn gives_up_after_max_retries() {
        let provider = RetryProvider::new(FlakyProvider {
            calls: AtomicUsize::new(0),
            fail_first: 10,
            status: 503,
        })
        .with_max_retries(2)
        .with_base_delay(Duration::from_millis(1));

        let err = provider.complete(vec![], vec![]).await.unwrap_err();
        assert!(matches!(err, LlmError::Api { status: 503, .. }));
    }
}
