//! Token accounting: heuristic estimation plus self-calibration against
//! real provider usage.
//!
//! The estimator is a heuristic (words × 1.3 vs chars ÷ 4, whichever is
//! larger) because tokenizers are model-specific. [`TokenCalibration`] closes
//! the gap: every LLM response reports its true prompt-token count, and the
//! strategy records the estimate that produced it, so the ratio between them
//! can be learned online and applied to future estimates.
//!
//! Context assemblers calibrate themselves — they scan the events they are
//! already given for `(estimated, actual)` pairs, so no extra wiring is
//! needed.

use crate::event::{kinds, meta_keys, Event};
use crate::llm::types::ChatMessage;
use crate::llm::ContentPart;
use std::sync::atomic::{AtomicU64, Ordering};

/// Rough token cost charged for one image part.
///
/// Real cost depends on resolution and provider tiling; this is a
/// deliberately conservative flat estimate so images cannot silently blow a
/// context budget that only counted text.
pub const IMAGE_TOKEN_ESTIMATE: usize = 1_200;

/// Approximate token count for a string.
///
/// Uses the higher of two heuristics so both prose (word-heavy) and
/// JSON/code (character-heavy with short tokens) are reasonably bounded:
/// - words × 1.3  (good for natural language)
/// - chars ÷ 4    (good for dense JSON / code)
pub fn estimate_tokens(s: &str) -> usize {
    let by_words = ((s.split_whitespace().count() as f64) * 1.3).ceil() as usize + 4;
    let by_chars = s.len() / 4 + 4;
    by_words.max(by_chars)
}

/// Approximate token count for a full message list.
///
/// Text content, tool-call arguments, and image parts all contribute.
pub fn messages_token_count(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .map(|m| {
            let content = m.content.as_deref().map(estimate_tokens).unwrap_or(0);
            let images = m
                .parts
                .iter()
                .filter(|p| matches!(p, ContentPart::Image { .. }))
                .count()
                * IMAGE_TOKEN_ESTIMATE;
            let tool_calls: usize = m
                .tool_calls
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .map(|tc| {
                    estimate_tokens(&tc.function.arguments) + estimate_tokens(&tc.function.name) + 4
                })
                .sum();
            content + images + tool_calls + 4
        })
        .sum()
}

// ── Calibration ───────────────────────────────────────────────────────────────

/// Scale factor stored as parts-per-thousand so it fits in an atomic.
const SCALE: f64 = 1000.0;
/// Weight of each new sample in the exponentially-weighted moving average.
const EWMA_ALPHA: f64 = 0.3;
/// Clamp the learned ratio to a sane band so one anomalous response
/// (e.g. a cached-prompt hit reported oddly) cannot wreck the estimator.
const MIN_RATIO: f64 = 0.25;
const MAX_RATIO: f64 = 4.0;

/// Learns the ratio between estimated and actual prompt tokens.
///
/// Shared cheaply via `Arc`; all methods are lock-free.
///
/// ```
/// use eventage::agent::tokens::TokenCalibration;
///
/// let cal = TokenCalibration::new();
/// // Estimator said 1000, provider actually charged 1300.
/// cal.observe(1000, 1300);
/// assert!(cal.ratio() > 1.0);
/// assert!(cal.adjust(1000) > 1000);
/// ```
#[derive(Debug)]
pub struct TokenCalibration {
    /// Current ratio (actual ÷ estimated) × [`SCALE`].
    ratio_scaled: AtomicU64,
    samples: AtomicU64,
    /// ID of the last event folded in, so repeated assembles over the same
    /// log do not double-count the same sample.
    last_sample: AtomicU64,
}

impl Default for TokenCalibration {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenCalibration {
    pub fn new() -> Self {
        Self {
            ratio_scaled: AtomicU64::new(SCALE as u64),
            samples: AtomicU64::new(0),
            last_sample: AtomicU64::new(0),
        }
    }

    /// Fold one `(estimated, actual)` observation into the moving average.
    pub fn observe(&self, estimated: usize, actual: u32) {
        if estimated == 0 || actual == 0 {
            return;
        }
        let sample = (actual as f64 / estimated as f64).clamp(MIN_RATIO, MAX_RATIO);
        let current = self.ratio();
        let updated = if self.samples.load(Ordering::Relaxed) == 0 {
            sample
        } else {
            current * (1.0 - EWMA_ALPHA) + sample * EWMA_ALPHA
        };
        self.ratio_scaled
            .store((updated * SCALE) as u64, Ordering::Relaxed);
        self.samples.fetch_add(1, Ordering::Relaxed);
    }

    /// Current actual÷estimated ratio (`1.0` until the first observation).
    pub fn ratio(&self) -> f64 {
        self.ratio_scaled.load(Ordering::Relaxed) as f64 / SCALE
    }

    /// Number of observations folded in so far.
    pub fn samples(&self) -> u64 {
        self.samples.load(Ordering::Relaxed)
    }

    /// Apply the learned ratio to a raw estimate.
    pub fn adjust(&self, estimate: usize) -> usize {
        (estimate as f64 * self.ratio()).round() as usize
    }

    /// Scan `events` for the most recent `assistant.message` carrying both an
    /// estimate and a real prompt-token count, and fold it in.
    ///
    /// Cheap and idempotent: each event is only ever counted once.
    pub fn observe_events(&self, events: &[Event]) {
        let sample = events.iter().rev().find_map(|e| {
            if e.kind != kinds::ASSISTANT_MESSAGE {
                return None;
            }
            let estimated = e
                .metadata
                .get(meta_keys::LLM_ESTIMATED_INPUT_TOKENS)
                .and_then(|v| v.as_u64())?;
            let actual = e
                .metadata
                .get(meta_keys::LLM_INPUT_TOKENS)
                .and_then(|v| v.as_u64())
                .filter(|v| *v > 0)?;
            Some((e.id, estimated, actual))
        });

        let Some((id, estimated, actual)) = sample else {
            return;
        };
        // Use the low 64 bits of the event UUID as a cheap "seen" marker.
        let marker = id.as_u128() as u64;
        if self.last_sample.swap(marker, Ordering::Relaxed) == marker {
            return;
        }
        self.observe(estimated as usize, actual as u32);
    }

    /// Calibrated token count for a message list.
    pub fn count(&self, messages: &[ChatMessage]) -> usize {
        self.adjust(messages_token_count(messages))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn images_are_counted() {
        let text_only = vec![ChatMessage::user("hello")];
        let with_image = vec![ChatMessage::user_with_parts(vec![
            ContentPart::text("hello"),
            ContentPart::image_url("https://x/y.png"),
        ])];
        assert!(
            messages_token_count(&with_image)
                >= messages_token_count(&text_only) + IMAGE_TOKEN_ESTIMATE,
            "an image must add materially to the estimate"
        );
    }

    #[test]
    fn calibration_moves_toward_observed_ratio() {
        let cal = TokenCalibration::new();
        assert_eq!(cal.ratio(), 1.0);
        assert_eq!(cal.adjust(100), 100);

        // Provider consistently charges 50% more than we estimate.
        for _ in 0..20 {
            cal.observe(1000, 1500);
        }
        assert!(
            (cal.ratio() - 1.5).abs() < 0.05,
            "ratio should converge to 1.5, got {}",
            cal.ratio()
        );
        assert!(cal.adjust(1000) > 1400);
    }

    #[test]
    fn outliers_are_clamped() {
        let cal = TokenCalibration::new();
        cal.observe(10, 100_000); // absurd sample
        assert!(cal.ratio() <= MAX_RATIO);
        cal.observe(100_000, 1);
        assert!(cal.ratio() >= MIN_RATIO);
    }

    #[test]
    fn zero_values_are_ignored() {
        let cal = TokenCalibration::new();
        cal.observe(0, 100);
        cal.observe(100, 0);
        assert_eq!(cal.samples(), 0);
        assert_eq!(cal.ratio(), 1.0);
    }

    #[test]
    fn observe_events_is_idempotent() {
        let cal = TokenCalibration::new();
        let event = Event::new(kinds::ASSISTANT_MESSAGE, json!({"content": "x"}))
            .with_meta(meta_keys::LLM_ESTIMATED_INPUT_TOKENS, json!(1000))
            .with_meta(meta_keys::LLM_INPUT_TOKENS, json!(2000));
        let events = vec![event];

        cal.observe_events(&events);
        cal.observe_events(&events);
        cal.observe_events(&events);
        assert_eq!(cal.samples(), 1, "same event must not be counted twice");
        assert!((cal.ratio() - 2.0).abs() < 0.01);
    }
}
