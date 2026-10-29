//! Host-authored output pacing: a streamed answer released no faster than a rate.
//!
//! The control plane's `start_attempt` reply may carry
//! `output_tokens_per_second` for the attempt it reserved. It does so only on
//! a self-hosted rung (`upstream_priority`), only for a request whose host set
//! a class pace, and, when the host set a saturation threshold, only while
//! that rung's in-flight load is at or above it
//! (`exp/runtime/gateway/class_throttles.py`). Without it nothing here runs:
//! `OutputPacer::new(None)` is `None` and the stream loops never sleep. With
//! it, a token bucket (one second of burst, starting full) delays each released
//! event until the caller's stream stays at or under the rate.
//!
//! Pacing only delays delivery: the GPU still generates at full speed, so it
//! frees no capacity by itself. The vLLM priority and the per-class in-flight
//! cap are the levers that do.
//!
//! Tokens are estimated as visible bytes over four, the approximation the data
//! plane already uses for reasoning and input estimates; the gateway holds no
//! tokenizer. Text, refusal, tool-argument and reasoning deltas count; framing,
//! usage and terminals cost nothing, so the final frames are never delayed.
//! Pacing only delays delivery: it never changes a byte, a count, or the bill.

use std::time::{Duration, Instant};

use crate::events::Event;

/// Bytes per estimated token, matching `reasoning_estimate`.
const BYTES_PER_TOKEN: u64 = 4;

/// A token bucket over estimated output tokens.
#[derive(Debug, Clone)]
pub(crate) struct OutputPacer {
    rate: f64,
    burst: f64,
    available: f64,
    refilled: Instant,
}

impl OutputPacer {
    /// A pacer at `tokens_per_second`, or `None` (never pace) when the rate
    /// is absent, non-finite or not positive.
    pub(crate) fn new(tokens_per_second: Option<f64>) -> Option<Self> {
        let rate = tokens_per_second.filter(|rate| rate.is_finite() && *rate > 0.0)?;
        let burst = rate.max(1.0);
        Some(Self {
            rate,
            burst,
            available: burst,
            refilled: Instant::now(),
        })
    }

    /// Spend `tokens` at `now` and return how long the caller must wait
    /// before releasing them. A debt carries over, so the long-run rate holds
    /// even when one delta is larger than the burst.
    pub(crate) fn delay(&mut self, tokens: u64, now: Instant) -> Duration {
        let elapsed = now.saturating_duration_since(self.refilled).as_secs_f64();
        self.refilled = now;
        self.available = (self.available + elapsed * self.rate).min(self.burst);
        self.available -= tokens as f64;
        if self.available >= 0.0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64(-self.available / self.rate)
        }
    }
}

/// The estimated output tokens one released event carries.
pub(crate) fn event_output_tokens(event: &Event) -> u64 {
    let text = match event {
        Event::TextDelta(text) | Event::RefusalDelta(text) | Event::ReasoningTextDelta(text) => {
            text.as_str()
        }
        Event::ProviderTextDelta { delta, .. }
        | Event::ProviderRefusalDelta { delta, .. }
        | Event::ToolArgumentsDelta { delta, .. }
        | Event::ReasoningSummaryDelta { delta, .. }
        | Event::ThinkingDelta { delta, .. }
        | Event::ReasoningContentDelta { delta, .. } => delta.as_str(),
        _ => return 0,
    };
    (text.len() as u64).div_ceil(BYTES_PER_TOKEN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_or_invalid_rate_never_paces() {
        for rate in [
            None,
            Some(0.0),
            Some(-5.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
        ] {
            assert!(OutputPacer::new(rate).is_none(), "{rate:?}");
        }
    }

    #[test]
    fn burst_is_free_and_debt_waits_at_the_rate() {
        let mut pacer = OutputPacer::new(Some(10.0)).unwrap();
        let start = pacer.refilled;
        // One second of burst (10 tokens) releases at once.
        assert_eq!(pacer.delay(10, start), Duration::ZERO);
        // The next 5 tokens owe half a second.
        assert_eq!(pacer.delay(5, start), Duration::from_millis(500));
        // After that half second elapses the debt is paid; 5 more owe another half.
        let later = start + Duration::from_millis(500);
        assert_eq!(pacer.delay(5, later), Duration::from_millis(500));
    }

    #[test]
    fn idle_time_refills_only_up_to_the_burst() {
        let mut pacer = OutputPacer::new(Some(10.0)).unwrap();
        let start = pacer.refilled;
        assert_eq!(pacer.delay(10, start), Duration::ZERO);
        let idle = start + Duration::from_secs(60);
        assert_eq!(pacer.delay(10, idle), Duration::ZERO);
        assert_eq!(pacer.delay(10, idle), Duration::from_secs(1));
    }

    #[test]
    fn sustained_stream_converges_to_the_rate() {
        let mut pacer = OutputPacer::new(Some(50.0)).unwrap();
        let start = pacer.refilled;
        let mut now = start;
        // 1,000 one-token deltas: the first 50 are burst, the rest wait 20 ms each.
        for _ in 0..1_000 {
            now += pacer.delay(1, now);
        }
        let total = now.duration_since(start).as_secs_f64();
        assert!((total - 19.0).abs() < 0.01, "{total}");
    }

    #[test]
    fn only_visible_and_reasoning_deltas_cost_tokens() {
        assert_eq!(event_output_tokens(&Event::TextDelta("abcdefgh".into())), 2);
        assert_eq!(
            event_output_tokens(&Event::TextDelta("abcdefghi".into())),
            3
        );
        assert_eq!(
            event_output_tokens(&Event::ReasoningTextDelta("abcd".into())),
            1
        );
        assert_eq!(
            event_output_tokens(&Event::ThinkingDelta {
                index: 0,
                delta: "abcd".into()
            }),
            1
        );
        assert_eq!(event_output_tokens(&Event::Completed), 0);
        assert_eq!(event_output_tokens(&Event::TextDelta(String::new())), 0);
    }
}
