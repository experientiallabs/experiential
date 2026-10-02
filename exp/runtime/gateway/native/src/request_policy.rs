//! Typed request-owned dispatch budgets and discretionary retry waits.

use crate::throttle_backoff::jitter_unit;
use serde::Deserialize;
use std::time::Duration;

/// A caller wait override. No variant changes retry or failover eligibility.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Backoff {
    None,
    Exponential {
        base_delay_ms: u64,
        max_delay_ms: u64,
        multiplier: f64,
    },
}

impl Backoff {
    /// Validate the bridge boundary as strictly as the public request decoder.
    pub fn valid(self) -> bool {
        match self {
            Self::None => true,
            Self::Exponential {
                base_delay_ms,
                max_delay_ms,
                multiplier,
            } => {
                (1..=10_000).contains(&base_delay_ms)
                    && (base_delay_ms..=60_000).contains(&max_delay_ms)
                    && multiplier.is_finite()
                    && (1.0..=4.0).contains(&multiplier)
            }
        }
    }

    /// Choose an equal-jittered wait, respecting the provider floor and deadline.
    /// A declined floor means no same-route redial, never an immediate hammer.
    pub fn delay(
        self,
        ordinal: u32,
        retry_after: Option<u32>,
        remaining: Duration,
        request_id: &str,
    ) -> Option<Duration> {
        self.delay_with_jitter(
            ordinal,
            retry_after,
            remaining,
            jitter_unit(request_id, ordinal),
        )
    }

    fn delay_with_jitter(
        self,
        ordinal: u32,
        retry_after: Option<u32>,
        remaining: Duration,
        jitter: f64,
    ) -> Option<Duration> {
        let floor = Duration::from_secs(u64::from(retry_after.unwrap_or(0)));
        let delay = match self {
            Self::None => {
                if !floor.is_zero() {
                    return None;
                }
                Duration::ZERO
            }
            Self::Exponential {
                base_delay_ms,
                max_delay_ms,
                multiplier,
            } => {
                let ceiling = Duration::from_millis(max_delay_ms);
                if floor > ceiling {
                    return None;
                }
                let exponential = ((base_delay_ms as f64) * multiplier.powf(ordinal as f64))
                    .min(max_delay_ms as f64);
                let jittered = Duration::from_secs_f64(
                    exponential / 1000.0 * (0.5 + jitter.clamp(0.0, 1.0) * 0.5),
                );
                floor.max(jittered)
            }
        };
        // Leave time to make progress instead of sleeping out the deadline.
        (delay < remaining).then_some(delay)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_bounds_and_provider_floor_are_hard() {
        let backoff = Backoff::Exponential {
            base_delay_ms: 500,
            max_delay_ms: 8000,
            multiplier: 3.0,
        };
        assert!(backoff.valid());
        let remaining = Duration::from_secs(60);
        assert_eq!(
            backoff.delay_with_jitter(1, None, remaining, 0.0),
            Some(Duration::from_millis(750))
        );
        assert_eq!(
            backoff.delay_with_jitter(1, None, remaining, 1.0),
            Some(Duration::from_millis(1500))
        );
        assert_eq!(
            backoff.delay_with_jitter(1, Some(2), remaining, 0.0),
            Some(Duration::from_secs(2))
        );
        assert_eq!(backoff.delay_with_jitter(1, Some(9), remaining, 0.0), None);
        assert_eq!(
            Backoff::None.delay_with_jitter(0, Some(2), Duration::from_secs(2), 0.0),
            None
        );
        assert_eq!(
            Backoff::None.delay_with_jitter(0, Some(2), remaining, 0.0),
            None
        );
    }
}
