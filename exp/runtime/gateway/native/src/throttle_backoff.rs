//! Backoff-and-redial schedule for provider throttles.
//!
//! A pool may author a `throttle_redial` schedule (the python
//! `GatewayThrottleRedialPolicy`); admission carries it to the data plane as
//! one of the frozen retry-policy facts, and marks per rung whether a
//! throttle there is worth waiting for on this request. The waterfall then
//! answers a pre-commit throttle on such a rung by waiting and re-dialing the
//! SAME deployment (its warm prompt cache is worth the wait) before the
//! ladder advances, and a throttle reaches the caller only once every rung is
//! exhausted, carrying the largest `Retry-After` any rung stated.
//!
//! This module owns the pure wait decision so the schedule, the
//! `Retry-After` precedence, and the deadline and first-byte caps are
//! testable without a provider or a control plane.

use std::hash::{BuildHasher, RandomState};
use std::time::Duration;

use serde::Deserialize;

use crate::errors::{Failure, FailureClass};

/// The pool's frozen backoff-and-redial schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct ThrottleRedial {
    /// Redials of a throttled rung after its throttled dispatch, per rung.
    pub max_attempts: u32,
    /// Wait before the first redial; each later redial doubles it.
    pub base_delay_ms: u64,
    /// Ceiling on any single wait, including a provider `Retry-After`.
    pub max_delay_ms: u64,
}

/// Everything the wait decision needs about one throttled attempt.
#[derive(Debug, Clone, Copy)]
pub struct BackoffQuery {
    pub schedule: ThrottleRedial,
    /// Redials already made on this rung (the throttled dispatch excluded).
    pub redials_so_far: u32,
    /// The provider's parsed `Retry-After`, when it stated one.
    pub retry_after_seconds: Option<u32>,
    /// What the request deadline leaves right now.
    pub remaining_deadline: Duration,
    /// The rung's first-byte allowance for one attempt.
    pub first_byte_allowance: Duration,
    /// A unit-interval draw spreading the exponential wait.
    pub jitter_unit: f64,
}

impl BackoffQuery {
    /// The wait before re-dialing the throttled rung, or `None` when the
    /// ladder should advance instead.
    ///
    /// The exponential wait is `base * 2^n` capped at the schedule ceiling
    /// and equal-jittered into `[wait / 2, wait]`. A provider `Retry-After`
    /// within the ceiling is a floor on that wait (the provider knows its
    /// window); one above the ceiling means the rung is out for longer than
    /// the pool is willing to wait, so no redial happens. A wait never
    /// exceeds the rung's first-byte allowance (the gateway waits no longer
    /// for a throttled rung than it would for a silent one) and must leave
    /// the redial its own first-byte allowance under the request deadline.
    pub fn delay(&self) -> Option<Duration> {
        if self.redials_so_far >= self.schedule.max_attempts {
            return None;
        }
        let ceiling = Duration::from_millis(self.schedule.max_delay_ms);
        let base = Duration::from_millis(self.schedule.base_delay_ms);
        let exponential = base
            .checked_mul(1u32.checked_shl(self.redials_so_far).unwrap_or(u32::MAX))
            .unwrap_or(ceiling)
            .min(ceiling);
        let half = exponential / 2;
        let jittered = half + half.mul_f64(self.jitter_unit.clamp(0.0, 1.0));
        let wait = match self.retry_after_seconds {
            Some(seconds) => {
                let stated = Duration::from_secs(u64::from(seconds));
                if stated > ceiling {
                    return None;
                }
                stated.max(jittered)
            }
            None => jittered,
        };
        if wait > self.first_byte_allowance {
            return None;
        }
        if wait + self.first_byte_allowance > self.remaining_deadline {
            return None;
        }
        Some(wait)
    }
}

/// A unit-interval jitter draw for one redial, seeded per process and spread
/// by the request and its attempt ordinal so concurrent throttled requests
/// do not re-dial in lockstep.
pub fn jitter_unit(request_id: &str, attempt_ordinal: u32) -> f64 {
    let hash = RandomState::new().hash_one((request_id, attempt_ordinal));
    // The top 53 bits map exactly onto the f64 mantissa range [0, 1).
    (hash >> 11) as f64 / (1u64 << 53) as f64
}

/// Remember the largest `Retry-After` any throttled attempt stated.
pub fn track_retry_after(largest: &mut Option<u32>, failure: &Failure) {
    if failure.failure_class != FailureClass::Throttled {
        return;
    }
    if let Some(seconds) = failure.retry_after_seconds {
        *largest = Some(largest.map_or(seconds, |seen| seen.max(seconds)));
    }
}

/// Floor a throttled exhaustion's `Retry-After` at the largest one seen on
/// the ladder, so the caller is told the longest wait any rung asked for
/// rather than only the last rung's.
pub fn with_largest_retry_after(mut failure: Failure, largest: Option<u32>) -> Failure {
    if failure.failure_class != FailureClass::Throttled {
        return failure;
    }
    if let Some(seen) = largest {
        failure.retry_after_seconds = Some(
            failure
                .retry_after_seconds
                .map_or(seen, |own| own.max(seen)),
        );
    }
    failure
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEDULE: ThrottleRedial = ThrottleRedial {
        max_attempts: 3,
        base_delay_ms: 500,
        max_delay_ms: 8_000,
    };

    fn query(redials_so_far: u32, retry_after_seconds: Option<u32>) -> BackoffQuery {
        BackoffQuery {
            schedule: SCHEDULE,
            redials_so_far,
            retry_after_seconds,
            remaining_deadline: Duration::from_secs(300),
            first_byte_allowance: Duration::from_secs(15),
            // The top of the jitter band, so the exponential is exact.
            jitter_unit: 1.0,
        }
    }

    #[test]
    fn exponential_schedule_doubles_from_the_base_and_caps_at_the_ceiling() {
        assert_eq!(query(0, None).delay(), Some(Duration::from_millis(500)));
        assert_eq!(query(1, None).delay(), Some(Duration::from_millis(1_000)));
        assert_eq!(query(2, None).delay(), Some(Duration::from_millis(2_000)));
        // The per-rung redial cap ends the redials; the ladder advances.
        assert_eq!(query(3, None).delay(), None);
        let long = BackoffQuery {
            schedule: ThrottleRedial {
                max_attempts: 6,
                base_delay_ms: 3_000,
                max_delay_ms: 8_000,
            },
            ..query(5, None)
        };
        assert_eq!(long.delay(), Some(Duration::from_millis(8_000)));
    }

    #[test]
    fn jitter_spreads_the_exponential_over_its_upper_half_only() {
        let low = BackoffQuery {
            jitter_unit: 0.0,
            ..query(1, None)
        };
        assert_eq!(low.delay(), Some(Duration::from_millis(500)));
        let middle = BackoffQuery {
            jitter_unit: 0.5,
            ..query(1, None)
        };
        assert_eq!(middle.delay(), Some(Duration::from_millis(750)));
        // Out-of-range draws clamp instead of escaping the band.
        let wild = BackoffQuery {
            jitter_unit: 7.0,
            ..query(1, None)
        };
        assert_eq!(wild.delay(), Some(Duration::from_millis(1_000)));
    }

    #[test]
    fn retry_after_within_the_ceiling_floors_the_wait_and_above_it_declines() {
        // The provider's stated wait wins over a shorter exponential.
        assert_eq!(query(0, Some(3)).delay(), Some(Duration::from_secs(3)));
        // A longer exponential wins over a shorter stated wait.
        let late = BackoffQuery {
            schedule: ThrottleRedial {
                max_attempts: 6,
                base_delay_ms: 4_000,
                max_delay_ms: 8_000,
            },
            ..query(1, Some(3))
        };
        assert_eq!(late.delay(), Some(Duration::from_secs(8)));
        // Exactly the ceiling is still honored.
        assert_eq!(query(0, Some(8)).delay(), Some(Duration::from_secs(8)));
        // Above the ceiling the rung is out longer than the pool waits.
        assert_eq!(query(0, Some(9)).delay(), None);
    }

    #[test]
    fn the_wait_is_capped_by_the_first_byte_allowance_and_the_deadline() {
        // A wait longer than the rung's first-byte allowance is declined.
        let short_allowance = BackoffQuery {
            first_byte_allowance: Duration::from_millis(400),
            ..query(0, None)
        };
        assert_eq!(short_allowance.delay(), None);
        // The wait plus the redial's own first-byte allowance must fit.
        let tight = BackoffQuery {
            remaining_deadline: Duration::from_millis(15_400),
            ..query(0, None)
        };
        assert_eq!(tight.delay(), None);
        let fits = BackoffQuery {
            remaining_deadline: Duration::from_millis(15_500),
            ..query(0, None)
        };
        assert_eq!(fits.delay(), Some(Duration::from_millis(500)));
        // An expired deadline never waits.
        let expired = BackoffQuery {
            remaining_deadline: Duration::ZERO,
            ..query(0, None)
        };
        assert_eq!(expired.delay(), None);
    }

    #[test]
    fn jitter_draws_stay_in_the_unit_interval_and_vary_by_ordinal() {
        let first = jitter_unit("request", 1);
        let second = jitter_unit("request", 2);
        for draw in [first, second] {
            assert!((0.0..1.0).contains(&draw));
        }
        assert_ne!(first, second);
    }

    #[test]
    fn the_largest_retry_after_seen_floors_a_throttled_exhaustion() {
        let mut largest = None;
        track_retry_after(
            &mut largest,
            &Failure::new(FailureClass::ProviderInternal, "boom"),
        );
        assert_eq!(largest, None);
        let mut short = Failure::new(FailureClass::Throttled, "slow");
        short.retry_after_seconds = Some(7);
        track_retry_after(&mut largest, &short);
        let mut long = Failure::new(FailureClass::Throttled, "slow");
        long.retry_after_seconds = Some(30);
        track_retry_after(&mut largest, &long);
        track_retry_after(&mut largest, &short);
        assert_eq!(largest, Some(30));
        // The exhausting throttle advertises the longest wait any rung asked for.
        let exhausted = with_largest_retry_after(short.clone(), largest);
        assert_eq!(exhausted.retry_after_seconds, Some(30));
        // A rung asking for even longer keeps its own.
        let mut longest = Failure::new(FailureClass::Throttled, "slow");
        longest.retry_after_seconds = Some(45);
        assert_eq!(
            with_largest_retry_after(longest, largest).retry_after_seconds,
            Some(45)
        );
        // A non-throttle exhaustion is untouched.
        let internal = Failure::new(FailureClass::ProviderInternal, "boom");
        assert_eq!(
            with_largest_retry_after(internal, largest).retry_after_seconds,
            None
        );
    }
}
