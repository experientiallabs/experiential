//! The serving default of the first-token stall bound, clamped so it can fire.
//!
//! The relay's first-token bound is only useful while the request still has
//! budget to fail over: a stall the request deadline reaches first ends as a
//! terminal deadline failure, never as the failover-eligible first-token
//! timeout. The engine's own defaults put the request timeout at 120 s and
//! the first-token allowance at 120 s (a thinking model on a chat wire
//! answers its first token a minute or more after its headers), so unclamped
//! the default bound would never run. Hosts that grant long request budgets
//! (the platform: 1500 s) keep the full allowance.

use std::time::Duration;

/// The share of the request budget the first-token bound may consume before
/// the ladder must still have room to fail over.
const FIRST_TOKEN_SHARE_OF_REQUEST_BUDGET: f64 = 0.75;

/// The serving default for the first-token bound: the configured allowance,
/// never more than three quarters of the request budget, never zero.
pub(crate) fn first_token_bound(
    first_token_seconds: f64,
    request_timeout_seconds: f64,
) -> Duration {
    let cap = request_timeout_seconds.max(0.0) * FIRST_TOKEN_SHARE_OF_REQUEST_BUDGET;
    Duration::from_secs_f64(first_token_seconds.min(cap).max(0.001))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_request_budget_keeps_the_full_allowance() {
        assert_eq!(
            first_token_bound(120.0, 1500.0),
            Duration::from_secs_f64(120.0)
        );
    }

    #[test]
    fn the_engine_defaults_leave_room_to_fail_over() {
        // 120 s allowance under a 120 s request budget would never fire; the
        // clamp leaves a quarter of the budget for the fallback rung.
        assert_eq!(
            first_token_bound(120.0, 120.0),
            Duration::from_secs_f64(90.0)
        );
    }

    #[test]
    fn a_shorter_configured_allowance_is_honored_as_is() {
        assert_eq!(
            first_token_bound(15.0, 120.0),
            Duration::from_secs_f64(15.0)
        );
    }

    #[test]
    fn the_bound_never_collapses_to_zero() {
        assert_eq!(first_token_bound(0.0, 0.0), Duration::from_secs_f64(0.001));
    }
}
