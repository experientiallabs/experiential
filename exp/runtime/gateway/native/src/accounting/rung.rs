//! Dispatch-time local admission and output pacing.

use super::loads::{Loads, Reservation, Shed};
use super::route_config::{Lane, Registration};
use super::selection::FailoverMode;
use super::sticky::Sticky;

#[allow(clippy::too_many_arguments)]
pub(super) fn reserve(
    loads: &mut Loads,
    sticky: &mut Sticky,
    config: &Registration,
    lane: &Lane,
    input_tokens: u64,
    force: bool,
    rate_retry: bool,
    failover: bool,
    first: bool,
    admitted: f64,
    now: f64,
) -> Result<Option<String>, Shed> {
    let p = &lane.policy;
    let fast = lane.fast(config);
    let class_cap = config
        .class
        .zip(config.class_cap)
        .filter(|_| p.self_hosted && !fast);
    let authored = p.concurrency_bound.is_some()
        || p.requests_per_minute.is_some()
        || p.tokens_per_minute.is_some()
        || p.fleet.is_some();
    if !authored && loads.default_bound.is_none() && class_cap.is_none() {
        return Ok(None);
    }
    let default = p.concurrency_bound.is_none();
    let bound = p.concurrency_bound.or(loads.default_bound);
    let fresh_fraction = p.fresh_session_spill_fraction.filter(|_| {
        lane.stage.failover_mode == FailoverMode::MaximizeCacheAffinity
            && (config.fingerprint.is_some() || config.caching_off)
    });
    let mut warm = !config.caching_off;
    if let Some(fingerprint) = config
        .fingerprint
        .as_ref()
        .filter(|_| fresh_fraction.is_some() && !lane.cache_placed)
    {
        warm = if config.recovery_scoped {
            config.verified_warm_deployment.as_ref() == Some(&lane.key.1)
                && now < admitted + config.verified_warm_seconds
        } else {
            sticky.bound(fingerprint.as_bytes(), now) == Some(lane.key.1.as_str())
        };
    }
    let refusing = default || p.refusing;
    let factor = match config.priority {
        1 => {
            if default {
                1.25
            } else {
                p.priority_overflow_paying
                    .unwrap_or(1.25)
                    .min(p.priority_overflow_pro.unwrap_or(1.5))
            }
        }
        2 => {
            if default {
                1.5
            } else {
                p.priority_overflow_pro.unwrap_or(1.5)
            }
        }
        _ => 1.0,
    };
    let request = Reservation {
        organization_id: config.organization_id.clone(),
        weight: config.weight,
        bound,
        fair_share: true,
        requests_per_minute: p.requests_per_minute,
        tokens_per_minute: p.tokens_per_minute,
        cache_priority_alpha: p.cache_priority_alpha,
        reserved_tokens: if p.tokens_per_minute.is_some() {
            input_tokens.saturating_add(lane.output_tokens)
        } else {
            0
        },
        warm_session: warm,
        fresh_spill_fraction: fresh_fraction,
        force,
        hard_bound: refusing && config.priority == 0 && !failover,
        overflow_ceiling: bound
            .filter(|_| {
                config.priority > 0 && (refusing || (config.selected_route && first)) && !failover
            })
            .map(|n| n as f64 * factor),
        rate_retry,
        class_cap,
        fast,
        fast_reserve: lane.fast_reserve(),
        fast_reserve_hard: refusing,
        failover_cap: bound.filter(|_| failover).map(|n| n.div_ceil(2).max(1)),
    };
    loads
        .reserve(&lane.load_key(), &request, now)
        .map(Some)
        .map_err(|mut shed| {
            if default && shed.reason == "queue_bound" {
                shed.default_bound = true;
            }
            shed
        })
}

pub(super) fn pace(loads: &Loads, config: &Registration, lane: &Lane) -> Option<f64> {
    let pace = config.pace?;
    if !lane.policy.self_hosted || lane.fast(config) {
        return None;
    }
    if let Some(threshold) = config.pace_saturation {
        let bound = lane.policy.concurrency_bound.or(loads.default_bound)?;
        let key = lane.load_key();
        let (held, bound) = match lane.fast_reserve() {
            Some(reserve) => (
                loads.inflight(&key, None) - loads.fast_inflight(&key),
                bound - reserve,
            ),
            None => (loads.inflight(&key, None), bound),
        };
        if (held as f64) / (bound as f64) < threshold {
            return None;
        }
    }
    Some(pace)
}
