//! Request-owned retry budgets and the three-pass health claim walk.

use std::collections::{HashMap, HashSet};

use serde::Deserialize;

use crate::errors::{Failure, FailureClass};
use crate::waterfall::fallback_rules::{self, RungEligibility};

use super::health::{Claim, Health, HealthKey};
use super::loads::Shed;

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[allow(clippy::enum_variant_names)] // Names match the admitted routing vocabulary.
pub(super) enum FailoverMode {
    MaximizeAvailability,
    MaximizeCache,
    MaximizeCacheAffinity,
}

#[derive(Clone, Deserialize)]
pub(super) struct Stage {
    pub lead: usize,
    pub failover_mode: FailoverMode,
    #[serde(default, deserialize_with = "crate::lossless_json::deserialize")]
    pub throttle_cache_threshold: Option<f64>,
    pub throttle_redial: bool,
}

#[derive(Clone, Deserialize)]
pub(super) struct AttemptPolicy {
    pub maximum_total_attempts: u32,
    pub maximum_same_deployment_attempts: u32,
    pub physical_route_cap: Option<u32>,
}

impl AttemptPolicy {
    pub fn valid(&self) -> bool {
        (1..=8).contains(&self.maximum_total_attempts)
            && (1..=4).contains(&self.maximum_same_deployment_attempts)
            && self.physical_route_cap.is_none_or(|n| (1..=4).contains(&n))
    }
    pub fn permits(&self, total: u32, physical: u32) -> bool {
        total < self.maximum_total_attempts
            && self.physical_route_cap.is_none_or(|cap| physical < cap)
    }
}

pub(super) type Rules = Vec<Option<Vec<String>>>;

pub(super) fn eligible(rules: &Rules, failure: Option<&Failure>, unrestricted: bool) -> Vec<usize> {
    rules
        .iter()
        .enumerate()
        .filter_map(|(depth, rule)| {
            let permitted = match (rule, failure) {
                (None, _) => unrestricted,
                (Some(rule), Some(failure)) => {
                    fallback_rules::eligibility(Some(rule), failure) == RungEligibility::Matched
                }
                _ => false,
            };
            permitted.then_some(depth)
        })
        .collect()
}

pub(super) fn untried(
    rules: &Rules,
    failure: &Failure,
    dispatched: &[u32],
    current: usize,
    unrestricted: bool,
) -> Vec<usize> {
    eligible(rules, Some(failure), unrestricted)
        .into_iter()
        .filter(|depth| fallback_rules::untried(dispatched, current, *depth))
        .collect()
}

pub(super) fn claim(
    health: &mut Health,
    keys: &[HealthKey],
    start: usize,
    depths: &[usize],
    throttle_probe: bool,
    now: f64,
) -> Option<usize> {
    let depths: Vec<_> = depths
        .iter()
        .copied()
        .filter(|depth| *depth >= start)
        .collect();
    let ordered = health.order(keys, &depths, now);
    let modes = [
        Claim::Normal,
        Claim::LastResort,
        Claim::Forced,
        Claim::ThrottleProbe,
    ];
    for mode in modes.into_iter().take(if throttle_probe { 4 } else { 3 }) {
        for depth in &ordered {
            if health.claim(&keys[*depth], mode, now) {
                return Some(*depth);
            }
        }
    }
    None
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Throttle {
    Surfaced,
    Cold,
    Backoff,
}

impl Throttle {
    pub fn reason(self) -> &'static str {
        match self {
            Self::Surfaced => "throttle_surfaced_cache_preserving",
            Self::Cold => "throttle_failover_cold",
            Self::Backoff => "throttle_backoff",
        }
    }
}

fn throttle(failure: &Failure, threshold: Option<f64>, cached: f64) -> Option<Throttle> {
    threshold
        .filter(|_| failure.failure_class == FailureClass::Throttled)
        .map(|threshold| {
            if cached >= threshold {
                Throttle::Surfaced
            } else {
                Throttle::Cold
            }
        })
}

pub(super) struct Retry<'a> {
    pub failure: &'a Failure,
    pub current: usize,
    pub ordinary_counts: &'a [u32],
    pub physical_counts: &'a [u32],
    pub total: u32,
    pub policy: &'a AttemptPolicy,
    pub refusal_failover: bool,
    pub stage: &'a Stage,
    pub cached_fraction: f64,
    pub throttle_backoff: bool,
    pub redial_budget: u32,
    pub rules: &'a Rules,
}

pub(super) fn next(
    health: &mut Health,
    keys: &[HealthKey],
    retry: &Retry<'_>,
    now: f64,
) -> (Option<usize>, Option<Throttle>) {
    let candidate = next_depth(health, keys, retry, now);
    let mut disposition = throttle(
        retry.failure,
        retry.stage.throttle_cache_threshold,
        retry.cached_fraction,
    );
    if retry.stage.throttle_redial && retry.failure.failure_class == FailureClass::Throttled {
        disposition = candidate.map(|depth| {
            if depth == retry.current {
                Throttle::Backoff
            } else {
                Throttle::Cold
            }
        });
    }
    (candidate, disposition)
}

fn next_depth(
    health: &mut Health,
    keys: &[HealthKey],
    retry: &Retry<'_>,
    now: f64,
) -> Option<usize> {
    if retry.total >= retry.policy.maximum_total_attempts {
        return None;
    }
    let same = retry
        .policy
        .physical_route_cap
        .is_none_or(|cap| retry.physical_counts[retry.current] < cap);
    if same
        && retry.failure.retryable_same_deployment
        && retry.ordinary_counts[retry.current] < retry.policy.maximum_same_deployment_attempts
        && health.claim(&keys[retry.current], Claim::Normal, now)
    {
        return Some(retry.current);
    }
    let throttled = retry.failure.failure_class == FailureClass::Throttled;
    if retry.stage.throttle_redial && throttled {
        if retry.throttle_backoff
            && same
            && retry.redial_budget > 0
            && health.claim(&keys[retry.current], Claim::ThrottleRedial, now)
        {
            return Some(retry.current);
        }
    } else {
        let disposition = throttle(
            retry.failure,
            retry.stage.throttle_cache_threshold,
            retry.cached_fraction,
        );
        if disposition == Some(Throttle::Surfaced)
            || (disposition.is_none()
                && retry.stage.failover_mode == FailoverMode::MaximizeCache
                && throttled)
        {
            return None;
        }
    }
    let unrestricted = retry.failure.failover_eligible
        || (retry.failure.failure_class == FailureClass::Refusal && retry.refusal_failover);
    let depths = untried(
        retry.rules,
        retry.failure,
        retry.physical_counts,
        retry.current,
        unrestricted,
    );
    claim(health, keys, 0, &depths, false, now)
}

pub(super) struct FailoverPass {
    pub untried: Vec<usize>,
    ladder: Vec<usize>,
    pub may_force: bool,
    forced: HashSet<usize>,
    pub forcing: bool,
}

impl FailoverPass {
    pub fn begin(
        rules: &Rules,
        failure: Option<&Failure>,
        current: Option<usize>,
        dispatched: &[u32],
        refusal_failover: bool,
    ) -> Option<Self> {
        let failure = failure?;
        let current = current?;
        if !failure.failover_eligible
            && !(failure.failure_class == FailureClass::Refusal && refusal_failover)
        {
            return None;
        }
        let untried = untried(rules, failure, dispatched, current, true);
        Some(Self {
            ladder: untried.clone(),
            untried,
            may_force: matches!(
                failure.failure_class,
                FailureClass::Transport | FailureClass::ProviderInternal
            ),
            forced: HashSet::new(),
            forcing: false,
        })
    }
    pub fn sideways(
        &mut self,
        health: &mut Health,
        keys: &[HealthKey],
        candidate: usize,
        now: f64,
    ) -> Option<usize> {
        if self.forcing {
            self.forcing = false;
            return None;
        }
        self.ladder.retain(|depth| *depth != candidate);
        claim(health, keys, 0, &self.ladder, false, now)
    }
    pub fn forced_target(
        &mut self,
        sheds: &[(usize, String)],
        records: &HashMap<usize, Shed>,
        health: &mut Health,
        keys: &[HealthKey],
        now: f64,
    ) -> Option<usize> {
        if !self.may_force {
            return None;
        }
        for (depth, _) in sheds {
            let Some(shed) = records.get(depth) else {
                continue;
            };
            if self.forced.contains(depth)
                || !self.untried.contains(depth)
                || shed.overflow_ceiling
                || shed.reason == "rate_limit"
            {
                continue;
            }
            self.forced.insert(*depth);
            if claim(health, keys, *depth, &[*depth], false, now) != Some(*depth) {
                continue;
            }
            self.forcing = true;
            return Some(*depth);
        }
        None
    }
    pub fn capacity_refusal(&self, records: &HashMap<usize, Shed>) -> bool {
        self.may_force && self.untried.iter().any(|depth| records.contains_key(depth))
    }
}

pub(super) fn overflow_target(
    priority: u8,
    selected_route: bool,
    refusing: &[bool],
    sheds: &[(usize, String)],
    records: &HashMap<usize, Shed>,
) -> Option<usize> {
    let historical = |depth: usize, shed: Option<&Shed>| {
        (!refusing[depth] && !shed.is_some_and(|shed| shed.default_bound || shed.overflow_ceiling))
            .then_some(depth)
    };
    for (depth, _) in sheds {
        let shed = records.get(depth);
        if priority > 0 {
            if shed.is_some_and(|shed| shed.overflow_ceiling) {
                continue;
            }
            if shed.is_some_and(|shed| shed.reason == "rate_limit") {
                return if selected_route {
                    None
                } else {
                    historical(*depth, shed)
                };
            }
            return Some(*depth);
        }
        if !shed.is_some_and(|shed| shed.fleet) {
            return historical(*depth, shed);
        }
    }
    None
}

pub(super) fn disclosure(
    stage: &Stage,
    candidate: usize,
    sheds: &[(usize, String)],
    forced: bool,
    sticky: bool,
    redial: bool,
) -> (Option<String>, Option<usize>) {
    if redial {
        return (Some(Throttle::Backoff.reason().into()), None);
    }
    let (reason, target) = if stage.failover_mode == FailoverMode::MaximizeCacheAffinity {
        let reason = if forced {
            "saturated_overflow"
        } else if candidate == stage.lead {
            if sticky && candidate == 0 {
                "affinity_sticky"
            } else {
                "affinity"
            }
        } else {
            sheds
                .iter()
                .find(|(depth, _)| *depth == stage.lead)
                .map_or("rung_dead", |(_, reason)| reason)
        };
        (reason, stage.lead)
    } else if forced {
        let Some((target, _)) = sheds.first() else {
            return (None, None);
        };
        ("saturated_overflow", *target)
    } else if let Some((target, reason)) = sheds.first() {
        (reason.as_str(), *target)
    } else {
        return (None, None);
    };
    (Some(reason.into()), (candidate != target).then_some(target))
}
