//! State and soft admission decisions for a physical lane.

use std::collections::{HashMap, VecDeque};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct Reservation {
    pub organization_id: String,
    pub weight: u64,
    pub bound: Option<u64>,
    pub fair_share: bool,
    pub requests_per_minute: Option<u64>,
    pub tokens_per_minute: Option<u64>,
    #[serde(default, deserialize_with = "crate::lossless_json::deserialize")]
    pub cache_priority_alpha: Option<f64>,
    pub reserved_tokens: u64,
    pub warm_session: bool,
    #[serde(default, deserialize_with = "crate::lossless_json::deserialize")]
    pub fresh_spill_fraction: Option<f64>,
    pub force: bool,
    pub hard_bound: bool,
    #[serde(default, deserialize_with = "crate::lossless_json::deserialize")]
    pub overflow_ceiling: Option<f64>,
    pub rate_retry: bool,
    pub class_cap: Option<(i64, u64)>,
    pub fast: bool,
    pub fast_reserve: Option<u64>,
    pub fast_reserve_hard: bool,
    pub failover_cap: Option<u64>,
}

impl Default for Reservation {
    fn default() -> Self {
        Self {
            organization_id: String::new(),
            weight: 1,
            bound: None,
            fair_share: false,
            requests_per_minute: None,
            tokens_per_minute: None,
            cache_priority_alpha: None,
            reserved_tokens: 0,
            warm_session: true,
            fresh_spill_fraction: None,
            force: false,
            hard_bound: false,
            overflow_ceiling: None,
            rate_retry: false,
            class_cap: None,
            fast: false,
            fast_reserve: None,
            fast_reserve_hard: true,
            failover_cap: None,
        }
    }
}

impl Reservation {
    pub fn valid(&self) -> bool {
        !self.organization_id.is_empty()
            && self.weight > 0
            && self.bound.is_none_or(|n| n > 0)
            && self
                .fast_reserve
                .is_none_or(|n| self.bound.is_some_and(|bound| n <= bound))
            && [
                self.cache_priority_alpha,
                self.fresh_spill_fraction,
                self.overflow_ceiling,
            ]
            .into_iter()
            .flatten()
            .all(|v| v.is_finite() && v >= 0.0)
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub(super) struct Shed {
    pub reason: &'static str,
    pub learned_requests_per_minute: Option<f64>,
    pub overflow_ceiling: bool,
    pub default_bound: bool,
    pub fleet: bool,
    pub fleet_refuse: bool,
}

impl Shed {
    pub fn new(reason: &'static str) -> Self {
        Self {
            reason,
            learned_requests_per_minute: None,
            overflow_ceiling: false,
            default_bound: false,
            fleet: false,
            fleet_refuse: false,
        }
    }
    pub fn hard() -> Self {
        Self {
            overflow_ceiling: true,
            ..Self::new("queue_bound")
        }
    }
}

#[derive(Default)]
pub(super) struct Organization {
    pub inflight: u64,
    pub last_seen: f64,
    pub weight: u64,
    pub fast: u64,
}

pub(super) struct CacheSignal {
    pub fraction: f64,
    pub sampled_at: f64,
}

#[derive(Default)]
pub(super) struct Rung {
    pub total: u64,
    // Python's insertion order is retained for fractional fairness summation.
    pub organizations: IndexMap<String, Organization>,
    pub cache: HashMap<String, CacheSignal>,
    pub cache_pruned_at: f64,
    pub window: VecDeque<(f64, u64)>,
    pub window_tokens: u128,
    pub learned_rpm: Option<f64>,
    pub learned_throttled_at: f64,
    pub learned_crept_at: f64,
    pub classes: HashMap<i64, u64>,
    pub fast: u64,
    pub failover_forced: u64,
}

impl Rung {
    pub fn prune_cache(&mut self, now: f64) {
        if now - self.cache_pruned_at < 60.0 {
            return;
        }
        self.cache_pruned_at = now;
        self.cache
            .retain(|_, signal| signal.sampled_at >= now - 3600.0);
    }
    pub fn prune(&mut self, now: f64, activity_window: f64) {
        self.organizations
            .retain(|_, load| load.inflight > 0 || load.last_seen >= now - activity_window);
        self.prune_cache(now);
        self.prune_window(now);
    }
    pub fn prune_window(&mut self, now: f64) {
        while self.window.front().is_some_and(|(at, _)| *at <= now - 60.0) {
            self.window_tokens -= u128::from(self.window.pop_front().unwrap().1);
        }
    }
    fn working_rpm(&mut self, authored: Option<u64>, now: f64) -> Option<f64> {
        if let Some(mut learned) = self.learned_rpm {
            if now - self.learned_throttled_at >= 6.0 * 3600.0 {
                self.learned_rpm = None;
            } else {
                let steps = ((now - self.learned_crept_at) / 60.0).floor().max(0.0) as u64;
                for _ in 0..steps {
                    learned += (0.05 * learned).ceil().max(1.0);
                    if authored.is_some_and(|n| learned >= n as f64) {
                        learned = authored.unwrap() as f64;
                        break;
                    }
                }
                if steps > 0 {
                    self.learned_crept_at += steps as f64 * 60.0;
                }
                self.learned_rpm = Some(learned);
            }
        }
        match (authored, self.learned_rpm) {
            (None, learned) => learned,
            (Some(authored), None) => Some(authored as f64),
            (Some(authored), Some(learned)) => Some((authored as f64).min(learned)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn shed(
        &mut self,
        request: &Reservation,
        now: f64,
        activity_window: f64,
        bound: Option<u64>,
        held: u64,
        regular_only: bool,
        skip_rate: bool,
    ) -> Option<Shed> {
        if let Some(bound) = bound {
            if held >= bound {
                return Some(Shed::new("queue_bound"));
            }
            if !request.warm_session
                && request
                    .fresh_spill_fraction
                    .is_some_and(|fraction| held as f64 >= bound as f64 * fraction)
            {
                return Some(Shed::new("fresh_session_spill"));
            }
        }
        let rpm = self.working_rpm(request.requests_per_minute, now);
        if !skip_rate
            && (rpm.is_some_and(|rpm| self.window.len() as f64 + 1.0 > rpm)
                || request.tokens_per_minute.is_some_and(|cap| {
                    self.window_tokens > 0
                        && self.window_tokens + u128::from(request.reserved_tokens)
                            > u128::from(cap)
                }))
        {
            return Some(Shed {
                learned_requests_per_minute: self.learned_rpm,
                ..Shed::new("rate_limit")
            });
        }
        let bound = bound.filter(|_| request.fair_share)?;
        let organization = &self.organizations[&request.organization_id];
        let congestion = held as f64 / bound as f64;
        let weighted: Vec<_> = self
            .organizations
            .iter()
            .filter(|(_, o)| {
                o.inflight > 0 || o.last_seen >= organization.last_seen - activity_window
            })
            .map(|(id, o)| {
                let fraction = self.cache.get(id).map_or(0.0, |s| s.fraction);
                let weight = o.weight as f64
                    * (1.0 + request.cache_priority_alpha.unwrap_or(0.0) * congestion * fraction);
                (id, o, weight)
            })
            .collect();
        let total_weight: f64 = weighted.iter().map(|(_, _, weight)| weight).sum();
        let own_weight = weighted
            .iter()
            .find(|(id, _, _)| *id == &request.organization_id)
            .unwrap()
            .2;
        let holding = |o: &Organization| {
            if regular_only {
                o.inflight - o.fast
            } else {
                o.inflight
            }
        };
        if (holding(organization) + 1) as f64 <= bound as f64 * own_weight / total_weight {
            return None;
        }
        let deficit: f64 = weighted
            .iter()
            .filter(|(id, _, _)| *id != &request.organization_id)
            .map(|(_, o, weight)| {
                (bound as f64 * weight / total_weight - holding(o) as f64).max(0.0)
            })
            .sum();
        (u128::from(held) + 1 + deficit as u128 > u128::from(bound))
            .then(|| Shed::new("fair_share_shed"))
    }
}
