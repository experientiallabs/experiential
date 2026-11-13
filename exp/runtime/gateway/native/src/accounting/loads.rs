//! Native local concurrency, rate windows, fairness and cache-affinity signals.

use std::collections::HashMap;

use uuid::Uuid;

use super::health::LaneKey;
use super::load_state::{CacheSignal, Rung};
pub(super) use super::load_state::{Reservation, Shed};

struct Ticket {
    lane: LaneKey,
    org: String,
    class: Option<i64>,
    fast: bool,
    failover: bool,
    window: (f64, u64),
}

pub(super) struct Loads {
    pub default_bound: Option<u64>,
    activity_window: f64,
    rungs: HashMap<LaneKey, Rung>,
    tickets: HashMap<String, Ticket>,
    attempts: HashMap<String, String>,
}

impl Loads {
    pub fn new(default_bound: Option<u64>, activity_window: f64) -> Option<Self> {
        (default_bound.is_none_or(|v| v > 0)
            && activity_window.is_finite()
            && activity_window > 0.0)
            .then(|| Self {
                default_bound,
                activity_window,
                rungs: HashMap::new(),
                tickets: HashMap::new(),
                attempts: HashMap::new(),
            })
    }

    /// Validated admission policy is passed unchanged to every physical reservation.
    pub fn reserve(
        &mut self,
        key: &LaneKey,
        request: &Reservation,
        now: f64,
    ) -> Result<String, Shed> {
        let rung = self.rungs.entry(key.clone()).or_default();
        let org = rung
            .organizations
            .entry(request.organization_id.clone())
            .or_default();
        org.last_seen = now;
        org.weight = request.weight;
        rung.prune(now, self.activity_window);
        let mut request = request.clone();
        let mut bound = request.bound;
        if let (Some(bound), Some(reserve)) = (bound, request.fast_reserve) {
            let full = rung.total >= bound;
            let regular_full = rung.total - rung.fast >= bound - reserve;
            if request.fast_reserve_hard {
                if full || (!request.fast && regular_full) {
                    return Err(Shed::hard());
                }
            } else if !request.force && full {
                return Err(Shed::new("queue_bound"));
            }
        } else if let Some(bound) = bound {
            if (request.fast && rung.fast >= (bound - bound / 2).max(1))
                || (!request.fast && rung.fast > 0 && rung.total - rung.fast >= (bound / 2).max(1))
            {
                return Err(Shed::hard());
            }
        }
        if request.fast {
            bound = None;
            request.fair_share = false;
            request.overflow_ceiling = None;
        }
        let mut occupied = rung.total;
        if let (Some(cap), Some(reserve)) = (bound, request.fast_reserve) {
            if request.fast_reserve_hard || !request.force {
                bound = Some(cap - reserve);
                occupied = rung.total - rung.fast;
            }
        }
        if request.hard_bound && bound.is_some_and(|bound| occupied >= bound) {
            return Err(Shed::new("queue_bound"));
        }
        if request
            .class_cap
            .is_some_and(|(class, cap)| rung.classes.get(&class).copied().unwrap_or(0) >= cap)
            || (request.force
                && request
                    .failover_cap
                    .is_some_and(|cap| rung.failover_forced >= cap))
            || (request.force
                && request
                    .overflow_ceiling
                    .is_some_and(|cap| rung.total as f64 + 1.0 > cap))
        {
            return Err(Shed::hard());
        }
        if !request.force || request.rate_retry {
            let bound = if request.force && request.overflow_ceiling.is_some() {
                None
            } else {
                bound
            };
            if let Some(shed) = rung.shed(
                &request,
                now,
                self.activity_window,
                bound,
                occupied,
                request.fast_reserve.is_some() && (request.fast_reserve_hard || !request.force),
                request.rate_retry,
            ) {
                return Err(shed);
            }
        }
        let org = rung
            .organizations
            .get_mut(&request.organization_id)
            .unwrap();
        org.inflight += 1;
        rung.total += 1;
        rung.window.push_back((now, request.reserved_tokens));
        rung.window_tokens += u128::from(request.reserved_tokens);
        let class = request.class_cap.map(|(class, _)| class);
        if let Some(class) = class {
            *rung.classes.entry(class).or_default() += 1;
        }
        if request.fast {
            rung.fast += 1;
            org.fast += 1;
        }
        let failover = request.force && request.failover_cap.is_some();
        if failover {
            rung.failover_forced += 1;
        }
        let id = format!("rung-{}", Uuid::new_v4().simple());
        self.tickets.insert(
            id.clone(),
            Ticket {
                lane: key.clone(),
                org: request.organization_id,
                class,
                fast: request.fast,
                failover,
                window: (now, request.reserved_tokens),
            },
        );
        Ok(id)
    }

    pub fn can_admit(&mut self, key: &LaneKey, request: &Reservation, now: f64) -> bool {
        let Some(rung) = self.rungs.get_mut(key) else {
            return true;
        };
        let org = rung
            .organizations
            .entry(request.organization_id.clone())
            .or_default();
        org.last_seen = now;
        org.weight = request.weight;
        rung.prune(now, self.activity_window);
        let mut bound = request.bound;
        let mut occupied = rung.total;
        if let (Some(cap), Some(reserve)) = (bound, request.fast_reserve) {
            if rung.total >= cap {
                return false;
            }
            bound = Some(cap - reserve);
            occupied = rung.total - rung.fast;
        }
        let request = Reservation {
            warm_session: true,
            fresh_spill_fraction: None,
            ..request.clone()
        };
        rung.shed(
            &request,
            now,
            self.activity_window,
            bound,
            occupied,
            request.fast_reserve.is_some(),
            false,
        )
        .is_none()
    }

    pub fn record_throttle(&mut self, key: &LaneKey, now: f64) {
        let rung = self.rungs.entry(key.clone()).or_default();
        rung.prune_window(now);
        rung.learned_rpm = Some((rung.window.len() as f64 * 0.9).max(0.1));
        rung.learned_throttled_at = now;
        rung.learned_crept_at = now;
    }

    pub fn record_settle(&mut self, key: &LaneKey, org: &str, cached: u64, input: u64, now: f64) {
        if input == 0 {
            return;
        }
        let sample = cached.min(input) as f64 / input as f64;
        let rung = self.rungs.entry(key.clone()).or_default();
        rung.prune_cache(now);
        if let Some(signal) = rung.cache.get_mut(org) {
            let elapsed = (now - signal.sampled_at).max(1.0);
            let retained = 0.5_f64.powf(elapsed / 600.0);
            signal.fraction = signal.fraction * retained + sample * (1.0 - retained);
            signal.sampled_at = now;
        } else {
            rung.cache.insert(
                org.into(),
                CacheSignal {
                    fraction: sample,
                    sampled_at: now,
                },
            );
        }
    }

    pub fn cached_fraction(&self, key: &LaneKey, org: &str, now: f64) -> f64 {
        self.rungs
            .get(key)
            .and_then(|r| r.cache.get(org))
            .filter(|s| s.sampled_at >= now - 3600.0)
            .map_or(0.0, |s| s.fraction)
    }

    pub fn bind(&mut self, ticket: &str, attempt: &str) {
        if self.tickets.contains_key(ticket) {
            self.attempts.insert(attempt.into(), ticket.into());
        }
    }

    pub fn release_attempt(&mut self, attempt: &str) {
        if let Some(ticket) = self.attempts.remove(attempt) {
            self.release(&ticket);
        }
    }

    pub fn release(&mut self, ticket: &str) {
        let Some(ticket) = self.tickets.remove(ticket) else {
            return;
        };
        let Some(rung) = self.rungs.get_mut(&ticket.lane) else {
            return;
        };
        if ticket.failover {
            rung.failover_forced = rung.failover_forced.saturating_sub(1);
        }
        if let Some(class) = ticket.class {
            if let Some(count) = rung.classes.get_mut(&class) {
                *count = count.saturating_sub(1);
            }
        }
        if ticket.fast {
            rung.fast = rung.fast.saturating_sub(1);
        }
        if let Some(org) = rung.organizations.get_mut(&ticket.org) {
            if org.inflight > 0 {
                org.inflight -= 1;
                if ticket.fast {
                    org.fast = org.fast.saturating_sub(1);
                }
            }
        }
        rung.total = rung.total.saturating_sub(1);
    }

    /// A fleet refusal retracts the local dispatch window as well as the slot.
    pub fn retract(&mut self, ticket: &str) {
        if let Some(held) = self.tickets.get(ticket) {
            if let Some(rung) = self.rungs.get_mut(&held.lane) {
                if let Some(index) = rung.window.iter().position(|v| *v == held.window) {
                    rung.window.remove(index);
                    rung.window_tokens -= u128::from(held.window.1);
                }
            }
        }
        self.release(ticket);
    }

    pub fn restamp(&mut self, ticket: &str, now: f64) {
        let Some(held) = self.tickets.get_mut(ticket) else {
            return;
        };
        if now <= held.window.0 {
            return;
        }
        if let Some(rung) = self.rungs.get_mut(&held.lane) {
            if let Some(index) = rung.window.iter().position(|v| *v == held.window) {
                rung.window.remove(index);
                held.window.0 = now;
                rung.window.push_back(held.window);
            }
        }
    }

    pub fn inflight(&self, key: &LaneKey, org: Option<&str>) -> u64 {
        self.rungs.get(key).map_or(0, |r| match org {
            None => r.total,
            Some(org) => r.organizations.get(org).map_or(0, |o| o.inflight),
        })
    }
    pub fn fast_inflight(&self, key: &LaneKey) -> u64 {
        self.rungs.get(key).map_or(0, |r| r.fast)
    }
    pub fn failover_inflight(&self, key: &LaneKey) -> u64 {
        self.rungs.get(key).map_or(0, |r| r.failover_forced)
    }
    pub fn learned_ceilings(&self) -> HashMap<String, f64> {
        self.rungs
            .iter()
            .filter_map(|((id, connection), r)| {
                r.learned_rpm.map(|n| {
                    (
                        format!("{id}:{}", connection.chars().take(8).collect::<String>()),
                        n,
                    )
                })
            })
            .collect()
    }
}
