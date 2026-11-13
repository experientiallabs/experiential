//! Circuit, throttle and physical-lane ejection state shared by native executions.
//!
//! The owner serializes access. Monotonic seconds are supplied by that owner so
//! replay tests can exercise the same boundary instants as the admission policy.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::Deserialize;

use crate::errors::{Failure, FailureClass, HeaderTimeout};

pub(super) type HealthKey = (String, String, String);
pub(super) type LaneKey = (String, String);

#[derive(Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct Policy {
    pub failure_threshold: u32,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub open_seconds: f64,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub throttle_seconds: f64,
    pub ejection_failures: usize,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub ejection_window_seconds: f64,
    pub ejection_header_timeouts: usize,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub ejection_header_window_seconds: f64,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub ejection_cooldown_seconds: f64,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            failure_threshold: 2,
            open_seconds: 30.0,
            throttle_seconds: 30.0,
            ejection_failures: 3,
            ejection_window_seconds: 10.0,
            ejection_header_timeouts: 2,
            ejection_header_window_seconds: 600.0,
            ejection_cooldown_seconds: 30.0,
        }
    }
}

impl Policy {
    pub fn valid(&self) -> bool {
        self.failure_threshold > 0
            && [
                self.open_seconds,
                self.throttle_seconds,
                self.ejection_window_seconds,
                self.ejection_header_window_seconds,
                self.ejection_cooldown_seconds,
            ]
            .into_iter()
            .all(|v| v.is_finite() && v > 0.0)
    }
}

#[derive(Default)]
struct Circuit {
    failures: u32,
    open_until: f64,
    throttle_until: f64,
    half_open: bool,
    last_resort: bool,
    refusals: u64,
    /// The provider-account failure (authentication, not found, quota) holding
    /// the circuit open, reduced to its class, message and ownership. Never
    /// recorded for a tenant origin, whose failures stay opaque to callers.
    account: Option<Failure>,
    /// End of the throttle windows observed after `account` was recorded; only
    /// these, never an older window, outrank the recorded account failure.
    newer_throttle_until: f64,
    stated_throttle_until: f64,
    probe_attempts: HashSet<String>,
}

#[derive(Default)]
struct Lane {
    failures: VecDeque<f64>,
    headers: VecDeque<f64>,
    ejected_until: f64,
    probe_started: Option<f64>,
}

pub(super) struct Health {
    policy: Policy,
    circuits: HashMap<HealthKey, Circuit>,
    lanes: HashMap<LaneKey, Lane>,
    probe_paced_at: HashMap<LaneKey, f64>,
}

#[derive(Clone, Copy)]
pub(super) enum Claim {
    Normal,
    LastResort,
    Forced,
    ThrottleRedial,
    ThrottleProbe,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Cause {
    Transport,
    Credential,
    Throttle,
}

pub(super) fn cause(class: FailureClass) -> Option<Cause> {
    use FailureClass::*;
    match class {
        ProviderAuthentication | ProviderNotFound | ProviderQuota => Some(Cause::Credential),
        Transport | Timeout | MalformedResponse | ProviderInternal => Some(Cause::Transport),
        Throttled => Some(Cause::Throttle),
        _ => None,
    }
}

fn lane(key: &HealthKey) -> LaneKey {
    (key.1.clone(), key.2.clone())
}

impl Health {
    pub fn new(policy: Policy) -> Option<Self> {
        policy.valid().then(|| Self {
            policy,
            circuits: HashMap::new(),
            lanes: HashMap::new(),
            probe_paced_at: HashMap::new(),
        })
    }

    pub fn ejected(&self, key: &HealthKey, now: f64) -> bool {
        self.lanes.get(&lane(key)).is_some_and(|state| {
            state.ejected_until != 0.0
                && (now < state.ejected_until
                    || state
                        .probe_started
                        .is_some_and(|start| now - start < self.policy.ejection_cooldown_seconds))
        })
    }

    pub fn order(&self, keys: &[HealthKey], candidates: &[usize], now: f64) -> Vec<usize> {
        let mut ordered = candidates.to_vec();
        ordered.sort_by_key(|index| self.ejected(&keys[*index], now));
        ordered
    }

    pub fn claim(&mut self, key: &HealthKey, mode: Claim, now: f64) -> bool {
        let state = self.circuits.entry(key.clone()).or_default();
        if matches!(mode, Claim::ThrottleRedial) {
            return state.open_until <= now;
        }
        if matches!(mode, Claim::ThrottleProbe) {
            if state.throttle_until <= now || state.stated_throttle_until > now {
                return false;
            }
            let physical = lane(key);
            if self
                .probe_paced_at
                .get(&physical)
                .is_some_and(|last| now - last < 1.0)
            {
                return false;
            }
            self.probe_paced_at.insert(physical, now);
        } else if state.throttle_until > now {
            return false;
        }
        match mode {
            Claim::Normal => {
                if state.open_until > now {
                    return false;
                }
                if state.failures >= self.policy.failure_threshold {
                    if state.half_open {
                        return false;
                    }
                    state.half_open = true;
                }
            }
            Claim::LastResort => {
                if state.last_resort {
                    return false;
                }
                state.last_resort = true;
            }
            Claim::Forced | Claim::ThrottleProbe => (),
            Claim::ThrottleRedial => unreachable!(),
        }
        if let Some(state) = self.lanes.get_mut(&lane(key)) {
            if state.ejected_until != 0.0
                && now >= state.ejected_until
                && state
                    .probe_started
                    .is_none_or(|start| now - start >= self.policy.ejection_cooldown_seconds)
            {
                state.probe_started = Some(now);
            }
        }
        true
    }

    /// Bind a no-failure reservation to the hintless window current at reservation.
    pub fn bind_throttle_probe(&mut self, key: &HealthKey, id: &str, probe: bool, now: f64) {
        if !probe {
            return;
        }
        if let Some(state) = self.circuits.get_mut(key) {
            if state.throttle_until > now && state.stated_throttle_until <= now {
                state.probe_attempts.insert(id.into());
            }
        }
    }

    /// An opened response and a successful terminal restore the circuit equally.
    pub fn succeeded(&mut self, key: &HealthKey, attempt: Option<&str>, now: f64) {
        let state = self.circuits.entry(key.clone()).or_default();
        state.failures = 0;
        state.open_until = 0.0;
        state.half_open = false;
        state.last_resort = false;
        state.account = None;
        if attempt.is_some_and(|id| state.probe_attempts.remove(id))
            && state.throttle_until > now
            && state.stated_throttle_until <= now
        {
            state.throttle_until = state.stated_throttle_until;
            state.newer_throttle_until =
                state.newer_throttle_until.min(state.stated_throttle_until);
        }
        self.lanes.remove(&lane(key));
        // Provider quota windows survive an otherwise successful response.
    }

    pub fn failed(&mut self, key: &HealthKey, failure: &Failure, attempt: Option<&str>, now: f64) {
        self.lane_failed(key, failure, now);
        let state = self.circuits.entry(key.clone()).or_default();
        if let Some(id) = attempt {
            state.probe_attempts.remove(id);
        }
        state.half_open = false;
        state.last_resort = false;
        match cause(failure.failure_class) {
            Some(Cause::Throttle) => {
                let window = failure
                    .retry_after_seconds
                    .map_or(self.policy.throttle_seconds, |seconds| {
                        f64::from(seconds).clamp(5.0, 6.0 * 3600.0)
                    });
                state.throttle_until = state.throttle_until.max(now + window);
                state.newer_throttle_until = state.newer_throttle_until.max(now + window);
                state.probe_attempts.clear();
                if failure.retry_after_seconds.is_some() {
                    state.stated_throttle_until = state.stated_throttle_until.max(now + window);
                }
            }
            Some(Cause::Credential) => {
                state.failures = self.policy.failure_threshold;
                state.open_until = now + self.policy.open_seconds;
                state.account = (!failure.tenant_origin).then(|| {
                    let mut account = Failure::new(failure.failure_class, &failure.safe_message);
                    account.customer_owned = failure.customer_owned;
                    account
                });
                state.newer_throttle_until = 0.0;
            }
            Some(Cause::Transport) => {
                state.failures = state.failures.saturating_add(1);
                if state.failures >= self.policy.failure_threshold {
                    state.open_until = now + self.policy.open_seconds;
                    state.account = None;
                }
            }
            None if failure.failure_class == FailureClass::Refusal => {
                state.refusals = state.refusals.saturating_add(1);
            }
            None => (),
        }
    }

    pub fn exhausted(&mut self, key: &HealthKey, reset_seconds: f64, now: f64) {
        if !reset_seconds.is_finite() {
            return;
        }
        let state = self.circuits.entry(key.clone()).or_default();
        let window_end = now + reset_seconds.clamp(5.0, 8.0 * 86400.0);
        state.throttle_until = state.throttle_until.max(window_end);
        state.newer_throttle_until = state.newer_throttle_until.max(window_end);
        state.stated_throttle_until = state.stated_throttle_until.max(window_end);
    }

    pub fn suppressed(&self, key: &HealthKey, now: f64) -> bool {
        self.circuits
            .get(key)
            .is_some_and(|s| s.throttle_until > now || s.open_until > now)
    }

    pub fn throttled_remaining(&self, keys: &[HealthKey], now: f64) -> Option<f64> {
        if keys.is_empty() {
            return None;
        }
        let mut remaining: f64 = 0.0;
        for key in keys {
            let state = self.circuits.get(key)?;
            if state.throttle_until <= now {
                return None;
            }
            remaining = remaining.max(state.throttle_until - now);
        }
        Some(remaining)
    }

    /// The account failure holding EVERY judged circuit open, named by the
    /// first key: each circuit refuses a claim right now (inside its cooldown,
    /// or its half-open probe in flight) and runs no newer throttle window.
    pub fn circuit_failure(&self, keys: &[HealthKey], now: f64) -> Option<Failure> {
        let mut first: Option<&Failure> = None;
        for key in keys {
            let state = self.circuits.get(key)?;
            let account = state.account.as_ref()?;
            if (state.open_until <= now && !state.half_open) || state.newer_throttle_until > now {
                return None;
            }
            first.get_or_insert(account);
        }
        first.cloned()
    }

    pub fn release(&mut self, key: &HealthKey, attempt: Option<&str>) {
        if let Some(state) = self.circuits.get_mut(key) {
            state.half_open = false;
            state.last_resort = false;
            if let Some(id) = attempt {
                state.probe_attempts.remove(id);
            }
        }
        if let Some(state) = self.lanes.get_mut(&lane(key)) {
            state.probe_started = None;
        }
    }

    fn lane_failed(&mut self, key: &HealthKey, failure: &Failure, now: f64) {
        if self.policy.ejection_failures == 0 {
            return;
        }
        let dead = matches!(
            failure.failure_class,
            FailureClass::Transport | FailureClass::ProviderInternal
        );
        let hung = self.policy.ejection_header_timeouts > 0
            && failure.header_timeout == Some(HeaderTimeout::Allowance);
        let lane = lane(key);
        if !dead && !hung {
            if let Some(state) = self.lanes.get_mut(&lane) {
                state.probe_started = None;
            }
            return;
        }
        let state = self.lanes.entry(lane).or_default();
        state.probe_started = None;
        if state.ejected_until != 0.0 && now >= state.ejected_until {
            state.ejected_until = now + self.policy.ejection_cooldown_seconds;
            return;
        }
        let (streak, threshold, window) = if dead {
            (
                &mut state.failures,
                self.policy.ejection_failures,
                self.policy.ejection_window_seconds,
            )
        } else {
            (
                &mut state.headers,
                self.policy.ejection_header_timeouts,
                self.policy.ejection_header_window_seconds,
            )
        };
        streak.push_back(now);
        while streak.len() > threshold {
            streak.pop_front();
        }
        if streak.len() == threshold && now - streak.front().unwrap() <= window {
            state.ejected_until = now + self.policy.ejection_cooldown_seconds;
            state.failures.clear();
            state.headers.clear();
        }
    }
}
