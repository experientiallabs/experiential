//! Bounded, tenant-scoped cache evidence and recovery leases shared with admission.

use hashlink::LinkedHashMap;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub(super) mod python;

pub(super) type Key = [String; 4];

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
pub(super) struct Operational {
    pub provider: String,
    pub exact_model_id: String,
    pub endpoint_scope: String,
    pub region_scope: Option<String>,
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
pub(super) struct Scope {
    #[serde(flatten)]
    pub operational: Operational,
    pub credential_scope: String,
    pub organization_id: String,
}

impl Scope {
    fn negative(&self, cause: &str) -> Vec<String> {
        let mut key = vec![
            cause.into(),
            self.operational.provider.clone(),
            self.operational.exact_model_id.clone(),
            self.operational.endpoint_scope.clone(),
            self.operational.region_scope.clone().unwrap_or_default(),
        ];
        if cause != "transport" {
            key.extend([self.organization_id.clone(), self.credential_scope.clone()]);
        }
        key
    }
    fn matches(&self, other: &Operational) -> bool {
        !self.operational.endpoint_scope.is_empty()
            && self.operational.region_scope.is_some()
            && self.operational == *other
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub(super) struct Departure {
    pub scope: Scope,
    pub cause: String,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub failed_at: f64,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub retry_at: f64,
    pub causes: BTreeSet<String>,
    pub cause_observed_at: Vec<(String, f64)>,
}

#[derive(Clone, Deserialize, Serialize)]
pub(super) struct Evidence {
    pub scope: Scope,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub expires_at: f64,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub age_deadline: f64,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub observed_at: f64,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub window_started_at: f64,
}

#[derive(Default, Serialize)]
pub(super) struct History {
    pub evidence: IndexMap<String, Evidence>,
    pub departures: IndexMap<String, Departure>,
    pub retained: Option<String>,
    pub retained_until: f64,
    pub retained_age_deadline: f64,
    pub retained_observed_at: f64,
}

#[derive(Deserialize)]
pub(super) struct Observation {
    pub scope: Operational,
    pub cause: String,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub observed_at: f64,
    pub healthy: bool,
    pub authoritative: bool,
}

#[derive(Deserialize)]
pub(super) struct Lease {
    pub lease_id: String,
    pub scope: Operational,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub expires_at: f64,
}

#[derive(Deserialize)]
pub(super) struct Snapshot {
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub loaded_at: f64,
    pub observations: Vec<Observation>,
    pub leases: Vec<Lease>,
}

#[derive(Serialize)]
pub(super) struct Decision {
    pub deployment_id: Option<String>,
    pub reason: &'static str,
    pub trial: bool,
    pub warm_remaining_seconds: f64,
}
impl Default for Decision {
    fn default() -> Self {
        Self {
            deployment_id: None,
            reason: "normal_selection",
            trial: false,
            warm_remaining_seconds: 0.0,
        }
    }
}
impl Decision {
    fn recovered(id: &str, remaining: f64) -> Self {
        Self {
            deployment_id: Some(id.into()),
            reason: "recovered_preferred_route",
            trial: true,
            warm_remaining_seconds: remaining,
        }
    }
}

#[derive(Deserialize)]
pub(super) struct Success {
    pub cached_tokens: i64,
    pub cache_write_tokens: i64,
    #[serde(default, deserialize_with = "crate::lossless_json::deserialize")]
    pub retention_seconds: Option<f64>,
    #[serde(default, deserialize_with = "crate::lossless_json::deserialize")]
    pub sticky_seconds: Option<f64>,
}

pub(super) struct Registry {
    maximum: usize,
    max_age: f64,
    observation_lifetime: f64,
    cooldown: f64,
    sessions: LinkedHashMap<Key, History>,
    consumed: HashMap<String, f64>,
    negative: LinkedHashMap<Vec<String>, f64>,
}
impl Registry {
    pub fn new(
        maximum: usize,
        max_age: f64,
        observation_lifetime: f64,
        cooldown: f64,
    ) -> Option<Self> {
        if maximum == 0
            || !max_age.is_finite()
            || max_age <= 0.0
            || !observation_lifetime.is_finite()
            || observation_lifetime <= 0.0
            || !cooldown.is_finite()
        {
            return None;
        }
        Some(Self {
            maximum,
            max_age,
            observation_lifetime,
            cooldown: cooldown.max(0.0),
            sessions: LinkedHashMap::new(),
            consumed: HashMap::new(),
            negative: LinkedHashMap::new(),
        })
    }
    fn history(&mut self, key: &Key) -> &mut History {
        if !self.sessions.contains_key(key) {
            self.sessions.insert(key.clone(), History::default());
        }
        self.sessions.to_back(key);
        while self.sessions.len() > self.maximum {
            self.sessions.pop_front();
        }
        self.sessions.get_mut(key).unwrap()
    }
    pub fn depart(
        &mut self,
        key: &Key,
        deployment: &str,
        scope: Scope,
        cause: &str,
        retry: f64,
        now: f64,
    ) {
        let negative = scope.negative(cause);
        let at = now.max(self.negative.get(&negative).copied().unwrap_or(now));
        self.negative.insert(negative.clone(), at);
        self.negative.to_back(&negative);
        while self.negative.len() > self.maximum {
            self.negative.pop_front();
        }
        let wait = self.cooldown.max(retry);
        let history = self.history(key);
        if history
            .evidence
            .get(deployment)
            .is_some_and(|e| e.scope == scope && e.observed_at > now)
        {
            return;
        }
        let previous = history.departures.get(deployment).cloned();
        let mut receipts = BTreeMap::from([(cause.to_string(), now)]);
        if let Some(prior) = previous.as_ref().filter(|p| p.scope == scope) {
            for (cause, receipt) in &prior.cause_observed_at {
                receipts
                    .entry(cause.clone())
                    .and_modify(|v| *v = v.max(*receipt))
                    .or_insert(*receipt);
            }
        }
        if let Some(mut prior) = previous.clone().filter(|p| p.failed_at >= now) {
            if prior.scope == scope {
                prior.causes.insert(cause.into());
                prior.cause_observed_at = receipts.into_iter().collect();
                prior.retry_at = prior.retry_at.max(now + wait);
                history.departures.insert(deployment.into(), prior);
            }
            return;
        }
        if history.departures.len() >= 256 && !history.departures.contains_key(deployment) {
            history.departures.shift_remove_index(0);
        }
        let mut causes = BTreeSet::from([cause.to_string()]);
        let mut retry_at = now + wait;
        if let Some(prior) = previous.filter(|p| p.scope == scope) {
            causes.extend(prior.causes);
            retry_at = retry_at.max(prior.retry_at);
        }
        history.departures.insert(
            deployment.into(),
            Departure {
                scope,
                cause: cause.into(),
                failed_at: now,
                retry_at,
                causes,
                cause_observed_at: receipts.into_iter().collect(),
            },
        );
    }
    pub fn success(
        &mut self,
        key: &Key,
        deployment: &str,
        scope: Scope,
        meter: Success,
        observed: f64,
        current: f64,
    ) {
        let Some(retention) = meter.retention_seconds.filter(|v| *v > 0.0) else {
            return;
        };
        if scope.credential_scope.is_empty()
            || meter.cached_tokens.max(meter.cache_write_tokens) <= 0
            || observed > current
        {
            return;
        }
        let ttl = retention.min(self.max_age);
        let expired = observed + ttl <= current;
        if !expired {
            self.history(key);
        }
        let Some(history) = self.sessions.get_mut(key) else {
            return;
        };
        let old = history.evidence.get(deployment).cloned();
        let mut age = observed + 4.0 * ttl;
        if let Some(mut prior) = old.clone().filter(|e| e.observed_at > observed) {
            if prior.scope == scope && observed + ttl > prior.window_started_at {
                age = age.min(prior.age_deadline);
                prior.expires_at = prior.expires_at.min(age);
                prior.age_deadline = age;
                prior.window_started_at = observed.min(prior.window_started_at);
                history.evidence.insert(deployment.into(), prior);
                if history.retained.as_deref() == Some(deployment) {
                    if let Some(sticky) = meter.sticky_seconds.filter(|v| *v > 0.0) {
                        history.retained_age_deadline = history
                            .retained_age_deadline
                            .min(observed + 4.0 * ttl.min(sticky));
                        history.retained_until = history
                            .retained_until
                            .min(history.retained_age_deadline)
                            .min(age);
                    }
                }
            }
            return;
        }
        if expired {
            return;
        }
        let mut window_started_at = observed;
        if let Some(prior) = old
            .as_ref()
            .filter(|e| e.scope == scope && e.expires_at > observed)
        {
            age = age.min(prior.age_deadline);
            window_started_at = prior.window_started_at;
        }
        history.evidence.insert(
            deployment.into(),
            Evidence {
                scope: scope.clone(),
                expires_at: (observed + ttl).min(age),
                age_deadline: age,
                observed_at: observed,
                window_started_at,
            },
        );
        if let Some(mut prior) = history
            .departures
            .get(deployment)
            .cloned()
            .filter(|p| p.scope == scope)
        {
            let receipts: HashMap<_, _> = prior.cause_observed_at.iter().cloned().collect();
            prior.causes.retain(|cause| {
                receipts.get(cause).copied().unwrap_or(prior.failed_at).max(
                    self.negative
                        .get(&scope.negative(cause))
                        .copied()
                        .unwrap_or(observed),
                ) >= observed
            });
            if prior.causes.is_empty() {
                history.departures.shift_remove(deployment);
            } else {
                prior
                    .cause_observed_at
                    .retain(|(cause, _)| prior.causes.contains(cause));
                history.departures.insert(deployment.into(), prior);
            }
        }
        if history.evidence.len() > 256 {
            history.evidence.shift_remove_index(0);
        }
        if let Some(sticky) = meter.sticky_seconds.filter(|v| *v > 0.0) {
            if observed < history.retained_observed_at
                || history
                    .departures
                    .get(deployment)
                    .is_some_and(|d| d.failed_at >= observed)
            {
                return;
            }
            let lifetime = ttl.min(sticky);
            if history.retained.as_deref() != Some(deployment)
                || old.as_ref().is_none_or(|e| e.scope != scope)
            {
                history.retained_age_deadline = observed + 4.0 * lifetime;
            }
            history.retained = Some(deployment.into());
            history.retained_observed_at = observed;
            history.retained_until = (observed + lifetime)
                .min(history.retained_age_deadline)
                .min(age);
        }
    }
    pub fn retained(&self, key: &Key, live: bool, now: f64) -> bool {
        self.sessions
            .get(key)
            .is_some_and(|h| h.retained.is_some() && (!live || h.retained_until > now))
    }
    pub fn inspect(&self, key: &Key) -> Option<&History> {
        self.sessions.get(key)
    }
    pub fn choose<E>(
        &mut self,
        key: &Key,
        candidates: &[(String, Scope)],
        mut eligible: impl FnMut(&str) -> Result<bool, E>,
        snapshot: Option<&Snapshot>,
        mut capacity: Option<impl FnMut(&str) -> Result<bool, E>>,
        now: f64,
    ) -> Result<Decision, E> {
        let Some(history) = self.sessions.get(key).filter(|h| h.retained.is_some()) else {
            return Ok(Decision::default());
        };
        if history.retained_until <= now {
            self.sessions.to_back(key).unwrap().retained = None;
            return Ok(Decision {
                reason: "cache_expired",
                ..Decision::default()
            });
        }
        self.sessions.to_back(key);
        let history = self.sessions.get(key).unwrap();
        let retained = history.retained.as_ref().unwrap();
        let scopes: HashMap<_, _> = candidates.iter().map(|(id, scope)| (id, scope)).collect();
        let Some(evidence) = history
            .evidence
            .get(retained)
            .filter(|e| e.expires_at > now && scopes.get(retained).is_some_and(|s| **s == e.scope))
        else {
            return Ok(Decision::default());
        };
        if !eligible(retained)?
            || history.departures.get(retained).is_some_and(|d| {
                d.scope == evidence.scope
                    && d.failed_at >= evidence.observed_at
                    && ["transport", "credential", "throttle"]
                        .iter()
                        .any(|c| d.causes.contains(*c))
            })
        {
            return Ok(Decision::default());
        }
        for (id, scope) in candidates {
            if id == retained {
                break;
            }
            let Some(warm) = history
                .evidence
                .get(id)
                .filter(|e| e.expires_at > now && e.scope == *scope)
            else {
                continue;
            };
            let Some(departed) = history.departures.get(id) else {
                continue;
            };
            if !eligible(id)? || departed.scope != *scope || now < departed.retry_at {
                continue;
            }
            if let Some(capacity) = capacity.as_mut() {
                if !capacity(id)? {
                    continue;
                }
            }
            let mut causes = departed.causes.clone();
            for cause in ["transport", "credential", "throttle"] {
                if self
                    .negative
                    .get(&scope.negative(cause))
                    .is_some_and(|at| *at >= warm.observed_at)
                {
                    causes.insert(cause.into());
                }
            }
            if causes.contains("credential") || causes.contains("throttle") {
                continue;
            }
            if !causes.contains("transport") {
                if let Some(capacity) = capacity.as_mut() {
                    if capacity(id)? {
                        return Ok(Decision::recovered(id, warm.expires_at - now));
                    }
                }
                continue;
            }
            let Some(snapshot) = snapshot.filter(|s| {
                0.0 <= now - s.loaded_at && now - s.loaded_at <= self.observation_lifetime
            }) else {
                continue;
            };
            let Some(negative) = self.negative.get(&scope.negative("transport")) else {
                continue;
            };
            let observations: Vec<_> = snapshot
                .observations
                .iter()
                .filter(|o| {
                    o.cause == "transport"
                        && scope.matches(&o.scope)
                        && departed.failed_at.max(*negative) < o.observed_at
                        && o.observed_at <= now
                        && now - o.observed_at <= self.observation_lifetime
                })
                .collect();
            let Some(latest) = observations.iter().map(|o| o.observed_at).reduce(f64::max) else {
                continue;
            };
            if !observations
                .iter()
                .filter(|o| o.observed_at == latest)
                .all(|o| o.healthy && (departed.cause != "credential" || o.authoritative))
            {
                continue;
            }
            self.consumed.retain(|_, expiry| *expiry > now);
            for lease in &snapshot.leases {
                if !self.consumed.contains_key(&lease.lease_id)
                    && lease.expires_at > now
                    && scope.operational == lease.scope
                {
                    if self.consumed.len() >= self.maximum {
                        break;
                    }
                    self.consumed
                        .insert(lease.lease_id.clone(), lease.expires_at);
                    return Ok(Decision::recovered(id, warm.expires_at - now));
                }
            }
        }
        Ok(Decision {
            deployment_id: Some(retained.clone()),
            reason: "retained_warm_fallback",
            trial: false,
            warm_remaining_seconds: evidence.expires_at.min(history.retained_until) - now,
        })
    }
}
