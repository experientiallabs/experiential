//! Physical dispatch selection, reservation and settlement under one request owner.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::errors::{Failure, FailureClass, PublicError};
use crate::native_callbacks::NativeTarget;
use crate::waterfall::fallback_rules;

use super::health::{Claim, HealthKey};
use super::loads::Shed;
use super::ownership::Driver;
use super::route_config::Registration;
use super::selection::{self, FailoverPass, Retry, Throttle};
use super::shared::{Policies, Shared};
use super::{prices, rung, terminal};

pub(super) struct Attempt {
    pub depth: usize,
    pub terms: terminal::Terms,
    pub funded: bool,
    pub terminal: Option<Value>,
    pub recovery_observed: Option<f64>,
    pub recovery_failure: Option<Failure>,
    pub recovery_recorded: bool,
    pub recovery_key: Option<super::session_recovery::Key>,
    pub result: Option<Value>,
    pub fleet: Option<String>,
    pub cancelled: bool,
    pub late_terminal: Option<Value>,
}

pub(super) struct State {
    pub total: u32,
    pub physical: Vec<u32>,
    pub ordinary: Vec<u32>,
    pub redials: Vec<u32>,
    pub denied: HashSet<String>,
    pub attempts: HashMap<String, Attempt>,
    pub active: Option<String>,
    pub final_depth: Option<usize>,
    pub features: super::features::Features,
    pub search: Option<super::search_dispatch::Session>,
    pub input_tokens: u64,
    pub counted_input_tokens: u64,
    pub recovery_key: Option<super::session_recovery::Key>,
}

pub(super) struct Dispatch {
    pub config: Registration,
    pub shared: Arc<Shared>,
    pub host: Arc<NativeTarget>,
    admitted: f64,
    state: Mutex<State>,
    admitted_attempts: Mutex<HashMap<String, terminal::Terms>>,
    pub(super) inspection: Option<super::inspection::Inspection>,
    pub(super) deadline: std::time::Instant,
    pub(super) resources: super::resources::Resources,
    cache: Option<super::cache::Cache>,
}

#[derive(Deserialize)]
struct Start {
    attempt_ordinal: u32,
    current_depth: Option<usize>,
    failure: Option<Failure>,
    #[serde(default)]
    throttle_backoff: bool,
    #[serde(default)]
    tool_search_round: bool,
    #[serde(default)]
    reasoning_repair: bool,
}

impl Dispatch {
    #[cfg(test)]
    pub fn new(
        config: Registration,
        shared: Arc<Shared>,
        host: Arc<NativeTarget>,
    ) -> Result<Arc<Self>, PublicError> {
        Self::with_resources(config, shared, host, super::resources::Resources::default())
    }

    pub fn with_resources(
        mut config: Registration,
        shared: Arc<Shared>,
        host: Arc<NativeTarget>,
        resources: super::resources::Resources,
    ) -> Result<Arc<Self>, PublicError> {
        config.validate()?;
        if [
            "start_attempt",
            "settle",
            "abandon",
            "price",
            "record_late_meter",
        ]
        .iter()
        .any(|method| !host.handles(method))
            || (config.features.continuation.is_some() && !host.handles("remember"))
            || (config.features.cache.is_some() && !host.handles("cache_transaction"))
            || (config.lanes.iter().any(|l| l.policy.fleet.is_some())
                && ["fleet_acquire", "fleet_release"]
                    .iter()
                    .any(|method| !host.handles(method)))
        {
            return Err(PublicError::internal());
        }
        let n = config.lanes.len();
        let deadline = std::time::Instant::now()
            + std::time::Duration::from_secs_f64(config.remaining_seconds);
        let inspection = config.features.output_checks.as_ref().map(|checks| {
            let mut inspection = super::inspection::Inspection::new(checks, deadline);
            inspection.reporting = resources.reporting.clone();
            inspection
        });
        let mut features = std::mem::take(&mut config.features);
        let search = features
            .search
            .take()
            .map(super::search_dispatch::Session::new)
            .transpose()?;
        let cache = features.cache.take().map(super::cache::Cache::new);
        let input_tokens = config.input_tokens;
        let counted_input_tokens = config.counted_input_tokens;
        let recovery_key = config
            .recovery
            .as_ref()
            .filter(|r| !r.key[3].is_empty())
            .map(|r| r.key.clone());
        Ok(Arc::new(Self {
            config,
            admitted: shared.now(),
            shared,
            host,
            inspection,
            deadline,
            resources,
            cache,
            admitted_attempts: Mutex::new(HashMap::new()),
            state: Mutex::new(State {
                total: 0,
                physical: vec![0; n],
                ordinary: vec![0; n],
                redials: vec![0; n],
                denied: HashSet::new(),
                attempts: HashMap::new(),
                active: None,
                final_depth: None,
                features,
                search,
                input_tokens,
                counted_input_tokens,
                recovery_key,
            }),
        }))
    }

    pub(super) fn host(&self, method: &str, input: &Value) -> Result<Value, PublicError> {
        crate::lossless_json::from_str(&self.host.call(method, &input.to_string())?)
            .map_err(|_| PublicError::internal())
    }

    fn start(&self, data: &Value, state: &mut State) -> Result<Value, PublicError> {
        let start: Start =
            crate::lossless_json::from_value(data.clone()).map_err(|_| PublicError::internal())?;
        let c = &self.config;
        if start.attempt_ordinal != state.total
            || state.active.is_some()
            || start.current_depth.is_some_and(|d| d >= c.lanes.len())
        {
            return Err(PublicError::internal());
        }
        let selected = c.selected_route && state.total == 0;
        let keys: Vec<_> = c.lanes.iter().map(|l| l.key.clone()).collect();
        let mut ladder = if selected {
            vec![0]
        } else {
            selection::eligible(&c.rules, start.failure.as_ref(), true)
        };
        let tool_round =
            start.tool_search_round && start.current_depth.is_some() && start.failure.is_none();
        let repair = start.reasoning_repair && start.current_depth.is_some();
        let mut failover = FailoverPass::begin(
            &c.rules,
            start.failure.as_ref(),
            start.current_depth,
            &state.physical,
            c.refusal_failover,
        );
        let (mut candidate, disposition) = {
            let mut policies = self
                .shared
                .policies
                .lock()
                .map_err(|_| PublicError::internal())?;
            let now = self.shared.now();
            if let (Some(failure), Some(current)) = (&start.failure, start.current_depth) {
                let cached = policies.loads.cached_fraction(
                    &c.lanes[current].load_key(),
                    &c.organization_id,
                    now,
                );
                selection::next(
                    &mut policies.health,
                    &keys,
                    &Retry {
                        failure,
                        current,
                        ordinary_counts: &state.ordinary,
                        physical_counts: &state.physical,
                        total: state.total,
                        policy: &c.attempt_policy,
                        refusal_failover: c.refusal_failover,
                        stage: &c.lanes[current].stage,
                        cached_fraction: cached,
                        throttle_backoff: start.throttle_backoff,
                        redial_budget: c.redial_budgets[current]
                            .saturating_sub(state.redials[current]),
                        rules: &c.rules,
                    },
                    now,
                )
            } else if selected {
                (
                    policies
                        .health
                        .claim(&keys[0], Claim::Normal, now)
                        .then_some(0),
                    None,
                )
            } else {
                if repair {
                    ladder = vec![start.current_depth.unwrap()];
                }
                (
                    selection::claim(
                        &mut policies.health,
                        &keys,
                        if tool_round || repair {
                            start.current_depth.unwrap()
                        } else {
                            0
                        },
                        &ladder,
                        now,
                    ),
                    None,
                )
            }
        };
        if let Some(disposition) = disposition {
            super::counters::increment(match disposition {
                Throttle::Surfaced => &self.shared.counters.surfaced,
                Throttle::Cold => &self.shared.counters.cold,
                Throttle::Backoff => &self.shared.counters.backoff,
            });
        }
        let redial = (disposition == Some(Throttle::Backoff))
            .then_some(start.current_depth)
            .flatten();
        let mut sheds = Vec::<(usize, String)>::new();
        if disposition == Some(Throttle::Cold) {
            sheds.push((start.current_depth.unwrap(), Throttle::Cold.reason().into()));
        }
        let mut records = HashMap::<usize, Shed>::new();
        let mut forced = false;
        let mut capacity = false;
        let mut failure = start.failure.clone();
        loop {
            let now = self.shared.now();
            if now >= self.admitted + c.remaining_seconds {
                if let Some(depth) = candidate {
                    self.release_probe(&keys[depth])?;
                }
                failure = Some(Failure::new(FailureClass::Timeout,
                    "the request reached the gateway's time limit before any provider finished answering; retry the request"));
                break;
            }
            if candidate.is_none() {
                if !sheds.is_empty() && failure.is_none() && !forced {
                    candidate = if selected && c.priority == 0 {
                        None
                    } else {
                        self.overflow(&sheds, &records)
                    };
                    forced = candidate.is_some();
                    if candidate.is_none() {
                        capacity = true;
                        failure = Some(self.capacity());
                        break;
                    }
                } else {
                    if let Some(pass) = &mut failover {
                        let mut policies = self
                            .shared
                            .policies
                            .lock()
                            .map_err(|_| PublicError::internal())?;
                        candidate =
                            pass.forced_target(&sheds, &records, &mut policies.health, &keys, now);
                        if candidate.is_none() && pass.capacity_refusal(&records) {
                            failure = Some(self.capacity());
                        }
                    }
                    if candidate.is_none() {
                        break;
                    }
                    forced = true;
                }
            }
            let depth = candidate.unwrap();
            let lane = &c.lanes[depth];
            if state.denied.contains(&lane.pool_id) {
                self.release_probe(&keys[depth])?;
                candidate = self.sideways(&mut failover, &keys, &ladder, depth)?;
                continue;
            }
            if !c.attempt_policy.permits(state.total, state.physical[depth]) {
                self.release_probe(&keys[depth])?;
                failure.get_or_insert_with(|| {
                    let mut f = Failure::new(FailureClass::InvalidRequest,
                        "The request exhausted gateway.retry attempt limits before producing an answer. Increase the requested limits and resend.");
                    f.rejected_parameter = Some("gateway.retry".into()); f
                });
                break;
            }
            let wall = SystemTime::now();
            let (prices, tier, maximum) = match prices::select(
                &lane.prices,
                c.service_tier.as_deref(),
                lane.forwards_tier,
                lane.customer_managed,
                &if state.search.is_some() {
                    lane.ceiling.search_input(state.input_tokens)?
                } else {
                    lane.ceiling.clone()
                },
                wall,
            ) {
                Ok(value) => value,
                Err(error) => {
                    self.release_probe(&keys[depth])?;
                    return Err(error);
                }
            };
            let ticket = {
                let mut policies = self
                    .shared
                    .policies
                    .lock()
                    .map_err(|_| PublicError::internal())?;
                let Policies { loads, sticky, .. } = &mut *policies;
                rung::reserve(
                    loads,
                    sticky,
                    c,
                    lane,
                    state.input_tokens,
                    forced,
                    forced && Some(depth) == redial,
                    failover.as_ref().is_some_and(|p| p.forcing),
                    state.total == 0,
                    self.admitted,
                    now,
                )
            };
            let (ticket, fleet) = self.fleet(depth, ticket, state.input_tokens)?;
            let ticket = match ticket {
                Ok(ticket) => ticket,
                Err(shed) => {
                    super::counters::increment(&self.shared.counters.sheds);
                    if shed.reason == "rate_limit" {
                        super::counters::increment(&self.shared.counters.rates);
                    }
                    if shed.reason == "fresh_session" {
                        super::counters::increment(&self.shared.counters.fresh);
                    }
                    if let (Some(recovery), Some(key)) = (&c.recovery, &state.recovery_key) {
                        recovery.departed(
                            key,
                            &self.shared,
                            depth,
                            &lane.key.1,
                            "local_capacity",
                            0.0,
                            super::session_observations::now(),
                        );
                    }
                    sheds.push((depth, shed.reason.into()));
                    self.release_probe(&keys[depth])?;
                    forced = !selected
                        && (shed.fleet_refuse
                            || (!shed.fleet
                                && ((Some(depth) == redial && shed.reason == "rate_limit")
                                    || (failure.is_none() && lane.keeps_pin))));
                    let retry_rate = Some(depth) == redial && shed.reason == "rate_limit";
                    records.insert(depth, shed);
                    if forced
                        && !retry_rate
                        && self
                            .overflow(&[(depth, records[&depth].reason.into())], &records)
                            .is_none()
                    {
                        capacity = true;
                        failure = Some(self.capacity());
                        break;
                    }
                    if !forced {
                        candidate = self.sideways(&mut failover, &keys, &ladder, depth)?;
                    }
                    continue;
                }
            };
            let (mut reason, mut preferred) = selection::disclosure(
                &lane.stage,
                depth,
                &sheds,
                forced,
                c.sticky_preferred,
                Some(depth) == redial,
            );
            if tool_round {
                reason = Some("tool_search_round".into());
            } else if Some(depth) != start.current_depth
                && (reason.is_none() || reason.as_deref() == Some("rung_dead"))
            {
                let policies = self
                    .shared
                    .policies
                    .lock()
                    .map_err(|_| PublicError::internal())?;
                let eligible = failover.as_ref().map_or(&ladder, |p| &p.untried);
                if let Some(before) = eligible
                    .iter()
                    .take_while(|d| **d != depth)
                    .find(|d| policies.health.ejected(&keys[**d], now))
                {
                    reason = Some("lane_ejected".into());
                    preferred = Some(*before);
                }
            }
            if depth == 0
                && state.total == 0
                && matches!(
                    reason.as_deref(),
                    None | Some("affinity" | "affinity_sticky")
                )
                && c.recovery_reason.is_some()
            {
                reason = c.recovery_reason.clone();
            }
            let reserved = self.host("start_attempt", &json!({"request_id": c.request_id, "dispatch": {
                "ordinal": state.total, "depth": depth, "maximum_cost_nano": maximum,
                "reserved_input_tokens": state.input_tokens, "reserved_output_tokens": lane.output_tokens,
                "route_reason": lane.route_reason, "fallback_reason": self.fallback(depth, &start),
                "dispatch_reason": reason, "preferred_deployment_id": preferred.map(|d| &c.lanes[d].key.1),
                "prices": prices, "service_tier": tier, "started_at": DateTime::<Utc>::from(wall),
            }}));
            let reserved = match reserved {
                Ok(value) => value,
                Err(error) => {
                    self.release_unbound(&keys[depth], ticket.as_deref(), fleet.as_deref())?;
                    return Err(error);
                }
            };
            if let Some(refused) = reserved.get("refused") {
                self.release_unbound(&keys[depth], ticket.as_deref(), fleet.as_deref())?;
                match refused["scope_kind"].as_str() {
                    Some("pool") if refused["scope_id"].as_str() == Some(lane.pool_id.as_str()) => {
                        state.denied.insert(lane.pool_id.clone());
                    }
                    Some("deployment")
                        if refused["scope_id"].as_str() == Some(lane.key.1.as_str()) => {}
                    Some("request") => {
                        let error: PublicError =
                            crate::lossless_json::from_value(refused["error"].clone())
                                .map_err(|_| PublicError::internal())?;
                        // A request-wide refusal is the customer's terminal. The owner
                        // keeps it as this request's intent before its single write,
                        // then answers with the refusal.
                        let quota = Failure::new(
                            FailureClass::QuotaExceeded,
                            "monthly gateway allocation is exhausted",
                        );
                        return Ok(json!({"refused": {"failure": quota, "error": error}}));
                    }
                    _ => return Err(PublicError::internal()),
                }
                failure = Some(
                    if refused["scope_kind"] == "pool" || depth + 1 == c.lanes.len() {
                        Failure::new(
                            FailureClass::QuotaExceeded,
                            "monthly gateway allocation is exhausted",
                        )
                    } else {
                        Failure::new(
                            FailureClass::ProviderInternal,
                            "all certified routes are temporarily unavailable",
                        )
                    },
                );
                forced = false;
                candidate = self.sideways(&mut failover, &keys, &ladder, depth)?;
                continue;
            }
            let Some(id) = reserved["attempt_id"]
                .as_str()
                .filter(|id| !id.is_empty() && !state.attempts.contains_key(*id))
            else {
                self.release_unbound(&keys[depth], ticket.as_deref(), fleet.as_deref())?;
                return Err(PublicError::internal());
            };
            let id = id.to_string();
            let pace = {
                let mut policies = self
                    .shared
                    .policies
                    .lock()
                    .map_err(|_| PublicError::internal())?;
                if let Some(ticket) = &ticket {
                    policies.loads.bind(ticket, &id);
                }
                if c.bind_sticky {
                    if let (Some(fingerprint), Some(ttl)) =
                        (&c.fingerprint, lane.policy.sticky_spill_seconds)
                    {
                        policies.sticky.bind(
                            fingerprint.as_bytes(),
                            &lane.key.1,
                            ttl,
                            self.shared.now(),
                        );
                    }
                }
                rung::pace(&policies.loads, c, lane)
            };
            state.physical[depth] += 1;
            if !tool_round && !repair {
                state.ordinary[depth] += 1;
            }
            if Some(depth) == redial {
                state.redials[depth] += 1;
            }
            if forced {
                super::counters::increment(&self.shared.counters.overflows);
            }
            if forced && Some(depth) == redial {
                super::counters::increment(&self.shared.counters.forced_backoff);
            }
            state.total += 1;
            state.active = Some(id.clone());
            state.attempts.insert(
                id.clone(),
                Attempt {
                    depth,
                    funded: reserved["promo_funded"] == true,
                    terminal: None,
                    recovery_observed: None,
                    recovery_failure: None,
                    recovery_recorded: false,
                    recovery_key: state.recovery_key.clone(),
                    result: None,
                    fleet,
                    cancelled: false,
                    late_terminal: None,
                    terms: terminal::Terms {
                        surface: c.surface.clone(),
                        counted_input_tokens: state.counted_input_tokens,
                        output_ceiling: lane.output_ceiling,
                        output_budget: Some(lane.output_tokens),
                        service_tier: tier,
                        self_served: lane.fast(c)
                            || (lane.policy.priority_tier_attested && lane.priority_card()),
                        paid_prework: c.paid_prework,
                        web_search_requests: c.web_search_requests,
                    },
                },
            );
            self.admitted_attempts
                .lock()
                .map_err(|_| PublicError::internal())?
                .insert(id.clone(), state.attempts[&id].terms.clone());
            let mut result = json!({"attempt_id": id, "route_depth": depth});
            if let Some(pace) = pace {
                result["output_tokens_per_second"] = json!(pace);
            }
            return Ok(result);
        }
        let failure = match failure {
            Some(failure) => failure,
            None => self.unclaimed(&keys, &ladder, selected, repair)?,
        };
        // The request owner retains this terminal before the first write so
        // failed or incomplete acknowledgements remain recoverable.
        Ok(json!({"exhausted": true, "failure": failure,
            "certify_no_effects": capacity && !c.paid_prework}))
    }

    /// Answers a walk that found no claimable rung and has no failure of its
    /// own. The account failure holding every claimable circuit open wins;
    /// else a throttle window over the claimable depths, naming the caller's
    /// controls that kept the request off other rungs; else unavailable.
    ///
    /// A selected first dial may claim only its lead, so the selector is named
    /// when another control narrowed the route or a first-dial sibling sits
    /// outside a throttle window. A reasoning-repair redial may claim only the
    /// rung it repairs whatever the caller sent, so it names no control.
    fn unclaimed(
        &self,
        keys: &[HealthKey],
        ladder: &[usize],
        first: bool,
        repair: bool,
    ) -> Result<Failure, PublicError> {
        let judged = |depths: &[usize]| depths.iter().map(|d| keys[*d].clone()).collect::<Vec<_>>();
        let policies = self
            .shared
            .policies
            .lock()
            .map_err(|_| PublicError::internal())?;
        let now = self.shared.now();
        let claimable = judged(ladder);
        if let Some(account) = policies.health.circuit_failure(&claimable, now) {
            return Ok(account);
        }
        let Some(remaining) = policies.health.throttled_remaining(&claimable, now) else {
            return Ok(Failure::new(
                FailureClass::ProviderInternal,
                "all exact-model deployments are unavailable",
            ));
        };
        let mut controls: Vec<&str> = self
            .config
            .caller_narrowing
            .iter()
            .map(String::as_str)
            .collect();
        // Only a sibling an unpinned first dial may claim makes removing the selector a remedy.
        let siblings: Vec<usize> = selection::eligible(&self.config.rules, None, true)
            .into_iter()
            .filter(|depth| *depth != 0)
            .collect();
        let sibling_free = !siblings.is_empty()
            && policies
                .health
                .throttled_remaining(&judged(&siblings), now)
                .is_none();
        if first && (!controls.is_empty() || sibling_free) {
            controls.insert(0, "gateway.routing.route_id");
        }
        let seconds = remaining.ceil().max(5.0) as u32;
        let message = if repair || controls.is_empty() {
            format!(
                "all exact-model deployments are inside a provider throttle window; retry in {seconds}s"
            )
        } else {
            let selection = controls.join(" and ");
            format!(
                "the route selected by {selection} is inside a provider throttle window; \
                 retry in {seconds}s, or remove {selection} to let the gateway use this \
                 model's other routes"
            )
        };
        let mut failure = Failure::new(FailureClass::Throttled, &message);
        failure.retry_after_seconds = Some(seconds);
        Ok(failure)
    }

    fn capacity(&self) -> Failure {
        super::counters::increment(&self.shared.counters.refusals);
        let mut failure = Failure::new(FailureClass::Throttled, &self.config.capacity_message);
        failure.retry_after_seconds = Some(5);
        failure
    }
    fn overflow(&self, sheds: &[(usize, String)], records: &HashMap<usize, Shed>) -> Option<usize> {
        selection::overflow_target(
            self.config.priority,
            self.config.selected_route,
            &self
                .config
                .lanes
                .iter()
                .map(|l| l.policy.refusing)
                .collect::<Vec<_>>(),
            sheds,
            records,
        )
    }
    fn release_probe(&self, key: &HealthKey) -> Result<(), PublicError> {
        self.shared
            .policies
            .lock()
            .map_err(|_| PublicError::internal())?
            .health
            .release(key);
        Ok(())
    }
    fn sideways(
        &self,
        failover: &mut Option<FailoverPass>,
        keys: &[HealthKey],
        ladder: &[usize],
        depth: usize,
    ) -> Result<Option<usize>, PublicError> {
        let mut p = self
            .shared
            .policies
            .lock()
            .map_err(|_| PublicError::internal())?;
        Ok(match failover {
            Some(pass) => pass.sideways(&mut p.health, keys, depth, self.shared.now()),
            None => selection::claim(&mut p.health, keys, depth + 1, ladder, self.shared.now()),
        })
    }
    fn fallback(&self, depth: usize, start: &Start) -> Option<String> {
        let Some(failure) = start.failure.as_ref() else {
            return self.config.fallback_reason.clone();
        };
        if start.current_depth.is_none_or(|d| d == depth) {
            return self.config.fallback_reason.clone();
        }
        let Some(rules) = self.config.rules[depth].as_ref() else {
            return self.config.fallback_reason.clone();
        };
        let token = fallback_rules::failure_token(failure);
        let matched = rules.iter().find(|r| r.as_str() == token).or_else(|| {
            rules
                .iter()
                .find(|r| r.as_str() == "refusal" && failure.failure_class == FailureClass::Refusal)
        });
        matched
            .map(|token| format!("failover_only_on:{token}"))
            .or_else(|| self.config.fallback_reason.clone())
    }
}

impl Driver for Dispatch {
    fn validate_terminal(&self, input: &Value) -> Result<(), PublicError> {
        let id = input["attempt_id"]
            .as_str()
            .ok_or_else(PublicError::internal)?;
        let terms = self
            .admitted_attempts
            .lock()
            .map_err(|_| PublicError::internal())?
            .get(id)
            .cloned()
            .ok_or_else(PublicError::internal)?;
        terminal::validate(input, &terms)
    }
    fn call(&self, method: &str, input: &Value) -> Result<Value, PublicError> {
        if [
            "prepare_explicit_cache",
            "finish_explicit_cache",
            "renew_explicit_cache",
        ]
        .contains(&method)
        {
            let Some(cache) = &self.cache else {
                return if method == "prepare_explicit_cache" {
                    Ok(json!({"state": "disabled"}))
                } else {
                    Err(PublicError::internal())
                };
            };
            return cache.call(
                method,
                input,
                &self.config.request_id,
                &|transaction| self.host("cache_transaction", &transaction),
                &|| {
                    if std::time::Instant::now() >= self.deadline {
                        return Ok(None);
                    }
                    let state = self.state.lock().map_err(|_| PublicError::internal())?;
                    Ok(state
                        .active
                        .as_ref()
                        .and_then(|id| state.attempts.get(id).map(|a| (id, a)))
                        .filter(|(_, a)| !a.cancelled && a.terminal.is_none())
                        .map(|(id, a)| super::cache::Active {
                            attempt: id.clone(),
                            depth: a.depth,
                            deployment: self.config.lanes[a.depth].key.1.clone(),
                        }))
                },
            );
        }
        if method == "guardrail_input_status" {
            return match self.state.try_lock() {
                Ok(_guard) => Ok(self.resources.status()),
                Err(std::sync::TryLockError::WouldBlock) => Ok(json!({"action": "pending"})),
                Err(_) => Err(PublicError::internal()),
            };
        }
        if method == "tool_search_round" {
            if let Some(input) = &self.resources.input {
                input.wait();
            }
        }
        let mut state = self.state.lock().map_err(|_| PublicError::internal())?;
        match method {
            "start_attempt" => self.start(input, &mut state),
            "price" | "settle" => self.settle(method, input, &mut state),
            "abandon" => {
                let certify = input["certify_no_effects"] == true
                    && state.total == 0
                    && !self.config.paid_prework;
                self.finish(input, &mut state, certify)
            }
            "tool_search_round" => self.search_round(input, &mut state),
            "enforce_output" => self
                .inspection
                .as_ref()
                .ok_or_else(PublicError::internal)?
                .output(input),
            "enforce_output_segment" => self
                .inspection
                .as_ref()
                .ok_or_else(PublicError::internal)?
                .segment(input),
            "remember" => {
                let depth = state
                    .active
                    .as_ref()
                    .and_then(|id| state.attempts.get(id))
                    .map(|a| a.depth)
                    .or(state.final_depth);
                let Some(context) = &state.features.continuation else {
                    return Ok(json!({}));
                };
                match context.remember(&self.config.request_id, depth, input)? {
                    Some(payload) => self.host("remember", &payload),
                    None => Ok(json!({})),
                }
            }
            _ => {
                let depth = state
                    .active
                    .as_ref()
                    .and_then(|id| state.attempts.get(id))
                    .map(|a| a.depth);
                state
                    .features
                    .call(method, &self.config.request_id, depth, input)
                    .unwrap_or_else(|| self.host(method, input))
            }
        }
    }
}
