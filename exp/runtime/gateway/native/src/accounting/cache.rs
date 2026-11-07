//! Request-local cache claims; host I/O never owns the execution lock.

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, MutexGuard};

use serde::Deserialize;
use serde_json::{json, Value};

use super::cache_plan::{Binding, Configuration, Plan};
use super::cache_policy::{self as policy, Authority, Claim, Observation, Offer, Result};
use crate::errors::PublicError;

#[derive(Clone, PartialEq)]
pub(super) struct Active {
    pub attempt: String,
    pub depth: usize,
    pub deployment: String,
}

#[derive(Clone)]
struct Operation {
    active: Active,
    offer: Offer,
    binding: Binding,
    observed: Option<Observation>,
    recording: bool,
    recorded: bool,
}

#[derive(Default)]
struct State {
    prepared: HashSet<String>,
    operations: HashMap<String, Operation>,
    renewals: HashMap<String, (f64, f64)>,
}

pub(super) struct Cache {
    config: Configuration,
    state: Mutex<State>,
    clock: fn() -> Result<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Boundary {
    request_id: String,
    deployment_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Finish {
    request_id: String,
    deployment_id: String,
    operation_id: String,
    outcome: String,
    http_status: Option<u16>,
    name: Option<String>,
    expire_time: Option<String>,
    create_time: Option<String>,
    total_tokens: Option<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Renewed {
    request_id: String,
    deployment_id: String,
    operation_id: String,
    outcome: String,
    expire_time: Option<String>,
}

impl Cache {
    pub fn new(config: Configuration) -> Self {
        Self {
            config,
            state: Mutex::default(),
            clock: policy::now,
        }
    }
    fn state(&self) -> Result<MutexGuard<'_, State>> {
        self.state.lock().map_err(|_| PublicError::internal())
    }

    pub fn call(
        &self,
        method: &str,
        input: &Value,
        request: &str,
        host: &impl Fn(Value) -> Result<Value>,
        active: &impl Fn() -> Result<Option<Active>>,
    ) -> Result<Value> {
        let result = (|| {
            policy::require(input.to_string().len() <= 16384)?;
            match method {
                "prepare_explicit_cache" => self.prepare(input, request, host, active),
                "finish_explicit_cache" => self.finish(input, request, host, active),
                "renew_explicit_cache" => self.renew(input, request, host),
                _ => Err(PublicError::internal()),
            }
        })();
        if result.is_err() && (self.config.automatic || method == "renew_explicit_cache") {
            Ok(unavailable())
        } else {
            result.map_err(|_| boundary_error())
        }
    }

    fn prepare(
        &self,
        input: &Value,
        request: &str,
        host: &impl Fn(Value) -> Result<Value>,
        current: &impl Fn() -> Result<Option<Active>>,
    ) -> Result<Value> {
        let selected: Boundary =
            crate::lossless_json::from_value(input.clone()).map_err(|_| PublicError::internal())?;
        boundary(request, &selected.request_id, &selected.deployment_id)?;
        let active = current()?.ok_or_else(PublicError::internal)?;
        let mut binding = self
            .config
            .bindings
            .get(active.depth)
            .and_then(Option::as_ref)
            .ok_or_else(PublicError::internal)?
            .clone();
        policy::require(
            selected.deployment_id == binding.deployment_id
                && active.deployment == binding.deployment_id,
        )?;
        if !self.state()?.prepared.insert(active.attempt.clone()) {
            return Ok(unavailable());
        }
        let authority = host(json!({"action": "authority", "depth": active.depth}))?;
        let authority: Option<Authority> =
            crate::lossless_json::from_value(authority).map_err(|_| PublicError::internal())?;
        if let Some(authority) = &authority {
            authority.validate()?;
            if !binding.automatic_plans.is_empty() {
                let plans = binding
                    .automatic_plans
                    .iter()
                    .map(|p| p.bind(authority.vertex_project.as_ref()))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();
                if plans.is_empty() {
                    return Ok(unavailable());
                }
                let candidates: Vec<_> = plans
                    .iter()
                    .map(|p| {
                        json!({"prefix_sha256": p.prefix_sha256,
                    "prefix_bytes": p.conservative_input_bound})
                    })
                    .collect();
                let selected = host(
                    json!({"action": "select", "authority": authority, "candidates": candidates}),
                )?;
                if selected.is_null() {
                    return Ok(unavailable());
                }
                let digest = selected.as_str().ok_or_else(PublicError::internal)?;
                let mut matches = plans.into_iter().filter(|p| p.prefix_sha256 == digest);
                binding.plan = matches.next().ok_or_else(PublicError::internal)?;
                policy::require(matches.next().is_none())?;
            }
        }
        if current()?.as_ref() != Some(&active) {
            return Ok(unavailable());
        }
        let Some(authority) = authority else {
            return Ok(json!({"state": "disabled"}));
        };
        let Some(plan) = binding.plan.bind(authority.vertex_project.as_ref())? else {
            return Ok(unavailable());
        };
        binding.plan = plan;
        let offer = authority.offer(
            format!("cache-{}", uuid::Uuid::new_v4().simple()),
            request,
            &active.attempt,
            binding
                .plan
                .account(&binding.connection, &authority.credential_scope)?,
            &binding.plan.prefix_sha256,
            &binding.plan.resource_prefix,
            binding.plan.conservative_input_bound,
            (self.clock)()?.trunc(),
        )?;
        let Some(offer) = offer else {
            return Ok(unavailable());
        };
        let started = (self.clock)()?;
        policy::require(started >= offer.requested_at)?;
        if started + 5.0 >= offer.expires_at {
            return Ok(unavailable());
        }
        let claim: Claim =
            crate::lossless_json::from_value(host(json!({"action": "claim", "offer": offer}))?)
                .map_err(|_| PublicError::internal())?;
        let observed = (self.clock)()?;
        policy::require(observed >= started)?;
        offer.claim(&claim, observed, false)?;
        // A claim that expired during I/O never grants HTTP permission. It stays
        // owned by the host, including any uncertain renewal hold.
        if offer.claim(&claim, observed, true).is_err() {
            return Ok(unavailable());
        }
        let release = match &claim {
            Claim::Ready {
                renewal_operation_id,
                ..
            } => renewal_operation_id.as_deref(),
            _ => None,
        };
        let result = (|| {
            if current()?.as_ref() != Some(&active) {
                return Ok((unavailable(), false));
            }
            match &claim {
                Claim::Ready {
                    renewal_operation_id,
                    renew_until,
                    expires_at,
                    ..
                } => {
                    let horizon = renew_until.map(f64::trunc).filter(|v| *v > *expires_at);
                    let used = if let (Some(id), Some(until)) = (renewal_operation_id, horizon) {
                        self.state()?
                            .renewals
                            .insert(id.clone(), (until, *expires_at));
                        true
                    } else {
                        false
                    };
                    Ok((ready(&binding.plan, &claim)?, used))
                }
                Claim::Creator { .. } => {
                    let mut payload = binding.plan.create_payload.clone();
                    let object = payload.as_object_mut().ok_or_else(PublicError::internal)?;
                    object.remove("ttl");
                    object.insert(
                        "expireTime".into(),
                        json!(policy::iso_time(offer.expires_at)?),
                    );
                    let response = json!({"state": "create", "operation_id": offer.operation_id,
                        "resource_prefix": binding.plan.resource_prefix, "url": binding.plan.create_url,
                        "payload": payload, "expires_at": offer.expires_at,
                        "measure_tokens": !binding.automatic_plans.is_empty(), "minimum_tokens": authority.minimum_input_tokens});
                    self.state()?.operations.insert(
                        offer.operation_id.clone(),
                        Operation {
                            active,
                            offer,
                            binding,
                            observed: None,
                            recording: false,
                            recorded: false,
                        },
                    );
                    Ok((response, false))
                }
                Claim::Unavailable { .. } => Ok((unavailable(), false)),
            }
        })();
        if !result.as_ref().is_ok_and(|(_, used)| *used) {
            if let Some(id) = release {
                // Best effort only. A lost release keeps its conservative hold.
                let _ = host(
                    json!({"action": "renew", "renewal": {"operation_id": id, "outcome": "failed",
                    "observed_at": (self.clock)()?, "expire_time": null}}),
                );
            }
        }
        result.map(|(response, _)| response)
    }

    fn finish(
        &self,
        input: &Value,
        request: &str,
        host: &impl Fn(Value) -> Result<Value>,
        current: &impl Fn() -> Result<Option<Active>>,
    ) -> Result<Value> {
        let selected: Finish =
            crate::lossless_json::from_value(input.clone()).map_err(|_| PublicError::internal())?;
        selected.validate(request)?;
        let operation = self
            .state()?
            .operations
            .get(&selected.operation_id)
            .cloned()
            .ok_or_else(PublicError::internal)?;
        policy::require(
            operation.active.deployment == selected.deployment_id
                && current()?.as_ref() == Some(&operation.active),
        )?;
        let (result, record) = {
            let mut state = self.state()?;
            let operation = state
                .operations
                .get_mut(&selected.operation_id)
                .ok_or_else(PublicError::internal)?;
            // The first writer freezes receipt time under the same lock as intent.
            // A retry that read the operation before that writer must reuse it.
            let observed_at = match &operation.observed {
                Some(o) => o.observed_at,
                None => (self.clock)()?,
            };
            let result = selected.observation(operation, observed_at)?;
            policy::require(
                !operation.recording
                    && operation.observed.as_ref().is_none_or(|old| old == &result),
            )?;
            operation.observed = Some(result.clone());
            operation.recording = !operation.recorded;
            (result, !operation.recorded)
        };
        if record {
            let written = operation
                .offer
                .observation(&result)
                .and_then(|_| host(json!({"action": "record", "result": result})));
            let mut state = self.state()?;
            let operation = state
                .operations
                .get_mut(&selected.operation_id)
                .ok_or_else(PublicError::internal)?;
            operation.recording = false;
            operation.recorded = written.is_ok();
            written?;
        }
        if result.outcome != "ready" || current()?.as_ref() != Some(&operation.active) {
            return Ok(unavailable());
        }
        let claim = Claim::Ready {
            resource_name: result.resource_name.ok_or_else(PublicError::internal)?,
            expires_at: result.expire_time.ok_or_else(PublicError::internal)?,
            token_count: result.total_tokens.ok_or_else(PublicError::internal)?,
            renewal_operation_id: None,
            renew_until: None,
        };
        if operation
            .offer
            .claim(&claim, (self.clock)()?, true)
            .is_err()
        {
            return Ok(unavailable());
        }
        ready(&operation.binding.plan, &claim)
    }

    fn renew(
        &self,
        input: &Value,
        request: &str,
        host: &impl Fn(Value) -> Result<Value>,
    ) -> Result<Value> {
        let selected: Renewed =
            crate::lossless_json::from_value(input.clone()).map_err(|_| PublicError::internal())?;
        boundary(request, &selected.request_id, &selected.deployment_id)?;
        policy::scope(&selected.operation_id, 128)?;
        policy::require(["renewed", "failed", "unknown"].contains(&selected.outcome.as_str()))?;
        optional_text(selected.expire_time.as_deref(), 64)?;
        let (until, expiry) = self
            .state()?
            .renewals
            .remove(&selected.operation_id)
            .ok_or_else(PublicError::internal)?;
        let now = (self.clock)()?;
        let mut result = json!({"operation_id": selected.operation_id, "outcome": "unknown", "observed_at": now, "expire_time": null});
        if selected.outcome == "failed" && selected.expire_time.is_none() {
            result["outcome"] = json!("failed");
        }
        if selected.outcome == "renewed" {
            if let Ok(time) = policy::parse_time(selected.expire_time.as_deref()) {
                if time > now.max(expiry) && time <= until {
                    result["outcome"] = json!("renewed");
                    result["expire_time"] = json!(time);
                }
            }
        }
        host(json!({"action": "renew", "renewal": result}))?;
        Ok(json!({"state": "recorded"}))
    }
}

impl Finish {
    fn validate(&self, request: &str) -> Result<()> {
        boundary(request, &self.request_id, &self.deployment_id)?;
        policy::scope(&self.operation_id, 128)?;
        policy::require(["ready", "unknown", "not_created"].contains(&self.outcome.as_str()))?;
        optional_text(self.name.as_deref(), 1024)?;
        optional_text(self.expire_time.as_deref(), 64)?;
        optional_text(self.create_time.as_deref(), 64)?;
        if let Some(tokens) = self.total_tokens {
            policy::integer(tokens, 1)?;
        }
        policy::require(self.http_status.is_none_or(|s| (100..=599).contains(&s)))
    }
    fn observation(&self, operation: &Operation, observed_at: f64) -> Result<Observation> {
        let unknown = Observation {
            operation_id: self.operation_id.clone(),
            outcome: "unknown".into(),
            observed_at,
            resource_name: None,
            total_tokens: None,
            expire_time: None,
            http_status: self.http_status,
            create_time: None,
        };
        if self.outcome == "not_created" {
            policy::require(
                !operation.binding.automatic_plans.is_empty()
                    && self.name.is_none()
                    && self.http_status.is_none()
                    && self.total_tokens.is_none()
                    && self.expire_time.is_none()
                    && self.create_time.is_none(),
            )?;
            return Ok(Observation {
                outcome: "not_created".into(),
                ..unknown
            });
        }
        if self.outcome != "ready" {
            return Ok(unknown);
        }
        let Ok(expire_time) = policy::parse_time(self.expire_time.as_deref()) else {
            return Ok(unknown);
        };
        let mut ready = Observation {
            outcome: "ready".into(),
            resource_name: self.name.clone(),
            total_tokens: self.total_tokens,
            expire_time: Some(expire_time),
            ..unknown.clone()
        };
        if operation
            .binding
            .plan
            .apply(self.name.as_deref().unwrap_or(""))
            .is_err()
            || operation.offer.observation(&ready).is_err()
        {
            return Ok(unknown);
        }
        if let Ok(created) = policy::parse_time(self.create_time.as_deref()) {
            if created > 0.0 && created <= expire_time && created <= observed_at {
                ready.create_time = Some(created);
            }
        }
        Ok(ready)
    }
}

fn boundary(request: &str, selected: &str, deployment: &str) -> Result<()> {
    policy::require(
        selected == request
            && !selected.is_empty()
            && selected.chars().count() <= 128
            && !deployment.is_empty()
            && deployment.chars().count() <= 256,
    )
}
fn optional_text(text: Option<&str>, maximum: usize) -> Result<()> {
    policy::require(text.is_none_or(|s| s.chars().count() <= maximum))
}
fn unavailable() -> Value {
    json!({"state": "unavailable"})
}
fn ready(plan: &Plan, claim: &Claim) -> Result<Value> {
    let Claim::Ready {
        resource_name,
        expires_at,
        renewal_operation_id,
        renew_until,
        ..
    } = claim
    else {
        return Err(PublicError::internal());
    };
    let mut response = json!({"state": "ready", "resource_name": resource_name, "resource_prefix": plan.resource_prefix,
        "expires_at": expires_at, "payload": plan.apply(resource_name)?});
    if let (Some(operation), Some(until)) = (
        renewal_operation_id,
        renew_until.map(f64::trunc).filter(|v| *v > *expires_at),
    ) {
        response["renew"] = json!({"operation_id": operation, "expires_at": until, "expire_time": policy::iso_time(until)?});
    }
    Ok(response)
}
fn boundary_error() -> PublicError {
    crate::errors::Failure::new(
        crate::errors::FailureClass::Internal,
        "explicit cache accounting is unavailable; contact the operator",
    )
    .public_error()
}

#[cfg(test)]
#[path = "cache_tests.rs"]
mod tests;
