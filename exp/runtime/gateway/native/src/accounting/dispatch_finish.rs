//! Durable terminal delivery, health observations and native fleet lease release.

use std::time::SystemTime;

use serde_json::{json, Value};

use crate::errors::{Failure, FailureClass, PublicError};

use super::dispatch::{Attempt, Dispatch, State};
use super::health::HealthKey;
use super::loads::Shed;
use super::{observations, terminal};

type FleetTicket = (Result<Option<String>, Shed>, Option<String>);

impl Dispatch {
    pub(super) fn fleet(
        &self,
        depth: usize,
        ticket: Result<Option<String>, Shed>,
        input_tokens: u64,
    ) -> Result<FleetTicket, PublicError> {
        let c = &self.config;
        let lane = &c.lanes[depth];
        let (Some(policy), Ok(Some(id))) = (&lane.policy.fleet, &ticket) else {
            return Ok((ticket, None));
        };
        let response = self.host(
            "fleet_acquire",
            &json!({"request_id": c.request_id,
            "ticket": id, "deployment_id": lane.key.1, "connection": lane.key.2,
            "policy": policy, "reserved_tokens": input_tokens.saturating_add(lane.output_tokens)}),
        );
        let response = match response {
            Ok(response) => response,
            Err(error) => {
                let mut p = self
                    .shared
                    .policies
                    .lock()
                    .map_err(|_| PublicError::internal())?;
                p.loads.retract(id);
                p.health.release(&lane.key);
                return Err(error);
            }
        };
        if let Some(reason) = response["shed"].as_str() {
            let mut p = self
                .shared
                .policies
                .lock()
                .map_err(|_| PublicError::internal())?;
            p.loads.retract(id);
            let refusing = policy["saturation"] == "refuse";
            return Ok((
                Err(Shed {
                    reason: if reason == "rate" && !refusing {
                        "rate_limit"
                    } else {
                        "queue_bound"
                    },
                    fleet: true,
                    fleet_refuse: refusing,
                    ..Shed::hard()
                }),
                None,
            ));
        }
        let Some(lease) = response["lease"].as_str().filter(|s| !s.is_empty()) else {
            let mut p = self
                .shared
                .policies
                .lock()
                .map_err(|_| PublicError::internal())?;
            p.loads.retract(id);
            p.health.release(&lane.key);
            return Err(PublicError::internal());
        };
        let lease = lease.to_string();
        self.shared
            .policies
            .lock()
            .map_err(|_| PublicError::internal())?
            .loads
            .restamp(id, self.shared.now());
        Ok((ticket, Some(lease)))
    }

    pub(super) fn release_unbound(
        &self,
        key: &HealthKey,
        ticket: Option<&str>,
        fleet: Option<&str>,
    ) -> Result<(), PublicError> {
        {
            let mut p = self
                .shared
                .policies
                .lock()
                .map_err(|_| PublicError::internal())?;
            p.health.release(key);
            if let Some(ticket) = ticket {
                p.loads.release(ticket);
            }
        }
        if let Some(lease) = fleet {
            // Best-effort, like every lease release: the host's expiry reclaims a failed one.
            let _ = self.host("fleet_release", &json!({"request_id": self.config.request_id, "lease": lease,
                "throttled": false, "used_tokens": null, "output_tokens": null, "first_token_at": null}));
        }
        Ok(())
    }

    pub(super) fn settle(
        &self,
        method: &str,
        data: &Value,
        state: &mut State,
    ) -> Result<Value, PublicError> {
        let id = data["attempt_id"]
            .as_str()
            .ok_or_else(PublicError::internal)?;
        let attempt = state
            .attempts
            .get_mut(id)
            .ok_or_else(PublicError::internal)?;
        // A meter observed during a cancellation write remains recoverable;
        // the cancellation's cached result cannot certify that meter as paid.
        if attempt.cancelled {
            if let Some(result) = &attempt.result {
                return Ok(result.clone());
            }
            if attempt.late_terminal.is_none() {
                attempt.late_terminal = Some(terminal::normalize(
                    data,
                    &attempt.terms,
                    0.0,
                    SystemTime::now(),
                )?);
            }
            let result = self.host(
                "record_late_meter",
                &json!({"request_id": self.config.request_id,
                "attempt_id": id, "terminal": attempt.late_terminal}),
            )?;
            if method == "settle" {
                attempt.result = Some(result.clone());
            }
            return Ok(result);
        }
        if let Some(result) = &attempt.result {
            return Ok(result.clone());
        }
        if attempt.terminal.is_none() {
            let cached = self
                .shared
                .policies
                .lock()
                .map_err(|_| PublicError::internal())?
                .loads
                .cached_fraction(
                    &self.config.lanes[attempt.depth].load_key(),
                    &self.config.organization_id,
                    self.shared.now(),
                );
            let mut terminal =
                terminal::normalize(data, &attempt.terms, cached, SystemTime::now())?;
            self.resources.terminal(&mut terminal);
            attempt.recovery_observed = Some(super::session_observations::now());
            attempt.recovery_failure = if terminal["failure"].is_object() {
                Some(
                    crate::lossless_json::from_value(terminal["failure"].clone())
                        .map_err(|_| PublicError::internal())?,
                )
            } else {
                None
            };
            if let Some(failure) = &mut attempt.recovery_failure {
                if failure.customer_owned
                    && matches!(
                        data["failure"]["failure_class"].as_str(),
                        Some("provider_authentication" | "provider_quota")
                    )
                {
                    failure.failure_class =
                        crate::lossless_json::from_value(data["failure"]["failure_class"].clone())
                            .map_err(|_| PublicError::internal())?;
                }
            }
            attempt.terminal = Some(terminal);
        }
        let terminal = attempt.terminal.as_ref().unwrap();
        let result = self.host(
            method,
            &json!({"request_id": self.config.request_id, "attempt_id": id, "terminal": terminal}),
        )?;
        if method == "price" {
            return Ok(result);
        }
        self.resources.settled(&self.config.request_id, &result);
        if !attempt.recovery_recorded {
            if let (Some(recovery), Some(observed), Some(key)) = (
                &self.config.recovery,
                attempt.recovery_observed,
                &attempt.recovery_key,
            ) {
                recovery.settled(
                    key,
                    &self.shared,
                    attempt.depth,
                    &self.config.lanes[attempt.depth].key.1,
                    terminal,
                    attempt.recovery_failure.as_ref(),
                    observed,
                );
            }
            attempt.recovery_recorded = true;
        }
        self.released(id, attempt, data)?;
        attempt.result = Some(result.clone());
        if data["finalize"] == true {
            self.resources.close(&self.config.request_id);
        }
        if data["finalize"] == true
            && attempt
                .terminal
                .as_ref()
                .is_some_and(|t| t["input_guardrail_denied"] != true)
            && matches!(data["outcome"].as_str(), Some("completed" | "incomplete"))
        {
            state.final_depth = Some(attempt.depth);
        }
        if state.active.as_deref() == Some(id) {
            state.active = None;
        }
        Ok(result)
    }

    fn released(&self, id: &str, attempt: &mut Attempt, data: &Value) -> Result<(), PublicError> {
        let lane = &self.config.lanes[attempt.depth];
        let terminal = attempt
            .terminal
            .as_ref()
            .ok_or_else(PublicError::internal)?;
        let mut failure: Option<Failure> = if data["failure"].is_object() {
            Some(
                crate::lossless_json::from_value(data["failure"].clone())
                    .map_err(|_| PublicError::internal())?,
            )
        } else {
            None
        };
        let observed = observations::parse(&data["rate_limit_headers"], SystemTime::now());
        if let Some(failure) = &mut failure {
            if failure.customer_owned
                && matches!(
                    failure.failure_class,
                    FailureClass::ProviderAuthentication | FailureClass::ProviderQuota
                )
            {
                failure.failure_class = FailureClass::InvalidRequest;
            }
            if failure.failure_class == FailureClass::Throttled
                && failure.retry_after_seconds.is_none()
            {
                failure.retry_after_seconds = observed
                    .retry_after_seconds
                    .or_else(|| observed.exhausted_wait())
                    .map(|n| n as u32);
            }
        }
        let throttled = failure
            .as_ref()
            .is_some_and(|f| f.failure_class == FailureClass::Throttled);
        // Each lease is released once and best-effort: a failed release is not
        // retried, and the host's own lease expiry reclaims it. A settlement
        // replayed by recovery therefore never releases the same lease twice.
        if let Some(lease) = attempt.fleet.take() {
            let usage = &data["usage"];
            let total = usage["input_tokens"]
                .as_u64()
                .zip(usage["output_tokens"].as_u64())
                .and_then(|(i, o)| i.checked_add(o))
                .filter(|n| *n > 0);
            let _ = self.host("fleet_release", &json!({"request_id": self.config.request_id, "lease": lease,
                "throttled": throttled, "used_tokens": total,
                "output_tokens": total.and_then(|_| usage["output_tokens"].as_u64()), "first_token_at": terminal["first_token_at"]}));
        }
        let mut p = self
            .shared
            .policies
            .lock()
            .map_err(|_| PublicError::internal())?;
        p.loads.release_attempt(id);
        if data["graceful_transport_retirement"] == true {
            p.health.release(&lane.key);
            return Ok(());
        }
        if data["opened"] == true {
            p.health.succeeded(&lane.key);
        }
        let now = self.shared.now();
        if let Some(failure) = &failure {
            if throttled
                && (lane.policy.concurrency_bound.is_some()
                    || lane.policy.requests_per_minute.is_some()
                    || lane.policy.tokens_per_minute.is_some())
            {
                p.loads.record_throttle(&lane.load_key(), now);
            }
            p.health.failed(&lane.key, failure, now);
        } else {
            p.health.succeeded(&lane.key);
        }
        if let Some(wait) = observed.exhausted_wait() {
            p.health.exhausted(&lane.key, wait as f64, now);
        }
        if terminal["usage_estimated"] != true && !attempt.funded {
            if let Some(input) = terminal["usage"]["input_tokens"].as_u64() {
                p.loads.record_settle(
                    &lane.load_key(),
                    &self.config.organization_id,
                    terminal["usage"]["cached_input_tokens"]
                        .as_u64()
                        .unwrap_or(0),
                    input,
                    now,
                );
            }
        }
        Ok(())
    }

    pub(super) fn finish(
        &self,
        data: &Value,
        state: &mut State,
        certify: bool,
    ) -> Result<Value, PublicError> {
        let c = &self.config;
        let failure = if data["failure"].is_object() {
            data["failure"].clone()
        } else {
            json!({"failure_class": "cancelled", "safe_message": "gateway request was cancelled"})
        };
        let evidence = json!({"outcome": "failed", "failure": failure, "finalize": true});
        let terms = terminal::Terms {
            surface: c.surface.clone(),
            counted_input_tokens: state.counted_input_tokens,
            output_ceiling: c.output_ceiling,
            output_budget: None,
            service_tier: Value::Null,
            self_served: false,
            paid_prework: c.paid_prework,
            web_search_requests: c.web_search_requests,
        };
        let mut terminal = terminal::normalize(&evidence, &terms, 0.0, SystemTime::now())?;
        let searches = data["web_search_requests"]
            .as_u64()
            .unwrap_or(0)
            .max(c.web_search_requests);
        // Active reservations settle from the terminal; unreserved requests
        // use the request-level count. Both retain the same paid prework.
        terminal["observations"]["web_search_requests"] = json!(searches);
        self.resources.terminal(&mut terminal);
        self.resources.close(&c.request_id);
        let result = self.host(
            "abandon",
            &json!({"request_id": c.request_id, "terminal": terminal,
            "refusal_sqlstate": null, "certify_no_effects": certify,
            "web_search_requests": searches}),
        )?;
        if result["terminalized"] == true {
            self.resources.settled(&c.request_id, &result);
            if let Some(id) = state.active.clone() {
                let attempt = state
                    .attempts
                    .get_mut(&id)
                    .ok_or_else(PublicError::internal)?;
                if attempt.terminal.is_some() && !attempt.cancelled {
                    // The owner must deliver observed evidence before cancellation.
                    return Err(PublicError::internal());
                }
                attempt.terminal.get_or_insert(terminal);
                attempt.cancelled = true;
                self.released(&id, attempt, &evidence)?;
                state.active = None;
            }
        }
        Ok(result)
    }
}
