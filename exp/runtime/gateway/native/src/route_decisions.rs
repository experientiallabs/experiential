//! Native `POST /v1/systemone`: buffered TypeSafe decisions, not chat.
//!
//! Admission builds the exact provider payload and owns authorization and
//! reservations. Each physical dispatch uses the shared attempt lifecycle;
//! only a fully validated, billable answer may settle as completed or reach
//! the caller. This surface has no streaming or replay protocol.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::admission::{acquire_permit, new_guard, wire_drift_response};
use crate::dialects::{Dialect, MAXIMUM_RETAINED_OUTPUT_BYTES, OUTPUT_OVERFLOW_MESSAGE};
use crate::encode::{compact_json, stable_public_id};
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::Usage;
use crate::metrics::{classify_escalation, METRICS};
use crate::rate_limit_headers::harvest_rate_limit_headers;
use crate::relay::{collection_public_error, remaining};
use crate::respond::{
    bearer_key, client_ip, error_response, escalation_error, json_response, latin1_header,
    read_body,
};
use crate::server::AppState;
use crate::settlement::AttemptGuard;
use crate::upstream::open_stream;
use crate::waterfall::{successor_possible, DeploymentWire, RoutePolicy, StartResponse};

const PROBABILITY_TOLERANCE: f64 = 1e-6;

#[derive(Debug, Clone, Deserialize)]
struct DecisionsAdmission {
    request_id: String,
    alias: String,
    alias_revision_id: String,
    exact_model_id: String,
    route_reason: String,
    route: Vec<DeploymentWire>,
    questions: Map<String, Value>,
    maximum_total_attempts: u32,
    maximum_same_deployment_attempts: u32,
}

impl DecisionsAdmission {
    fn policy(&self) -> RoutePolicy {
        RoutePolicy {
            maximum_total_attempts: self.maximum_total_attempts.max(1),
            maximum_same_deployment_attempts: self.maximum_same_deployment_attempts.max(1),
            refusal_failover: false,
            throttle_redial: None,
            physical_route_cap: None,
            backoff: None,
        }
    }
}

struct Served {
    depth: usize,
    body: Value,
    usage: Usage,
}

pub(crate) async fn decisions(
    State(state): State<AppState>,
    request: axum::extract::Request,
) -> Response {
    state.handled_requests.fetch_add(1, Ordering::Relaxed);
    let started = Instant::now();
    let deadline = started + state.request_timeout;
    let (parts, raw_body) = request.into_parts();
    let headers = parts.headers;
    let body = match tokio::time::timeout_at(deadline.into(), read_body(raw_body)).await {
        Ok(Ok(body)) => body,
        Ok(Err(error)) => return error_response(&error),
        Err(_) => return error_response(&timeout_failure().public_error()),
    };
    let raw_key = match bearer_key(&headers) {
        Ok(key) => key,
        Err(error) => return error_response(&error),
    };
    let authenticate = compact_json(&json!({"raw_key": raw_key}));
    if let Err(error) = state.bridge.call("authenticate", authenticate).await {
        return error_response(&error);
    }
    let body_text = match String::from_utf8(body.to_vec()) {
        Ok(text) => text,
        Err(_) => return error_response(&PublicError::invalid_json()),
    };
    let client_request_id = latin1_header(&headers, "x-client-request-id");
    let argument = admission_argument(&raw_key, &body_text, &headers);
    let admission_text = match state.bridge.call("admit_decisions", argument).await {
        Ok(text) => text,
        Err(error) => return error_response(&error),
    };
    let admission_value: Value = match serde_json::from_str(&admission_text) {
        Ok(value) => value,
        Err(_) => return error_response(&PublicError::internal()),
    };
    if let Some(reason) = admission_value.get("escalate") {
        METRICS.record_escalation(classify_escalation(reason.as_str().unwrap_or_default()));
        return error_response(&escalation_error());
    }
    let admission: DecisionsAdmission = match serde_json::from_value(admission_value.clone()) {
        Ok(admission) => admission,
        Err(_) => return wire_drift_response(&state, &admission_value, started).await,
    };
    let mut guard = new_guard(&state, admission.request_id.clone(), started);
    let _permit = match acquire_permit(&state, &mut guard, deadline).await {
        Ok(permit) => permit,
        Err(response) => return *response,
    };
    match run_ladder(&state, &admission, &raw_key, &mut guard, deadline).await {
        Err(error) => error_response(&error),
        Ok(served) => {
            if !guard
                .settle("completed", Some(&served.usage), &[], None, true)
                .await
            {
                return error_response(&PublicError::internal());
            }
            json_response(
                StatusCode::OK,
                &served.body,
                &served_headers(&admission, served.depth, client_request_id),
            )
        }
    }
}

/// Forward authority, attribution, and trusted proxy IP without caller idempotency.
fn admission_argument(raw_key: &str, body: &str, headers: &HeaderMap) -> String {
    compact_json(&json!({
        "raw_key": raw_key,
        "body": body,
        "client_ip": client_ip(headers),
        "app_referer": latin1_header(headers, "http-referer"),
        "app_title": latin1_header(headers, "x-title"),
    }))
}

/// Reserve every physical dispatch through the same authority as other routes.
async fn run_ladder(
    state: &AppState,
    admission: &DecisionsAdmission,
    raw_key: &str,
    guard: &mut AttemptGuard,
    deadline: Instant,
) -> Result<Served, PublicError> {
    let policy = admission.policy();
    let mut total_attempts = 0;
    let mut counts = vec![0; admission.route.len()];
    let mut current_depth = None;
    let mut last_failure: Option<Failure> = None;
    loop {
        let argument = compact_json(&json!({
            "request_id": admission.request_id,
            "raw_key": raw_key,
            "attempt_ordinal": total_attempts,
            "current_depth": current_depth,
            "failure": last_failure.as_ref().map(|failure| json!({
                "failure_class": failure.failure_class.as_str(),
                "safe_message": failure.safe_message,
                "retryable_same_deployment": failure.retryable_same_deployment,
                "failover_eligible": failure.failover_eligible,
                "rejected_parameter": failure.rejected_parameter,
                "provider_detail": failure.provider_detail,
                "customer_owned": failure.customer_owned,
                "retry_after_seconds": failure.retry_after_seconds,
            })),
        }));
        let started_text = match state.bridge.call("start_attempt", argument).await {
            Ok(text) => text,
            Err(error) => {
                guard.disarm_finalized("failed");
                return Err(error);
            }
        };
        let started: StartResponse = match serde_json::from_str(&started_text) {
            Ok(started) => started,
            Err(_) => {
                guard.abandon(&wire_failure()).await;
                return Err(PublicError::internal());
            }
        };
        if started.exhausted {
            guard.disarm_finalized("failed");
            let failure = started.failure.or(last_failure).unwrap_or_else(|| {
                Failure::new(
                    FailureClass::ProviderInternal,
                    "all exact-model deployments are unavailable",
                )
            });
            return Err(collection_public_error(&failure.boundary()));
        }
        let (Some(attempt_id), Some(depth)) = (started.attempt_id, started.route_depth) else {
            guard.abandon(&wire_failure()).await;
            return Err(PublicError::internal());
        };
        guard.rebind(attempt_id);
        let Some(wire) = admission.route.get(depth) else {
            guard
                .settle("failed", None, &[], Some(&wire_failure()), true)
                .await;
            return Err(PublicError::internal());
        };
        // Re-check caps locally before dispatch, even if the bridge returns an
        // inconsistent reservation. Unknown provider idempotency is no license
        // to repeat an already dispatched decision.
        if total_attempts >= policy.maximum_total_attempts
            || counts[depth] >= policy.maximum_same_deployment_attempts
        {
            guard
                .settle("failed", None, &[], Some(&wire_failure()), true)
                .await;
            return Err(PublicError::internal());
        }
        if current_depth == Some(depth) {
            METRICS.record_open_retry();
        }
        total_attempts += 1;
        counts[depth] += 1;
        match dispatch(&state.http, wire, deadline, admission, guard).await {
            Ok((body, usage)) => return Ok(Served { depth, body, usage }),
            Err(failure) => {
                let failure = route_failure(failure, wire, policy, counts[depth]);
                let boundary = failure.clone().boundary();
                let possible = successor_possible(
                    policy,
                    &admission.route,
                    deadline,
                    total_attempts,
                    counts[depth],
                    depth,
                    &failure,
                    false,
                );
                if !guard
                    .settle("failed", None, &[], Some(&boundary), !possible)
                    .await
                {
                    return Err(PublicError::internal());
                }
                if !possible {
                    return Err(collection_public_error(&boundary));
                }
                current_depth = Some(depth);
                last_failure = Some(failure);
            }
        }
    }
}

/// Preserve credential ownership and send only allowed retry choices to authority.
fn route_failure(
    mut failure: Failure,
    wire: &DeploymentWire,
    policy: RoutePolicy,
    same_deployment_attempts: u32,
) -> Failure {
    if wire.billing_customer_managed {
        failure = crate::stream_errors::customer_credential_failure(failure, &wire.provider);
    }
    if !failure.decision_provider_rejected {
        // The provider may have executed the decision. With no idempotency
        // contract, an uncertain outcome cannot authorize another dispatch.
        failure.retryable_same_deployment = false;
        failure.failover_eligible = false;
    } else if same_deployment_attempts >= policy.maximum_same_deployment_attempts {
        failure.retryable_same_deployment = false;
    }
    failure
}

fn wire_failure() -> Failure {
    Failure::new(
        FailureClass::Internal,
        "gateway decision wire contract failed",
    )
}

fn timeout_failure() -> Failure {
    Failure::new(
        FailureClass::Timeout,
        "provider did not finish the decision response in time",
    )
    .with_retry(false, false)
}

async fn dispatch(
    http: &crate::upstream::UpstreamClient,
    wire: &DeploymentWire,
    deadline: Instant,
    admission: &DecisionsAdmission,
    guard: &mut AttemptGuard,
) -> Result<(Value, Usage), Failure> {
    if wire.dialect != "typesafe_systemone" || !wire.timeout_seconds.is_finite() {
        return Err(wire_failure());
    }
    if remaining(deadline).is_zero() {
        return Err(timeout_failure());
    }
    let phase_timeout =
        Duration::try_from_secs_f64(wire.timeout_seconds.max(0.001)).map_err(|_| wire_failure())?;
    let response = open_stream(
        http,
        &wire.url,
        &wire.headers,
        &wire.idempotency_key,
        &wire.upstream_payload,
        None,
        remaining(deadline).min(phase_timeout),
        Dialect::TypesafeSystemone,
    )
    .await?;
    collect_response(response, deadline, phase_timeout, admission, guard).await
}

/// Buffer one opened response while its cancellation guard remains live.
async fn collect_response(
    response: reqwest::Response,
    deadline: Instant,
    phase_timeout: Duration,
    admission: &DecisionsAdmission,
    guard: &mut AttemptGuard,
) -> Result<(Value, Usage), Failure> {
    // Mark the dispatch before buffering so cancellation records an opened
    // attempt, not a failed open, and the successful answer has the same fact.
    guard.mark_opened();
    guard.record_rate_limit_headers(harvest_rate_limit_headers(response.headers()));
    if response
        .content_length()
        .is_some_and(|length| length > MAXIMUM_RETAINED_OUTPUT_BYTES as u64)
    {
        return Err(malformed(OUTPUT_OVERFLOW_MESSAGE));
    }
    let bytes = read_bounded_body(response, deadline, phase_timeout).await?;
    let payload = strict_json(&bytes)?;
    // Billing evidence is independent of decision validity. Keep it on the
    // attempt guard before any error or cancellation can discard the answer.
    if let Ok(usage) = decision_usage(&payload) {
        guard.record_decision_usage(usage);
    }
    let served = public_decisions(payload, admission)?;
    if remaining(deadline).is_zero() {
        return Err(timeout_failure());
    }
    Ok(served)
}

/// Cap retained bytes even without Content-Length, under per-read and request bounds.
async fn read_bounded_body(
    mut response: reqwest::Response,
    deadline: Instant,
    phase_timeout: Duration,
) -> Result<Vec<u8>, Failure> {
    let mut body = Vec::new();
    loop {
        let bound = remaining(deadline).min(phase_timeout);
        if bound.is_zero() {
            return Err(timeout_failure());
        }
        let chunk = match tokio::time::timeout(bound, response.chunk()).await {
            Ok(Ok(Some(chunk))) => chunk,
            Ok(Ok(None)) => return Ok(body),
            Ok(Err(_)) => {
                return Err(Failure::new(
                    FailureClass::Transport,
                    "provider connection failed while sending the decision response",
                )
                .with_retry(false, false))
            }
            Err(_) => return Err(timeout_failure()),
        };
        if chunk.len() > MAXIMUM_RETAINED_OUTPUT_BYTES.saturating_sub(body.len()) {
            return Err(malformed(OUTPUT_OVERFLOW_MESSAGE));
        }
        body.extend_from_slice(&chunk);
    }
}

fn malformed(reason: &str) -> Failure {
    Failure::new(FailureClass::MalformedResponse, reason).with_retry(false, false)
}

/// Strict JSON parsing rejects duplicate keys at every depth, including answers.
fn strict_json(bytes: &[u8]) -> Result<Value, Failure> {
    struct Unique(Value);
    impl<'de> Deserialize<'de> for Unique {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct Visitor;
            impl<'de> serde::de::Visitor<'de> for Visitor {
                type Value = Unique;
                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    f.write_str("JSON with unique object keys")
                }
                fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<Unique, E> {
                    Ok(Unique(Value::from(value)))
                }
                fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Unique, E> {
                    Ok(Unique(Value::from(value)))
                }
                fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Unique, E> {
                    Ok(Unique(Value::from(value)))
                }
                fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Unique, E> {
                    serde_json::Number::from_f64(value)
                        .map(|number| Unique(Value::Number(number)))
                        .ok_or_else(|| E::custom("non-finite JSON number"))
                }
                fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Unique, E> {
                    Ok(Unique(Value::from(value)))
                }
                fn visit_unit<E: serde::de::Error>(self) -> Result<Unique, E> {
                    Ok(Unique(Value::Null))
                }
                fn visit_seq<A: serde::de::SeqAccess<'de>>(
                    self,
                    mut seq: A,
                ) -> Result<Unique, A::Error> {
                    let mut values = Vec::new();
                    while let Some(Unique(value)) = seq.next_element()? {
                        values.push(value);
                    }
                    Ok(Unique(Value::Array(values)))
                }
                fn visit_map<A: serde::de::MapAccess<'de>>(
                    self,
                    mut map: A,
                ) -> Result<Unique, A::Error> {
                    let mut values = Map::new();
                    while let Some((key, Unique(value))) = map.next_entry::<String, Unique>()? {
                        if values.insert(key, value).is_some() {
                            return Err(serde::de::Error::custom("duplicate JSON key"));
                        }
                    }
                    Ok(Unique(Value::Object(values)))
                }
            }
            deserializer.deserialize_any(Visitor)
        }
    }
    serde_json::from_slice::<Unique>(bytes)
        .map(|value| value.0)
        .map_err(|_| malformed("decision response is not strict JSON"))
}

/// Validate against the admitted question definitions, never provider declarations.
fn public_decisions(
    payload: Value,
    admission: &DecisionsAdmission,
) -> Result<(Value, Usage), Failure> {
    let answers = payload
        .get("answers")
        .and_then(Value::as_object)
        .ok_or_else(|| malformed("decision response omitted its answers object"))?;
    if answers.len() != admission.questions.len()
        || !answers
            .keys()
            .all(|key| admission.questions.contains_key(key))
    {
        return Err(malformed(
            "decision response question IDs do not match the request",
        ));
    }
    let mut public_answers = Map::new();
    for (id, question) in &admission.questions {
        let answer = &answers[id];
        let kind = question
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(wire_failure)?;
        if answer.get("type").and_then(Value::as_str) != Some(kind) {
            return Err(malformed(
                "decision answer type does not match the question",
            ));
        }
        let public = match kind {
            "noul" => {
                probability(&answer["noul"])?;
                json!({"type": kind, "noul": answer["noul"]})
            }
            "choice" => {
                let criteria = question
                    .get("criteria")
                    .and_then(Value::as_object)
                    .ok_or_else(wire_failure)?;
                distribution(
                    &answer["probabilities"],
                    criteria.keys().map(String::as_str),
                )?;
                let choice = answer
                    .get("choice")
                    .and_then(Value::as_str)
                    .filter(|choice| criteria.contains_key(*choice))
                    .ok_or_else(|| malformed("decision choice is not a requested category"))?;
                probability(&answer["confidence"])?;
                let selected = probability(&answer["probabilities"][choice])?;
                for key in criteria.keys() {
                    if probability(&answer["probabilities"][key])? > selected {
                        return Err(malformed(
                            "decision choice is not a highest-probability category",
                        ));
                    }
                }
                json!({"type": kind, "choice": choice, "confidence": answer["confidence"],
                    "probabilities": answer["probabilities"]})
            }
            "score" => {
                let criteria = question
                    .get("criteria")
                    .and_then(Value::as_array)
                    .ok_or_else(wire_failure)?;
                let legend: Map<String, Value> = criteria
                    .iter()
                    .enumerate()
                    .map(|(index, value)| (index.to_string(), value.clone()))
                    .collect();
                if answer.get("legend").and_then(Value::as_object) != Some(&legend) {
                    return Err(malformed(
                        "decision score legend does not match the requested criteria",
                    ));
                }
                distribution(&answer["probabilities"], legend.keys().map(String::as_str))?;
                probability(&answer["confidence"])?;
                let score = answer
                    .get("score")
                    .and_then(Value::as_f64)
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| malformed("decision score is not finite numeric data"))?;
                let expected = (0..criteria.len()).try_fold(0.0, |sum, index| {
                    probability(&answer["probabilities"][index.to_string()])
                        .map(|value| sum + index as f64 * value)
                })?;
                if score < 0.0
                    || score > criteria.len().saturating_sub(1) as f64
                    || (score - expected).abs() > PROBABILITY_TOLERANCE
                {
                    return Err(malformed(
                        "decision score does not match its probability distribution",
                    ));
                }
                json!({"type": kind, "score": answer["score"], "confidence": answer["confidence"],
                    "legend": answer["legend"], "probabilities": answer["probabilities"]})
            }
            _ => return Err(wire_failure()),
        };
        public_answers.insert(id.clone(), public);
    }
    let usage = decision_usage(&payload)?;
    Ok((
        json!({"id": stable_public_id("decision", &admission.request_id),
        "model": admission.alias, "answers": public_answers,
        "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}}),
        usage,
    ))
}

/// Validate the provider's billing evidence without trusting the decision answers.
fn decision_usage(payload: &Value) -> Result<Usage, Failure> {
    let input_tokens = token_count(&payload["usage"]["input_tokens"])?;
    let output_tokens = token_count(&payload["usage"]["output_tokens"])?;
    if input_tokens == 0 && output_tokens == 0 {
        return Err(malformed(
            "decision response reported no billable token usage",
        ));
    }
    Ok(Usage {
        input_tokens: Some(input_tokens),
        output_tokens: Some(output_tokens),
        cached_input_tokens: None,
        cache_creation_input_tokens: None,
        cache_creation_1h_input_tokens: None,
        reasoning_tokens: None,
    })
}

fn probability(value: &Value) -> Result<f64, Failure> {
    value
        .as_f64()
        .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
        .ok_or_else(|| {
            malformed("decision probability or confidence must be finite and between zero and one")
        })
}

fn distribution<'a>(value: &Value, keys: impl Iterator<Item = &'a str>) -> Result<(), Failure> {
    let probabilities = value
        .as_object()
        .ok_or_else(|| malformed("decision omitted its probability object"))?;
    let mut count = 0;
    let mut sum = 0.0;
    for key in keys {
        count += 1;
        sum += probability(probabilities.get(key).unwrap_or(&Value::Null))?;
    }
    if probabilities.len() != count || (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(malformed(
            "decision probability keys or total do not match the request",
        ));
    }
    Ok(())
}

fn token_count(value: &Value) -> Result<u64, Failure> {
    value.as_u64().filter(|value| *value <= i64::MAX as u64)
        .ok_or_else(|| malformed("decision usage requires nonnegative integer input_tokens and output_tokens within i64"))
}

fn served_headers(
    admission: &DecisionsAdmission,
    depth: usize,
    client_request_id: Option<String>,
) -> Vec<(String, String)> {
    let wire = &admission.route[depth];
    let mut headers = vec![
        ("x-request-id".to_string(), admission.request_id.clone()),
        ("x-gateway-alias".to_string(), admission.alias.clone()),
        (
            "x-gateway-alias-revision".to_string(),
            admission.alias_revision_id.clone(),
        ),
        (
            "x-gateway-canonical-model".to_string(),
            admission.exact_model_id.clone(),
        ),
        ("x-gateway-provider".to_string(), wire.provider.clone()),
        (
            "x-gateway-deployment".to_string(),
            wire.deployment_id.clone(),
        ),
        ("x-gateway-route-depth".to_string(), depth.to_string()),
        (
            "x-gateway-route-reason".to_string(),
            admission.route_reason.clone(),
        ),
    ];
    if let Some(value) = client_request_id {
        headers.push(("x-client-request-id".to_string(), value));
    }
    headers
}

#[cfg(test)]
mod tests;
