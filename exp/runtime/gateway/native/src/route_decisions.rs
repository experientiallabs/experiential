//! Native decisions: `POST /v1/systemone` (TypeSafe SystemOne wire) and
//! `POST /v1/decisions` (OpenAI Decisions API wire), buffered, not chat.
//!
//! Both wires share admission authority, the attempt ladder, settlement and
//! billing evidence; they differ only in the admission bridge method, the
//! upstream dialect, and how a provider answer is validated against the
//! admitted questions.
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
use serde::{Deserialize, Serialize};
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
    read_body, with_app_identity,
};
use crate::server::AppState;
use crate::settlement::AttemptGuard;
use crate::upstream::open_decision_stream;
use crate::waterfall::{successor_possible, DeploymentWire, RoutePolicy, StartResponse};
use body_budget::reserve_body;
use openai::{openai_decision_usage, public_openai_decisions};
use residual::residual_envelope;

const PROBABILITY_TOLERANCE: f64 = 1e-6;

#[derive(Debug, Clone, Default, Deserialize)]
struct DecisionsAdmission {
    request_id: String,
    alias: String,
    alias_revision_id: String,
    exact_model_id: String,
    route_reason: String,
    route: Vec<DeploymentWire>,
    /// SystemOne question definitions keyed by question ID.
    #[serde(default)]
    questions: Map<String, Value>,
    /// Which decisions wire this admission dispatches; required, so a stale
    /// or malformed producer fails closed as wire drift.
    wire: DecisionWire,
    /// OpenAI question definitions, in request order (OpenAI wire only).
    #[serde(default)]
    openai_questions: Vec<Value>,
    maximum_total_attempts: u32,
    maximum_same_deployment_attempts: u32,
}

/// The upstream decisions wire an admission dispatches.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum DecisionWire {
    /// TypeSafe SystemOne (`/v1/systemone`).
    #[default]
    Systemone,
    /// OpenAI Decisions API (`/v1/decisions`).
    Openai,
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

/// `POST /v1/systemone`: the TypeSafe SystemOne decisions wire.
pub(crate) async fn decisions(
    State(state): State<AppState>,
    request: axum::extract::Request,
) -> Response {
    serve(state, request, "admit_decisions", DecisionWire::Systemone).await
}

/// `POST /v1/decisions`: the OpenAI Decisions API wire.
pub(crate) async fn openai_decisions(
    State(state): State<AppState>,
    request: axum::extract::Request,
) -> Response {
    serve(
        state,
        request,
        "admit_openai_decisions",
        DecisionWire::Openai,
    )
    .await
}

async fn serve(
    state: AppState,
    request: axum::extract::Request,
    admission_method: &'static str,
    expected_wire: DecisionWire,
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
    // Held for the whole request: the body and its admitted payloads stay
    // resident until the provider answer is relayed.
    let _body_budget = match reserve_body(body.len(), deadline).await {
        Ok(permit) => permit,
        Err(error) => return error_response(&error),
    };
    let raw_key = match bearer_key(&headers) {
        Ok(key) => key,
        Err(error) => return error_response(&error),
    };
    let authenticate = compact_json(&json!({"raw_key": raw_key}));
    if let Err(error) = state.bridge.call("authenticate", authenticate).await {
        return error_response(&error);
    }
    let Ok(body_text) = std::str::from_utf8(&body) else {
        return error_response(&PublicError::invalid_json());
    };
    let client_request_id = latin1_header(&headers, "x-client-request-id");
    let argument = admission_argument(&raw_key, body_text, &headers);
    drop(body);
    let admission_text = match state.bridge.call(admission_method, argument).await {
        Ok(text) => text,
        Err(error) => return error_response(&error),
    };
    let admission = match parse_admission(&admission_text, expected_wire) {
        Ok(admission) => admission,
        Err(Parsed::Escalate(reason)) => {
            METRICS.record_escalation(classify_escalation(&reason));
            return error_response(&escalation_error(&reason));
        }
        Err(Parsed::Drift(value)) => return wire_drift_response(&state, &value, started).await,
        Err(Parsed::Malformed) => return error_response(&PublicError::internal()),
    };
    drop(admission_text);
    let mut guard = new_guard(&state, admission.request_id.clone(), started);
    let _permit = match acquire_permit(&state, &mut guard, deadline, None).await {
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

/// An admission bridge answer that is not a dispatchable decision route.
enum Parsed {
    Escalate(String),
    Drift(Value),
    Malformed,
}

#[derive(Deserialize)]
struct EscalationProbe {
    #[serde(default)]
    escalate: Option<Value>,
}

/// Decode the admission straight from its text: the route carries every
/// rung's provider payload, so no intermediate `Value` copy is kept.
fn parse_admission(text: &str, expected_wire: DecisionWire) -> Result<DecisionsAdmission, Parsed> {
    let Ok(probe) = serde_json::from_str::<EscalationProbe>(text) else {
        return Err(Parsed::Malformed);
    };
    if let Some(reason) = probe.escalate {
        return Err(Parsed::Escalate(
            reason.as_str().unwrap_or_default().to_string(),
        ));
    }
    match serde_json::from_str::<DecisionsAdmission>(text) {
        Ok(admission) if admission.wire == expected_wire => Ok(admission),
        _ => Err(Parsed::Drift(
            serde_json::from_str(text).unwrap_or(Value::Null),
        )),
    }
}

#[derive(Serialize)]
struct AdmissionArgument<'a> {
    body: &'a str,
    /// UTF-8 length of `body`, so admission bounds it without re-encoding.
    body_bytes: usize,
    #[serde(flatten)]
    fields: Map<String, Value>,
}

/// Forward authority, app identity, and trusted proxy IP without caller idempotency.
fn admission_argument(raw_key: &str, body: &str, headers: &HeaderMap) -> String {
    let mut fields = json!({
        "raw_key": raw_key,
        "client_ip": client_ip(headers),
    });
    with_app_identity(&mut fields, headers);
    let Value::Object(fields) = fields else {
        unreachable!("the admission fields are built as an object")
    };
    serde_json::to_string(&AdmissionArgument {
        body,
        body_bytes: body.len(),
        fields,
    })
    .unwrap_or_else(|_| "null".to_string())
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
            let mut error = collection_public_error(&failure.boundary());
            error.known_unbilled = started.known_unbilled;
            return Err(error);
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
        let http = state.http.for_origin(wire.tenant_origin);
        match dispatch(http, wire, deadline, admission, guard).await {
            Ok((body, usage)) => return Ok(Served { depth, body, usage }),
            Err(failure) => {
                let failure = route_failure(failure, wire, policy, counts[depth])
                    .with_tenant_origin(wire.tenant_origin);
                let boundary = failure.clone().boundary();
                let possible = successor_possible(
                    policy,
                    &admission.route,
                    deadline,
                    total_attempts,
                    counts[depth],
                    depth,
                    &counts,
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
    let dialect = match admission.wire {
        DecisionWire::Systemone if wire.dialect == "typesafe_systemone" => {
            Dialect::TypesafeSystemone
        }
        // Direct OpenAI connections speak the Responses dialect for chat; the
        // decisions endpoint shares their OpenAI error envelope.
        DecisionWire::Openai if wire.dialect == "openai_responses" => Dialect::OpenAiCompatible,
        _ => return Err(wire_failure()),
    };
    if !wire.timeout_seconds.is_finite() {
        return Err(wire_failure());
    }
    if remaining(deadline).is_zero() {
        return Err(timeout_failure());
    }
    let phase_timeout =
        Duration::try_from_secs_f64(wire.timeout_seconds.max(0.001)).map_err(|_| wire_failure())?;
    let response = open_decision_stream(
        http,
        &wire.url,
        &wire.headers,
        &wire.idempotency_key,
        &wire.upstream_payload,
        remaining(deadline).min(phase_timeout),
        dialect,
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
    let usage = match admission.wire {
        DecisionWire::Systemone => decision_usage(&payload),
        DecisionWire::Openai => openai_decision_usage(&payload),
    };
    if let Ok(usage) = usage {
        guard.record_decision_usage(usage);
    }
    let served = match admission.wire {
        DecisionWire::Systemone => public_decisions(payload, admission)?,
        DecisionWire::Openai => public_openai_decisions(payload, admission)?,
    };
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
                let probabilities = distribution(
                    &answer["probabilities"],
                    criteria.keys().map(String::as_str),
                )?;
                let choice = answer
                    .get("choice")
                    .and_then(Value::as_str)
                    .filter(|choice| criteria.contains_key(*choice))
                    .ok_or_else(|| malformed("decision choice is not a requested category"))?;
                probability(&answer["confidence"])?;
                let selected = criteria
                    .keys()
                    .position(|key| key == choice)
                    .ok_or_else(wire_failure)?;
                if !choice_can_be_highest(selected, &probabilities) {
                    return Err(malformed(
                        "decision choice is not a highest-probability category",
                    ));
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
                let probabilities = (0..criteria.len())
                    .map(|index| probability(&answer["probabilities"][index.to_string()]))
                    .collect::<Result<Vec<_>, _>>()?;
                if score < 0.0
                    || score > criteria.len().saturating_sub(1) as f64
                    || !score_matches_distribution(score, &probabilities)
                {
                    return Err(score_mismatch(
                        "decision score does not match its probability distribution",
                        score,
                        &probabilities,
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

/// Require a unit total, allowing only the hundredth-rounding envelope.
fn unit_total(values: &[f64]) -> Result<(), Failure> {
    let sum: f64 = values.iter().sum();
    if (sum - 1.0).abs() > PROBABILITY_TOLERANCE
        && !rounding_bounds(values).is_some_and(|bounds| admits_unit_total(&bounds))
    {
        return Err(malformed("decision probabilities do not sum to one"));
    }
    Ok(())
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
        cache_creation_12h_input_tokens: None,
        reasoning_tokens: None,
        billed_units: None,
        cache_writes_read_back: false,
        image_output_tokens: None,
    })
}

/// The probability-weighted level average recomputed from the wire bins.
fn expected_score(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .enumerate()
        .map(|(index, value)| index as f64 * value)
        .sum()
}

/// Check a score without treating rounded wire probabilities as exact model values.
fn score_matches_distribution(score: f64, probabilities: &[f64]) -> bool {
    if (score - expected_score(probabilities)).abs() <= PROBABILITY_TOLERANCE {
        return true;
    }
    // A single unit distribution must explain all bins together, not a separate
    // error allowance for each weighted term. Only hundredth-valued bins get an
    // envelope; finer bins stay strict.
    let Some((minimum, maximum)) = score_envelope(probabilities) else {
        return false;
    };
    if score_meets_envelope(score, minimum, maximum) {
        return true;
    }
    // TypeSafe also publishes bins rounded to hundredths whose residual was
    // folded into one highest bin so the total reads exactly one, while its
    // score keeps the unrounded expectation (production 2026-10-23: 0.71 beside
    // [0.49, 0.37, 0.11, 0.03], whose independent envelope tops out at 0.70).
    // That folded bin can sit up to half a hundredth per other bin away from
    // its latent value, so it gets its own exact envelope of that convention.
    residual_envelope(probabilities)
        .is_some_and(|(minimum, maximum)| score_meets_envelope(score, minimum, maximum))
}

/// Whether a score, read at the precision it was reported to, meets an
/// envelope in half-hundredths.
fn score_meets_envelope(score: f64, minimum: i64, maximum: i64) -> bool {
    // The score is rounded at the precision the provider REPORTED it to
    // (TypeSafe answers `score=3.9` beside bins whose latent expectation is
    // 3.87, production 2026-10-17), so it stands for the closed interval of
    // half a unit in its last decimal place, and passes only when that
    // interval meets the envelope. A whole-number score is read at tenths,
    // the coarsest precision the provider reports, so `4` cannot claim a
    // half-unit allowance; scores finer than thousandths stay strict. No
    // output is repaired or normalized, and structurally invalid
    // distributions never reach this check.
    let Some((units, scale)) = reported_precision(score) else {
        return false;
    };
    // The envelope is in half-hundredths (1/200); the score interval is
    // [(2u-1)/(2s), (2u+1)/(2s)]. Cross-multiplied to stay in integers.
    (2 * units + 1) * 100 >= minimum * scale && (2 * units - 1) * 100 <= maximum * scale
}

/// The score in units of its reported decimal place, and that place's scale:
/// tenths at the coarsest, thousandths at the finest, `None` finer still.
fn reported_precision(score: f64) -> Option<(i64, i64)> {
    [(10.0, 10), (100.0, 100), (1000.0, 1000)]
        .into_iter()
        .find_map(|(factor, scale)| decimal_units(score, factor).map(|units| (units, scale)))
}

/// Extreme expectations, in half-hundredths, of every unit distribution whose
/// bins round to the published hundredth bins; `None` when the bins are finer
/// than hundredths or no unit distribution explains them.
fn score_envelope(probabilities: &[f64]) -> Option<(i64, i64)> {
    let bounds = rounding_bounds(probabilities)?;
    if !admits_unit_total(&bounds) {
        return None;
    }
    let lower_sum: i64 = bounds.iter().map(|(low, _)| low).sum();
    let base: i64 = bounds
        .iter()
        .enumerate()
        .map(|(index, (low, _))| index as i64 * low)
        .sum();
    let minimum = rounded_expectation(base, 200 - lower_sum, bounds.iter().enumerate());
    let maximum = rounded_expectation(base, 200 - lower_sum, bounds.iter().enumerate().rev());
    Some((minimum, maximum))
}

/// A score rejection that names the numbers it was judged on.
///
/// The reason is replaced by the generic malformed-response message on the
/// public error, but rides into the operator log and the ledger detail
/// (`Failure::boundary`), so a rejected score can be told apart as provider
/// inconsistency or an overly strict check. Every value is a numeric model
/// output or a gateway constant, never prompt content.
fn score_mismatch(reason: &str, score: f64, probabilities: &[f64]) -> Failure {
    let bins = probabilities
        .iter()
        .map(f64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let render = |envelope: Option<(i64, i64)>| {
        envelope.map_or_else(
            || "none".to_string(),
            |(minimum, maximum)| format!("[{},{}]", minimum as f64 / 200.0, maximum as f64 / 200.0),
        )
    };
    malformed(&format!(
        "{reason} (score={score}, expected={}, probabilities=[{bins}], \
         rounding_envelope={}, residual_envelope={}, tolerance={PROBABILITY_TOLERANCE})",
        expected_score(probabilities),
        render(score_envelope(probabilities)),
        render(residual_envelope(probabilities)),
    ))
}

/// Check a selected category without treating rounded wire probabilities as exact.
fn choice_can_be_highest(selected: usize, probabilities: &[f64]) -> bool {
    if probabilities
        .iter()
        .all(|value| *value <= probabilities[selected])
    {
        return true;
    }
    // Rounding to hundredths can split a near tie, so only hundredth-valued bins
    // get an envelope. One unit distribution must make the selected bin highest
    // while every bin stays inside its own rounding interval.
    let Some(bounds) = rounding_bounds(probabilities) else {
        return false;
    };
    let (low, high) = bounds[selected];
    (low..=high).any(|level| {
        let mut lower_sum = level;
        let mut upper_sum = level;
        for (index, (low, high)) in bounds.iter().enumerate() {
            if index == selected {
                continue;
            }
            if *low > level {
                return false;
            }
            lower_sum += low;
            upper_sum += (*high).min(level);
        }
        lower_sum <= 200 && upper_sum >= 200
    })
}

/// Half-hundredth rounding intervals of hundredth-valued probabilities, clipped to
/// `[0, 1]`. Integer units avoid floating-point boundary tolerances; finer data
/// gets no interval and stays strict.
fn rounding_bounds(probabilities: &[f64]) -> Option<Vec<(i64, i64)>> {
    probabilities
        .iter()
        .map(|value| hundredths(*value).map(|bin| ((2 * bin - 1).max(0), (2 * bin + 1).min(200))))
        .collect()
}

/// Whether one unit distribution fits inside every closed rounding interval.
fn admits_unit_total(bounds: &[(i64, i64)]) -> bool {
    let lower_sum: i64 = bounds.iter().map(|(low, _)| low).sum();
    let upper_sum: i64 = bounds.iter().map(|(_, high)| high).sum();
    lower_sum <= 200 && upper_sum >= 200
}

/// Recognize decimal hundredths, including their binary floating-point residue.
fn hundredths(value: f64) -> Option<i64> {
    decimal_units(value, 100.0)
}

/// Recognize a value with at most the decimal precision `scale` names.
fn decimal_units(value: f64, scale: f64) -> Option<i64> {
    let scaled = value * scale;
    let rounded = scaled.round();
    // Allow binary representation and multiplication roundoff, not extra decimal
    // precision. A relative bound also keeps tiny nonzero bins distinct from zero.
    let roundoff = 2.0 * f64::EPSILON * scaled.abs();
    ((scaled - rounded).abs() <= roundoff).then_some(rounded as i64)
}

/// Extremize a weighted sum by filling the lowest or highest indices first.
fn rounded_expectation<'a>(
    mut value: i64,
    mut remaining: i64,
    bins: impl Iterator<Item = (usize, &'a (i64, i64))>,
) -> i64 {
    for (index, (low, high)) in bins {
        let added = remaining.min(high - low);
        value += index as i64 * added;
        remaining -= added;
    }
    value
}

fn probability(value: &Value) -> Result<f64, Failure> {
    value
        .as_f64()
        .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
        .ok_or_else(|| {
            malformed("decision probability or confidence must be finite and between zero and one")
        })
}

/// Validate exact keys and a unit total, returning probabilities in key order.
fn distribution<'a>(
    value: &Value,
    keys: impl Iterator<Item = &'a str>,
) -> Result<Vec<f64>, Failure> {
    let probabilities = value
        .as_object()
        .ok_or_else(|| malformed("decision omitted its probability object"))?;
    let values = keys
        .map(|key| probability(probabilities.get(key).unwrap_or(&Value::Null)))
        .collect::<Result<Vec<_>, _>>()?;
    let sum: f64 = values.iter().sum();
    // Independently rounded hundredth bins may publish 0.99 or 1.01; accept that
    // total only when one unit distribution fits every bin's rounding interval.
    // Finer-precision totals keep the exact normalization check.
    if probabilities.len() != values.len()
        || ((sum - 1.0).abs() > PROBABILITY_TOLERANCE
            && !rounding_bounds(&values).is_some_and(|bounds| admits_unit_total(&bounds)))
    {
        return Err(malformed(
            "decision probability keys or total do not match the request",
        ));
    }
    Ok(values)
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
#[path = "route_decisions/admission_tests.rs"]
mod admission_tests;
#[path = "route_decisions/body_budget.rs"]
mod body_budget;
#[path = "route_decisions/openai.rs"]
mod openai;
#[cfg(test)]
#[path = "route_decisions/openai_tests.rs"]
mod openai_tests;
#[path = "route_decisions/residual.rs"]
mod residual;
#[cfg(test)]
#[path = "route_decisions/score_tests.rs"]
mod score_tests;
#[cfg(test)]
mod tests;
