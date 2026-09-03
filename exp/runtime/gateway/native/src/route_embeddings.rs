//! The native embeddings surface: the `POST /v1/embeddings` handler.
//!
//! Embeddings are message-less and never stream, so the surface skips the
//! relay, replay, and reasoning machinery entirely: admission returns one
//! fully built OpenAI-wire `/embeddings` payload per certified deployment, the
//! ladder below reserves each physical attempt through `start_attempt`,
//! POSTs the payload, buffers the JSON answer, validates it against the
//! request (one vector per input, a reported `prompt_tokens`), and settles the
//! winning attempt with input-only usage. Failures walk the same
//! same-deployment/failover ladder as chat under the control plane's
//! deployment-health and budget authority. An inbound `Idempotency-Key` is
//! ignored: the surface has no replay protocol yet, and keying it here would
//! share the chat replay namespace.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Response;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::admission::{acquire_permit, new_guard, wire_drift_response};
use crate::dialects::{Dialect, MAXIMUM_RETAINED_OUTPUT_BYTES, OUTPUT_OVERFLOW_MESSAGE};
use crate::encode::compact_json;
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::Usage;
use crate::metrics::{classify_escalation, METRICS};
use crate::relay::{collection_public_error, remaining};
use crate::respond::{
    bearer_key, error_response, escalation_error, json_response, latin1_header, read_body,
};
use crate::server::AppState;
use crate::settlement::AttemptGuard;
use crate::upstream::open_stream;
use crate::waterfall::{successor_possible, DeploymentWire, RoutePolicy, StartResponse};

/// The wire configuration returned by one successful embeddings admission.
#[derive(Debug, Clone, Deserialize)]
struct EmbeddingsAdmission {
    request_id: String,
    alias: String,
    alias_revision_id: String,
    exact_model_id: String,
    route_reason: String,
    route: Vec<DeploymentWire>,
    /// Number of input strings; the provider must answer with exactly this
    /// many vectors.
    input_count: usize,
    maximum_total_attempts: u32,
    maximum_same_deployment_attempts: u32,
}

impl EmbeddingsAdmission {
    fn policy(&self) -> RoutePolicy {
        RoutePolicy {
            maximum_total_attempts: self.maximum_total_attempts.max(1),
            maximum_same_deployment_attempts: self.maximum_same_deployment_attempts.max(1),
            refusal_failover: false,
        }
    }
}

/// One validated provider answer: the public body and its billable usage.
struct Served {
    depth: usize,
    body: Value,
    usage: Usage,
}

pub(crate) async fn embeddings(
    State(state): State<AppState>,
    request: axum::extract::Request,
) -> Response {
    state.handled_requests.fetch_add(1, Ordering::Relaxed);
    let started = Instant::now();
    let deadline = started + state.request_timeout;
    let (parts, raw_body) = request.into_parts();
    let headers = parts.headers;
    let body = match read_body(raw_body).await {
        Ok(body) => body,
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
    let body_text = match String::from_utf8(body.to_vec()) {
        Ok(text) => text,
        Err(_) => return error_response(&PublicError::invalid_json()),
    };
    let client_request_id = latin1_header(&headers, "x-client-request-id");

    let admit_argument = compact_json(&json!({"raw_key": raw_key, "body": body_text}));
    let admission_text = match state.bridge.call("admit_embeddings", admit_argument).await {
        Ok(text) => text,
        Err(error) => return error_response(&error),
    };
    let admission_value: Value = match serde_json::from_str(&admission_text) {
        Ok(value) => value,
        Err(_) => return error_response(&PublicError::internal()),
    };
    if let Some(reason) = admission_value.get("escalate") {
        METRICS.record_escalation(classify_escalation(reason.as_str().unwrap_or_default()));
        // The accepted request is already finalized control-plane side;
        // startup validation guarantees native servability, so fail closed.
        return error_response(&escalation_error());
    }
    let admission: EmbeddingsAdmission = match serde_json::from_value(admission_value.clone()) {
        Ok(admission) => admission,
        Err(_) => return wire_drift_response(&state, &admission_value, started).await,
    };
    let mut guard = new_guard(&state, admission.request_id.clone(), started);
    let permit = match acquire_permit(&state, &mut guard, deadline).await {
        Ok(permit) => permit,
        Err(response) => return *response,
    };
    let _permit = permit;

    match run_ladder(&state, &admission, &raw_key, &mut guard, deadline).await {
        Err(error) => error_response(&error),
        Ok(served) => {
            if !guard
                .settle("completed", Some(&served.usage), &[], None, true)
                .await
            {
                // The ledger never learned the outcome; the caller must not
                // receive vectors the gateway cannot account for.
                return error_response(&PublicError::internal());
            }
            let response_headers = served_headers(&admission, served.depth, client_request_id);
            json_response(StatusCode::OK, &served.body, &response_headers)
        }
    }
}

/// Walk the certified ladder to one validated provider answer or the public
/// error of the exhausting failure. Every reserved attempt settles exactly
/// once through `guard`; on `Ok` the winning attempt is still open for the
/// caller's finalizing settlement.
async fn run_ladder(
    state: &AppState,
    admission: &EmbeddingsAdmission,
    raw_key: &str,
    guard: &mut AttemptGuard,
    deadline: Instant,
) -> Result<Served, PublicError> {
    let policy = admission.policy();
    let mut total_attempts: u32 = 0;
    let mut counts: Vec<u32> = vec![0; admission.route.len()];
    let mut current_depth: Option<usize> = None;
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
            })),
        }));
        let started_text = match state.bridge.call("start_attempt", argument).await {
            Ok(text) => text,
            Err(error) => {
                // The control plane finalized the request before raising.
                guard.disarm_finalized("failed");
                return Err(error);
            }
        };
        let started: StartResponse = match serde_json::from_str(&started_text) {
            Ok(started) => started,
            Err(_) => {
                guard
                    .abandon(&Failure::new(
                        FailureClass::Internal,
                        "gateway attempt wire contract failed",
                    ))
                    .await;
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
            guard
                .abandon(&Failure::new(
                    FailureClass::Internal,
                    "gateway attempt wire contract failed",
                ))
                .await;
            return Err(PublicError::internal());
        };
        let Some(wire) = admission.route.get(depth) else {
            guard.rebind(attempt_id);
            let failure = Failure::new(
                FailureClass::Internal,
                "gateway attempt wire contract failed",
            );
            guard
                .settle("failed", None, &[], Some(&failure), true)
                .await;
            return Err(PublicError::internal());
        };
        if current_depth == Some(depth) {
            METRICS.record_open_retry();
        }
        guard.rebind(attempt_id);
        total_attempts += 1;
        counts[depth] += 1;
        match dispatch(&state.http, wire, deadline, admission).await {
            Ok((body, usage)) => return Ok(Served { depth, body, usage }),
            Err((failure, opened)) => {
                if opened {
                    guard.mark_opened();
                }
                let boundary = failure.clone().boundary();
                let possible = successor_possible(
                    policy,
                    admission.route.len(),
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
                if possible {
                    current_depth = Some(depth);
                    last_failure = Some(failure);
                    continue;
                }
                return Err(collection_public_error(&boundary));
            }
        }
    }
}

/// POST one admitted payload and validate the buffered answer. The error
/// carries whether the provider dispatch opened (for deployment-health
/// recording); a rejected open never opened.
async fn dispatch(
    http: &reqwest::Client,
    wire: &DeploymentWire,
    deadline: Instant,
    admission: &EmbeddingsAdmission,
) -> Result<(Value, Usage), (Failure, bool)> {
    let phase_timeout = Duration::from_secs_f64(wire.timeout_seconds.max(0.001));
    let bound = remaining(deadline).min(phase_timeout);
    // Every embeddings wire is the OpenAI `/embeddings` shape, whichever chat
    // dialect the connection speaks, so client-error attribution reads the
    // OpenAI error envelope.
    let response = open_stream(
        http,
        &wire.url,
        &wire.headers,
        &wire.idempotency_key,
        &wire.upstream_payload,
        None,
        bound,
        Dialect::OpenAiCompatible,
    )
    .await
    .map_err(|failure| (failure, false))?;
    if response
        .content_length()
        .is_some_and(|length| length > MAXIMUM_RETAINED_OUTPUT_BYTES as u64)
    {
        return Err((
            Failure::new(FailureClass::MalformedResponse, OUTPUT_OVERFLOW_MESSAGE),
            true,
        ));
    }
    let bytes = read_bounded_body(response, deadline, phase_timeout)
        .await
        .map_err(|failure| (failure, true))?;
    let payload: Value = match serde_json::from_slice(&bytes) {
        Ok(payload) => payload,
        Err(_) => return Err((malformed("embeddings response is not JSON"), true)),
    };
    public_embeddings(payload, admission).map_err(|failure| (failure, true))
}

/// Read one successful provider body chunk by chunk under the retained-output
/// cap, so an upstream that omits `Content-Length` cannot make the gateway
/// buffer an unbounded answer before the size check runs. Each chunk read is
/// bounded by the deployment's transport timeout and the request deadline.
async fn read_bounded_body(
    mut response: reqwest::Response,
    deadline: Instant,
    phase_timeout: Duration,
) -> Result<Vec<u8>, Failure> {
    let mut body: Vec<u8> = Vec::new();
    loop {
        let bound = remaining(deadline).min(phase_timeout);
        let chunk = match tokio::time::timeout(bound, response.chunk()).await {
            Ok(Ok(Some(chunk))) => chunk,
            Ok(Ok(None)) => return Ok(body),
            Ok(Err(_)) => {
                return Err(Failure::new(
                    FailureClass::Transport,
                    "provider connection failed while sending the embeddings response",
                )
                .with_retry(true, true))
            }
            Err(_) => {
                return Err(Failure::new(
                    FailureClass::Timeout,
                    "provider did not finish the embeddings response in time",
                )
                .with_retry(false, true))
            }
        };
        if body.len() + chunk.len() > MAXIMUM_RETAINED_OUTPUT_BYTES {
            return Err(Failure::new(
                FailureClass::MalformedResponse,
                OUTPUT_OVERFLOW_MESSAGE,
            ));
        }
        body.extend_from_slice(&chunk);
    }
}

/// A provider answer the gateway cannot bill or relay; never retried on the
/// same deployment, eligible for a certified fallback.
fn malformed(reason: &str) -> Failure {
    Failure::new(FailureClass::MalformedResponse, reason).with_retry(false, true)
}

/// Validate one provider `/embeddings` body against the admitted request and
/// rebuild it as the public answer: exactly one vector per input, in input
/// order, the public alias as `model`, and the billed `prompt_tokens`.
/// Vectors pass through untouched (`float` arrays or `base64` strings), so the
/// caller receives the provider's exact values.
fn public_embeddings(
    payload: Value,
    admission: &EmbeddingsAdmission,
) -> Result<(Value, Usage), Failure> {
    let object = payload
        .as_object()
        .ok_or_else(|| malformed("embeddings response is not an object"))?;
    let data = object
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("embeddings response omitted the data array"))?;
    if data.len() != admission.input_count {
        return Err(malformed(
            "embeddings response count does not match the request input count",
        ));
    }
    let mut ordered: Vec<Option<Value>> = vec![None; admission.input_count];
    for (position, item) in data.iter().enumerate() {
        let entry = item
            .as_object()
            .ok_or_else(|| malformed("embeddings response item is not an object"))?;
        let index = match entry.get("index") {
            None => position,
            Some(value) => value
                .as_u64()
                .and_then(|index| usize::try_from(index).ok())
                .ok_or_else(|| malformed("embeddings response index is not an integer"))?,
        };
        if index >= admission.input_count || ordered[index].is_some() {
            return Err(malformed(
                "embeddings response indexes are not unique input indexes",
            ));
        }
        let vector = entry
            .get("embedding")
            .filter(|vector| vector.is_array() || vector.is_string())
            .ok_or_else(|| malformed("embeddings response item omitted its vector"))?;
        ordered[index] = Some(json!({
            "object": "embedding",
            "index": index,
            "embedding": vector,
        }));
    }
    let data: Vec<Value> = ordered
        .into_iter()
        .map(|item| item.ok_or_else(|| malformed("embeddings response omitted an input index")))
        .collect::<Result<_, _>>()?;
    // The surface bills on this count, so an omitted or malformed
    // prompt_tokens is a malformed response, never a zero.
    let prompt_tokens = object
        .get("usage")
        .and_then(Value::as_object)
        .and_then(|usage| usage.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .ok_or_else(|| malformed("embeddings response omitted usage.prompt_tokens"))?;
    let mut public = Map::new();
    public.insert("object".to_string(), Value::from("list"));
    public.insert("data".to_string(), Value::Array(data));
    public.insert("model".to_string(), Value::from(admission.alias.as_str()));
    public.insert(
        "usage".to_string(),
        json!({"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens}),
    );
    let usage = Usage {
        input_tokens: Some(prompt_tokens),
        output_tokens: Some(0),
        cached_input_tokens: None,
        cache_creation_input_tokens: None,
        reasoning_tokens: None,
    };
    Ok((Value::Object(public), usage))
}

/// The gateway identity headers of one served embeddings response: the
/// commit-independent and commit-dependent sets the chat surface emits.
fn served_headers(
    admission: &EmbeddingsAdmission,
    depth: usize,
    client_request_id: Option<String>,
) -> Vec<(String, String)> {
    let (provider, deployment_id) = admission
        .route
        .get(depth)
        .map(|wire| (wire.provider.clone(), wire.deployment_id.clone()))
        .unwrap_or_default();
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
        ("x-gateway-provider".to_string(), provider),
        ("x-gateway-deployment".to_string(), deployment_id),
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
mod tests {
    use super::*;

    fn admission(input_count: usize) -> EmbeddingsAdmission {
        EmbeddingsAdmission {
            request_id: "request-1".to_string(),
            alias: "embedder".to_string(),
            alias_revision_id: "revision-one".to_string(),
            exact_model_id: "model-exact".to_string(),
            route_reason: "direct".to_string(),
            route: Vec::new(),
            input_count,
            maximum_total_attempts: 8,
            maximum_same_deployment_attempts: 2,
        }
    }

    #[test]
    fn provider_answer_is_reordered_relabeled_and_billed_on_prompt_tokens() {
        let payload = json!({
            "object": "list",
            "model": "provider-model-exact",
            "data": [
                {"object": "embedding", "index": 1, "embedding": [0.5, 0.5]},
                {"object": "embedding", "index": 0, "embedding": "AACAPwAAAAA="},
            ],
            "usage": {"prompt_tokens": 7, "total_tokens": 7},
        });
        let (body, usage) = public_embeddings(payload, &admission(2)).expect("valid answer");
        assert_eq!(
            body,
            json!({
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": "AACAPwAAAAA="},
                    {"object": "embedding", "index": 1, "embedding": [0.5, 0.5]},
                ],
                "model": "embedder",
                "usage": {"prompt_tokens": 7, "total_tokens": 7},
            })
        );
        assert_eq!(usage.input_tokens, Some(7));
        assert_eq!(usage.output_tokens, Some(0));
    }

    #[test]
    fn count_mismatch_duplicate_index_and_missing_usage_are_malformed() {
        let short = json!({"data": [{"embedding": [1.0]}], "usage": {"prompt_tokens": 1}});
        let duplicated = json!({
            "data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [1.0]}],
            "usage": {"prompt_tokens": 1},
        });
        let unbilled = json!({"data": [{"embedding": [1.0]}, {"embedding": [2.0]}]});
        for payload in [short, duplicated, unbilled] {
            let failure = public_embeddings(payload, &admission(2)).expect_err("malformed");
            assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
            assert!(failure.failover_eligible);
            assert!(!failure.retryable_same_deployment);
        }
    }
}
