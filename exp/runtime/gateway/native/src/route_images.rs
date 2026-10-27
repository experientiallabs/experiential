//! The native image surface: the `POST /v1/images/generations` handler and
//! the shared ladder `POST /v1/images/edits` reuses.
//!
//! Prompt in (plus input images for an edit), images out, never streamed:
//! admission returns one fully built provider payload per certified
//! deployment and the ladder below reserves each attempt through
//! `start_attempt`, POSTs the payload, buffers the JSON answer under the
//! retained-output cap, validates it against the request, and settles the
//! winning attempt. Two answer shapes exist, named per rung by admission:
//!
//! - the OpenAI Images API: one image per requested `n` and a reported token
//!   usage, settled on the provider's prompt and image token counts; a rung
//!   whose lane prices images per image (dall-e, xAI Grok Imagine) settles
//!   billed `image` units instead, one per image returned, never its tokens;
//! - Stability AI's Bedrock InvokeModel answer (`images`, `seeds`,
//!   `finish_reasons`): one base64 image, always billed per image. Its body
//!   is frozen at admission and signed (SigV4) immediately before each
//!   physical POST; a content-filter finish reason is a refusal billed $0.
//!
//! A token-billed answer without usage is refused as unbillable, so no image
//! is ever handed out unaccounted.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Response;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::admission::{acquire_permit, new_guard, wire_drift_response};
use crate::bridge::Bridge;
use crate::dialects::{Dialect, MAXIMUM_RETAINED_OUTPUT_BYTES, OUTPUT_OVERFLOW_MESSAGE};
use crate::dispatch_capacity::DispatchPermit;
use crate::encode::compact_json;
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{BilledUnits, Usage};
use crate::metrics::{classify_escalation, METRICS};
use crate::relay::{collection_public_error, remaining};
use crate::respond::{
    bearer_key, error_response, escalation_error, json_response, latin1_header, read_body,
    with_app_identity,
};
use crate::server::AppState;
use crate::settlement::AttemptGuard;
use crate::upstream::open_stream;
use crate::waterfall::{
    customer_owned, dispatch_headers, successor_possible, DeploymentWire, RoutePolicy,
    StartResponse,
};

/// The wire configuration returned by one successful images admission.
#[derive(Debug, Clone, Deserialize)]
struct ImagesAdmission {
    request_id: String,
    alias: String,
    alias_revision_id: String,
    exact_model_id: String,
    route_reason: String,
    route: Vec<DeploymentWire>,
    /// The requested `n`; the provider must answer with exactly this many images.
    image_count: usize,
    /// Per-rung unit billing aligned with `route`: `Some` bills `image` units at
    /// that variant, `None` (or an absent list) bills the provider's tokens.
    #[serde(default)]
    unit_billing: Vec<Option<ImageUnitBilling>>,
    /// Per-rung answer shapes aligned with `route` (absent means OpenAI).
    #[serde(default)]
    wire_shapes: Vec<WireShape>,
    maximum_total_attempts: u32,
    maximum_same_deployment_attempts: u32,
}

impl ImagesAdmission {
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

/// One per-image priced rung's billing: the variant its unit card prices.
#[derive(Debug, Clone, Deserialize)]
struct ImageUnitBilling {
    #[serde(default)]
    variant: String,
}

/// The answer shape one rung's provider returns.
#[derive(Debug, Clone, Deserialize)]
struct WireShape {
    /// `openai` (the Images API) or `stability` (Bedrock InvokeModel).
    shape: String,
    /// The encoded format a Stability rung returns (`png`, `jpeg`, `webp`).
    #[serde(default)]
    output_format: Option<String>,
}

impl WireShape {
    fn stability(&self) -> bool {
        self.shape == "stability"
    }
}

struct Served {
    depth: usize,
    body: Value,
    usage: Usage,
}

pub(crate) async fn images(
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
    let admit_value = json!({"raw_key": raw_key, "body": body_text});
    serve_images(
        &state,
        &headers,
        started,
        deadline,
        &raw_key,
        "admit_images",
        admit_value,
        None,
    )
    .await
}

/// Admit one parsed image request through `admit_method`, then walk its
/// ladder and answer: shared by the generations and edits routes.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn serve_images(
    state: &AppState,
    headers: &axum::http::HeaderMap,
    started: Instant,
    deadline: Instant,
    raw_key: &str,
    admit_method: &'static str,
    mut admit_value: Value,
    upload_permit: Option<DispatchPermit>,
) -> Response {
    let client_request_id = latin1_header(headers, "x-client-request-id");
    with_app_identity(&mut admit_value, headers);
    let admit_argument = compact_json(&admit_value);
    let admission_text = match state.bridge.call(admit_method, admit_argument).await {
        Ok(text) => text,
        Err(error) => return error_response(&error),
    };
    let admission_value: Value = match serde_json::from_str(&admission_text) {
        Ok(value) => value,
        Err(_) => return error_response(&PublicError::internal()),
    };
    if let Some(reason) = admission_value.get("escalate") {
        METRICS.record_escalation(classify_escalation(reason.as_str().unwrap_or_default()));
        return error_response(&escalation_error(reason.as_str().unwrap_or_default()));
    }
    let admission: ImagesAdmission = match serde_json::from_value(admission_value.clone()) {
        Ok(admission) => admission,
        Err(_) => return wire_drift_response(state, &admission_value, started).await,
    };
    let mut guard = new_guard(state, admission.request_id.clone(), started);
    let permit = match upload_permit {
        Some(permit) => {
            guard.hold_dispatch(permit.clone());
            permit
        }
        None => match acquire_permit(state, &mut guard, deadline, None).await {
            Ok(permit) => permit,
            Err(response) => return *response,
        },
    };
    let _permit = permit;

    match run_ladder(state, &admission, raw_key, &mut guard, deadline).await {
        Err(error) => error_response(&error),
        Ok(served) => {
            if !guard
                .settle("completed", Some(&served.usage), &[], None, true)
                .await
            {
                // Never hand out images the ledger did not record.
                return error_response(&PublicError::internal());
            }
            let response_headers = served_headers(&admission, served.depth, client_request_id);
            json_response(StatusCode::OK, &served.body, &response_headers)
        }
    }
}

/// Walk the certified ladder to one validated provider answer or the public
/// error of the exhausting failure (same contract as the embeddings ladder).
async fn run_ladder(
    state: &AppState,
    admission: &ImagesAdmission,
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
            let mut error = collection_public_error(&failure.boundary());
            error.known_unbilled = started.known_unbilled;
            return Err(error);
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
        let units = admission.unit_billing.get(depth).cloned().flatten();
        let shape = admission.wire_shapes.get(depth);
        let http = state.http.for_origin(wire.tenant_origin);
        let target = DispatchTarget {
            bridge: &state.bridge,
            request_id: &admission.request_id,
            wire,
            shape,
        };
        match dispatch(http, &target, deadline, admission, units.as_ref()).await {
            Ok((body, usage)) => return Ok(Served { depth, body, usage }),
            Err((failure, opened)) => {
                let failure = failure.with_tenant_origin(wire.tenant_origin);
                if opened {
                    guard.mark_opened();
                }
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

/// One attempt's dispatch target: the rung's wire and answer shape, plus the
/// bridge a body-signing rung signs through.
struct DispatchTarget<'a> {
    bridge: &'a Bridge,
    request_id: &'a str,
    wire: &'a DeploymentWire,
    shape: Option<&'a WireShape>,
}

/// POST one admitted payload and validate the buffered answer; the error
/// carries whether the provider dispatch opened. A body-signing rung
/// (Bedrock) is signed here, immediately before the POST, and its frozen
/// body is sent verbatim; a customer-managed rung's credential failure is
/// the customer's to fix, never operator deadness.
async fn dispatch(
    http: &crate::upstream::UpstreamClient,
    target: &DispatchTarget<'_>,
    deadline: Instant,
    admission: &ImagesAdmission,
    units: Option<&ImageUnitBilling>,
) -> Result<(Value, Usage), (Failure, bool)> {
    let wire = target.wire;
    let stability = target.shape.is_some_and(WireShape::stability);
    let headers = dispatch_headers(target.bridge, target.request_id, wire)
        .await
        .map_err(|_| {
            (
                Failure::new(
                    FailureClass::ProviderAuthentication,
                    "provider dispatch signing failed",
                ),
                false,
            )
        })?;
    let dialect = if stability {
        Dialect::BedrockConverseStream
    } else {
        Dialect::OpenAiCompatible
    };
    let phase_timeout = Duration::from_secs_f64(wire.timeout_seconds.max(0.001));
    let bound = remaining(deadline).min(phase_timeout);
    let response = open_stream(
        http,
        &wire.url,
        &headers,
        &wire.idempotency_key,
        &wire.upstream_payload,
        wire.upstream_body.as_deref(),
        bound,
        dialect,
    )
    .await
    .map_err(|failure| (customer_owned(failure, wire), false))?;
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
        Err(_) => return Err((malformed("images response is not JSON"), true)),
    };
    let answer = match target.shape.filter(|shape| shape.stability()) {
        Some(shape) => public_stability_images(payload, admission, units, shape),
        None => public_images(payload, admission, units),
    };
    answer.map_err(|failure| (failure, true))
}

/// Read one provider body chunk by chunk under the retained-output cap (a
/// generated image set is large; the cap still bounds it).
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
                    "provider connection failed while sending the images response",
                )
                .with_retry(true, true))
            }
            Err(_) => {
                return Err(Failure::new(
                    FailureClass::Timeout,
                    "provider did not finish the images response in time",
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

fn malformed(reason: &str) -> Failure {
    Failure::new(FailureClass::MalformedResponse, reason).with_retry(false, true)
}

/// Validate one provider `/images/generations` body against the admitted
/// request and rebuild it as the public answer: exactly `n` images, each a
/// `b64_json` or `url` string (passed through untouched, with any
/// `revised_prompt`), and a billable meter: `units` bills one `image` unit per
/// image returned, otherwise the provider's token usage must be present. The
/// public body keeps the provider's `created`, any `usage`, and the echoed
/// rendering facts.
fn public_images(
    payload: Value,
    admission: &ImagesAdmission,
    units: Option<&ImageUnitBilling>,
) -> Result<(Value, Usage), Failure> {
    let object = payload
        .as_object()
        .ok_or_else(|| malformed("images response is not an object"))?;
    let data = object
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("images response omitted the data array"))?;
    if data.len() != admission.image_count {
        return Err(malformed(
            "images response count does not match the requested n",
        ));
    }
    let mut images: Vec<Value> = Vec::with_capacity(data.len());
    for item in data {
        let entry = item
            .as_object()
            .ok_or_else(|| malformed("images response item is not an object"))?;
        let mut public_item = Map::new();
        let mut carried = false;
        for key in ["b64_json", "url", "revised_prompt"] {
            if let Some(value) = entry.get(key).filter(|value| value.is_string()) {
                carried |= key != "revised_prompt";
                public_item.insert(key.to_string(), value.clone());
            }
        }
        if !carried {
            return Err(malformed(
                "images response item carried neither b64_json nor url",
            ));
        }
        images.push(Value::Object(public_item));
    }
    let mut public = Map::new();
    public.insert(
        "created".to_string(),
        object.get("created").cloned().unwrap_or(Value::from(0)),
    );
    let count = images.len() as u64;
    public.insert("data".to_string(), Value::Array(images));
    for key in ["background", "output_format", "quality", "size"] {
        if let Some(value) = object.get(key) {
            public.insert(key.to_string(), value.clone());
        }
    }
    if let Some(units) = units {
        // A per-image lane bills the images returned; any token counts the
        // provider reports are passed through for the caller, never billed.
        if let Some(usage) = object.get("usage").filter(|usage| usage.is_object()) {
            public.insert("usage".to_string(), usage.clone());
        }
        let usage = Usage {
            billed_units: Some(BilledUnits {
                kind: "image".to_string(),
                variant: units.variant.clone(),
                quantity_milli: count.saturating_mul(1000),
            }),
            ..Usage::default()
        };
        return Ok((Value::Object(public), usage));
    }
    // A token-priced lane bills on these counts: an answer without token
    // usage is unbillable here and is refused, never a zero.
    let usage = object
        .get("usage")
        .and_then(Value::as_object)
        .ok_or_else(|| malformed("images response omitted its token usage"))?;
    let input_tokens = usage
        .get("input_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| malformed("images response omitted usage.input_tokens"))?;
    let output_tokens = usage
        .get("output_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| malformed("images response omitted usage.output_tokens"))?;
    public.insert("usage".to_string(), Value::Object(usage.clone()));
    let usage = Usage {
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
    };
    Ok((Value::Object(public), usage))
}

/// Validate one Stability (Bedrock InvokeModel) answer and rebuild it as the
/// public Images answer: `n` base64 images (Stability returns one per call)
/// in `data[].b64_json`, the requested `output_format`, and one billed
/// `image` unit per image. A content-filter finish reason is a refusal (the
/// model's answer to the request, never rung deadness); an inference error
/// is a retryable provider failure.
fn public_stability_images(
    payload: Value,
    admission: &ImagesAdmission,
    units: Option<&ImageUnitBilling>,
    shape: &WireShape,
) -> Result<(Value, Usage), Failure> {
    let object = payload
        .as_object()
        .ok_or_else(|| malformed("images response is not an object"))?;
    if let Some(reasons) = object.get("finish_reasons").and_then(Value::as_array) {
        if let Some(reason) = reasons.iter().find_map(Value::as_str) {
            return Err(match reason.strip_prefix("Filter reason:") {
                Some(what) => Failure::new(
                    FailureClass::Refusal,
                    &format!(
                        "the image model's content filter refused the request ({})",
                        filter_subject(what)
                    ),
                ),
                None => Failure::new(
                    FailureClass::ProviderInternal,
                    "the image model reported an inference error",
                )
                .with_retry(true, true),
            });
        }
    }
    let encoded = object
        .get("images")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("images response omitted the images array"))?;
    if encoded.len() != admission.image_count {
        return Err(malformed(
            "images response count does not match the requested n",
        ));
    }
    let mut images: Vec<Value> = Vec::with_capacity(encoded.len());
    for item in encoded {
        let data = item
            .as_str()
            .filter(|data| !data.is_empty())
            .ok_or_else(|| malformed("images response item is not a base64 string"))?;
        images.push(json!({"b64_json": data}));
    }
    // Admission routes a Stability rung only with a per-image price card.
    let units = units.ok_or_else(|| malformed("a Stability image rung has no per-image price"))?;
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or_default();
    let count = images.len() as u64;
    let mut public = Map::new();
    public.insert("created".to_string(), Value::from(created));
    public.insert("data".to_string(), Value::Array(images));
    if let Some(format) = &shape.output_format {
        public.insert("output_format".to_string(), Value::from(format.clone()));
    }
    let usage = Usage {
        billed_units: Some(BilledUnits {
            kind: "image".to_string(),
            variant: units.variant.clone(),
            quantity_milli: count.saturating_mul(1000),
        }),
        ..Usage::default()
    };
    Ok((Value::Object(public), usage))
}

/// Name what Stability's content filter screened (`prompt`, `output image`,
/// `input image`) from its finish reason, as a bounded vocabulary only.
fn filter_subject(reason: &str) -> &'static str {
    match reason.trim() {
        "prompt" => "prompt",
        "output image" => "output image",
        "input image" => "input image",
        _ => "content",
    }
}

fn served_headers(
    admission: &ImagesAdmission,
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

    fn admission(image_count: usize) -> ImagesAdmission {
        ImagesAdmission {
            request_id: "request-1".to_string(),
            alias: "painter".to_string(),
            alias_revision_id: "revision-one".to_string(),
            exact_model_id: "model-exact".to_string(),
            route_reason: "direct".to_string(),
            route: Vec::new(),
            image_count,
            unit_billing: Vec::new(),
            wire_shapes: Vec::new(),
            maximum_total_attempts: 8,
            maximum_same_deployment_attempts: 2,
        }
    }

    #[test]
    fn provider_answer_keeps_images_and_bills_token_usage() {
        let payload = json!({
            "created": 1_700_000_000,
            "background": "opaque",
            "output_format": "png",
            "quality": "low",
            "size": "1024x1024",
            "data": [
                {"b64_json": "aW1hZ2U=", "revised_prompt": "a cat"},
                {"url": "https://example.invalid/img.png"},
            ],
            "usage": {
                "input_tokens": 11,
                "output_tokens": 272,
                "total_tokens": 283,
                "input_tokens_details": {"text_tokens": 11, "image_tokens": 0},
            },
        });
        let (body, usage) =
            public_images(payload.clone(), &admission(2), None).expect("valid answer");
        assert_eq!(body["data"], payload["data"]);
        assert_eq!(body["usage"], payload["usage"]);
        assert_eq!(body["created"], json!(1_700_000_000));
        assert_eq!(body["quality"], json!("low"));
        assert_eq!(usage.input_tokens, Some(11));
        assert_eq!(usage.output_tokens, Some(272));
    }

    #[test]
    fn unbilled_short_or_imageless_answers_are_malformed() {
        let unbilled = json!({"created": 1, "data": [{"b64_json": "aW1hZ2U="}]});
        let short =
            json!({"data": [{"b64_json": "x"}], "usage": {"input_tokens": 1, "output_tokens": 1}});
        let imageless = json!({
            "data": [{"revised_prompt": "only text"}, {"b64_json": "x"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        });
        for payload in [unbilled, short, imageless] {
            let failure = public_images(payload, &admission(2), None).expect_err("malformed");
            assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
            assert!(failure.failover_eligible);
        }
    }

    #[test]
    fn a_per_image_rung_bills_the_images_returned_not_tokens() {
        let payload = json!({
            "created": 1,
            "data": [{"url": "https://example.invalid/a.png"}, {"b64_json": "x"}],
            "usage": {"input_tokens": 9, "output_tokens": 0},
        });
        let units = ImageUnitBilling {
            variant: "hd/1024x1024".to_string(),
        };
        let (body, usage) =
            public_images(payload.clone(), &admission(2), Some(&units)).expect("valid answer");
        assert_eq!(body["usage"], payload["usage"]);
        assert_eq!(usage.input_tokens, None);
        assert_eq!(usage.output_tokens, None);
        let billed = usage.billed_units.expect("image units");
        assert_eq!(billed.kind, "image");
        assert_eq!(billed.variant, "hd/1024x1024");
        assert_eq!(billed.quantity_milli, 2_000);
        let usageless = json!({"data": [{"url": "https://example.invalid/a.png"}]});
        let (body, usage) =
            public_images(usageless, &admission(1), Some(&units)).expect("usageless answer");
        assert!(body.get("usage").is_none());
        assert_eq!(usage.billed_units.expect("units").quantity_milli, 1_000);
    }

    fn stability_shape() -> WireShape {
        WireShape {
            shape: "stability".to_string(),
            output_format: Some("jpeg".to_string()),
        }
    }

    fn flat_units() -> ImageUnitBilling {
        ImageUnitBilling {
            variant: String::new(),
        }
    }

    #[test]
    fn a_stability_answer_becomes_one_billed_b64_image() {
        let payload = json!({
            "seeds": [2_130_420_379_u64],
            "finish_reasons": [null],
            "images": ["aW1hZ2U="],
        });
        let (body, usage) = public_stability_images(
            payload,
            &admission(1),
            Some(&flat_units()),
            &stability_shape(),
        )
        .expect("valid answer");
        assert_eq!(body["data"], json!([{"b64_json": "aW1hZ2U="}]));
        assert_eq!(body["output_format"], "jpeg");
        assert!(body["created"].as_u64().is_some_and(|created| created > 0));
        assert!(
            body.get("seeds").is_none(),
            "provider-only fields stay private"
        );
        let billed = usage.billed_units.expect("image units");
        assert_eq!(billed.kind, "image");
        assert_eq!(billed.variant, "");
        assert_eq!(billed.quantity_milli, 1_000);
        assert_eq!(usage.input_tokens, None);
    }

    #[test]
    fn a_stability_content_filter_is_a_refusal_naming_only_its_subject() {
        for (reason, subject) in [
            ("Filter reason: prompt", "prompt"),
            ("Filter reason: output image", "output image"),
            ("Filter reason: input image", "input image"),
            ("Filter reason: something new", "content"),
        ] {
            let payload = json!({"finish_reasons": [reason]});
            let failure = public_stability_images(
                payload,
                &admission(1),
                Some(&flat_units()),
                &stability_shape(),
            )
            .expect_err("filtered");
            assert_eq!(failure.failure_class, FailureClass::Refusal);
            assert!(failure.safe_message.ends_with(&format!("({subject})")));
            assert!(!failure.retryable_same_deployment);
            assert!(!failure.failover_eligible);
        }
    }

    #[test]
    fn a_stability_inference_error_is_a_retryable_provider_failure() {
        let payload = json!({"finish_reasons": ["Inference error"], "images": []});
        let failure = public_stability_images(
            payload,
            &admission(1),
            Some(&flat_units()),
            &stability_shape(),
        )
        .expect_err("inference error");
        assert_eq!(failure.failure_class, FailureClass::ProviderInternal);
        assert!(failure.retryable_same_deployment);
        assert!(failure.failover_eligible);
    }

    #[test]
    fn a_stability_answer_without_its_image_or_price_is_malformed() {
        for payload in [
            json!({"finish_reasons": [null]}),
            json!({"finish_reasons": [null], "images": []}),
            json!({"finish_reasons": [null], "images": [""]}),
            json!({"finish_reasons": [null], "images": [7]}),
            json!(["not", "an", "object"]),
        ] {
            let failure = public_stability_images(
                payload,
                &admission(1),
                Some(&flat_units()),
                &stability_shape(),
            )
            .expect_err("malformed");
            assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
        }
        let unpriced = public_stability_images(
            json!({"finish_reasons": [null], "images": ["aW1hZ2U="]}),
            &admission(1),
            None,
            &stability_shape(),
        )
        .expect_err("no per-image price");
        assert_eq!(unpriced.failure_class, FailureClass::MalformedResponse);
    }

    #[test]
    fn admission_wire_shapes_default_to_the_openai_answer() {
        let without: ImagesAdmission = serde_json::from_value(json!({
            "request_id": "r", "alias": "a", "alias_revision_id": "v",
            "exact_model_id": "m", "route_reason": "direct", "route": [],
            "image_count": 1, "maximum_total_attempts": 1,
            "maximum_same_deployment_attempts": 1,
        }))
        .expect("older admission shape");
        assert!(without.wire_shapes.is_empty());
        let with: ImagesAdmission = serde_json::from_value(json!({
            "request_id": "r", "alias": "a", "alias_revision_id": "v",
            "exact_model_id": "m", "route_reason": "direct", "route": [],
            "image_count": 1, "maximum_total_attempts": 1,
            "maximum_same_deployment_attempts": 1,
            "wire_shapes": [{"shape": "stability", "output_format": "png"}, {"shape": "openai"}],
        }))
        .expect("shaped admission");
        assert!(with.wire_shapes[0].stability());
        assert!(!with.wire_shapes[1].stability());
        assert_eq!(with.wire_shapes[1].output_format, None);
    }
}
