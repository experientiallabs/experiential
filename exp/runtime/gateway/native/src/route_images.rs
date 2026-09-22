//! The native image-generation surface: the `POST /v1/images/generations`
//! handler.
//!
//! Prompt in, images out, never streamed: admission returns one fully built
//! OpenAI-wire `/images/generations` payload per certified deployment and the
//! ladder below reserves each attempt through `start_attempt`, POSTs the
//! payload, buffers the JSON answer under the retained-output cap, validates
//! it against the request (up to `n` images for OpenRouter, exactly `n` on
//! other wires, a reported token
//! usage), and settles the winning attempt with the provider's prompt and
//! image token counts. A provider that answers without usage (the per-image
//! priced dall-e models) is refused as an unbillable answer until the typed
//! billed-units ledger lands, so no image is ever handed out unaccounted.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Response;
use base64::Engine;
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
    bearer_key, error_response as gateway_error_response, escalation_error, json_response,
    latin1_header, read_body,
};
use crate::server::AppState;
use crate::settlement::AttemptGuard;
use crate::upstream::open_stream;
use crate::waterfall::{DeploymentWire, StartResponse};

/// The wire configuration returned by one successful images admission.
#[derive(Debug, Clone, Deserialize)]
struct ImagesAdmission {
    request_id: String,
    alias: String,
    alias_revision_id: String,
    exact_model_id: String,
    route_reason: String,
    route: Vec<DeploymentWire>,
    /// The requested `n`: OpenRouter may return fewer, other wires require exactly n.
    image_count: usize,
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
    let client_request_id = latin1_header(&headers, "x-client-request-id");

    let admit_argument = compact_json(&json!({"raw_key": raw_key, "body": body_text}));
    let admission_text = match state.bridge.call("admit_images", admit_argument).await {
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
    let admission: ImagesAdmission = match serde_json::from_value(admission_value.clone()) {
        Ok(admission) => admission,
        Err(_) => return wire_drift_response(&state, &admission_value, started).await,
    };
    let mut guard = new_guard(&state, admission.request_id.clone(), started);
    let permit = match acquire_permit(&state, &mut guard, deadline).await {
        Ok(permit) => permit,
        Err(response) => return *response,
    };
    let _permit = permit;

    match run_once(&state, &admission, &raw_key, &mut guard, deadline).await {
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

/// Suppress automatic SDK retries: image requests have no keyed replay contract.
fn error_response(error: &PublicError) -> Response {
    let mut response = gateway_error_response(error);
    response.headers_mut().insert(
        "x-should-retry",
        axum::http::HeaderValue::from_static("false"),
    );
    response
}

/// Dispatch the admitted candidate once. No failure can dispatch another generation.
async fn run_once(
    state: &AppState,
    admission: &ImagesAdmission,
    raw_key: &str,
    guard: &mut AttemptGuard,
    deadline: Instant,
) -> Result<Served, PublicError> {
    let argument = compact_json(&json!({
        "request_id": admission.request_id,
        "raw_key": raw_key,
        "attempt_ordinal": 0,
        "current_depth": null,
        "failure": null,
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
        let failure = started.failure.unwrap_or_else(|| {
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
    guard.rebind(attempt_id);
    match dispatch(&state.http, wire, deadline, admission).await {
        Ok((body, usage)) => Ok(Served { depth, body, usage }),
        Err((failure, opened, usage)) => {
            // No provider idempotency contract protects image generation.
            // Even a header timeout or 5xx can follow completed billable work.
            let failure = failure.with_retry(false, false);
            if opened {
                guard.mark_opened();
            }
            let boundary = failure.clone().boundary();
            if !guard
                .settle("failed", usage.as_deref(), &[], Some(&boundary), true)
                .await
            {
                return Err(PublicError::internal());
            }
            Err(collection_public_error(&boundary))
        }
    }
}

/// POST one admitted payload and validate the buffered answer; the error
/// carries whether the provider dispatch opened.
async fn dispatch(
    http: &reqwest::Client,
    wire: &DeploymentWire,
    deadline: Instant,
    admission: &ImagesAdmission,
) -> Result<(Value, Usage), (Failure, bool, Option<Box<Usage>>)> {
    let phase_timeout = Duration::from_secs_f64(wire.timeout_seconds.max(0.001));
    let bound = remaining(deadline).min(phase_timeout);
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
    .map_err(|failure| (failure, false, None))?;
    if response
        .content_length()
        .is_some_and(|length| length > MAXIMUM_RETAINED_OUTPUT_BYTES as u64)
    {
        return Err((
            Failure::new(FailureClass::MalformedResponse, OUTPUT_OVERFLOW_MESSAGE),
            true,
            None,
        ));
    }
    let bytes = read_bounded_body(response, deadline, phase_timeout)
        .await
        .map_err(|failure| (failure, true, None))?;
    let payload: Value = match serde_json::from_slice(&bytes) {
        Ok(payload) => payload,
        Err(_) => return Err((malformed("images response is not JSON"), true, None)),
    };
    let openrouter = wire.provider == "openrouter";
    let observed = image_usage(&payload, openrouter).ok().map(Box::new);
    public_images(payload, admission, openrouter).map_err(|failure| (failure, true, observed))
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
/// request and rebuild it as the public answer: up to `n` images on OpenRouter,
/// exactly `n` on other wires, each a
/// `b64_json` or `url` string (passed through untouched, with any
/// `revised_prompt`), and a billable token usage. The public body keeps the
/// provider's `created`, `usage`, and the echoed rendering facts.
fn public_images(
    payload: Value,
    admission: &ImagesAdmission,
    openrouter: bool,
) -> Result<(Value, Usage), Failure> {
    let object = payload
        .as_object()
        .ok_or_else(|| malformed("images response is not an object"))?;
    let data = object
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("images response omitted the data array"))?;
    if data.is_empty()
        || data.len() > admission.image_count
        || (!openrouter && data.len() != admission.image_count)
    {
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
        for key in ["b64_json", "url", "revised_prompt", "media_type"] {
            if let Some(value) = entry.get(key).filter(|value| value.is_string()) {
                let text = value.as_str().unwrap_or_default();
                if key == "b64_json" {
                    let decoded = base64::engine::general_purpose::STANDARD
                        .decode(text)
                        .map_err(|_| malformed("images response contains invalid base64"))?;
                    if decoded.is_empty() {
                        return Err(malformed("images response contains an empty image"));
                    }
                    carried = true;
                }
                if key == "url" {
                    if openrouter || !(text.starts_with("https://") || text.starts_with("http://"))
                    {
                        return Err(malformed("images response contains an unusable image URL"));
                    }
                    carried = true;
                }
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
    // The surface bills on these counts: an answer without token usage (the
    // per-image priced models) is unbillable here and is refused, never a zero.
    let usage = object
        .get("usage")
        .and_then(Value::as_object)
        .ok_or_else(|| malformed("images response omitted its token usage"))?;
    let observed = image_usage(&payload, openrouter)?;
    let input_tokens = observed.input_tokens.unwrap_or_default();
    let output_tokens = observed.output_tokens.unwrap_or_default();
    let mut public = Map::new();
    public.insert(
        "created".to_string(),
        object.get("created").cloned().unwrap_or(Value::from(0)),
    );
    public.insert("data".to_string(), Value::Array(images));
    for key in ["background", "output_format", "quality", "size"] {
        if let Some(value) = object.get(key) {
            public.insert(key.to_string(), value.clone());
        }
    }
    let mut public_usage = usage.clone();
    if openrouter {
        public_usage.remove("prompt_tokens");
        public_usage.remove("completion_tokens");
        public_usage.insert("input_tokens".to_string(), json!(input_tokens));
        public_usage.insert("output_tokens".to_string(), json!(output_tokens));
    }
    public.insert("usage".to_string(), Value::Object(public_usage));
    Ok((Value::Object(public), observed))
}

/// Preserve observed counts even when an image body fails validation.
fn image_usage(payload: &Value, openrouter: bool) -> Result<Usage, Failure> {
    let usage = payload
        .get("usage")
        .and_then(Value::as_object)
        .ok_or_else(|| malformed("images response omitted its token usage"))?;
    let input_tokens = usage
        .get(if openrouter {
            "prompt_tokens"
        } else {
            "input_tokens"
        })
        .and_then(Value::as_u64)
        .ok_or_else(|| malformed("images response omitted input token usage"))?;
    let output_tokens = usage
        .get(if openrouter {
            "completion_tokens"
        } else {
            "output_tokens"
        })
        .and_then(Value::as_u64)
        .ok_or_else(|| malformed("images response omitted output token usage"))?;
    Ok(Usage {
        input_tokens: Some(input_tokens),
        output_tokens: Some(output_tokens),
        cached_input_tokens: None,
        cache_creation_input_tokens: None,
        cache_creation_1h_input_tokens: None,
        reasoning_tokens: None,
    })
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
            public_images(payload.clone(), &admission(2), false).expect("valid answer");
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
            let failure = public_images(payload, &admission(2), false).expect_err("malformed");
            assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
            assert!(failure.failover_eligible);
        }
    }
    #[test]
    fn openrouter_usage_and_partial_count_are_mapped_without_losing_media() {
        let payload = json!({
            "created": 17,
            "data": [{"b64_json": "aW1hZ2U=", "media_type": "image/png"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 272, "total_tokens": 275, "cost": 0.008184}
        });
        let (body, usage) = public_images(payload, &admission(2), true).unwrap();
        assert_eq!(body["data"][0]["media_type"], "image/png");
        assert_eq!(body["usage"]["input_tokens"], 3);
        assert_eq!(body["usage"]["output_tokens"], 272);
        assert_eq!(body["usage"]["cost"], 0.008184);
        assert_eq!(usage.input_tokens, Some(3));
        assert_eq!(usage.output_tokens, Some(272));
    }

    #[test]
    fn invalid_base64_and_sandbox_references_are_not_images() {
        for item in [
            json!({"b64_json":""}),
            json!({"b64_json":"!!!"}),
            json!({"url":"sandbox:/mnt/data/0.png"}),
        ] {
            let payload = json!({"data":[item], "usage":{"prompt_tokens":1,"completion_tokens":1}});
            assert!(public_images(payload, &admission(1), true).is_err());
        }
    }
}
