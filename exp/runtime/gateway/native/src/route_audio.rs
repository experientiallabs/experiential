//! The native audio surfaces: `POST /v1/audio/speech` and
//! `POST /v1/audio/transcriptions`.
//!
//! Both are buffered, never streamed to the caller, and share one failover
//! ladder. Admission returns each certified rung's OpenAI-wire payload and how
//! the rung bills (`units` from its unit card, `tokens` from provider usage):
//!
//! - Speech posts the JSON payload. A `units` rung returns the audio file and
//!   bills the input characters the gateway counted; a `tokens` rung was asked
//!   for the provider's SSE format, whose `speech.audio.delta` events carry the
//!   audio in base64 and whose `speech.audio.done` event reports the usage, so
//!   the audio is reassembled here and the usage billed.
//! - Transcription parses the caller's upload (multipart `file`, or JSON
//!   `input_audio` base64) and keeps the audio bytes on this side of the
//!   bridge: admission sees only the upload's size, name, type, SHA-256, and
//!   the duration measured from its demuxed packets (an unmeasurable upload is
//!   refused). Each attempt re-encodes the audio as a
//!   multipart upload beside the rung's text fields. A `units` rung bills the
//!   provider's metered seconds (`usage.seconds`, or `verbose_json`'s
//!   `duration`); a `tokens` rung bills `usage.input_tokens`/`output_tokens`.
//!
//! An answer without the meter its rung bills on is refused as malformed and
//! fails over, so no audio or transcript is handed out unaccounted. An inbound
//! `Idempotency-Key` is ignored: neither surface has a replay protocol.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::Response;
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokio::sync::OwnedSemaphorePermit;

use crate::admission::{new_guard, wire_drift_response};
use crate::audio_upload::{measure_upload, read_upload, AudioUpload};
use crate::dialects::{Dialect, MAXIMUM_RETAINED_OUTPUT_BYTES, OUTPUT_OVERFLOW_MESSAGE};
use crate::encode::compact_json;
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{BilledUnits, Usage, MAXIMUM_LEDGER_COUNT};
use crate::metrics::{classify_escalation, METRICS};
use crate::rate_limit_headers::harvest_rate_limit_headers;
use crate::relay::{collection_public_error, remaining};
use crate::respond::{
    bearer_key, error_response, escalation_error, latin1_header, with_app_identity,
};
use crate::server::AppState;
use crate::settlement::AttemptGuard;
use crate::upstream::{open_stream, open_upload};
use crate::waterfall::{successor_possible, DeploymentWire, RoutePolicy, StartResponse};

/// The wire configuration returned by one successful audio admission.
#[derive(Debug, Clone, Deserialize)]
struct AudioAdmission {
    request_id: String,
    alias: String,
    alias_revision_id: String,
    exact_model_id: String,
    route_reason: String,
    route: Vec<DeploymentWire>,
    /// Per-rung billing mode, aligned with `route`: `units` or `tokens`.
    billing_modes: Vec<String>,
    /// Speech only: the characters a `units` rung bills.
    #[serde(default)]
    input_characters: Option<u64>,
    /// Transcription only: the caller's response format (`json`, `verbose_json`, `text`).
    #[serde(default)]
    response_format: Option<String>,
    /// Transcription only: the most milliseconds the reservation covers; a
    /// provider meter above it is inconsistent with the admitted upload.
    #[serde(default)]
    maximum_audio_milli: Option<u64>,
    /// Per-rung `[input, output]` token ceilings the reservation covers (token
    /// rungs only); a provider usage above either is inconsistent with the hold.
    #[serde(default)]
    token_ceilings: Vec<Option<[u64; 2]>>,
    maximum_total_attempts: u32,
    maximum_same_deployment_attempts: u32,
}

impl AudioAdmission {
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

    fn tokens_billed(&self, depth: usize) -> bool {
        self.billing_modes.get(depth).map(String::as_str) == Some("tokens")
    }
}

/// What one attempt dispatches.
enum Job<'a> {
    Speech,
    Transcription(&'a AudioUpload),
}

/// One validated provider answer: the public body, its content type, and the meter.
struct Served {
    depth: usize,
    body: Vec<u8>,
    content_type: String,
    usage: Usage,
}

/// `POST /v1/audio/speech`.
pub(crate) async fn speech(
    State(state): State<AppState>,
    request: axum::extract::Request,
) -> Response {
    state.handled_requests.fetch_add(1, Ordering::Relaxed);
    let started = Instant::now();
    let (parts, raw_body) = request.into_parts();
    let headers = parts.headers;
    let raw_key = match authenticate(&state, &headers).await {
        Ok(key) => key,
        Err(error) => return error_response(&error),
    };
    // Like transcription, the body is read only under a permit and the
    // request deadline, so the concurrency limit bounds every buffered body.
    let deadline = started + state.request_timeout;
    let permit = match pre_admission_permit(&state, deadline).await {
        Ok(permit) => permit,
        Err(error) => return error_response(&error),
    };
    let read = axum::body::to_bytes(raw_body, MAXIMUM_SPEECH_BODY_BYTES);
    let body = match tokio::time::timeout_at(deadline.into(), read).await {
        Ok(Ok(body)) => body,
        Ok(Err(_)) => return error_response(&PublicError::request_too_large()),
        Err(_) => return error_response(&upload_deadline_error()),
    };
    let body_text = match String::from_utf8(body.to_vec()) {
        Ok(text) => text,
        Err(_) => return error_response(&PublicError::invalid_json()),
    };
    serve(
        &state,
        &headers,
        &raw_key,
        started,
        "admit_speech",
        "body",
        Value::String(body_text),
        Job::Speech,
        permit,
    )
    .await
}

/// `POST /v1/audio/transcriptions`.
pub(crate) async fn transcriptions(
    State(state): State<AppState>,
    request: axum::extract::Request,
) -> Response {
    state.handled_requests.fetch_add(1, Ordering::Relaxed);
    let started = Instant::now();
    let headers = request.headers().clone();
    // The key is checked before the upload is read, so an unknown key gets the
    // uniform 401 whatever its body, and no unauthenticated upload is parsed.
    let raw_key = match authenticate(&state, &headers).await {
        Ok(key) => key,
        Err(error) => return error_response(&error),
    };
    // The upload (up to 25 MB, possibly sent slowly) is read and measured only
    // under a concurrency permit and the request deadline, so the configured
    // request limit bounds this route's buffered memory like every other route.
    let deadline = started + state.request_timeout;
    let permit = match pre_admission_permit(&state, deadline).await {
        Ok(permit) => permit,
        Err(error) => return error_response(&error),
    };
    let read = tokio::time::timeout_at(deadline.into(), read_upload(&state, request)).await;
    let (fields, upload) = match read {
        Ok(Ok(read)) => read,
        Ok(Err(error)) => return error_response(&error),
        Err(_) => return error_response(&upload_deadline_error()),
    };
    // Measured under the still-held permit, never under a cancelling timeout.
    let (fields, upload, permit) = match measure_upload(fields, upload, permit).await {
        Ok(measured) => measured,
        Err(error) => return error_response(&error),
    };
    serve(
        &state,
        &headers,
        &raw_key,
        started,
        "admit_transcription",
        "upload",
        Value::String(compact_json(&Value::Object(fields))),
        Job::Transcription(&upload),
        permit,
    )
    .await
}

/// Largest speech request body read: its two long text fields (`input` and
/// `instructions`, 4,096 characters each) fully `\uXXXX`-escaped, plus room
/// for the short fields, so an oversized body never reaches the bridge.
const MAXIMUM_SPEECH_BODY_BYTES: usize = 128 * 1024;

/// Take one data-plane permit before the body is read (no admission or attempt
/// guard exists yet), under the same request deadline.
async fn pre_admission_permit(
    state: &AppState,
    deadline: Instant,
) -> Result<OwnedSemaphorePermit, PublicError> {
    match tokio::time::timeout_at(deadline.into(), state.permits.clone().acquire_owned()).await {
        Ok(Ok(permit)) => Ok(permit),
        Ok(Err(_)) => Err(PublicError::draining()),
        Err(_) => Err(Failure::new(
            FailureClass::Timeout,
            "gateway execution queue deadline exceeded",
        )
        .public_error()),
    }
}

fn upload_deadline_error() -> PublicError {
    PublicError::new(
        408,
        "request_timeout",
        "The audio upload did not arrive before the request deadline.",
        "invalid_request_error",
    )
}

/// Authenticate, admit, walk the ladder, settle, and answer one audio request.
#[allow(clippy::too_many_arguments)]
async fn serve(
    state: &AppState,
    headers: &HeaderMap,
    raw_key: &str,
    started: Instant,
    admit_method: &'static str,
    body_key: &str,
    body_value: Value,
    job: Job<'_>,
    permit: OwnedSemaphorePermit,
) -> Response {
    let deadline = started + state.request_timeout;
    if Instant::now() >= deadline {
        // A slow upload or duration probe spent the budget: admit nothing, so
        // no attempt is opened that the ladder could never run.
        return error_response(&upload_deadline_error());
    }
    let client_request_id = latin1_header(headers, "x-client-request-id");
    let remaining_milli = u64::try_from(remaining(deadline).as_millis()).unwrap_or(u64::MAX);
    let mut admit_value = json!({"raw_key": raw_key, "remaining_milli": remaining_milli});
    admit_value[body_key] = body_value;
    with_app_identity(&mut admit_value, headers);
    let admission_text = match state
        .bridge
        .call(admit_method, compact_json(&admit_value))
        .await
    {
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
    let admission = match serde_json::from_value::<AudioAdmission>(admission_value.clone()) {
        Ok(admission) if admission.billing_modes.len() == admission.route.len() => admission,
        _ => return wire_drift_response(state, &admission_value, started).await,
    };
    let mut guard = new_guard(state, admission.request_id.clone(), started);
    let _permit = permit;
    match run_ladder(state, &admission, raw_key, &mut guard, deadline, &job).await {
        Err(error) => error_response(&error),
        Ok(served) => {
            if !guard
                .settle("completed", Some(&served.usage), &[], None, true)
                .await
            {
                // Never hand out audio or a transcript the ledger did not record.
                return error_response(&PublicError::internal());
            }
            let mut response_headers = served_headers(&admission, served.depth, client_request_id);
            response_headers.push((header::CONTENT_TYPE.to_string(), served.content_type));
            binary_response(served.body, &response_headers)
        }
    }
}

/// Check the caller's bearer key against the control plane before any body is read.
async fn authenticate(state: &AppState, headers: &HeaderMap) -> Result<String, PublicError> {
    let raw_key = bearer_key(headers)?;
    state
        .bridge
        .call("authenticate", compact_json(&json!({"raw_key": raw_key})))
        .await?;
    Ok(raw_key)
}

fn binary_response(body: Vec<u8>, headers: &[(String, String)]) -> Response {
    let mut builder = Response::builder().status(StatusCode::OK);
    for (name, value) in headers {
        builder = builder.header(name, value);
    }
    builder
        .body(Body::from(body))
        .unwrap_or_else(|_| error_response(&PublicError::internal()))
}

/// Walk the certified ladder to one validated provider answer or the public
/// error of the exhausting failure (same contract as the images ladder).
async fn run_ladder(
    state: &AppState,
    admission: &AudioAdmission,
    raw_key: &str,
    guard: &mut AttemptGuard,
    deadline: Instant,
    job: &Job<'_>,
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
        let outcome = match job {
            Job::Speech => dispatch_speech(state, wire, deadline, admission, depth, guard).await,
            Job::Transcription(upload) => {
                dispatch_transcription(state, wire, deadline, admission, depth, upload, guard).await
            }
        };
        match outcome {
            Ok((body, content_type, usage)) => {
                return Ok(Served {
                    depth,
                    body,
                    content_type,
                    usage,
                })
            }
            Err((mut failure, opened)) => {
                // A customer's own credential or account failing is theirs to
                // fix: classify it as the chat waterfall does.
                if wire.billing_customer_managed {
                    failure =
                        crate::stream_errors::customer_credential_failure(failure, &wire.provider);
                }
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

type Dispatched = Result<(Vec<u8>, String, Usage), (Failure, bool)>;

async fn dispatch_speech(
    state: &AppState,
    wire: &DeploymentWire,
    deadline: Instant,
    admission: &AudioAdmission,
    depth: usize,
    guard: &mut AttemptGuard,
) -> Dispatched {
    let phase_timeout = Duration::from_secs_f64(wire.timeout_seconds.max(0.001));
    let bound = remaining(deadline).min(phase_timeout);
    // From here the provider may execute and bill even if no header arrives.
    guard.mark_dispatched();
    let response = open_stream(
        &state.http,
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
    // Opened the moment headers arrive, as on the chat ladder: settlement
    // records the open and the provider's rate-limit observations.
    guard.mark_opened();
    guard.record_rate_limit_headers(harvest_rate_limit_headers(response.headers()));
    let upstream_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let bytes = read_bounded_body(response, deadline, phase_timeout)
        .await
        .map_err(|f| (f, true))?;
    let format = wire
        .upstream_payload
        .get("response_format")
        .and_then(Value::as_str)
        .unwrap_or("mp3");
    if admission.tokens_billed(depth) {
        let (audio, usage) = reassemble_speech_events(&bytes).map_err(|f| (f, true))?;
        let ceilings = admission.token_ceilings.get(depth).copied().flatten();
        let usage = within_ceilings(usage, ceilings, "speech").map_err(|f| (f, true))?;
        return Ok((audio, speech_content_type(format).to_string(), usage));
    }
    if bytes.is_empty() {
        return Err((malformed("speech response carried no audio"), true));
    }
    // Anything a provider names that is not audio (an error document behind a
    // 200 from a misbehaving proxy) must never be billed or handed out as audio.
    if upstream_type
        .as_deref()
        .is_some_and(|value| !is_audio_type(value))
    {
        return Err((malformed("speech response was not audio"), true));
    }
    let characters = admission.input_characters.unwrap_or(0);
    let usage = Usage {
        billed_units: Some(BilledUnits {
            kind: "character".to_string(),
            variant: String::new(),
            quantity_milli: characters.saturating_mul(1000),
        }),
        ..Usage::default()
    };
    let content_type = upstream_type.unwrap_or_else(|| speech_content_type(format).to_string());
    Ok((bytes, content_type, usage))
}

/// Whether a named content type can carry speech audio: any `audio/*`, the
/// Ogg container, or an untyped binary stream.
fn is_audio_type(value: &str) -> bool {
    let essence = value.split(';').next().unwrap_or_default().trim();
    let essence = essence.to_ascii_lowercase();
    essence.starts_with("audio/")
        || essence == "application/ogg"
        || essence == "application/octet-stream"
}

/// The content type of one requested speech format (OpenAI's own mapping).
fn speech_content_type(format: &str) -> &'static str {
    match format {
        "opus" => "audio/ogg",
        "aac" => "audio/aac",
        "flac" => "audio/flac",
        "wav" => "audio/wav",
        "pcm" => "audio/pcm",
        _ => "audio/mpeg",
    }
}

/// Reassemble one SSE speech answer: concatenate every `speech.audio.delta`
/// event's base64 audio and read the `speech.audio.done` event's token usage.
fn reassemble_speech_events(body: &[u8]) -> Result<(Vec<u8>, Usage), Failure> {
    let text =
        std::str::from_utf8(body).map_err(|_| malformed("speech event stream is not text"))?;
    let mut audio: Vec<u8> = Vec::new();
    let mut usage: Option<Usage> = None;
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let event: Value =
            serde_json::from_str(data).map_err(|_| malformed("speech event is not JSON"))?;
        match event.get("type").and_then(Value::as_str) {
            Some("speech.audio.delta") => {
                let chunk = event
                    .get("audio")
                    .and_then(Value::as_str)
                    .ok_or_else(|| malformed("speech audio delta carried no audio"))?;
                let decoded = base64::engine::general_purpose::STANDARD
                    .decode(chunk)
                    .map_err(|_| malformed("speech audio delta is not base64"))?;
                if audio.len() + decoded.len() > MAXIMUM_RETAINED_OUTPUT_BYTES {
                    return Err(Failure::new(
                        FailureClass::MalformedResponse,
                        OUTPUT_OVERFLOW_MESSAGE,
                    ));
                }
                audio.extend_from_slice(&decoded);
            }
            Some("speech.audio.done") => {
                let meter = event
                    .get("usage")
                    .and_then(Value::as_object)
                    .ok_or_else(|| malformed("speech stream omitted its token usage"))?;
                usage = Some(token_usage(meter, "speech")?);
            }
            _ => {}
        }
    }
    let usage = usage.ok_or_else(|| malformed("speech stream ended without its usage event"))?;
    if audio.is_empty() {
        return Err(malformed("speech stream carried no audio"));
    }
    Ok((audio, usage))
}

/// Refuse a token usage above the rung's reserved ceilings as malformed: the
/// settle must stay inside the hold the reservation took.
fn within_ceilings(
    usage: Usage,
    ceilings: Option<[u64; 2]>,
    surface: &str,
) -> Result<Usage, Failure> {
    let Some([input, output]) = ceilings else {
        return Ok(usage);
    };
    if usage.input_tokens.unwrap_or(0) > input || usage.output_tokens.unwrap_or(0) > output {
        return Err(malformed(&format!(
            "{surface} usage exceeds the reserved token ceiling"
        )));
    }
    Ok(usage)
}

fn token_usage(meter: &Map<String, Value>, surface: &str) -> Result<Usage, Failure> {
    // A count the ledger cannot store is as malformed as a missing one.
    let count = |key: &str| {
        meter
            .get(key)
            .and_then(Value::as_u64)
            .filter(|count| *count <= MAXIMUM_LEDGER_COUNT)
    };
    let (input, output) = (count("input_tokens"), count("output_tokens"));
    match (input, output) {
        (Some(input), Some(output)) => Ok(Usage {
            input_tokens: Some(input),
            output_tokens: Some(output),
            ..Usage::default()
        }),
        _ => Err(malformed(&format!(
            "{surface} usage omitted its token counts"
        ))),
    }
}

async fn dispatch_transcription(
    state: &AppState,
    wire: &DeploymentWire,
    deadline: Instant,
    admission: &AudioAdmission,
    depth: usize,
    upload: &AudioUpload,
    guard: &mut AttemptGuard,
) -> Dispatched {
    let phase_timeout = Duration::from_secs_f64(wire.timeout_seconds.max(0.001));
    let bound = remaining(deadline).min(phase_timeout);
    let boundary = format!(
        "exp-gateway-{}",
        admission
            .request_id
            .replace(|c: char| !c.is_ascii_alphanumeric(), "")
    );
    let body = encode_multipart(&boundary, &wire.upstream_payload, upload);
    guard.mark_dispatched();
    let response = open_upload(
        &state.http,
        &wire.url,
        &wire.headers,
        &wire.idempotency_key,
        body,
        &format!("multipart/form-data; boundary={boundary}"),
        &wire.upstream_payload,
        bound,
    )
    .await
    .map_err(|failure| (failure, false))?;
    // Opened the moment headers arrive, as on the chat ladder: settlement
    // records the open and the provider's rate-limit observations.
    guard.mark_opened();
    guard.record_rate_limit_headers(harvest_rate_limit_headers(response.headers()));
    let bytes = read_bounded_body(response, deadline, phase_timeout)
        .await
        .map_err(|f| (f, true))?;
    let payload: Value = serde_json::from_slice(&bytes)
        .map_err(|_| (malformed("transcription response is not JSON"), true))?;
    public_transcription(payload, admission, depth).map_err(|failure| (failure, true))
}

/// Validate one provider transcription answer and render the caller's format.
fn public_transcription(
    payload: Value,
    admission: &AudioAdmission,
    depth: usize,
) -> Result<(Vec<u8>, String, Usage), Failure> {
    let object = payload
        .as_object()
        .ok_or_else(|| malformed("transcription response is not an object"))?;
    let text = object
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| malformed("transcription response omitted its text"))?;
    let meter = object.get("usage").and_then(Value::as_object);
    let usage = if admission.tokens_billed(depth) {
        within_ceilings(
            token_usage(
                meter.ok_or_else(|| malformed("transcription response omitted its usage"))?,
                "transcription",
            )?,
            admission.token_ceilings.get(depth).copied().flatten(),
            "transcription",
        )?
    } else {
        let seconds = meter
            .filter(|meter| meter.get("type").and_then(Value::as_str) == Some("duration"))
            .and_then(|meter| meter.get("seconds"))
            .or_else(|| object.get("duration"))
            .and_then(Value::as_f64)
            .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
            .ok_or_else(|| malformed("transcription response omitted its metered duration"))?;
        let quantity_milli = (seconds * 1000.0).ceil() as u64;
        // A meter longer than the measured upload the reservation covered is
        // inconsistent with the audio sent; refusing it keeps the settle
        // inside the hold (and the ledger's integer range).
        if admission
            .maximum_audio_milli
            .is_none_or(|maximum| quantity_milli > maximum)
        {
            return Err(malformed(
                "transcription meter exceeds the admitted audio duration",
            ));
        }
        Usage {
            billed_units: Some(BilledUnits {
                kind: "audio_second".to_string(),
                variant: String::new(),
                quantity_milli,
            }),
            ..Usage::default()
        }
    };
    if admission.response_format.as_deref() == Some("text") {
        return Ok((
            text.as_bytes().to_vec(),
            "text/plain; charset=utf-8".to_string(),
            usage,
        ));
    }
    let body = serde_json::to_vec(&payload)
        .map_err(|_| malformed("transcription response is not JSON"))?;
    Ok((body, "application/json".to_string(), usage))
}

/// Encode one multipart upload: every text field (a list is a repeated part
/// under its own name) and the audio as the `file` part.
fn encode_multipart(boundary: &str, fields: &Value, upload: &AudioUpload) -> Vec<u8> {
    let mut body: Vec<u8> = Vec::with_capacity(upload.bytes.len() + 1024);
    let mut text_part = |name: &str, value: &str| {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"{}\"\r\n\r\n",
                escape_quoted(name)
            )
            .as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    };
    if let Some(object) = fields.as_object() {
        for (name, value) in object {
            match value {
                Value::String(text) => text_part(name, text),
                Value::Array(items) => {
                    for item in items.iter().filter_map(Value::as_str) {
                        text_part(name, item);
                    }
                }
                Value::Number(number) => text_part(name, &number.to_string()),
                _ => {}
            }
        }
    }
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"file\"; filename=\"{}\"\r\nContent-Type: {}\r\n\r\n",
            escape_quoted(&upload.filename),
            upload.content_type
        )
        .as_bytes(),
    );
    body.extend_from_slice(&upload.bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    body
}

fn escape_quoted(value: &str) -> String {
    value
        .chars()
        .filter(|c| !matches!(c, '"' | '\r' | '\n'))
        .collect()
}

/// Read one provider body chunk by chunk under the retained-output cap.
async fn read_bounded_body(
    mut response: reqwest::Response,
    deadline: Instant,
    phase_timeout: Duration,
) -> Result<Vec<u8>, Failure> {
    if response
        .content_length()
        .is_some_and(|length| length > MAXIMUM_RETAINED_OUTPUT_BYTES as u64)
    {
        return Err(Failure::new(
            FailureClass::MalformedResponse,
            OUTPUT_OVERFLOW_MESSAGE,
        ));
    }
    let mut body: Vec<u8> = Vec::new();
    loop {
        let bound = remaining(deadline).min(phase_timeout);
        let chunk = match tokio::time::timeout(bound, response.chunk()).await {
            Ok(Ok(Some(chunk))) => chunk,
            Ok(Ok(None)) => return Ok(body),
            Ok(Err(_)) => {
                return Err(Failure::new(
                    FailureClass::Transport,
                    "provider connection failed while sending the audio response",
                )
                .with_retry(true, true))
            }
            Err(_) => {
                return Err(Failure::new(
                    FailureClass::Timeout,
                    "provider did not finish the audio response in time",
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

fn served_headers(
    admission: &AudioAdmission,
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
#[path = "route_audio_tests.rs"]
mod tests;
