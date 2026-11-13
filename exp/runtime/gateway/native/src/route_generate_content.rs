//! The native Gemini API surface: `POST /v1beta/models/{model}:generateContent`
//! and `:streamGenerateContent?alt=sse`, the `:countTokens` estimate, the
//! Google-shaped model listing, and Google's error envelope. Google defines no
//! idempotency header, so this surface never joins the keyed replay stores,
//! exactly like the Anthropic Messages surface.

use crate::guardrails::input::acquire_guarded_attempt;
use std::sync::atomic::Ordering;
use std::time::Instant;

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode, Uri};
use axum::response::Response;
use bytes::Bytes;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::admission::{
    acquire_permit_with, apply_output_guardrail, new_guard, served_headers, Admission,
};
use crate::capture::reasoning::{checkpoint_winner, observe_winner};
use crate::encode::compact_json;
use crate::encode_generate_content::{
    completed_generate_content_body, google_error_body, refusal_failure, AggregatedGenerateContent,
    GenerateContentEncoder,
};
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{Event, Usage};
use crate::guardrails::StreamGuardrails;
use crate::metrics::{classify_escalation, METRICS};
use crate::reasoning_display::ReasoningOutput;
use crate::relay::{collect_committed, collection_public_error, track_event};
use crate::respond::{
    bearer_key, client_ip, complete_visible_refusal, escalation_error, json_response,
    outward_event, price_stream_end, read_body, send_bounded, settle_stream_end, sse_body_response,
    with_app_identity,
};
use crate::respond::{log_stream_exit, stream_delivery::Delivery};
use crate::route_chat::{seal_reasoning_candidate, seal_reasoning_events};
use crate::server::AppState;
use crate::settlement::AttemptGuard;
use crate::tool_search::{adopt_outcome, disclose_after_collection};
use crate::waterfall::{
    billed_empty_completion, unreported_empty_completion, CommittedAttempt, Served, SettledAttempt,
    WaterfallContext, Won,
};

/// The generation methods every served model answers on this surface.
const GENERATION_METHODS: [&str; 3] = ["generateContent", "streamGenerateContent", "countTokens"];

/// Google-enveloped variant of `error_response` for the Gemini surface.
pub(crate) fn google_error_response(error: &PublicError) -> Response {
    let mut builder = Response::builder()
        .status(
            StatusCode::from_u16(error.status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        )
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(wait) = error.retry_after_seconds {
        builder = builder.header(header::RETRY_AFTER, wait.to_string());
    }
    if error.status_code == 429 && error.known_unbilled {
        builder = builder.header("x-gateway-admission-refused", "true");
    }
    builder
        .body(Body::from(compact_json(&google_error_body(error))))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

/// The `key` query parameter Google's REST convention carries, if any.
fn query_key(uri: &Uri) -> Option<String> {
    let query = uri.query()?;
    query.split('&').find_map(|pair| {
        let (name, value) = pair.split_once('=')?;
        (name == "key" && !value.is_empty()).then(|| {
            // The virtual key is URL-safe text; percent decoding covers an
            // encoded prefix underscore or padding a strict client adds.
            percent_decode(value)
        })
    })
}

/// Whether the request asked for Google's SSE streaming shape (`alt=sse`).
fn asks_for_sse(uri: &Uri) -> bool {
    uri.query()
        .is_some_and(|query| query.split('&').any(|pair| pair == "alt=sse"))
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(&value[index + 1..index + 3], 16) {
                decoded.push(byte);
                index += 3;
                continue;
            }
        }
        decoded.push(bytes[index]);
        index += 1;
    }
    String::from_utf8_lossy(&decoded).into_owned()
}

/// Google callers present `x-goog-api-key` (the official SDK default), the
/// `?key=` query parameter (the REST convention), or a standard Bearer
/// header; all three carry the same virtual key.
pub(crate) fn generate_content_api_key(
    headers: &HeaderMap,
    uri: &Uri,
) -> Result<String, PublicError> {
    if let Some(value) = headers
        .get("x-goog-api-key")
        .and_then(|value| value.to_str().ok())
    {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }
    if let Some(key) = query_key(uri) {
        return Ok(key);
    }
    bearer_key(headers).map_err(|_| {
        PublicError::new(
            401,
            "invalid_key",
            "A valid API key is required: send x-goog-api-key, ?key=, or Authorization: Bearer.",
            "authentication_error",
        )
    })
}

/// Split Google's `{model}:{action}` path segment; a segment with no action
/// is a model name alone (the detail read).
fn split_action(segment: &str) -> (String, Option<String>) {
    match segment.rsplit_once(':') {
        Some((model, action)) if !action.is_empty() && !model.is_empty() => {
            (model.to_string(), Some(action.to_string()))
        }
        _ => (segment.to_string(), None),
    }
}

/// Google's `Model` object for one granted alias, listing exactly the
/// methods this surface serves (the SDK reads `supportedGenerationMethods`).
fn google_model(alias: &str) -> Value {
    json!({
        "name": format!("models/{alias}"),
        "displayName": alias,
        "supportedGenerationMethods": GENERATION_METHODS,
    })
}

fn unknown_action(action: &str) -> PublicError {
    PublicError::new(
        404,
        "unknown_route",
        &format!(
            "Unknown model action ':{action}'. The gateway serves :generateContent, \
             :streamGenerateContent (with ?alt=sse) and :countTokens."
        ),
        "invalid_request_error",
    )
}

/// `GET /v1beta/models`: the granted aliases in Google's listing shape.
pub(crate) async fn generate_content_models(
    State(state): State<AppState>,
    request: axum::extract::Request,
) -> Response {
    state.handled_requests.fetch_add(1, Ordering::Relaxed);
    let raw_key = match generate_content_api_key(request.headers(), request.uri()) {
        Ok(key) => key,
        Err(error) => return google_error_response(&error),
    };
    let argument = compact_json(&json!({"raw_key": raw_key}));
    let listing = match state.bridge.call("models", argument).await {
        Ok(text) => text,
        Err(error) => return google_error_response(&error),
    };
    let payload: Value = match serde_json::from_str(&listing) {
        Ok(payload) => payload,
        Err(_) => return google_error_response(&PublicError::internal()),
    };
    let models: Vec<Value> = payload["data"]
        .as_array()
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| entry["id"].as_str())
                .map(google_model)
                .collect()
        })
        .unwrap_or_default();
    json_response(StatusCode::OK, &json!({"models": models}), &[])
}

/// `GET /v1beta/models/{model}` and `POST /v1beta/models/{model}:{action}`.
pub(crate) async fn generate_content_model_action(
    State(state): State<AppState>,
    Path(segment): Path<String>,
    request: axum::extract::Request,
) -> Response {
    if request.method() == axum::http::Method::GET {
        // GET has no action form, so a colon belongs to the alias itself
        // (`coding:free` is one granted spelling, not an action on `coding`).
        return model_detail(state, segment, request).await;
    }
    let (model, action) = split_action(&segment);
    match action {
        None => google_error_response(&PublicError::new(
            404,
            "unknown_route",
            "POST a model action: /v1beta/models/{model}:generateContent.",
            "invalid_request_error",
        )),
        Some(action) => match action.as_str() {
            "generateContent" => generate(state, model, request, false).await,
            "streamGenerateContent" => {
                if !asks_for_sse(request.uri()) {
                    return google_error_response(&PublicError::new(
                        400,
                        "unsupported_parameter",
                        "streamGenerateContent is served as server-sent events only: add ?alt=sse \
                         (the official SDK does), or call :generateContent for one JSON body.",
                        "invalid_request_error",
                    ));
                }
                generate(state, model, request, true).await
            }
            "countTokens" => count_tokens(state, model, request).await,
            other => google_error_response(&unknown_action(other)),
        },
    }
}

async fn model_detail(state: AppState, model: String, request: axum::extract::Request) -> Response {
    state.handled_requests.fetch_add(1, Ordering::Relaxed);
    let raw_key = match generate_content_api_key(request.headers(), request.uri()) {
        Ok(key) => key,
        Err(error) => return google_error_response(&error),
    };
    let argument = compact_json(&json!({"raw_key": raw_key, "model_id": model}));
    match state.bridge.call("model_detail", argument).await {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(payload) => match payload["id"].as_str() {
                Some(alias) => json_response(StatusCode::OK, &google_model(alias), &[]),
                None => google_error_response(&PublicError::internal()),
            },
            Err(_) => google_error_response(&PublicError::internal()),
        },
        Err(error) => google_error_response(&error),
    }
}

/// `:countTokens`: the gateway's own count in Google's `{"totalTokens": N}`
/// shape, authenticated and granted like a generation but never accepted,
/// reserved or charged.
async fn count_tokens(state: AppState, model: String, request: axum::extract::Request) -> Response {
    state.handled_requests.fetch_add(1, Ordering::Relaxed);
    let (parts, raw_body) = request.into_parts();
    let body = match read_body(raw_body).await {
        Ok(body) => body,
        Err(error) => return google_error_response(&error),
    };
    let raw_key = match generate_content_api_key(&parts.headers, &parts.uri) {
        Ok(key) => key,
        Err(error) => return google_error_response(&error),
    };
    let authenticate = compact_json(&json!({"raw_key": raw_key}));
    if let Err(error) = state.bridge.call("authenticate", authenticate).await {
        return google_error_response(&error);
    }
    let body_text = match String::from_utf8(body.to_vec()) {
        Ok(text) => text,
        Err(_) => return google_error_response(&PublicError::invalid_json()),
    };
    let argument = compact_json(&json!({
        "raw_key": raw_key,
        "body": body_text,
        "surface": "generate_content",
        "model": model,
    }));
    match state.bridge.call("count_tokens", argument).await {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(payload) => json_response(StatusCode::OK, &payload, &[]),
            Err(_) => google_error_response(&PublicError::internal()),
        },
        Err(error) => google_error_response(&error),
    }
}

async fn generate(
    state: AppState,
    model: String,
    request: axum::extract::Request,
    stream: bool,
) -> Response {
    state.handled_requests.fetch_add(1, Ordering::Relaxed);
    let started = Instant::now();
    let deadline = started + state.request_timeout;
    let (parts, raw_body) = request.into_parts();
    let headers = parts.headers;
    let body = match read_body(raw_body).await {
        Ok(body) => body,
        Err(error) => return google_error_response(&error),
    };
    let raw_key = match generate_content_api_key(&headers, &parts.uri) {
        Ok(key) => key,
        Err(error) => return google_error_response(&error),
    };
    let authenticate = compact_json(&json!({"raw_key": raw_key}));
    if let Err(error) = state.bridge.call("authenticate", authenticate).await {
        return google_error_response(&error);
    }
    let body_text = match String::from_utf8(body.to_vec()) {
        Ok(text) => text,
        Err(_) => return google_error_response(&PublicError::invalid_json()),
    };
    let mut admit_value = json!({
        "raw_key": raw_key,
        "body": body_text,
        "surface": if stream { "stream_generate_content" } else { "generate_content" },
        "model": model,
        "client_ip": client_ip(&headers),
        "capture_session_id": crate::capture::session_id(&headers),
    });
    with_app_identity(&mut admit_value, &headers);
    let admission_text = match state.bridge.call("admit", compact_json(&admit_value)).await {
        Ok(text) => text,
        Err(error) => return google_error_response(&error),
    };
    let admission_value: Value = match serde_json::from_str(&admission_text) {
        Ok(value) => value,
        Err(_) => return google_error_response(&PublicError::internal()),
    };
    if let Some(reason) = admission_value.get("escalate") {
        METRICS.record_escalation(classify_escalation(reason.as_str().unwrap_or_default()));
        return google_error_response(&escalation_error(reason.as_str().unwrap_or_default()));
    }
    let mut admission: Admission = match serde_json::from_value(admission_value.clone()) {
        Ok(admission) => admission,
        Err(_) => return wire_drift_response(&state, &admission_value, started).await,
    };
    let mut guard = new_guard(&state, admission.request_id.clone(), started);
    guard.record_web_search(admission.web_search_meter());
    let permit = match acquire_permit_with(
        &state,
        &mut guard,
        deadline,
        admission.class,
        google_error_response,
    )
    .await
    {
        Ok(permit) => permit,
        Err(response) => return *response,
    };
    let context = WaterfallContext {
        bridge: &state.bridge,
        http: &state.http,
        request_id: &admission.request_id,
        raw_key: &raw_key,
        caller_scope: admission.caller_scope.as_deref(),
        route: &admission.route,
        policy: admission.policy(),
        deadline,
        pre_first_byte_deadline: state.first_byte_budget(deadline, &admission.route, &body_text),
        time_to_first_byte: state.time_to_first_byte,
        time_to_first_byte_slope_seconds_per_million_input_tokens: state
            .time_to_first_byte_slope_seconds_per_million_input_tokens,
        time_to_first_token: state.time_to_first_token,
        approximate_input_tokens: crate::waterfall::approximate_input_tokens(&body_text),
        chat_logprobs: false,
        capture_logprobs: !admission.buffers_output()
            && state.capture.as_ref().is_some_and(|capture| {
                capture.captures_logprobs()
                    && crate::capture::logprobs::route_may_inject(&admission.route)
                    && capture.wants_logprobs(&admission.request_id)
            }),
        output_less_retention: None,
        output_token_cap: admission.maximum_output_tokens,
        tool_search: admission.tool_search.as_ref(),
        output_guardrails: admission.output_guardrail.enforces().then_some(
            crate::waterfall::OutputGuardrailContext {
                web_search: admission.web_search.as_ref(),
                responses: false,
            },
        ),
    };
    let mut won =
        acquire_guarded_attempt(&context, &mut guard, admission.guardrail_input_pending).await;
    adopt_outcome(&mut admission, &mut won);
    won = checkpoint_winner(
        state.capture.as_ref(),
        &admission,
        &mut guard,
        won,
        deadline,
    )
    .await;
    observe_winner(state.capture.clone(), &admission, &guard, &mut won, false);
    let capture = state.capture.clone();
    let capture_request_id = admission.request_id.clone();
    let response = match won {
        Won::Failed(error) => google_error_response(&error),
        Won::Settled(settled) => settled_response(&admission, settled).await,
        Won::Committed(committed) => {
            let committed = *committed;
            let incremental = admission.stream_incremental(committed.depth);
            if admission.buffers_output() && !incremental {
                guarded(state, admission, guard, committed, deadline, permit).await
            } else if admission.stream {
                stream_response(admission, guard, committed, deadline, permit, incremental).await
            } else {
                completed(admission, guard, committed, deadline, permit).await
            }
        }
    };
    crate::capture::response::capture_response(capture, &capture_request_id, response)
}

/// Abandon a durably accepted request whose admission reply failed to parse.
async fn wire_drift_response(
    state: &AppState,
    admission_value: &Value,
    started: Instant,
) -> Response {
    let request_id = admission_value
        .get("request_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if !request_id.is_empty() {
        let mut guard = new_guard(state, request_id, started);
        guard
            .abandon(&Failure::new(
                FailureClass::Internal,
                "gateway admission wire contract failed",
            ))
            .await;
    }
    google_error_response(&PublicError::internal())
}

fn aggregate(
    admission: &Admission,
    events: &[Event],
    carrier: Option<&str>,
    output: ReasoningOutput,
) -> Result<AggregatedGenerateContent, PublicError> {
    completed_generate_content_body(
        &admission.request_id,
        &admission.alias,
        events,
        &admission.ignored_parameters,
        carrier,
        output,
        admission.include_thoughts,
    )
}

fn encode_sse(
    admission: &Admission,
    events: &[Event],
    carrier: Option<&str>,
    output: ReasoningOutput,
) -> Result<Vec<u8>, PublicError> {
    let mut encoder = GenerateContentEncoder::new(
        &admission.request_id,
        &admission.alias,
        admission.ignored_parameters.clone(),
        admission.include_thoughts,
    );
    encoder.set_reasoning_output(output);
    if let Some(carrier) = carrier {
        encoder.set_reasoning_content_carrier(carrier.to_string());
    }
    let mut body = Vec::new();
    for event in events {
        for frame in encoder.feed(event)? {
            body.extend_from_slice(frame.as_bytes());
        }
    }
    Ok(body)
}

/// Answer one attempt the waterfall already settled.
async fn settled_response(admission: &Admission, settled: SettledAttempt) -> Response {
    let served = settled.served();
    let mut events = settled.events;
    let refusal_completed = complete_visible_refusal(&mut events);
    if refusal_completed.is_none() {
        if let Some(Event::Failed(failure)) = events.last() {
            let error = collection_public_error(&failure.clone().boundary());
            if admission.stream {
                let body = match encode_sse(admission, &events, None, ReasoningOutput::default()) {
                    Ok(body) => body,
                    Err(error) => return google_error_response(&error),
                };
                return sse_body_response(&served_headers(admission, None, served), body);
            }
            return google_error_response(&error);
        }
    }
    let headers = served_headers(admission, None, served);
    let exposed = admission.reasoning_output_at(settled.depth);
    if admission.stream {
        return match encode_sse(admission, &events, None, exposed) {
            Ok(body) => sse_body_response(&headers, body),
            Err(error) => google_error_response(&error),
        };
    }
    match aggregate(admission, &events, None, exposed) {
        Ok(aggregated) => match &aggregated.failure {
            Some(failure) => google_error_response(&failure.clone().boundary().public_error()),
            None => json_response(StatusCode::OK, &aggregated.body, &headers),
        },
        Err(error) => google_error_response(&error),
    }
}

/// Aggregate one committed non-streaming or guarded attempt and answer it,
/// settling exactly once.
#[allow(clippy::too_many_arguments)]
async fn respond_from_events(
    admission: Admission,
    mut guard: AttemptGuard,
    served: Served,
    mut events: Vec<Event>,
    usage: Option<Usage>,
    tool_names: Vec<String>,
    stream_body: bool,
) -> Response {
    let depth = served.depth;
    let refusal_completed = complete_visible_refusal(&mut events);
    let exposed = admission.reasoning_output_at(depth);
    let carrier = if refusal_completed.is_some() || exposed.exposed {
        None
    } else {
        match seal_reasoning_events(&guard.bridge, &admission.request_id, depth, &events).await {
            Ok(carrier) => carrier,
            Err(failure) => {
                guard
                    .settle("failed", usage.as_ref(), &tool_names, Some(&failure), true)
                    .await;
                return google_error_response(&failure.public_error());
            }
        }
    };
    let aggregated = match aggregate(&admission, &events, carrier.as_deref(), exposed) {
        Ok(aggregated) => aggregated,
        Err(error) => {
            guard
                .settle(
                    "failed",
                    usage.as_ref(),
                    &tool_names,
                    Some(
                        &Failure::new(
                            FailureClass::MalformedResponse,
                            "provider stream ended without a terminal event",
                        )
                        .boundary(),
                    ),
                    true,
                )
                .await;
            return google_error_response(&error);
        }
    };
    if let Some(failure) = &aggregated.failure {
        let failure = failure.clone().boundary();
        let error = failure.public_error();
        guard
            .settle(
                "failed",
                aggregated.usage.as_ref().or(usage.as_ref()),
                &aggregated.tool_names,
                Some(&failure),
                true,
            )
            .await;
        return google_error_response(&error);
    }
    let terminal = events.iter().rev().find(|event| event.is_terminal());
    let empty_completion = refusal_completed.is_none()
        && aggregated.empty
        && terminal.is_some_and(|terminal| {
            let usage = aggregated.usage.as_ref().or(usage.as_ref());
            billed_empty_completion(terminal, usage) || unreported_empty_completion(terminal, usage)
        });
    let settled = if empty_completion {
        // The committed rung closed the turn with nothing this surface can
        // render: a typed 200 under `x-gateway-warning: empty_completion`,
        // the typed `empty_completion` failure in the ledger at $0.
        guard
            .settle(
                "failed",
                aggregated.usage.as_ref().or(usage.as_ref()),
                &aggregated.tool_names,
                Some(&Failure::empty_completion()),
                true,
            )
            .await
    } else if let Some(refusal) = &refusal_completed {
        guard
            .settle(
                "failed",
                aggregated.usage.as_ref().or(usage.as_ref()),
                &aggregated.tool_names,
                Some(refusal),
                true,
            )
            .await
    } else {
        let outcome = if aggregated.incomplete {
            "incomplete"
        } else {
            "completed"
        };
        guard
            .settle(
                outcome,
                aggregated.usage.as_ref().or(usage.as_ref()),
                &aggregated.tool_names,
                None,
                true,
            )
            .await
    };
    if !settled {
        return google_error_response(&guard.settlement_error());
    }
    let served = Served {
        empty_completion,
        ..served
    };
    let headers = served_headers(&admission, None, served);
    if stream_body {
        return match encode_sse(&admission, &events, carrier.as_deref(), exposed) {
            Ok(body) => sse_body_response(&headers, body),
            Err(error) => google_error_response(&error),
        };
    }
    json_response(StatusCode::OK, &aggregated.body, &headers)
}

async fn completed(
    mut admission: Admission,
    mut guard: AttemptGuard,
    mut committed: CommittedAttempt,
    deadline: Instant,
    permit: crate::dispatch_capacity::DispatchPermit,
) -> Response {
    let _permit = permit;
    let phase_timeout = admission.phase_timeout(committed.depth);
    let collection =
        collect_committed(&mut committed, deadline, phase_timeout, guard.started).await;
    guard.record_first_token(committed.relay.first_token_at());
    let events = match collection {
        Ok(events) => events,
        Err(failure) => {
            let failure = failure.boundary();
            let error = collection_public_error(&failure);
            guard
                .settle(
                    "failed",
                    committed.usage.as_ref(),
                    &committed.tool_names,
                    Some(&failure),
                    true,
                )
                .await;
            return google_error_response(&error);
        }
    };
    disclose_after_collection(&mut admission, &committed);
    respond_from_events(
        admission,
        guard,
        committed.served(),
        events,
        committed.usage,
        committed.tool_names,
        false,
    )
    .await
}

async fn guarded(
    state: AppState,
    mut admission: Admission,
    mut guard: AttemptGuard,
    mut committed: CommittedAttempt,
    deadline: Instant,
    permit: crate::dispatch_capacity::DispatchPermit,
) -> Response {
    let _permit = permit;
    let phase_timeout = admission.phase_timeout(committed.depth);
    let collection =
        collect_committed(&mut committed, deadline, phase_timeout, guard.started).await;
    guard.record_first_token(committed.relay.first_token_at());
    let collected = match collection {
        Ok(events) => events,
        Err(failure) => {
            let failure = failure.boundary();
            let error = collection_public_error(&failure);
            guard
                .settle(
                    "failed",
                    committed.usage.as_ref(),
                    &committed.tool_names,
                    Some(&failure),
                    true,
                )
                .await;
            return google_error_response(&error);
        }
    };
    disclose_after_collection(&mut admission, &committed);
    let events = match apply_output_guardrail(&state, &admission, collected, deadline, false).await
    {
        Ok(events) => events,
        Err(failure) => {
            guard
                .settle(
                    "failed",
                    committed.usage.as_ref(),
                    &committed.tool_names,
                    Some(&failure),
                    true,
                )
                .await;
            return google_error_response(&failure.public_error());
        }
    };
    let stream_body = admission.stream;
    respond_from_events(
        admission,
        guard,
        committed.served(),
        events,
        committed.usage,
        committed.tool_names,
        stream_body,
    )
    .await
}

async fn stream_response(
    admission: Admission,
    guard: AttemptGuard,
    committed: CommittedAttempt,
    deadline: Instant,
    permit: crate::dispatch_capacity::DispatchPermit,
    incremental_guardrail: bool,
) -> Response {
    let (sender, receiver) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    let header_pairs = served_headers(&admission, None, committed.served());
    let request_id = admission.request_id.clone();
    let phase_timeout = admission.phase_timeout(committed.depth);
    let task_hold = guard.hold_task();
    tokio::spawn(async move {
        let _task = task_hold;
        let _permit = permit;
        let mut guard = guard;
        let mut committed = committed;
        let mut delivery = Delivery::new(sender.clone(), false).paced(committed.pace);
        let mut encoder = GenerateContentEncoder::new(
            &request_id,
            &admission.alias,
            admission.ignored_parameters.clone(),
            admission.include_thoughts,
        );
        encoder.set_reasoning_output(admission.reasoning_output_at(committed.depth));
        let mut usage: Option<Usage> = committed.usage.take();
        let mut tool_names: Vec<String> = std::mem::take(&mut committed.tool_names);
        let mut visible_refusal = committed.visible_refusal;
        let mut terminal: Option<Event> = None;
        let mut output_guardrails = StreamGuardrails::new(&request_id, incremental_guardrail);
        let mut empty_completion = false;

        macro_rules! fail_stream {
            ($failure:expr) => {{
                committed.relay.close_transport();
                log_stream_exit(&request_id, "provider_or_encoding_failure");
                let failure = $failure.boundary();
                guard
                    .settle("failed", usage.as_ref(), &tool_names, Some(&failure), true)
                    .await;
                emit_failure(&sender, deadline, &mut encoder, &failure).await;
                return;
            }};
        }

        guard.record_first_token(committed.relay.first_token_at());
        let mut prefix: std::collections::VecDeque<Event> = committed.prefix.drain(..).collect();
        let mut send_deadline = deadline;
        loop {
            if committed.relay.deadline_cut(deadline, &mut send_deadline) {
                fail_stream!(crate::relay::deadline_while_generating())
            }
            let event = if let Some(event) = prefix.pop_front() {
                event
            } else {
                match delivery
                    .next(
                        committed
                            .relay
                            .next_event(deadline, phase_timeout, guard.started),
                    )
                    .await
                {
                    None => {
                        committed.relay.close_transport();
                        log_stream_exit(&request_id, "subscriber_closed");
                        guard.settle_cancelled(usage.as_ref(), &tool_names).await;
                        return;
                    }
                    Some(Ok(Some(event))) => event,
                    Some(Ok(None)) => {
                        usage = committed.relay.usage_before_failure(usage.take());
                        fail_stream!(Failure::new(
                            FailureClass::MalformedResponse,
                            "provider stream ended without a terminal event",
                        ))
                    }
                    Some(Err(failure)) => {
                        usage = committed.relay.usage_before_failure(usage.take());
                        fail_stream!(failure)
                    }
                }
            };
            track_event(&event, &mut usage, &mut tool_names);
            if matches!(event, Event::Failed(_)) {
                usage = committed.relay.usage_before_failure(usage.take());
            }
            guard.record_first_token(committed.relay.first_token_at());
            let outward = outward_event(&event, &mut visible_refusal);
            let outward_events = match output_guardrails
                .release(&guard.bridge, outward, event.is_terminal())
                .await
            {
                Ok(events) => events,
                Err(failure) => fail_stream!(failure),
            };
            if event.is_terminal() {
                committed.relay.close_transport();
                if matches!(event, Event::Completed | Event::StoppedAtSequence(_)) {
                    let candidate = match encoder.reasoning_carrier_candidate() {
                        Ok(candidate) => candidate,
                        Err(_) => {
                            fail_stream!(Failure::new(
                                FailureClass::MalformedResponse,
                                "provider returned malformed reasoning continuation data",
                            ))
                        }
                    };
                    if !admission.reasoning_output_at(committed.depth).exposed {
                        match seal_reasoning_candidate(
                            &guard.bridge,
                            &request_id,
                            committed.depth,
                            candidate,
                        )
                        .await
                        {
                            Ok(Some(carrier)) => encoder.set_reasoning_content_carrier(carrier),
                            Ok(None) => {}
                            Err(failure) => fail_stream!(failure),
                        }
                    }
                    if !encoder.has_content()
                        && (billed_empty_completion(&event, usage.as_ref())
                            || unreported_empty_completion(&event, usage.as_ref()))
                    {
                        empty_completion = true;
                    }
                }
                terminal = Some(event.clone());
                let settled = if empty_completion {
                    guard
                        .price_or_settle(
                            "failed",
                            usage.as_ref(),
                            &tool_names,
                            Some(&Failure::empty_completion()),
                        )
                        .await
                } else if encoder.refusal_seen() && !matches!(event, Event::Failed(_)) {
                    // A refusal the provider closed with a plain stop has no
                    // part on this surface: the encoder answers Google's error
                    // frame, so the ledger records the typed refusal, exactly
                    // as the non-streaming aggregate settles it.
                    guard
                        .price_or_settle(
                            "failed",
                            usage.as_ref(),
                            &tool_names,
                            Some(&refusal_failure().boundary()),
                        )
                        .await
                } else {
                    price_stream_end(&mut guard, &event, usage.as_ref(), &tool_names).await
                };
                if !settled {
                    log_stream_exit(&request_id, "settlement_unavailable");
                    return;
                }
            }
            for outward in delivery.pace(outward_events, deadline).await {
                let encoded = match encoder.feed(&outward) {
                    Ok(encoded) => encoded,
                    Err(_) => {
                        if terminal.is_some() {
                            log_stream_exit(&request_id, "terminal_encoding_failed");
                            return;
                        }
                        fail_stream!(Failure::new(
                            FailureClass::Internal,
                            "gateway could not encode the provider stream",
                        ))
                    }
                };
                for data in encoded {
                    if !delivery.send(send_deadline, Bytes::from(data)).await {
                        committed.relay.close_transport();
                        log_stream_exit(&request_id, "subscriber_closed_or_delivery_deadline");
                        settle_stream_end(
                            &mut guard,
                            terminal.as_ref(),
                            usage.as_ref(),
                            &tool_names,
                            true,
                        )
                        .await;
                        return;
                    }
                }
                if outward.is_terminal() {
                    drop(delivery);
                    drop(sender);
                    guard.settle_priced().await;
                    return;
                }
            }
        }
    });

    let body = Body::from_stream(ReceiverStream::new(receiver));
    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8");
    for (name, value) in &header_pairs {
        if let (Ok(name), Ok(value)) = (
            header::HeaderName::try_from(name.as_str()),
            HeaderValue::try_from(value.as_str()),
        ) {
            builder = builder.header(name, value);
        }
    }
    builder
        .body(body)
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

/// Emit the sanitized failure frame (Google's error envelope as one data
/// frame) when the stream has not already reached a terminal.
async fn emit_failure(
    sender: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    deadline: Instant,
    encoder: &mut GenerateContentEncoder,
    failure: &Failure,
) {
    if encoder.saw_terminal() {
        return;
    }
    let frames = encoder
        .feed(&Event::Failed(failure.clone()))
        .unwrap_or_default();
    let deadline = crate::respond::terminal_send_deadline(deadline);
    for frame in frames {
        if !send_bounded(sender, deadline, Bytes::from(frame)).await {
            return;
        }
    }
}

#[cfg(test)]
#[path = "route_generate_content_tests.rs"]
mod tests;
