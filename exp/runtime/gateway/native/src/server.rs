//! The axum data plane: routes, admission, upstream relay, and settlement.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::serve::ListenerExt;
use axum::Router;
use bytes::Bytes;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokio::sync::{mpsc, Semaphore};
use tokio_stream::wrappers::ReceiverStream;

use crate::bridge::Bridge;
use crate::decode::{decode_chat, DecodedChat};
use crate::dialects::{
    build_payload, Dialect, Normalizer, WireHints, MAXIMUM_RETAINED_OUTPUT_BYTES,
    OUTPUT_OVERFLOW_MESSAGE,
};
use crate::encode::{chat_data, compact_json, completed_chat_body, ChatSseEncoder};
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{Event, Usage};
use crate::sse::SseDecoder;
use crate::upstream::{completion_timeout_seconds, open_stream};

/// Serve-time configuration passed from `exp run --engine rust`.
#[derive(Debug, Clone, Deserialize)]
pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    #[serde(default = "default_max_active_requests")]
    pub max_active_requests: usize,
    #[serde(default = "default_request_timeout_seconds")]
    pub request_timeout_seconds: f64,
    #[serde(default = "default_callback_permits")]
    pub callback_permits: usize,
    /// Loopback port of the embedded python engine serving escalated requests.
    pub fallback_port: u16,
    #[serde(default = "default_graceful_timeout_seconds")]
    pub graceful_timeout_seconds: f64,
}

fn default_graceful_timeout_seconds() -> f64 {
    10.0
}

fn default_max_active_requests() -> usize {
    64
}

fn default_request_timeout_seconds() -> f64 {
    120.0
}

fn default_callback_permits() -> usize {
    4
}

/// Shared server state.
#[derive(Clone)]
struct AppState {
    bridge: Arc<Bridge>,
    http: reqwest::Client,
    permits: Arc<Semaphore>,
    request_timeout: Duration,
    fallback_base: String,
}

/// The wire configuration returned by one successful admission.
#[derive(Debug, Clone, Deserialize)]
struct Admission {
    request_id: String,
    attempt_id: String,
    alias: String,
    alias_revision_id: String,
    dialect: String,
    url: String,
    headers: HashMap<String, String>,
    model_id: String,
    #[serde(default = "default_true")]
    supports_temperature: bool,
    #[serde(default)]
    reasoning_effort: Option<String>,
    #[serde(default = "default_token_limit_key")]
    token_limit_key: String,
    timeout_seconds: f64,
    idempotency_key: String,
    exact_model_id: String,
    provider: String,
    deployment_id: String,
    route_reason: String,
}

fn default_true() -> bool {
    true
}

fn default_token_limit_key() -> String {
    "max_tokens".to_string()
}

/// Run the data plane until shutdown; returns after graceful stop.
pub async fn run(bridge: Arc<Bridge>, config: ServeConfig) -> Result<(), String> {
    let http = crate::upstream::build_client()?;
    let state = AppState {
        bridge,
        http,
        permits: Arc::new(Semaphore::new(config.max_active_requests.max(1))),
        request_timeout: Duration::from_secs_f64(config.request_timeout_seconds),
        fallback_base: format!("http://127.0.0.1:{}", config.fallback_port),
    };
    let app = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/models/{model_id}", get(model_detail))
        .route("/v1/chat/completions", post(chat))
        .route("/health/live", get(health_live))
        .route("/health/ready", get(health_ready))
        .route("/usage.json", get(usage_json))
        .fallback(proxy_fallback)
        .with_state(state);
    let listener = tokio::net::TcpListener::bind((config.host.as_str(), config.port))
        .await
        .map_err(|error| format!("failed to bind {}:{}: {error}", config.host, config.port))?
        // Small SSE frames must not sit behind Nagle's algorithm.
        .tap_io(|stream| {
            let _ = stream.set_nodelay(true);
        });
    let graceful = Duration::from_secs_f64(config.graceful_timeout_seconds.max(0.1));
    let server = axum::serve(listener, app).with_graceful_shutdown(async {
        let _ = tokio::signal::ctrl_c().await;
    });
    // SIGINT starts the graceful drain above; the arm below bounds it, so a
    // stuck stream cannot hold shutdown past the configured timeout.
    tokio::select! {
        outcome = server => outcome.map_err(|error| format!("gateway server failed: {error}")),
        _ = async {
            let _ = tokio::signal::ctrl_c().await;
            tokio::time::sleep(graceful).await;
        } => Ok(()),
    }
}

fn error_response(error: &PublicError) -> Response {
    let mut builder = Response::builder()
        .status(StatusCode::from_u16(error.status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR))
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(wait) = error.retry_after_seconds {
        builder = builder.header(header::RETRY_AFTER, wait.to_string());
    }
    builder
        .body(Body::from(compact_json(&error.json_body())))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

fn json_response(status: StatusCode, payload: &Value, headers: &[(String, String)]) -> Response {
    let mut builder = Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json");
    for (name, value) in headers {
        builder = builder.header(name, value);
    }
    builder
        .body(Body::from(compact_json(payload)))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

fn bearer_key(headers: &HeaderMap) -> Result<String, PublicError> {
    let value = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .ok_or_else(PublicError::invalid_key)?;
    let key = value.strip_prefix("Bearer ").ok_or_else(PublicError::invalid_key)?;
    let trimmed = key.trim();
    if trimmed.is_empty() {
        return Err(PublicError::invalid_key());
    }
    Ok(trimmed.to_string())
}

fn header_string(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
}

async fn health_live() -> Response {
    json_response(StatusCode::OK, &json!({"status": "live"}), &[])
}

async fn health_ready(State(state): State<AppState>) -> Response {
    match state.bridge.call("readiness", "{}".to_string()).await {
        Ok(text) if text == "true" => {
            json_response(StatusCode::OK, &json!({"status": "ready"}), &[])
        }
        _ => json_response(
            StatusCode::SERVICE_UNAVAILABLE,
            &json!({"status": "not_ready"}),
            &[],
        ),
    }
}

async fn usage_json(State(state): State<AppState>) -> Response {
    match state.bridge.call("usage_json", "{}".to_string()).await {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(payload) => json_response(StatusCode::OK, &payload, &[]),
            Err(_) => error_response(&PublicError::internal()),
        },
        Err(error) => error_response(&error),
    }
}

/// Replay one HTTP request against the embedded python engine and stream the
/// response back unchanged. Serves every surface the native plane does not
/// implement (Responses, replay-keyed chat, usage pages, unknown routes).
async fn proxy_to_python(
    state: &AppState,
    method: reqwest::Method,
    path_and_query: &str,
    headers: &HeaderMap,
    body: Bytes,
) -> Response {
    let url = format!("{}{}", state.fallback_base, path_and_query);
    let mut request = state.http.request(method, url);
    for (name, value) in headers {
        let lowered = name.as_str().to_ascii_lowercase();
        if matches!(
            lowered.as_str(),
            "host" | "connection" | "content-length" | "transfer-encoding" | "keep-alive"
                | "te" | "trailer" | "upgrade"
        ) {
            continue;
        }
        request = request.header(name, value);
    }
    let upstream = match request.body(body).send().await {
        Ok(upstream) => upstream,
        Err(_) => {
            return error_response(&PublicError::new(
                502,
                "fallback_engine_unavailable",
                "The python fallback engine did not answer. Retry shortly; if this persists, restart the gateway.",
                "api_error",
            ))
        }
    };
    let status = StatusCode::from_u16(upstream.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let mut builder = Response::builder().status(status);
    for (name, value) in upstream.headers() {
        let lowered = name.as_str().to_ascii_lowercase();
        if matches!(
            lowered.as_str(),
            "connection" | "transfer-encoding" | "keep-alive" | "te" | "trailer" | "upgrade"
        ) {
            continue;
        }
        builder = builder.header(name, value);
    }
    builder
        .body(Body::from_stream(upstream.bytes_stream()))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

/// Route every path the native plane does not own to the python engine.
async fn proxy_fallback(
    State(state): State<AppState>,
    request: axum::extract::Request,
) -> Response {
    let (parts, body) = request.into_parts();
    let bytes = match axum::body::to_bytes(body, 512 * 1024 * 1024).await {
        Ok(bytes) => bytes,
        Err(_) => return error_response(&PublicError::invalid_json()),
    };
    let method = reqwest::Method::from_bytes(parts.method.as_str().as_bytes())
        .unwrap_or(reqwest::Method::GET);
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|value| value.as_str().to_string())
        .unwrap_or_else(|| parts.uri.path().to_string());
    proxy_to_python(&state, method, &path_and_query, &parts.headers, bytes).await
}

async fn models(State(state): State<AppState>, headers: HeaderMap) -> Response {
    let raw_key = match bearer_key(&headers) {
        Ok(key) => key,
        Err(error) => return error_response(&error),
    };
    let argument = compact_json(&json!({"raw_key": raw_key}));
    match state.bridge.call("models", argument).await {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(payload) => json_response(StatusCode::OK, &payload, &[]),
            Err(_) => error_response(&PublicError::internal()),
        },
        Err(error) => error_response(&error),
    }
}

async fn model_detail(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    let raw_key = match bearer_key(&headers) {
        Ok(key) => key,
        Err(error) => return error_response(&error),
    };
    let argument = compact_json(&json!({"raw_key": raw_key, "model_id": model_id}));
    match state.bridge.call("model_detail", argument).await {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(payload) => json_response(StatusCode::OK, &payload, &[]),
            Err(_) => error_response(&PublicError::internal()),
        },
        Err(error) => error_response(&error),
    }
}

/// Commit-independent headers, mirroring `commit_independent_headers`.
fn commit_independent(admission: &Admission, client_request_id: Option<&str>) -> Vec<(String, String)> {
    let mut headers = vec![
        ("x-request-id".to_string(), admission.request_id.clone()),
        ("x-gateway-alias".to_string(), admission.alias.clone()),
        (
            "x-gateway-alias-revision".to_string(),
            admission.alias_revision_id.clone(),
        ),
    ];
    if let Some(id) = client_request_id {
        headers.push(("x-client-request-id".to_string(), id.to_string()));
    }
    headers
}

/// Commit-dependent headers, mirroring `commit_dependent_headers`.
fn commit_dependent(admission: &Admission) -> Vec<(String, String)> {
    vec![
        (
            "x-gateway-canonical-model".to_string(),
            admission.exact_model_id.clone(),
        ),
        ("x-gateway-provider".to_string(), admission.provider.clone()),
        (
            "x-gateway-deployment".to_string(),
            admission.deployment_id.clone(),
        ),
        ("x-gateway-route-depth".to_string(), "0".to_string()),
        (
            "x-gateway-route-reason".to_string(),
            admission.route_reason.clone(),
        ),
    ]
}

fn settle_argument(
    admission: &Admission,
    outcome: &str,
    usage: Option<&Usage>,
    tool_names: &[String],
    failure: Option<&Failure>,
) -> String {
    compact_json(&json!({
        "request_id": admission.request_id,
        "attempt_id": admission.attempt_id,
        "outcome": outcome,
        "usage": usage.map(|usage| json!({
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
        })),
        "tool_names": tool_names,
        "failure": failure.map(|failure| json!({
            "failure_class": failure.failure_class.as_str(),
            "safe_message": failure.safe_message,
        })),
    }))
}

async fn settle(
    bridge: &Bridge,
    admission: &Admission,
    outcome: &str,
    usage: Option<&Usage>,
    tool_names: &[String],
    failure: Option<&Failure>,
) {
    let argument = settle_argument(admission, outcome, usage, tool_names, failure);
    // The control plane keeps the in-flight entry on a failed terminal write,
    // so a transient ledger failure is retried here with bounded backoff; a
    // persistent failure stays latched as accounting-unhealthy control-plane
    // side and is left for restart reconciliation.
    for backoff_ms in [0u64, 100, 500, 2_000] {
        if backoff_ms > 0 {
            tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
        }
        if bridge.call("settle", argument.clone()).await.is_ok() {
            return;
        }
    }
}

async fn chat(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let started = Instant::now();
    let deadline = started + state.request_timeout;

    let raw_key = match bearer_key(&headers) {
        Ok(key) => key,
        Err(error) => return error_response(&error),
    };
    let authenticate = compact_json(&json!({"raw_key": raw_key}));
    if let Err(error) = state.bridge.call("authenticate", authenticate).await {
        return error_response(&error);
    }

    let payload: Value = match serde_json::from_slice(&body) {
        Ok(payload) => payload,
        Err(_) => return error_response(&PublicError::invalid_json()),
    };
    let object: &Map<String, Value> = match payload.as_object() {
        Some(object) => object,
        None => return error_response(&PublicError::not_json_object()),
    };
    let idempotency_key = header_string(&headers, "idempotency-key");
    let client_request_id = header_string(&headers, "x-client-request-id");
    if idempotency_key.is_some() || client_request_id.is_some() {
        // Replay-keyed chat keeps the python engine's idempotency semantics.
        return proxy_to_python(
            &state,
            reqwest::Method::POST,
            "/v1/chat/completions",
            &headers,
            body,
        )
        .await;
    }
    let decoded = match decode_chat(object, idempotency_key.as_deref(), client_request_id.as_deref())
    {
        Ok(decoded) => decoded,
        Err(error) => return error_response(&error),
    };

    // Logical admission mirrors the executor's bounded active-request permit.
    let permit = match tokio::time::timeout_at(
        deadline.into(),
        state.permits.clone().acquire_owned(),
    )
    .await
    {
        Ok(Ok(permit)) => permit,
        Ok(Err(_)) => return error_response(&PublicError::draining()),
        Err(_) => {
            return error_response(
                &Failure::new(
                    FailureClass::Timeout,
                    "gateway execution queue deadline exceeded",
                )
                .public_error(),
            )
        }
    };

    let admit_argument = compact_json(&json!({
        "raw_key": raw_key,
        "alias": decoded.alias,
        "request": decoded.canonical,
        "stream": decoded.stream,
    }));
    let admission: Admission = match state.bridge.call("admit", admit_argument).await {
        Ok(text) => match serde_json::from_str(&text) {
            Ok(admission) => admission,
            Err(_) => return error_response(&PublicError::internal()),
        },
        Err(error) => {
            if error.code == "native_unsupported" {
                drop(permit);
                return proxy_to_python(
                    &state,
                    reqwest::Method::POST,
                    "/v1/chat/completions",
                    &headers,
                    body,
                )
                .await;
            }
            return error_response(&error);
        }
    };

    let dialect = match Dialect::from_str(&admission.dialect) {
        Some(dialect) => dialect,
        None => {
            settle(
                &state.bridge,
                &admission,
                "failed",
                None,
                &[],
                Some(&Failure::new(
                    FailureClass::Internal,
                    "gateway engine does not support the resolved provider dialect",
                )),
            )
            .await;
            return error_response(&PublicError::internal());
        }
    };
    let hints = WireHints {
        model_id: admission.model_id.clone(),
        supports_temperature: admission.supports_temperature,
        reasoning_effort: admission.reasoning_effort.clone(),
        token_limit_key: admission.token_limit_key.clone(),
    };
    let upstream_payload = match build_payload(dialect, &decoded.canonical, &hints) {
        Ok(payload) => payload,
        Err(failure) => {
            let error = failure.public_error();
            settle(&state.bridge, &admission, "failed", None, &[], Some(&failure)).await;
            return error_response(&error);
        }
    };

    let maximum_output_tokens = decoded
        .canonical
        .get("maximum_output_tokens")
        .and_then(Value::as_u64);
    let phase_timeout = Duration::from_secs_f64(completion_timeout_seconds(
        admission.timeout_seconds,
        maximum_output_tokens,
    ));
    let open_bound = remaining(deadline).min(phase_timeout);
    let response = match open_stream(
        &state.http,
        &admission.url,
        &admission.headers,
        &admission.idempotency_key,
        &upstream_payload,
        open_bound,
    )
    .await
    {
        Ok(response) => response,
        Err(failure) => {
            let error = failure.public_error();
            settle(&state.bridge, &admission, "failed", None, &[], Some(&failure)).await;
            return error_response(&error);
        }
    };

    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or(0);

    if decoded.stream {
        stream_response(
            state,
            admission,
            decoded,
            dialect,
            response,
            created_at,
            deadline,
            phase_timeout,
            permit,
        )
        .await
    } else {
        completed_response(
            state,
            admission,
            decoded,
            dialect,
            response,
            created_at,
            deadline,
            phase_timeout,
            permit,
        )
        .await
    }
}

fn remaining(deadline: Instant) -> Duration {
    deadline.saturating_duration_since(Instant::now())
}

/// Approximate retained size of one aggregated event, in bytes.
fn event_retained_bytes(event: &Event) -> usize {
    match event {
        Event::TextDelta(text) | Event::RefusalDelta(text) => text.len(),
        Event::ToolArgumentsDelta { delta, .. } => delta.len(),
        // Completed-call bytes were already charged as argument deltas.
        Event::ToolCallCompleted { .. } => 64,
        _ => 64,
    }
}

/// Map one collection failure to its public error, honoring the shared
/// aggregate-output overflow contract.
fn collection_public_error(failure: &Failure) -> PublicError {
    if failure.safe_message == OUTPUT_OVERFLOW_MESSAGE {
        return PublicError::provider_output_too_large();
    }
    failure.public_error()
}

/// Drain one upstream SSE response into normalized events.
async fn collect_events(
    response: reqwest::Response,
    dialect: Dialect,
    deadline: Instant,
    phase_timeout: Duration,
) -> Result<Vec<Event>, Failure> {
    let mut normalizer = Normalizer::new(dialect);
    let mut decoder = SseDecoder::new();
    let mut events = Vec::new();
    let mut retained_bytes = 0usize;
    let mut byte_stream = response.bytes_stream();
    loop {
        let bound = remaining(deadline).min(phase_timeout);
        let chunk = match tokio::time::timeout(bound, byte_stream.next()).await {
            Ok(Some(Ok(chunk))) => chunk,
            Ok(Some(Err(_))) => {
                return Err(Failure::new(
                    FailureClass::Transport,
                    "provider transport failed",
                ))
            }
            Ok(None) => break,
            Err(_) => {
                return Err(Failure::new(
                    FailureClass::Timeout,
                    "provider response stream timed out",
                ))
            }
        };
        let frames = decoder
            .feed(&chunk)
            .map_err(|message| Failure::new(FailureClass::MalformedResponse, &message))?;
        for frame in frames {
            for event in normalizer.feed(&frame)? {
                retained_bytes = retained_bytes.saturating_add(event_retained_bytes(&event));
                if retained_bytes > MAXIMUM_RETAINED_OUTPUT_BYTES {
                    return Err(Failure::new(
                        FailureClass::ProviderInternal,
                        OUTPUT_OVERFLOW_MESSAGE,
                    ));
                }
                events.push(event);
            }
            if normalizer.saw_terminal() {
                return Ok(events);
            }
        }
    }
    if let Some(frame) = decoder
        .finish()
        .map_err(|message| Failure::new(FailureClass::MalformedResponse, &message))?
    {
        events.extend(normalizer.feed(&frame)?);
    }
    normalizer.stream_ended()?;
    Ok(events)
}

#[allow(clippy::too_many_arguments)]
async fn completed_response(
    state: AppState,
    admission: Admission,
    decoded: DecodedChat,
    dialect: Dialect,
    response: reqwest::Response,
    created_at: i64,
    deadline: Instant,
    phase_timeout: Duration,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    let _permit = permit;
    let events = match collect_events(response, dialect, deadline, phase_timeout).await {
        Ok(events) => events,
        Err(failure) => {
            let error = collection_public_error(&failure);
            settle(&state.bridge, &admission, "failed", None, &[], Some(&failure)).await;
            return error_response(&error);
        }
    };
    let aggregated =
        match completed_chat_body(&admission.request_id, &admission.alias, created_at, &events) {
            Ok(aggregated) => aggregated,
            Err(error) => {
                settle(
                    &state.bridge,
                    &admission,
                    "failed",
                    None,
                    &[],
                    Some(&Failure::new(
                        FailureClass::MalformedResponse,
                        "provider stream ended without a terminal event",
                    )),
                )
                .await;
                return error_response(&error);
            }
        };
    if let Some(failure) = &aggregated.failure {
        let error = failure.public_error();
        settle(
            &state.bridge,
            &admission,
            "failed",
            aggregated.usage.as_ref(),
            &aggregated.tool_names,
            Some(failure),
        )
        .await;
        return error_response(&error);
    }
    let outcome = if aggregated.incomplete { "incomplete" } else { "completed" };
    settle(
        &state.bridge,
        &admission,
        outcome,
        aggregated.usage.as_ref(),
        &aggregated.tool_names,
        None,
    )
    .await;
    let mut headers = commit_independent(&admission, decoded.client_request_id.as_deref());
    headers.extend(commit_dependent(&admission));
    json_response(StatusCode::OK, &aggregated.body, &headers)
}

#[allow(clippy::too_many_arguments)]
async fn stream_response(
    state: AppState,
    admission: Admission,
    decoded: DecodedChat,
    dialect: Dialect,
    response: reqwest::Response,
    created_at: i64,
    deadline: Instant,
    phase_timeout: Duration,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    let (sender, receiver) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    let header_pairs = commit_independent(&admission, decoded.client_request_id.as_deref());
    let bridge = state.bridge.clone();
    let include_usage = decoded.include_usage;
    tokio::spawn(async move {
        let _permit = permit;
        let mut encoder = ChatSseEncoder::new(
            &admission.request_id,
            &admission.alias,
            created_at,
            include_usage,
        );
        let mut normalizer = Normalizer::new(dialect);
        let mut decoder = SseDecoder::new();
        let mut usage: Option<Usage> = None;
        let mut tool_names: Vec<String> = Vec::new();
        let mut terminal: Option<Event> = None;

        let start_frames = match encoder.start() {
            Ok(frames) => frames,
            Err(_) => Vec::new(),
        };
        for frame in start_frames {
            if sender.send(Ok(Bytes::from(frame))).await.is_err() {
                settle_cancelled(&bridge, &admission, usage.as_ref(), &tool_names).await;
                return;
            }
        }

        let mut byte_stream = response.bytes_stream();
        'outer: loop {
            let bound = remaining(deadline).min(phase_timeout);
            let chunk = match tokio::time::timeout(bound, byte_stream.next()).await {
                Ok(Some(Ok(chunk))) => chunk,
                Ok(Some(Err(_))) => {
                    let failure = Failure::new(FailureClass::Transport, "provider transport failed");
                    emit_failure(&sender, &mut encoder, &failure).await;
                    settle(&bridge, &admission, "failed", usage.as_ref(), &tool_names, Some(&failure))
                        .await;
                    return;
                }
                Ok(None) => break 'outer,
                Err(_) => {
                    let failure = Failure::new(
                        FailureClass::Timeout,
                        "provider response stream timed out",
                    );
                    emit_failure(&sender, &mut encoder, &failure).await;
                    settle(&bridge, &admission, "failed", usage.as_ref(), &tool_names, Some(&failure))
                        .await;
                    return;
                }
            };
            let frames = match decoder.feed(&chunk) {
                Ok(frames) => frames,
                Err(message) => {
                    let failure = Failure::new(FailureClass::MalformedResponse, &message);
                    emit_failure(&sender, &mut encoder, &failure).await;
                    settle(&bridge, &admission, "failed", usage.as_ref(), &tool_names, Some(&failure))
                        .await;
                    return;
                }
            };
            for frame in frames {
                let events = match normalizer.feed(&frame) {
                    Ok(events) => events,
                    Err(failure) => {
                        emit_failure(&sender, &mut encoder, &failure).await;
                        settle(
                            &bridge,
                            &admission,
                            "failed",
                            usage.as_ref(),
                            &tool_names,
                            Some(&failure),
                        )
                        .await;
                        return;
                    }
                };
                for event in events {
                    track_event(&event, &mut usage, &mut tool_names);
                    let encoded = match encoder.feed(&event) {
                        Ok(encoded) => encoded,
                        Err(_) => Vec::new(),
                    };
                    for data in encoded {
                        if sender.send(Ok(Bytes::from(data))).await.is_err() {
                            settle_cancelled(&bridge, &admission, usage.as_ref(), &tool_names).await;
                            return;
                        }
                    }
                    if event.is_terminal() {
                        terminal = Some(event);
                        break 'outer;
                    }
                }
            }
        }

        match terminal {
            Some(Event::Failed(failure)) => {
                settle(&bridge, &admission, "failed", usage.as_ref(), &tool_names, Some(&failure))
                    .await;
            }
            Some(Event::Incomplete) => {
                settle(&bridge, &admission, "incomplete", usage.as_ref(), &tool_names, None).await;
            }
            Some(_) => {
                settle(&bridge, &admission, "completed", usage.as_ref(), &tool_names, None).await;
            }
            None => {
                let failure = Failure::new(
                    FailureClass::MalformedResponse,
                    "provider stream ended without a terminal event",
                );
                emit_failure(&sender, &mut encoder, &failure).await;
                settle(&bridge, &admission, "failed", usage.as_ref(), &tool_names, Some(&failure))
                    .await;
            }
        }
    });

    let body = Body::from_stream(ReceiverStream::new(receiver));
    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream");
    for (name, value) in &header_pairs {
        if let (Ok(name), Ok(value)) = (
            header::HeaderName::try_from(name.as_str()),
            HeaderValue::try_from(value.as_str()),
        ) {
            builder = builder.header(name, value);
        }
    }
    builder.body(body).unwrap_or_else(|_| Response::new(Body::empty()))
}

fn track_event(event: &Event, usage: &mut Option<Usage>, tool_names: &mut Vec<String>) {
    match event {
        Event::Usage(candidate) if candidate.has_token_counts() => {
            *usage = Some(candidate.clone());
        }
        Event::ToolCallCompleted { call, .. } => {
            if !tool_names.contains(&call.name) {
                tool_names.push(call.name.clone());
            }
        }
        _ => {}
    }
}

/// Emit the encoder's sanitized failure frame and done sentinel when the
/// stream has not already reached a terminal.
async fn emit_failure(
    sender: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    encoder: &mut ChatSseEncoder,
    failure: &Failure,
) {
    if encoder.saw_terminal() {
        return;
    }
    let frames = encoder
        .feed(&Event::Failed(failure.clone()))
        .unwrap_or_else(|_| {
            vec![
                chat_data(&failure.public_error().json_body()),
                "data: [DONE]\n\n".to_string(),
            ]
        });
    for frame in frames {
        if sender.send(Ok(Bytes::from(frame))).await.is_err() {
            return;
        }
    }
}

async fn settle_cancelled(
    bridge: &Bridge,
    admission: &Admission,
    usage: Option<&Usage>,
    tool_names: &[String],
) {
    settle(
        bridge,
        admission,
        "failed",
        usage,
        tool_names,
        Some(&Failure::new(
            FailureClass::Cancelled,
            "gateway request was cancelled",
        )),
    )
    .await;
}
