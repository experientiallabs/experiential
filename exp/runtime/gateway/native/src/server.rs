//! The axum data plane: routes, admission, upstream relay, and settlement.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
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
use serde_json::{json, Value};
use tokio::sync::{mpsc, Semaphore};
use tokio_stream::wrappers::ReceiverStream;

use crate::bridge::Bridge;
use crate::dialects::{
    Dialect, Normalizer, MAXIMUM_RETAINED_OUTPUT_BYTES, OUTPUT_OVERFLOW_MESSAGE,
};
use crate::encode::{chat_data, compact_json, completed_chat_body, ChatSseEncoder};
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{Event, Usage};
use crate::sse::SseDecoder;
use crate::upstream::open_stream;

/// Largest accepted request body on every native-served or proxied route.
/// Bounded so one client cannot hold unbounded gateway memory; far above any
/// real chat history (the python engine imposes no explicit cap).
const MAXIMUM_REQUEST_BODY_BYTES: usize = 64 * 1024 * 1024;

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
    /// Settlement writes still in flight, held open through graceful shutdown.
    pending_settlements: Arc<AtomicUsize>,
}

/// The wire configuration returned by one successful admission. The upstream
/// payload arrives fully built by the shared python dialect builders, so the
/// two engines cannot drift at the provider boundary.
#[derive(Debug, Clone, Deserialize)]
struct Admission {
    request_id: String,
    attempt_id: String,
    alias: String,
    alias_revision_id: String,
    stream: bool,
    include_usage: bool,
    dialect: String,
    url: String,
    headers: HashMap<String, String>,
    timeout_seconds: f64,
    upstream_payload: Value,
    idempotency_key: String,
    exact_model_id: String,
    provider: String,
    deployment_id: String,
    route_reason: String,
}

/// Run the data plane until shutdown; returns after graceful stop.
pub async fn run(bridge: Arc<Bridge>, config: ServeConfig) -> Result<(), String> {
    let http = crate::upstream::build_client()?;
    let pending_settlements = Arc::new(AtomicUsize::new(0));
    let state = AppState {
        bridge,
        http,
        permits: Arc::new(Semaphore::new(config.max_active_requests.max(1))),
        request_timeout: Duration::from_secs_f64(config.request_timeout_seconds),
        fallback_base: format!("http://127.0.0.1:{}", config.fallback_port),
        pending_settlements: pending_settlements.clone(),
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
    let server = axum::serve(listener, app).with_graceful_shutdown(shutdown_signal());
    // A signal starts the graceful drain above; the arm below bounds it, so a
    // stuck stream cannot hold shutdown past the configured timeout.
    let outcome = tokio::select! {
        outcome = server => outcome.map_err(|error| format!("gateway server failed: {error}")),
        _ = async {
            shutdown_signal().await;
            tokio::time::sleep(graceful).await;
        } => Ok(()),
    };
    // Terminal accounting writes spawned by disconnect guards must land
    // before the runtime is dropped; bound the wait by the graceful timeout.
    let drain_deadline = Instant::now() + graceful;
    while pending_settlements.load(Ordering::SeqCst) > 0 && Instant::now() < drain_deadline {
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    outcome
}

/// Resolve on SIGINT or SIGTERM, the process-manager stop signals.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut terminate =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(stream) => stream,
                Err(_) => {
                    let _ = tokio::signal::ctrl_c().await;
                    return;
                }
            };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

fn error_response(error: &PublicError) -> Response {
    let mut builder = Response::builder()
        .status(
            StatusCode::from_u16(error.status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        )
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
    let key = value
        .strip_prefix("Bearer ")
        .ok_or_else(PublicError::invalid_key)?;
    let trimmed = key.trim();
    if trimmed.is_empty() {
        return Err(PublicError::invalid_key());
    }
    Ok(trimmed.to_string())
}

/// Read one request body under the shared explicit cap.
async fn read_body(body: Body) -> Result<Bytes, PublicError> {
    axum::body::to_bytes(body, MAXIMUM_REQUEST_BODY_BYTES)
        .await
        .map_err(|_| PublicError::request_too_large())
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
/// implement (Responses, replay-keyed chat, escalated aliases, unknown routes).
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
            "host"
                | "connection"
                | "content-length"
                | "transfer-encoding"
                | "keep-alive"
                | "te"
                | "trailer"
                | "upgrade"
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
    let bytes = match read_body(body).await {
        Ok(bytes) => bytes,
        Err(error) => return error_response(&error),
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
/// Replay-keyed requests are proxied, so no client request identity exists
/// on the native path.
fn commit_independent(admission: &Admission) -> Vec<(String, String)> {
    vec![
        ("x-request-id".to_string(), admission.request_id.clone()),
        ("x-gateway-alias".to_string(), admission.alias.clone()),
        (
            "x-gateway-alias-revision".to_string(),
            admission.alias_revision_id.clone(),
        ),
    ]
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
    request_id: &str,
    attempt_id: &str,
    outcome: &str,
    usage: Option<&Usage>,
    tool_names: &[String],
    failure: Option<&Failure>,
) -> String {
    compact_json(&json!({
        "request_id": request_id,
        "attempt_id": attempt_id,
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

/// Deliver one settlement with bounded backoff; the control plane keeps the
/// in-flight entry on a failed terminal write, so retries can still land. A
/// persistent failure stays latched as accounting-unhealthy control-plane
/// side and is reconciled at the next startup.
async fn deliver_settlement(bridge: &Bridge, argument: String) -> bool {
    for backoff_ms in [0u64, 100, 500, 2_000] {
        if backoff_ms > 0 {
            tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
        }
        if bridge.call("settle", argument.clone()).await.is_ok() {
            return true;
        }
    }
    // The control plane keeps the in-flight entry; its sweep keeps retrying
    // and latches readiness if the loss is durable. Leave an operator signal.
    eprintln!("exp-gateway-native: settlement not yet durable after retries: {argument}");
    false
}

/// Exactly-once settlement owner for one admitted attempt.
///
/// Every admitted request settles through this guard. If the owning future is
/// dropped before an explicit settlement lands (client disconnect cancels the
/// handler, a panic unwinds the stream task), `Drop` spawns a cancellation
/// settlement so the ledger row and its budget reservation are always closed.
struct AttemptGuard {
    bridge: Arc<Bridge>,
    request_id: String,
    attempt_id: String,
    pending: Arc<AtomicUsize>,
    armed: bool,
}

impl AttemptGuard {
    fn new(state: &AppState, request_id: String, attempt_id: String) -> Self {
        Self {
            bridge: state.bridge.clone(),
            request_id,
            attempt_id,
            pending: state.pending_settlements.clone(),
            armed: true,
        }
    }

    /// Durably settle this attempt; disarms the drop backstop afterwards.
    /// Returns whether the terminal write reached the ledger.
    async fn settle(
        &mut self,
        outcome: &str,
        usage: Option<&Usage>,
        tool_names: &[String],
        failure: Option<&Failure>,
    ) -> bool {
        let argument = settle_argument(
            &self.request_id,
            &self.attempt_id,
            outcome,
            usage,
            tool_names,
            failure,
        );
        let delivered = deliver_settlement(&self.bridge, argument).await;
        self.armed = false;
        delivered
    }

    async fn settle_cancelled(&mut self, usage: Option<&Usage>, tool_names: &[String]) -> bool {
        self.settle(
            "failed",
            usage,
            tool_names,
            Some(&Failure::new(
                FailureClass::Cancelled,
                "gateway request was cancelled",
            )),
        )
        .await
    }
}

impl Drop for AttemptGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            // Runtime teardown; startup reconciliation closes the row.
            return;
        };
        let bridge = self.bridge.clone();
        let pending = self.pending.clone();
        pending.fetch_add(1, Ordering::SeqCst);
        let argument = settle_argument(
            &self.request_id,
            &self.attempt_id,
            "failed",
            None,
            &[],
            Some(&Failure::new(
                FailureClass::Cancelled,
                "gateway request was cancelled",
            )),
        );
        handle.spawn(async move {
            deliver_settlement(&bridge, argument).await;
            pending.fetch_sub(1, Ordering::SeqCst);
        });
    }
}

async fn chat(State(state): State<AppState>, request: axum::extract::Request) -> Response {
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

    // Replay-keyed chat keeps the python engine's idempotency semantics.
    // Presence is checked on the raw header map so a non-UTF8 value still
    // escalates instead of silently dropping replay behavior.
    if headers.contains_key("idempotency-key") || headers.contains_key("x-client-request-id") {
        return proxy_to_python(
            &state,
            reqwest::Method::POST,
            "/v1/chat/completions",
            &headers,
            body,
        )
        .await;
    }
    let body_text = match String::from_utf8(body.to_vec()) {
        Ok(text) => text,
        Err(_) => return error_response(&PublicError::invalid_json()),
    };

    let admit_argument = compact_json(&json!({"raw_key": raw_key, "body": body_text}));
    let admission_text = match state.bridge.call("admit", admit_argument).await {
        Ok(text) => text,
        Err(error) => return error_response(&error),
    };
    let admission_value: Value = match serde_json::from_str(&admission_text) {
        Ok(value) => value,
        Err(_) => return error_response(&PublicError::internal()),
    };
    if admission_value.get("escalate").is_some() {
        // No ledger row exists; the python engine owns this request end to end.
        return proxy_to_python(
            &state,
            reqwest::Method::POST,
            "/v1/chat/completions",
            &headers,
            body,
        )
        .await;
    }
    let admission: Admission = match serde_json::from_value(admission_value.clone()) {
        Ok(admission) => admission,
        Err(_) => {
            // The attempt is durably started; close it before failing so
            // wire-contract drift cannot leak ledger rows.
            let request_id = admission_value
                .get("request_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let attempt_id = admission_value
                .get("attempt_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            if !request_id.is_empty() && !attempt_id.is_empty() {
                let mut guard = AttemptGuard::new(&state, request_id, attempt_id);
                guard
                    .settle(
                        "failed",
                        None,
                        &[],
                        Some(&Failure::new(
                            FailureClass::Internal,
                            "gateway admission wire contract failed",
                        )),
                    )
                    .await;
            }
            return error_response(&PublicError::internal());
        }
    };
    let mut guard = AttemptGuard::new(
        &state,
        admission.request_id.clone(),
        admission.attempt_id.clone(),
    );

    let dialect = match Dialect::from_str(&admission.dialect) {
        Some(dialect) => dialect,
        None => {
            guard
                .settle(
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

    // The connection's raw timeout bounds each transport phase (open, then
    // every chunk read), exactly like the python streaming path.
    let phase_timeout = Duration::from_secs_f64(admission.timeout_seconds.max(0.001));

    // The bounded active-dispatch permit is awaited after admission, like the
    // python executor: protocol and authority errors answer immediately even
    // at capacity, and a queue-deadline expiry settles the started attempt.
    let permit =
        match tokio::time::timeout_at(deadline.into(), state.permits.clone().acquire_owned()).await
        {
            Ok(Ok(permit)) => permit,
            Ok(Err(_)) => {
                guard
                    .settle(
                        "failed",
                        None,
                        &[],
                        Some(&Failure::new(
                            FailureClass::Cancelled,
                            "gateway is draining and is not accepting new requests",
                        )),
                    )
                    .await;
                return error_response(&PublicError::draining());
            }
            Err(_) => {
                let failure = Failure::new(
                    FailureClass::Timeout,
                    "gateway execution queue deadline exceeded",
                );
                let error = failure.public_error();
                guard.settle("failed", None, &[], Some(&failure)).await;
                return error_response(&error);
            }
        };

    // One bounded same-deployment retry at the open phase, mirroring the
    // python executor's retry policy before any byte reaches the client.
    let mut response = None;
    for attempt in 0..2u8 {
        let open_bound = remaining(deadline).min(phase_timeout);
        match open_stream(
            &state.http,
            &admission.url,
            &admission.headers,
            &admission.idempotency_key,
            &admission.upstream_payload,
            open_bound,
        )
        .await
        {
            Ok(opened) => {
                response = Some(opened);
                break;
            }
            Err(transport) => {
                if attempt == 0
                    && transport.retryable_same_deployment
                    && !remaining(deadline).is_zero()
                {
                    continue;
                }
                let failure = transport.failure;
                let error = failure.public_error();
                guard.settle("failed", None, &[], Some(&failure)).await;
                return error_response(&error);
            }
        }
    }
    let response = match response {
        Some(response) => response,
        None => {
            let failure = Failure::new(FailureClass::Internal, "provider dispatch failed");
            guard.settle("failed", None, &[], Some(&failure)).await;
            return error_response(&PublicError::internal());
        }
    };

    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or(0);

    if admission.stream {
        stream_response(
            admission,
            guard,
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
            admission,
            guard,
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

/// Deliver one public frame bounded by the request deadline, so a connected
/// client that stops reading cannot pin the admission permit and the upstream
/// connection forever. `false` means the client is gone or out of time.
async fn send_bounded(
    sender: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    deadline: Instant,
    data: Bytes,
) -> bool {
    let bound = remaining(deadline);
    if bound.is_zero() {
        return false;
    }
    matches!(
        tokio::time::timeout(bound, sender.send(Ok(data))).await,
        Ok(Ok(()))
    )
}

/// Classify one mid-stream chunk timeout the way the python transport does:
/// a stalled provider read is a transport failure unless the request's own
/// deadline is exhausted.
fn stream_timeout_failure(deadline: Instant) -> Failure {
    if remaining(deadline).is_zero() {
        Failure::new(FailureClass::Timeout, "gateway execution deadline exceeded")
    } else {
        Failure::new(
            FailureClass::Transport,
            "provider transport failed; retry the request",
        )
    }
}

/// Approximate retained size of one aggregated event, in bytes. Completed
/// tool calls charge their full argument text, matching the python engine's
/// bounded aggregation, which also charges the completed call after its
/// streamed deltas.
fn event_retained_bytes(event: &Event) -> usize {
    match event {
        Event::TextDelta(text) | Event::RefusalDelta(text) => text.len(),
        Event::ToolArgumentsDelta { delta, .. } => delta.len(),
        Event::ToolCallCompleted { call, .. } => call.raw_arguments.len().max(64),
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
    let mut retain = |events: &mut Vec<Event>, event: Event| -> Result<(), Failure> {
        retained_bytes = retained_bytes.saturating_add(event_retained_bytes(&event));
        if retained_bytes > MAXIMUM_RETAINED_OUTPUT_BYTES {
            return Err(Failure::new(
                FailureClass::ProviderInternal,
                OUTPUT_OVERFLOW_MESSAGE,
            ));
        }
        events.push(event);
        Ok(())
    };
    let mut byte_stream = response.bytes_stream();
    loop {
        let bound = remaining(deadline).min(phase_timeout);
        let chunk = match tokio::time::timeout(bound, byte_stream.next()).await {
            Ok(Some(Ok(chunk))) => chunk,
            Ok(Some(Err(_))) => {
                return Err(Failure::new(
                    FailureClass::Transport,
                    "provider transport failed; retry the request",
                ))
            }
            Ok(None) => break,
            Err(_) => return Err(stream_timeout_failure(deadline)),
        };
        let frames = decoder
            .feed(&chunk)
            .map_err(|message| Failure::new(FailureClass::MalformedResponse, &message))?;
        for frame in frames {
            for event in normalizer.feed(&frame)? {
                retain(&mut events, event)?;
            }
            if normalizer.saw_terminal() {
                return Ok(events);
            }
        }
    }
    // Recover a final unterminated SSE frame at EOF, exactly like the python
    // decoder, so a provider that omits the closing blank line still settles
    // by its terminal event.
    if let Some(frame) = decoder
        .finish()
        .map_err(|message| Failure::new(FailureClass::MalformedResponse, &message))?
    {
        for event in normalizer.feed(&frame)? {
            retain(&mut events, event)?;
        }
        if normalizer.saw_terminal() {
            return Ok(events);
        }
    }
    normalizer.stream_ended()?;
    Ok(events)
}

#[allow(clippy::too_many_arguments)]
async fn completed_response(
    admission: Admission,
    mut guard: AttemptGuard,
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
            let failure = failure.boundary();
            let error = collection_public_error(&failure);
            guard.settle("failed", None, &[], Some(&failure)).await;
            return error_response(&error);
        }
    };
    let aggregated =
        match completed_chat_body(&admission.request_id, &admission.alias, created_at, &events) {
            Ok(aggregated) => aggregated,
            Err(error) => {
                guard
                    .settle(
                        "failed",
                        None,
                        &[],
                        Some(
                            &Failure::new(
                                FailureClass::MalformedResponse,
                                "provider stream ended without a terminal event",
                            )
                            .boundary(),
                        ),
                    )
                    .await;
                return error_response(&error);
            }
        };
    if let Some(failure) = &aggregated.failure {
        let failure = failure.clone().boundary();
        let error = failure.public_error();
        guard
            .settle(
                "failed",
                aggregated.usage.as_ref(),
                &aggregated.tool_names,
                Some(&failure),
            )
            .await;
        return error_response(&error);
    }
    let outcome = if aggregated.incomplete {
        "incomplete"
    } else {
        "completed"
    };
    let settled = guard
        .settle(
            outcome,
            aggregated.usage.as_ref(),
            &aggregated.tool_names,
            None,
        )
        .await;
    if !settled {
        // Success is only reported once the terminal accounting write landed.
        return error_response(&PublicError::internal());
    }
    let mut headers = commit_independent(&admission);
    headers.extend(commit_dependent(&admission));
    json_response(StatusCode::OK, &aggregated.body, &headers)
}

#[allow(clippy::too_many_arguments)]
async fn stream_response(
    admission: Admission,
    guard: AttemptGuard,
    dialect: Dialect,
    response: reqwest::Response,
    created_at: i64,
    deadline: Instant,
    phase_timeout: Duration,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    let (sender, receiver) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    let mut header_pairs = commit_independent(&admission);
    header_pairs.extend(commit_dependent(&admission));
    let include_usage = admission.include_usage;
    let request_id = admission.request_id.clone();
    let alias = admission.alias.clone();
    tokio::spawn(async move {
        let _permit = permit;
        let mut guard = guard;
        let mut encoder = ChatSseEncoder::new(&request_id, &alias, created_at, include_usage);
        let mut normalizer = Normalizer::new(dialect);
        let mut decoder = SseDecoder::new();
        let mut usage: Option<Usage> = None;
        let mut tool_names: Vec<String> = Vec::new();
        let mut terminal: Option<Event> = None;

        macro_rules! fail_stream {
            ($failure:expr) => {{
                let failure = $failure.boundary();
                emit_failure(&sender, deadline, &mut encoder, &failure).await;
                guard
                    .settle("failed", usage.as_ref(), &tool_names, Some(&failure))
                    .await;
                return;
            }};
        }

        let start_frames = match encoder.start() {
            Ok(frames) => frames,
            Err(_) => {
                fail_stream!(Failure::new(
                    FailureClass::Internal,
                    "gateway could not encode the provider stream",
                ))
            }
        };
        for frame in start_frames {
            if !send_bounded(&sender, deadline, Bytes::from(frame)).await {
                guard.settle_cancelled(usage.as_ref(), &tool_names).await;
                return;
            }
        }

        let mut byte_stream = response.bytes_stream();
        'outer: loop {
            let bound = remaining(deadline).min(phase_timeout);
            let chunk = match tokio::time::timeout(bound, byte_stream.next()).await {
                Ok(Some(Ok(chunk))) => chunk,
                Ok(Some(Err(_))) => {
                    fail_stream!(Failure::new(
                        FailureClass::Transport,
                        "provider transport failed; retry the request",
                    ))
                }
                Ok(None) => break 'outer,
                Err(_) => fail_stream!(stream_timeout_failure(deadline)),
            };
            let frames = match decoder.feed(&chunk) {
                Ok(frames) => frames,
                Err(message) => {
                    fail_stream!(Failure::new(FailureClass::MalformedResponse, &message))
                }
            };
            for frame in frames {
                let events = match normalizer.feed(&frame) {
                    Ok(events) => events,
                    Err(failure) => fail_stream!(failure),
                };
                for event in events {
                    track_event(&event, &mut usage, &mut tool_names);
                    // The terminal is recorded before its frames flush, so a
                    // disconnect during the final flush still settles by the
                    // provider's outcome instead of as a cancellation.
                    if event.is_terminal() {
                        terminal = Some(event.clone());
                    }
                    let encoded = match encoder.feed(&event) {
                        Ok(encoded) => encoded,
                        Err(_) => {
                            fail_stream!(Failure::new(
                                FailureClass::Internal,
                                "gateway could not encode the provider stream",
                            ))
                        }
                    };
                    for data in encoded {
                        if !send_bounded(&sender, deadline, Bytes::from(data)).await {
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
                    if terminal.is_some() {
                        break 'outer;
                    }
                }
            }
        }

        // Recover a final unterminated SSE frame at EOF (python decoder
        // parity) before deciding the stream ended without a terminal.
        if terminal.is_none() {
            let tail = match decoder.finish() {
                Ok(tail) => tail,
                Err(message) => {
                    fail_stream!(Failure::new(FailureClass::MalformedResponse, &message))
                }
            };
            if let Some(frame) = tail {
                let events = match normalizer.feed(&frame) {
                    Ok(events) => events,
                    Err(failure) => fail_stream!(failure),
                };
                for event in events {
                    track_event(&event, &mut usage, &mut tool_names);
                    if event.is_terminal() {
                        terminal = Some(event.clone());
                    }
                    let encoded = match encoder.feed(&event) {
                        Ok(encoded) => encoded,
                        Err(_) => {
                            fail_stream!(Failure::new(
                                FailureClass::Internal,
                                "gateway could not encode the provider stream",
                            ))
                        }
                    };
                    for data in encoded {
                        if !send_bounded(&sender, deadline, Bytes::from(data)).await {
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
                }
            }
        }

        if terminal.is_none() {
            let failure = Failure::new(
                FailureClass::MalformedResponse,
                "provider stream ended without a terminal event",
            )
            .boundary();
            emit_failure(&sender, deadline, &mut encoder, &failure).await;
            guard
                .settle("failed", usage.as_ref(), &tool_names, Some(&failure))
                .await;
            return;
        }
        settle_stream_end(
            &mut guard,
            terminal.as_ref(),
            usage.as_ref(),
            &tool_names,
            false,
        )
        .await;
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
    builder
        .body(body)
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

/// Settle a stream that reached its end (or lost its client). With a terminal
/// observed, the provider's outcome wins even on disconnect; without one, a
/// disconnect settles as cancelled.
async fn settle_stream_end(
    guard: &mut AttemptGuard,
    terminal: Option<&Event>,
    usage: Option<&Usage>,
    tool_names: &[String],
    disconnected: bool,
) {
    match terminal {
        Some(Event::Failed(failure)) => {
            let failure = failure.clone().boundary();
            guard
                .settle("failed", usage, tool_names, Some(&failure))
                .await;
        }
        Some(Event::Incomplete) => {
            guard.settle("incomplete", usage, tool_names, None).await;
        }
        Some(_) => {
            guard.settle("completed", usage, tool_names, None).await;
        }
        None => {
            if disconnected {
                guard.settle_cancelled(usage, tool_names).await;
            }
        }
    }
}

fn track_event(event: &Event, usage: &mut Option<Usage>, tool_names: &mut Vec<String>) {
    match event {
        Event::Usage(candidate) if candidate.has_token_counts() => {
            *usage = Some(candidate.clone());
        }
        Event::ToolCallCompleted { call, .. } if !tool_names.contains(&call.name) => {
            tool_names.push(call.name.clone());
        }
        _ => {}
    }
}

/// Emit the encoder's sanitized failure frame and done sentinel when the
/// stream has not already reached a terminal.
async fn emit_failure(
    sender: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    deadline: Instant,
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
        if !send_bounded(sender, deadline, Bytes::from(frame)).await {
            return;
        }
    }
}
