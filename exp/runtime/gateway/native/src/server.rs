//! The axum data plane: routes, admission, waterfall execution, settlement.

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
use crate::dialects::MAXIMUM_RETAINED_OUTPUT_BYTES;
use crate::encode::{chat_data, compact_json, completed_chat_body, ChatSseEncoder};
use crate::encode_responses::{completed_responses_body, ResponsesEnvelope, ResponsesSseEncoder};
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{CompletedToolCall, Event, Usage};
use crate::guardrails;
use crate::metrics::{classify_escalation, METRICS};
use crate::replay::{CachedResponse, Claim, OwnerLease, ReplayKey, ReplayStore};
use crate::waterfall::{
    acquire_attempt, collect_committed, collection_public_error, event_retained_bytes, remaining,
    track_event, AttemptGuard, CommittedAttempt, DeploymentWire, RoutePolicy, SettledAttempt,
    WaterfallContext, Won,
};

/// Largest accepted request body on every native-served or proxied route.
/// Bounded so one client cannot hold unbounded gateway memory; far above any
/// real chat history (the python engine imposes no explicit cap).
const MAXIMUM_REQUEST_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Largest SSE capture retained for keyed-stream replay, matching the python
/// engine's `_STREAM_REPLAY_CAPTURE_BYTES`.
const STREAM_REPLAY_CAPTURE_BYTES: usize = 64 * 1024 * 1024;

/// Serve-time configuration passed from `exp --engine rust`.
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
    #[serde(default = "default_native_usage_enabled")]
    pub native_usage_enabled: bool,
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

fn default_native_usage_enabled() -> bool {
    true
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
    /// Requests handled since start; the idle reclaim loop trims the
    /// allocator once per burst when this advances and the plane is idle.
    handled_requests: Arc<AtomicUsize>,
    /// Proxied requests still relaying to or from the python engine. Proxy
    /// traffic holds no active-request permit, so the reclaim loop needs
    /// this separate in-flight signal to avoid trimming under proxy load.
    active_proxies: Arc<AtomicUsize>,
    /// Bounded in-process keyed-response replay, the native mirror of the
    /// python engine's `BoundedReplayStore`.
    replays: Arc<ReplayStore>,
}

/// Holds one in-flight proxy count from admission until the relayed
/// response body is fully forwarded or dropped.
struct ProxyGuard(Arc<AtomicUsize>);

impl ProxyGuard {
    /// Count one proxied request as in flight.
    fn new(counter: Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::SeqCst);
        METRICS.record_proxied();
        METRICS.enter_proxy();
        Self(counter)
    }
}

impl Drop for ProxyGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
        METRICS.exit_proxy();
    }
}

/// The wire configuration returned by one successful admission: the full
/// ordered certified route (one wire configuration per deployment, each with
/// its payload fully built by the shared python dialect builders) plus the
/// frozen retry-policy facts. No attempt is started at admission; each
/// physical dispatch is reserved through the `start_attempt` callback.
#[derive(Debug, Clone, Deserialize)]
struct Admission {
    request_id: String,
    alias: String,
    alias_revision_id: String,
    stream: bool,
    include_usage: bool,
    exact_model_id: String,
    route_reason: String,
    route: Vec<DeploymentWire>,
    maximum_total_attempts: u32,
    maximum_same_deployment_attempts: u32,
    #[serde(default)]
    refusal_failover: bool,
    /// Responses-only request-reflecting envelope fields; chat admissions
    /// omit it.
    #[serde(default)]
    envelope: Option<ResponsesEnvelope>,
    /// When true, buffer the winning completion and call `enforce_output` once
    /// before any caller byte or replay retention. Unguarded admissions omit
    /// the flag (default false) and never invoke that callback.
    #[serde(default)]
    output_guardrail: bool,
}

impl Admission {
    fn policy(&self) -> RoutePolicy {
        RoutePolicy {
            maximum_total_attempts: self.maximum_total_attempts.max(1),
            maximum_same_deployment_attempts: self.maximum_same_deployment_attempts.max(1),
            refusal_failover: self.refusal_failover,
        }
    }

    /// The per-chunk transport bound of the deployment serving `depth`.
    fn phase_timeout(&self, depth: usize) -> Duration {
        let seconds = self
            .route
            .get(depth)
            .map(|wire| wire.timeout_seconds)
            .unwrap_or(60.0);
        Duration::from_secs_f64(seconds.max(0.001))
    }
}

/// Run the data plane until shutdown; returns after graceful stop.
pub async fn run(bridge: Arc<Bridge>, config: ServeConfig) -> Result<(), String> {
    let http = crate::upstream::build_client()?;
    let pending_settlements = Arc::new(AtomicUsize::new(0));
    let max_active_requests = config.max_active_requests.max(1);
    let handled_requests = Arc::new(AtomicUsize::new(0));
    let active_proxies = Arc::new(AtomicUsize::new(0));
    let state = AppState {
        bridge,
        http,
        permits: Arc::new(Semaphore::new(max_active_requests)),
        request_timeout: Duration::from_secs_f64(config.request_timeout_seconds),
        fallback_base: format!("http://127.0.0.1:{}", config.fallback_port),
        pending_settlements: pending_settlements.clone(),
        handled_requests: handled_requests.clone(),
        active_proxies: active_proxies.clone(),
        replays: Arc::new(ReplayStore::new()),
    };
    tokio::spawn(crate::memory::reclaim_when_idle(
        state.permits.clone(),
        max_active_requests,
        handled_requests,
        pending_settlements.clone(),
        active_proxies,
    ));
    let app = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/models/{model_id}", get(model_detail))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/responses", post(responses))
        .route("/health/live", get(health_live))
        .route("/health/ready", get(health_ready))
        .route("/metrics.json", get(metrics_json))
        .route("/metrics", get(metrics_text));
    let app = if config.native_usage_enabled {
        app.route("/usage.json", get(usage_json).fallback(proxy_fallback))
            .route("/usage", get(usage_page).fallback(proxy_fallback))
    } else {
        app
    };
    let app = app.fallback(proxy_fallback).with_state(state);
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

/// Build the usage callback argument: an anonymous request reads the
/// organization-wide report, a Bearer key scopes it to the key's identity.
fn usage_argument(headers: &HeaderMap) -> Result<String, PublicError> {
    if headers.get(header::AUTHORIZATION).is_none() {
        return Ok("{}".to_string());
    }
    let raw_key = bearer_key(headers)?;
    Ok(compact_json(&json!({"raw_key": raw_key})))
}

async fn usage_json(State(state): State<AppState>, headers: HeaderMap) -> Response {
    let argument = match usage_argument(&headers) {
        Ok(argument) => argument,
        Err(error) => return error_response(&error),
    };
    match state.bridge.call("usage_json", argument).await {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(payload) => json_response(StatusCode::OK, &payload, &[]),
            Err(_) => error_response(&PublicError::internal()),
        },
        Err(error) => error_response(&error),
    }
}

async fn usage_page(State(state): State<AppState>, headers: HeaderMap) -> Response {
    let argument = match usage_argument(&headers) {
        Ok(argument) => argument,
        Err(error) => return error_response(&error),
    };
    match state.bridge.call("usage_page", argument).await {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(payload) => match payload.get("html").and_then(Value::as_str) {
                Some(html) => Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
                    .body(Body::from(html.to_string()))
                    .unwrap_or_else(|_| Response::new(Body::empty())),
                None => error_response(&PublicError::internal()),
            },
            Err(_) => error_response(&PublicError::internal()),
        },
        Err(error) => error_response(&error),
    }
}

/// Serve the content-free metrics snapshot. The control plane composes the
/// data-plane registry with its own sweep counters so this body and the
/// programmatic python snapshot are one and the same.
async fn metrics_json(State(state): State<AppState>) -> Response {
    match state.bridge.call("metrics_json", "{}".to_string()).await {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(payload) => json_response(StatusCode::OK, &payload, &[]),
            Err(_) => error_response(&PublicError::internal()),
        },
        Err(error) => error_response(&error),
    }
}

/// Serve the same content-free snapshot rendered in the Prometheus text
/// exposition format. The control plane renders the body; this handler only
/// unwraps the `text` field and stamps the exposition content type.
async fn metrics_text(State(state): State<AppState>) -> Response {
    match state.bridge.call("metrics_text", "{}".to_string()).await {
        Ok(text) => match serde_json::from_str::<Value>(&text) {
            Ok(payload) => match payload.get("text").and_then(Value::as_str) {
                Some(exposition) => Response::builder()
                    .status(StatusCode::OK)
                    .header(
                        header::CONTENT_TYPE,
                        "text/plain; version=0.0.4; charset=utf-8",
                    )
                    .body(Body::from(exposition.to_string()))
                    .unwrap_or_else(|_| Response::new(Body::empty())),
                None => error_response(&PublicError::internal()),
            },
            Err(_) => error_response(&PublicError::internal()),
        },
        Err(error) => error_response(&error),
    }
}

/// Replay one HTTP request against the embedded python engine and stream the
/// response back unchanged. Serves every surface the native plane does not
/// implement (replay-keyed Responses, escalated aliases, unknown routes).
async fn proxy_to_python(
    state: &AppState,
    method: reqwest::Method,
    path_and_query: &str,
    headers: &HeaderMap,
    body: Bytes,
) -> Response {
    let guard = ProxyGuard::new(state.active_proxies.clone());
    let url = format!("{}{}", state.fallback_base, path_and_query);
    // Connect failures are retried a bounded number of times: nothing has been
    // written to the python engine yet, so a replay cannot double-execute, and
    // a transient accept-queue overflow under concurrent load must not surface
    // as a caller-visible 502.
    let mut upstream = None;
    for attempt in 0..3u8 {
        let mut request = state.http.request(method.clone(), url.clone());
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
        match request.body(body.clone()).send().await {
            Ok(response) => {
                upstream = Some(response);
                break;
            }
            Err(error) => {
                if attempt < 2 && error.is_connect() {
                    tokio::time::sleep(Duration::from_millis(25 << attempt)).await;
                    continue;
                }
                METRICS.record_fallback_unavailable();
                return error_response(&PublicError::new(
                    502,
                    "fallback_engine_unavailable",
                    "The python fallback engine did not answer. Retry shortly; if this persists, restart the gateway.",
                    "api_error",
                ));
            }
        }
    }
    let Some(upstream) = upstream else {
        return error_response(&PublicError::internal());
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
    // The guard rides the relayed stream so the proxied request stays
    // counted until its body finishes or the client disconnects.
    let relayed = upstream.bytes_stream().map(move |chunk| {
        let _held = &guard;
        chunk
    });
    builder
        .body(Body::from_stream(relayed))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

/// Route every path the native plane does not own to the python engine.
async fn proxy_fallback(
    State(state): State<AppState>,
    request: axum::extract::Request,
) -> Response {
    state.handled_requests.fetch_add(1, Ordering::Relaxed);
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

/// Commit-independent headers, mirroring `commit_independent_headers`,
/// including the caller's echoed request identity when one was supplied.
fn commit_independent(
    admission: &Admission,
    client_request_id: Option<&str>,
) -> Vec<(String, String)> {
    let mut headers = vec![
        ("x-request-id".to_string(), admission.request_id.clone()),
        ("x-gateway-alias".to_string(), admission.alias.clone()),
        (
            "x-gateway-alias-revision".to_string(),
            admission.alias_revision_id.clone(),
        ),
    ];
    if let Some(value) = client_request_id {
        headers.push(("x-client-request-id".to_string(), value.to_string()));
    }
    headers
}

/// Read one header value as latin-1 text, the same byte-transparent decoding
/// the python engine's ASGI server applies, so any HTTP-legal value produces
/// the identical caller-operation string on both engines.
fn latin1_header(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .map(|value| value.as_bytes().iter().map(|&byte| byte as char).collect())
}

/// Re-encode one latin-1 decoded header value to its original bytes.
pub(crate) fn latin1_bytes(value: &str) -> Vec<u8> {
    value
        .chars()
        .map(|character| {
            let code = character as u32;
            if code < 256 {
                code as u8
            } else {
                b'?'
            }
        })
        .collect()
}

/// Build one exact HTTP response from a stored keyed result, mirroring the
/// python engine's `_cached_response`.
fn cached_response(cached: &CachedResponse) -> Response {
    let mut builder = Response::builder()
        .status(
            StatusCode::from_u16(cached.status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        )
        .header(header::CONTENT_TYPE, cached.media_type.as_str());
    for (name, value) in &cached.headers {
        if let (Ok(name), Ok(value)) = (
            header::HeaderName::try_from(name.as_str()),
            HeaderValue::from_bytes(&latin1_bytes(value)),
        ) {
            builder = builder.header(name, value);
        }
    }
    builder
        .body(Body::from(Bytes::copy_from_slice(&cached.body)))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

/// Append one frame while it remains within the replay capture ceiling,
/// mirroring the python engine's `capture_frame`.
fn capture_frame(buffer: &mut Vec<u8>, data: &[u8], replayable: bool) -> bool {
    if !replayable || buffer.len() + data.len() > STREAM_REPLAY_CAPTURE_BYTES {
        return false;
    }
    buffer.extend_from_slice(data);
    true
}

/// Commit-dependent headers, mirroring `commit_dependent_headers`: the
/// deployment identity and route depth that actually served the request.
fn commit_dependent(admission: &Admission, depth: usize) -> Vec<(String, String)> {
    let (provider, deployment_id) = admission
        .route
        .get(depth)
        .map(|wire| (wire.provider.clone(), wire.deployment_id.clone()))
        .unwrap_or_default();
    vec![
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
    ]
}

async fn chat(State(state): State<AppState>, request: axum::extract::Request) -> Response {
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

    // Replay-keyed chat runs the python engine's exact idempotency protocol
    // natively: the shared control plane computes the tenant-scoped replay
    // key (or escalates a request the native path cannot serve), then the
    // bounded replay store dedupes concurrent duplicates and replays the
    // owner's exact stored response. Headers are decoded latin-1 so any
    // HTTP-legal value matches the python engine's view byte for byte.
    let idempotency_key = latin1_header(&headers, "idempotency-key");
    let client_request_id = latin1_header(&headers, "x-client-request-id");
    let mut lease: Option<OwnerLease> = None;
    if idempotency_key.is_some() || client_request_id.is_some() {
        let scope_argument = compact_json(&json!({
            "raw_key": raw_key,
            "body": body_text,
            "idempotency_key": idempotency_key,
            "client_request_id": client_request_id,
        }));
        let scope_text = match state.bridge.call("claim_scope", scope_argument).await {
            Ok(text) => text,
            Err(error) => return error_response(&error),
        };
        let scope_value: Value = match serde_json::from_str(&scope_text) {
            Ok(value) => value,
            Err(_) => return error_response(&PublicError::internal()),
        };
        if let Some(reason) = scope_value.get("escalate") {
            METRICS.record_escalation(classify_escalation(reason.as_str().unwrap_or_default()));
            // No replay claim exists; the python engine owns this request
            // end to end, including its own replay store.
            return proxy_to_python(
                &state,
                reqwest::Method::POST,
                "/v1/chat/completions",
                &headers,
                body,
            )
            .await;
        }
        let key: ReplayKey = match serde_json::from_value(scope_value) {
            Ok(key) => key,
            Err(_) => return error_response(&PublicError::internal()),
        };
        match state.replays.claim(key).await {
            Err(error) => return error_response(&error),
            Ok(Claim::Replay(cached)) => return cached_response(&cached),
            Ok(Claim::Join(joiner)) => {
                // Joining never touches the ledger or budget: only the owner
                // accounts for the single provider call.
                return match joiner.result().await {
                    Ok(cached) => cached_response(&cached),
                    Err(error) => error_response(&error),
                };
            }
            Ok(Claim::Owner(owner)) => lease = Some(owner),
        }
    }

    let admit_argument = compact_json(&json!({
        "raw_key": raw_key,
        "body": body_text,
        "idempotency_key": idempotency_key,
        "client_request_id": client_request_id,
    }));
    let admission_text = match state.bridge.call("admit", admit_argument).await {
        Ok(text) => text,
        Err(error) => {
            // A failed keyed admission abandons ownership so waiting
            // duplicates fail closed instead of hanging.
            if let Some(mut owner) = lease.take() {
                owner.abandon().await;
            }
            return error_response(&error);
        }
    };
    let admission_value: Value = match serde_json::from_str(&admission_text) {
        Ok(value) => value,
        Err(_) => return error_response(&PublicError::internal()),
    };
    if let Some(reason) = admission_value.get("escalate") {
        METRICS.record_escalation(classify_escalation(reason.as_str().unwrap_or_default()));
        // No ledger row exists; the python engine owns this request end to end.
        if let Some(mut owner) = lease.take() {
            owner.abandon().await;
        }
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
            // The request is durably accepted; abandon it before failing so
            // wire-contract drift cannot leak an open request row.
            if let Some(mut owner) = lease.take() {
                owner.abandon().await;
            }
            return wire_drift_response(&state, &admission_value, started).await;
        }
    };
    let mut guard = new_guard(&state, admission.request_id.clone(), started);
    // The replay key was authorized independently of admission. If an alias
    // activation landed between the two, the admitted work belongs to a newer
    // revision than the claimed replay scope, so the request fails closed:
    // executing without ownership would let a concurrent duplicate own the
    // new revision's key and run the same keyed operation a second time.
    if lease
        .as_ref()
        .is_some_and(|owner| owner.alias_revision_id() != admission.alias_revision_id)
    {
        if let Some(mut owner) = lease.take() {
            owner.abandon().await;
        }
        guard
            .abandon(&Failure::new(
                FailureClass::Internal,
                "the alias revision changed during keyed admission",
            ))
            .await;
        let mut error = PublicError::new(
            409,
            "idempotency_replay_unavailable",
            "The alias revision changed while the keyed request was admitted. Retry the request.",
            "api_error",
        );
        error.param = Some("Idempotency-Key".to_string());
        return error_response(&error);
    }

    let permit = match acquire_permit(&state, &mut guard, deadline).await {
        Ok(permit) => permit,
        Err(response) => {
            if let Some(mut owner) = lease.take() {
                owner.abandon().await;
            }
            return *response;
        }
    };

    // Run the certified waterfall to its committed or terminal attempt.
    let context = WaterfallContext {
        bridge: &state.bridge,
        http: &state.http,
        request_id: &admission.request_id,
        raw_key: &raw_key,
        route: &admission.route,
        policy: admission.policy(),
        deadline,
    };
    let won = acquire_attempt(&context, &mut guard).await;

    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or(0);

    match won {
        Won::Failed(error) => {
            if let Some(mut owner) = lease.take() {
                owner.abandon().await;
            }
            error_response(&error)
        }
        Won::Settled(settled) => {
            settled_chat_response(&admission, settled, created_at, lease, client_request_id).await
        }
        Won::Committed(committed) => {
            let committed = *committed;
            if admission.output_guardrail {
                guarded_chat_response(
                    admission,
                    guard,
                    committed,
                    created_at,
                    deadline,
                    permit,
                    lease,
                    client_request_id,
                )
                .await
            } else if admission.stream {
                stream_response(
                    admission,
                    guard,
                    committed,
                    created_at,
                    deadline,
                    permit,
                    lease,
                    client_request_id,
                )
                .await
            } else {
                completed_response(
                    admission,
                    guard,
                    committed,
                    created_at,
                    deadline,
                    permit,
                    lease,
                    client_request_id,
                )
                .await
            }
        }
    }
}

/// Build one request guard bound to this server's settlement bookkeeping.
fn new_guard(state: &AppState, request_id: String, started: Instant) -> AttemptGuard {
    AttemptGuard::new(
        state.bridge.clone(),
        state.pending_settlements.clone(),
        request_id,
        started,
    )
}

/// Abandon one accepted request whose admission body failed to deserialize.
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
    error_response(&PublicError::internal())
}

/// Wait for one bounded active-dispatch permit after admission, like the
/// python executor: protocol and authority errors answer immediately even at
/// capacity, and a queue-deadline expiry terminalizes the accepted request.
async fn acquire_permit(
    state: &AppState,
    guard: &mut AttemptGuard,
    deadline: Instant,
) -> Result<tokio::sync::OwnedSemaphorePermit, Box<Response>> {
    let permit_wait_started = Instant::now();
    match tokio::time::timeout_at(deadline.into(), state.permits.clone().acquire_owned()).await {
        Ok(Ok(permit)) => {
            METRICS.permit_wait_ms.record(permit_wait_started.elapsed());
            Ok(permit)
        }
        Ok(Err(_)) => {
            guard
                .abandon(&Failure::new(
                    FailureClass::Cancelled,
                    "gateway is draining and is not accepting new requests",
                ))
                .await;
            Err(Box::new(error_response(&PublicError::draining())))
        }
        Err(_) => {
            let failure = Failure::new(
                FailureClass::Timeout,
                "gateway execution queue deadline exceeded",
            );
            let error = failure.public_error();
            guard.abandon(&failure).await;
            Err(Box::new(error_response(&error)))
        }
    }
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

/// Replace a typed refusal terminal with a public completion when refusal
/// output already reached (or is reaching) the caller, mirroring the python
/// executor's committed-refusal rule. Returns the recorded refusal failure.
fn complete_visible_refusal(events: &mut [Event]) -> Option<Failure> {
    let visible = events
        .iter()
        .any(|event| matches!(event, Event::RefusalDelta(_)));
    if !visible {
        return None;
    }
    if let Some(last) = events.last_mut() {
        if let Event::Failed(failure) = last {
            if failure.failure_class == FailureClass::Refusal {
                let failure = failure.clone();
                *last = Event::Completed;
                return Some(failure);
            }
        }
    }
    None
}

/// Answer one attempt that the waterfall already settled: a successful
/// terminal with no semantic output, or an exhausted ladder flushing its
/// bounded withheld refusal output ahead of the failing terminal.
async fn settled_chat_response(
    admission: &Admission,
    settled: SettledAttempt,
    created_at: i64,
    mut lease: Option<OwnerLease>,
    client_request_id: Option<String>,
) -> Response {
    let mut events = settled.events;
    let refusal_completed = complete_visible_refusal(&mut events);
    if refusal_completed.is_none() {
        if let Some(Event::Failed(failure)) = events.last() {
            let error = collection_public_error(&failure.clone().boundary());
            if admission.stream {
                // The withheld refusal output and its failing terminal flush
                // outward as the stream's only frames.
                let body = match encode_chat_sse(admission, created_at, &events) {
                    Ok(body) => body,
                    Err(error) => return error_response(&error),
                };
                let mut headers = commit_independent(admission, client_request_id.as_deref());
                headers.extend(commit_dependent(admission, settled.depth));
                if let Some(mut owner) = lease.take() {
                    owner.abandon().await;
                }
                return sse_body_response(&headers, body);
            }
            if let Some(mut owner) = lease.take() {
                owner.abandon().await;
            }
            return error_response(&error);
        }
    }
    let mut headers = commit_independent(admission, client_request_id.as_deref());
    headers.extend(commit_dependent(admission, settled.depth));
    if admission.stream {
        let body = match encode_chat_sse(admission, created_at, &events) {
            Ok(body) => body,
            Err(error) => return error_response(&error),
        };
        if let Some(mut owner) = lease.take() {
            let mut sorted = headers.clone();
            sorted.sort();
            let cached = CachedResponse {
                status_code: 200,
                media_type: "text/event-stream".to_string(),
                headers: sorted,
                body: body.clone(),
            };
            return match owner.complete(cached.clone()).await {
                Ok(()) => cached_response(&cached),
                Err(error) => error_response(&error),
            };
        }
        return sse_body_response(&headers, body);
    }
    let aggregated =
        match completed_chat_body(&admission.request_id, &admission.alias, created_at, &events) {
            Ok(aggregated) => aggregated,
            Err(error) => return error_response(&error),
        };
    if let Some(failure) = &aggregated.failure {
        if let Some(mut owner) = lease.take() {
            owner.abandon().await;
        }
        return error_response(&failure.clone().boundary().public_error());
    }
    if let Some(mut owner) = lease.take() {
        let mut sorted = headers.clone();
        sorted.sort();
        let cached = CachedResponse {
            status_code: 200,
            media_type: "application/json".to_string(),
            headers: sorted,
            body: compact_json(&aggregated.body).into_bytes(),
        };
        return match owner.complete(cached.clone()).await {
            Ok(()) => cached_response(&cached),
            Err(error) => error_response(&error),
        };
    }
    json_response(StatusCode::OK, &aggregated.body, &headers)
}

async fn apply_output_guardrail(
    admission: &Admission,
    bridge: &Bridge,
    events: Vec<Event>,
) -> Result<Vec<Event>, Failure> {
    if !admission.output_guardrail {
        return Ok(events);
    }
    guardrails::enforce_collected_output(bridge, &admission.request_id, events).await
}

fn encode_chat_sse(
    admission: &Admission,
    created_at: i64,
    events: &[Event],
) -> Result<Vec<u8>, PublicError> {
    let mut encoder = ChatSseEncoder::new(
        &admission.request_id,
        &admission.alias,
        created_at,
        admission.include_usage,
    );
    let mut body = Vec::new();
    for frame in encoder.start().map_err(|_| PublicError::internal())? {
        body.extend_from_slice(frame.as_bytes());
    }
    for event in events {
        for frame in encoder.feed(event).map_err(|_| PublicError::internal())? {
            body.extend_from_slice(frame.as_bytes());
        }
    }
    Ok(body)
}

fn sse_body_response(headers: &[(String, String)], body: Vec<u8>) -> Response {
    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream");
    for (name, value) in headers {
        if let (Ok(header_name), Ok(header_value)) = (
            header::HeaderName::try_from(name.as_str()),
            HeaderValue::try_from(value.as_str()),
        ) {
            builder = builder.header(header_name, header_value);
        }
    }
    builder
        .body(Body::from(body))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

/// Aggregate one committed non-streaming or guarded chat attempt and answer
/// it, settling exactly once and publishing keyed results.
#[allow(clippy::too_many_arguments)]
async fn respond_from_chat_events(
    admission: Admission,
    mut guard: AttemptGuard,
    depth: usize,
    mut events: Vec<Event>,
    usage: Option<Usage>,
    tool_names: Vec<String>,
    created_at: i64,
    mut lease: Option<OwnerLease>,
    client_request_id: Option<String>,
    stream_body: bool,
) -> Response {
    let refusal_completed = complete_visible_refusal(&mut events);
    let aggregated =
        match completed_chat_body(&admission.request_id, &admission.alias, created_at, &events) {
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
                if let Some(mut owner) = lease.take() {
                    owner.abandon().await;
                }
                return error_response(&error);
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
        if let Some(mut owner) = lease.take() {
            owner.abandon().await;
        }
        return error_response(&error);
    }
    let settled = if let Some(refusal) = &refusal_completed {
        // The caller saw the refusal output, so the public result completes;
        // the ledger still records the provider's typed refusal.
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
        // Success is only reported once the terminal accounting write landed.
        if let Some(mut owner) = lease.take() {
            owner.abandon().await;
        }
        return error_response(&PublicError::internal());
    }
    let mut headers = commit_independent(&admission, client_request_id.as_deref());
    headers.extend(commit_dependent(&admission, depth));
    if stream_body {
        let body = match encode_chat_sse(&admission, created_at, &events) {
            Ok(body) => body,
            Err(error) => return error_response(&error),
        };
        if let Some(mut owner) = lease.take() {
            let mut sorted = headers.clone();
            sorted.sort();
            let cached = CachedResponse {
                status_code: 200,
                media_type: "text/event-stream".to_string(),
                headers: sorted,
                body: body.clone(),
            };
            return match owner.complete(cached.clone()).await {
                Ok(()) => cached_response(&cached),
                Err(error) => error_response(&error),
            };
        }
        return sse_body_response(&headers, body);
    }
    if let Some(mut owner) = lease.take() {
        // Publish the exact response body and headers, then answer from the
        // stored copy, matching the python engine's `_cached_response`.
        let mut sorted = headers.clone();
        sorted.sort();
        let cached = CachedResponse {
            status_code: 200,
            media_type: "application/json".to_string(),
            headers: sorted,
            body: compact_json(&aggregated.body).into_bytes(),
        };
        return match owner.complete(cached.clone()).await {
            Ok(()) => cached_response(&cached),
            Err(error) => error_response(&error),
        };
    }
    json_response(StatusCode::OK, &aggregated.body, &headers)
}

#[allow(clippy::too_many_arguments)]
async fn completed_response(
    admission: Admission,
    mut guard: AttemptGuard,
    mut committed: CommittedAttempt,
    created_at: i64,
    deadline: Instant,
    permit: tokio::sync::OwnedSemaphorePermit,
    mut lease: Option<OwnerLease>,
    client_request_id: Option<String>,
) -> Response {
    let _permit = permit;
    let phase_timeout = admission.phase_timeout(committed.depth);
    let events =
        match collect_committed(&mut committed, deadline, phase_timeout, guard.started).await {
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
                if let Some(mut owner) = lease.take() {
                    owner.abandon().await;
                }
                return error_response(&error);
            }
        };
    respond_from_chat_events(
        admission,
        guard,
        committed.depth,
        events,
        committed.usage,
        committed.tool_names,
        created_at,
        lease,
        client_request_id,
        false,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn guarded_chat_response(
    admission: Admission,
    mut guard: AttemptGuard,
    mut committed: CommittedAttempt,
    created_at: i64,
    deadline: Instant,
    permit: tokio::sync::OwnedSemaphorePermit,
    mut lease: Option<OwnerLease>,
    client_request_id: Option<String>,
) -> Response {
    let _permit = permit;
    let phase_timeout = admission.phase_timeout(committed.depth);
    let collected =
        match collect_committed(&mut committed, deadline, phase_timeout, guard.started).await {
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
                if let Some(mut owner) = lease.take() {
                    owner.abandon().await;
                }
                return error_response(&error);
            }
        };
    let events = match apply_output_guardrail(&admission, &guard.bridge, collected).await {
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
            if let Some(mut owner) = lease.take() {
                owner.abandon().await;
            }
            return error_response(&failure.public_error());
        }
    };
    let stream_body = admission.stream;
    respond_from_chat_events(
        admission,
        guard,
        committed.depth,
        events,
        committed.usage,
        committed.tool_names,
        created_at,
        lease,
        client_request_id,
        stream_body,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn stream_response(
    admission: Admission,
    guard: AttemptGuard,
    committed: CommittedAttempt,
    created_at: i64,
    deadline: Instant,
    permit: tokio::sync::OwnedSemaphorePermit,
    lease: Option<OwnerLease>,
    client_request_id: Option<String>,
) -> Response {
    let (sender, receiver) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    let mut header_pairs = commit_independent(&admission, client_request_id.as_deref());
    header_pairs.extend(commit_dependent(&admission, committed.depth));
    let include_usage = admission.include_usage;
    let request_id = admission.request_id.clone();
    let alias = admission.alias.clone();
    let phase_timeout = admission.phase_timeout(committed.depth);
    let cached_headers = {
        let mut sorted = header_pairs.clone();
        sorted.sort();
        sorted
    };
    let task_hold = guard.hold_task();
    tokio::spawn(async move {
        let _task = task_hold;
        let _permit = permit;
        let mut guard = guard;
        let mut lease = lease;
        let mut committed = committed;
        let mut encoder = ChatSseEncoder::new(&request_id, &alias, created_at, include_usage);
        let mut usage: Option<Usage> = committed.usage.take();
        let mut tool_names: Vec<String> = std::mem::take(&mut committed.tool_names);
        let mut visible_refusal = committed.visible_refusal;
        let mut terminal: Option<Event> = None;
        // Keyed streams capture every public frame so the owner can publish
        // the exact byte stream; terminal frames are withheld until that
        // publication succeeds, matching the python engine's `_stream_body`.
        let mut capture: Vec<u8> = Vec::new();
        let mut replayable = lease.is_some();

        macro_rules! fail_stream {
            ($failure:expr) => {{
                let failure = $failure.boundary();
                let frames = failure_frames(&mut encoder, &failure);
                guard
                    .settle("failed", usage.as_ref(), &tool_names, Some(&failure), true)
                    .await;
                finish_stream_terminal(
                    &sender,
                    deadline,
                    &mut lease,
                    replayable,
                    &mut capture,
                    &cached_headers,
                    frames,
                )
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
            let data = Bytes::from(frame);
            if lease.is_some() {
                replayable = capture_frame(&mut capture, &data, replayable);
            }
            if !send_bounded(&sender, deadline, data).await {
                guard.settle_cancelled(usage.as_ref(), &tool_names).await;
                return;
            }
        }

        let mut prefix: std::collections::VecDeque<Event> = committed.prefix.drain(..).collect();
        loop {
            let event = if let Some(event) = prefix.pop_front() {
                event
            } else {
                match committed
                    .relay
                    .next_event(deadline, phase_timeout, guard.started)
                    .await
                {
                    Ok(Some(event)) => event,
                    Ok(None) => {
                        fail_stream!(Failure::new(
                            FailureClass::MalformedResponse,
                            "provider stream ended without a terminal event",
                        ))
                    }
                    Err(failure) => fail_stream!(failure),
                }
            };
            track_event(&event, &mut usage, &mut tool_names);
            if matches!(event, Event::RefusalDelta(_)) {
                visible_refusal = true;
            }
            // A typed refusal after visible refusal output completes
            // publicly; the ledger still records the provider's refusal.
            let outward = match &event {
                Event::Failed(failure)
                    if failure.failure_class == FailureClass::Refusal && visible_refusal =>
                {
                    Event::Completed
                }
                other => other.clone(),
            };
            if event.is_terminal() {
                terminal = Some(event.clone());
                if !settle_stream_end(&mut guard, Some(&event), usage.as_ref(), &tool_names, false)
                    .await
                {
                    return;
                }
            }
            let encoded = match encoder.feed(&outward) {
                Ok(encoded) => encoded,
                Err(_) => {
                    if terminal.is_some() {
                        // The attempt already settled by its provider
                        // terminal; the stream simply ends short.
                        return;
                    }
                    fail_stream!(Failure::new(
                        FailureClass::Internal,
                        "gateway could not encode the provider stream",
                    ))
                }
            };
            if terminal.is_some() {
                // Terminal frames flow through the shared publication tail so
                // keyed owners publish the exact byte stream first.
                finish_stream_terminal(
                    &sender,
                    deadline,
                    &mut lease,
                    replayable,
                    &mut capture,
                    &cached_headers,
                    encoded.into_iter().map(Bytes::from).collect(),
                )
                .await;
                return;
            }
            for data in encoded {
                let data = Bytes::from(data);
                if lease.is_some() {
                    replayable = capture_frame(&mut capture, &data, replayable);
                }
                if !send_bounded(&sender, deadline, data).await {
                    settle_stream_end(&mut guard, None, usage.as_ref(), &tool_names, true).await;
                    return;
                }
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
) -> bool {
    match terminal {
        Some(Event::Failed(failure)) => {
            let failure = failure.clone().boundary();
            guard
                .settle("failed", usage, tool_names, Some(&failure), true)
                .await
        }
        Some(Event::Incomplete) => {
            guard
                .settle("incomplete", usage, tool_names, None, true)
                .await
        }
        Some(_) => {
            guard
                .settle("completed", usage, tool_names, None, true)
                .await
        }
        None => {
            if disconnected {
                guard.settle_cancelled(usage, tool_names).await
            } else {
                true
            }
        }
    }
}

async fn responses(State(state): State<AppState>, request: axum::extract::Request) -> Response {
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

    // Replay-keyed Responses keeps the python engine's idempotency semantics.
    // Presence is checked on the raw header map so a non-UTF8 value still
    // escalates instead of silently dropping replay behavior.
    if headers.contains_key("idempotency-key") || headers.contains_key("x-client-request-id") {
        return proxy_to_python(
            &state,
            reqwest::Method::POST,
            "/v1/responses",
            &headers,
            body,
        )
        .await;
    }
    let body_text = match String::from_utf8(body.to_vec()) {
        Ok(text) => text,
        Err(_) => return error_response(&PublicError::invalid_json()),
    };

    let admit_argument = compact_json(&json!({
        "raw_key": raw_key,
        "body": body_text,
        "surface": "responses",
    }));
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
            "/v1/responses",
            &headers,
            body,
        )
        .await;
    }
    let admission: Admission = match serde_json::from_value(admission_value.clone()) {
        Ok(admission) => admission,
        Err(_) => return wire_drift_response(&state, &admission_value, started).await,
    };
    let mut guard = new_guard(&state, admission.request_id.clone(), started);

    let permit = match acquire_permit(&state, &mut guard, deadline).await {
        Ok(permit) => permit,
        Err(response) => return *response,
    };

    let context = WaterfallContext {
        bridge: &state.bridge,
        http: &state.http,
        request_id: &admission.request_id,
        raw_key: &raw_key,
        route: &admission.route,
        policy: admission.policy(),
        deadline,
    };
    let won = acquire_attempt(&context, &mut guard).await;

    // Responses envelopes carry a float wall clock, like the python encoder.
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs_f64())
        .unwrap_or(0.0);

    match won {
        Won::Failed(error) => error_response(&error),
        Won::Settled(settled) => settled_responses_response(&admission, settled, created_at),
        Won::Committed(committed) => {
            let committed = *committed;
            if admission.output_guardrail {
                guarded_responses(
                    state, admission, guard, committed, created_at, deadline, permit,
                )
                .await
            } else if admission.stream {
                stream_responses(
                    state.clone(),
                    admission,
                    guard,
                    committed,
                    created_at,
                    deadline,
                    permit,
                )
                .await
            } else {
                completed_responses(
                    &state, admission, guard, committed, created_at, deadline, permit,
                )
                .await
            }
        }
    }
}

/// Aggregated assistant output tracked while relaying one Responses stream,
/// mirroring `assistant_message` inputs for continuation retention.
#[derive(Default)]
struct ResponsesRetention {
    text: String,
    refusal: bool,
    tool_calls: Vec<CompletedToolCall>,
    retained_bytes: usize,
    overflowed: bool,
}

impl ResponsesRetention {
    fn track(&mut self, event: &Event) {
        if self.overflowed {
            return;
        }
        self.retained_bytes = self
            .retained_bytes
            .saturating_add(event_retained_bytes(event));
        if self.retained_bytes > MAXIMUM_RETAINED_OUTPUT_BYTES {
            // Retention is bounded like the python service's aggregation:
            // the stream keeps flowing but nothing oversize is remembered.
            self.overflowed = true;
            self.text.clear();
            self.tool_calls.clear();
            return;
        }
        match event {
            Event::TextDelta(delta) => self.text.push_str(delta),
            Event::RefusalDelta(_) => self.refusal = true,
            Event::ToolCallCompleted { call, .. } => self.tool_calls.push(call.clone()),
            _ => {}
        }
    }
}

/// Build the retention payload consumed by the control plane's `remember`.
fn remember_argument(request_id: &str, retention: &ResponsesRetention) -> String {
    compact_json(&json!({
        "request_id": request_id,
        "text": retention.text,
        "refusal": retention.refusal,
        "tool_calls": retention
            .tool_calls
            .iter()
            .map(|call| json!({
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.raw_arguments,
            }))
            .collect::<Vec<Value>>(),
    }))
}

/// Retain one completed Responses continuation before the terminal frames
/// flush, mirroring the python service's ordering. Returns the public error
/// when bounded retention fails closed.
async fn remember_continuation(
    state: &AppState,
    request_id: &str,
    retention: &ResponsesRetention,
) -> Result<(), PublicError> {
    if retention.overflowed
        || retention.refusal
        || (retention.text.is_empty() && retention.tool_calls.is_empty())
    {
        return Ok(());
    }
    state
        .bridge
        .call("remember", remember_argument(request_id, retention))
        .await
        .map(|_| ())
}

/// Answer one Responses attempt that the waterfall already settled: a
/// successful terminal with no semantic output, or an exhausted ladder
/// flushing withheld refusal output ahead of the failing terminal.
fn settled_responses_response(
    admission: &Admission,
    settled: SettledAttempt,
    created_at: f64,
) -> Response {
    let mut events = settled.events;
    let refusal_completed = complete_visible_refusal(&mut events);
    if refusal_completed.is_none() {
        if let Some(Event::Failed(failure)) = events.last() {
            if !admission.stream {
                return error_response(&collection_public_error(&failure.clone().boundary()));
            }
        }
    }
    let mut headers = commit_independent(admission, None);
    headers.extend(commit_dependent(admission, settled.depth));
    if admission.stream {
        let body = match encode_responses_sse(admission, created_at, &events) {
            Ok(body) => body,
            Err(error) => return error_response(&error),
        };
        return sse_body_response(&headers, body);
    }
    let envelope = admission.envelope.clone().unwrap_or_default();
    let aggregated = match completed_responses_body(
        &admission.request_id,
        &admission.alias,
        created_at,
        envelope,
        &events,
    ) {
        Ok(aggregated) => aggregated,
        Err(error) => return error_response(&error),
    };
    if let Some(failure) = &aggregated.failure {
        return error_response(&failure.clone().boundary().public_error());
    }
    json_response(StatusCode::OK, &aggregated.body, &headers)
}

/// Aggregate one committed Responses attempt and answer it, retaining the
/// continuation and settling exactly once.
#[allow(clippy::too_many_arguments)]
async fn respond_from_responses_events(
    state: &AppState,
    admission: Admission,
    mut guard: AttemptGuard,
    depth: usize,
    mut events: Vec<Event>,
    usage: Option<Usage>,
    tool_names: Vec<String>,
    created_at: f64,
    stream_body: bool,
) -> Response {
    let refusal_completed = complete_visible_refusal(&mut events);
    let envelope = admission.envelope.clone().unwrap_or_default();
    let aggregated = match completed_responses_body(
        &admission.request_id,
        &admission.alias,
        created_at,
        envelope,
        &events,
    ) {
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
            return error_response(&error);
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
        return error_response(&error);
    }
    // Retention runs while the attempt row is still in flight so the control
    // plane can resolve its namespaced continuation context, and before the
    // body is answered so an oversize continuation fails closed like python.
    let retention = ResponsesRetention {
        text: aggregated.text.clone(),
        refusal: !aggregated.refusal.is_empty() || refusal_completed.is_some(),
        tool_calls: aggregated.tool_calls.clone(),
        ..ResponsesRetention::default()
    };
    let remembered = remember_continuation(state, &admission.request_id, &retention).await;
    let settled = if let Some(refusal) = &refusal_completed {
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
    if let Err(error) = remembered {
        // The provider outcome already settled above, exactly like the python
        // executor; only the HTTP result reports the retention failure.
        return error_response(&error);
    }
    if !settled {
        // Success is only reported once the terminal accounting write landed.
        return error_response(&PublicError::internal());
    }
    let mut headers = commit_independent(&admission, None);
    headers.extend(commit_dependent(&admission, depth));
    if stream_body {
        let body = match encode_responses_sse(&admission, created_at, &events) {
            Ok(body) => body,
            Err(error) => return error_response(&error),
        };
        return sse_body_response(&headers, body);
    }
    json_response(StatusCode::OK, &aggregated.body, &headers)
}

#[allow(clippy::too_many_arguments)]
async fn completed_responses(
    state: &AppState,
    admission: Admission,
    mut guard: AttemptGuard,
    mut committed: CommittedAttempt,
    created_at: f64,
    deadline: Instant,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    let _permit = permit;
    let phase_timeout = admission.phase_timeout(committed.depth);
    let events =
        match collect_committed(&mut committed, deadline, phase_timeout, guard.started).await {
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
                return error_response(&error);
            }
        };
    respond_from_responses_events(
        state,
        admission,
        guard,
        committed.depth,
        events,
        committed.usage,
        committed.tool_names,
        created_at,
        false,
    )
    .await
}

fn encode_responses_sse(
    admission: &Admission,
    created_at: f64,
    events: &[Event],
) -> Result<Vec<u8>, PublicError> {
    let envelope = admission.envelope.clone().unwrap_or_default();
    let mut encoder = ResponsesSseEncoder::new(
        &admission.request_id,
        &admission.alias,
        created_at,
        envelope,
    );
    let mut body = Vec::new();
    for frame in encoder.start()? {
        body.extend_from_slice(frame.as_bytes());
    }
    for event in events {
        for frame in encoder.feed(event)? {
            body.extend_from_slice(frame.as_bytes());
        }
    }
    Ok(body)
}

#[allow(clippy::too_many_arguments)]
async fn guarded_responses(
    state: AppState,
    admission: Admission,
    mut guard: AttemptGuard,
    mut committed: CommittedAttempt,
    created_at: f64,
    deadline: Instant,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    let _permit = permit;
    let phase_timeout = admission.phase_timeout(committed.depth);
    let collected =
        match collect_committed(&mut committed, deadline, phase_timeout, guard.started).await {
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
                return error_response(&error);
            }
        };
    let events = match apply_output_guardrail(&admission, &guard.bridge, collected).await {
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
            return error_response(&failure.public_error());
        }
    };
    let stream_body = admission.stream;
    respond_from_responses_events(
        &state,
        admission,
        guard,
        committed.depth,
        events,
        committed.usage,
        committed.tool_names,
        created_at,
        stream_body,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn stream_responses(
    state: AppState,
    admission: Admission,
    guard: AttemptGuard,
    committed: CommittedAttempt,
    created_at: f64,
    deadline: Instant,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    let (sender, receiver) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    let mut header_pairs = commit_independent(&admission, None);
    header_pairs.extend(commit_dependent(&admission, committed.depth));
    let request_id = admission.request_id.clone();
    let alias = admission.alias.clone();
    let envelope = admission.envelope.clone().unwrap_or_default();
    let phase_timeout = admission.phase_timeout(committed.depth);
    let task_hold = guard.hold_task();
    tokio::spawn(async move {
        let _task = task_hold;
        let _permit = permit;
        let mut guard = guard;
        let mut committed = committed;
        let mut encoder = ResponsesSseEncoder::new(&request_id, &alias, created_at, envelope);
        let mut usage: Option<Usage> = committed.usage.take();
        let mut tool_names: Vec<String> = std::mem::take(&mut committed.tool_names);
        let mut visible_refusal = committed.visible_refusal;
        let mut terminal: Option<Event> = None;
        let mut retention = ResponsesRetention::default();
        // Terminal frames are withheld until continuation retention lands,
        // mirroring the python stream body's ordering.
        let terminal_frames: Vec<String>;

        macro_rules! fail_stream {
            ($failure:expr) => {{
                let failure = $failure.boundary();
                emit_responses_failure(&sender, deadline, &mut encoder, &failure).await;
                guard
                    .settle("failed", usage.as_ref(), &tool_names, Some(&failure), true)
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

        let mut prefix: std::collections::VecDeque<Event> = committed.prefix.drain(..).collect();
        loop {
            let event = if let Some(event) = prefix.pop_front() {
                event
            } else {
                match committed
                    .relay
                    .next_event(deadline, phase_timeout, guard.started)
                    .await
                {
                    Ok(Some(event)) => event,
                    Ok(None) => {
                        fail_stream!(Failure::new(
                            FailureClass::MalformedResponse,
                            "provider stream ended without a terminal event",
                        ))
                    }
                    Err(failure) => fail_stream!(failure),
                }
            };
            track_event(&event, &mut usage, &mut tool_names);
            retention.track(&event);
            if matches!(event, Event::RefusalDelta(_)) {
                visible_refusal = true;
            }
            let outward = match &event {
                Event::Failed(failure)
                    if failure.failure_class == FailureClass::Refusal && visible_refusal =>
                {
                    Event::Completed
                }
                other => other.clone(),
            };
            // The terminal is recorded before its frames flush, so a
            // disconnect during the final flush still settles by the
            // provider's outcome instead of as a cancellation.
            if event.is_terminal() {
                terminal = Some(event.clone());
            }
            let encoded = match encoder.feed(&outward) {
                Ok(encoded) => encoded,
                Err(_) => {
                    fail_stream!(Failure::new(
                        FailureClass::Internal,
                        "gateway could not encode the provider stream",
                    ))
                }
            };
            if terminal.is_some() {
                terminal_frames = encoded;
                break;
            }
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

        // Retention runs before the terminal frames flush; a bounded
        // retention failure truncates the stream before its terminal, the
        // same observable behavior as the python service.
        let retainable = !matches!(terminal, Some(Event::Failed(_)));
        if retainable {
            if let Err(_error) =
                remember_continuation(&state, &admission.request_id, &retention).await
            {
                settle_stream_end(
                    &mut guard,
                    terminal.as_ref(),
                    usage.as_ref(),
                    &tool_names,
                    false,
                )
                .await;
                return;
            }
        }
        for data in terminal_frames {
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

/// Emit the Responses encoder's sanitized failure lifecycle when the stream
/// has not already reached a terminal.
async fn emit_responses_failure(
    sender: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    deadline: Instant,
    encoder: &mut ResponsesSseEncoder,
    failure: &Failure,
) {
    if encoder.saw_terminal() {
        return;
    }
    let frames = encoder
        .feed(&Event::Failed(failure.clone()))
        .unwrap_or_default();
    for frame in frames {
        if !send_bounded(sender, deadline, Bytes::from(frame)).await {
            return;
        }
    }
}

/// Build the encoder's sanitized failure frame and done sentinel when the
/// stream has not already reached a terminal.
fn failure_frames(encoder: &mut ChatSseEncoder, failure: &Failure) -> Vec<Bytes> {
    if encoder.saw_terminal() {
        return Vec::new();
    }
    encoder
        .feed(&Event::Failed(failure.clone()))
        .unwrap_or_else(|_| {
            vec![
                chat_data(&failure.public_error().json_body()),
                "data: [DONE]\n\n".to_string(),
            ]
        })
        .into_iter()
        .map(Bytes::from)
        .collect()
}

/// Close one stream that reached a terminal outcome: publish the keyed
/// capture (or abandon it), then flush the withheld terminal frames.
///
/// Mirrors the python engine's `_stream_body` tail: a keyed stream that
/// cannot be retained (capture overflow or a rejected publication) ends
/// without its terminal frames, so the caller observes a truncated stream
/// rather than an unreplayable success, and every waiting duplicate fails
/// closed instead of hanging.
async fn finish_stream_terminal(
    sender: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    deadline: Instant,
    lease: &mut Option<OwnerLease>,
    mut replayable: bool,
    capture: &mut Vec<u8>,
    cached_headers: &[(String, String)],
    frames: Vec<Bytes>,
) {
    if lease.is_some() {
        for data in &frames {
            replayable = capture_frame(capture, data, replayable);
        }
    }
    if let Some(mut owner) = lease.take() {
        if replayable {
            let cached = CachedResponse {
                status_code: 200,
                media_type: "text/event-stream".to_string(),
                headers: cached_headers.to_vec(),
                body: std::mem::take(capture),
            };
            if owner.complete(cached).await.is_err() {
                return;
            }
        } else {
            owner.abandon().await;
            return;
        }
    }
    for data in frames {
        if !send_bounded(sender, deadline, data).await {
            return;
        }
    }
}
