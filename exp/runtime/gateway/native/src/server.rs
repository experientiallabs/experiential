//! Server composition for the axum data plane: serve-time configuration, the
//! shared application state, router construction with graceful shutdown, and
//! the small control-plane-backed routes (health, models, usage, metrics).
//! The chat, Responses, and Messages surfaces live in their `route_*`
//! modules; unknown routes answer the native 404 in the OpenAI envelope.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::serve::ListenerExt;
use axum::Router;
use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::Semaphore;

use crate::bridge::Bridge;
use crate::encode::compact_json;
use crate::errors::PublicError;
use crate::replay::ReplayStore;
use crate::respond::{bearer_key, error_response, json_response, unknown_route_error};
use crate::route_batches::{
    batches_cancel, batches_create, batches_list, batches_retrieve, files_body_limit,
    files_content, files_create, files_retrieve,
};
use crate::route_chat::chat;
use crate::route_messages::{messages, messages_count_tokens};
use crate::route_responses::responses;
use crate::route_responses_ws::responses_ws;

/// Serve-time configuration passed from `exp --engine rust`.
#[derive(Debug, Clone, Deserialize)]
pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    #[serde(default = "default_max_active_requests")]
    pub max_active_requests: usize,
    #[serde(default = "default_request_timeout_seconds")]
    pub request_timeout_seconds: f64,
    /// Fail-fast bound on the TCP+TLS connect phase of every provider call.
    #[serde(default = "default_connect_timeout_seconds")]
    pub connect_timeout_seconds: f64,
    /// Fail-fast bound on the wait for a provider's first streamed byte. This
    /// never caps total generation time: once the first byte arrives, reads
    /// are paced by the deployment's own per-chunk timeout.
    #[serde(default = "default_time_to_first_byte_seconds")]
    pub time_to_first_byte_seconds: f64,
    /// Input-scaled first-byte allowance added on top of the flat bound, in
    /// seconds per million approximate input tokens (request bytes / 4), so
    /// a very large prompt's prefill is not misread as a dead lane.
    #[serde(default = "default_time_to_first_byte_seconds_per_million_input_tokens")]
    pub time_to_first_byte_seconds_per_million_input_tokens: f64,
    #[serde(default = "default_callback_permits")]
    pub callback_permits: usize,
    #[serde(default = "default_native_usage_enabled")]
    pub native_usage_enabled: bool,
    #[serde(default = "default_graceful_timeout_seconds")]
    pub graceful_timeout_seconds: f64,
}

fn default_graceful_timeout_seconds() -> f64 {
    10.0
}

fn default_connect_timeout_seconds() -> f64 {
    5.0
}

fn default_time_to_first_byte_seconds() -> f64 {
    15.0
}

fn default_time_to_first_byte_seconds_per_million_input_tokens() -> f64 {
    240.0
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
pub(crate) struct AppState {
    pub(crate) bridge: Arc<Bridge>,
    pub(crate) http: reqwest::Client,
    pub(crate) permits: Arc<Semaphore>,
    pub(crate) request_timeout: Duration,
    /// Fail-fast flat bound on the wait for the first provider byte per attempt.
    pub(crate) time_to_first_byte: Duration,
    /// Default input-scaled first-byte allowance in seconds per million
    /// approximate input tokens.
    pub(crate) time_to_first_byte_slope_seconds_per_million_input_tokens: f64,
    /// Settlement writes still in flight, held open through graceful shutdown.
    pub(crate) pending_settlements: Arc<AtomicUsize>,
    /// Requests handled since start; the idle reclaim loop trims the
    /// allocator once per burst when this advances and the plane is idle.
    pub(crate) handled_requests: Arc<AtomicUsize>,
    /// Bounded in-process keyed-response replay, the native mirror of the
    /// python engine's `BoundedReplayStore`.
    pub(crate) replays: Arc<ReplayStore>,
}

/// Run the data plane until shutdown; returns after graceful stop.
///
/// `shutdown` optionally carries an embedder-owned stop signal beside the
/// process signals, so a host can stop the plane without sending SIGINT.
pub async fn run(
    bridge: Arc<Bridge>,
    config: ServeConfig,
    shutdown: Option<tokio::sync::watch::Receiver<bool>>,
    on_listening: Option<Py<PyAny>>,
) -> Result<(), String> {
    let connect_timeout = Duration::from_secs_f64(config.connect_timeout_seconds.max(0.001));
    let http = crate::upstream::build_client(connect_timeout)?;
    let pending_settlements = Arc::new(AtomicUsize::new(0));
    let max_active_requests = config.max_active_requests.max(1);
    let handled_requests = Arc::new(AtomicUsize::new(0));
    let state = AppState {
        bridge,
        http,
        permits: Arc::new(Semaphore::new(max_active_requests)),
        request_timeout: Duration::from_secs_f64(config.request_timeout_seconds),
        time_to_first_byte: Duration::from_secs_f64(config.time_to_first_byte_seconds.max(0.001)),
        time_to_first_byte_slope_seconds_per_million_input_tokens: config
            .time_to_first_byte_seconds_per_million_input_tokens
            .max(0.0),
        pending_settlements: pending_settlements.clone(),
        handled_requests: handled_requests.clone(),
        replays: Arc::new(ReplayStore::new()),
    };
    tokio::spawn(crate::memory::reclaim_when_idle(
        state.permits.clone(),
        max_active_requests,
        handled_requests,
        pending_settlements.clone(),
    ));
    let app = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/models/{model_id}", get(model_detail))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/responses", post(responses).get(responses_ws))
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(messages_count_tokens))
        .route("/v1/batches", post(batches_create).get(batches_list))
        .route("/v1/batches/{batch_id}", get(batches_retrieve))
        .route("/v1/batches/{batch_id}/cancel", post(batches_cancel))
        .route("/v1/files", post(files_create).layer(files_body_limit()))
        .route("/v1/files/{file_id}", get(files_retrieve))
        .route("/v1/files/{file_id}/content", get(files_content))
        .route("/health/live", get(health_live))
        .route("/health/ready", get(health_ready))
        .route("/metrics.json", get(metrics_json))
        .route("/metrics", get(metrics_text));
    let app = if config.native_usage_enabled {
        app.route("/usage.json", get(usage_json))
            .route("/usage", get(usage_page))
    } else {
        app
    };
    let app = app.fallback(unknown_route).with_state(state);
    let listener = tokio::net::TcpListener::bind((config.host.as_str(), config.port))
        .await
        .map_err(|error| format!("failed to bind {}:{}: {error}", config.host, config.port))?
        // Small SSE frames must not sit behind Nagle's algorithm.
        .tap_io(|stream| {
            let _ = stream.set_nodelay(true);
        });
    // The socket is bound and queuing connections, so the embedder may now
    // truthfully announce readiness; a callback failure aborts the launch
    // before any traffic is accepted.
    if let Some(callback) = on_listening {
        Python::attach(|py| callback.call0(py))
            .map_err(|error| format!("on_listening callback failed: {error}"))?;
    }
    let graceful = Duration::from_secs_f64(config.graceful_timeout_seconds.max(0.1));
    let server =
        axum::serve(listener, app).with_graceful_shutdown(shutdown_requested(shutdown.clone()));
    // A stop request starts the graceful drain above; the arm below bounds
    // it, so a stuck stream cannot hold shutdown past the configured timeout.
    let outcome = tokio::select! {
        outcome = server => outcome.map_err(|error| format!("gateway server failed: {error}")),
        _ = async {
            shutdown_requested(shutdown).await;
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

/// Resolve on SIGINT, SIGTERM, or an embedder-owned stop request.
async fn shutdown_requested(receiver: Option<tokio::sync::watch::Receiver<bool>>) {
    match receiver {
        Some(mut receiver) => {
            let requested = async move {
                while !*receiver.borrow() {
                    if receiver.changed().await.is_err() {
                        // A dropped sender never requests a stop; wait on the
                        // process signals alone.
                        std::future::pending::<()>().await;
                    }
                }
            };
            tokio::select! {
                _ = shutdown_signal() => {}
                _ = requested => {}
            }
        }
        None => shutdown_signal().await,
    }
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

/// Answer any route the native plane does not own with the shared 404.
async fn unknown_route(State(state): State<AppState>) -> Response {
    state.handled_requests.fetch_add(1, Ordering::Relaxed);
    error_response(&unknown_route_error())
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
