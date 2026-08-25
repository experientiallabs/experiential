//! Relay to the embedded python fallback engine: the in-flight proxy count,
//! the bounded-retry request replay, and the catch-all route handler that
//! forwards every path the native plane does not own (or answers the native
//! error in rust-only mode).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use futures_util::StreamExt;

use crate::errors::PublicError;
use crate::metrics::METRICS;
use crate::respond::{error_response, read_body};
use crate::server::AppState;

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

/// The native 404 for a route the gateway does not serve, in the OpenAI
/// error envelope.
pub(crate) fn unknown_route_error() -> PublicError {
    PublicError::new(
        404,
        "unknown_route",
        "Unknown route. The gateway serves /v1/models, /v1/chat/completions, \
         /v1/responses, /v1/messages, /v1/messages/count_tokens, /health/live, \
         /health/ready, /metrics, /metrics.json, /usage, and /usage.json.",
        "invalid_request_error",
    )
}

/// The rust-only answer to an escalation disposition: startup validation
/// guarantees every granted alias is natively servable, so reaching this is
/// an internal error rather than a degraded route.
pub(crate) fn no_fallback_escalation_error() -> PublicError {
    PublicError::new(
        500,
        "internal_error",
        "The gateway cannot serve this request natively and no fallback \
         engine is configured. Ask the gateway operator to inspect the \
         server logs.",
        "api_error",
    )
}

/// Replay one HTTP request against the embedded python engine and stream the
/// response back unchanged. Serves the surfaces the native plane escalates
/// (providers without a native dialect, host-ineligible routes) plus unknown
/// routes while a fallback engine exists; `absent_error` answers the caller
/// in rust-only mode.
pub(crate) async fn proxy_to_python(
    state: &AppState,
    method: reqwest::Method,
    path_and_query: &str,
    headers: &HeaderMap,
    body: Bytes,
    absent_error: PublicError,
) -> Response {
    let Some(fallback_base) = state.fallback_base.clone() else {
        return error_response(&absent_error);
    };
    let guard = ProxyGuard::new(state.active_proxies.clone());
    let url = format!("{fallback_base}{path_and_query}");
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
pub(crate) async fn proxy_fallback(
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
    proxy_to_python(
        &state,
        method,
        &path_and_query,
        &parts.headers,
        bytes,
        unknown_route_error(),
    )
    .await
}
