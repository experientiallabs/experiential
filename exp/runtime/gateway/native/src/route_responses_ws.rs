//! WebSocket transport for the OpenAI Responses surface: the GET
//! `/v1/responses` upgrade handler and its per-connection frame loop,
//! mirroring the Responses-over-WebSocket contract served by
//! api.openai.com. A client frame is the standard Responses request body
//! tagged `"type": "response.create"`; every server frame is exactly one
//! standard Responses stream event JSON (the SSE `data:` payload), with
//! request-level failures carried as wrapped in-band
//! `{"type": "error", "error": ..., "status": ...}` frames. The transport
//! is an adapter only: every generating request crosses the exact HTTP
//! handler (`route_responses::responses`) with the connection's upgrade
//! headers, so decode, admission, replay, waterfall, and settlement are
//! identical on both transports. A `generate: false` frame is the
//! connection prewarm: it is answered with an empty completed response
//! envelope and never touches admission or the ledger.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::ws::rejection::WebSocketUpgradeRejection;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::{header, HeaderMap, Method};
use axum::response::Response;
use futures_util::StreamExt;
use serde_json::{json, Map, Value};

use crate::encode::compact_json;
use crate::encode_responses::{ResponsesEnvelope, ResponsesSseEncoder};
use crate::errors::PublicError;
use crate::events::Event;
use crate::respond::{bearer_key, error_response};
use crate::route_responses::responses;
use crate::server::AppState;
use crate::sse::SseDecoder;

/// Bound on a buffered non-streaming (error) response body read back from
/// the HTTP handler; its bodies are single compact JSON envelopes.
const MAXIMUM_ADAPTED_BODY_BYTES: usize = 1_000_000;

/// Byte-compatible rejection for `"stream": false`, which the WebSocket
/// transport cannot honor (api.openai.com answers this exact message).
const STREAM_FALSE_MESSAGE: &str = "The 'stream' parameter is not supported on /v1/responses.";

/// Monotonic component of prewarm response identities within one process.
static PREWARM_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Answer the `/v1/responses` WebSocket upgrade.
///
/// The bearer key is authenticated before the upgrade is accepted, so an
/// unknown key rejects the handshake with the same plain HTTP 401 the POST
/// surface answers (matching api.openai.com). A GET without a well-formed
/// upgrade fails closed with 426, the one status the Codex client treats
/// as "use HTTP instead".
pub(crate) async fn responses_ws(
    State(state): State<AppState>,
    headers: HeaderMap,
    upgrade: Result<WebSocketUpgrade, WebSocketUpgradeRejection>,
) -> Response {
    state.handled_requests.fetch_add(1, Ordering::Relaxed);
    let upgrade = match upgrade {
        Ok(upgrade) => upgrade,
        Err(_) => return error_response(&upgrade_required_error()),
    };
    let raw_key = match bearer_key(&headers) {
        Ok(key) => key,
        Err(error) => return error_response(&error),
    };
    // Upgrade-time rejection mirrors api.openai.com; the shared HTTP handler
    // still authenticates every later frame with the connection's headers,
    // so key revocation applies mid-connection at the next request.
    let authenticate = compact_json(&json!({"raw_key": raw_key}));
    if let Err(error) = state.bridge.call("authenticate", authenticate).await {
        return error_response(&error);
    }
    let connection_headers = request_headers(&headers);
    upgrade.on_upgrade(move |socket| serve_socket(state, connection_headers, socket))
}

/// The 426 answered for a GET that is not a well-formed WebSocket upgrade.
fn upgrade_required_error() -> PublicError {
    PublicError::new(
        426,
        "upgrade_required",
        "GET /v1/responses is served only as a WebSocket upgrade. Open a \
         WebSocket connection, or POST the request over HTTP.",
        "invalid_request_error",
    )
}

/// Copy the upgrade headers every synthesized per-frame request carries,
/// dropping the handshake-only fields that do not describe the request.
fn request_headers(headers: &HeaderMap) -> HeaderMap {
    let mut carried = headers.clone();
    for name in [
        header::CONNECTION,
        header::UPGRADE,
        header::CONTENT_LENGTH,
        header::SEC_WEBSOCKET_KEY,
        header::SEC_WEBSOCKET_VERSION,
        header::SEC_WEBSOCKET_EXTENSIONS,
        header::SEC_WEBSOCKET_PROTOCOL,
    ] {
        carried.remove(name);
    }
    carried
}

/// Serve one accepted connection: sequential request frames, each answered
/// with its full event stream before the next frame is read.
async fn serve_socket(state: AppState, headers: HeaderMap, mut socket: WebSocket) {
    while let Some(message) = socket.recv().await {
        let message = match message {
            Ok(message) => message,
            Err(_) => return,
        };
        match message {
            Message::Text(text) => {
                if handle_frame(&state, &headers, &mut socket, text.as_str())
                    .await
                    .is_err()
                {
                    return;
                }
            }
            Message::Binary(_) => {
                let error = PublicError::new(
                    400,
                    "invalid_request",
                    "Binary WebSocket frames are not supported. Send one JSON \
                     response.create object per text frame.",
                    "invalid_request_error",
                );
                if send_public_error(&mut socket, &error).await.is_err() {
                    return;
                }
            }
            Message::Close(_) => return,
            // The axum WebSocket answers pings itself; pongs carry no work.
            Message::Ping(_) | Message::Pong(_) => {}
        }
    }
}

/// Decode one request frame and stream its response frames back.
///
/// Returns `Err(())` only when the socket is unusable; request-level
/// failures are answered in-band and keep the connection open.
async fn handle_frame(
    state: &AppState,
    headers: &HeaderMap,
    socket: &mut WebSocket,
    text: &str,
) -> Result<(), ()> {
    let mut value: Value = match serde_json::from_str(text) {
        Ok(Value::Object(entries)) => Value::Object(entries),
        _ => return send_public_error(socket, &PublicError::invalid_json()).await,
    };
    let body = value.as_object_mut().expect("frame decoded as an object");
    match body.remove("type") {
        Some(Value::String(kind)) if kind == "response.create" => {}
        _ => {
            let error = PublicError::new(
                400,
                "invalid_request",
                "WebSocket request frames must be tagged \"type\": \
                 \"response.create\".",
                "invalid_request_error",
            );
            return send_public_error(socket, &error).await;
        }
    }
    // `generate: false` is the transport prewarm: acknowledge the connection
    // with an empty completed response and perform no model work.
    if let Some(generate) = body.remove("generate") {
        if generate == Value::Bool(false) {
            return send_prewarm_ack(socket, &value).await;
        }
    }
    // The WebSocket surface always streams; only an explicit opt-out is a
    // request error (api.openai.com answers this exact wrapped 400).
    match body.get("stream") {
        Some(Value::Bool(false)) => {
            let error = PublicError::new(
                400,
                "invalid_request",
                STREAM_FALSE_MESSAGE,
                "invalid_request_error",
            );
            return send_public_error(socket, &error).await;
        }
        _ => {
            body.insert("stream".to_string(), Value::Bool(true));
        }
    }

    let mut request = axum::http::Request::builder()
        .method(Method::POST)
        .uri("/v1/responses")
        .body(axum::body::Body::from(compact_json(&value)))
        .expect("static request line is valid");
    *request.headers_mut() = headers.clone();
    let response = responses(State(state.clone()), request).await;
    relay_response(socket, response).await
}

/// Re-frame one HTTP handler response onto the socket: an event-stream body
/// becomes one text frame per SSE event; anything else becomes one wrapped
/// in-band error frame.
async fn relay_response(socket: &mut WebSocket, response: Response) -> Result<(), ()> {
    let status = response.status();
    let is_event_stream = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.starts_with("text/event-stream"));
    if status.is_success() && is_event_stream {
        let mut decoder = SseDecoder::new();
        // Dropping the body mid-stream cancels the in-flight attempt through
        // the same disconnect guards an HTTP client disconnect triggers.
        let mut body = response.into_body().into_data_stream();
        while let Some(chunk) = body.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(_) => {
                    let error = PublicError::internal();
                    return send_public_error(socket, &error).await;
                }
            };
            let events = match decoder.feed(&chunk) {
                Ok(events) => events,
                Err(_) => {
                    let error = PublicError::internal();
                    return send_public_error(socket, &error).await;
                }
            };
            for event in events {
                if socket.send(Message::Text(event.data.into())).await.is_err() {
                    return Err(());
                }
            }
        }
        return Ok(());
    }

    let retry_after = response
        .headers()
        .get(header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let body = match axum::body::to_bytes(response.into_body(), MAXIMUM_ADAPTED_BODY_BYTES).await {
        Ok(body) => body,
        Err(_) => return send_public_error(socket, &PublicError::internal()).await,
    };
    let error = serde_json::from_slice::<Value>(&body)
        .ok()
        .and_then(|mut envelope| {
            envelope
                .as_object_mut()
                .and_then(|entries| entries.remove("error"))
        })
        .unwrap_or_else(|| error_object(&PublicError::internal()));
    send_wrapped_error(socket, status.as_u16(), error, retry_after).await
}

/// Answer one prewarm frame with created, in-progress, and empty completed
/// lifecycle events reflecting the request envelope, without admission,
/// ledger, or provider work.
async fn send_prewarm_ack(socket: &mut WebSocket, request: &Value) -> Result<(), ()> {
    let model = request
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let envelope: ResponsesEnvelope =
        serde_json::from_value(request.clone()).unwrap_or_default();
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs_f64())
        .unwrap_or_default();
    let sequence = PREWARM_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let request_id = format!("prewarm:{created_at}:{sequence}");
    let mut encoder = ResponsesSseEncoder::new(&request_id, &model, created_at, envelope);
    let mut frames = match encoder.start() {
        Ok(frames) => frames,
        Err(error) => return send_public_error(socket, &error).await,
    };
    match encoder.feed(&Event::Completed) {
        Ok(terminal) => frames.extend(terminal),
        Err(error) => return send_public_error(socket, &error).await,
    }
    for frame in frames {
        let Some(data) = sse_frame_data(&frame) else {
            return send_public_error(socket, &PublicError::internal()).await;
        };
        if socket
            .send(Message::Text(data.to_string().into()))
            .await
            .is_err()
        {
            return Err(());
        }
    }
    Ok(())
}

/// Extract the single-line `data:` payload from one encoder SSE frame.
fn sse_frame_data(frame: &str) -> Option<&str> {
    frame
        .split_once("\ndata: ")
        .map(|(_, tail)| tail)
        .and_then(|tail| tail.strip_suffix("\n\n"))
}

/// The bare error object inside one `PublicError`'s OpenAI envelope.
fn error_object(error: &PublicError) -> Value {
    let mut envelope = error.json_body();
    envelope
        .as_object_mut()
        .and_then(|entries| entries.remove("error"))
        .expect("public error envelope carries an error object")
}

/// Send one wrapped in-band error frame built from a `PublicError`.
async fn send_public_error(socket: &mut WebSocket, error: &PublicError) -> Result<(), ()> {
    let envelope = error_object(error);
    let retry_after = error.retry_after_seconds.map(|wait| wait.to_string());
    send_wrapped_error(socket, error.status_code, envelope, retry_after).await
}

/// Send the wrapped `{"type": "error", "error": ..., "status": ...}` frame
/// the Responses-over-WebSocket contract uses for request-level failures.
async fn send_wrapped_error(
    socket: &mut WebSocket,
    status: u16,
    error: Value,
    retry_after: Option<String>,
) -> Result<(), ()> {
    let mut frame = Map::new();
    frame.insert("type".to_string(), Value::String("error".to_string()));
    frame.insert("error".to_string(), error);
    frame.insert("status".to_string(), json!(status));
    if let Some(wait) = retry_after {
        frame.insert("headers".to_string(), json!({ "retry-after": wait }));
    }
    let payload = compact_json(&Value::Object(frame));
    socket
        .send(Message::Text(payload.into()))
        .await
        .map_err(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_frame_data_extracts_the_payload() {
        let frame = "event: response.created\ndata: {\"type\":\"response.created\"}\n\n";
        assert_eq!(sse_frame_data(frame), Some("{\"type\":\"response.created\"}"));
    }

    #[test]
    fn sse_frame_data_rejects_a_malformed_frame() {
        assert_eq!(sse_frame_data("data only"), None);
        assert_eq!(sse_frame_data("event: x\ndata: {}"), None);
    }

    #[test]
    fn upgrade_required_error_is_the_codex_fallback_status() {
        assert_eq!(upgrade_required_error().status_code, 426);
    }

    #[test]
    fn request_headers_drop_the_handshake_fields() {
        let mut headers = HeaderMap::new();
        headers.insert(header::AUTHORIZATION, "Bearer xpl_test".parse().unwrap());
        headers.insert(header::CONNECTION, "Upgrade".parse().unwrap());
        headers.insert(header::UPGRADE, "websocket".parse().unwrap());
        headers.insert(header::SEC_WEBSOCKET_KEY, "abc".parse().unwrap());
        headers.insert(header::SEC_WEBSOCKET_VERSION, "13".parse().unwrap());
        headers.insert("x-client-request-id", "session-1".parse().unwrap());
        let carried = request_headers(&headers);
        assert!(carried.contains_key(header::AUTHORIZATION));
        assert!(carried.contains_key("x-client-request-id"));
        assert!(!carried.contains_key(header::CONNECTION));
        assert!(!carried.contains_key(header::UPGRADE));
        assert!(!carried.contains_key(header::SEC_WEBSOCKET_KEY));
        assert!(!carried.contains_key(header::SEC_WEBSOCKET_VERSION));
    }
}
