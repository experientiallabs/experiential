//! Shared HTTP response construction and stream delivery helpers used by
//! every native surface: error and JSON bodies, latin-1 header transparency,
//! keyed-replay response materialization and capture, and the bounded SSE
//! delivery tail with its terminal settlement rules.

use std::time::Instant;

use axum::body::Body;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use serde_json::Value;
use tokio::sync::mpsc;

use crate::encode::compact_json;
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{Event, Usage};
use crate::relay::remaining;
use crate::replay::{CachedResponse, OwnerLease};
use crate::settlement::AttemptGuard;

/// Largest accepted request body on every native-served or proxied route.
/// Bounded so one client cannot hold unbounded gateway memory; far above any
/// real chat history (the python engine imposes no explicit cap).
pub(crate) const MAXIMUM_REQUEST_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Largest SSE capture retained for keyed-stream replay, matching the python
/// engine's `_STREAM_REPLAY_CAPTURE_BYTES`.
pub(crate) const STREAM_REPLAY_CAPTURE_BYTES: usize = 64 * 1024 * 1024;

pub(crate) fn error_response(error: &PublicError) -> Response {
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

pub(crate) fn json_response(
    status: StatusCode,
    payload: &Value,
    headers: &[(String, String)],
) -> Response {
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

pub(crate) fn bearer_key(headers: &HeaderMap) -> Result<String, PublicError> {
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
pub(crate) async fn read_body(body: Body) -> Result<Bytes, PublicError> {
    axum::body::to_bytes(body, MAXIMUM_REQUEST_BODY_BYTES)
        .await
        .map_err(|_| PublicError::request_too_large())
}

/// Read one header value as latin-1 text, the same byte-transparent decoding
/// the python engine's ASGI server applies, so any HTTP-legal value produces
/// the identical caller-operation string on both engines.
pub(crate) fn latin1_header(headers: &HeaderMap, name: &str) -> Option<String> {
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
pub(crate) fn cached_response(cached: &CachedResponse) -> Response {
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
pub(crate) fn capture_frame(buffer: &mut Vec<u8>, data: &[u8], replayable: bool) -> bool {
    if !replayable || buffer.len() + data.len() > STREAM_REPLAY_CAPTURE_BYTES {
        return false;
    }
    buffer.extend_from_slice(data);
    true
}

/// Deliver one public frame bounded by the request deadline, so a connected
/// client that stops reading cannot pin the admission permit and the upstream
/// connection forever. `false` means the client is gone or out of time.
pub(crate) async fn send_bounded(
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
pub(crate) fn complete_visible_refusal(events: &mut [Event]) -> Option<Failure> {
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

pub(crate) fn sse_body_response(headers: &[(String, String)], body: Vec<u8>) -> Response {
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

/// Settle a stream that reached its end (or lost its client). With a terminal
/// observed, the provider's outcome wins even on disconnect; without one, a
/// disconnect settles as cancelled.
pub(crate) async fn settle_stream_end(
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

/// Close one stream that reached a terminal outcome: publish the keyed
/// capture (or abandon it), then flush the withheld terminal frames.
///
/// Mirrors the python engine's `_stream_body` tail: a keyed stream that
/// cannot be retained (capture overflow or a rejected publication) ends
/// without its terminal frames, so the caller observes a truncated stream
/// rather than an unreplayable success, and every waiting duplicate fails
/// closed instead of hanging.
pub(crate) async fn finish_stream_terminal(
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
