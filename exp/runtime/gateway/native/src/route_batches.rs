//! The asynchronous batch surfaces: `/v1/batches*` and `/v1/files*`.
//!
//! Every handler is a thin relay: it authenticates nothing itself, shapes one
//! JSON argument carrying the caller's bearer key, and forwards it to the
//! python control plane's batch methods over the bridge. The plane returns a
//! uniform `{status, body}` envelope that maps directly onto the HTTP
//! response, so all batch semantics live in exactly one place.

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Multipart, Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::Response;
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::encode::compact_json;
use crate::errors::PublicError;
use crate::respond::{bearer_key, error_response, json_response, read_body};
use crate::server::AppState;

/// Native transport cap for one uploaded batch input file. The engine's own
/// contract allows 100MB; the transport adds headroom for multipart framing.
pub(crate) const MAXIMUM_FILE_UPLOAD_BYTES: usize = 110 * 1024 * 1024;

/// The multipart limit layer for the files upload route.
pub(crate) fn files_body_limit() -> DefaultBodyLimit {
    DefaultBodyLimit::max(MAXIMUM_FILE_UPLOAD_BYTES)
}

/// Relay one plane call and map its `{status, body}` envelope onto HTTP.
async fn plane_call(state: &AppState, method: &'static str, payload: Value) -> Response {
    let argument = compact_json(&payload);
    let rendered = match state.bridge.call(method, argument).await {
        Ok(rendered) => rendered,
        Err(error) => return error_response(&error),
    };
    envelope_response(&rendered)
}

/// Map one rendered `{status, body}` envelope string onto an HTTP response.
fn envelope_response(rendered: &str) -> Response {
    let parsed: Value = match serde_json::from_str(rendered) {
        Ok(parsed) => parsed,
        Err(_) => return error_response(&PublicError::internal()),
    };
    let status = parsed
        .get("status")
        .and_then(Value::as_u64)
        .and_then(|status| u16::try_from(status).ok())
        .and_then(|status| StatusCode::from_u16(status).ok())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let body = parsed.get("body").cloned().unwrap_or(Value::Null);
    json_response(status, &body, &[])
}

/// Hold one aggregate admission permit for the duration of a batch handler.
///
/// Batch bodies share the same concurrency budget as synchronous requests,
/// so a burst of large uploads cannot bypass the plane's memory admission.
async fn acquire_batch_permit(
    state: &AppState,
) -> Result<tokio::sync::OwnedSemaphorePermit, PublicError> {
    match tokio::time::timeout(state.request_timeout, state.permits.clone().acquire_owned()).await {
        Ok(Ok(permit)) => Ok(permit),
        Ok(Err(_)) => Err(PublicError::draining()),
        Err(_) => Err(PublicError::new(
            429,
            "unavailable_route",
            "The gateway is at capacity; retry the batch request shortly.",
            "api_error",
        )),
    }
}

/// Authenticate the presented key over the bridge before any body work.
///
/// Mirrors the synchronous routes: authentication happens before the plane
/// buffers request content, so an unauthenticated caller can never make the
/// gateway allocate an upload.
async fn pre_authenticate(state: &AppState, headers: &HeaderMap) -> Result<String, PublicError> {
    let key = bearer_key(headers)?;
    let authenticate = compact_json(&json!({"raw_key": key.clone()}));
    state.bridge.call("authenticate", authenticate).await?;
    Ok(key)
}

/// Shape the shared argument object carrying the caller's bearer key.
fn keyed_payload(headers: &HeaderMap) -> Result<Map<String, Value>, PublicError> {
    let key = bearer_key(headers)?;
    let mut payload = Map::new();
    payload.insert("bearer_key".to_string(), Value::String(key));
    Ok(payload)
}

/// `POST /v1/batches`: submit one batch job from an uploaded input file.
pub(crate) async fn batches_create(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let _permit = match acquire_batch_permit(&state).await {
        Ok(permit) => permit,
        Err(error) => return error_response(&error),
    };
    let key = match pre_authenticate(&state, &headers).await {
        Ok(key) => key,
        Err(error) => return error_response(&error),
    };
    let mut payload = Map::new();
    payload.insert("bearer_key".to_string(), Value::String(key));
    let raw = match read_body(body).await {
        Ok(raw) => raw,
        Err(error) => return error_response(&error),
    };
    let parsed: Value = match serde_json::from_slice(&raw) {
        Ok(Value::Object(fields)) => Value::Object(fields),
        _ => {
            return error_response(&PublicError::new(
                400,
                "invalid_request",
                "The request body must be a JSON object.",
                "invalid_request_error",
            ))
        }
    };
    for field in ["input_file_id", "endpoint", "metadata"] {
        if let Some(value) = parsed.get(field) {
            payload.insert(field.to_string(), value.clone());
        }
    }
    plane_call(&state, "batch_create", Value::Object(payload)).await
}

/// Pagination query parameters for the batch listing.
#[derive(Debug, Deserialize)]
pub(crate) struct BatchListQuery {
    limit: Option<u32>,
    after: Option<String>,
}

/// `GET /v1/batches`: list the caller's batch jobs, newest first.
pub(crate) async fn batches_list(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<BatchListQuery>,
) -> Response {
    let _permit = match acquire_batch_permit(&state).await {
        Ok(permit) => permit,
        Err(error) => return error_response(&error),
    };
    let mut payload = match keyed_payload(&headers) {
        Ok(payload) => payload,
        Err(error) => return error_response(&error),
    };
    if let Some(limit) = query.limit {
        payload.insert("limit".to_string(), json!(limit));
    }
    if let Some(after) = query.after {
        payload.insert("after".to_string(), Value::String(after));
    }
    plane_call(&state, "batch_list", Value::Object(payload)).await
}

/// `GET /v1/batches/{batch_id}`: return one owned batch object.
pub(crate) async fn batches_retrieve(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(batch_id): Path<String>,
) -> Response {
    let _permit = match acquire_batch_permit(&state).await {
        Ok(permit) => permit,
        Err(error) => return error_response(&error),
    };
    let mut payload = match keyed_payload(&headers) {
        Ok(payload) => payload,
        Err(error) => return error_response(&error),
    };
    payload.insert("batch_id".to_string(), Value::String(batch_id));
    plane_call(&state, "batch_retrieve", Value::Object(payload)).await
}

/// `POST /v1/batches/{batch_id}/cancel`: request cancellation of one job.
pub(crate) async fn batches_cancel(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(batch_id): Path<String>,
) -> Response {
    let _permit = match acquire_batch_permit(&state).await {
        Ok(permit) => permit,
        Err(error) => return error_response(&error),
    };
    let mut payload = match keyed_payload(&headers) {
        Ok(payload) => payload,
        Err(error) => return error_response(&error),
    };
    payload.insert("batch_id".to_string(), Value::String(batch_id));
    plane_call(&state, "batch_cancel", Value::Object(payload)).await
}

/// Map one multipart read failure onto an honest client error.
fn multipart_error_response(error: &axum::extract::multipart::MultipartError) -> Response {
    let status = error.status();
    let public = if status == StatusCode::PAYLOAD_TOO_LARGE {
        PublicError::request_too_large()
    } else {
        PublicError::new(
            400,
            "invalid_request",
            "The multipart upload is malformed or truncated.",
            "invalid_request_error",
        )
    };
    error_response(&public)
}

/// `POST /v1/files`: store one multipart batch input file.
pub(crate) async fn files_create(
    State(state): State<AppState>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Response {
    let _permit = match acquire_batch_permit(&state).await {
        Ok(permit) => permit,
        Err(error) => return error_response(&error),
    };
    let key = match pre_authenticate(&state, &headers).await {
        Ok(key) => key,
        Err(error) => return error_response(&error),
    };
    let mut payload = Map::new();
    payload.insert("bearer_key".to_string(), Value::String(key));
    let mut purpose: Option<String> = None;
    let mut filename: Option<String> = None;
    let mut content: Option<Vec<u8>> = None;
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(error) => return multipart_error_response(&error),
        };
        match field.name().unwrap_or_default() {
            "purpose" => match field.text().await {
                Ok(text) => purpose = Some(text),
                Err(error) => return multipart_error_response(&error),
            },
            "file" => {
                filename = field.file_name().map(str::to_string);
                match field.bytes().await {
                    Ok(bytes) => content = Some(bytes.to_vec()),
                    Err(error) => return multipart_error_response(&error),
                }
            }
            _ => {}
        }
    }
    let Some(content) = content else {
        return error_response(&PublicError::new(
            400,
            "invalid_request",
            "The upload must carry a multipart part named file.",
            "invalid_request_error",
        ));
    };
    if content.len() > MAXIMUM_FILE_UPLOAD_BYTES {
        return error_response(&PublicError::request_too_large());
    }
    payload.insert(
        "purpose".to_string(),
        Value::String(purpose.unwrap_or_default()),
    );
    payload.insert(
        "filename".to_string(),
        Value::String(filename.unwrap_or_else(|| "batch.jsonl".to_string())),
    );
    payload.insert(
        "content_b64".to_string(),
        Value::String(base64::engine::general_purpose::STANDARD.encode(content)),
    );
    plane_call(&state, "file_create", Value::Object(payload)).await
}

/// `GET /v1/files/{file_id}`: return one owned file's metadata object.
pub(crate) async fn files_retrieve(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(file_id): Path<String>,
) -> Response {
    let _permit = match acquire_batch_permit(&state).await {
        Ok(permit) => permit,
        Err(error) => return error_response(&error),
    };
    let mut payload = match keyed_payload(&headers) {
        Ok(payload) => payload,
        Err(error) => return error_response(&error),
    };
    payload.insert("file_id".to_string(), Value::String(file_id));
    plane_call(&state, "file_retrieve", Value::Object(payload)).await
}

/// `GET /v1/files/{file_id}/content`: stream one owned file's raw bytes.
pub(crate) async fn files_content(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(file_id): Path<String>,
) -> Response {
    let _permit = match acquire_batch_permit(&state).await {
        Ok(permit) => permit,
        Err(error) => return error_response(&error),
    };
    let mut payload = match keyed_payload(&headers) {
        Ok(payload) => payload,
        Err(error) => return error_response(&error),
    };
    payload.insert("file_id".to_string(), Value::String(file_id));
    let argument = compact_json(&Value::Object(payload));
    let rendered = match state.bridge.call("file_content", argument).await {
        Ok(rendered) => rendered,
        Err(error) => return error_response(&error),
    };
    let parsed: Value = match serde_json::from_str(&rendered) {
        Ok(parsed) => parsed,
        Err(_) => return error_response(&PublicError::internal()),
    };
    let status = parsed.get("status").and_then(Value::as_u64).unwrap_or(500);
    if status != 200 {
        return envelope_response(&rendered);
    }
    let encoded = parsed
        .get("body")
        .and_then(|body| body.get("content_b64"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(encoded) else {
        return error_response(&PublicError::internal());
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/jsonl")
        .body(Body::from(decoded))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}
