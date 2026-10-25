//! The asynchronous video surfaces: `/v1/videos*`, mirroring OpenAI's Videos API.
//!
//! Create, list, retrieve, and delete are thin relays: the handler
//! authenticates the bearer key over the bridge, shapes one JSON argument,
//! and the python control plane's `video_*` methods own every semantic
//! (ownership, reservation, failover, polling, settlement). Create accepts
//! the official SDK's multipart form (text fields plus an optional
//! `input_reference` file part) or a JSON body.
//!
//! Content is the one route the data plane executes itself: the plane answers
//! where the owned, completed asset lives, and the bytes stream from the
//! provider to the caller without crossing the bridge. Provider headers go to
//! that first URL only; redirects are followed by hand, re-validated against
//! the upstream address policy, and never carry them.

use std::collections::HashMap;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, FromRequest, Multipart, Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::Response;
use base64::Engine as _;
use serde_json::{json, Map, Value};

use crate::encode::compact_json;
use crate::errors::PublicError;
use crate::respond::{error_response, json_response, with_app_identity};
use crate::route_batches::{acquire_batch_permit, pre_authenticate};
use crate::server::AppState;

/// Largest first-frame image accepted (mirrors `MAXIMUM_REFERENCE_IMAGE_BYTES`).
const MAXIMUM_REFERENCE_IMAGE_BYTES: usize = 10 * 1024 * 1024;

/// Longest text form field accepted: the 4,000-character prompt in four-byte UTF-8.
const MAXIMUM_TEXT_FIELD_BYTES: usize = 16 * 1024;

/// Most text fields one create may carry; every supported field fits well under it.
const MAXIMUM_TEXT_FIELDS: usize = 16;

/// Redirect hops followed when downloading one asset.
const MAXIMUM_CONTENT_REDIRECTS: usize = 3;

/// The create route's body limit: a base64 JSON data URL of the largest image
/// plus framing headroom.
pub(crate) fn videos_body_limit() -> DefaultBodyLimit {
    DefaultBodyLimit::max(MAXIMUM_REFERENCE_IMAGE_BYTES / 3 * 4 + 1024 * 1024)
}

fn invalid(message: &str, param: Option<&str>) -> PublicError {
    let mut error = PublicError::new(400, "invalid_request", message, "invalid_request_error");
    error.param = param.map(str::to_string);
    error
}

/// Relay one plane call whose success is a JSON object answered with 200.
async fn plane_json(state: &AppState, method: &'static str, payload: Value) -> Response {
    let rendered = match state.bridge.call(method, compact_json(&payload)).await {
        Ok(rendered) => rendered,
        Err(error) => return error_response(&error),
    };
    match serde_json::from_str::<Value>(&rendered) {
        Ok(body) => json_response(StatusCode::OK, &body, &[]),
        Err(_) => error_response(&PublicError::internal()),
    }
}

/// Read one multipart text field, refusing it by name past the field cap.
async fn read_text_field(
    mut field: axum::extract::multipart::Field<'_>,
    name: &str,
) -> Result<String, PublicError> {
    let mut bytes: Vec<u8> = Vec::new();
    while let Some(chunk) = field
        .chunk()
        .await
        .map_err(|_| invalid("A multipart text field is malformed.", None))?
    {
        if bytes.len() + chunk.len() > MAXIMUM_TEXT_FIELD_BYTES {
            return Err(invalid(
                &format!("The {name} field is longer than {MAXIMUM_TEXT_FIELD_BYTES} bytes."),
                Some(name),
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    String::from_utf8(bytes).map_err(|_| invalid("A multipart text field is not UTF-8.", None))
}

/// Parse the SDK's multipart create into text fields and the optional image part.
async fn parse_multipart(
    state: &AppState,
    request: axum::extract::Request,
) -> Result<(Map<String, Value>, Option<Value>), PublicError> {
    let mut multipart = Multipart::from_request(request, state)
        .await
        .map_err(|_| invalid("The multipart form is malformed.", None))?;
    let mut fields = Map::new();
    let mut reference: Option<Value> = None;
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(error) if error.status() == StatusCode::PAYLOAD_TOO_LARGE => {
                return Err(PublicError::request_too_large())
            }
            Err(_) => {
                return Err(invalid(
                    "The multipart form is malformed or truncated.",
                    None,
                ))
            }
        };
        let name = field.name().unwrap_or_default().to_string();
        if name == "input_reference" && field.file_name().is_some() {
            if reference.is_some() {
                return Err(invalid(
                    "Send exactly one input_reference file part.",
                    Some("input_reference"),
                ));
            }
            let content_type = field
                .content_type()
                .unwrap_or("application/octet-stream")
                .to_ascii_lowercase();
            let bytes = field
                .bytes()
                .await
                .map_err(|_| invalid("The input_reference part is truncated.", None))?;
            if bytes.len() > MAXIMUM_REFERENCE_IMAGE_BYTES {
                return Err(invalid(
                    "input_reference must be at most 10 MiB.",
                    Some("input_reference"),
                ));
            }
            reference = Some(json!({
                "content_type": content_type,
                "data": base64::engine::general_purpose::STANDARD.encode(&bytes),
            }));
            continue;
        }
        let text = read_text_field(field, &name).await?;
        if fields.len() >= MAXIMUM_TEXT_FIELDS || fields.contains_key(&name) {
            return Err(invalid(
                "The form repeats a field or carries too many fields.",
                Some(&name),
            ));
        }
        fields.insert(name, Value::String(text));
    }
    Ok((fields, reference))
}

/// Read the create body (multipart or JSON) into fields and the optional image part.
async fn read_create(
    state: &AppState,
    request: axum::extract::Request,
) -> Result<(Map<String, Value>, Option<Value>), PublicError> {
    let content_type = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if content_type.starts_with("multipart/form-data") {
        return parse_multipart(state, request).await;
    }
    let limit = MAXIMUM_REFERENCE_IMAGE_BYTES / 3 * 4 + 1024 * 1024;
    let bytes = axum::body::to_bytes(request.into_body(), limit)
        .await
        .map_err(|_| PublicError::request_too_large())?;
    match serde_json::from_slice::<Value>(&bytes) {
        Ok(Value::Object(fields)) => Ok((fields, None)),
        _ => Err(invalid(
            "The request body must be a JSON object or multipart form data.",
            None,
        )),
    }
}

/// `POST /v1/videos`: create one video job.
pub(crate) async fn videos_create(
    State(state): State<AppState>,
    headers: HeaderMap,
    request: axum::extract::Request,
) -> Response {
    let _permit = match acquire_batch_permit(&state, &headers).await {
        Ok(permit) => permit,
        Err(error) => return error_response(&error),
    };
    let key = match pre_authenticate(&state, &headers).await {
        Ok(key) => key,
        Err(error) => return error_response(&error),
    };
    let (fields, reference) = match read_create(&state, request).await {
        Ok(parsed) => parsed,
        Err(error) => return error_response(&error),
    };
    let mut payload = json!({
        "bearer_key": key,
        "fields": Value::Object(fields),
        "reference": reference.unwrap_or(Value::Null),
    });
    with_app_identity(&mut payload, &headers);
    plane_json(&state, "video_create", payload).await
}

/// `GET /v1/videos`: list the caller's videos.
pub(crate) async fn videos_list(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    let key = match pre_authenticate(&state, &headers).await {
        Ok(key) => key,
        Err(error) => return error_response(&error),
    };
    let mut payload = json!({"bearer_key": key});
    for name in ["limit", "after", "order"] {
        if let Some(value) = query.get(name) {
            payload[name] = Value::String(value.clone());
        }
    }
    plane_json(&state, "video_list", payload).await
}

/// `GET /v1/videos/{video_id}`: return one owned video, refreshed from its provider.
pub(crate) async fn videos_retrieve(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(video_id): Path<String>,
) -> Response {
    let key = match pre_authenticate(&state, &headers).await {
        Ok(key) => key,
        Err(error) => return error_response(&error),
    };
    plane_json(
        &state,
        "video_retrieve",
        json!({"bearer_key": key, "video_id": video_id}),
    )
    .await
}

/// `DELETE /v1/videos/{video_id}`: delete one owned, finished video.
pub(crate) async fn videos_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(video_id): Path<String>,
) -> Response {
    let key = match pre_authenticate(&state, &headers).await {
        Ok(key) => key,
        Err(error) => return error_response(&error),
    };
    plane_json(
        &state,
        "video_delete",
        json!({"bearer_key": key, "video_id": video_id}),
    )
    .await
}

/// One download location the plane resolved for an owned, completed video.
struct ContentFetch {
    url: String,
    headers: Vec<(String, String)>,
    content_type: String,
}

/// Parse the plane's `{url, headers, content_type}` answer.
fn content_fetch(rendered: &str) -> Option<ContentFetch> {
    let parsed: Value = serde_json::from_str(rendered).ok()?;
    let url = parsed.get("url")?.as_str()?.to_string();
    let headers = parsed
        .get("headers")
        .and_then(Value::as_object)
        .map(|map| {
            map.iter()
                .filter_map(|(name, value)| Some((name.clone(), value.as_str()?.to_string())))
                .collect()
        })
        .unwrap_or_default();
    let content_type = parsed
        .get("content_type")
        .and_then(Value::as_str)
        .unwrap_or("video/mp4")
        .to_string();
    Some(ContentFetch {
        url,
        headers,
        content_type,
    })
}

fn asset_unavailable() -> PublicError {
    PublicError::new(
        404,
        "video_expired",
        "The video asset is no longer available from its provider.",
        "invalid_request_error",
    )
}

fn asset_failed() -> PublicError {
    PublicError::new(
        502,
        "upstream_error",
        "The video asset could not be downloaded from its provider; retry shortly.",
        "api_error",
    )
}

/// Resolve one redirect `Location` against the URL that answered it.
fn redirect_target(current: &str, response: &reqwest::Response) -> Option<String> {
    let location = response.headers().get(header::LOCATION)?.to_str().ok()?;
    let base = reqwest::Url::parse(current).ok()?;
    base.join(location).ok().map(String::from)
}

/// Download one asset, following redirects without the provider headers.
async fn open_asset(
    state: &AppState,
    fetch: &ContentFetch,
) -> Result<reqwest::Response, PublicError> {
    let mut url = fetch.url.clone();
    let bound = state.request_timeout.min(Duration::from_secs(120));
    for hop in 0..=MAXIMUM_CONTENT_REDIRECTS {
        let mut builder = state.http.get(&url).map_err(|_| asset_failed())?;
        if hop == 0 {
            for (name, value) in &fetch.headers {
                builder = builder.header(name.as_str(), value.as_str());
            }
        }
        let response = tokio::time::timeout(bound, builder.send())
            .await
            .map_err(|_| asset_failed())?
            .map_err(|_| asset_failed())?;
        let status = response.status();
        if status.is_redirection() {
            url = redirect_target(&url, &response).ok_or_else(asset_failed)?;
            continue;
        }
        if status == reqwest::StatusCode::NOT_FOUND || status == reqwest::StatusCode::GONE {
            return Err(asset_unavailable());
        }
        if !status.is_success() {
            return Err(asset_failed());
        }
        return Ok(response);
    }
    Err(asset_failed())
}

/// `GET /v1/videos/{video_id}/content`: stream one owned, completed video.
pub(crate) async fn videos_content(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(video_id): Path<String>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    let key = match pre_authenticate(&state, &headers).await {
        Ok(key) => key,
        Err(error) => return error_response(&error),
    };
    let mut payload = json!({"bearer_key": key, "video_id": video_id});
    if let Some(variant) = query.get("variant") {
        payload["variant"] = Value::String(variant.clone());
    }
    let rendered = match state
        .bridge
        .call("video_content", compact_json(&payload))
        .await
    {
        Ok(rendered) => rendered,
        Err(error) => return error_response(&error),
    };
    let Some(fetch) = content_fetch(&rendered) else {
        return error_response(&PublicError::internal());
    };
    let upstream = match open_asset(&state, &fetch).await {
        Ok(upstream) => upstream,
        Err(error) => return error_response(&error),
    };
    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, fetch.content_type.as_str());
    if let Some(length) = upstream.content_length() {
        builder = builder.header(header::CONTENT_LENGTH, length);
    }
    builder
        .body(Body::from_stream(upstream.bytes_stream()))
        .unwrap_or_else(|_| error_response(&PublicError::internal()))
}

#[cfg(test)]
#[path = "route_videos_tests.rs"]
mod tests;
