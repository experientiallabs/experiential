//! The native image-edit surface: the `POST /v1/images/edits` handler.
//!
//! The official SDKs send an edit as `multipart/form-data`: the image to edit
//! in an `image` part (or a one-entry `image[]` list), an optional `mask`, an
//! optional `style_image` (style transfer), and text fields. This handler
//! parses the upload, base64-encodes the image parts (Bedrock Stability
//! models take base64 inside the JSON body the control plane freezes and
//! signs, so the images cross the bridge once, bounded below), and hands the
//! request to the shared images ladder through `admit_image_edits`.

use std::sync::atomic::Ordering;
use std::time::Instant;

use axum::extract::{DefaultBodyLimit, FromRequest, Multipart, State};
use axum::http::{header, StatusCode};
use axum::response::Response;
use base64::Engine as _;
use serde_json::{json, Map, Value};

use crate::admission::pre_admission_permit;
use crate::encode::compact_json;
use crate::errors::PublicError;
use crate::respond::{bearer_key, error_response};
use crate::route_images::serve_images;
use crate::server::AppState;

#[cfg(test)]
#[path = "route_image_edits_tests.rs"]
mod tests;

/// Largest decoded image part accepted (`MAXIMUM_EDIT_IMAGE_BYTES`).
pub(crate) const MAXIMUM_EDIT_IMAGE_BYTES: usize = 20 * 1024 * 1024;

/// Image parts an edit may carry; each at most once.
const FILE_PARTS: [&str; 4] = ["image", "image[]", "mask", "style_image"];

/// Longest text field accepted, in bytes: a 32,000-character prompt in
/// four-byte UTF-8.
const MAXIMUM_TEXT_FIELD_BYTES: usize = 128 * 1024;

/// Most text fields one upload may carry; every supported field fits under it.
const MAXIMUM_TEXT_FIELDS: usize = 40;

/// The edit route's body limit: three full-size image parts plus framing.
pub(crate) fn image_edits_body_limit() -> DefaultBodyLimit {
    DefaultBodyLimit::max(3 * MAXIMUM_EDIT_IMAGE_BYTES + 2 * 1024 * 1024)
}

fn invalid_upload(message: &str, param: Option<&str>) -> PublicError {
    let mut error = PublicError::new(400, "invalid_request", message, "invalid_request_error");
    error.param = param.map(str::to_string);
    error
}

pub(crate) async fn image_edits(
    State(state): State<AppState>,
    request: axum::extract::Request,
) -> Response {
    state.handled_requests.fetch_add(1, Ordering::Relaxed);
    let started = Instant::now();
    let deadline = started + state.request_timeout;
    let headers = request.headers().clone();
    let raw_key = match bearer_key(&headers) {
        Ok(key) => key,
        Err(error) => return error_response(&error),
    };
    let authenticate = compact_json(&json!({"raw_key": raw_key}));
    if let Err(error) = state.bridge.call("authenticate", authenticate).await {
        return error_response(&error);
    }
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !content_type.starts_with("multipart/form-data") {
        return error_response(&invalid_upload(
            "Send an image edit as multipart/form-data with an image part.",
            Some("image"),
        ));
    }
    // Bound buffered uploads and their lifetime before parsing any body bytes.
    // The shared ladder retains this same permit through terminal accounting.
    let permit = match pre_admission_permit(&state, deadline).await {
        Ok(permit) => permit,
        Err(error) => return error_response(&error),
    };
    let read = tokio::time::timeout_at(deadline.into(), parse_edit_upload(&state, request)).await;
    let (fields, files) = match read {
        Ok(Ok(parsed)) => parsed,
        Ok(Err(error)) => return error_response(&error),
        Err(_) => return error_response(&upload_deadline_error()),
    };
    if Instant::now() >= deadline {
        return error_response(&upload_deadline_error());
    }
    let admit_value = json!({
        "raw_key": raw_key,
        "upload": {"fields": fields, "files": files},
    });
    serve_images(
        &state,
        &headers,
        started,
        deadline,
        &raw_key,
        "admit_image_edits",
        admit_value,
        Some(permit),
    )
    .await
}

fn upload_deadline_error() -> PublicError {
    PublicError::new(
        408,
        "request_timeout",
        "The image upload did not arrive before the request deadline.",
        "invalid_request_error",
    )
}

/// Parse one edit upload into its text fields and base64 image parts.
/// Refuses (by name) a repeated or oversized image part, an oversized or
/// non-UTF-8 text field, and too many fields, while streaming the parts.
async fn parse_edit_upload(
    state: &AppState,
    request: axum::extract::Request,
) -> Result<(Map<String, Value>, Map<String, Value>), PublicError> {
    let mut multipart = Multipart::from_request(request, state)
        .await
        .map_err(|_| invalid_upload("The multipart upload is malformed.", None))?;
    let mut fields = Map::new();
    let mut files = Map::new();
    loop {
        let mut field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(error) if error.status() == StatusCode::PAYLOAD_TOO_LARGE => {
                return Err(PublicError::request_too_large())
            }
            Err(_) => {
                return Err(invalid_upload(
                    "The multipart upload is malformed or truncated.",
                    None,
                ))
            }
        };
        let name = field.name().unwrap_or_default().to_string();
        if FILE_PARTS.contains(&name.as_str()) || field.file_name().is_some() {
            let param = name.trim_end_matches("[]").to_string();
            if files.contains_key(&name) {
                return Err(invalid_upload(
                    &format!("Send exactly one {param} part; every edit model takes one."),
                    Some(&param),
                ));
            }
            let mut bytes: Vec<u8> = Vec::new();
            while let Some(chunk) = field.chunk().await.map_err(|_| {
                invalid_upload("A multipart image part is malformed or truncated.", None)
            })? {
                if bytes.len() + chunk.len() > MAXIMUM_EDIT_IMAGE_BYTES {
                    return Err(invalid_upload(
                        &format!(
                            "The {param} part is larger than {} MB.",
                            MAXIMUM_EDIT_IMAGE_BYTES / (1024 * 1024)
                        ),
                        Some(&param),
                    ));
                }
                bytes.extend_from_slice(&chunk);
            }
            if bytes.is_empty() {
                return Err(invalid_upload(
                    &format!("The {param} part is empty."),
                    Some(&param),
                ));
            }
            let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
            files.insert(name, Value::String(encoded));
            continue;
        }
        if fields.len() >= MAXIMUM_TEXT_FIELDS {
            return Err(invalid_upload(
                "The upload carries too many form fields.",
                None,
            ));
        }
        if fields.contains_key(&name) {
            return Err(invalid_upload(
                &format!("Send the {name} field once."),
                Some(&name),
            ));
        }
        let mut bytes: Vec<u8> = Vec::new();
        while let Some(chunk) = field
            .chunk()
            .await
            .map_err(|_| invalid_upload("A multipart text field is malformed.", None))?
        {
            if bytes.len() + chunk.len() > MAXIMUM_TEXT_FIELD_BYTES {
                return Err(invalid_upload(
                    &format!("The {name} field is longer than {MAXIMUM_TEXT_FIELD_BYTES} bytes."),
                    Some(&name),
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        let text = String::from_utf8(bytes)
            .map_err(|_| invalid_upload("A multipart text field is not UTF-8.", Some(&name)))?;
        fields.insert(name, Value::String(text));
    }
    Ok((fields, files))
}
