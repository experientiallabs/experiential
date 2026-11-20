//! The native image-edit surface: the `POST /v1/images/edits` handler.
//!
//! The official SDKs send an edit as `multipart/form-data`: the image to edit
//! in an `image` part (or repeated `image[]` parts, up to 16 source images),
//! an optional `mask`, an optional `style_image` (style transfer), and text
//! fields. This handler parses the upload, base64-encodes the image parts
//! (Bedrock Stability and xAI take base64 inside the JSON body the control
//! plane freezes, so the images cross the bridge once, bounded below), and
//! hands the request to the shared images ladder through `admit_image_edits`.
//! It also keeps the raw parts: a rung whose admission names the OpenAI
//! multipart edit wire is sent the control plane's text fields plus these
//! parts as its own `multipart/form-data` body (`encode_edit_multipart`).

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

/// Image parts an edit may carry; each at most once except `image[]`.
const FILE_PARTS: [&str; 4] = ["image", "image[]", "mask", "style_image"];

/// The repeatable source-image part (the SDKs' list spelling of `image`).
const REPEATED_IMAGE_PART: &str = "image[]";

/// Most source images one edit carries (OpenAI's GPT image models take 16;
/// a rung that takes fewer refuses more at admission).
pub(crate) const MAXIMUM_EDIT_IMAGES: usize = 16;

/// One uploaded image part, kept in upload order for a multipart edit rung.
#[derive(Debug, Clone)]
pub(crate) struct EditPart {
    /// The caller's part name (`image`, `image[]`, `mask`, `style_image`).
    pub(crate) name: String,
    pub(crate) bytes: Vec<u8>,
    /// The sniffed (else declared) media type.
    pub(crate) content_type: String,
}

/// Longest text field accepted, in bytes: a 32,000-character prompt in
/// four-byte UTF-8.
const MAXIMUM_TEXT_FIELD_BYTES: usize = 128 * 1024;

/// Most text fields one upload may carry; every supported field fits under it.
const MAXIMUM_TEXT_FIELDS: usize = 40;

/// The edit route's body limit: three full-size image parts plus framing.
/// It bounds the whole upload, not each part: the parts are held in memory
/// and cross the bridge base64-encoded, so a 16-image edit fits when its
/// images together stay under about 62 MB (a larger upload answers 413).
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
    let (fields, files, parts) = match read {
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
        Some(&parts),
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

/// One parsed edit upload: text fields, base64 image parts for admission
/// (`image[]` as a list in upload order), and the raw parts in upload order.
type ParsedEdit = (Map<String, Value>, Map<String, Value>, Vec<EditPart>);

/// Parse one edit upload into its text fields and image parts. Refuses (by
/// name) a repeated (other than `image[]`), too many, or oversized image
/// part, an oversized or non-UTF-8 text field, and too many fields, while
/// streaming the parts.
async fn parse_edit_upload(
    state: &AppState,
    request: axum::extract::Request,
) -> Result<ParsedEdit, PublicError> {
    let mut multipart = Multipart::from_request(request, state)
        .await
        .map_err(|_| invalid_upload("The multipart upload is malformed.", None))?;
    let mut fields = Map::new();
    let mut files = Map::new();
    let mut parts: Vec<EditPart> = Vec::new();
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
            let repeated = name == REPEATED_IMAGE_PART;
            if files.contains_key(&name) && !repeated {
                return Err(invalid_upload(
                    &format!("Send the {param} part once."),
                    Some(&param),
                ));
            }
            if repeated
                && files
                    .get(&name)
                    .and_then(Value::as_array)
                    .is_some_and(|images| images.len() >= MAXIMUM_EDIT_IMAGES)
            {
                return Err(invalid_upload(
                    &format!("An image edit carries at most {MAXIMUM_EDIT_IMAGES} images."),
                    Some("image"),
                ));
            }
            let declared = field.content_type().map(str::to_string);
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
            let encoded = Value::String(base64::engine::general_purpose::STANDARD.encode(&bytes));
            if repeated {
                match files.get_mut(&name) {
                    Some(Value::Array(images)) => images.push(encoded),
                    _ => {
                        files.insert(name.clone(), Value::Array(vec![encoded]));
                    }
                }
            } else {
                files.insert(name.clone(), encoded);
            }
            let content_type = sniff_image_type(&bytes)
                .map(str::to_string)
                .or(declared)
                .unwrap_or_else(|| "application/octet-stream".to_string());
            parts.push(EditPart {
                name,
                bytes,
                content_type,
            });
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
    Ok((fields, files, parts))
}

/// Identify a PNG, JPEG, or WebP image from its leading bytes.
fn sniff_image_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        Some("image/jpeg")
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

/// The file extension a part is named with upstream (OpenAI reads the type
/// from the part, but a named file keeps the upload self-describing).
fn image_extension(content_type: &str) -> &'static str {
    match content_type {
        "image/jpeg" => "jpg",
        "image/webp" => "webp",
        _ => "png",
    }
}

fn escape_quoted(value: &str) -> String {
    value
        .chars()
        .filter(|c| !matches!(c, '"' | '\r' | '\n'))
        .collect()
}

/// Encode one OpenAI Images edit as `multipart/form-data`: the control
/// plane's text `fields` (numbers as their decimal text), then the source
/// images as `image` (one) or repeated `image[]` parts (several) in upload
/// order, then any `mask`. A `style_image` is never sent: admission refuses
/// it on this wire.
pub(crate) fn encode_edit_multipart(boundary: &str, fields: &Value, parts: &[EditPart]) -> Vec<u8> {
    let size = parts
        .iter()
        .map(|part| part.bytes.len() + 256)
        .sum::<usize>()
        + 1024;
    let mut body: Vec<u8> = Vec::with_capacity(size);
    if let Some(object) = fields.as_object() {
        for (name, value) in object {
            let text = match value {
                Value::String(text) => text.clone(),
                Value::Number(number) => number.to_string(),
                Value::Bool(flag) => flag.to_string(),
                _ => continue,
            };
            body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            body.extend_from_slice(
                format!(
                    "Content-Disposition: form-data; name=\"{}\"\r\n\r\n",
                    escape_quoted(name)
                )
                .as_bytes(),
            );
            body.extend_from_slice(text.as_bytes());
            body.extend_from_slice(b"\r\n");
        }
    }
    let images: Vec<&EditPart> = parts
        .iter()
        .filter(|part| part.name == "image" || part.name == REPEATED_IMAGE_PART)
        .collect();
    let image_name = if images.len() > 1 {
        REPEATED_IMAGE_PART
    } else {
        "image"
    };
    let masks = parts.iter().filter(|part| part.name == "mask");
    let named = images
        .iter()
        .copied()
        .enumerate()
        .map(|(index, part)| (image_name, format!("image-{index}"), part))
        .chain(masks.map(|part| ("mask", "mask".to_string(), part)));
    for (name, stem, part) in named {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"{name}\"; filename=\"{stem}.{}\"\r\nContent-Type: {}\r\n\r\n",
                image_extension(&part.content_type),
                escape_quoted(&part.content_type),
            )
            .as_bytes(),
        );
        body.extend_from_slice(&part.bytes);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    body
}
