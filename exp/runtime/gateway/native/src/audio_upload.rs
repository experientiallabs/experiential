//! Parse and measure one `POST /v1/audio/transcriptions` upload.
//!
//! The upload is a multipart `file` part (the official SDKs' shape) or a JSON
//! `input_audio` base64 object. The audio bytes stay in the data plane; the
//! text fields plus the audio's measured facts (size, name, type, SHA-256, and
//! the duration demuxed from its packets) are what admission validates. A
//! `source_url` is refused: the gateway never fetches a caller-named URL.

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, FromRequest, Multipart};
use axum::http::{header, StatusCode};
use base64::Engine as _;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::audio_duration;
use crate::errors::PublicError;
use crate::server::AppState;

/// The OpenAI transcription upload cap.
pub(crate) const MAXIMUM_AUDIO_UPLOAD_BYTES: usize = 25 * 1024 * 1024;

/// The longest transcription admitted, in milliseconds (4 hours; mirrors
/// `MAXIMUM_TRANSCRIPTION_AUDIO_MILLI`). A 25 MB upload reaches it only at
/// about 14 kbps, so real recordings fit, while a corrupt or crafted timeline
/// can never size a reservation past it.
const MAXIMUM_TRANSCRIPTION_AUDIO_MILLI: u64 = 4 * 60 * 60 * 1000;

/// The transcription route's body limit: a base64 JSON upload of the largest
/// audio file plus framing headroom.
pub(crate) fn transcription_body_limit() -> DefaultBodyLimit {
    DefaultBodyLimit::max(MAXIMUM_AUDIO_UPLOAD_BYTES / 3 * 4 + 1024 * 1024)
}

/// One parsed transcription upload: the audio bytes stay here.
pub(crate) struct AudioUpload {
    pub(crate) bytes: Vec<u8>,
    pub(crate) filename: String,
    pub(crate) content_type: String,
}

fn invalid_upload(message: &str) -> PublicError {
    PublicError::new(400, "invalid_request", message, "invalid_request_error")
}

/// Read the caller's transcription upload (multipart or JSON) into its text
/// fields and the audio bytes, which never cross the bridge.
pub(crate) async fn read_upload(
    state: &AppState,
    request: axum::extract::Request,
) -> Result<(Map<String, Value>, AudioUpload), PublicError> {
    let content_type = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let (fields, upload) = if content_type.starts_with("multipart/form-data") {
        parse_multipart(state, request).await?
    } else {
        parse_json_upload(request).await?
    };
    if upload.bytes.len() > MAXIMUM_AUDIO_UPLOAD_BYTES {
        return Err(PublicError::request_too_large());
    }
    Ok((fields, upload))
}

/// Measure one read upload and add its facts to the fields admission validates.
///
/// Runs after the timed read and under the caller's still-held permit: the
/// blocking probe cannot be cancelled, so it must never outlive the permit
/// that bounds concurrent uploads (a deadline here would release the permit
/// while the probe kept running).
pub(crate) async fn measure_upload(
    mut fields: Map<String, Value>,
    upload: AudioUpload,
) -> Result<(Map<String, Value>, AudioUpload), PublicError> {
    let extension = upload
        .filename
        .rsplit('.')
        .next()
        .unwrap_or_default()
        .to_string();
    // One owned copy for the blocking probe, which the demuxer consumes.
    let bytes = upload.bytes.clone();
    let measured =
        tokio::task::spawn_blocking(move || audio_duration::duration_milli(bytes, &extension))
            .await
            .ok()
            .flatten();
    if measured.is_some_and(|milli| milli > MAXIMUM_TRANSCRIPTION_AUDIO_MILLI) {
        let mut error = PublicError::new(
            400,
            "invalid_request",
            "The audio is longer than 4 hours. Split it and transcribe each part.",
            "invalid_request_error",
        );
        error.param = Some("file".to_string());
        return Err(error);
    }
    let Some(duration_milli) = measured else {
        let mut error = PublicError::new(
            400,
            "invalid_request",
            "The audio's duration could not be measured. Send one audio track as Opus (ogg or \
             webm), AAC (m4a or mp4), MP3, FLAC, or WAV.",
            "invalid_request_error",
        );
        error.param = Some("file".to_string());
        return Err(error);
    };
    let digest = Sha256::digest(&upload.bytes);
    let sha256: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    let mut file = Map::new();
    file.insert("bytes".to_string(), json!(upload.bytes.len()));
    file.insert("filename".to_string(), json!(upload.filename));
    file.insert("content_type".to_string(), json!(upload.content_type));
    file.insert("sha256".to_string(), json!(sha256));
    file.insert("duration_milli".to_string(), json!(duration_milli));
    fields.insert("file".to_string(), Value::Object(file));
    Ok((fields, upload))
}

async fn parse_multipart(
    state: &AppState,
    request: axum::extract::Request,
) -> Result<(Map<String, Value>, AudioUpload), PublicError> {
    let mut multipart = Multipart::from_request(request, state)
        .await
        .map_err(|_| invalid_upload("The multipart upload is malformed."))?;
    let mut fields = Map::new();
    let mut upload: Option<AudioUpload> = None;
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(error) if error.status() == StatusCode::PAYLOAD_TOO_LARGE => {
                return Err(PublicError::request_too_large())
            }
            Err(_) => {
                return Err(invalid_upload(
                    "The multipart upload is malformed or truncated.",
                ))
            }
        };
        let name = field.name().unwrap_or_default().to_string();
        if name == "file" {
            let filename = field.file_name().unwrap_or("audio").to_string();
            let content_type = field
                .content_type()
                .unwrap_or("application/octet-stream")
                .to_string();
            let bytes = field.bytes().await.map_err(|_| {
                invalid_upload("The multipart file part is malformed or truncated.")
            })?;
            upload = Some(AudioUpload {
                bytes: bytes.to_vec(),
                filename,
                content_type,
            });
            continue;
        }
        let text = field
            .text()
            .await
            .map_err(|_| invalid_upload("A multipart text field is malformed."))?;
        insert_form_field(&mut fields, &name, text)?;
    }
    let upload = upload
        .ok_or_else(|| invalid_upload("The upload must carry a multipart part named file."))?;
    Ok((fields, upload))
}

/// Most entries one `name[]` list field may carry (`timestamp_granularities`
/// has two distinct values).
const MAXIMUM_LIST_FIELD_ENTRIES: usize = 2;

/// Most text fields one upload may carry; every supported field fits well under it.
const MAXIMUM_TEXT_FIELDS: usize = 32;

/// File one text form field under its JSON name: `name[]` parts accumulate
/// into a list, and the numeric `temperature` is parsed so admission
/// validates a number rather than its text. Refuses (with the field name) a
/// list past its bound or an upload past the field-count bound while parsing,
/// so a flood of tiny parts never grows past them.
fn insert_form_field(
    fields: &mut Map<String, Value>,
    name: &str,
    text: String,
) -> Result<(), PublicError> {
    if let Some(list_name) = name.strip_suffix("[]") {
        if !fields.contains_key(list_name) && fields.len() >= MAXIMUM_TEXT_FIELDS {
            return Err(invalid_upload("The upload carries too many form fields."));
        }
        let entry = fields
            .entry(list_name.to_string())
            .or_insert_with(|| Value::Array(Vec::new()));
        if let Value::Array(items) = entry {
            if items.len() >= MAXIMUM_LIST_FIELD_ENTRIES
                || items
                    .iter()
                    .any(|item| item.as_str() == Some(text.as_str()))
            {
                let mut error = invalid_upload(&format!(
                    "{list_name} takes at most {MAXIMUM_LIST_FIELD_ENTRIES} distinct values."
                ));
                error.param = Some(list_name.to_string());
                return Err(error);
            }
            items.push(Value::String(text));
        }
        return Ok(());
    }
    if !fields.contains_key(name) && fields.len() >= MAXIMUM_TEXT_FIELDS {
        return Err(invalid_upload("The upload carries too many form fields."));
    }
    let value = if name == "temperature" {
        text.trim()
            .parse::<f64>()
            .ok()
            .map_or(Value::String(text), |number| json!(number))
    } else {
        Value::String(text)
    };
    fields.insert(name.to_string(), value);
    Ok(())
}

/// Parse a JSON upload: `input_audio: {data: <base64>, format}` beside the
/// text fields. A `source_url` is refused: the gateway never fetches a
/// caller-named URL on its own network.
async fn parse_json_upload(
    request: axum::extract::Request,
) -> Result<(Map<String, Value>, AudioUpload), PublicError> {
    let body = read_bounded_upload(request.into_body()).await?;
    let mut fields = match serde_json::from_slice::<Value>(&body) {
        Ok(Value::Object(fields)) => fields,
        _ => return Err(PublicError::invalid_json()),
    };
    if fields.contains_key("source_url") {
        return Err(PublicError::new(
            400,
            "unsupported_parameter",
            "source_url is not supported. Upload the audio as a multipart file or as input_audio base64.",
            "invalid_request_error",
        ));
    }
    let Some(Value::Object(input_audio)) = fields.remove("input_audio") else {
        return Err(invalid_upload(
            "The upload must carry a multipart file part or an input_audio object.",
        ));
    };
    let data = input_audio
        .get("data")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_upload("input_audio.data must be base64 audio."))?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|_| invalid_upload("input_audio.data is not valid base64."))?;
    let format = input_audio
        .get("format")
        .and_then(Value::as_str)
        .unwrap_or("wav")
        .to_ascii_lowercase();
    let content_type = match format.as_str() {
        "wav" => "audio/wav",
        "mp3" | "mpeg" => "audio/mpeg",
        "ogg" | "opus" => "audio/ogg",
        "flac" => "audio/flac",
        "webm" => "audio/webm",
        "m4a" | "mp4" => "audio/mp4",
        _ => {
            let mut error = invalid_upload(
                "input_audio.format must be one of flac, m4a, mp3, mp4, mpeg, ogg, opus, wav, \
                 or webm.",
            );
            error.param = Some("input_audio.format".to_string());
            return Err(error);
        }
    };
    let upload = AudioUpload {
        bytes,
        filename: format!("audio.{format}"),
        content_type: content_type.to_string(),
    };
    Ok((fields, upload))
}

async fn read_bounded_upload(body: Body) -> Result<Vec<u8>, PublicError> {
    axum::body::to_bytes(body, MAXIMUM_AUDIO_UPLOAD_BYTES / 3 * 4 + 1024 * 1024)
        .await
        .map(|bytes| bytes.to_vec())
        .map_err(|_| PublicError::request_too_large())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn form_fields_accumulate_lists_and_parse_temperature() {
        let mut fields = Map::new();
        insert_form_field(&mut fields, "timestamp_granularities[]", "word".to_string()).unwrap();
        insert_form_field(
            &mut fields,
            "timestamp_granularities[]",
            "segment".to_string(),
        )
        .unwrap();
        insert_form_field(&mut fields, "temperature", "0.2".to_string()).unwrap();
        insert_form_field(&mut fields, "language", "en".to_string()).unwrap();
        assert_eq!(
            fields["timestamp_granularities"],
            json!(["word", "segment"])
        );
        assert_eq!(fields["temperature"], json!(0.2));
        assert_eq!(fields["language"], json!("en"));
    }

    #[test]
    fn repeated_or_excess_list_entries_are_refused_while_parsing() {
        let mut fields = Map::new();
        let name = "timestamp_granularities[]";
        insert_form_field(&mut fields, name, "word".to_string()).unwrap();
        let repeated = insert_form_field(&mut fields, name, "word".to_string()).unwrap_err();
        assert_eq!(repeated.param.as_deref(), Some("timestamp_granularities"));
        insert_form_field(&mut fields, name, "segment".to_string()).unwrap();
        assert!(insert_form_field(&mut fields, name, "word".to_string()).is_err());
        for index in 0..MAXIMUM_TEXT_FIELDS {
            let _ = insert_form_field(&mut fields, &format!("f{index}"), String::new());
        }
        assert!(insert_form_field(&mut fields, "one_too_many", String::new()).is_err());
    }
}
