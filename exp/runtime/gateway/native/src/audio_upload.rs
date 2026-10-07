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
use tokio::sync::OwnedSemaphorePermit;

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
/// Runs after the timed read and takes the caller's permit into the blocking
/// probe, which cannot be cancelled: the permit is released only when the
/// probe exits (never by a dropped handler or a deadline), and is handed back
/// with the measured upload.
pub(crate) async fn measure_upload(
    mut fields: Map<String, Value>,
    upload: AudioUpload,
    permit: OwnedSemaphorePermit,
) -> Result<(Map<String, Value>, AudioUpload, OwnedSemaphorePermit), PublicError> {
    let extension = upload
        .filename
        .rsplit('.')
        .next()
        .unwrap_or_default()
        .to_string();
    // One owned copy for the blocking probe, which the demuxer consumes. The
    // permit travels with it: a cancelled handler cannot release capacity
    // while the uncancellable probe still runs, and it comes back on success.
    let bytes = upload.bytes.clone();
    let (measured, permit) = match tokio::task::spawn_blocking(move || {
        (audio_duration::duration_milli(bytes, &extension), permit)
    })
    .await
    {
        Ok((measured, permit)) => (measured, Some(permit)),
        Err(_) => (None, None),
    };
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
    let permit = permit.ok_or_else(PublicError::internal)?;
    Ok((fields, upload, permit))
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
            if upload.is_some() {
                let mut error = invalid_upload("The upload must carry exactly one file part.");
                error.param = Some("file".to_string());
                return Err(error);
            }
            let filename = field.file_name().unwrap_or("audio").to_string();
            let content_type = field
                .content_type()
                .unwrap_or("application/octet-stream")
                .to_string();
            // The canonical contract's limits, enforced before anything is
            // copied into the probe or the bridge argument.
            if filename.chars().count() > MAXIMUM_FILENAME_CHARACTERS
                || content_type.chars().count() > MAXIMUM_CONTENT_TYPE_CHARACTERS
            {
                let mut error = invalid_upload(&format!(
                    "The file part's filename must be at most {MAXIMUM_FILENAME_CHARACTERS} \
                     characters and its content type at most {MAXIMUM_CONTENT_TYPE_CHARACTERS}."
                ));
                error.param = Some("file".to_string());
                return Err(error);
            }
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
        let text = read_text_field(field, &name).await?;
        insert_form_field(&mut fields, &name, text)?;
    }
    let upload = upload
        .ok_or_else(|| invalid_upload("The upload must carry a multipart part named file."))?;
    Ok((fields, upload))
}

/// Longest upload filename accepted (`TranscriptionRequest.audio_filename`).
const MAXIMUM_FILENAME_CHARACTERS: usize = 256;

/// Longest upload content type accepted (`TranscriptionRequest.audio_content_type`).
const MAXIMUM_CONTENT_TYPE_CHARACTERS: usize = 128;

/// Longest text form field accepted, in bytes: the longest supported field
/// (`prompt`, 4,096 characters) in four-byte UTF-8, so no single field can
/// carry most of the upload body across the bridge.
const MAXIMUM_TEXT_FIELD_BYTES: usize = 16 * 1024;

/// Read one multipart text field, refusing it (by name) the moment it passes
/// [`MAXIMUM_TEXT_FIELD_BYTES`] rather than after buffering all of it.
async fn read_text_field(
    mut field: axum::extract::multipart::Field<'_>,
    name: &str,
) -> Result<String, PublicError> {
    let mut bytes: Vec<u8> = Vec::new();
    while let Some(chunk) = field
        .chunk()
        .await
        .map_err(|_| invalid_upload("A multipart text field is malformed."))?
    {
        if bytes.len() + chunk.len() > MAXIMUM_TEXT_FIELD_BYTES {
            let mut error = invalid_upload(&format!(
                "The {name} field is longer than {MAXIMUM_TEXT_FIELD_BYTES} bytes."
            ));
            error.param = Some(name.trim_end_matches("[]").to_string());
            return Err(error);
        }
        bytes.extend_from_slice(&chunk);
    }
    String::from_utf8(bytes).map_err(|_| invalid_upload("A multipart text field is not UTF-8."))
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
    if fields.len() >= MAXIMUM_TEXT_FIELDS {
        return Err(invalid_upload("The upload carries too many form fields."));
    }
    let value = if name == "temperature" {
        text.trim()
            .parse::<f64>()
            .ok()
            // NaN, infinities, and overflows stay text so validation refuses
            // them, instead of becoming JSON null (the provider default).
            .filter(|number| number.is_finite())
            .map_or(Value::String(text), |number| json!(number))
    } else {
        Value::String(text)
    };
    if fields.contains_key(name) {
        // Last-value-wins could route or bill a model the caller did not mean.
        let mut error = invalid_upload(&format!("The {name} field appears more than once."));
        error.param = Some(name.to_string());
        return Err(error);
    }
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
    // A closed object: a misspelled or extra key is refused, never ignored.
    if let Some(unknown) = input_audio
        .keys()
        .find(|key| !matches!(key.as_str(), "data" | "format"))
    {
        let mut error = invalid_upload(&format!(
            "input_audio.{unknown} is not supported; input_audio takes data and format."
        ));
        error.param = Some(format!("input_audio.{unknown}"));
        return Err(error);
    }
    let data = input_audio
        .get("data")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_upload("input_audio.data must be base64 audio."))?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|_| invalid_upload("input_audio.data is not valid base64."))?;
    // Omitted means WAV; a supplied format must be one of the named strings.
    let format = match input_audio.get("format") {
        None => "wav".to_string(),
        Some(Value::String(format)) => format.to_ascii_lowercase(),
        Some(_) => String::new(),
    };
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
        for invalid in ["NaN", "inf", "-Infinity", "1e999"] {
            let mut single = Map::new();
            insert_form_field(&mut single, "temperature", invalid.to_string()).unwrap();
            assert_eq!(single["temperature"], json!(invalid), "{invalid}");
        }
        let repeated = insert_form_field(&mut fields, "language", "fr".to_string()).unwrap_err();
        assert_eq!(repeated.param.as_deref(), Some("language"));
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
