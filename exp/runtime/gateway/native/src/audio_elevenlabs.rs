//! ElevenLabs transcription answers rendered onto the OpenAI audio wire.
//!
//! An ElevenLabs `/speech-to-text` answer is one JSON shape whatever the
//! caller asked for: `{text, language_code, words: [{text, type, start, end}],
//! audio_duration_secs}`. The gateway bills `audio_duration_secs` (bounded by
//! the admitted upload, exactly like the OpenAI duration meter) and renders
//! the caller's OpenAI `response_format` from it: `json` is the transcript
//! with an OpenAI-shaped duration usage, `text` the bare transcript, and
//! `verbose_json` the transcript with its language, duration, and word
//! timestamps (ElevenLabs reports no segments).

use serde_json::{json, Map, Value};

use crate::errors::Failure;
use crate::events::Usage;
use crate::route_audio::malformed;

/// The wire dialect admission names an ElevenLabs audio rung by.
pub(crate) const ELEVENLABS_AUDIO_DIALECT: &str = "elevenlabs_audio";

/// Validate one ElevenLabs transcription answer, bill its metered seconds
/// through `meter`, and render the caller's response format.
pub(crate) fn elevenlabs_transcription(
    payload: Value,
    response_format: Option<&str>,
    meter: impl Fn(f64) -> Result<Usage, Failure>,
) -> Result<(Vec<u8>, String, Usage), Failure> {
    let object = payload
        .as_object()
        .ok_or_else(|| malformed("transcription response is not an object"))?;
    let text = object
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| malformed("transcription response omitted its text"))?;
    let seconds = object
        .get("audio_duration_secs")
        .and_then(Value::as_f64)
        .ok_or_else(|| malformed("transcription response omitted its metered duration"))?;
    let usage = meter(seconds)?;
    let rendered = match response_format.unwrap_or("json") {
        "text" => {
            return Ok((
                text.as_bytes().to_vec(),
                "text/plain; charset=utf-8".to_string(),
                usage,
            ))
        }
        "verbose_json" => json!({
            "task": "transcribe",
            "language": object.get("language_code").cloned().unwrap_or(Value::Null),
            "duration": seconds,
            "text": text,
            "words": words(object),
        }),
        _ => json!({"text": text, "usage": {"type": "duration", "seconds": seconds}}),
    };
    let body = serde_json::to_vec(&rendered)
        .map_err(|_| malformed("transcription response is not JSON"))?;
    Ok((body, "application/json".to_string(), usage))
}

/// The answer's spoken words as OpenAI word timestamps (`spacing` and
/// `audio_event` entries are not words).
fn words(object: &Map<String, Value>) -> Vec<Value> {
    object
        .get("words")
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter(|entry| entry.get("type").and_then(Value::as_str) == Some("word"))
                .map(|entry| {
                    json!({
                        "word": entry.get("text").cloned().unwrap_or(Value::Null),
                        "start": entry.get("start").cloned().unwrap_or(Value::Null),
                        "end": entry.get("end").cloned().unwrap_or(Value::Null),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
#[path = "audio_elevenlabs_tests.rs"]
mod tests;
