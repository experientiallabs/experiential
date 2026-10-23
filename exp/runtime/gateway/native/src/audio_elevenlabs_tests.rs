//! Unit tests for rendering ElevenLabs transcription answers.

use super::*;
use crate::events::BilledUnits;

fn answer() -> Value {
    json!({
        "language_code": "eng",
        "language_probability": 0.99,
        "text": "Hello there.",
        "words": [
            {"text": "Hello", "type": "word", "start": 0.0, "end": 0.4},
            {"text": " ", "type": "spacing", "start": 0.4, "end": 0.45},
            {"text": "there.", "type": "word", "start": 0.45, "end": 0.9},
        ],
        "audio_duration_secs": 2.25,
    })
}

fn seconds_meter(seconds: f64) -> Result<Usage, Failure> {
    Ok(Usage {
        billed_units: Some(BilledUnits {
            kind: "audio_second".to_string(),
            variant: String::new(),
            quantity_milli: (seconds * 1000.0).ceil() as u64,
        }),
        ..Usage::default()
    })
}

#[test]
fn json_bills_audio_duration_and_carries_an_openai_duration_usage() {
    let (body, content_type, usage) =
        elevenlabs_transcription(answer(), None, seconds_meter).expect("valid");
    assert_eq!(content_type, "application/json");
    let rendered: Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(
        rendered,
        json!({"text": "Hello there.", "usage": {"type": "duration", "seconds": 2.25}})
    );
    assert_eq!(usage.billed_units.expect("units").quantity_milli, 2_250);
}

#[test]
fn text_and_verbose_json_render_from_the_one_answer() {
    let (text, content_type, _) =
        elevenlabs_transcription(answer(), Some("text"), seconds_meter).expect("valid");
    assert_eq!(text, b"Hello there.");
    assert!(content_type.starts_with("text/plain"));
    let (verbose, _, _) =
        elevenlabs_transcription(answer(), Some("verbose_json"), seconds_meter).expect("valid");
    let rendered: Value = serde_json::from_slice(&verbose).expect("json");
    assert_eq!(rendered["task"], "transcribe");
    assert_eq!(rendered["language"], "eng");
    assert_eq!(rendered["duration"], 2.25);
    assert_eq!(
        rendered["words"],
        json!([
            {"word": "Hello", "start": 0.0, "end": 0.4},
            {"word": "there.", "start": 0.45, "end": 0.9},
        ])
    );
}

#[test]
fn an_answer_without_text_or_duration_is_malformed_and_the_meter_bounds_it() {
    let mut no_duration = answer();
    no_duration
        .as_object_mut()
        .expect("object")
        .remove("audio_duration_secs");
    assert!(elevenlabs_transcription(no_duration, None, seconds_meter).is_err());
    let mut no_text = answer();
    no_text.as_object_mut().expect("object").remove("text");
    assert!(elevenlabs_transcription(no_text, None, seconds_meter).is_err());
    let refused = elevenlabs_transcription(answer(), None, |_| {
        Err(malformed(
            "transcription meter exceeds the admitted audio duration",
        ))
    });
    assert!(refused.is_err());
}
