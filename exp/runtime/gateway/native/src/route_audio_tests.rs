//! Unit tests for the audio surfaces' pure helpers.

use super::*;

fn admission(modes: &[&str], response_format: Option<&str>) -> AudioAdmission {
    AudioAdmission {
        request_id: "request-1".to_string(),
        alias: "voice".to_string(),
        alias_revision_id: "revision-one".to_string(),
        exact_model_id: "model-exact".to_string(),
        route_reason: "direct".to_string(),
        route: Vec::new(),
        billing_modes: modes.iter().map(|mode| (*mode).to_string()).collect(),
        input_characters: Some(12),
        response_format: response_format.map(str::to_string),
        maximum_audio_milli: Some(4_000),
        token_ceilings: vec![Some([100, 10])],
        maximum_total_attempts: 8,
        maximum_same_deployment_attempts: 2,
    }
}

#[test]
fn sse_speech_reassembles_audio_and_bills_the_done_usage() {
    let audio = base64::engine::general_purpose::STANDARD;
    let body = format!(
        "data: {}\n\ndata: {}\n\ndata: {}\n\n",
        json!({"type": "speech.audio.delta", "audio": audio.encode(b"ab")}),
        json!({"type": "speech.audio.delta", "audio": audio.encode(b"cd")}),
        json!({"type": "speech.audio.done", "usage": {"input_tokens": 7, "output_tokens": 90, "total_tokens": 97}}),
    );
    let (bytes, usage) = reassemble_speech_events(body.as_bytes()).expect("valid stream");
    assert_eq!(bytes, b"abcd");
    assert_eq!(
        (usage.input_tokens, usage.output_tokens),
        (Some(7), Some(90))
    );
    assert!(usage.billed_units.is_none());
}

#[test]
fn sse_speech_without_its_usage_or_audio_is_malformed() {
    let audio = base64::engine::general_purpose::STANDARD;
    let unbilled = format!(
        "data: {}\n\n",
        json!({"type": "speech.audio.delta", "audio": audio.encode(b"ab")})
    );
    let silent = format!(
        "data: {}\n\n",
        json!({"type": "speech.audio.done", "usage": {"input_tokens": 1, "output_tokens": 1}})
    );
    for body in [unbilled, silent] {
        let failure = reassemble_speech_events(body.as_bytes()).expect_err("malformed");
        assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
        assert!(failure.failover_eligible);
    }
}

#[test]
fn transcription_bills_metered_seconds_and_renders_text_on_request() {
    let payload = json!({"text": "hello world", "usage": {"type": "duration", "seconds": 3.2}});
    let (body, content_type, usage) =
        public_transcription(payload.clone(), &admission(&["units"], Some("text")), 0)
            .expect("valid");
    assert_eq!(body, b"hello world");
    assert!(content_type.starts_with("text/plain"));
    let billed = usage.billed_units.expect("billed seconds");
    assert_eq!(
        (billed.kind.as_str(), billed.quantity_milli),
        ("audio_second", 3_200)
    );
    let (json_body, json_type, _) =
        public_transcription(payload.clone(), &admission(&["units"], None), 0).expect("valid");
    assert_eq!(
        serde_json::from_slice::<Value>(&json_body).unwrap(),
        payload
    );
    assert_eq!(json_type, "application/json");
}

#[test]
fn verbose_duration_meters_seconds_and_token_rungs_bill_tokens() {
    let verbose = json!({"text": "hi", "duration": 1.0001, "segments": []});
    let (_, _, usage) =
        public_transcription(verbose, &admission(&["units"], Some("verbose_json")), 0)
            .expect("valid");
    assert_eq!(usage.billed_units.expect("seconds").quantity_milli, 1_001);
    let tokens =
        json!({"text": "hi", "usage": {"type": "tokens", "input_tokens": 50, "output_tokens": 3}});
    let (_, _, usage) =
        public_transcription(tokens, &admission(&["tokens"], None), 0).expect("valid");
    assert_eq!(
        (usage.input_tokens, usage.output_tokens),
        (Some(50), Some(3))
    );
}

#[test]
fn transcription_without_its_meter_or_text_is_malformed() {
    let unmetered = json!({"text": "hi"});
    let textless = json!({"usage": {"type": "duration", "seconds": 1}});
    for payload in [unmetered, textless] {
        let failure =
            public_transcription(payload, &admission(&["units"], None), 0).expect_err("malformed");
        assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
    }
    let token_rung_without_tokens =
        json!({"text": "hi", "usage": {"type": "duration", "seconds": 1}});
    assert!(
        public_transcription(token_rung_without_tokens, &admission(&["tokens"], None), 0).is_err()
    );
}

#[test]
fn multipart_encoding_repeats_list_fields_and_carries_the_audio() {
    let upload = AudioUpload {
        bytes: b"RIFFaudio".to_vec(),
        filename: "clip\".wav".to_string(),
        content_type: "audio/wav".to_string(),
    };
    let fields = json!({
        "model": "whisper-1",
        "response_format": "verbose_json",
        "timestamp_granularities[]": ["word", "segment"],
    });
    let body = String::from_utf8(encode_multipart("B", &fields, &upload)).expect("utf-8 test body");
    assert!(body.contains("name=\"model\"\r\n\r\nwhisper-1\r\n"));
    assert_eq!(
        body.matches("name=\"timestamp_granularities[]\"").count(),
        2
    );
    assert!(body.contains(
        "filename=\"clip.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFFaudio\r\n--B--\r\n"
    ));
}

#[test]
fn a_meter_longer_than_the_admitted_audio_is_malformed() {
    let long = json!({"text": "hi", "usage": {"type": "duration", "seconds": 4.5}});
    let failure =
        public_transcription(long, &admission(&["units"], None), 0).expect_err("over the hold");
    assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
}

#[test]
fn token_usage_above_the_reserved_ceilings_is_malformed() {
    let over =
        json!({"text": "hi", "usage": {"type": "tokens", "input_tokens": 50, "output_tokens": 11}});
    let failure =
        public_transcription(over, &admission(&["tokens"], None), 0).expect_err("over the hold");
    assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
}

#[test]
fn only_audio_content_types_are_billed_as_speech() {
    for audio in [
        "audio/mpeg",
        "Audio/Ogg; codecs=opus",
        "application/ogg",
        "application/octet-stream",
    ] {
        assert!(is_audio_type(audio), "{audio}");
    }
    for document in [
        "application/json",
        "application/problem+json",
        "application/xml",
        "text/html; charset=utf-8",
        "",
    ] {
        assert!(!is_audio_type(document), "{document}");
    }
}
