//! Cloud Code Assist envelope frames normalize exactly like the bare Gemini
//! frames they wrap, and usage comes from the inner `usageMetadata`.

use super::super::{drain_stream_fixture, drain_stream_fixture_with, Dialect, Normalizer};
use super::*;
use serde_json::json;

const GEMINI: Dialect = Dialect::GeminiGenerateContent;

fn sse(payload: &Value) -> Vec<u8> {
    format!("data: {payload}\n\n").into_bytes()
}

fn enveloped(inner: &Value) -> Value {
    json!({"response": inner, "traceId": "trace-1"})
}

fn code_assist_drain(chunks: &[Vec<u8>]) -> (Vec<Value>, Option<Failure>) {
    let mut normalizer = Normalizer::new(GEMINI);
    normalizer.set_code_assist_envelope(true);
    drain_stream_fixture_with(normalizer, GEMINI, chunks)
}

fn frames() -> Vec<Value> {
    vec![
        json!({"candidates":[{"content":{"role":"model","parts":[
            {"thought":true,"text":"thinking"},
            {"functionCall":{"name":"lookup","args":{"id":"a"}},"thoughtSignature":"sig=="}
        ]}}],"modelVersion":"gemini-3-flash"}),
        json!({"candidates":[{"content":{"role":"model","parts":[{"text":"done"}]},
            "finishReason":"STOP"}],
            "usageMetadata":{"promptTokenCount":11,"candidatesTokenCount":7,
                "thoughtsTokenCount":3,"cachedContentTokenCount":2,"totalTokenCount":21}}),
    ]
}

#[test]
fn enveloped_frames_normalize_like_the_bare_frames_they_wrap() {
    let bare: Vec<Vec<u8>> = frames().iter().map(sse).collect();
    let wrapped: Vec<Vec<u8>> = frames().iter().map(|f| sse(&enveloped(f))).collect();
    let expected = drain_stream_fixture(GEMINI, &bare);
    let observed = code_assist_drain(&wrapped);
    assert!(expected.1.is_none() && observed.1.is_none());
    assert_eq!(observed.0, expected.0);
    let usage = observed
        .0
        .iter()
        .rev()
        .find(|event| event["kind"] == "usage")
        .expect("usage from the inner usageMetadata");
    assert_eq!(usage["input_tokens"], json!(11));
    assert_eq!(usage["output_tokens"], json!(10));
}

#[test]
fn an_unflagged_normalizer_does_not_read_the_envelope() {
    let wrapped: Vec<Vec<u8>> = frames().iter().map(|f| sse(&enveloped(f))).collect();
    let (events, failure) = drain_stream_fixture(GEMINI, &wrapped);
    // Every frame reads as candidate-less, so nothing is produced.
    assert!(events.iter().all(|event| event["kind"] != "text_delta"));
    assert!(failure.is_some());
}

#[test]
fn an_error_frame_and_a_trace_only_frame_keep_their_bare_reading() {
    let error = sse(&json!({"error":{"code":503,"status":"UNAVAILABLE","message":"busy"}}));
    let (events, _) = code_assist_drain(&[error]);
    assert!(
        events.iter().any(|event| event["kind"] == "failed"),
        "{events:?}"
    );
    let trace_only = sse(&json!({"traceId":"t"}));
    let mut rest: Vec<Vec<u8>> = vec![trace_only];
    rest.extend(frames().iter().map(|f| sse(&enveloped(f))));
    let (events, failure) = code_assist_drain(&rest);
    assert!(failure.is_none());
    assert!(
        events.iter().any(|event| event["kind"] == "completed"),
        "{events:?}"
    );
}

#[test]
fn a_non_object_response_is_malformed() {
    let mut normalizer = Normalizer::new(GEMINI);
    normalizer.set_code_assist_envelope(true);
    let frame = crate::sse::SseEvent {
        data: json!({"response": "nope"}).to_string(),
        event: None,
    };
    let failure = normalizer.feed(&frame).expect_err("malformed");
    assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
}

#[test]
fn only_a_gemini_normalizer_honours_the_flag() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    normalizer.set_code_assist_envelope(true);
    assert!(!normalizer.code_assist_envelope);
}
