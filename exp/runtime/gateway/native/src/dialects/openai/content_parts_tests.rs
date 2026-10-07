//! Typed `content` part arrays on the OpenAI-compatible wire. Mistral answers
//! this shape whenever the model reasons; the fixtures under `testdata/` are
//! verbatim streams captured from Mistral's own API on 2026-10-06 for "What
//! is 17*23? Answer with just the number.".

use crate::dialects::{drain_stream_fixture, Dialect, Normalizer};
use crate::events::simplified_event;
use crate::sse::SseEvent;
use serde_json::{json, Value};

const SMALL_REASONING_STREAM: &str =
    include_str!("testdata/mistral_small_2603_reasoning_stream.sse");
const LARGE_STREAM: &str = include_str!("testdata/mistral_large_4_stream.sse");
const LARGE_TOOL_STREAM: &str = include_str!("testdata/mistral_large_4_tool_stream.sse");

/// Decode one captured stream and return (answer text, reasoning text, events).
fn decode(fixture: &str) -> (String, String, Vec<Value>) {
    let (events, failure) =
        drain_stream_fixture(Dialect::OpenAiCompatible, &[fixture.as_bytes().to_vec()]);
    assert!(
        failure.is_none(),
        "fixture must decode cleanly: {failure:?}"
    );
    let mut text = String::new();
    let mut reasoning = String::new();
    for event in &events {
        match event["kind"].as_str() {
            Some("text_delta") => text.push_str(event["text"].as_str().unwrap()),
            Some("reasoning_text_delta") => reasoning.push_str(event["text"].as_str().unwrap()),
            _ => {}
        }
    }
    (text, reasoning, events)
}

/// One chunk frame whose delta carries `content`.
fn content_frame(content: Value) -> SseEvent {
    SseEvent {
        event: None,
        data: json!({"choices": [{"index": 0, "delta": {"content": content}}]}).to_string(),
    }
}

/// Feed one frame and return its simplified events.
fn feed(normalizer: &mut Normalizer, content: Value) -> Vec<Value> {
    normalizer
        .feed(&content_frame(content))
        .unwrap()
        .iter()
        .map(simplified_event)
        .collect()
}

#[test]
fn reasoning_stream_keeps_the_answer_that_shares_the_closing_thinking_chunk() {
    // The answer's first token "3" arrives in the same array chunk as the
    // final thinking fragment, then "91" as a plain string.
    let (text, reasoning, _) = decode(SMALL_REASONING_STREAM);
    assert_eq!(text, "391");
    assert!(reasoning.starts_with("I need to calculate 17 multiplied by 23."));
    assert!(reasoning.ends_with("equals 391."));
    assert!(
        !reasoning.contains("[{"),
        "parts are flattened to their text"
    );
}

#[test]
fn always_reasoning_model_answer_arrives_only_inside_an_array_chunk() {
    // mistral-large-4 streams its whole answer as a text part beside an
    // empty thinking part on the finishing chunk.
    let (text, reasoning, events) = decode(LARGE_STREAM);
    assert_eq!(text, "391");
    assert!(!reasoning.is_empty());
    let last_reasoning = events
        .iter()
        .rposition(|event| event["kind"] == "reasoning_text_delta")
        .unwrap();
    let first_text = events
        .iter()
        .position(|event| event["kind"] == "text_delta")
        .unwrap();
    assert!(last_reasoning < first_text, "reasoning precedes the answer");
}

#[test]
fn reasoning_then_tool_call_stream_keeps_reasoning_and_the_call() {
    let (text, reasoning, events) = decode(LARGE_TOOL_STREAM);
    assert_eq!(text, "");
    assert!(reasoning.contains("Paris"));
    let call = events
        .iter()
        .find(|event| event["kind"] == "tool_call_completed")
        .expect("completed tool call");
    assert_eq!(call["name"], "get_weather");
    assert_eq!(call["raw_arguments"], "{\"city\": \"Paris\"}");
}

#[test]
fn part_order_is_preserved_within_one_chunk() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let events = feed(
        &mut normalizer,
        json!([
            {"type": "thinking", "thinking": [{"type": "text", "text": "plan"}], "closed": true},
            {"type": "text", "text": "answer"},
            {"type": "thinking", "thinking": "string form"},
            {"type": "reference", "reference_ids": [1]},
            {"type": "thinking", "thinking": []},
            {"type": "text", "text": ""}
        ]),
    );
    assert_eq!(
        events,
        vec![
            json!({"kind": "reasoning_text_delta", "text": "plan"}),
            json!({"kind": "text_delta", "text": "answer"}),
            json!({"kind": "reasoning_text_delta", "text": "string form"}),
        ]
    );
}

#[test]
fn thinking_parts_on_a_replay_route_seal_the_route_carrier() {
    let mut normalizer = Normalizer::new_with_reasoning_content_route(
        Dialect::OpenAiCompatible,
        Some("route".into()),
    );
    let events = feed(
        &mut normalizer,
        json!([
            {"type": "thinking", "thinking": [{"type": "text", "text": "plan"}]},
            {"type": "text", "text": "answer"}
        ]),
    );
    assert_eq!(events[0]["kind"], "reasoning_content_delta");
    assert_eq!(events[0]["route_sha256"], "route");
    assert_eq!(events[1], json!({"kind": "text_delta", "text": "answer"}));
    assert_eq!(events.len(), 2);
}

#[test]
fn malformed_parts_fail_loudly() {
    for content in [
        json!(["bare string"]),
        json!([{"type": "text", "text": 3}]),
        json!([{"type": "thinking", "thinking": 3}]),
        json!([{"type": "thinking", "thinking": ["bare"]}]),
    ] {
        let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
        assert!(
            normalizer.feed(&content_frame(content.clone())).is_err(),
            "{content} must be malformed"
        );
    }
}
