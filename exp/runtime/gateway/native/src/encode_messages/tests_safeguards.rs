//! Claude Code auto-mode safeguards: Anthropic sends its verdicts for the
//! request's `safeguards` field only on the final `message_delta`, at
//! `delta.safeguard_results`, keyed by tool_use id. These tests drive the
//! verbatim captures under `dialects/anthropic/testdata/` (api.anthropic.com,
//! 2026-10-08) through the Anthropic normalizer and every public encoder.

use super::*;
use crate::dialects::{Dialect, FrameDecoder, Normalizer};
use crate::encode::{completed_chat_body_with_ignored, ChatSseEncoder};
use crate::encode_responses::{completed_responses_body, ResponsesEnvelope, ResponsesSseEncoder};

const TOOL_USE_STREAM: &str =
    include_str!("../dialects/anthropic/testdata/anthropic_stream_tool_use.sse");
const END_TURN_STREAM: &str =
    include_str!("../dialects/anthropic/testdata/anthropic_stream_end_turn.sse");
const TOOL_USE_ID: &str = "toolu_01Cx7MmCyUmN8FSkrJrnpFMN";

/// Decode one raw Anthropic capture into normalized events.
fn normalize(stream: &str) -> Vec<Event> {
    let mut decoder = FrameDecoder::new(Dialect::AnthropicMessages);
    let mut normalizer = Normalizer::new(Dialect::AnthropicMessages);
    let mut events = Vec::new();
    let mut frames = decoder.feed(stream.as_bytes()).expect("capture decodes");
    frames.extend(decoder.finish().expect("capture closes"));
    for frame in frames {
        events.extend(normalizer.feed(&frame).expect("capture normalizes"));
    }
    assert!(events.last().is_some_and(Event::is_terminal));
    events
}

/// The upstream `message_delta.delta.safeguard_results` value of a capture.
fn upstream_results(stream: &str) -> Value {
    let line = stream
        .lines()
        .find(|line| line.starts_with("data: ") && line.contains("\"message_delta\""))
        .expect("capture has a message_delta");
    let payload: Value = serde_json::from_str(&line["data: ".len()..]).expect("JSON frame");
    payload["delta"]["safeguard_results"].clone()
}

/// Stream events through the public Messages SSE encoder.
fn messages_frames(events: &[Event]) -> Vec<String> {
    let mut encoder = MessagesSseEncoder::new("request-safeguards", "claude");
    let mut frames = encoder.start().expect("starts");
    for event in events {
        frames.extend(encoder.feed(event).expect("streams"));
    }
    frames
}

/// The JSON payload of the first frame with the given event name.
fn frame_payload(frames: &[String], name: &str) -> Value {
    let prefix = format!("event: {name}\n");
    let frame = frames
        .iter()
        .find(|frame| frame.starts_with(&prefix))
        .expect("frame present");
    let data = frame
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .expect("frame data");
    serde_json::from_str(data).expect("JSON frame")
}

#[test]
fn tool_use_capture_relays_safeguard_results_on_message_delta_unchanged() {
    let upstream = upstream_results(TOOL_USE_STREAM);
    assert!(upstream.is_array());
    let events = normalize(TOOL_USE_STREAM);
    assert!(events
        .iter()
        .any(|event| matches!(event, Event::SafeguardResults(value) if *value == upstream)));
    let frames = messages_frames(&events);
    let delta = frame_payload(&frames, "message_delta");
    assert_eq!(delta["delta"]["safeguard_results"], upstream);
    assert_eq!(
        compact_json(&delta["delta"]["safeguard_results"]),
        compact_json(&upstream)
    );
    assert_eq!(delta["delta"]["stop_reason"], json!("tool_use"));
    // The verdicts are keyed by tool_use id, so the id must reach the
    // caller exactly as the provider issued it.
    let block = frame_payload(&frames, "content_block_start");
    assert_eq!(block["content_block"]["type"], json!("tool_use"));
    assert_eq!(block["content_block"]["id"], json!(TOOL_USE_ID));
    assert!(upstream[0]["status"]["tool_uses"]
        .get(TOOL_USE_ID)
        .is_some());
}

#[test]
fn end_turn_capture_relays_its_empty_verdict_map() {
    let upstream = upstream_results(END_TURN_STREAM);
    assert_eq!(upstream[0]["status"]["tool_uses"], json!({}));
    let frames = messages_frames(&normalize(END_TURN_STREAM));
    let delta = frame_payload(&frames, "message_delta");
    assert_eq!(delta["delta"]["safeguard_results"], upstream);
    assert_eq!(delta["delta"]["stop_reason"], json!("end_turn"));
}

#[test]
fn an_upstream_without_safeguard_results_adds_no_key() {
    let stripped = TOOL_USE_STREAM
        .lines()
        .map(|line| match line.strip_prefix("data: ") {
            Some(data) if data.contains("\"message_delta\"") => {
                let mut payload: Value = serde_json::from_str(data).unwrap();
                payload["delta"]
                    .as_object_mut()
                    .unwrap()
                    .remove("safeguard_results");
                format!("data: {payload}")
            }
            _ => line.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n");
    let events = normalize(&stripped);
    assert!(!events
        .iter()
        .any(|event| matches!(event, Event::SafeguardResults(_))));
    let frames = messages_frames(&events);
    let delta = frame_payload(&frames, "message_delta");
    assert!(delta["delta"].get("safeguard_results").is_none());
    let body = completed_messages_body("request-safeguards", "claude", &events)
        .expect("aggregates")
        .body;
    assert!(body.get("safeguard_results").is_none());
}

#[test]
fn non_streaming_message_carries_safeguard_results_top_level() {
    let upstream = upstream_results(TOOL_USE_STREAM);
    let events = normalize(TOOL_USE_STREAM);
    let body = completed_messages_body("request-safeguards", "claude", &events)
        .expect("aggregates")
        .body;
    assert_eq!(body["safeguard_results"], upstream);
    assert_eq!(body["content"][0]["id"], json!(TOOL_USE_ID));
    assert_eq!(body["stop_reason"], json!("tool_use"));
}

#[test]
fn chat_and_responses_callers_never_see_safeguard_results() {
    let events = normalize(TOOL_USE_STREAM);
    let mut chat = ChatSseEncoder::new_with_ignored("request", "claude", 1, true, Vec::new());
    let mut chat_frames = chat.start().expect("starts");
    for event in &events {
        chat_frames.extend(chat.feed(event).expect("streams"));
    }
    assert!(chat_frames.iter().any(|frame| frame.contains(TOOL_USE_ID)));
    assert!(!chat_frames.join("").contains("safeguard_results"));
    let chat_body = completed_chat_body_with_ignored("request", "claude", 1, &events, &[], false)
        .expect("aggregates")
        .body;
    assert!(!chat_body.to_string().contains("safeguard_results"));

    let mut responses =
        ResponsesSseEncoder::new("request", "claude", 1, ResponsesEnvelope::default());
    let mut responses_frames = responses.start().expect("starts");
    for event in &events {
        responses_frames.extend(responses.feed(event).expect("streams"));
    }
    assert!(!responses_frames.join("").contains("safeguard_results"));
    let responses_body = completed_responses_body(
        "request",
        "claude",
        1,
        ResponsesEnvelope::default(),
        &events,
    )
    .expect("aggregates")
    .body;
    assert!(!responses_body.to_string().contains("safeguard_results"));
}
