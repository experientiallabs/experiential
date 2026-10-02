//! Inline tests for byte-exact Messages round trips (zero-argument tool calls,
//! non-ASCII thinking and long signatures), split from `tests` so each test
//! file stays within the repository line budget.

use super::*;
use crate::events::CompletedToolCall;

#[test]
fn zero_argument_tool_calls_encode_as_an_empty_object_on_both_paths() {
    // The completion-time `{}` seed (live wire, 2026-08-28) must serve a valid
    // tool_use block streaming and non-streaming.
    let events = vec![
        Event::ToolCallStarted {
            custom: false,
            namespace: None,
            caller: None,
            index: 0,
            call_id: "call-1".to_string(),
            name: "get_time".to_string(),
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: String::new(),
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: "{}".to_string(),
        },
        Event::ToolCallCompleted {
            index: 0,
            call: CompletedToolCall {
                namespace: None,
                caller: None,
                call_id: "call-1".to_string(),
                name: "get_time".to_string(),
                provider_item_id: None,
                provider_status: None,
                raw_arguments: "{}".to_string(),
                custom: false,
            },
        },
        Event::Completed,
    ];
    let mut encoder = MessagesSseEncoder::new("request-abc", "coding");
    let mut frames = encoder.start().expect("starts");
    for event in &events {
        frames.extend(encoder.feed(event).expect("streams the call"));
    }
    assert!(frames
        .iter()
        .any(|frame| frame.contains("\"type\":\"tool_use\"")));
    assert!(frames
        .last()
        .expect("frames")
        .starts_with("event: message_stop"));
    let streamed: String = frames
        .iter()
        .filter(|frame| frame.contains("\"input_json_delta\""))
        .map(|frame| {
            let data = frame.split("data: ").nth(1).expect("data line").trim();
            let payload: Value = serde_json::from_str(data).expect("json frame");
            payload["delta"]["partial_json"]
                .as_str()
                .expect("partial_json")
                .to_string()
        })
        .collect();
    assert_eq!(streamed, "{}");

    let aggregated = completed_messages_body("request-abc", "coding", &events).expect("aggregates");
    assert_eq!(
        aggregated.body["content"][0],
        json!({"type": "tool_use", "id": "call-1", "name": "get_time", "input": {}})
    );
    assert_eq!(aggregated.body["stop_reason"], json!("tool_use"));
}

#[test]
fn thinking_text_and_a_long_signature_survive_both_paths_byte_for_byte() {
    // The signature is an opaque value the provider verifies on replay, so
    // Unicode escaping or truncation would break every continued conversation.
    let thinking = "Grüß 事實 مرحبا 🤔🧠 σκέψη ⇒ done";
    let signature = format!("Eq{}", "A0b/+=".repeat(700));
    let redacted = "R3".repeat(1500);
    let events = vec![
        Event::ThinkingDelta {
            index: 0,
            delta: thinking.to_string(),
        },
        Event::ThinkingSignature {
            index: 0,
            signature: signature.clone(),
        },
        Event::RedactedThinking {
            index: 1,
            data: redacted.clone(),
        },
        Event::Completed,
    ];
    let mut encoder = MessagesSseEncoder::new("request-abc", "coding");
    let mut frames = encoder.start().expect("starts");
    for event in &events {
        frames.extend(encoder.feed(event).expect("streams thinking"));
    }
    let payloads: Vec<Value> = frames
        .iter()
        .filter_map(|frame| frame.split("data: ").nth(1))
        .map(|data| serde_json::from_str(data.trim()).expect("json frame"))
        .collect();
    let delta_of = |kind: &str, field: &str| -> String {
        payloads
            .iter()
            .filter(|payload| payload["delta"]["type"] == json!(kind))
            .map(|payload| payload["delta"][field].as_str().expect("field").to_string())
            .collect()
    };
    assert_eq!(delta_of("thinking_delta", "thinking"), thinking);
    assert_eq!(delta_of("signature_delta", "signature"), signature);

    let aggregated = completed_messages_body("request-abc", "coding", &events).expect("aggregates");
    assert_eq!(aggregated.body["content"][0]["thinking"], json!(thinking));
    assert_eq!(aggregated.body["content"][0]["signature"], json!(signature));
    assert_eq!(aggregated.body["content"][1]["data"], json!(redacted));
}
