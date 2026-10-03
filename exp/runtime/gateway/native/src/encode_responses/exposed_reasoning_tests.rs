//! A rung marked `reasoning_output_exposed` returns its plaintext reasoning on
//! the Responses surface as the route reasoning item's summary text, beside
//! the sealed carrier, while every other rung keeps it inside the gateway.

use super::*;

fn exposed_envelope() -> ResponsesEnvelope {
    ResponsesEnvelope {
        include_encrypted_reasoning: true,
        reasoning_output_exposed: true,
        ..ResponsesEnvelope::default()
    }
}

fn reasoning(delta: &str) -> Event {
    Event::ReasoningContentDelta {
        route_sha256: "a".repeat(64),
        delta: delta.to_string(),
    }
}

/// One GLM-style tool turn: two reasoning fragments, then one tool call.
fn exposed_tool_events() -> Vec<Event> {
    vec![
        reasoning("check the "),
        reasoning("weather"),
        Event::ToolCallStarted {
            custom: false,
            namespace: None,
            caller: None,
            index: 0,
            call_id: "call-one".to_string(),
            name: "lookup".to_string(),
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
                call_id: "call-one".to_string(),
                name: "lookup".to_string(),
                raw_arguments: "{}".to_string(),
                provider_item_id: None,
                provider_status: None,
                custom: false,
            },
        },
        Event::Completed,
    ]
}

fn payloads(frames: &[String]) -> Vec<Value> {
    frames
        .iter()
        .map(|frame| {
            let data = frame.split_once("data: ").expect("frame carries data").1;
            serde_json::from_str(data.trim_end()).expect("frame data is JSON")
        })
        .collect()
}

#[test]
fn exposed_stream_carries_summary_text_and_the_sealed_carrier() {
    let mut encoder =
        ResponsesSseEncoder::new("request-1", "glm", 1_700_000_000, exposed_envelope());
    encoder.start().expect("stream start must encode");
    encoder
        .set_reasoning_content_carrier("authenticated-carrier-v2".to_string())
        .expect("carrier must attach");
    let mut frames = Vec::new();
    for event in &exposed_tool_events() {
        frames.extend(encoder.feed(event).expect("Responses event must encode"));
    }
    let events = payloads(&frames);
    let types: Vec<&str> = events
        .iter()
        .map(|event| event["type"].as_str().expect("event type"))
        .collect();
    assert_eq!(
        &types[..5],
        [
            "response.output_item.added",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.delta",
            "response.output_item.added",
        ]
    );
    let deltas: Vec<&str> = events
        .iter()
        .filter(|event| event["type"] == "response.reasoning_summary_text.delta")
        .map(|event| event["delta"].as_str().expect("delta text"))
        .collect();
    assert_eq!(deltas, ["check the ", "weather"]);
    let text_done = events
        .iter()
        .find(|event| event["type"] == "response.reasoning_summary_text.done")
        .expect("summary text closes");
    assert_eq!(text_done["text"], json!("check the weather"));
    let reasoning_done = events
        .iter()
        .find(|event| {
            event["type"] == "response.output_item.done" && event["item"]["type"] == "reasoning"
        })
        .expect("reasoning item closes");
    assert_eq!(
        reasoning_done["item"]["summary"],
        json!([{"type": "summary_text", "text": "check the weather"}])
    );
    assert_eq!(
        reasoning_done["item"]["encrypted_content"],
        json!("authenticated-carrier-v2")
    );
}

#[test]
fn exposed_completed_body_carries_summary_text_and_the_sealed_carrier() {
    let completed = completed_responses_body_with_carrier(
        "request-1",
        "glm",
        1_700_000_000,
        exposed_envelope(),
        &exposed_tool_events(),
        Some("authenticated-carrier-v2"),
    )
    .expect("completed body must encode");
    let item = &completed.body["output"][0];
    assert_eq!(item["type"], json!("reasoning"));
    assert_eq!(
        item["summary"],
        json!([{"type": "summary_text", "text": "check the weather"}])
    );
    assert_eq!(item["encrypted_content"], json!("authenticated-carrier-v2"));
}

#[test]
fn exposed_answer_without_tools_returns_reasoning_with_no_carrier() {
    let events = vec![
        reasoning("think"),
        Event::TextDelta("answer".to_string()),
        Event::Completed,
    ];
    let completed = completed_responses_body(
        "request-1",
        "glm",
        1_700_000_000,
        exposed_envelope(),
        &events,
    )
    .expect("a tool-less turn needs no carrier");
    let output = completed.body["output"]
        .as_array()
        .expect("output is an array");
    assert_eq!(
        output[0]["summary"],
        json!([{"type": "summary_text", "text": "think"}])
    );
    assert!(output[0].get("encrypted_content").is_none());
    assert_eq!(output[1]["content"][0]["text"], json!("answer"));
}

#[test]
fn unexposed_rung_keeps_the_reasoning_item_empty() {
    let envelope = ResponsesEnvelope {
        include_encrypted_reasoning: true,
        ..ResponsesEnvelope::default()
    };
    let completed = completed_responses_body_with_carrier(
        "request-1",
        "glm",
        1_700_000_000,
        envelope,
        &exposed_tool_events(),
        Some("authenticated-carrier-v2"),
    )
    .expect("completed body must encode");
    assert_eq!(completed.body["output"][0]["summary"], json!([]));
    assert!(!completed.body.to_string().contains("weather"));
}

#[test]
fn the_exposure_flag_is_never_read_from_the_request() {
    let envelope: ResponsesEnvelope =
        serde_json::from_value(json!({"reasoning_output_exposed": true}))
            .expect("unknown request fields are ignored");
    assert!(!envelope.reasoning_output_exposed);
}
