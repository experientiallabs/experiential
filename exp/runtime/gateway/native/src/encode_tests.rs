//! Unit tests for the public Chat Completions encoders in `encode.rs`.

use super::*;

fn fireworks_tool_events() -> Vec<Event> {
    vec![
        Event::ReasoningContentDelta {
            route_sha256: "a".repeat(64),
            delta: "hidden provider reasoning".to_string(),
        },
        Event::ToolCallStarted {
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
            call: crate::events::CompletedToolCall {
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

#[test]
fn fireworks_chat_reasoning_round_trips_only_as_sealed_carrier() {
    let events = fireworks_tool_events();
    let mut stream =
        ChatSseEncoder::new_with_ignored("request-1", "coding", 1_700_000_000, false, Vec::new());
    stream.set_reasoning_content_carrier("authenticated-carrier-v2".to_string());
    let mut frames = stream.start().expect("stream start must encode");
    for event in &events {
        frames.extend(stream.feed(event).expect("event must encode"));
    }
    let public = frames.join("");
    assert!(!public.contains("hidden provider reasoning"));
    assert!(public.contains("authenticated-carrier-v2"));

    let completed = completed_chat_body_with_carrier(
        "request-1",
        "coding",
        1_700_000_000,
        &events,
        &[],
        Some("authenticated-carrier-v2"),
        false,
    )
    .expect("completed body must preserve the carrier");
    assert_eq!(
        completed.body["choices"][0]["message"]["reasoning_content"],
        json!("authenticated-carrier-v2")
    );
    assert!(!completed
        .body
        .to_string()
        .contains("hidden provider reasoning"));
}

#[test]
fn fireworks_chat_reasoning_fails_closed_without_carrier_or_unique_completion() {
    let events = fireworks_tool_events();
    assert!(completed_chat_body_with_ignored(
        "request-1",
        "coding",
        1_700_000_000,
        &events,
        &[],
        false,
    )
    .is_err());

    let mut duplicate = events[..events.len() - 1].to_vec();
    duplicate.push(events[3].clone());
    duplicate.push(Event::Completed);
    assert!(reasoning_carrier_candidate(&duplicate).is_err());
}

#[test]
fn reasoning_carrier_preserves_provider_tool_start_order() {
    let events = vec![
        Event::ReasoningContentDelta {
            route_sha256: "a".repeat(64),
            delta: "hidden".to_string(),
        },
        Event::ToolCallStarted {
            namespace: None,
            caller: None,
            index: 1,
            call_id: "call-one".to_string(),
            name: "first".to_string(),
        },
        Event::ToolCallStarted {
            namespace: None,
            caller: None,
            index: 0,
            call_id: "call-zero".to_string(),
            name: "second".to_string(),
        },
        Event::ToolCallCompleted {
            index: 0,
            call: crate::events::CompletedToolCall {
                namespace: None,
                caller: None,
                call_id: "call-zero".to_string(),
                name: "second".to_string(),
                raw_arguments: "{\"order\":0}".to_string(),
                provider_item_id: None,
                provider_status: None,
                custom: false,
            },
        },
        Event::ToolCallCompleted {
            index: 1,
            call: crate::events::CompletedToolCall {
                namespace: None,
                caller: None,
                call_id: "call-one".to_string(),
                name: "first".to_string(),
                raw_arguments: "{\"order\":1}".to_string(),
                provider_item_id: None,
                provider_status: None,
                custom: false,
            },
        },
    ];

    let candidate = reasoning_carrier_candidate(&events)
        .expect("provider events must validate")
        .expect("reasoning plus tools must produce a carrier");

    assert_eq!(
        candidate
            .tool_calls
            .iter()
            .map(|call| call.call_id.as_str())
            .collect::<Vec<_>>(),
        vec!["call-one", "call-zero"]
    );
}

#[test]
fn ignored_generation_controls_are_disclosed_by_both_chat_encoders() {
    let ignored = vec!["top_p".to_string(), "reasoning_effort".to_string()];
    let mut stream = ChatSseEncoder::new_with_ignored(
        "request-1",
        "coding",
        1_700_000_000,
        false,
        ignored.clone(),
    );
    let frames = stream.start().expect("stream start must encode");
    assert!(frames[0]
        .contains("\"x-experiential-ignored-parameters\":[\"top_p\",\"reasoning_effort\"]"));

    let completed = completed_chat_body_with_ignored(
        "request-1",
        "coding",
        1_700_000_000,
        &[Event::Completed],
        &ignored,
        false,
    )
    .expect("completed body must encode");
    assert_eq!(
        completed.body["x-experiential-ignored-parameters"],
        json!(["top_p", "reasoning_effort"])
    );
}

/// A non-tool reasoning turn on an exposure-gated rung returns the model's
/// plaintext reasoning for display, both streaming and non-streaming; an
/// unexposed rung keeps it stripped. There is no tool call, so no carrier.
#[test]
fn exposed_rung_returns_plaintext_reasoning_without_a_carrier() {
    let events = vec![
        Event::ReasoningContentDelta {
            route_sha256: "d".repeat(64),
            delta: "let me think: 17*23".to_string(),
        },
        Event::TextDelta("391".to_string()),
        Event::Completed,
    ];

    // Streaming: the plaintext streams as reasoning_content deltas.
    let mut exposed = ChatSseEncoder::new_with_ignored(
        "request-1",
        "hy4-preview",
        1_700_000_000,
        false,
        Vec::new(),
    );
    exposed.set_reasoning_output_exposed(true);
    let mut frames = exposed.start().expect("stream start must encode");
    for event in &events {
        frames.extend(exposed.feed(event).expect("event must encode"));
    }
    let public = frames.join("");
    assert!(public.contains("let me think: 17*23"));
    assert!(public.contains("\"reasoning_content\""));

    // An unexposed rung drops the very same reasoning stream.
    let mut hidden = ChatSseEncoder::new_with_ignored(
        "request-1",
        "hy4-preview",
        1_700_000_000,
        false,
        Vec::new(),
    );
    let mut hidden_frames = hidden.start().expect("stream start must encode");
    for event in &events {
        hidden_frames.extend(hidden.feed(event).expect("event must encode"));
    }
    assert!(!hidden_frames.join("").contains("let me think"));

    // Non-streaming: exposed returns plaintext, unexposed omits the field.
    let shown = completed_chat_body_with_ignored(
        "request-1",
        "hy4-preview",
        1_700_000_000,
        &events,
        &[],
        true,
    )
    .expect("completed body must encode");
    assert_eq!(
        shown.body["choices"][0]["message"]["reasoning_content"],
        json!("let me think: 17*23")
    );
    let stripped = completed_chat_body_with_ignored(
        "request-1",
        "hy4-preview",
        1_700_000_000,
        &events,
        &[],
        false,
    )
    .expect("completed body must encode");
    assert_eq!(
        stripped.body["choices"][0]["message"].get("reasoning_content"),
        None
    );
}

#[test]
fn chat_message_without_tool_calls_omits_the_key_instead_of_null() {
    // OpenAI omits `tool_calls` from a message that made none; strict
    // schema consumers reject `null` there while accepting an absent key.
    let events = vec![
        Event::TextDelta("Sunny in Bern.".to_string()),
        Event::Completed,
    ];
    let completed =
        completed_chat_body_with_ignored("request-1", "coding", 1_700_000_000, &events, &[], false)
            .expect("completed body must encode");
    let message = completed.body["choices"][0]["message"]
        .as_object()
        .expect("chat message is an object");
    assert!(!message.contains_key("tool_calls"));
    assert_eq!(message["content"], json!("Sunny in Bern."));
    assert_eq!(message["refusal"], Value::Null);
    assert_eq!(completed.body["choices"][0]["finish_reason"], json!("stop"));
}

#[test]
fn chat_message_with_tool_calls_carries_the_array() {
    let events = fireworks_tool_events();
    let completed = completed_chat_body_with_carrier(
        "request-1",
        "coding",
        1_700_000_000,
        &events,
        &[],
        Some("authenticated-carrier-v2"),
        false,
    )
    .expect("completed body must encode");
    let message = &completed.body["choices"][0]["message"];
    assert_eq!(message["tool_calls"].as_array().map(Vec::len), Some(1));
    assert_eq!(
        message["tool_calls"][0]["function"]["name"],
        json!("lookup")
    );
    assert_eq!(
        completed.body["choices"][0]["finish_reason"],
        json!("tool_calls")
    );
}
