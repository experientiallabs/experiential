//! Unit tests for the public Chat Completions encoders in `encode.rs`.

use super::*;

fn fireworks_tool_events() -> Vec<Event> {
    vec![
        Event::ReasoningContentDelta {
            route_sha256: "a".repeat(64),
            delta: "hidden provider reasoning".to_string(),
        },
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
            custom: false,
            namespace: None,
            caller: None,
            index: 1,
            call_id: "call-one".to_string(),
            name: "first".to_string(),
        },
        Event::ToolCallStarted {
            custom: false,
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

fn web_search_admission() -> crate::web_search::WebSearchAdmission {
    crate::web_search::WebSearchAdmission {
        query: "current stable Python".to_string(),
        requests: 1,
        results: vec![crate::web_search::WebSearchSource {
            url: "https://python.org/".to_string(),
            title: "Python".to_string(),
        }],
    }
}

fn web_search_chat_events(text: &str) -> Vec<Event> {
    vec![
        Event::TextDelta(text.to_string()),
        Event::Usage(Usage {
            input_tokens: Some(12),
            output_tokens: Some(7),
            ..Usage::default()
        }),
        Event::Completed,
    ]
}

fn chat_frames(
    web_search: Option<crate::web_search::WebSearchAdmission>,
    text: &str,
) -> Vec<String> {
    let mut encoder =
        ChatSseEncoder::new_with_ignored("request-1", "coding", 1_700_000_000, true, Vec::new());
    encoder.set_web_search(web_search);
    let mut frames = encoder.start().expect("starts");
    for event in &web_search_chat_events(text) {
        frames.extend(encoder.feed(event).expect("encodes"));
    }
    frames
}

fn chat_payload(frame: &str) -> Value {
    serde_json::from_str(frame.trim_start_matches("data: ").trim_end()).expect("chunk JSON")
}

#[test]
fn chat_stream_cites_the_search_right_before_the_finish_chunk_and_meters_usage() {
    let frames = chat_frames(
        Some(web_search_admission()),
        "See https://python.org/ today",
    );
    // role, content, annotations, finish, usage, [DONE]
    assert_eq!(frames.len(), 6);
    let annotations = chat_payload(&frames[2]);
    assert_eq!(
        annotations["choices"][0]["delta"],
        json!({"annotations": [{
            "type": "url_citation",
            "url_citation": {
                "url": "https://python.org/",
                "title": "Python",
                "start_index": 4,
                "end_index": 23,
            },
        }]})
    );
    assert_eq!(annotations["choices"][0]["finish_reason"], Value::Null);
    let finish = chat_payload(&frames[3]);
    assert_eq!(finish["choices"][0]["finish_reason"], json!("stop"));
    let usage = chat_payload(&frames[4]);
    assert_eq!(
        usage["usage"]["server_tool_use_details"],
        json!({"web_search_requests": 1})
    );
    assert_eq!(usage["usage"]["prompt_tokens"], json!(12));
    assert_eq!(frames[5], "data: [DONE]\n\n");
}

#[test]
fn chat_stream_skips_the_annotations_chunk_when_no_result_is_cited() {
    let frames = chat_frames(Some(web_search_admission()), "no links");
    // role, content, finish, usage, [DONE]
    assert_eq!(frames.len(), 5);
    assert!(!frames.join("").contains("annotations"));
    assert_eq!(
        chat_payload(&frames[3])["usage"]["server_tool_use_details"],
        json!({"web_search_requests": 1})
    );
}

#[test]
fn chat_stream_without_a_search_is_byte_identical() {
    let mut untouched =
        ChatSseEncoder::new_with_ignored("request-1", "coding", 1_700_000_000, true, Vec::new());
    let mut frames = untouched.start().expect("starts");
    for event in &web_search_chat_events("See https://python.org/ today") {
        frames.extend(untouched.feed(event).expect("encodes"));
    }
    assert_eq!(frames, chat_frames(None, "See https://python.org/ today"));
    let joined = frames.join("");
    assert!(!joined.contains("annotations"));
    assert!(!joined.contains("server_tool_use_details"));
}

#[test]
fn chat_aggregate_gains_annotations_and_the_usage_meter_only_when_searched() {
    let events = web_search_chat_events("See https://python.org/ today");
    let mut cited =
        completed_chat_body_with_ignored("request-1", "coding", 1_700_000_000, &events, &[], false)
            .expect("aggregates");
    let plain = compact_json(&cited.body);
    crate::web_search::annotate_chat_completion(&mut cited.body, &web_search_admission());
    let message = &cited.body["choices"][0]["message"];
    assert_eq!(
        message["annotations"][0]["url_citation"]["start_index"],
        json!(4)
    );
    assert_eq!(
        cited.body["usage"]["server_tool_use_details"],
        json!({"web_search_requests": 1})
    );
    assert!(!plain.contains("annotations"));
    assert!(!plain.contains("server_tool_use_details"));
}
