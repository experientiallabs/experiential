//! Unit tests for the public Chat Completions encoders in `encode.rs`.

use super::*;

#[test]
fn chat_unreported_meters_survive_two_hops_without_fabricating_zero() {
    for streaming in [false, true] {
        for count in [None, Some(0)] {
            let mut usage = Usage {
                input_tokens: Some(100),
                output_tokens: Some(10),
                cached_input_tokens: count,
                cache_creation_input_tokens: count,
                reasoning_tokens: count,
                ..Usage::default()
            };
            for _ in 0..2 {
                let wire = if streaming {
                    streaming_chat_usage(&usage)
                } else {
                    completed_chat_usage(Some(&usage))
                };
                assert_eq!(
                    wire.get("unreported_token_details").is_some(),
                    streaming && count.is_none()
                );
                usage = crate::events::openai_compatible_usage(&wire).unwrap();
                assert_eq!(usage.cached_input_tokens, count);
                assert_eq!(usage.cache_creation_input_tokens, count);
                assert_eq!(usage.reasoning_tokens, count);
            }
        }
    }
}

#[test]
fn chat_cache_ttl_survives_two_gateway_hops_without_inventing_zero() {
    for hour in [None, Some(0), Some(60)] {
        let usage = Usage {
            input_tokens: Some(120),
            output_tokens: Some(10),
            cached_input_tokens: Some(10),
            cache_creation_input_tokens: Some(100),
            cache_creation_1h_input_tokens: hour,
            reasoning_tokens: Some(5),
            billed_units: None,
            cache_writes_read_back: false,
        };
        for streaming in [false, true] {
            let mut current = usage.clone();
            for _ in 0..2 {
                let wire = if streaming {
                    streaming_chat_usage(&current)
                } else {
                    completed_chat_usage(Some(&current))
                };
                let details = wire["prompt_tokens_details"].as_object().unwrap();
                assert_eq!(
                    details.get("cache_write_1h_tokens"),
                    hour.map(|n| json!(n)).as_ref()
                );
                current = crate::events::openai_compatible_usage(&wire).unwrap();
                assert_eq!(current.cache_creation_input_tokens, Some(100));
                assert_eq!(current.cache_creation_1h_input_tokens, hour);
                assert_eq!(current.reasoning_tokens, Some(5));
            }
        }
    }
}

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
fn exposed_tool_reasoning_is_plaintext_not_concatenated_with_a_carrier() {
    let events = fireworks_tool_events();
    let mut stream = ChatSseEncoder::new_with_ignored("request", "model", 1, false, vec![]);
    stream.set_reasoning_output(true.into());
    stream.set_reasoning_content_carrier("opaque-carrier".into());
    let mut frames = stream.start().unwrap();
    for event in &events {
        frames.extend(stream.feed(event).unwrap());
    }
    let text: String = frames
        .iter()
        .filter_map(|frame| {
            let payload = frame.strip_prefix("data: ")?.trim();
            serde_json::from_str::<Value>(payload).ok()
        })
        .filter_map(|frame| {
            frame["choices"][0]["delta"]["reasoning_content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect();
    assert_eq!(text, "hidden provider reasoning");
    let completed = completed_chat_body_with_carrier(
        "request",
        "model",
        1,
        &events,
        &[],
        Some("opaque-carrier"),
        true,
    )
    .unwrap();
    assert_eq!(
        completed.body["choices"][0]["message"]["reasoning_content"],
        text
    );
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
    exposed.set_reasoning_output(true.into());
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
        include_sources: false,
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

fn custom_tool_events(custom: bool, name: &str, deltas: &[&str], raw: &str) -> Vec<Event> {
    let mut events = vec![Event::ToolCallStarted {
        custom,
        namespace: None,
        caller: None,
        index: 0,
        call_id: "call-custom".to_string(),
        name: name.to_string(),
    }];
    events.extend(deltas.iter().map(|delta| Event::ToolArgumentsDelta {
        index: 0,
        delta: (*delta).to_string(),
    }));
    events.push(Event::ToolCallCompleted {
        index: 0,
        call: crate::events::CompletedToolCall {
            namespace: None,
            caller: None,
            call_id: "call-custom".to_string(),
            name: name.to_string(),
            raw_arguments: raw.to_string(),
            provider_item_id: None,
            provider_status: None,
            custom,
        },
    });
    events.push(Event::Completed);
    events
}

fn chat_stream_payloads(events: &[Event]) -> Vec<Value> {
    let mut encoder =
        ChatSseEncoder::new_with_ignored("request-1", "coding", 1_700_000_000, false, Vec::new());
    let mut frames = encoder.start().expect("starts");
    for event in events {
        frames.extend(encoder.feed(event).expect("encodes"));
    }
    frames
        .iter()
        .filter(|frame| frame.starts_with("data: {"))
        .map(|frame| chat_payload(frame))
        .collect()
}

#[test]
fn chat_stream_renders_a_custom_tool_call_in_openais_chat_shape() {
    // Captured from api.openai.com Chat Completions (gpt-5-nano, 2026-10-09):
    // the start carries `type: custom` with an empty input, each delta
    // carries only `custom.input`, and the turn finishes with `tool_calls`.
    let events = custom_tool_events(true, "code_exec", &["print", "(2+2)"], "print(2+2)");
    let payloads = chat_stream_payloads(&events);
    let calls: Vec<&Value> = payloads
        .iter()
        .filter_map(|payload| payload["choices"][0]["delta"].get("tool_calls"))
        .collect();
    assert_eq!(
        calls,
        vec![
            &json!([{"index": 0, "id": "call-custom", "type": "custom",
                     "custom": {"name": "code_exec", "input": ""}}]),
            &json!([{"index": 0, "custom": {"input": "print"}}]),
            &json!([{"index": 0, "custom": {"input": "(2+2)"}}]),
        ]
    );
    let finish = payloads
        .iter()
        .find_map(|payload| payload["choices"][0]["finish_reason"].as_str())
        .expect("finish chunk");
    assert_eq!(finish, "tool_calls");
}

#[test]
fn chat_stream_refuses_a_completion_that_changes_the_custom_kind() {
    let mut events = custom_tool_events(false, "code_exec", &["{}"], "{}");
    if let Event::ToolCallCompleted { call, .. } = &mut events[2] {
        call.custom = true;
    }
    let mut encoder =
        ChatSseEncoder::new_with_ignored("request-1", "coding", 1_700_000_000, false, Vec::new());
    encoder.start().expect("starts");
    encoder.feed(&events[0]).expect("start encodes");
    encoder.feed(&events[1]).expect("delta encodes");
    assert!(encoder.feed(&events[2]).is_err());
}

#[test]
fn chat_aggregate_renders_a_custom_tool_call_in_openais_chat_shape() {
    let events = custom_tool_events(true, "code_exec", &["print(2+2)\n"], "print(2+2)\n");
    let completed =
        completed_chat_body_with_ignored("request-1", "coding", 1_700_000_000, &events, &[], false)
            .expect("completed body must encode");
    let choice = &completed.body["choices"][0];
    assert_eq!(
        choice["message"]["tool_calls"],
        json!([{"id": "call-custom", "type": "custom",
                "custom": {"name": "code_exec", "input": "print(2+2)\n"}}])
    );
    assert_eq!(choice["finish_reason"], json!("tool_calls"));
}

#[test]
fn chat_stream_unwraps_a_translated_custom_call_for_a_chat_caller() {
    // A foreign wire served the custom tool as a function with one `input`
    // string; the inverter restores the custom call and the Chat encoder
    // emits the caller's custom shape with the whole freeform input at once.
    let mut inverter = crate::codex_native_inversion::NativeToolInverter::default();
    inverter.translation = std::collections::HashMap::from([(
        "code_exec".to_string(),
        ("code_exec".to_string(), None, true),
    )]);
    let provider = custom_tool_events(
        false,
        "code_exec",
        &["{\"input\":\"print(", "2+2)\"}"],
        "{\"input\":\"print(2+2)\"}",
    );
    let mut inverted = Vec::new();
    for event in provider {
        inverted.extend(inverter.filter(event).expect("inverts"));
    }
    let payloads = chat_stream_payloads(&inverted);
    let calls: Vec<&Value> = payloads
        .iter()
        .filter_map(|payload| payload["choices"][0]["delta"].get("tool_calls"))
        .collect();
    assert_eq!(
        calls,
        vec![
            &json!([{"index": 0, "id": "call-custom", "type": "custom",
                     "custom": {"name": "code_exec", "input": ""}}]),
            &json!([{"index": 0, "custom": {"input": "print(2+2)"}}]),
        ]
    );
}

/// A Gemini automatic-cache creation read back in-call (10,663 of 11,933
/// input tokens written, none read from an earlier cache).
fn written_and_read_back() -> Usage {
    Usage {
        input_tokens: Some(11_933),
        output_tokens: Some(9),
        cached_input_tokens: Some(0),
        cache_creation_input_tokens: Some(10_663),
        cache_creation_1h_input_tokens: Some(0),
        cache_writes_read_back: true,
        ..Usage::default()
    }
}

#[test]
fn openai_shapes_report_writes_read_back_inside_cached_tokens_like_openrouter() {
    // OpenRouter's Gemini relay reports the creating call's written tokens in
    // `cached_tokens` too; the OpenAI-shape renderers match it.
    let usage = written_and_read_back();
    for wire in [
        streaming_chat_usage(&usage),
        completed_chat_usage(Some(&usage)),
    ] {
        let details = &wire["prompt_tokens_details"];
        assert_eq!(details["cached_tokens"], 10_663);
        assert_eq!(details["cache_write_tokens"], 10_663);
        assert_eq!(wire["prompt_tokens"], 11_933);
    }
    // A reuse on the same call adds to the read-back writes.
    let mixed = Usage {
        cached_input_tokens: Some(500),
        ..written_and_read_back()
    };
    assert_eq!(
        completed_chat_usage(Some(&mixed))["prompt_tokens_details"]["cached_tokens"],
        11_163
    );
}

#[test]
fn disjoint_writes_and_the_anthropic_shape_keep_reads_and_writes_apart() {
    // Writes that were not read back (Anthropic, Bedrock) stay disjoint.
    let disjoint = Usage {
        cache_writes_read_back: false,
        ..written_and_read_back()
    };
    assert_eq!(
        completed_chat_usage(Some(&disjoint))["prompt_tokens_details"]["cached_tokens"],
        0
    );
    // Anthropic clients sum the legs into the prompt size, so the Messages
    // shape never double reports a written token as read.
    let messages = crate::encode_messages::messages_usage(Some(&written_and_read_back()));
    assert_eq!(messages["cache_read_input_tokens"], 0);
    assert_eq!(messages["cache_creation_input_tokens"], 10_663);
    assert_eq!(messages["input_tokens"], 1_270);
    // Settlement's disjoint read leg is untouched by the reporting convention.
    assert_eq!(written_and_read_back().cached_input_tokens, Some(0));
}
