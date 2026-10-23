//! Tool-call item attribution tests for `encode_responses`: the namespace,
//! programmatic caller, and Gemini thought signature each re-emit on the
//! function_call item the caller echoes back.

use super::*;

#[test]
fn namespaced_tool_call_items_re_emit_namespace_to_the_caller() {
    // The caller replays the completed function_call item verbatim, so the
    // synthesized output items must carry the namespace or the provider
    // rejects the next turn ("Missing namespace for function_call ...").
    let events = vec![
        Event::ToolCallStarted {
            custom: false,
            index: 0,
            call_id: "call-ns".to_string(),
            name: "spawn_agent".to_string(),
            namespace: Some("collaboration".to_string()),
            caller: None,
            thought_signature: None,
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: "{}".to_string(),
        },
        Event::ToolCallCompleted {
            index: 0,
            call: crate::events::CompletedToolCall {
                call_id: "call-ns".to_string(),
                name: "spawn_agent".to_string(),
                namespace: Some("collaboration".to_string()),
                caller: None,
                provider_item_id: None,
                provider_status: None,
                raw_arguments: "{}".to_string(),
                custom: false,
                thought_signature: None,
            },
        },
        Event::Completed,
    ];
    let mut encoder = ResponsesSseEncoder::new(
        "request-1",
        "coding",
        1_700_000_000,
        ResponsesEnvelope::default(),
    );
    encoder.start().expect("stream start must encode");
    let mut frames = Vec::new();
    for event in &events {
        frames.extend(encoder.feed(event).expect("events must encode"));
    }
    let added = frames
        .iter()
        .find(|frame| frame.contains("response.output_item.added"))
        .expect("tool item start frame");
    assert!(added.contains("\"namespace\":\"collaboration\""));
    let done = frames
        .iter()
        .find(|frame| frame.contains("response.output_item.done"))
        .expect("tool item done frame");
    assert!(done.contains("\"namespace\":\"collaboration\""));

    let completed = completed_responses_body(
        "request-1",
        "coding",
        1_700_000_000,
        ResponsesEnvelope::default(),
        &events,
    )
    .expect("completed body must encode");
    assert_eq!(
        completed.body["output"][0]["namespace"],
        json!("collaboration")
    );

    // A namespace-free call keeps the exact pre-existing item shape.
    let plain = completed_responses_body(
        "request-2",
        "coding",
        1_700_000_000,
        ResponsesEnvelope::default(),
        &[
            Event::ToolCallStarted {
                custom: false,
                index: 0,
                call_id: "call-plain".to_string(),
                name: "lookup".to_string(),
                namespace: None,
                caller: None,
                thought_signature: None,
            },
            Event::ToolArgumentsDelta {
                index: 0,
                delta: "{}".to_string(),
            },
            Event::ToolCallCompleted {
                index: 0,
                call: crate::events::CompletedToolCall {
                    call_id: "call-plain".to_string(),
                    name: "lookup".to_string(),
                    namespace: None,
                    caller: None,
                    provider_item_id: None,
                    provider_status: None,
                    raw_arguments: "{}".to_string(),
                    custom: false,
                    thought_signature: None,
                },
            },
            Event::Completed,
        ],
    )
    .expect("completed body must encode");
    assert!(plain.body["output"][0]
        .as_object()
        .expect("tool item object")
        .get("namespace")
        .is_none());
}

/// Hosted tool items re-emit verbatim at gateway-owned indexes: the public
/// output index and sequence number are re-stamped, everything inside the
/// item and its progress payloads passes through untouched, and the final
/// envelope re-serves the provider's exact item JSON.

#[test]
fn caller_attributed_tool_call_items_re_emit_caller_to_the_caller() {
    // The caller replays the completed function_call item verbatim, so the
    // synthesized output items must carry the SDK 3.0 `caller` attribution
    // exactly as the provider emitted it.
    let caller = json!({"type": "program", "id": "prog_1"});
    let events = vec![
        Event::ToolCallStarted {
            custom: false,
            index: 0,
            call_id: "call-caller".to_string(),
            name: "lookup".to_string(),
            namespace: None,
            caller: Some(caller.clone()),
            thought_signature: None,
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: "{}".to_string(),
        },
        Event::ToolCallCompleted {
            index: 0,
            call: crate::events::CompletedToolCall {
                call_id: "call-caller".to_string(),
                name: "lookup".to_string(),
                namespace: None,
                caller: Some(caller.clone()),
                provider_item_id: None,
                provider_status: None,
                raw_arguments: "{}".to_string(),
                custom: false,
                thought_signature: None,
            },
        },
        Event::Completed,
    ];
    let mut encoder = ResponsesSseEncoder::new(
        "request-1",
        "coding",
        1_700_000_000,
        ResponsesEnvelope::default(),
    );
    encoder.start().expect("stream start must encode");
    let mut frames = Vec::new();
    for event in &events {
        frames.extend(encoder.feed(event).expect("events must encode"));
    }
    let added = frames
        .iter()
        .find(|frame| frame.contains("response.output_item.added"))
        .expect("tool item start frame");
    assert!(added.contains("\"caller\"") && added.contains("\"prog_1\""));
    let done = frames
        .iter()
        .find(|frame| frame.contains("response.output_item.done"))
        .expect("tool item done frame");
    assert!(done.contains("\"caller\"") && done.contains("\"prog_1\""));

    let completed = completed_responses_body(
        "request-1",
        "coding",
        1_700_000_000,
        ResponsesEnvelope::default(),
        &events,
    )
    .expect("completed body must encode");
    assert_eq!(completed.body["output"][0]["caller"], caller);
}

/// One complete Gemini-shaped tool turn: two parallel calls, of which Google
/// signs only the first.
fn signed_parallel_tool_events() -> Vec<Event> {
    let mut events = Vec::new();
    for (index, signature) in [(0u32, Some("c2lnbmF0dXJlLWE=")), (1, None)] {
        let call_id = format!("call-{index}");
        events.push(Event::ToolCallStarted {
            custom: false,
            index,
            call_id: call_id.clone(),
            name: "get_weather".to_string(),
            namespace: None,
            caller: None,
            thought_signature: signature.map(str::to_string),
        });
        events.push(Event::ToolArgumentsDelta {
            index,
            delta: "{}".to_string(),
        });
        events.push(Event::ToolCallCompleted {
            index,
            call: crate::events::CompletedToolCall {
                call_id,
                name: "get_weather".to_string(),
                namespace: None,
                caller: None,
                thought_signature: signature.map(str::to_string),
                provider_item_id: None,
                provider_status: None,
                raw_arguments: "{}".to_string(),
                custom: false,
            },
        });
    }
    events.push(Event::Completed);
    events
}

#[test]
fn gemini_thought_signatures_ride_function_call_items_as_extra_content() {
    // A caller echoing the function_call item returns the signature Gemini
    // requires on the replayed call; the unsigned call carries no field.
    let events = signed_parallel_tool_events();
    let signed = json!({"google": {"thought_signature": "c2lnbmF0dXJlLWE="}});
    let mut encoder = ResponsesSseEncoder::new(
        "request-1",
        "gemini",
        1_700_000_000,
        ResponsesEnvelope::default(),
    );
    encoder.start().expect("stream start must encode");
    let mut frames = Vec::new();
    for event in &events {
        frames.extend(encoder.feed(event).expect("events must encode"));
    }
    let items = |event_type: &str| -> Vec<Value> {
        frames
            .iter()
            .filter(|frame| frame.contains(event_type))
            .map(|frame| {
                let data = frame.split("data: ").nth(1).expect("frame data");
                serde_json::from_str::<Value>(data.trim()).expect("frame json")["item"].clone()
            })
            .collect()
    };
    for event_type in ["response.output_item.added", "response.output_item.done"] {
        let items = items(event_type);
        assert_eq!(items.len(), 2, "{event_type}");
        assert_eq!(items[0]["extra_content"], signed, "{event_type}");
        assert!(items[1].get("extra_content").is_none(), "{event_type}");
    }

    let completed = completed_responses_body(
        "request-1",
        "gemini",
        1_700_000_000,
        ResponsesEnvelope::default(),
        &events,
    )
    .expect("completed body must encode");
    assert_eq!(completed.body["output"][0]["extra_content"], signed);
    assert!(completed.body["output"][1].get("extra_content").is_none());
}

#[test]
fn a_completion_that_changes_the_streamed_signature_is_refused() {
    let mut events = signed_parallel_tool_events();
    if let Event::ToolCallCompleted { call, .. } = &mut events[2] {
        call.thought_signature = Some("b3RoZXI=".to_string());
    }
    let mut encoder = ResponsesSseEncoder::new(
        "request-1",
        "gemini",
        1_700_000_000,
        ResponsesEnvelope::default(),
    );
    encoder.start().expect("stream start must encode");
    let refused = events.iter().find_map(|event| encoder.feed(event).err());
    assert!(refused.is_some());
}
