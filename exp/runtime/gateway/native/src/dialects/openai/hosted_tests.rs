//! Hosted-tool pass-through regressions for the OpenAI Responses dialect
//! (split from `normalizer_tests.rs` for the module line budget).

use super::*;
use crate::dialects::{Dialect, Normalizer};
use crate::sse::SseEvent;

/// Build one typed SSE frame for the hosted-tool lifecycle tests.
fn hosted_frame(payload: serde_json::Value) -> SseEvent {
    SseEvent {
        event: None,
        data: payload.to_string(),
    }
}

#[test]
fn terminal_response_preserves_authoritative_output_probabilities() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    normalizer.enable_responses_logprobs(true);
    let terminal = hosted_frame(serde_json::json!({
        "type": "response.completed",
        "response": {
            "status": "completed",
            "output": [{
                "id": "msg_1",
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": "OK",
                    "logprobs": [{"token": "OK", "logprob": -0.125, "bytes": [79, 75]}]
                }]
            }]
        }
    }));
    let events = normalizer.feed(&terminal).expect("terminal normalizes");
    assert!(events.iter().any(|event| matches!(
        event,
        Event::ProviderResponsesLogprobs { phase, records, output_index: 0, .. }
        if phase == "terminal" && records[0]["token"] == "OK"
    )));
}

#[test]
fn message_item_done_preserves_each_content_part_probability_record() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    normalizer.enable_responses_logprobs(true);
    let done = hosted_frame(serde_json::json!({
        "type": "response.output_item.done",
        "output_index": 2,
        "item": {
            "id": "msg_2",
            "type": "message",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "A", "logprobs": [{"token": "A", "logprob": -0.1}]},
                {"type": "output_text", "text": "B", "logprobs": [{"token": "B", "logprob": -0.2}]}
            ]
        }
    }));
    let events = normalizer.feed(&done).expect("item completion normalizes");
    let probabilities: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            Event::ProviderResponsesLogprobs {
                content_index,
                phase,
                records,
                ..
            } => Some((*content_index, phase.as_str(), records)),
            _ => None,
        })
        .collect();
    assert_eq!(probabilities.len(), 2);
    assert_eq!(probabilities[0].0, 0);
    assert_eq!(probabilities[0].1, "item_done");
    assert_eq!(probabilities[0].2[0]["token"], "A");
    assert_eq!(probabilities[1].0, 1);
    assert_eq!(probabilities[1].2[0]["token"], "B");
}

#[test]
fn malformed_responses_probability_records_fail_closed() {
    for records in [
        serde_json::json!([{"token": 7, "logprob": -0.1}]),
        serde_json::json!([{"token": "x", "logprob": "bad"}]),
        serde_json::json!([{"token": "x", "logprob": -0.1, "top_logprobs": [{"token": "y", "logprob": "bad"}]}]),
    ] {
        let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
        normalizer.enable_responses_logprobs(true);
        let frame = hosted_frame(serde_json::json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "item_id": "msg_bad",
            "content_index": 0,
            "delta": "x",
            "logprobs": records,
        }));
        assert!(normalizer.feed(&frame).is_err());
    }
}

/// Frame shapes mirror the documented Responses web_search lifecycle
/// (`output_item.added`, the three `response.web_search_call.*` status
/// events, `output_item.done` with the final `action`, and a cited answer;
/// openai-python 3.x stream-event union, checked 2026-09-04). The 2026-09-04
/// incident class: the added frame alone killed the stream as malformed.
#[test]
fn web_search_call_items_pass_through_verbatim_with_their_lifecycle() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    normalizer.enable_responses_logprobs(true);
    let added = hosted_frame(serde_json::json!({
        "type": "response.output_item.added",
        "output_index": 0,
        "sequence_number": 2,
        "item": {"id": "ws_1", "type": "web_search_call", "status": "in_progress"},
    }));
    let events = normalizer
        .feed(&added)
        .expect("web_search start normalizes");
    match events.as_slice() {
        [Event::HostedToolItemStarted {
            output_index: 0,
            item_id,
            item_type,
            item,
        }] => {
            assert_eq!(item_id, "ws_1");
            assert_eq!(item_type, "web_search_call");
            assert!(item.contains("\"status\":\"in_progress\""));
        }
        other => panic!("unexpected events: {other:?}"),
    }
    for status_event in [
        "response.web_search_call.in_progress",
        "response.web_search_call.searching",
        "response.web_search_call.completed",
    ] {
        let frame = hosted_frame(serde_json::json!({
            "type": status_event,
            "item_id": "ws_1",
            "output_index": 0,
            "sequence_number": 3,
        }));
        let events = normalizer.feed(&frame).expect("status frame normalizes");
        match events.as_slice() {
            [Event::HostedToolItemProgress {
                output_index: 0,
                item_id,
                event_type,
                payload,
            }] => {
                assert_eq!(item_id, "ws_1");
                assert_eq!(event_type, status_event);
                assert!(payload.contains("\"item_id\":\"ws_1\""));
            }
            other => panic!("unexpected events: {other:?}"),
        }
    }
    let done = hosted_frame(serde_json::json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "id": "ws_1",
            "type": "web_search_call",
            "status": "completed",
            "action": {"type": "search", "query": "current stable Python"},
        },
    }));
    let events = normalizer.feed(&done).expect("web_search done normalizes");
    match events.as_slice() {
        [Event::HostedToolItemCompleted { item, .. }] => {
            assert!(item.contains("\"query\":\"current stable Python\""));
        }
        other => panic!("unexpected events: {other:?}"),
    }
    let text = hosted_frame(serde_json::json!({
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "output_index": 1,
        "content_index": 0,
        "delta": "Python 3.14.7.",
        "logprobs": [{"token": "Python", "logprob": -0.25, "bytes": [80, 121]}],
    }));
    let events = normalizer.feed(&text).expect("text normalizes");
    assert_eq!(
        events.len(),
        3,
        "message start plus probability and text delta"
    );
    assert!(events.iter().any(|event| matches!(
        event,
        Event::ProviderResponsesLogprobs { phase, content_index: 0, .. }
        if phase == "delta"
    )));
    let part_done = hosted_frame(serde_json::json!({
        "type": "response.content_part.done",
        "item_id": "msg_1",
        "output_index": 1,
        "content_index": 0,
        "part": {"type": "output_text", "text": "Python 3.14.7.", "logprobs": []},
    }));
    let events = normalizer.feed(&part_done).expect("part done normalizes");
    assert!(events.iter().any(|event| matches!(
        event,
        Event::ProviderResponsesLogprobs { phase, records, .. }
        if phase == "content_part_done" && records == &serde_json::json!([])
    )));
    let annotation = hosted_frame(serde_json::json!({
        "type": "response.output_text.annotation.added",
        "item_id": "msg_1",
        "output_index": 1,
        "content_index": 0,
        "annotation_index": 0,
        "sequence_number": 9,
        "annotation": {
            "type": "url_citation",
            "url": "https://www.python.org/doc/versions/",
            "title": "Python versions",
            "start_index": 0,
            "end_index": 14,
        },
    }));
    let events = normalizer.feed(&annotation).expect("annotation normalizes");
    match events.as_slice() {
        [Event::ProviderTextAnnotation {
            output_index: 1,
            item_id,
            annotation,
        }] => {
            assert_eq!(item_id, "msg_1");
            assert!(annotation.contains("\"type\":\"url_citation\""));
        }
        other => panic!("unexpected events: {other:?}"),
    }
    let terminal = hosted_frame(serde_json::json!({
        "type": "response.completed",
        "response": {
            "status": "completed",
            "usage": {"input_tokens": 320, "output_tokens": 41, "total_tokens": 361},
        },
    }));
    let events = normalizer.feed(&terminal).expect("terminal normalizes");
    // The message item never saw output_item.done, so the sweep closes it;
    // the hosted item already completed and is not re-emitted.
    assert!(events
        .iter()
        .all(|event| !matches!(event, Event::HostedToolItemCompleted { .. })));
    assert!(matches!(events.last(), Some(Event::Completed)));
    match &events[events.len() - 2] {
        Event::Usage(usage) => assert_eq!(usage.input_tokens, Some(320)),
        other => panic!("expected terminal usage, got {other:?}"),
    }
}

/// The documented MCP lifecycle: `mcp_list_tools` and `mcp_call` items with
/// their argument deltas and status frames all pass through, and the final
/// items carry the provider's own fields (`server_label`, `output`) verbatim.
#[test]
fn mcp_items_pass_through_with_argument_deltas() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    normalizer.enable_responses_logprobs(true);
    let listing = hosted_frame(serde_json::json!({
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {"id": "mcpl_1", "type": "mcp_list_tools", "server_label": "deepwiki", "tools": []},
    }));
    assert!(matches!(
        normalizer.feed(&listing).expect("listing normalizes").as_slice(),
        [Event::HostedToolItemStarted { item_type, .. }] if item_type == "mcp_list_tools"
    ));
    let listing_done = hosted_frame(serde_json::json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "id": "mcpl_1",
            "type": "mcp_list_tools",
            "server_label": "deepwiki",
            "tools": [{"name": "ask_question", "input_schema": {"type": "object"}}],
        },
    }));
    assert!(matches!(
        normalizer.feed(&listing_done).expect("listing done").as_slice(),
        [Event::HostedToolItemCompleted { item, .. }] if item.contains("ask_question")
    ));
    let call = hosted_frame(serde_json::json!({
        "type": "response.output_item.added",
        "output_index": 1,
        "item": {
            "id": "mcp_1",
            "type": "mcp_call",
            "server_label": "deepwiki",
            "name": "ask_question",
            "arguments": "",
            "status": "in_progress",
        },
    }));
    assert!(matches!(
        normalizer.feed(&call).expect("mcp call start").as_slice(),
        [Event::HostedToolItemStarted { item_type, .. }] if item_type == "mcp_call"
    ));
    for (event_type, extra) in [
        (
            "response.mcp_call_arguments.delta",
            ("delta", "{\"question\""),
        ),
        ("response.mcp_call_arguments.delta", ("delta", ": \"pi?\"}")),
        (
            "response.mcp_call_arguments.done",
            ("arguments", "{\"question\": \"pi?\"}"),
        ),
        ("response.mcp_call.completed", ("item_id", "mcp_1")),
    ] {
        let mut payload = serde_json::json!({
            "type": event_type,
            "item_id": "mcp_1",
            "output_index": 1,
            "sequence_number": 7,
        });
        payload[extra.0] = serde_json::json!(extra.1);
        let events = normalizer
            .feed(&hosted_frame(payload))
            .expect("mcp frame normalizes");
        assert!(matches!(
            events.as_slice(),
            [Event::HostedToolItemProgress { event_type: seen, .. }] if seen == event_type
        ));
    }
    let call_done = hosted_frame(serde_json::json!({
        "type": "response.output_item.done",
        "output_index": 1,
        "item": {
            "id": "mcp_1",
            "type": "mcp_call",
            "server_label": "deepwiki",
            "name": "ask_question",
            "arguments": "{\"question\": \"pi?\"}",
            "output": "3.14159",
            "status": "completed",
        },
    }));
    assert!(matches!(
        normalizer.feed(&call_done).expect("mcp call done").as_slice(),
        [Event::HostedToolItemCompleted { item, .. }] if item.contains("\"output\":\"3.14159\"")
    ));
    let terminal = hosted_frame(serde_json::json!({
        "type": "response.completed",
        "response": {"status": "completed", "usage": {"input_tokens": 5, "output_tokens": 3}},
    }));
    let events = normalizer.feed(&terminal).expect("terminal normalizes");
    assert!(matches!(events.last(), Some(Event::Completed)));
}

/// A truly unknown output-item type still fails the stream closed, and the
/// reason names the type so the next unknown shape is diagnosable from logs.
#[test]
fn unknown_output_item_types_fail_with_the_type_named() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    normalizer.enable_responses_logprobs(true);
    let added = hosted_frame(serde_json::json!({
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {"id": "x_1", "type": "telepathy_call"},
    }));
    let failure = normalizer
        .feed(&added)
        .expect_err("unknown type fails closed");
    assert!(
        failure.safe_message.contains("telepathy_call"),
        "reason must name the type: {}",
        failure.safe_message
    );

    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    normalizer.enable_responses_logprobs(true);
    let hostile = hosted_frame(serde_json::json!({
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {"id": "x_1", "type": "weird type! with spaces and payload"},
    }));
    let failure = normalizer
        .feed(&hostile)
        .expect_err("hostile type fails closed");
    assert!(
        failure.safe_message.contains("non-identifier"),
        "a non-identifier token must not be relayed: {}",
        failure.safe_message
    );
}

/// A hosted item whose `done` never arrives is closed by the terminal sweep
/// with its last-seen verbatim JSON, never rewritten and never dropped.
#[test]
fn hosted_items_without_done_close_at_the_terminal_with_the_last_seen_item() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    normalizer.enable_responses_logprobs(true);
    let added = hosted_frame(serde_json::json!({
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {"id": "img_1", "type": "image_generation_call", "status": "generating", "result": null},
    }));
    normalizer.feed(&added).expect("image start normalizes");
    let terminal = hosted_frame(serde_json::json!({
        "type": "response.incomplete",
        "response": {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {"input_tokens": 10, "output_tokens": 2},
        },
    }));
    let events = normalizer.feed(&terminal).expect("terminal normalizes");
    match events.first() {
        Some(Event::HostedToolItemCompleted { item, .. }) => {
            // The provider owns the status vocabulary, so the swept item is
            // re-served verbatim rather than patched to a gateway status.
            assert!(item.contains("\"status\":\"generating\""));
        }
        other => panic!("expected the swept hosted item first, got {other:?}"),
    }
    assert!(matches!(events.last(), Some(Event::Incomplete)));
}

/// A hosted progress frame for an item that never started names its event
/// type in the malformed reason.
#[test]
fn hosted_progress_before_its_item_fails_with_the_event_named() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    normalizer.enable_responses_logprobs(true);
    let frame = hosted_frame(serde_json::json!({
        "type": "response.mcp_call.completed",
        "item_id": "mcp_9",
        "output_index": 4,
        "sequence_number": 3,
    }));
    let failure = normalizer
        .feed(&frame)
        .expect_err("orphan progress fails closed");
    assert!(
        failure.safe_message.contains("response.mcp_call.completed"),
        "reason must name the event: {}",
        failure.safe_message
    );
}

/// The astra cyber-policy kill: a `response.failed` whose error names
/// `cyber_policy` classifies as a refusal that carries the bounded category
/// onto BOTH caller surfaces, while the raw provider prose stays ledger-only.
#[test]
fn a_cyber_policy_response_failed_carries_the_reason_on_both_surfaces() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    normalizer.enable_responses_logprobs(true);
    let failed = hosted_frame(serde_json::json!({
        "type": "response.failed",
        "response": {
            "status": "failed",
            "error": {"code": "cyber_policy", "message": "blocked by our cybersecurity policy"},
        },
    }));
    let events = normalizer
        .feed(&failed)
        .expect("failed terminal normalizes");
    let failure = match events.as_slice() {
        [Event::Failed(failure)] => failure.clone(),
        other => panic!("unexpected events: {other:?}"),
    };
    assert_eq!(failure.failure_class, crate::errors::FailureClass::Refusal);
    assert_eq!(
        failure.refusal_reason,
        Some(crate::errors::RefusalReason::CyberPolicy)
    );
    // The raw provider sentence reaches the ledger detail only.
    assert!(failure
        .provider_detail
        .as_deref()
        .is_some_and(|detail| detail.contains("cyber_policy")));

    let public = failure.public_error();
    assert_eq!(public.status_code, 400);
    assert_eq!(public.code, "refusal");
    assert_eq!(
        public.message,
        "provider refused the request: cybersecurity policy"
    );
    // Chat / Responses surface (OpenAI envelope).
    let chat_body = public.json_body();
    assert_eq!(chat_body["error"]["refusal_reason"], "cyber_policy");
    assert_eq!(chat_body["error"]["code"], "refusal");
    assert!(!chat_body
        .to_string()
        .contains("blocked by our cybersecurity"));
    // Messages surface (Anthropic envelope).
    let messages_body = crate::encode_messages::anthropic_error_body(&public);
    assert_eq!(messages_body["error"]["refusal_reason"], "cyber_policy");
    assert_eq!(messages_body["error"]["type"], "invalid_request_error");
    assert!(!messages_body
        .to_string()
        .contains("blocked by our cybersecurity"));
}

/// A failed terminal still folds the usage the provider billed.
#[test]
fn a_failed_terminal_reports_its_billed_usage() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    normalizer.enable_responses_logprobs(true);
    let failed = hosted_frame(serde_json::json!({
        "type": "response.failed",
        "response": {
            "status": "failed",
            "error": {"code": "server_error", "message": "boom"},
            "usage": {"input_tokens": 88, "output_tokens": 7},
        },
    }));
    let events = normalizer
        .feed(&failed)
        .expect("failed terminal normalizes");
    match events.as_slice() {
        [Event::Usage(usage), Event::Failed(failure)] => {
            assert_eq!(usage.input_tokens, Some(88));
            assert_eq!(
                failure.failure_class,
                crate::errors::FailureClass::ProviderInternal
            );
        }
        other => panic!("unexpected events: {other:?}"),
    }
}

#[test]
fn failed_terminal_replaces_provisional_reasoning_usage() {
    for additive in [false, true] {
        let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
        // Seed the owning accumulator's retained partial snapshot to exercise
        // failed-terminal retention independently of earlier event shapes.
        for usage in [
            serde_json::json!({"input_tokens": 100, "output_tokens": 10,
                "total_tokens": if additive { 110 } else { 115 }}),
            serde_json::json!({"output_tokens_details": {"reasoning_tokens": 5}}),
        ] {
            normalizer.usage = normalizer
                .openai_usage
                .update_responses(Some(&usage))
                .unwrap();
        }
        let events = normalizer
            .feed(&hosted_frame(serde_json::json!({
                "type": "response.failed", "response": {
                    "status": "failed", "error": {"code": "server_error"},
                    "usage": {"input_tokens": if additive { 100 } else { 105 },
                        "output_tokens": 10, "output_tokens_details": {"reasoning_tokens": 5},
                        "total_tokens": 115}
                }
            })))
            .unwrap();
        match events.as_slice() {
            [Event::Usage(usage), Event::Failed(_)] => {
                assert_eq!(usage.output_tokens, Some(if additive { 15 } else { 10 }));
                assert_eq!(usage.reasoning_tokens, Some(5));
            }
            other => panic!("unexpected events: {other:?}"),
        }
        assert_eq!(
            normalizer.observed_usage().unwrap().output_tokens,
            Some(if additive { 15 } else { 10 })
        );
    }
}

/// A status frame trailing the item's `done` is dropped, never a failed
/// stream: the final item already reached the caller and is the authority.
#[test]
fn hosted_progress_after_done_is_dropped_not_fatal() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    normalizer.enable_responses_logprobs(true);
    let added = hosted_frame(serde_json::json!({
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {"id": "ws_1", "type": "web_search_call", "status": "in_progress"},
    }));
    normalizer.feed(&added).expect("start normalizes");
    let done = hosted_frame(serde_json::json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {"id": "ws_1", "type": "web_search_call", "status": "completed"},
    }));
    normalizer.feed(&done).expect("done normalizes");
    let late = hosted_frame(serde_json::json!({
        "type": "response.web_search_call.completed",
        "item_id": "ws_1",
        "output_index": 0,
        "sequence_number": 11,
    }));
    assert!(normalizer
        .feed(&late)
        .expect("a trailing status frame is dropped")
        .is_empty());
}

#[test]
fn malformed_terminal_probabilities_preserve_reported_billing_usage() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    normalizer.enable_responses_logprobs(true);
    let frame = hosted_frame(serde_json::json!({
        "type": "response.completed", "response": {
            "status": "completed", "usage": {"input_tokens": 17, "output_tokens": 4},
            "output": [{"type": "message", "id": "msg-bad", "content": [
                {"type": "output_text", "text": "A", "logprobs": [{"token": "A"}]}
            ]}]
        }
    }));
    assert!(normalizer.feed(&frame).is_err());
    let usage = normalizer
        .observed_usage()
        .expect("usage survives validation failure");
    assert_eq!(usage.input_tokens, Some(17));
    assert_eq!(usage.output_tokens, Some(4));
}

#[test]
fn terminal_probability_indices_reject_overflow_and_malformed_values_without_losing_usage() {
    for output_index in [
        serde_json::json!(u64::from(u32::MAX) + 1),
        serde_json::json!(-1),
        serde_json::json!("0"),
        serde_json::json!(1.5),
        serde_json::json!(false),
        serde_json::Value::Null,
    ] {
        let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
        normalizer.enable_responses_logprobs(true);
        let frame = hosted_frame(serde_json::json!({
            "type": "response.completed", "response": {
                "status": "completed", "usage": {"input_tokens": 17, "output_tokens": 4},
                "output": [{"type": "message", "id": "msg-bad", "output_index": output_index,
                    "content": [{"type": "output_text", "text": "A", "logprobs": []}]}]
            }
        }));
        let failure = normalizer
            .feed(&frame)
            .expect_err("present invalid index must not fall back or wrap");
        assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
        let usage = normalizer
            .observed_usage()
            .expect("meter survives index refusal");
        assert_eq!(usage.input_tokens, Some(17));
        assert_eq!(usage.output_tokens, Some(4));
    }
}

#[test]
fn terminal_probability_indices_preserve_valid_values_and_fall_back_only_when_absent() {
    for explicit in [None, Some(u32::MAX)] {
        let mut item = serde_json::json!({"type":"message", "id":"msg-ok", "content":[
            {"type":"output_text", "text":"A", "logprobs":[]}
        ]});
        if let Some(index) = explicit {
            item["output_index"] = serde_json::json!(index);
        }
        let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
        normalizer.enable_responses_logprobs(true);
        let frame = hosted_frame(serde_json::json!({"type":"response.completed", "response":{
            "status":"completed", "output":[{"type":"function_call"}, item]
        }}));
        let events = normalizer
            .feed(&frame)
            .expect("valid index remains distinct");
        assert!(events.iter().any(|event| matches!(event,
            Event::ProviderResponsesLogprobs { output_index, .. } if *output_index == explicit.unwrap_or(1)
        )));
    }
}

#[test]
fn nullable_probabilities_are_accepted_only_on_optional_completed_parts() {
    for event_type in [
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
    ] {
        let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
        normalizer.enable_responses_logprobs(true);
        let frame = hosted_frame(serde_json::json!({
            "type": event_type, "output_index": 0, "item_id": "msg-null", "content_index": 0,
            "delta": "", "logprobs": null,
        }));
        let result = normalizer.feed(&frame);
        assert_eq!(result.is_ok(), event_type == "response.content_part.done");
    }
}
