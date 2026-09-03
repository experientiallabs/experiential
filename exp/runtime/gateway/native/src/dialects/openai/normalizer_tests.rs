//! OpenAI dialect normalizer regressions (moved from `openai.rs` for the
//! module line budget).

use super::*;
use crate::dialects::{Dialect, MAXIMUM_RETAINED_PROVIDER_ENTRIES, OUTPUT_OVERFLOW_MESSAGE};
use crate::sse::SseEvent;
fn reasoning_delta(output_index: u32, summary_index: u32, delta: &str) -> SseEvent {
    SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": format!("rs-{output_index}"),
            "output_index": output_index,
            "summary_index": summary_index,
            "delta": delta,
        })
        .to_string(),
    }
}

#[test]
fn custom_tool_call_stream_normalizes_like_a_freeform_tool_call() {
    // Exact payload shapes captured live from api.openai.com
    // (response.output_item.added / custom_tool_call_input.delta / .done,
    // 2026-08-30); input is opaque text, never JSON-validated.
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    let added = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "sequence_number": 2,
            "item": {
                "id": "ctc_live", "type": "custom_tool_call",
                "status": "in_progress",
                "call_id": "call_live", "input": "", "name": "exec",
            },
        })
        .to_string(),
    };
    let events = normalizer
        .feed(&added)
        .expect("custom start must normalize");
    assert!(matches!(
        events.as_slice(),
        [
            Event::ProviderOutputItemStarted {
                kind: ProviderOutputItemKind::CustomToolCall,
                ..
            },
            Event::ToolCallStarted { call_id, name, .. },
        ] if call_id == "call_live" && name == "exec"
    ));
    let delta = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.custom_tool_call_input.delta",
            "delta": "const r = 1;",
            "item_id": "ctc_live",
            "obfuscation": "x",
            "output_index": 0,
            "sequence_number": 3,
        })
        .to_string(),
    };
    let events = normalizer
        .feed(&delta)
        .expect("custom delta must normalize");
    assert!(matches!(
        events.as_slice(),
        [Event::ToolArgumentsDelta { delta, .. }] if delta == "const r = 1;"
    ));
    let done = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.custom_tool_call_input.done",
            "input": "const r = 1;",
            "item_id": "ctc_live",
            "output_index": 0,
            "sequence_number": 4,
        })
        .to_string(),
    };
    assert!(normalizer
        .feed(&done)
        .expect("matching input done must validate")
        .is_empty());
    let item_done = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "ctc_live", "type": "custom_tool_call",
                "status": "completed",
                "call_id": "call_live", "input": "const r = 1;", "name": "exec",
            },
        })
        .to_string(),
    };
    let events = normalizer
        .feed(&item_done)
        .expect("custom completion must normalize");
    assert!(matches!(
        events.as_slice(),
        [
            Event::ProviderOutputItemCompleted {
                kind: ProviderOutputItemKind::CustomToolCall,
                ..
            },
            Event::ToolCallCompleted { call, .. },
        ] if call.custom && call.raw_arguments == "const r = 1;" && call.name == "exec"
    ));
}

#[test]
fn compatible_reasoning_content_requires_fireworks_route_authority() {
    let frame = SseEvent {
        event: None,
        data: serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": {"reasoning_content": "provider private"},
                "finish_reason": null,
            }]
        })
        .to_string(),
    };
    let route_sha256 = "a".repeat(64);
    let mut authorized = Normalizer::new_with_reasoning_content_route(
        Dialect::OpenAiCompatible,
        Some(route_sha256.clone()),
    );
    let events = authorized
        .feed(&frame)
        .expect("authorized Fireworks reasoning must normalize");
    assert!(matches!(
        events.as_slice(),
        [Event::ReasoningContentDelta {
            route_sha256: route,
            delta,
        }] if route == &route_sha256 && delta == "provider private"
    ));

    let mut generic = Normalizer::new(Dialect::OpenAiCompatible);
    assert!(generic
        .feed(&frame)
        .expect("generic compatible extension is ignored")
        .is_empty());
}

#[test]
fn fireworks_reasoning_content_rejects_non_text_values() {
    let frame = SseEvent {
        event: None,
        data: serde_json::json!({
            "choices": [{
                "index": 0,
                "delta": {"reasoning_content": {"private": true}},
                "finish_reason": null,
            }]
        })
        .to_string(),
    };
    let mut normalizer = Normalizer::new_with_reasoning_content_route(
        Dialect::OpenAiCompatible,
        Some("a".repeat(64)),
    );

    assert!(normalizer.feed(&frame).is_err());
}

#[test]
fn responses_reasoning_summary_is_normalized_and_verified() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    let delta = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs-2",
            "output_index": 2,
            "summary_index": 1,
            "delta": "checked",
        })
        .to_string(),
    };
    let events = normalizer
        .feed(&delta)
        .expect("summary delta must normalize");
    assert!(matches!(
        events.as_slice(),
        [
            Event::ProviderOutputItemStarted {
                output_index: 2,
                item_id,
                kind: ProviderOutputItemKind::Reasoning,
                ..
            },
            Event::ReasoningSummaryDelta {
                output_index: 2,
                summary_index: 1,
                item_id: _,
                delta,
            }
        ] if item_id.as_deref() == Some("rs-2") && delta == "checked"
    ));

    let done = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.reasoning_summary_text.done",
            "item_id": "rs-2",
            "output_index": 2,
            "summary_index": 1,
            "text": "checked",
        })
        .to_string(),
    };
    assert!(normalizer
        .feed(&done)
        .expect("matching summary completion must validate")
        .is_empty());
}

#[test]
fn empty_reasoning_deltas_do_not_allocate_provider_state() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    for output_index in 0..=MAXIMUM_RETAINED_PROVIDER_ENTRIES as u32 {
        assert!(normalizer
            .feed(&reasoning_delta(output_index, 0, ""))
            .expect("empty summary delta must be ignored")
            .is_empty());
    }
    assert!(normalizer.reasoning_summaries.is_empty());

    assert_eq!(
        normalizer
            .feed(&reasoning_delta(0, 0, "bounded"))
            .expect("non-empty summary still fits")
            .len(),
        2
    );
}

#[test]
fn retained_provider_entries_are_bounded_across_tools_and_summaries() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    for index in 0..MAXIMUM_RETAINED_PROVIDER_ENTRIES as u32 {
        normalizer
            .reserve_tool_entry(index)
            .expect("entry below ceiling must fit");
        normalizer.tools.insert(
            index,
            ToolAccumulator::new(format!("call-{index}"), "lookup".to_string()),
        );
    }

    let failure = normalizer
        .feed(&reasoning_delta(0, 0, "overflow"))
        .expect_err("entry above ceiling must fail");
    assert_eq!(failure.failure_class, FailureClass::ProviderInternal);
    assert_eq!(failure.safe_message, OUTPUT_OVERFLOW_MESSAGE);
}

#[test]
fn completed_reasoning_items_pass_encrypted_content_through() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    let done = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "rs_provider",
                "type": "reasoning",
                "summary": [],
                "encrypted_content": "blob==",
                "status": "completed",
            },
        })
        .to_string(),
    };
    let events = normalizer.feed(&done).expect("reasoning item completes");
    assert!(matches!(
        events.as_slice(),
        [
            Event::ProviderOutputItemStarted {
                output_index: 0,
                item_id,
                kind: ProviderOutputItemKind::Reasoning,
                ..
            },
            Event::EncryptedReasoning {
                output_index: 0,
                item_id: _,
                encrypted_content,
            },
            Event::ProviderOutputItemCompleted {
                output_index: 0,
                status: Some(ProviderOutputItemStatus::Completed),
                ..
            },
        ] if item_id.as_deref() == Some("rs_provider") && encrypted_content == "blob=="
    ));

    // A reasoning item without the requested include stays silent.
    let bare = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {"id": "rs_2", "type": "reasoning", "summary": []},
        })
        .to_string(),
    };
    assert!(matches!(
        normalizer.feed(&bare).expect("bare item").as_slice(),
        [
            Event::ProviderOutputItemStarted {
                output_index: 1,
                item_id,
                kind: ProviderOutputItemKind::Reasoning,
                ..
            },
            Event::ProviderOutputItemCompleted {
                output_index: 1,
                ..
            },
        ] if item_id.as_deref() == Some("rs_2")
    ));
}

fn compatible_chunk(delta: serde_json::Value, finish_reason: Option<&str>) -> SseEvent {
    SseEvent {
        event: None,
        data: serde_json::json!({
            "id": "chatcmpl-dashscope",
            "object": "chat.completion.chunk",
            "created": 1_788_425_855,
            "model": "qwen3.8-flash",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        })
        .to_string(),
    }
}

#[test]
fn dashscope_argument_deltas_restate_an_empty_tool_call_id() {
    // DashScope's documented (and live, 2026-09-03) OpenAI-compatible tool
    // stream: the first delta names the call, every later argument delta
    // restates `"id": ""` with a null name. An empty placeholder is not a
    // changed identity, so the call must accumulate and complete normally
    // (this exact shape 502'd every qwen tool call as malformed_response).
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    assert!(normalizer
        .feed(&compatible_chunk(
            serde_json::json!({"role": "assistant", "content": ""}),
            None,
        ))
        .expect("role delta must normalize")
        .is_empty());
    let started = normalizer
        .feed(&compatible_chunk(
            serde_json::json!({"tool_calls": [{
                "index": 0,
                "id": "call_8f08d2b0fc0c4d8fab7123",
                "type": "function",
                "function": {"name": "get_current_weather", "arguments": "{\"location\":"},
            }]}),
            None,
        ))
        .expect("first tool delta must normalize");
    assert!(matches!(
        started.as_slice(),
        [
            Event::ToolCallStarted { call_id, name, .. },
            Event::ToolArgumentsDelta { delta, .. },
        ] if call_id == "call_8f08d2b0fc0c4d8fab7123"
            && name == "get_current_weather"
            && delta == "{\"location\":"
    ));
    let continued = normalizer
        .feed(&compatible_chunk(
            serde_json::json!({"tool_calls": [{
                "index": 0,
                "id": "",
                "type": "function",
                "function": {"arguments": " \"Hangzhou\"}", "name": null},
            }]}),
            None,
        ))
        .expect("an empty restated id is a placeholder, not a changed identity");
    assert!(matches!(
        continued.as_slice(),
        [Event::ToolArgumentsDelta { delta, .. }] if delta == " \"Hangzhou\"}"
    ));
    assert!(normalizer
        .feed(&compatible_chunk(serde_json::json!({}), Some("tool_calls")))
        .expect("finish chunk must normalize")
        .is_empty());
    let done = SseEvent {
        event: None,
        data: "[DONE]".to_string(),
    };
    let events = normalizer.feed(&done).expect("stream must complete");
    assert!(matches!(
        events.as_slice(),
        [Event::ToolCallCompleted { call, .. }, Event::Completed]
            if call.call_id == "call_8f08d2b0fc0c4d8fab7123"
                && call.name == "get_current_weather"
                && call.raw_arguments == "{\"location\": \"Hangzhou\"}"
    ));
}

#[test]
fn compatible_stream_still_rejects_a_changed_non_empty_tool_call_identity() {
    // The identity guard keeps its teeth: a later delta naming a DIFFERENT
    // non-empty id or name is still a malformed stream.
    for (id, name) in [
        ("call_other", "get_current_weather"),
        ("call_1", "other_tool"),
    ] {
        let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
        normalizer
            .feed(&compatible_chunk(
                serde_json::json!({"tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_current_weather", "arguments": ""},
                }]}),
                None,
            ))
            .expect("first tool delta must normalize");
        let failure = normalizer
            .feed(&compatible_chunk(
                serde_json::json!({"tool_calls": [{
                    "index": 0,
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }]}),
                None,
            ))
            .expect_err("a changed non-empty identity must stay malformed");
        assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
    }
}

#[test]
fn compatible_stream_folds_additive_reasoning_into_the_terminal_usage() {
    // Verbatim final frames from Azure Foundry grok-4.3 (silen-resource,
    // 2026-09-03, stream_options.include_usage): xAI reports 655 reasoning
    // tokens OUTSIDE completion_tokens=8, which its total_tokens identifies
    // (677 = 14 + 8 + 655), so the normalized usage carries the folded output
    // total with the reasoning subset intact, and the cached prompt leg passes
    // through.
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let text = SseEvent {
        event: None,
        data: serde_json::json!({
            "id": "8ca8705f-1504-4bec-a739-38e3726ff3d4",
            "object": "chat.completion.chunk",
            "created": 1788425522,
            "model": "grok-4.3",
            "choices": [{"index": 0, "delta": {"content": "Because"}, "finish_reason": null}],
            "system_fingerprint": "fp_39c5j0a3e9",
        })
        .to_string(),
    };
    assert!(matches!(
        normalizer.feed(&text).expect("text").as_slice(),
        [Event::TextDelta(delta)] if delta == "Because"
    ));
    let finish = SseEvent {
        event: None,
        data: serde_json::json!({
            "id": "8ca8705f-1504-4bec-a739-38e3726ff3d4",
            "object": "chat.completion.chunk",
            "created": 1788425522,
            "model": "grok-4.3",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "system_fingerprint": "fp_39c5j0a3e9",
        })
        .to_string(),
    };
    assert!(normalizer.feed(&finish).expect("finish").is_empty());
    let usage = SseEvent {
        event: None,
        data: serde_json::json!({
            "id": "8ca8705f-1504-4bec-a739-38e3726ff3d4",
            "object": "chat.completion.chunk",
            "created": 1788425522,
            "model": "grok-4.3",
            "choices": [],
            "usage": {
                "prompt_tokens": 14,
                "completion_tokens": 8,
                "total_tokens": 677,
                "prompt_tokens_details": {"text_tokens": 14, "audio_tokens": 0, "image_tokens": 0, "cached_tokens": 4},
                "completion_tokens_details": {"reasoning_tokens": 655, "audio_tokens": 0, "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0},
                "num_sources_used": 0,
                "cost_in_usd_ticks": 0,
            },
            "system_fingerprint": "fp_39c5j0a3e9",
            "service_tier": "default",
        })
        .to_string(),
    };
    assert!(normalizer.feed(&usage).expect("usage").is_empty());
    let done = SseEvent {
        event: None,
        data: "[DONE]".to_string(),
    };
    let events = normalizer.feed(&done).expect("terminal");
    match events.as_slice() {
        [Event::Usage(usage), Event::Completed] => {
            assert_eq!(usage.input_tokens, Some(14));
            assert_eq!(usage.output_tokens, Some(663));
            assert_eq!(usage.cached_input_tokens, Some(4));
            assert_eq!(usage.reasoning_tokens, Some(655));
        }
        other => panic!("unexpected events: {other:?}"),
    }
}

#[test]
fn a_tool_call_cut_off_by_the_output_budget_is_incomplete_not_malformed() {
    // Live shape (Tencent TokenHub glm-5.3, max_tokens=32, staging
    // 2026-09-03): the call starts, two argument fragments arrive, then the
    // provider finishes with `length`. The truncated call is dropped and the
    // stream ends Incomplete — the caller's remedy is a larger budget, so a
    // 502 "malformed response" was the wrong verdict.
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let started = normalizer
        .feed(&compatible_chunk(
            serde_json::json!({"tool_calls": [{
                "index": 0, "id": "call_73e9f9cfb9004dc1aaa71615", "type": "function",
                "function": {"name": "get_weather", "arguments": ""},
            }]}),
            None,
        ))
        .expect("tool start must normalize");
    // The empty first `arguments` rides as an empty delta, as on every wire.
    assert!(matches!(
        started.as_slice(),
        [
            Event::ToolCallStarted { .. },
            Event::ToolArgumentsDelta { .. }
        ]
    ));
    for fragment in ["{\"", "city"] {
        normalizer
            .feed(&compatible_chunk(
                serde_json::json!({"tool_calls": [{"index": 0, "function": {"arguments": fragment}}]}),
                None,
            ))
            .expect("argument fragments must normalize");
    }
    assert!(normalizer
        .feed(&compatible_chunk(serde_json::json!({}), Some("length")))
        .expect("length finish must normalize")
        .is_empty());
    let events = normalizer
        .feed(&SseEvent {
            event: None,
            data: "[DONE]".to_string(),
        })
        .expect("a length-truncated tool call must not be malformed");
    assert!(matches!(events.as_slice(), [Event::Incomplete]));
}

#[test]
fn a_complete_tool_call_still_completes_when_the_budget_ends_the_stream() {
    // finish_reason=length AFTER the arguments closed: the call is intact and
    // must still be delivered; only the terminal reads Incomplete.
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    normalizer
        .feed(&compatible_chunk(
            serde_json::json!({"tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"},
            }]}),
            Some("length"),
        ))
        .expect("tool chunk must normalize");
    let events = normalizer
        .feed(&SseEvent {
            event: None,
            data: "[DONE]".to_string(),
        })
        .expect("stream must finish");
    assert!(matches!(
        events.as_slice(),
        [Event::ToolCallCompleted { call, .. }, Event::Incomplete]
            if call.raw_arguments == "{\"city\": \"Paris\"}"
    ));
}

#[test]
fn unparsable_tool_arguments_stay_malformed_on_a_normal_finish() {
    // The strict contract holds whenever the provider claims it finished the
    // call: a `tool_calls`/`stop` terminal with a dangling fragment is still a
    // malformed stream, never silently dropped.
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    normalizer
        .feed(&compatible_chunk(
            serde_json::json!({"tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city"},
            }]}),
            Some("tool_calls"),
        ))
        .expect("tool chunk must normalize");
    let failure = normalizer
        .feed(&SseEvent {
            event: None,
            data: "[DONE]".to_string(),
        })
        .expect_err("a dangling fragment on a normal finish is malformed");
    assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
}
