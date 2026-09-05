//! Budget-truncation regressions for the OpenAI Responses dialect (split
//! from `normalizer_tests.rs` for the module line budget): the provider's
//! own output cut ends calls honestly instead of killing the stream.

use super::*;
use crate::dialects::{Dialect, Normalizer};
use crate::sse::SseEvent;

/// Frame shapes captured live from api.openai.com (gpt-6-astra, 2026-09-05):
/// the model hits max_output_tokens mid-arguments and the wire reports it
/// honestly with `function_call_arguments.done` carrying the PARTIAL bytes,
/// the item's own status `incomplete`, and a `response.incomplete` terminal.
///
/// Production incident (2026-09-05, ~5/min): the partial arguments failed the
/// strict JSON completion contract and killed the whole stream as
/// malformed_response. A provider-declared truncation drops the mid-fragment
/// call and surfaces Incomplete, mirroring the Chat lane's
/// finish_reason=length contract; the caller's remedy is a larger budget.
#[test]
fn astra_budget_truncated_function_call_surfaces_incomplete_not_malformed() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    let truncated_args = "{\"city\":\"Paris\",\"country\":\"France\",\"units\":\"metric";
    let added = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "fc_astra_live", "type": "function_call", "status": "in_progress",
                "arguments": "", "call_id": "call_astra_live", "name": "get_weather",
            },
        })
        .to_string(),
    };
    normalizer.feed(&added).expect("start normalizes");
    for fragment in [
        "{\"",
        "city",
        "\":\"",
        "Paris",
        "\",\"country\":\"France\",\"units\":\"metric",
    ] {
        let delta = SseEvent {
            event: None,
            data: serde_json::json!({
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_astra_live",
                "output_index": 0,
                "delta": fragment,
            })
            .to_string(),
        };
        normalizer.feed(&delta).expect("fragment normalizes");
    }
    let arguments_done = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.function_call_arguments.done",
            "item_id": "fc_astra_live",
            "output_index": 0,
            "arguments": truncated_args,
        })
        .to_string(),
    };
    normalizer
        .feed(&arguments_done)
        .expect("partial done matches streamed bytes");
    let item_done = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "fc_astra_live", "type": "function_call", "status": "incomplete",
                "arguments": truncated_args, "call_id": "call_astra_live", "name": "get_weather",
            },
        })
        .to_string(),
    };
    let events = normalizer
        .feed(&item_done)
        .expect("a provider-declared truncation must not kill the stream");
    assert!(matches!(
        events.as_slice(),
        [Event::ProviderOutputItemCompleted {
            kind: ProviderOutputItemKind::FunctionCall,
            status: Some(ProviderOutputItemStatus::Incomplete),
            ..
        }]
    ));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, Event::ToolCallCompleted { .. })),
        "a mid-fragment call is dropped, never completed"
    );
    let terminal = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.incomplete",
            "response": {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": {"input_tokens": 62, "output_tokens": 24, "total_tokens": 86},
            },
        })
        .to_string(),
    };
    let events = normalizer
        .feed(&terminal)
        .expect("incomplete terminal normalizes");
    assert!(matches!(events.last(), Some(Event::Incomplete)));
    match &events[events.len() - 2] {
        Event::Usage(usage) => assert_eq!(usage.output_tokens, Some(24)),
        other => panic!("expected terminal usage, got {other:?}"),
    }
}

/// A truncated item whose accumulated arguments still parse (the budget cut
/// after the closing brace) completes normally with its incomplete status.
#[test]
fn truncated_item_with_complete_arguments_still_completes_the_call() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    let added = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "fc_1", "type": "function_call", "status": "in_progress",
                "arguments": "{\"city\":\"Paris\"}", "call_id": "call_1", "name": "get_weather",
            },
        })
        .to_string(),
    };
    normalizer.feed(&added).expect("start normalizes");
    let item_done = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "fc_1", "type": "function_call", "status": "incomplete",
                "arguments": "{\"city\":\"Paris\"}", "call_id": "call_1", "name": "get_weather",
            },
        })
        .to_string(),
    };
    let events = normalizer
        .feed(&item_done)
        .expect("parsable truncation completes");
    assert!(matches!(
        events.last(),
        Some(Event::ToolCallCompleted { call, .. })
            if call.raw_arguments == "{\"city\":\"Paris\"}"
                && call.provider_status == Some(ProviderOutputItemStatus::Incomplete)
    ));
}

/// A call the terminal cut before its own item done: the incomplete terminal
/// sweeps it with the truncated-call contract instead of failing the stream.
#[test]
fn incomplete_terminal_drops_a_still_open_mid_fragment_call() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    let added = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "fc_1", "type": "function_call", "status": "in_progress",
                "arguments": "{\"ci", "call_id": "call_1", "name": "get_weather",
            },
        })
        .to_string(),
    };
    normalizer.feed(&added).expect("start normalizes");
    let terminal = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.incomplete",
            "response": {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": {"input_tokens": 9, "output_tokens": 4},
            },
        })
        .to_string(),
    };
    let events = normalizer
        .feed(&terminal)
        .expect("terminal sweeps the open call");
    assert!(matches!(events.last(), Some(Event::Incomplete)));
    assert!(!events
        .iter()
        .any(|event| matches!(event, Event::ToolCallCompleted { .. })));
}

/// A COMPLETED item with unparsable arguments keeps the strict contract, and
/// the reason now carries the byte count so logs name the corruption size.
#[test]
fn completed_item_with_invalid_arguments_stays_malformed_with_byte_count() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    let added = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "fc_1", "type": "function_call", "status": "in_progress",
                "arguments": "{\"ci", "call_id": "call_1", "name": "get_weather",
            },
        })
        .to_string(),
    };
    normalizer.feed(&added).expect("start normalizes");
    let item_done = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "fc_1", "type": "function_call", "status": "completed",
                "arguments": "{\"ci", "call_id": "call_1", "name": "get_weather",
            },
        })
        .to_string(),
    };
    let failure = normalizer
        .feed(&item_done)
        .expect_err("a completed call with corrupt arguments stays fail-closed");
    assert!(
        failure.safe_message.contains("not valid JSON")
            && failure.safe_message.contains("(4 bytes)"),
        "reason must name the corruption size: {}",
        failure.safe_message
    );
}

/// A content_filter (or safety) cut with a mid-fragment call keeps its
/// refusal classification: the sweep tolerates the provider-declared cut so
/// the stream reaches the refusal terminal instead of being pre-empted as a
/// malformed 502, and the dropped call is never served (the terminal is a
/// failure, so no output reaches the caller).
#[test]
fn content_filter_truncation_surfaces_the_refusal_not_malformed() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    let added = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "fc_1", "type": "function_call", "status": "in_progress",
                "arguments": "{\"ci", "call_id": "call_1", "name": "get_weather",
            },
        })
        .to_string(),
    };
    normalizer.feed(&added).expect("start normalizes");
    let terminal = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.incomplete",
            "response": {
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
                "usage": {"input_tokens": 9, "output_tokens": 4},
            },
        })
        .to_string(),
    };
    let events = normalizer
        .feed(&terminal)
        .expect("a filtered cut must reach its own refusal terminal");
    match events.last() {
        Some(Event::Failed(failure)) => {
            assert_eq!(failure.failure_class, crate::errors::FailureClass::Refusal);
        }
        other => panic!("expected the refusal terminal, got {other:?}"),
    }
    assert!(!events
        .iter()
        .any(|event| matches!(event, Event::ToolCallCompleted { .. })));
}
