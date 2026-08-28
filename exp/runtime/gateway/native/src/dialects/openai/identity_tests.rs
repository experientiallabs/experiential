//! OpenAI Responses output-item identity regressions.

use super::*;
use crate::dialects::Dialect;
use crate::sse::SseEvent;

#[test]
fn responses_output_item_identity_is_bounded_and_type_stable() {
    for item_id in [String::new(), "x".repeat(257)] {
        let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
        let frame = SseEvent {
            event: None,
            data: serde_json::json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"id": item_id, "type": "reasoning", "summary": []},
            })
            .to_string(),
        };
        assert_eq!(
            normalizer
                .feed(&frame)
                .expect_err("invalid item identity must fail")
                .failure_class,
            FailureClass::MalformedResponse
        );
    }

    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    let reasoning_start = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "rs-0", "type": "reasoning", "summary": []},
        })
        .to_string(),
    };
    normalizer
        .feed(&reasoning_start)
        .expect("reasoning identity must bind");
    let mismatched_summary = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs-other",
            "output_index": 0,
            "summary_index": 0,
            "delta": "mismatch",
        })
        .to_string(),
    };
    assert!(normalizer.feed(&mismatched_summary).is_err());

    let cross_type = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "fc-0",
                "type": "function_call",
                "call_id": "call-0",
                "name": "lookup",
                "arguments": "{}",
            },
        })
        .to_string(),
    };
    assert!(normalizer.feed(&cross_type).is_err());
}

#[test]
fn responses_function_call_completion_preserves_every_identity_field() {
    let start = SseEvent {
        event: None,
        data: serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "id": "fc-1",
                "type": "function_call",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": "{}",
            },
        })
        .to_string(),
    };
    for item in [
        serde_json::json!({
            "id": "fc-other", "type": "function_call", "call_id": "call-1",
            "name": "lookup", "arguments": "{}",
        }),
        serde_json::json!({
            "id": "fc-1", "type": "function_call", "call_id": "call-other",
            "name": "lookup", "arguments": "{}",
        }),
        serde_json::json!({
            "id": "fc-1", "type": "function_call", "call_id": "call-1",
            "name": "other", "arguments": "{}",
        }),
        serde_json::json!({
            "id": "fc-1", "type": "message", "call_id": "call-1",
            "name": "lookup", "arguments": "{}",
        }),
    ] {
        let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
        let events = normalizer.feed(&start).expect("function call must start");
        assert!(matches!(
            events.as_slice(),
            [
                Event::ProviderOutputItemStarted {
                    output_index: 1,
                    item_id,
                    kind: ProviderOutputItemKind::FunctionCall,
                    ..
                },
                Event::ToolCallStarted { index: 1, .. },
                Event::ToolArgumentsDelta { index: 1, .. },
            ] if item_id.as_deref() == Some("fc-1")
        ));
        let done = SseEvent {
            event: None,
            data: serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 1,
                "item": item,
            })
            .to_string(),
        };
        assert!(normalizer.feed(&done).is_err());
    }
}
