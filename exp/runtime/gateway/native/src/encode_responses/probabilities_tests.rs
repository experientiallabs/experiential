//! Probability-specific Responses lifecycle regressions.

use super::*;

#[test]
fn probability_item_done_closes_before_next_item_and_rejects_late_text() {
    let mut encoder = ResponsesSseEncoder::new("req", "model", 1, ResponsesEnvelope::default());
    encoder.start().unwrap();
    encoder
        .feed(&Event::ProviderResponsesLogprobs {
            output_index: 0,
            item_id: "a".into(),
            content_index: 0,
            phase: "delta".into(),
            records: json!([{"token":"A", "logprob":-0.1}]),
        })
        .unwrap();
    let closed = encoder
        .feed(&Event::ProviderOutputItemCompleted {
            output_index: 0,
            item_id: Some("a".into()),
            kind: ProviderOutputItemKind::Message,
            status: Some(ProviderOutputItemStatus::Completed),
            phase: None,
        })
        .unwrap();
    assert!(closed
        .iter()
        .any(|frame| frame.contains("response.output_item.done")));
    let next = encoder
        .feed(&Event::ProviderTextDelta {
            output_index: 1,
            item_id: "b".into(),
            delta: "B".into(),
        })
        .unwrap();
    assert!(next
        .iter()
        .any(|frame| frame.contains("response.output_item.added")));
    assert!(encoder
        .feed(&Event::ProviderTextDelta {
            output_index: 0,
            item_id: "a".into(),
            delta: "late".into()
        })
        .is_err());
    encoder
        .feed(&Event::ProviderResponsesLogprobs {
            output_index: 0,
            item_id: "a".into(),
            content_index: 0,
            phase: "terminal".into(),
            records: json!([{"token":"A", "logprob":-0.2}]),
        })
        .unwrap();
    assert_eq!(
        encoder.response("completed", None)["output"][0]["content"][0]["logprobs"][0]["logprob"],
        -0.2
    );
}

#[test]
fn terminal_only_probabilities_create_a_content_part_without_a_text_delta() {
    let mut encoder = ResponsesSseEncoder::new("req", "model", 1, ResponsesEnvelope::default());
    encoder.start().unwrap();
    let frames = encoder
        .feed(&Event::ProviderResponsesLogprobs {
            output_index: 0,
            item_id: "a".into(),
            content_index: 0,
            phase: "terminal".into(),
            records: json!([{"token":"A", "logprob":-0.1}]),
        })
        .unwrap();
    assert!(frames
        .iter()
        .any(|frame| frame.contains("response.output_item.added")));
    assert!(frames
        .iter()
        .any(|frame| frame.contains("response.content_part.added")));
    assert!(!frames
        .iter()
        .any(|frame| frame.contains("response.output_text.delta")));
    assert_eq!(
        encoder.response("completed", None)["output"][0]["content"][0]["logprobs"][0]["token"],
        "A"
    );
}

#[test]
fn inferred_probability_start_upgrades_once_but_explicit_duplicates_fail() {
    let mut encoder = ResponsesSseEncoder::new("req", "model", 1, ResponsesEnvelope::default());
    encoder.start().unwrap();
    encoder
        .feed(&Event::ProviderResponsesLogprobs {
            output_index: 0,
            item_id: "a".into(),
            content_index: 0,
            phase: "delta".into(),
            records: json!([]),
        })
        .unwrap();
    let start = Event::ProviderOutputItemStarted {
        output_index: 0,
        item_id: Some("a".into()),
        kind: ProviderOutputItemKind::Message,
        status: Some(ProviderOutputItemStatus::InProgress),
        phase: None,
    };
    assert!(encoder.feed(&start).unwrap().is_empty());
    assert!(encoder.feed(&start).is_err());
}

#[test]
fn every_first_probability_phase_normalizes_and_encodes_a_complete_lifecycle() {
    use crate::dialects::{Dialect, Normalizer};
    use crate::sse::SseEvent;
    for phase in [
        "delta",
        "text_done",
        "content_part_done",
        "item_done",
        "terminal",
    ] {
        let records = json!([{"token":"A", "logprob":-0.1, "bytes":[65], "top_logprobs":[]}]);
        let item = json!({"id":"a", "type":"message", "status":"completed", "role":"assistant",
            "content":[{"type":"output_text", "text":"A", "logprobs":records.clone()}]});
        let terminal = json!({"type":"response.completed", "response":{"status":"completed",
            "output":[item.clone()], "usage":{"input_tokens":1,"output_tokens":1}}});
        let mut frames = Vec::new();
        match phase {
            "delta" => frames.push(json!({"type":"response.output_text.delta", "output_index":0,
                "item_id":"a", "content_index":0,"delta":"", "logprobs":records.clone()})),
            "text_done" => frames.push(json!({"type":"response.output_text.done", "output_index":0,
                "item_id":"a", "content_index":0,"text":"", "logprobs":records.clone()})),
            "content_part_done" => frames.push(json!({"type":"response.content_part.done", "output_index":0,
                "item_id":"a", "content_index":0,"part":{"type":"output_text","text":"", "logprobs":records.clone()}})),
            "item_done" | "terminal" => {},
            _ => unreachable!(),
        }
        if phase != "terminal" {
            if phase != "item_done" {
                frames.push(
                    json!({"type":"response.output_text.delta", "output_index":0,
                    "item_id":"a", "content_index":0,"delta":"A", "logprobs":[]}),
                );
            }
            frames.push(json!({"type":"response.output_item.done", "output_index":0,"item":item}));
        }
        frames.push(terminal);
        let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
        normalizer.enable_responses_logprobs(true);
        let mut encoder =
            ResponsesSseEncoder::new("request", "model", 1, ResponsesEnvelope::default());
        encoder.start().unwrap();
        let mut public = Vec::new();
        for frame in frames {
            let events = normalizer
                .feed(&SseEvent {
                    event: None,
                    data: frame.to_string(),
                })
                .unwrap();
            for event in events {
                public.extend(encoder.feed(&event).unwrap());
            }
        }
        assert!(
            public
                .iter()
                .any(|frame| frame.contains("response.completed")),
            "{phase}"
        );
        assert_eq!(
            encoder.response("completed", None)["output"][0]["content"][0]["logprobs"],
            records,
            "{phase}"
        );
    }
}
