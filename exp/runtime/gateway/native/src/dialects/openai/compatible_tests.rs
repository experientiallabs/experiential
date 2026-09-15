//! Azure annotation-only frames share the compatible content and finish lifecycle.

use crate::dialects::{drain_stream_fixture, Dialect, Normalizer};
use crate::errors::FailureClass;
use crate::events::Event;
use crate::sse::SseEvent;
use serde_json::{json, Value};

fn frame(choice: Value) -> SseEvent {
    SseEvent {
        event: None,
        data: json!({"choices": [choice]}).to_string(),
    }
}

fn annotation(finish: Option<&str>) -> Value {
    json!({
        "index": 0, "finish_reason": finish,
        "content_filter_results": {
            "hate": {"filtered": finish == Some("content_filter"), "severity": "safe"}
        },
        "content_filter_offsets": {"check_offset": 49, "start_offset": 47, "end_offset": 49}
    })
}

#[test]
fn azure_annotations_preserve_text_finish_and_trailing_usage() {
    for after_stop in [false, true] {
        let mut frames = vec![
            json!({"choices": [], "prompt_filter_results": []}),
            json!({"choices": [{"index": 0, "delta": {"role": "assistant"}}]}),
            json!({"choices": [annotation(None)]}),
            json!({"choices": [{"index": 0, "delta": {"content": "OK"}}]}),
        ];
        let stop = json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]});
        let annotation = json!({"choices": [annotation(None)]});
        frames.extend(if after_stop {
            vec![stop, annotation]
        } else {
            vec![annotation, stop]
        });
        frames.push(json!({"choices": [], "usage": {"prompt_tokens": 13, "completion_tokens": 1, "total_tokens": 14}}));
        let wire = frames
            .iter()
            .map(|value| format!("data: {value}\n\n"))
            .collect::<String>()
            + "data: [DONE]\n\n";
        // Decode across arbitrary transport boundaries, including inside JSON.
        let chunks = wire
            .as_bytes()
            .chunks(7)
            .map(<[u8]>::to_vec)
            .collect::<Vec<_>>();
        let (events, failure) = drain_stream_fixture(Dialect::OpenAiCompatible, &chunks);
        assert!(failure.is_none(), "{failure:?}");
        assert_eq!(
            events,
            vec![
                json!({"kind": "text_delta", "text": "OK"}),
                json!({"kind": "usage", "input_tokens": 13, "output_tokens": 1, "cached_input_tokens": null, "reasoning_tokens": null}),
                json!({"kind": "completed"}),
            ]
        );
    }
}

#[test]
fn azure_filter_annotation_preserves_refusal_instead_of_completion() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let events = normalizer
        .feed(&frame(annotation(Some("content_filter"))))
        .unwrap();
    assert!(matches!(events.as_slice(), [Event::RefusalDelta(_)]));
    normalizer.feed(&frame(annotation(None))).unwrap();
    let events = normalizer
        .feed(&SseEvent {
            event: None,
            data: "[DONE]".into(),
        })
        .unwrap();
    assert!(
        matches!(events.as_slice(), [Event::Failed(failure)] if failure.failure_class == FailureClass::Refusal)
    );
}

#[test]
fn compatible_missing_or_invalid_deltas_still_fail_without_valid_annotations() {
    let mut cases = vec![
        json!({"index": 0}),
        json!({"index": 0, "finish_reason": "stop"}),
    ];
    for delta in [Value::Null, json!("bad"), json!([]), json!(0)] {
        let mut choice = annotation(None);
        choice["delta"] = delta;
        cases.push(choice);
    }
    for field in ["content_filter_results", "content_filter_offsets"] {
        for value in [Value::Null, json!("bad"), json!([])] {
            let mut choice = annotation(None);
            choice[field] = value;
            cases.push(choice);
        }
        let mut choice = annotation(None);
        choice.as_object_mut().unwrap().remove(field);
        cases.push(choice);
    }
    for choice in cases {
        let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
        assert!(
            normalizer.feed(&frame(choice.clone())).is_err(),
            "accepted {choice}"
        );
    }
}
