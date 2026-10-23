//! Regression tests for Chat token probability normalization and encoding.

use serde_json::Value;

use crate::dialects::{Dialect, Normalizer};
use crate::encode::{completed_chat_body_with_ignored, ChatSseEncoder};
use crate::events::{ChoiceLogprobsDelta, Event};

fn update(content: Option<Value>, refusal: Option<Value>) -> Event {
    Event::ChoiceLogprobsDelta(ChoiceLogprobsDelta {
        choice_index: 0,
        logprobs: Some(crate::events::ChoiceLogprobs {
            content: content.map(|value| serde_json::from_value(value).unwrap()),
            refusal: refusal.map(|value| serde_json::from_value(value).unwrap()),
        }),
    })
}

#[test]
fn encoder_does_not_repeat_probability_records_on_text_or_finish_chunks() {
    let event = update(
        Some(serde_json::json!([{"token":"a","logprob":-0.5,"bytes":[97],"top_logprobs":[]}])),
        None,
    );
    let mut encoder = ChatSseEncoder::new_with_ignored("r", "m", 1, false, vec![]);
    encoder.start().unwrap();
    let probability = encoder.feed(&event).unwrap().remove(0);
    let text = encoder
        .feed(&Event::TextDelta("a".into()))
        .unwrap()
        .remove(0);
    let finish = encoder.feed(&Event::Completed).unwrap();
    let probability: Value =
        serde_json::from_str(probability.trim_start_matches("data: ")).unwrap();
    let text: Value = serde_json::from_str(text.trim_start_matches("data: ")).unwrap();
    assert!(probability["choices"][0]["logprobs"]["content"].is_array());
    assert!(text["choices"][0]["logprobs"].is_null());
    assert!(finish[0].contains("\"logprobs\":null"));
}

#[test]
fn aggregate_appends_channels_and_preserves_empty_and_null() {
    let events = vec![
        update(Some(serde_json::json!([])), None),
        update(
            Some(serde_json::json!([{"token":"b","logprob":0.0,"top_logprobs":[]}])),
            None,
        ),
        Event::ChoiceLogprobsDelta(ChoiceLogprobsDelta {
            choice_index: 0,
            logprobs: None,
        }),
    ];
    let value = crate::logprobs::aggregate(&events).unwrap();
    assert_eq!(value.content.unwrap().len(), 1);
}

#[test]
fn parser_drops_bad_bytes_and_rejects_missing_alternatives_and_nonzero_choice() {
    // Out-of-range bytes are optional metadata: dropped, never a failure.
    let bad_bytes = serde_json::json!({"content":[{"token":"x","logprob":0.0,"bytes":[256],"top_logprobs":[]}]});
    let parsed = crate::logprobs::parse(Some(&bad_bytes), 0)
        .unwrap()
        .unwrap();
    assert_eq!(parsed.logprobs.unwrap().content.unwrap()[0].bytes, None);
    let missing = serde_json::json!({"content":[{"token":"x","logprob":0.0}]});
    assert!(crate::logprobs::parse(Some(&missing), 0).is_err());
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    normalizer.enable_chat_logprobs(true);
    let frame = crate::sse::SseEvent {
        event: None,
        data: serde_json::json!({"choices":[{"index":1,"delta":{},"logprobs":null}]}).to_string(),
    };
    assert!(normalizer.feed(&frame).is_err());
}

#[test]
fn aggregate_chat_body_contains_all_probability_records() {
    let events = vec![
        update(
            Some(serde_json::json!([{"token":"a","logprob":-1.0,"top_logprobs":[]}])),
            None,
        ),
        Event::TextDelta("a".into()),
        Event::Completed,
    ];
    let result = completed_chat_body_with_ignored("r", "m", 1, &events, &[], false).unwrap();
    assert_eq!(
        result.body["choices"][0]["logprobs"]["content"][0]["token"],
        "a"
    );
}

#[test]
fn malformed_unrequested_probabilities_do_not_change_ordinary_chat() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let frame = crate::sse::SseEvent {
        event: None,
        data: r#"{"choices":[{"index":0,"delta":{"content":"ok"},"logprobs":"bad"}]}"#.to_string(),
    };
    assert!(matches!(&normalizer.feed(&frame).unwrap()[0], Event::TextDelta(text) if text == "ok"));
}

#[test]
fn retained_probability_budget_covers_escaped_json_and_empty_records() {
    let event = update(
        Some(serde_json::json!([
            {"token":"\u{0000}","logprob":-9999.0,"bytes":[0,255],"top_logprobs":[]},
            {"token":"","logprob":0.0,"bytes":null,"top_logprobs":[]}
        ])),
        None,
    );
    let Event::ChoiceLogprobsDelta(delta) = event else {
        unreachable!()
    };
    assert!(delta.retained_bytes() >= serde_json::to_vec(&delta).unwrap().len());
}

/// One Novita (vLLM/SGLang behind pydantic) Chat stream chunk: a sampled token
/// whose `logprob` and alternatives carry the unrepresentable values such a
/// server writes for an effectively impossible token. `null` is what pydantic
/// v2 serializes -inf and NaN to; the texts are what other relays print.
fn novita_chunk(token: &str, logprob: Value, alternatives: Value) -> crate::sse::SseEvent {
    crate::sse::SseEvent {
        event: None,
        data: serde_json::json!({
            "id": "chatcmpl-novita",
            "object": "chat.completion.chunk",
            "created": 1_761_134_400,
            "model": "deepseek/deepseek-v4.1-flash-1017",
            "choices": [{
                "index": 0,
                "delta": {"content": token},
                "logprobs": {"content": [{
                    "token": token,
                    "logprob": logprob,
                    "bytes": token.as_bytes(),
                    "top_logprobs": alternatives,
                }]},
                "finish_reason": null,
            }],
        })
        .to_string(),
    }
}

fn novita_stream() -> Vec<crate::sse::SseEvent> {
    vec![
        novita_chunk(
            "a",
            serde_json::json!(null),
            serde_json::json!([
                {"token": "a", "logprob": null, "bytes": [97]},
                {"token": "b", "logprob": "-inf", "bytes": [98]},
            ]),
        ),
        novita_chunk(
            "b",
            serde_json::json!("-Infinity"),
            serde_json::json!([
                {"token": "b", "logprob": "NaN", "bytes": [98]},
                {"token": "c", "bytes": [99]},
            ]),
        ),
        novita_chunk(
            "c",
            serde_json::json!(-1.0e5),
            serde_json::json!([{"token": "c", "logprob": "-0.25", "bytes": [99]}]),
        ),
        novita_chunk(
            "d",
            serde_json::json!(-0.5),
            serde_json::json!([{"token": "d", "logprob": -0.5, "bytes": [100]}]),
        ),
    ]
}

fn novita_events() -> Vec<Event> {
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    normalizer.enable_chat_logprobs(true);
    let mut events = Vec::new();
    for frame in novita_stream() {
        events.extend(
            normalizer
                .feed(&frame)
                .expect("a logprob value never fails"),
        );
    }
    events.push(Event::Completed);
    events
}

#[test]
fn novita_unrepresentable_logprob_values_map_to_the_openai_negligible_value() {
    let events = novita_events();
    let logprobs = crate::logprobs::aggregate(&events).unwrap();
    let content = logprobs.content.unwrap();
    let values: Vec<f64> = content.iter().map(|record| record.logprob).collect();
    // null and "-Infinity" become -9999.0; a finite value below it is floored;
    // a finite value is kept exactly.
    assert_eq!(values, vec![-9999.0, -9999.0, -9999.0, -0.5]);
    let alternatives: Vec<Vec<f64>> = content
        .iter()
        .map(|record| record.top_logprobs.iter().map(|c| c.logprob).collect())
        .collect();
    // Every alternative is kept (the count the caller asked for); null, "-inf",
    // "NaN" and an absent value become -9999.0, a numeric text is its number.
    assert_eq!(
        alternatives,
        vec![
            vec![-9999.0, -9999.0],
            vec![-9999.0, -9999.0],
            vec![-0.25],
            vec![-0.5],
        ]
    );
    assert_eq!(content[1].top_logprobs[1].token, "c");
}

#[test]
fn novita_coerced_logprobs_encode_as_numbers_streaming_and_not() {
    let events = novita_events();
    let mut encoder = ChatSseEncoder::new_with_ignored("r", "m", 1, false, vec![]);
    encoder.start().unwrap();
    let mut seen = 0;
    for event in &events {
        for chunk in encoder.feed(event).unwrap() {
            let Ok(chunk) =
                serde_json::from_str::<Value>(chunk.trim().trim_start_matches("data: "))
            else {
                continue;
            };
            let Some(records) = chunk["choices"][0]["logprobs"]["content"].as_array() else {
                continue;
            };
            for record in records {
                seen += 1;
                assert!(record["logprob"].is_f64());
                for alternative in record["top_logprobs"].as_array().unwrap() {
                    assert!(alternative["logprob"].is_f64());
                }
            }
        }
    }
    assert_eq!(seen, 4);
    let body = completed_chat_body_with_ignored("r", "m", 1, &events, &[], false)
        .unwrap()
        .body;
    let content = body["choices"][0]["logprobs"]["content"]
        .as_array()
        .unwrap();
    assert_eq!(content.len(), 4);
    assert_eq!(content[0]["logprob"], -9999.0);
    assert_eq!(content[0]["top_logprobs"][1]["logprob"], -9999.0);
    assert_eq!(content[3]["logprob"], -0.5);
}

#[test]
fn logprob_values_never_fail_but_broken_structure_still_does() {
    for value in [
        serde_json::json!(null),
        serde_json::json!("-inf"),
        serde_json::json!("-Infinity"),
        serde_json::json!("NaN"),
        serde_json::json!("inf"),
        serde_json::json!("not a number"),
        serde_json::json!(true),
        serde_json::json!({}),
    ] {
        let channel = serde_json::json!({"content":[{"token":"x","logprob":value,"top_logprobs":[{"token":"y","logprob":value}]}]});
        let parsed = crate::logprobs::parse(Some(&channel), 0).unwrap().unwrap();
        let record = &parsed.logprobs.unwrap().content.unwrap()[0];
        assert_eq!(
            record.logprob,
            crate::logprobs::NEGLIGIBLE_LOGPROB,
            "{value}"
        );
        assert_eq!(
            record.top_logprobs[0].logprob,
            crate::logprobs::NEGLIGIBLE_LOGPROB
        );
    }
    for broken in [
        serde_json::json!({"content":"x"}),
        serde_json::json!({"content":[7]}),
        serde_json::json!({"content":[{"token":7,"logprob":-1.0,"top_logprobs":[]}]}),
        serde_json::json!({"content":[{"token":"x","logprob":-1.0,"top_logprobs":"y"}]}),
        serde_json::json!({"content":[{"token":"x","logprob":-1.0,"top_logprobs":[{"logprob":-1.0}]}]}),
    ] {
        assert!(
            crate::logprobs::parse(Some(&broken), 0).is_err(),
            "{broken}"
        );
    }
}

#[test]
fn capture_only_probabilities_keep_coerced_values_without_a_hole() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    normalizer.enable_logprobs_capture();
    for frame in novita_stream() {
        normalizer.feed(&frame).unwrap();
    }
    let (deltas, truncated) = normalizer.take_captured_logprobs().unwrap();
    assert!(!truncated);
    assert_eq!(deltas.len(), 4);
}
