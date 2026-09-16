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
fn parser_rejects_bad_bytes_missing_alternatives_and_nonzero_choice() {
    let bad_bytes = serde_json::json!({"content":[{"token":"x","logprob":0.0,"bytes":[256],"top_logprobs":[]}]});
    assert!(crate::logprobs::parse(Some(&bad_bytes), 0).is_err());
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
