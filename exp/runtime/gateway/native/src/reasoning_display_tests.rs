//! Reasoning display across the Chat, Messages, and Responses encoders.

use serde_json::{json, Value};

use super::*;
use crate::encode::{completed_chat_body_with_carrier, ChatSseEncoder};
use crate::encode_messages::{completed_messages_body_with_reasoning, MessagesSseEncoder};
use crate::encode_responses::{ResponsesEnvelope, ResponsesSseEncoder};

const DISPLAYED: ReasoningOutput = ReasoningOutput {
    exposed: false,
    displayed: true,
};

/// One turn whose reasoning arrives in every readable provider form.
fn every_form() -> Vec<Event> {
    vec![
        Event::ReasoningTextDelta("plain ".to_string()),
        Event::ReasoningTextDelta("thought".to_string()),
        Event::ReasoningSummaryDelta {
            output_index: 0,
            summary_index: 0,
            item_id: "rs_1".to_string(),
            delta: "summary".to_string(),
        },
        Event::ThinkingDelta {
            index: 1,
            delta: "thinking".to_string(),
        },
        Event::TextDelta("answer".to_string()),
        Event::Completed,
    ]
}

/// The JSON payloads of a stream's `data:` frames.
fn payloads(frames: &[String]) -> Vec<Value> {
    frames
        .iter()
        .flat_map(|frame| frame.lines())
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str(data).ok())
        .collect()
}

fn chat_stream(events: &[Event], output: ReasoningOutput) -> Vec<Value> {
    let mut encoder = ChatSseEncoder::new_with_ignored("request-1", "m", 1, false, Vec::new());
    encoder.set_reasoning_output(output);
    let mut frames = encoder.start().unwrap();
    for event in events {
        frames.extend(encoder.feed(event).unwrap());
    }
    payloads(&frames)
}

#[test]
fn chat_streams_every_readable_form_as_reasoning_with_unit_breaks() {
    let reasoning: String = chat_stream(&every_form(), DISPLAYED)
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["reasoning"].as_str())
        .collect();
    assert_eq!(reasoning, "plain thought\n\nsummary\n\nthinking");
}

#[test]
fn chat_display_off_returns_no_reasoning_field() {
    let chunks = chat_stream(&every_form(), ReasoningOutput::default());
    assert!(chunks
        .iter()
        .all(|chunk| chunk["choices"][0]["delta"].get("reasoning").is_none()));
}

#[test]
fn chat_display_never_replaces_exposed_reasoning_content() {
    let events = vec![
        Event::ReasoningContentDelta {
            route_sha256: "d".repeat(64),
            delta: "route thought".to_string(),
        },
        Event::TextDelta("answer".to_string()),
        Event::Completed,
    ];
    let both = ReasoningOutput {
        exposed: true,
        displayed: true,
    };
    let chunks = chat_stream(&events, both);
    let content: Vec<_> = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["reasoning_content"].as_str())
        .collect();
    assert_eq!(content, vec!["route thought"]);
    assert!(chunks
        .iter()
        .all(|chunk| chunk["choices"][0]["delta"].get("reasoning").is_none()));
}

#[test]
fn chat_completed_body_carries_flattened_reasoning() {
    let body =
        completed_chat_body_with_carrier("request-1", "m", 1, &every_form(), &[], None, DISPLAYED)
            .unwrap()
            .body;
    let message = &body["choices"][0]["message"];
    assert_eq!(
        message["reasoning"],
        json!("plain thought\n\nsummary\n\nthinking")
    );
    assert_eq!(message["content"], json!("answer"));
    assert!(message.get("reasoning_content").is_none());
    let hidden =
        completed_chat_body_with_carrier("request-1", "m", 1, &every_form(), &[], None, false)
            .unwrap()
            .body;
    assert!(hidden["choices"][0]["message"].get("reasoning").is_none());
}

#[test]
fn messages_stream_reasoning_that_resumes_after_text_in_a_fresh_block() {
    let events = vec![
        Event::ReasoningTextDelta("first".to_string()),
        Event::TextDelta("answer".to_string()),
        Event::ReasoningTextDelta("later".to_string()),
        Event::Completed,
    ];
    let mut encoder = MessagesSseEncoder::new_with_ignored("request-1", "m", Vec::new());
    encoder.set_reasoning_output(DISPLAYED);
    let mut frames = encoder.start().unwrap();
    for event in &events {
        frames.extend(encoder.feed(event).unwrap());
    }
    let chunks = payloads(&frames);
    let thinking: Vec<&str> = chunks
        .iter()
        .filter_map(|chunk| chunk["delta"]["thinking"].as_str())
        .collect();
    assert_eq!(thinking, vec!["first", "later"]);
    let opened = chunks
        .iter()
        .filter(|chunk| chunk["content_block"]["type"] == json!("thinking"))
        .count();
    assert_eq!(opened, 2);
}

#[test]
fn messages_render_non_anthropic_reasoning_as_one_unsigned_thinking_block() {
    let events = vec![
        Event::ReasoningTextDelta("plain".to_string()),
        Event::ReasoningSummaryDelta {
            output_index: 0,
            summary_index: 0,
            item_id: "rs_1".to_string(),
            delta: " summary".to_string(),
        },
        Event::TextDelta("answer".to_string()),
        Event::Completed,
    ];
    let mut encoder = MessagesSseEncoder::new_with_ignored("request-1", "m", Vec::new());
    encoder.set_reasoning_output(DISPLAYED);
    let mut frames = encoder.start().unwrap();
    for event in &events {
        frames.extend(encoder.feed(event).unwrap());
    }
    let chunks = payloads(&frames);
    let thinking: String = chunks
        .iter()
        .filter_map(|chunk| chunk["delta"]["thinking"].as_str())
        .collect();
    // Provider units stay separate paragraphs, exactly as on Chat.
    assert_eq!(thinking, "plain\n\n summary");
    assert!(chunks.iter().any(|chunk| chunk["content_block"]
        == json!({"type": "thinking", "thinking": "", "signature": ""})));

    let body =
        completed_messages_body_with_reasoning("request-1", "m", &events, &[], None, DISPLAYED)
            .unwrap()
            .body;
    assert_eq!(
        body["content"][0],
        json!({"type": "thinking", "thinking": "plain\n\n summary", "signature": ""})
    );
    let hidden =
        completed_messages_body_with_reasoning("request-1", "m", &events, &[], None, false)
            .unwrap()
            .body;
    assert_eq!(hidden["content"][0]["type"], json!("text"));
}

#[test]
fn responses_stream_display_text_as_a_carrier_free_reasoning_item() {
    let events = vec![
        Event::ReasoningTextDelta("plain thought".to_string()),
        Event::TextDelta("answer".to_string()),
        Event::Completed,
    ];
    for displayed in [true, false] {
        let envelope = ResponsesEnvelope {
            reasoning_displayed: displayed,
            ..ResponsesEnvelope::default()
        };
        let mut encoder = ResponsesSseEncoder::new("request-1", "m", 1, envelope);
        let mut frames = encoder.start().unwrap();
        for event in &events {
            frames.extend(encoder.feed(event).unwrap());
        }
        let chunks = payloads(&frames);
        let summary: String = chunks
            .iter()
            .filter(|chunk| chunk["type"] == "response.reasoning_summary_text.delta")
            .filter_map(|chunk| chunk["delta"].as_str())
            .collect();
        let completed = chunks
            .iter()
            .find(|chunk| chunk["type"] == "response.completed")
            .unwrap();
        let output = completed["response"]["output"].as_array().unwrap();
        if displayed {
            assert_eq!(summary, "plain thought");
            assert_eq!(output[0]["type"], json!("reasoning"));
            assert_eq!(
                output[0]["summary"],
                json!([{"type": "summary_text", "text": "plain thought"}])
            );
            assert!(output[0].get("encrypted_content").is_none());
        } else {
            assert!(summary.is_empty());
            assert!(output.iter().all(|item| item["type"] != "reasoning"));
        }
    }
}

#[test]
fn messages_text_only_counts_readable_forms_the_rung_shows() {
    let route = Event::ReasoningContentDelta {
        route_sha256: "d".repeat(64),
        delta: "route".to_string(),
    };
    let text = Event::ReasoningTextDelta("text".to_string());
    assert_eq!(unsigned_thinking_text(&route, false.into()), None);
    assert_eq!(unsigned_thinking_text(&route, true.into()), Some("route"));
    assert_eq!(unsigned_thinking_text(&route, DISPLAYED), Some("route"));
    assert_eq!(unsigned_thinking_text(&text, true.into()), None);
    assert_eq!(unsigned_thinking_text(&text, DISPLAYED), Some("text"));
    let thinking = Event::ThinkingDelta {
        index: 0,
        delta: "signed".to_string(),
    };
    assert_eq!(unsigned_thinking_text(&thinking, DISPLAYED), None);
}

#[test]
fn messages_aggregate_keeps_interleaved_reasoning_in_provider_order() {
    let events = vec![
        Event::ReasoningTextDelta("first".to_string()),
        Event::TextDelta("partial ".to_string()),
        Event::ReasoningTextDelta("second".to_string()),
        Event::TextDelta("answer".to_string()),
        Event::Completed,
    ];
    let body =
        completed_messages_body_with_reasoning("request-1", "m", &events, &[], None, DISPLAYED)
            .unwrap()
            .body;
    assert_eq!(
        body["content"],
        json!([
            {"type": "thinking", "thinking": "first", "signature": ""},
            {"type": "text", "text": "partial "},
            {"type": "thinking", "thinking": "second", "signature": ""},
            {"type": "text", "text": "answer"},
        ])
    );
}

#[test]
fn responses_withheld_rung_drops_provider_summaries_and_thinking() {
    for withheld in [false, true] {
        let envelope = ResponsesEnvelope {
            reasoning_withheld: withheld,
            ..ResponsesEnvelope::default()
        };
        let mut encoder = ResponsesSseEncoder::new("request-1", "m", 1, envelope);
        let mut frames = encoder.start().unwrap();
        for event in &every_form() {
            frames.extend(encoder.feed(event).unwrap());
        }
        let deltas: Vec<String> = payloads(&frames)
            .iter()
            .filter(|chunk| chunk["type"] == "response.reasoning_summary_text.delta")
            .filter_map(|chunk| chunk["delta"].as_str().map(str::to_string))
            .collect();
        let expected: Vec<String> = if withheld {
            Vec::new()
        } else {
            vec!["summary".to_string(), "thinking".to_string()]
        };
        assert_eq!(deltas, expected);
    }
}

#[test]
fn resumed_display_reasoning_opens_fresh_items_in_provider_order() {
    let events = vec![
        Event::ReasoningTextDelta("first".to_string()),
        Event::TextDelta("partial ".to_string()),
        Event::ReasoningSummaryDelta {
            output_index: 0,
            summary_index: 0,
            item_id: "rs_1".to_string(),
            delta: "second".to_string(),
        },
        Event::TextDelta("answer".to_string()),
        Event::Completed,
    ];
    // Messages: the resumed block starts without a synthetic paragraph break.
    let mut encoder = MessagesSseEncoder::new_with_ignored("request-1", "m", Vec::new());
    encoder.set_reasoning_output(DISPLAYED);
    let mut frames = encoder.start().unwrap();
    for event in &events {
        frames.extend(encoder.feed(event).unwrap());
    }
    let thinking: Vec<String> = payloads(&frames)
        .iter()
        .filter_map(|chunk| chunk["delta"]["thinking"].as_str().map(str::to_string))
        .collect();
    assert_eq!(thinking, vec!["first".to_string(), "second".to_string()]);

    // Responses: display text that resumes after the message opens a new item.
    let text_events = vec![
        Event::ReasoningTextDelta("first".to_string()),
        Event::TextDelta("partial ".to_string()),
        Event::ReasoningTextDelta("second".to_string()),
        Event::TextDelta("answer".to_string()),
        Event::Completed,
    ];
    let envelope = ResponsesEnvelope {
        reasoning_displayed: true,
        ..ResponsesEnvelope::default()
    };
    let mut encoder = ResponsesSseEncoder::new("request-1", "m", 1, envelope);
    let mut frames = encoder.start().unwrap();
    for event in &text_events {
        frames.extend(encoder.feed(event).unwrap());
    }
    let chunks = payloads(&frames);
    let completed = chunks
        .iter()
        .find(|chunk| chunk["type"] == "response.completed")
        .unwrap();
    let kinds: Vec<&str> = completed["response"]["output"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["type"].as_str().unwrap())
        .collect();
    // Answer text shares one message item; the resumed reasoning is a new
    // item after it instead of being appended to the first.
    assert_eq!(kinds, vec!["reasoning", "message", "reasoning"]);
}
