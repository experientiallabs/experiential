//! Streamed `output_item.done` order must equal `response.completed.output`:
//! clients that record items as they complete (Codex) replay that order.

use super::*;

fn read_marker_call(index: u32, call_id: &str, arguments: &str) -> [Event; 3] {
    [
        Event::ToolCallStarted {
            custom: false,
            namespace: None,
            caller: None,
            index,
            call_id: call_id.to_string(),
            name: "read_marker".to_string(),
            thought_signature: None,
        },
        Event::ToolArgumentsDelta {
            index,
            delta: arguments.to_string(),
        },
        Event::ToolCallCompleted {
            index,
            call: CompletedToolCall {
                namespace: None,
                caller: None,
                call_id: call_id.to_string(),
                name: "read_marker".to_string(),
                raw_arguments: arguments.to_string(),
                provider_item_id: None,
                provider_status: None,
                custom: false,
                thought_signature: None,
            },
        },
    ]
}

/// Stream `events` and return (item ids in `output_item.done` order, item ids
/// in `response.completed.output` order).
fn done_and_completed_order(events: &[Event]) -> (Vec<Value>, Vec<Value>) {
    let mut encoder = ResponsesSseEncoder::new(
        "request-1",
        "deepseek",
        1_700_000_000,
        ResponsesEnvelope::default(),
    );
    let mut frames = encoder.start().expect("stream start must encode");
    for event in events {
        frames.extend(encoder.feed(event).expect("event must encode"));
    }
    let payloads: Vec<Value> = frames
        .iter()
        .filter_map(|frame| frame.split_once("data: ").map(|(_, data)| data.trim()))
        .map(|data| serde_json::from_str(data).expect("frame data must be JSON"))
        .collect();
    let identity = |item: &Value| item.get("call_id").unwrap_or(&item["id"]).clone();
    let done: Vec<Value> = payloads
        .iter()
        .filter(|payload| payload["type"] == json!("response.output_item.done"))
        .map(|payload| identity(&payload["item"]))
        .collect();
    let terminal = payloads.last().expect("response must terminate");
    assert_eq!(terminal["type"], json!("response.completed"));
    let completed = terminal["response"]["output"]
        .as_array()
        .expect("completed output is an array")
        .iter()
        .map(identity)
        .collect();
    (done, completed)
}

#[test]
fn a_progress_message_completes_before_the_calls_that_follow_it() {
    // A chat-wire turn: the model narrates, then calls two tools; the chat
    // dialect completes the calls at the provider's finish, while the
    // message is still open. The calls used to complete first, so Codex
    // replayed call, call, message, outputs: a history DeepSeek 400s.
    let mut events = vec![Event::TextDelta(
        "I will read the synthetic markers now.".to_string(),
    )];
    events.extend(read_marker_call(0, "call_fixture_1", "{\"index\":1}"));
    events.extend(read_marker_call(1, "call_fixture_2", "{\"index\":2}"));
    events.push(Event::Completed);
    let (done, completed) = done_and_completed_order(&events);
    assert_eq!(done, completed);
    assert_eq!(
        completed,
        vec![
            json!(stable_public_id("msg", "request-1")),
            json!("call_fixture_1"),
            json!("call_fixture_2"),
        ]
    );
}

#[test]
fn thinking_and_text_complete_before_the_calls_that_follow_them() {
    let mut events = vec![
        Event::ThinkingDelta {
            index: 0,
            delta: "plan".to_string(),
        },
        Event::TextDelta("Reading.".to_string()),
    ];
    events.extend(read_marker_call(1, "call_fixture_1", "{\"index\":1}"));
    events.push(Event::Completed);
    let (done, completed) = done_and_completed_order(&events);
    assert_eq!(done, completed);
    assert_eq!(completed.len(), 3);
}

#[test]
fn a_call_with_nothing_open_before_it_still_completes_at_once() {
    let mut encoder = ResponsesSseEncoder::new(
        "request-1",
        "deepseek",
        1_700_000_000,
        ResponsesEnvelope::default(),
    );
    encoder.start().expect("stream start must encode");
    let [started, arguments, completed] = read_marker_call(0, "call_fixture_1", "{\"index\":1}");
    encoder.feed(&started).expect("start encodes");
    encoder.feed(&arguments).expect("arguments encode");
    let frames = encoder.feed(&completed).expect("completion encodes");
    assert!(frames
        .iter()
        .any(|frame| frame.contains("response.output_item.done")));
}

#[test]
fn a_deferred_call_refuses_a_second_completion() {
    let mut encoder = ResponsesSseEncoder::new(
        "request-1",
        "deepseek",
        1_700_000_000,
        ResponsesEnvelope::default(),
    );
    encoder.start().expect("stream start must encode");
    encoder
        .feed(&Event::TextDelta("Reading.".to_string()))
        .expect("text encodes");
    let [started, arguments, completed] = read_marker_call(0, "call_fixture_1", "{\"index\":1}");
    encoder.feed(&started).expect("start encodes");
    encoder.feed(&arguments).expect("arguments encode");
    assert!(encoder
        .feed(&completed)
        .expect("completion defers")
        .is_empty());
    assert!(encoder.feed(&completed).is_err());
}
