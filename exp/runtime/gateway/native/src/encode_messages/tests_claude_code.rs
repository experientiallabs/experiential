//! Inline tests for `encode_messages` that pin the Claude Code / Harbor
//! contracts (2026-09-11): carrier sealing on stop-sequence terminals, the
//! upstream start-frame meters on `message_start`, and mixed
//! reasoning+content+tool replies on the aggregate body. A second submodule
//! file so each stays within the repository line budget.

use super::tests::{frame_names, tool_turn_reasoning_events};
use super::*;

#[test]
fn a_stop_sequence_ending_a_reasoning_tool_turn_still_needs_and_emits_the_carrier() {
    // Both completing terminals demand the sealed carrier, so the route must
    // seal on `StoppedAtSequence` too (Greptile on #897): with the carrier the
    // trailing redacted block and the terminal frames flow; without it the
    // encoder fails closed instead of ending the stream short.
    let mut events = tool_turn_reasoning_events();
    events.pop();
    events.push(Event::StoppedAtSequence("STOP".to_string()));
    let carrier = "x-experiential-hunyuan-reasoning-v1:ZGVw:c2VhbGVk";
    let mut encoder = MessagesSseEncoder::new("request-abc", "coding");
    encoder.set_reasoning_content_carrier(carrier.to_string());
    let mut frames = encoder.start().expect("starts");
    for event in &events {
        frames.extend(encoder.feed(event).expect("streams"));
    }
    let names = frame_names(&frames);
    assert_eq!(names[names.len() - 2..], ["message_delta", "message_stop"]);
    assert!(frames
        .iter()
        .any(|frame| frame.contains("\"redacted_thinking\"")));

    let mut unsealed = MessagesSseEncoder::new("request-abc", "coding");
    unsealed.start().expect("starts");
    let error = events
        .iter()
        .find_map(|event| unsealed.feed(event).err())
        .expect("terminal without a carrier fails");
    assert!(error.message.contains("not sealed"));
}

#[test]
fn start_frame_carries_the_upstream_start_usage_when_known() {
    // An Anthropic upstream reports its input and cache meters on its own
    // message_start; the gateway's message_start mirrors them (input excludes
    // cached reads, as on the terminal frame) instead of the zero placeholder.
    // An OpenAI-wire upstream reports nothing before its final chunk, so the
    // placeholder stays and the final meters ride message_delta.
    let mut encoder = MessagesSseEncoder::new("request-abc", "coding");
    encoder.set_initial_usage(Some(Usage {
        input_tokens: Some(2230),
        output_tokens: Some(25),
        cached_input_tokens: Some(2000),
        cache_creation_input_tokens: None,
        reasoning_tokens: None,
    }));
    let frames = encoder.start().expect("starts");
    let start: Value = serde_json::from_str(
        frames[0]
            .lines()
            .nth(1)
            .and_then(|line| line.strip_prefix("data: "))
            .expect("data line"),
    )
    .expect("json");
    assert_eq!(
        start["message"]["usage"],
        json!({"input_tokens": 230, "output_tokens": 25, "cache_read_input_tokens": 2000})
    );

    let mut bare = MessagesSseEncoder::new("request-abc", "coding");
    bare.set_initial_usage(None);
    let frames = bare.start().expect("starts");
    let start: Value = serde_json::from_str(
        frames[0]
            .lines()
            .nth(1)
            .and_then(|line| line.strip_prefix("data: "))
            .expect("data line"),
    )
    .expect("json");
    assert_eq!(
        start["message"]["usage"],
        json!({"input_tokens": 0, "output_tokens": 0})
    );
}

#[test]
fn a_reply_with_reasoning_text_and_a_tool_call_renders_every_part_on_the_aggregate_body() {
    // Production 2026-09-11 (request-4e333079…, hy4-preview, non-streaming,
    // tools present, budget 32000): the body carried ONE thinking block and
    // stop_reason end_turn. If the model had also produced text or a tool
    // call, the aggregate path must render them beside the thinking block;
    // a thinking-only body therefore reflects the model's own output.
    let mut events = vec![
        Event::ReasoningContentDelta {
            route_sha256: "route-a".to_string(),
            delta: "plan first".to_string(),
        },
        Event::TextDelta("Running the lookup.".to_string()),
    ];
    events.extend(tool_turn_reasoning_events().into_iter().skip(1));
    let aggregated = completed_messages_body_with_reasoning(
        "request-abc",
        "coding",
        &events,
        &[],
        Some("x-experiential-hunyuan-reasoning-v1:sealed"),
        true,
    )
    .expect("aggregates");
    let content = aggregated.body["content"]
        .as_array()
        .expect("content array");
    let types: Vec<&str> = content
        .iter()
        .map(|block| block["type"].as_str().expect("typed block"))
        .collect();
    assert!(types.contains(&"thinking"), "{types:?}");
    assert!(types.contains(&"text"), "{types:?}");
    assert!(types.contains(&"tool_use"), "{types:?}");
    assert_eq!(aggregated.body["stop_reason"], json!("tool_use"));
    let text = content
        .iter()
        .find(|block| block["type"] == json!("text"))
        .expect("text block");
    assert_eq!(text["text"], json!("Running the lookup."));
    let tool = content
        .iter()
        .find(|block| block["type"] == json!("tool_use"))
        .expect("tool block");
    assert_eq!(tool["name"], json!("lookup"));
}

#[test]
fn start_frame_carries_the_pre_dispatch_estimate_when_the_upstream_reports_nothing() {
    // An OpenAI-wire upstream reports usage only in its final chunk, so the
    // start frame would otherwise carry the zero placeholder that Claude Code
    // reads as "0 input tokens". The admission's pre-dispatch prompt estimate
    // fills it with Anthropic's `output_tokens: 1` placeholder; the estimate
    // is display-only (message_delta and the ledger keep the provider's
    // report).
    let mut encoder = MessagesSseEncoder::new("request-abc", "coding");
    encoder.set_initial_usage(None);
    encoder.set_pre_dispatch_input_estimate(Some(1234));
    let frames = encoder.start().expect("starts");
    let start: Value = serde_json::from_str(
        frames[0]
            .lines()
            .nth(1)
            .and_then(|line| line.strip_prefix("data: "))
            .expect("data line"),
    )
    .expect("json");
    assert_eq!(
        start["message"]["usage"],
        json!({"input_tokens": 1234, "output_tokens": 1})
    );
}

#[test]
fn upstream_start_usage_outranks_the_pre_dispatch_estimate() {
    // An Anthropic upstream's own start meters are the truth; the estimate
    // never overrides a reported count, whichever order the route sets them.
    let mut encoder = MessagesSseEncoder::new("request-abc", "coding");
    encoder.set_pre_dispatch_input_estimate(Some(1234));
    encoder.set_initial_usage(Some(Usage {
        input_tokens: Some(30),
        output_tokens: Some(1),
        cached_input_tokens: Some(10),
        cache_creation_input_tokens: None,
        reasoning_tokens: None,
    }));
    let frames = encoder.start().expect("starts");
    let start: Value = serde_json::from_str(
        frames[0]
            .lines()
            .nth(1)
            .and_then(|line| line.strip_prefix("data: "))
            .expect("data line"),
    )
    .expect("json");
    assert_eq!(
        start["message"]["usage"],
        json!({"input_tokens": 20, "output_tokens": 1, "cache_read_input_tokens": 10})
    );
}
