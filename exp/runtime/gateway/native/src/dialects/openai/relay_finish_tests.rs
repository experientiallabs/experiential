//! OpenAI-compatible RELAY finish behaviour: a tool call whose arguments end
//! mid-token under a normal `stop`/`tool_calls` finish is the provider's cut
//! misreported (ledger 2026-09-07) and settles Incomplete; corruption inside
//! the arguments, or a prefix that could never be the arguments object, keeps
//! the strict malformed contract.

use super::super::{Dialect, Normalizer};
use crate::events::Event;
use crate::sse::SseEvent;

fn compatible_chunk(delta: serde_json::Value, finish_reason: Option<&str>) -> SseEvent {
    SseEvent {
        event: None,
        data: serde_json::json!({
            "id": "chatcmpl-relay",
            "object": "chat.completion.chunk",
            "created": 1_788_425_855,
            "model": "deepseek-v4-flash",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        })
        .to_string(),
    }
}

#[test]
fn a_relay_stop_finish_with_arguments_cut_mid_fragment_settles_incomplete() {
    // OpenRouter / Tencent relays close deepseek tool calls with
    // finish_reason `tool_calls` while the arguments end mid-string (ledger
    // 2026-09-07). The cut call is dropped and the turn is Incomplete, never a
    // malformed 502 that churns the ladder.
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    normalizer
        .feed(&compatible_chunk(
            serde_json::json!({"tool_calls": [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "write_file", "arguments": "{\"path\": \"a.txt\", \"content\": \"partial te"},
            }]}),
            None,
        ))
        .expect("tool delta normalizes");
    normalizer
        .feed(&compatible_chunk(serde_json::json!({}), Some("tool_calls")))
        .expect("finish chunk normalizes");
    let events = normalizer
        .feed(&SseEvent {
            event: None,
            data: "[DONE]".to_string(),
        })
        .expect("a mid-fragment cut is truncation, not corruption");
    assert!(
        events
            .iter()
            .any(|event| matches!(event, Event::Incomplete)),
        "the turn settles Incomplete: {events:?}"
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, Event::ToolCallCompleted { .. })),
        "the cut call is never served: {events:?}"
    );
}

#[test]
fn a_relay_stop_finish_with_a_syntax_error_inside_arguments_stays_malformed() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    normalizer
        .feed(&compatible_chunk(
            serde_json::json!({"tool_calls": [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "write_file", "arguments": "{\"path\": }"},
            }]}),
            None,
        ))
        .expect("tool delta normalizes");
    normalizer
        .feed(&compatible_chunk(serde_json::json!({}), Some("tool_calls")))
        .expect("finish chunk normalizes");
    let failure = normalizer
        .feed(&SseEvent {
            event: None,
            data: "[DONE]".to_string(),
        })
        .expect_err("a syntax error inside a served answer stays fail-closed");
    assert!(failure.safe_message.contains("not valid JSON"));
}

#[test]
fn a_cut_non_object_prefix_stays_malformed_on_a_relay_finish() {
    // An unfinished ARRAY could never have become the arguments object, so its
    // early end is corruption, not a cut call.
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    normalizer
        .feed(&compatible_chunk(
            serde_json::json!({"tool_calls": [{
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "write_file", "arguments": "[1,2"},
            }]}),
            None,
        ))
        .expect("tool delta normalizes");
    normalizer
        .feed(&compatible_chunk(serde_json::json!({}), Some("tool_calls")))
        .expect("finish chunk normalizes");
    let failure = normalizer
        .feed(&SseEvent {
            event: None,
            data: "[DONE]".to_string(),
        })
        .expect_err("a non-object prefix is corruption");
    assert!(failure.safe_message.contains("not valid JSON"));
}

#[test]
fn a_dangling_fragment_on_a_normal_relay_finish_is_truncation_not_malformed() {
    // Until 2026-09-07 a `tool_calls`/`stop` terminal with a dangling fragment
    // was a malformed stream. Relays (OpenRouter, Tencent, the house vLLM
    // lanes) then showed 57 such attempts in 12h, fragments from 1 KB to
    // 87 KB, every one ending mid-token: a model never emits a well-formed
    // answer that stops mid-string, so the shape is the provider's cut
    // misreported, and the turn now settles Incomplete with the cut call
    // dropped. A syntax error INSIDE the arguments keeps the strict contract
    // (see the test below).
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
    let events = normalizer
        .feed(&SseEvent {
            event: None,
            data: "[DONE]".to_string(),
        })
        .expect("a dangling fragment on a normal relay finish is truncation");
    assert!(
        matches!(events.as_slice(), [Event::Incomplete]),
        "{events:?}"
    );
}
