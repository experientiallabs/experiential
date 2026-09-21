//! Lenient decoding of provider tool-call streams that used to fail closed
//! (split from `normalizer_tests.rs` for the module line budget): relays that
//! stream no tool id, nameless placeholder entries, metadata frames without
//! `choices`, a reasoning summary whose `done` text differs, and a stream
//! that closes after output without its terminal frame.

use super::super::{drain_stream_fixture, Dialect, Normalizer};
use crate::events::Event;
use crate::sse::SseEvent;

fn compatible_chunk(delta: serde_json::Value, finish_reason: Option<&str>) -> SseEvent {
    SseEvent {
        event: None,
        data: serde_json::json!({
            "id": "chatcmpl-relay",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        })
        .to_string(),
    }
}

fn done() -> SseEvent {
    SseEvent {
        event: None,
        data: "[DONE]".to_string(),
    }
}

fn raw_chunk(payload: serde_json::Value) -> Vec<u8> {
    format!("data: {payload}\n\n").into_bytes()
}

#[test]
fn a_null_tool_call_id_is_minted_by_the_gateway_and_a_late_id_is_ignored() {
    // Z.ai GLM relayed by Fireworks/OpenRouter opens the call with `"id":
    // null` (glm-4.6v-flash, glm-5.3-flash: 10 attempts 2026-09-05..09, every
    // one a 502 "tool ID must be text"). The caller pairs its tool result by
    // id and nothing else, so the gateway mints one; a real id restated later
    // never re-identifies the call the caller already holds.
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let started = normalizer
        .feed(&compatible_chunk(
            serde_json::json!({"tool_calls": [{
                "index": 0, "id": null, "type": "function",
                "function": {"name": "lookup", "arguments": "{\"q\":"},
            }]}),
            None,
        ))
        .expect("a null id is minted, not malformed");
    let minted = match started.as_slice() {
        [Event::ToolCallStarted { call_id, name, .. }, Event::ToolArgumentsDelta { .. }] => {
            assert_eq!(name, "lookup");
            assert!(call_id.starts_with("call_gw0_"), "{call_id}");
            call_id.clone()
        }
        other => panic!("unexpected events: {other:?}"),
    };
    normalizer
        .feed(&compatible_chunk(
            serde_json::json!({"tool_calls": [{
                "index": 0, "id": "call_late_real", "type": "function",
                "function": {"arguments": "\"x\"}"},
            }]}),
            Some("tool_calls"),
        ))
        .expect("a late real id on a minted call is ignored");
    let events = normalizer.feed(&done()).expect("stream completes");
    assert!(matches!(
        events.as_slice(),
        [Event::ToolCallCompleted { call, .. }, Event::Completed]
            if call.call_id == minted && call.raw_arguments == "{\"q\":\"x\"}"
    ));
}

#[test]
fn minted_ids_never_repeat_within_a_process() {
    // Two streams decoded in the same clock tick must not share an id: the
    // client pairs its tool result by id alone.
    let mint = || {
        let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
        let events = normalizer
            .feed(&compatible_chunk(
                serde_json::json!({"tool_calls": [{
                    "index": 0, "id": null, "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }]}),
                None,
            ))
            .expect("a null id is minted");
        match events.as_slice() {
            [Event::ToolCallStarted { call_id, .. }, ..] => call_id.clone(),
            other => panic!("unexpected events: {other:?}"),
        }
    };
    let ids: std::collections::BTreeSet<String> = (0..64).map(|_| mint()).collect();
    assert_eq!(ids.len(), 64, "{ids:?}");
    assert!(ids.iter().all(|id| id.len() <= 64), "replay bound: {ids:?}");
}

#[test]
fn an_empty_tool_call_id_is_minted_too() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let started = normalizer
        .feed(&compatible_chunk(
            serde_json::json!({"tool_calls": [{
                "index": 2, "id": "", "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }]}),
            None,
        ))
        .expect("an empty id is minted");
    assert!(matches!(
        started.as_slice(),
        [Event::ToolCallStarted { call_id, .. }, Event::ToolArgumentsDelta { .. }]
            if call_id.starts_with("call_gw2_")
    ));
}

#[test]
fn a_non_text_tool_call_id_stays_malformed() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let failure = normalizer
        .feed(&compatible_chunk(
            serde_json::json!({"tool_calls": [{
                "index": 0, "id": 7, "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }]}),
            None,
        ))
        .expect_err("a numeric id is the malformed shape it always was");
    assert!(failure.safe_message.contains("tool ID must be text"));
}

#[test]
fn a_nameless_argument_free_placeholder_is_dropped_not_failed() {
    // OpenRouter's GLM and Hunyuan relays emit an empty tool-call entry
    // (`id: "", name: "", arguments: ""`) beside a text answer (10 Messages
    // attempts 2026-09-12..15, every one "streamed tool call is incomplete
    // (2 bytes)"). The entry names no call, so nothing is started for it and
    // the turn completes on its text.
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let text = normalizer
        .feed(&compatible_chunk(
            serde_json::json!({"content": "Done."}),
            None,
        ))
        .expect("text normalizes");
    assert!(matches!(text.as_slice(), [Event::TextDelta(text)] if text == "Done."));
    let phantom = normalizer
        .feed(&compatible_chunk(
            serde_json::json!({"tool_calls": [{
                "index": 0, "id": "", "type": "function",
                "function": {"name": "", "arguments": ""},
            }]}),
            Some("stop"),
        ))
        .expect("a nameless placeholder is not a malformed stream");
    assert!(phantom.is_empty(), "nothing is started for it: {phantom:?}");
    let events = normalizer.feed(&done()).expect("stream completes");
    assert!(
        matches!(events.as_slice(), [Event::Completed]),
        "{events:?}"
    );
}

#[test]
fn a_nameless_entry_that_argues_still_fails_closed() {
    // A name cannot be invented: arguments with no call to attach them to
    // keep the strict contract.
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    normalizer
        .feed(&compatible_chunk(
            serde_json::json!({"tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": "", "arguments": "{\"q\":1}"},
            }]}),
            Some("tool_calls"),
        ))
        .expect("the entry accumulates silently");
    let failure = normalizer
        .feed(&done())
        .expect_err("arguments without a name cannot become a call");
    assert!(failure
        .safe_message
        .contains("streamed tool call is incomplete"));
}

#[test]
fn a_late_name_starts_the_call_with_its_buffered_arguments() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let silent = normalizer
        .feed(&compatible_chunk(
            serde_json::json!({"tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": null, "arguments": "{\"q\":"},
            }]}),
            None,
        ))
        .expect("a nameless start accumulates");
    assert!(silent.is_empty(), "{silent:?}");
    let named = normalizer
        .feed(&compatible_chunk(
            serde_json::json!({"tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": "lookup", "arguments": "1}"},
            }]}),
            None,
        ))
        .expect("the late name starts the call");
    assert!(
        matches!(
            named.as_slice(),
            [
                Event::ToolCallStarted { call_id, name, .. },
                Event::ToolArgumentsDelta { delta: buffered, .. },
                Event::ToolArgumentsDelta { delta: fresh, .. },
            ] if call_id == "call_1" && name == "lookup" && buffered == "{\"q\":" && fresh == "1}"
        ),
        "{named:?}"
    );
}

#[test]
fn a_frame_without_choices_is_metadata_only() {
    // Novita closes some streams with a usage-only chunk carrying no
    // `choices` key at all (19 attempts 2026-09-14..15, each a 502
    // "choices must be an array"). Its usage is taken; nothing else is read.
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    normalizer
        .feed(&compatible_chunk(
            serde_json::json!({"content": "Hi"}),
            Some("stop"),
        ))
        .expect("text normalizes");
    let usage_only = SseEvent {
        event: None,
        data: serde_json::json!({
            "id": "chatcmpl-relay", "object": "chat.completion.chunk",
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        })
        .to_string(),
    };
    assert!(normalizer
        .feed(&usage_only)
        .expect("a choices-less frame is metadata")
        .is_empty());
    let events = normalizer.feed(&done()).expect("stream completes");
    assert!(matches!(
        events.as_slice(),
        [Event::Usage(usage), Event::Completed] if usage.output_tokens == Some(1)
    ));
    // A non-array `choices`, and a choices-less frame carrying anything but
    // chunk metadata, keep the strict contract.
    for bad in [
        serde_json::json!({"choices": {"index": 0}}),
        serde_json::json!({"id": "x", "delta": {"content": "smuggled"}}),
        serde_json::json!({"id": "x", "choices": null, "message": {"content": "flat"}}),
        serde_json::json!({"id": "x", "choices": null, "finish_reason": "stop"}),
    ] {
        let mut strict = Normalizer::new(Dialect::OpenAiCompatible);
        let frame = SseEvent {
            event: None,
            data: bad.to_string(),
        };
        assert!(strict
            .feed(&frame)
            .expect_err("stays malformed")
            .safe_message
            .contains("choices must be an array"));
    }
}

#[test]
fn novita_trailing_sla_metrics_frame_with_null_choices_is_metadata() {
    // The exact key set from the production operator line (novita
    // deepseek-v4.1-flash, 48 attempts 2026-09-15): a trailing chunk after the
    // finish with `choices: null`, no usage, and the relay's own
    // `sla_metrics`. It carries nothing a decoder needs, so the stream still
    // settles by its finish and its usage; a fixed metadata allowlist that
    // did not know `sla_metrics` (or `choices` itself) failed the whole
    // completed answer as malformed.
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    normalizer
        .feed(&compatible_chunk(
            serde_json::json!({"content": "Hi"}),
            None,
        ))
        .expect("text normalizes");
    let finish = SseEvent {
        event: None,
        data: serde_json::json!({
            "id": "chatcmpl-novita", "object": "chat.completion.chunk", "created": 1_789_000_000,
            "model": "deepseek/deepseek-v4.1-flash",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        })
        .to_string(),
    };
    assert!(normalizer
        .feed(&finish)
        .expect("finish normalizes")
        .is_empty());
    let trailing = SseEvent {
        event: None,
        data: serde_json::json!({
            "id": "chatcmpl-novita", "object": "chat.completion.chunk", "created": 1_789_000_000,
            "model": "deepseek/deepseek-v4.1-flash", "system_fingerprint": null,
            "choices": null,
            "sla_metrics": {"ttft_ms": 412, "tpot_ms": 9, "tokens_per_second": 108.3},
        })
        .to_string(),
    };
    assert!(normalizer
        .feed(&trailing)
        .expect("a choices-less frame with only relay metadata is not malformed")
        .is_empty());
    let events = normalizer.feed(&done()).expect("stream completes");
    assert!(matches!(
        events.as_slice(),
        [Event::Usage(usage), Event::Completed] if usage.output_tokens == Some(3)
    ));
}

#[test]
fn a_compatible_stream_that_drops_done_after_its_finish_settles_by_that_finish() {
    // The finish frame arrived and only `[DONE]` is missing: the declared
    // finish stands, exactly as if the sentinel had been sent.
    let chunks = vec![
        raw_chunk(serde_json::json!({"choices": [{"index": 0, "delta": {"content": "Hi"}}]})),
        raw_chunk(serde_json::json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        })),
    ];
    let (events, failure) = drain_stream_fixture(Dialect::OpenAiCompatible, &chunks);
    assert!(failure.is_none(), "{failure:?}");
    assert_eq!(
        events.last().map(|event| event["kind"].clone()),
        Some("completed".into())
    );
}

#[test]
fn a_compatible_stream_that_closes_mid_output_settles_incomplete_with_cut_calls_dropped() {
    // No finish frame at all: the provider closed the stream on its output,
    // which is a cut, so the turn settles Incomplete; a call left open
    // mid-fragment is dropped like every other cut call.
    let chunks = vec![
        raw_chunk(serde_json::json!({"choices": [{"index": 0, "delta": {"content": "Hi"}}]})),
        raw_chunk(
            serde_json::json!({"choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": "lookup", "arguments": "{\"q\":\"Par"},
            }]}}]}),
        ),
    ];
    let (events, failure) = drain_stream_fixture(Dialect::OpenAiCompatible, &chunks);
    assert!(failure.is_none(), "{failure:?}");
    assert!(!events
        .iter()
        .any(|event| event["kind"] == "tool_call_completed"));
    assert_eq!(
        events.last().map(|event| event["kind"].clone()),
        Some("incomplete".into())
    );
}

#[test]
fn a_stream_that_closes_before_any_output_stays_terminal_less() {
    // Nothing was served, so nothing is preserved: the relay's
    // `ended_without_terminal` (failover-eligible) still applies.
    let chunks = vec![raw_chunk(
        serde_json::json!({"choices": [{"index": 0, "delta": {"role": "assistant"}}]}),
    )];
    let (events, failure) = drain_stream_fixture(Dialect::OpenAiCompatible, &chunks);
    assert!(events.is_empty());
    assert!(failure
        .expect("terminal-less")
        .safe_message
        .contains("without a terminal event"));
}

#[test]
fn a_responses_stream_that_closes_after_output_closes_its_items_and_settles_incomplete() {
    // gpt-5.6-luna on OpenAI's own Responses wire: 33 of 34 terminal-less
    // streams in 14 days had already served output (2026-09-14).
    let chunks = vec![
        raw_chunk(serde_json::json!({
            "type": "response.output_item.added", "output_index": 0,
            "item": {"id": "msg_1", "type": "message", "status": "in_progress", "role": "assistant", "content": []},
        })),
        raw_chunk(serde_json::json!({
            "type": "response.output_text.delta", "output_index": 0, "item_id": "msg_1",
            "content_index": 0, "delta": "Hello",
        })),
    ];
    let (events, failure) = drain_stream_fixture(Dialect::OpenAiResponses, &chunks);
    assert!(failure.is_none(), "{failure:?}");
    let kinds: Vec<_> = events.iter().map(|event| event["kind"].clone()).collect();
    assert!(
        kinds
            .iter()
            .any(|kind| kind == "provider_output_item_completed"),
        "open items are closed: {kinds:?}"
    );
    assert_eq!(kinds.last(), Some(&serde_json::Value::from("incomplete")));
}

#[test]
fn a_reasoning_summary_whose_done_text_differs_is_tolerated() {
    // The summary is display-only prose already relayed as deltas; a `done`
    // text that differs (gpt-5.6-luna, 9 streams 2026-09-12..13) is logged,
    // never a reason to fail the served answer.
    let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
    let frame = |payload: serde_json::Value| SseEvent {
        event: None,
        data: payload.to_string(),
    };
    normalizer
        .feed(&frame(serde_json::json!({
            "type": "response.reasoning_summary_text.delta", "output_index": 0,
            "summary_index": 0, "item_id": "rs_1", "delta": "Thinking about",
        })))
        .expect("delta normalizes");
    let events = normalizer
        .feed(&frame(serde_json::json!({
            "type": "response.reasoning_summary_text.done", "output_index": 0,
            "summary_index": 0, "item_id": "rs_1", "text": "Thinking about Paris.",
        })))
        .expect("a differing done text is tolerated");
    assert!(events.is_empty(), "the streamed text stands: {events:?}");
}
