use super::*;

fn drain_capped(cap: Option<u64>, frames: &[Value]) -> (Vec<Event>, bool) {
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    normalizer.set_output_token_cap(cap);
    let mut events = Vec::new();
    for payload in frames {
        events.extend(
            normalizer
                .feed(&SseEvent {
                    event: None,
                    data: payload.to_string(),
                })
                .expect("frame normalizes"),
        );
    }
    events.extend(
        normalizer
            .feed(&SseEvent {
                event: None,
                data: "[DONE]".to_string(),
            })
            .expect("sentinel normalizes"),
    );
    (events, normalizer.unreported_stop_read_as_length())
}

/// Meta's muse lanes (2026-10-17) end a stream that hit `max_tokens` with
/// `finish_reason: stop` and no usage frame, while every natural stop
/// reports usage. On a capped request that usage-less stop is the budget
/// truncation and settles Incomplete, so the caller sees `length`.
#[test]
fn capped_stop_without_usage_is_the_budget_truncation() {
    let frames = [text_frame("Packet 1470 of the"), finish_frame("stop")];
    let (events, read_as_length) = drain_capped(Some(40_000), &frames);
    assert!(read_as_length);
    assert!(
        matches!(events.last(), Some(Event::Incomplete)),
        "{events:?}"
    );
}

/// A reported stop, an uncapped stop, and a usage-less tool-call finish keep
/// their declared ending: only the capped usage-less `stop` is reread.
#[test]
fn reported_uncapped_or_tool_finishes_keep_their_declared_ending() {
    let usage = json!({"choices": [], "usage": {"prompt_tokens": 15, "completion_tokens": 54}});
    let reported = [text_frame("ok"), finish_frame("stop"), usage];
    let (events, read_as_length) = drain_capped(Some(40_000), &reported);
    assert!(!read_as_length);
    assert!(
        matches!(events.last(), Some(Event::Completed)),
        "{events:?}"
    );

    let uncapped = [text_frame("ok"), finish_frame("stop")];
    let (events, read_as_length) = drain_capped(None, &uncapped);
    assert!(!read_as_length);
    assert!(
        matches!(events.last(), Some(Event::Completed)),
        "{events:?}"
    );

    let tool = [
        json!({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1",
            "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]}}]}),
        finish_frame("tool_calls"),
    ];
    let (events, read_as_length) = drain_capped(Some(40_000), &tool);
    assert!(!read_as_length);
    assert!(
        matches!(events.last(), Some(Event::Completed)),
        "{events:?}"
    );
}
