//! Gemini finish freezes content while transport trailers complete the meter.

use super::*;
use crate::dialects::{drain_stream_fixture, Dialect};
use serde_json::json;

fn frame(payload: Value) -> Vec<u8> {
    format!("data: {payload}\n\n").into_bytes()
}

fn text() -> Vec<u8> {
    frame(json!({"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}))
}

fn stop() -> Vec<u8> {
    frame(json!({"candidates":[{"finishReason":"STOP"}]}))
}

fn meter() -> Value {
    json!({"promptTokenCount":7,"candidatesTokenCount":2,"cachedContentTokenCount":3})
}

fn assert_usage(chunks: Vec<Vec<u8>>, expected: Option<(u64, u64, u64)>) {
    let (events, failure) = drain_stream_fixture(Dialect::GeminiGenerateContent, &chunks);
    assert!(failure.is_none(), "{failure:?}");
    assert_eq!(
        events.first().unwrap(),
        &json!({"kind":"text_delta","text":"hi"})
    );
    assert_eq!(events.last().unwrap(), &json!({"kind":"completed"}));
    assert_eq!(
        events.iter().filter(|e| e["kind"] == "completed").count(),
        1
    );
    let usages: Vec<_> = events.iter().filter(|e| e["kind"] == "usage").collect();
    match expected {
        Some((input, output, cache)) => {
            assert_eq!(usages.len(), 1, "{events:?}");
            assert_eq!(usages[0]["input_tokens"], input);
            assert_eq!(usages[0]["output_tokens"], output);
            assert_eq!(usages[0]["cached_input_tokens"], cache);
        }
        None => assert!(usages.is_empty(), "absent whole object stays unknown"),
    }
    assert_eq!(events.len(), if expected.is_some() { 3 } else { 2 });
}

#[test]
fn meters_before_with_and_after_finish_survive_every_chunk_boundary() {
    for frames in [
        vec![text(), frame(json!({"usageMetadata":meter()})), stop()],
        vec![
            text(),
            frame(json!({"candidates":[{"finishReason":"STOP"}],"usageMetadata":meter()})),
        ],
        vec![text(), stop(), frame(json!({"usageMetadata":meter()}))],
    ] {
        assert_usage(frames.clone(), Some((7, 2, 3)));
        let wire = frames.concat();
        for split in 0..=wire.len() {
            assert_usage(
                vec![wire[..split].to_vec(), wire[split..].to_vec()],
                Some((7, 2, 3)),
            );
        }
    }
}

#[test]
fn partial_and_empty_trailers_preserve_counts_and_add_new_cache_evidence() {
    assert_usage(
        vec![
            text(),
            frame(json!({"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":2}})),
            stop(),
            frame(json!({"usageMetadata":{"cachedContentTokenCount":3}})),
            frame(json!({"usageMetadata":{}})),
            frame(json!({"usageMetadata":null})),
        ],
        Some((7, 2, 3)),
    );
    // Additive thought and candidate legs can arrive in different snapshots.
    assert_usage(
        vec![
            text(),
            frame(json!({"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":2}})),
            stop(),
            frame(json!({"usageMetadata":{"thoughtsTokenCount":4,"cachedContentTokenCount":3}})),
        ],
        Some((7, 6, 3)),
    );
}

#[test]
fn absent_metadata_is_unknown_but_present_empty_metadata_is_explicit_zero() {
    assert_usage(vec![text(), stop()], None);
    assert_usage(
        vec![text(), stop(), frame(json!({"usageMetadata":null}))],
        None,
    );
    assert_usage(
        vec![text(), stop(), frame(json!({"usageMetadata":{}}))],
        Some((0, 0, 0)),
    );
}

#[test]
fn late_content_tools_errors_and_second_finish_cannot_change_the_answer() {
    assert_usage(
        vec![
            text(),
            stop(),
            frame(json!({
                "candidates":[{"content":{"parts":[{"text":"late"},{"functionCall":{"name":"late","args":{}}}]},"finishReason":"MAX_TOKENS"}],
                "error":{"status":"UNAVAILABLE","message":"late failure"},
                "usageMetadata":meter(),
            })),
        ],
        Some((7, 2, 3)),
    );
}

#[test]
fn impossible_cache_suffix_keeps_last_consistent_usage() {
    assert_usage(
        vec![
            text(),
            frame(json!({"usageMetadata":meter()})),
            stop(),
            frame(json!({"usageMetadata":{"cachedContentTokenCount":10}})),
        ],
        Some((7, 2, 3)),
    );
}

#[test]
fn cache_only_first_report_waits_for_primary_counts_or_stays_unknown() {
    let partial = vec![
        text(),
        stop(),
        frame(json!({"usageMetadata":{"cachedContentTokenCount":3}})),
    ];
    assert_usage(partial.clone(), None);
    let mut completed = partial;
    completed.push(frame(
        json!({"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":2}}),
    ));
    assert_usage(completed, Some((7, 2, 3)));
}

#[test]
fn malformed_or_partial_trailer_keeps_the_declared_finish_and_best_meter() {
    for tail in [
        b"data: {broken\n\n".to_vec(),
        b"data: {\"usageMetadata\":".to_vec(),
        frame(json!({"usageMetadata":{"promptTokenCount":"bad"}})),
    ] {
        assert_usage(
            vec![
                text(),
                frame(json!({"usageMetadata":meter()})),
                stop(),
                tail,
            ],
            Some((7, 2, 3)),
        );
    }
    // An unterminated but complete SSE data line is valid at EOF.
    assert_usage(
        vec![
            text(),
            stop(),
            format!("data: {}", json!({"usageMetadata":meter()})).into_bytes(),
        ],
        Some((7, 2, 3)),
    );
}
