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
fn oversized_cache_suffix_waits_for_later_input_even_with_a_valid_baseline() {
    for later_input in [None, Some(12)] {
        let mut frames = vec![
            text(),
            frame(json!({"usageMetadata":meter()})),
            stop(),
            frame(json!({"usageMetadata":{"cachedContentTokenCount":10}})),
            frame(json!({"usageMetadata":{}})),
        ];
        if let Some(input) = later_input {
            frames.push(frame(json!({"usageMetadata":{"promptTokenCount":input}})));
        }
        let expected = if later_input.is_some() {
            (12, 2, 10)
        } else {
            (7, 2, 3)
        };
        assert_usage(frames.clone(), Some(expected));
        assert_usage(vec![frames.concat()], Some(expected));
    }
}

#[test]
fn unreconciled_cache_does_not_hide_later_consistent_primary_counts() {
    let mut frames = vec![
        text(),
        frame(json!({"usageMetadata":meter()})),
        stop(),
        frame(json!({"usageMetadata":{"cachedContentTokenCount":1000}})),
        frame(json!({"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":5}})),
    ];
    assert_usage(frames.clone(), Some((12, 5, 3)));
    frames.push(frame(json!({"usageMetadata":{"promptTokenCount":1200}})));
    assert_usage(frames, Some((1200, 5, 1000)));
    // A newer frame that itself contradicts the subset relation does not
    // authorize publishing its other counts as a consistent snapshot.
    assert_usage(
        vec![
            text(),
            frame(json!({"usageMetadata":meter()})),
            stop(),
            frame(
                json!({"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":5,"cachedContentTokenCount":1000}}),
            ),
        ],
        Some((7, 2, 3)),
    );
}

#[test]
fn pending_cache_does_not_hide_a_newer_valid_cache_subset() {
    for order in [[0, 1], [1, 0]] {
        for finish_at in 0..=2 {
            for empty in [false, true] {
                let reports = [
                    json!({"cachedContentTokenCount":1000}),
                    json!({"promptTokenCount":12,"candidatesTokenCount":5,"cachedContentTokenCount":6}),
                ];
                let mut frames = vec![text(), frame(json!({"usageMetadata":meter()}))];
                for index in 0..=2 {
                    if index == finish_at {
                        frames.push(stop());
                    }
                    if index < 2 {
                        frames.push(frame(json!({"usageMetadata":reports[order[index]]})));
                    }
                    if empty {
                        frames.push(frame(json!({"usageMetadata":{}})));
                    }
                }
                assert_usage(frames.clone(), Some((12, 5, 6)));
                assert_usage(vec![frames.concat()], Some((12, 5, 6)));
                frames.push(frame(json!({"usageMetadata":{"promptTokenCount":1200}})));
                assert_usage(frames, Some((1200, 5, 1000)));
            }
        }
    }
}

#[test]
fn multiple_pending_subsets_reconcile_independently_in_either_order() {
    for reports in [
        vec![
            json!({"cachedContentTokenCount":10}),
            json!({"cachedContentTokenCount":1000}),
        ],
        vec![
            json!({"cachedContentTokenCount":1000}),
            json!({"cachedContentTokenCount":10}),
        ],
        vec![
            json!({"promptTokenCount":8,"candidatesTokenCount":99,"cachedContentTokenCount":10}),
            json!({"cachedContentTokenCount":1000}),
        ],
    ] {
        let mut frames = vec![text(), frame(json!({"usageMetadata":meter()})), stop()];
        for report in reports {
            frames.push(frame(json!({"usageMetadata":report})));
        }
        frames.push(frame(json!({"usageMetadata":{"promptTokenCount":12}})));
        assert_usage(frames.clone(), Some((12, 2, 10)));
        assert_usage(vec![frames.concat()], Some((12, 2, 10)));
        frames.push(frame(json!({"usageMetadata":{"promptTokenCount":1200}})));
        assert_usage(frames, Some((1200, 2, 1000)));
    }
    for cache_first in [false, true] {
        for cache in [6, 10] {
            let mut reports = vec![
                json!({"cachedContentTokenCount":cache}),
                json!({"promptTokenCount":12}),
            ];
            if !cache_first {
                reports.reverse();
            }
            let mut frames = vec![
                text(),
                frame(json!({"usageMetadata":meter()})),
                stop(),
                frame(json!({"usageMetadata":{"cachedContentTokenCount":1000}})),
            ];
            frames.extend(
                reports
                    .into_iter()
                    .map(|report| frame(json!({"usageMetadata":report}))),
            );
            assert_usage(frames, Some((12, 2, cache)));
        }
    }
}

#[test]
fn pending_cache_capacity_is_bounded_deduplicated_and_transactional() {
    use crate::dialects::MAXIMUM_RETAINED_PROVIDER_ENTRIES;
    let mut normalizer = Normalizer::new(Dialect::GeminiGenerateContent);
    for payload in [
        json!({"usageMetadata":meter()}),
        json!({"candidates":[{"finishReason":"STOP"}]}),
    ] {
        normalizer
            .feed(&crate::sse::SseEvent {
                event: None,
                data: payload.to_string(),
            })
            .unwrap();
    }
    for cache in 10..10 + MAXIMUM_RETAINED_PROVIDER_ENTRIES as u64 {
        normalizer
            .observe_gemini_usage(&json!({"cachedContentTokenCount":cache}))
            .unwrap();
    }
    // A duplicate at capacity is not an additional retained entry.
    normalizer
        .observe_gemini_usage(&json!({"cachedContentTokenCount":10}))
        .unwrap();
    let overflow = normalizer
        .observe_gemini_usage(&json!({"candidatesTokenCount":99,"cachedContentTokenCount":5000}))
        .unwrap_err();
    assert!(overflow.safe_message.contains("pending cache"));
    let usage = normalizer.observed_usage().unwrap();
    assert_eq!(
        (
            usage.input_tokens,
            usage.output_tokens,
            usage.cached_input_tokens
        ),
        (Some(7), Some(2), Some(3))
    );
    // Reconciled entries free capacity, and the rejected report mutated no leg.
    normalizer
        .observe_gemini_usage(&json!({"promptTokenCount":12}))
        .unwrap();
    let usage = normalizer.observed_usage().unwrap();
    assert_eq!(
        (
            usage.input_tokens,
            usage.output_tokens,
            usage.cached_input_tokens
        ),
        (Some(12), Some(2), Some(12))
    );
    normalizer
        .observe_gemini_usage(&json!({"cachedContentTokenCount":5000}))
        .unwrap();
    let mut frames = vec![text(), frame(json!({"usageMetadata":meter()})), stop()];
    frames.extend(
        (10..11 + MAXIMUM_RETAINED_PROVIDER_ENTRIES as u64)
            .map(|cache| frame(json!({"usageMetadata":{"cachedContentTokenCount":cache}}))),
    );
    assert_usage(frames, Some((7, 2, 3)));
}

#[test]
fn empty_baseline_cannot_publish_output_before_pending_cache_reconciles() {
    for empty_first in [false, true] {
        let mut frames = vec![text(), stop()];
        if empty_first {
            frames.push(frame(json!({"usageMetadata":{}})));
        }
        frames.extend([
            frame(json!({"usageMetadata":{"cachedContentTokenCount":3}})),
            frame(json!({"usageMetadata":{"candidatesTokenCount":2,"thoughtsTokenCount":4}})),
            frame(json!({"usageMetadata":{}})),
        ]);
        assert_usage(frames.clone(), empty_first.then_some((0, 0, 0)));
        frames.push(frame(json!({"usageMetadata":{"promptTokenCount":7}})));
        assert_usage(frames, Some((7, 6, 3)));
    }
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
fn pending_cache_preserves_output_legs_before_the_first_input_report() {
    assert_usage(
        vec![
            text(),
            stop(),
            frame(json!({"usageMetadata":{"cachedContentTokenCount":3}})),
            frame(json!({"usageMetadata":{"candidatesTokenCount":2,"thoughtsTokenCount":4}})),
            frame(json!({"usageMetadata":{}})),
            frame(json!({"usageMetadata":{"promptTokenCount":7}})),
        ],
        Some((7, 6, 3)),
    );
}

#[test]
fn explicitly_contradictory_snapshot_cannot_pollute_a_later_empty_suffix() {
    for prior in [false, true] {
        let mut frames = vec![text(), stop()];
        if prior {
            frames.push(frame(json!({"usageMetadata":meter()})));
        }
        frames.push(frame(json!({"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":40,"cachedContentTokenCount":500}})));
        frames.push(frame(json!({"usageMetadata":{}})));
        assert_usage(frames, prior.then_some((7, 2, 3)));
    }
}

#[test]
fn split_usage_fields_are_order_independent_across_finish_and_empty_frames() {
    let fields = [
        json!({"promptTokenCount":7}),
        json!({"candidatesTokenCount":2}),
        json!({"thoughtsTokenCount":4}),
        json!({"cachedContentTokenCount":3}),
    ];
    let combined = json!({"promptTokenCount":7,"candidatesTokenCount":2,"thoughtsTokenCount":4,"cachedContentTokenCount":3});
    let mut cases = 0;
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let order = [a, b, c, d];
                    if (0..4).any(|i| (i + 1..4).any(|j| order[i] == order[j])) {
                        continue;
                    }
                    for finish_at in 0..=4 {
                        for prior in [false, true] {
                            for empty in [false, true] {
                                let mut normalizer =
                                    Normalizer::new(Dialect::GeminiGenerateContent);
                                let mut payloads = vec![
                                    json!({"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}),
                                ];
                                if prior {
                                    payloads.push(json!({"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":1,"thoughtsTokenCount":1,"cachedContentTokenCount":1}}));
                                }
                                for index in 0..=4 {
                                    if index == finish_at {
                                        payloads
                                            .push(json!({"candidates":[{"finishReason":"STOP"}]}));
                                    }
                                    if index < 4 {
                                        payloads
                                            .push(json!({"usageMetadata":fields[order[index]]}));
                                    }
                                    if empty {
                                        payloads.push(json!({"usageMetadata":{}}));
                                    }
                                }
                                let mut events = Vec::new();
                                for payload in payloads {
                                    events.extend(
                                        normalizer
                                            .feed(&crate::sse::SseEvent {
                                                event: None,
                                                data: payload.to_string(),
                                            })
                                            .unwrap(),
                                    );
                                    if let Some(usage) = normalizer.observed_usage() {
                                        assert!(
                                            usage.cached_input_tokens <= usage.input_tokens,
                                            "invalid intermediate meter: {usage:?}"
                                        );
                                    }
                                }
                                events.extend(normalizer.on_stream_end().unwrap());
                                let actual = events
                                    .iter()
                                    .find_map(|event| match event {
                                        Event::Usage(usage) => {
                                            Some(crate::events::simplified_event(&Event::Usage(
                                                usage.clone(),
                                            )))
                                        }
                                        _ => None,
                                    })
                                    .unwrap();
                                let expected = crate::events::simplified_event(&Event::Usage(
                                    crate::events::gemini_usage(&combined).unwrap(),
                                ));
                                assert_eq!(actual, expected, "order={order:?} finish={finish_at} prior={prior} empty={empty}");
                                assert_eq!(events.iter().filter(|e| e.is_terminal()).count(), 1);
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 480);
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
