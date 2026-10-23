//! Sparse and stale usage-report coalescing across one attempt.

use super::*;
use serde_json::json;

#[test]
fn established_reasoning_accounting_survives_sparse_growth_and_stale_totals() {
    for chat in [false, true] {
        for additive in [false, true] {
            let (input, output, details) = if chat {
                (
                    "prompt_tokens",
                    "completion_tokens",
                    "completion_tokens_details",
                )
            } else {
                ("input_tokens", "output_tokens", "output_tokens_details")
            };
            let first = json!({
                input: 100, output: 10, details: {"reasoning_tokens": 5},
                "total_tokens": if additive { 115 } else { 110 }
            });
            let mut accumulator = OpenAiUsageAccumulator::default();
            for (report, expected_output, expected_reasoning) in [
                (first.clone(), if additive { 15 } else { 10 }, 5),
                // For additive accounting the retained old total now matches
                // input + raw output, but it must not change the dialect.
                (json!({output: 15}), if additive { 20 } else { 15 }, 5),
                (json!({output: 20}), if additive { 25 } else { 20 }, 5),
                (
                    json!({details: {"reasoning_tokens": 8}}),
                    if additive { 28 } else { 20 },
                    8,
                ),
                (first, if additive { 28 } else { 20 }, 8),
                (json!({}), if additive { 28 } else { 20 }, 8),
                (
                    json!({"total_tokens": if additive { 128 } else { 120 }}),
                    if additive { 28 } else { 20 },
                    8,
                ),
            ] {
                let observed = accumulator
                    .update(report.as_object().unwrap(), chat)
                    .unwrap();
                assert_eq!(observed.input_tokens, Some(100));
                assert_eq!(observed.output_tokens, Some(expected_output), "{report}");
                assert_eq!(observed.reasoning_tokens, Some(expected_reasoning));
            }
        }
    }
}

#[test]
fn additive_reasoning_growth_rejects_overflow_without_advancing_raw_state() {
    let mut accumulator = OpenAiUsageAccumulator::default();
    let first = json!({
        "input_tokens": 100, "output_tokens": 10,
        "output_tokens_details": {"reasoning_tokens": 5}, "total_tokens": 115
    });
    accumulator.update_responses(Some(&first)).unwrap();
    assert!(accumulator
        .update_responses(Some(&json!({"output_tokens": MAXIMUM_LEDGER_COUNT})))
        .is_err());
    let observed = accumulator
        .update_responses(Some(&json!({"output_tokens": 20})))
        .unwrap()
        .unwrap();
    assert_eq!(observed.output_tokens, Some(25));
    assert_eq!(observed.reasoning_tokens, Some(5));
}

#[test]
fn undecided_reasoning_waits_for_a_positive_decisive_total() {
    for first in [
        json!({"input_tokens": 100, "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 5}}),
        json!({"input_tokens": 100, "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 102}),
    ] {
        let mut accumulator = OpenAiUsageAccumulator::default();
        accumulator.update_responses(Some(&first)).unwrap();
        let observed = accumulator
            .update_responses(Some(&json!({
                "output_tokens": 10, "output_tokens_details": {"reasoning_tokens": 5},
                "total_tokens": 110
            })))
            .unwrap()
            .unwrap();
        assert_eq!(observed.output_tokens, Some(10));
        assert!(matches!(
            accumulator.reasoning_accounting,
            Some(ReasoningAccounting::Subset)
        ));
    }
}

#[test]
fn missing_reasoning_does_not_let_a_stale_total_establish_subset_accounting() {
    let mut accumulator = OpenAiUsageAccumulator::default();
    for report in [
        json!({"input_tokens": 100, "output_tokens": 10, "total_tokens": 115}),
        json!({"output_tokens": 15, "output_tokens_details": {"reasoning_tokens": 5}}),
    ] {
        accumulator.update_responses(Some(&report)).unwrap();
    }
    let observed = accumulator
        .update_responses(Some(&json!({
            "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 5},
            "total_tokens": 125
        })))
        .unwrap()
        .unwrap();
    assert_eq!(observed.output_tokens, Some(25));
}

#[test]
fn late_totals_cannot_establish_accounting_after_raw_growth() {
    for (later_output, late_total) in [(15, 115), (20, 120)] {
        let mut accumulator = OpenAiUsageAccumulator::default();
        for report in [
            json!({"input_tokens": 100, "output_tokens": 10, "total_tokens": 115}),
            json!({"output_tokens": 15, "output_tokens_details": {"reasoning_tokens": 5}}),
            json!({"output_tokens": later_output}),
            json!({"total_tokens": late_total}),
        ] {
            accumulator.update_responses(Some(&report)).unwrap();
        }
        assert!(accumulator.reasoning_accounting.is_none());
        for additive in [false, true] {
            let mut decided = accumulator.clone();
            let observed = decided
                .update_responses(Some(&json!({
                    "output_tokens": 25, "output_tokens_details": {"reasoning_tokens": 5},
                    "total_tokens": if additive { 130 } else { 125 }
                })))
                .unwrap()
                .unwrap();
            assert_eq!(observed.output_tokens, Some(if additive { 30 } else { 25 }));
        }
    }
}

#[test]
fn coherent_old_snapshot_establishes_mode_from_its_own_raw_counts() {
    let mut accumulator = OpenAiUsageAccumulator::default();
    accumulator
        .update_responses(Some(&json!({"input_tokens": 100, "output_tokens": 20})))
        .unwrap();
    let observed = accumulator
        .update_responses(Some(&json!({
            "input_tokens": 100, "output_tokens": 10,
            "output_tokens_details": {"reasoning_tokens": 5}, "total_tokens": 115
        })))
        .unwrap()
        .unwrap();
    assert_eq!(observed.output_tokens, Some(25));
}

#[test]
fn changed_input_cannot_be_reused_in_a_delayed_counter_sample() {
    let mut accumulator = OpenAiUsageAccumulator::default();
    for report in [
        json!({"input_tokens": 100, "output_tokens": 10, "total_tokens": 115}),
        json!({"input_tokens": 105}),
        json!({"output_tokens": 10, "output_tokens_details": {"reasoning_tokens": 5},
            "total_tokens": 115}),
    ] {
        accumulator.update_responses(Some(&report)).unwrap();
    }
    assert!(accumulator.reasoning_accounting.is_none());
    let observed = accumulator
        .update_responses(Some(&json!({
            "input_tokens": 105, "output_tokens": 15,
            "output_tokens_details": {"reasoning_tokens": 5}, "total_tokens": 125
        })))
        .unwrap()
        .unwrap();
    assert_eq!(observed.output_tokens, Some(20));
}

#[test]
fn unchanged_initial_fragments_preserve_total_first_accounting() {
    for additive in [false, true] {
        let pieces = [
            json!({"input_tokens": 100}),
            json!({"output_tokens": 10}),
            json!({"output_tokens_details": {"reasoning_tokens": 5}}),
            json!({"total_tokens": if additive { 115 } else { 110 }}),
        ];
        for order in [[0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 1, 0], [1, 0, 3, 2]] {
            let mut accumulator = OpenAiUsageAccumulator::default();
            let mut observed = None;
            for index in order {
                observed = accumulator.update_responses(Some(&pieces[index])).unwrap();
            }
            assert_eq!(
                observed.unwrap().output_tokens,
                Some(if additive { 15 } else { 10 })
            );
        }
    }
}

#[test]
fn unreported_meter_marker_rejects_malformed_or_contradictory_evidence() {
    for marker in [
        json!(null),
        json!({}),
        json!("cached_tokens"),
        json!([1]),
        json!(["other"]),
        json!(["cached_tokens", "cached_tokens"]),
    ] {
        assert!(OpenAiUsageAccumulator::default()
            .update_chat(&json!({
                "prompt_tokens": 10, "completion_tokens": 10,
                "unreported_token_details": marker
            }))
            .is_err());
    }
    for field in [
        "cached_tokens",
        "cache_write_tokens",
        "cache_write_1h_tokens",
        "reasoning_tokens",
    ] {
        for count in [json!(1), json!(-1), json!(true), json!(0.5), json!("0")] {
            let group = if field == "reasoning_tokens" {
                "completion_tokens_details"
            } else {
                "prompt_tokens_details"
            };
            let mut report = json!({
                "prompt_tokens": 10, "completion_tokens": 10,
                "unreported_token_details": [field]
            });
            report[group] = json!({field: count});
            assert!(OpenAiUsageAccumulator::default()
                .update_chat(&report)
                .is_err());
        }
    }
}

#[test]
fn sparse_cache_ttl_is_retained_checked_and_invalidated_on_write_growth() {
    let mut accumulator = OpenAiUsageAccumulator::default();
    let partial = accumulator
        .update_chat(&json!({
            "prompt_tokens_details": {"cache_write_1h_tokens": 3}
        }))
        .unwrap();
    assert_eq!(partial.cache_creation_1h_input_tokens, None);
    assert!(accumulator
        .update_chat(&json!({
            "prompt_tokens": 20,
            "prompt_tokens_details": {"cache_write_tokens": 2}
        }))
        .is_err());
    let complete = accumulator
        .update_chat(&json!({
            "prompt_tokens": 20, "completion_tokens": 2,
            "prompt_tokens_details": {"cache_write_tokens": 5}
        }))
        .unwrap();
    assert_eq!(complete.cache_creation_1h_input_tokens, Some(3));
    let grown = accumulator
        .update_chat(&json!({
            "prompt_tokens_details": {"cache_write_tokens": 10}
        }))
        .unwrap();
    assert_eq!(grown.cache_creation_1h_input_tokens, None);
    let zero = accumulator
        .update_chat(&json!({
            "prompt_tokens_details": {"cache_write_1h_tokens": 0}
        }))
        .unwrap();
    assert_eq!(zero.cache_creation_1h_input_tokens, Some(0));
}

#[test]
fn cache_ttl_refuses_malformed_counts_and_non_subsets() {
    for bad in [
        json!(-1),
        json!(true),
        json!(1.5),
        json!("2"),
        json!(101),
        json!(MAXIMUM_LEDGER_COUNT + 1),
    ] {
        assert!(OpenAiUsageAccumulator::default()
            .update_chat(&json!({
                "prompt_tokens": 120, "completion_tokens": 10,
                "prompt_tokens_details": {"cache_write_tokens": 100, "cache_write_1h_tokens": bad}
            }))
            .is_err());
        assert!(OpenAiUsageAccumulator::default()
            .update_responses(Some(&json!({
                "input_tokens": 120, "output_tokens": 10,
                "input_tokens_details": {"cache_write_tokens": 100, "cache_write_1h_tokens": bad}
            })))
            .is_err());
    }
}

#[test]
fn responses_sparse_raw_counters_match_whole_report_and_never_fold_twice() {
    let full = json!({"input_tokens":100,"output_tokens":10,"output_tokens_details":{"reasoning_tokens":5},"total_tokens":115});
    let expected = openai_usage(Some(&full)).unwrap().unwrap();
    let mut accumulator = OpenAiUsageAccumulator::default();
    for report in [
        json!({"input_tokens":100}),
        json!({"output_tokens":10}),
        json!({"output_tokens_details":{"reasoning_tokens":5},"total_tokens":115}),
        full.clone(),
        full,
    ] {
        let result = accumulator
            .update_responses(Some(&report))
            .unwrap()
            .unwrap();
        if report.get("total_tokens").is_some() {
            assert_eq!(result.input_tokens, expected.input_tokens);
            assert_eq!(result.output_tokens, expected.output_tokens);
            assert_eq!(result.reasoning_tokens, expected.reasoning_tokens);
        }
    }
    assert_eq!(expected.output_tokens, Some(15));
}

#[test]
fn sparse_response_cache_is_fitted_when_input_arrives_later() {
    let mut accumulator = OpenAiUsageAccumulator::default();
    accumulator
        .update_responses(Some(&json!({"input_tokens_details":{"cached_tokens":200}})))
        .unwrap();
    // An input below the earlier read count is raised to cover it, while the
    // provider's own raw count stays retained for later reports.
    let fitted = accumulator
        .update_responses(Some(&json!({"input_tokens":100,"output_tokens":1})))
        .unwrap()
        .unwrap();
    assert_eq!(fitted.input_tokens, Some(200));
    assert_eq!(fitted.cached_input_tokens, Some(200));
    let valid = accumulator
        .update_responses(Some(&json!({"input_tokens":250,"output_tokens":1})))
        .unwrap()
        .unwrap();
    assert_eq!(valid.input_tokens, Some(250));
    assert_eq!(valid.cached_input_tokens, Some(200));
}
