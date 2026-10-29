//! OpenAI 12-hour cache-write metering on Responses usage.

use super::super::OpenAiUsageAccumulator;
use serde_json::{json, Value};

/// The live `gpt-5.6-luna` Responses usage for a `ttl: "12h"` request (2026-10-20).
fn live_twelve_hour_usage() -> Value {
    json!({
        "input_tokens": 3250,
        "input_tokens_details": {
            "cache_write_tokens": 3247,
            "cache_write_tokens_details": {"ttl_12h_tokens": 3247, "ttl_30m_tokens": 0},
            "cached_tokens": 0
        },
        "output_tokens": 5,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 3255
    })
}

#[test]
fn responses_usage_meters_the_twelve_hour_write_subset() {
    let usage = OpenAiUsageAccumulator::default()
        .update_responses(Some(&live_twelve_hour_usage()))
        .unwrap()
        .unwrap();
    assert_eq!(usage.cache_creation_input_tokens, Some(3247));
    assert_eq!(usage.cache_creation_12h_input_tokens, Some(3247));
    assert_eq!(usage.cache_creation_1h_input_tokens, None);
}

#[test]
fn a_mixed_split_keeps_only_the_twelve_hour_part() {
    let mut wire = live_twelve_hour_usage();
    wire["input_tokens_details"]["cache_write_tokens_details"] =
        json!({"ttl_12h_tokens": 2000, "ttl_30m_tokens": 1247});
    let usage = OpenAiUsageAccumulator::default()
        .update_responses(Some(&wire))
        .unwrap()
        .unwrap();
    assert_eq!(usage.cache_creation_12h_input_tokens, Some(2000));
}

#[test]
fn twelve_hour_writes_beyond_total_writes_fail_the_stream() {
    let mut wire = live_twelve_hour_usage();
    wire["input_tokens_details"]["cache_write_tokens_details"]["ttl_12h_tokens"] = json!(3248);
    assert!(OpenAiUsageAccumulator::default()
        .update_responses(Some(&wire))
        .is_err());
    wire["input_tokens_details"]["cache_write_tokens_details"] = json!("12h");
    assert!(OpenAiUsageAccumulator::default()
        .update_responses(Some(&wire))
        .is_err());
}

#[test]
fn a_twelve_hour_request_without_a_split_bills_every_write_at_twelve_hours() {
    let mut wire = live_twelve_hour_usage();
    wire["input_tokens_details"]
        .as_object_mut()
        .unwrap()
        .remove("cache_write_tokens_details");
    let mut requested = OpenAiUsageAccumulator::default();
    requested.set_twelve_hour_writes_requested(true);
    let usage = requested.update_responses(Some(&wire)).unwrap().unwrap();
    assert_eq!(usage.cache_creation_12h_input_tokens, Some(3247));
    // The default lifetime reports no split and has no 12-hour subset.
    let usage = OpenAiUsageAccumulator::default()
        .update_responses(Some(&wire))
        .unwrap()
        .unwrap();
    assert_eq!(usage.cache_creation_12h_input_tokens, None);
}
