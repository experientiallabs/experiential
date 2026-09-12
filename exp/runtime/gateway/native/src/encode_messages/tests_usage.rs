//! Inline tests for the Anthropic usage object every Messages frame carries,
//! split from `tests` so each test file stays within the repository line
//! budget.

use serde_json::json;

use super::usage::messages_usage;
use crate::events::Usage;

#[test]
fn usage_reports_cached_reads_out_of_the_input_total() {
    let usage = Usage {
        input_tokens: Some(10),
        output_tokens: Some(4),
        cached_input_tokens: Some(3),
        cache_creation_input_tokens: None,
        reasoning_tokens: None,
    };
    assert_eq!(
        messages_usage(Some(&usage)),
        json!({
            "input_tokens": 7,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 3,
            "output_tokens": 4,
        })
    );
    assert_eq!(
        messages_usage(None),
        json!({
            "input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 0,
        })
    );
}

#[test]
fn usage_carries_both_cache_legs_as_zero_when_the_provider_reports_none() {
    // An OpenAI-wire provider that reports no `prompt_tokens_details` is an
    // uncached completion in Anthropic's shape, not a completion with the
    // cache fields missing: Claude Code and the SDK accumulators read the
    // legs by key.
    let usage = Usage {
        input_tokens: Some(437),
        output_tokens: Some(12),
        cached_input_tokens: None,
        cache_creation_input_tokens: None,
        reasoning_tokens: None,
    };
    assert_eq!(
        messages_usage(Some(&usage)),
        json!({
            "input_tokens": 437,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 12,
        })
    );
}

#[test]
fn usage_reports_openai_wire_cached_tokens_as_cache_reads() {
    // The folded total from `prompt_tokens_details.cached_tokens` (a real
    // Claude Code turn on an OpenAI-compatible rung: 437 prompt tokens of
    // which 256 were a cached prefix) comes back as uncached input plus the
    // cache-read leg.
    let usage = Usage {
        input_tokens: Some(437),
        output_tokens: Some(12),
        cached_input_tokens: Some(256),
        cache_creation_input_tokens: None,
        reasoning_tokens: None,
    };
    assert_eq!(
        messages_usage(Some(&usage)),
        json!({
            "input_tokens": 181,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 256,
            "output_tokens": 12,
        })
    );
}

#[test]
fn usage_reports_both_cache_legs_out_of_the_folded_input_total() {
    // A live cached turn folds input as uncached + read + write for the
    // ledger; callers get the provider's own shape back, with each cache
    // leg on its own field (Claude Code displays cache_creation on turn 1).
    let usage = Usage {
        input_tokens: Some(45_543),
        output_tokens: Some(9),
        cached_input_tokens: Some(0),
        cache_creation_input_tokens: Some(45_338),
        reasoning_tokens: None,
    };
    assert_eq!(
        messages_usage(Some(&usage)),
        json!({
            "input_tokens": 205,
            "cache_creation_input_tokens": 45_338,
            "cache_read_input_tokens": 0,
            "output_tokens": 9,
        })
    );
}
