//! OpenRouter's reported per-request charge on Chat Completions usage.

use serde_json::{json, Number, Value};

use super::{openrouter_cost_nano_usd, usd_to_nano_usd_ceil};
use crate::dialects::{Dialect, Normalizer};
use crate::events::{Event, Usage};
use crate::sse::SseEvent;

/// OpenRouter's usage object for a $0.0016 non-BYOK generation (shape
/// captured from openrouter.ai).
fn openrouter_usage() -> Value {
    json!({
        "prompt_tokens": 14,
        "completion_tokens": 543,
        "total_tokens": 557,
        "cost": 0.0016,
        "is_byok": false,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        "cost_details": {"upstream_inference_cost": null},
        "completion_tokens_details": {"reasoning_tokens": 480, "image_tokens": 0},
    })
}

/// One SSE data frame carrying `value`.
fn data(value: &Value) -> SseEvent {
    SseEvent {
        event: None,
        data: value.to_string(),
    }
}

/// Stream a text answer whose final chunk carries `usage` through a Chat
/// Completions normalizer and return the terminal usage event it settles.
fn streamed_usage(dialect: Dialect, reports_cost: bool, usage: Value) -> Usage {
    let mut normalizer = Normalizer::new(dialect);
    normalizer.set_provider_reported_cost(reports_cost);
    let frames = [
        json!({"choices": [{"index": 0, "delta": {"content": "OK"}}]}),
        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        json!({"choices": [], "provider": "Fireworks", "usage": usage}),
    ];
    let mut events = Vec::new();
    for frame in &frames {
        events.extend(normalizer.feed(&data(frame)).expect("valid frame"));
    }
    events.extend(
        normalizer
            .feed(&SseEvent {
                event: None,
                data: "[DONE]".to_string(),
            })
            .expect("valid sentinel"),
    );
    assert!(
        matches!(events.last(), Some(Event::Completed)),
        "{events:?}"
    );
    events
        .into_iter()
        .find_map(|event| match event {
            Event::Usage(usage) => Some(usage),
            _ => None,
        })
        .expect("terminal usage")
}

/// Convert one JSON number literal.
fn nano(literal: &str) -> Option<u64> {
    let number: Number = serde_json::from_str(literal).expect("number literal");
    usd_to_nano_usd_ceil(&number)
}

#[test]
fn usd_converts_to_whole_nano_usd_from_the_sent_decimal_rounding_up() {
    assert_eq!(nano("0.0016"), Some(1_600_000));
    // 0.000123 * 1e9 is 123000.00000000001 in binary floating point; the
    // decimal text is exact, so no spurious nano is added.
    assert_eq!(nano("0.000123"), Some(123_000));
    assert_eq!(nano("0"), Some(0));
    assert_eq!(nano("0.0"), Some(0));
    assert_eq!(nano("2"), Some(2_000_000_000));
    assert_eq!(nano("1.5"), Some(1_500_000_000));
    // Sub-nano remainders round up to the next whole nano-USD.
    assert_eq!(nano("0.0000000001"), Some(1));
    assert_eq!(nano("0.0000012345678"), Some(1_235));
    assert_eq!(nano("1e-7"), Some(100));
    assert_eq!(nano("-0.001"), None);
    assert_eq!(nano("1e300"), None);
    assert_eq!(nano("18446744073709551615"), None);
}

#[test]
fn a_streamed_openrouter_final_usage_chunk_reports_its_cost() {
    let usage = streamed_usage(Dialect::OpenAiCompatible, true, openrouter_usage());
    assert_eq!(usage.provider_reported_cost_nano_usd, Some(1_600_000));
    assert_eq!(usage.input_tokens, Some(14));
    assert_eq!(usage.output_tokens, Some(543));
}

#[test]
fn an_absent_byok_flag_still_reports_the_cost() {
    let mut wire = openrouter_usage();
    wire.as_object_mut().expect("object").remove("is_byok");
    let usage = streamed_usage(Dialect::OpenAiCompatible, true, wire);
    assert_eq!(usage.provider_reported_cost_nano_usd, Some(1_600_000));
}

#[test]
fn a_byok_answer_reports_no_cost() {
    let mut wire = openrouter_usage();
    wire["is_byok"] = json!(true);
    wire["cost_details"] = json!({"upstream_inference_cost": 0.0312});
    let usage = streamed_usage(Dialect::OpenAiCompatible, true, wire);
    assert_eq!(usage.provider_reported_cost_nano_usd, None);
    assert_eq!(usage.input_tokens, Some(14));
    // A non-boolean flag is no evidence the answer was not BYOK.
    let mut wire = openrouter_usage();
    wire["is_byok"] = json!("false");
    let object = wire.as_object().expect("object");
    assert_eq!(openrouter_cost_nano_usd(object), None);
}

#[test]
fn a_missing_cost_reports_none_even_beside_an_upstream_cost() {
    let mut wire = openrouter_usage();
    wire.as_object_mut().expect("object").remove("cost");
    wire["cost_details"] = json!({"upstream_inference_cost": 0.0016});
    let usage = streamed_usage(Dialect::OpenAiCompatible, true, wire);
    assert_eq!(usage.provider_reported_cost_nano_usd, None);
    assert_eq!(usage.output_tokens, Some(543));
    // A malformed cost is no report either, and never fails the stream.
    let mut wire = openrouter_usage();
    wire["cost"] = json!("0.0016");
    let usage = streamed_usage(Dialect::OpenAiCompatible, true, wire);
    assert_eq!(usage.provider_reported_cost_nano_usd, None);
}

#[test]
fn a_non_openrouter_rung_reports_no_cost_from_the_same_usage() {
    let usage = streamed_usage(Dialect::OpenAiCompatible, false, openrouter_usage());
    assert_eq!(usage.provider_reported_cost_nano_usd, None);
    assert_eq!(usage.input_tokens, Some(14));
}

#[test]
fn a_later_usage_report_without_cost_keeps_the_reported_cost() {
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    normalizer.set_provider_reported_cost(true);
    let mut sparse = openrouter_usage();
    sparse.as_object_mut().expect("object").remove("cost");
    for usage in [openrouter_usage(), sparse] {
        normalizer
            .feed(&data(&json!({"choices": [], "usage": usage})))
            .expect("valid usage");
    }
    let usage = normalizer.observed_usage().expect("observed usage");
    assert_eq!(usage.provider_reported_cost_nano_usd, Some(1_600_000));
}

#[test]
fn a_null_byok_flag_is_no_evidence_the_answer_was_not_byok() {
    let mut wire = openrouter_usage();
    wire["is_byok"] = Value::Null;
    let usage = streamed_usage(Dialect::OpenAiCompatible, true, wire);
    assert_eq!(usage.provider_reported_cost_nano_usd, None);
}

#[test]
fn an_openrouter_responses_terminal_reports_its_cost() {
    let completed = |byok: bool| {
        json!({
            "type": "response.completed",
            "response": {"status": "completed", "usage": {
                "input_tokens": 12, "output_tokens": 3, "total_tokens": 15,
                "cost": 0.000045, "is_byok": byok,
            }},
        })
    };
    for (reports_cost, byok, expected) in [
        (true, false, Some(45_000)),
        (true, true, None),
        (false, false, None),
    ] {
        let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
        normalizer.set_provider_reported_cost(reports_cost);
        let events = normalizer
            .feed(&SseEvent {
                event: Some("response.completed".to_string()),
                data: completed(byok).to_string(),
            })
            .expect("valid terminal");
        let usage = events
            .into_iter()
            .find_map(|event| match event {
                Event::Usage(usage) => Some(usage),
                _ => None,
            })
            .expect("terminal usage");
        assert_eq!(usage.input_tokens, Some(12));
        assert_eq!(usage.provider_reported_cost_nano_usd, expected);
    }
}
