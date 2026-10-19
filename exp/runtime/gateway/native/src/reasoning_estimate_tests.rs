use serde_json::json;

use super::*;
use crate::events::openai_compatible_usage;

fn usage(output: u64, reasoning: Option<u64>) -> Usage {
    Usage {
        input_tokens: Some(48),
        output_tokens: Some(output),
        reasoning_tokens: reasoning,
        ..Usage::default()
    }
}

fn thinking_then_answer(answer: &str) -> Vec<Event> {
    vec![
        Event::ThinkingDelta {
            index: 0,
            delta: "1001 = 7 x 143, so it is composite.".into(),
        },
        Event::TextDelta(answer.into()),
    ]
}

#[test]
fn a_thinking_turn_without_a_count_estimates_output_less_the_visible_answer() {
    // "No.\n\n1001 = 7 × 143" is 20 bytes (× is two), ceil(20 / 4) = 5 tokens, the same
    // one-line answer OpenRouter reported 240 of 259 for on Claude Haiku 4.5.
    let estimate = ReasoningEstimate::from_events(&thinking_then_answer("No.\n\n1001 = 7 × 143"));
    assert_eq!(estimate.estimate(&usage(259, None)), Some(254));
}

#[test]
fn tool_arguments_and_refusals_count_as_visible_output() {
    let mut events = thinking_then_answer("");
    events.push(Event::ToolArgumentsDelta {
        index: 0,
        delta: "{\"path\":\"a\"}".into(),
    });
    events.push(Event::RefusalDelta("no".into()));
    // 12 + 2 bytes -> ceil(14 / 4) = 4 tokens.
    assert_eq!(
        ReasoningEstimate::from_events(&events).estimate(&usage(10, None)),
        Some(6)
    );
}

#[test]
fn signed_or_redacted_thinking_without_text_still_counts_as_thinking() {
    let events = vec![
        Event::RedactedThinking {
            index: 0,
            data: "opaque".into(),
        },
        Event::TextDelta("ok".into()),
    ];
    assert_eq!(
        ReasoningEstimate::from_events(&events).estimate(&usage(40, None)),
        Some(39)
    );
}

#[test]
fn no_estimate_without_thinking_with_a_reported_count_or_without_an_output_total() {
    let plain = ReasoningEstimate::from_events(&[Event::TextDelta("ready".into())]);
    assert_eq!(plain.estimate(&usage(5, None)), None);

    let thought = ReasoningEstimate::from_events(&thinking_then_answer("ok"));
    // A provider count is never replaced.
    assert_eq!(thought.estimate(&usage(40, Some(12))), None);
    let unknown_output = Usage {
        output_tokens: None,
        ..usage(0, None)
    };
    assert_eq!(thought.estimate(&unknown_output), None);
    // A visible answer longer than the output total floors at zero.
    let long = ReasoningEstimate::from_events(&thinking_then_answer(&"x".repeat(400)));
    assert_eq!(long.estimate(&usage(20, None)), Some(0));
}

#[test]
fn annotate_moves_reasoning_from_unreported_to_estimated() {
    let estimate = ReasoningEstimate::from_events(&thinking_then_answer("No."));
    let reported = usage(30, None);
    let mut wire = json!({
        "prompt_tokens": 48,
        "completion_tokens": 30,
        "completion_tokens_details": {"reasoning_tokens": 0},
        "unreported_token_details": ["cached_tokens", "reasoning_tokens"],
    });
    estimate.annotate(&mut wire, Some(&reported), "completion_tokens_details");
    assert_eq!(wire["completion_tokens_details"]["reasoning_tokens"], 29);
    assert_eq!(wire["unreported_token_details"], json!(["cached_tokens"]));
    assert_eq!(wire["estimated_token_details"], json!(["reasoning_tokens"]));

    // A null details object (the non-streaming Chat shape) becomes one.
    let mut null_details = json!({"completion_tokens": 30, "completion_tokens_details": null});
    estimate.annotate(
        &mut null_details,
        Some(&reported),
        "completion_tokens_details",
    );
    assert_eq!(
        null_details["completion_tokens_details"],
        json!({"reasoning_tokens": 29})
    );
    assert!(null_details.get("unreported_token_details").is_none());

    // A reported count leaves the wire byte-identical.
    let mut known = json!({"completion_tokens_details": {"reasoning_tokens": 7}});
    let before = known.clone();
    estimate.annotate(
        &mut known,
        Some(&usage(30, Some(7))),
        "completion_tokens_details",
    );
    assert_eq!(known, before);
}

#[test]
fn an_upstream_estimate_is_never_parsed_as_a_billed_reasoning_subset() {
    let parsed = openai_compatible_usage(&json!({
        "prompt_tokens": 48,
        "completion_tokens": 259,
        "total_tokens": 307,
        "completion_tokens_details": {"reasoning_tokens": 253},
        "estimated_token_details": ["reasoning_tokens"],
    }))
    .expect("an estimated count parses");
    assert_eq!(parsed.output_tokens, Some(259));
    assert_eq!(parsed.reasoning_tokens, None);

    let foreign = openai_compatible_usage(&json!({
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "estimated_token_details": ["cached_tokens"],
    }));
    assert!(foreign.is_err());
}

#[test]
fn bedrock_summarized_thinking_estimates_like_the_messages_wire() {
    // Bedrock Converse streams Claude's summarized thinking as reasoningContent
    // text (normalized to ReasoningTextDelta) and reports only outputTokens, so
    // the estimate is the same output-less-visible count the Messages wire gets
    // (live 2026-10-19, us.anthropic.claude-opus-4-8 at high: 294 output tokens).
    let events = vec![
        Event::ReasoningTextDelta("Head start 60 km, closing 24 km/h, so 2.5 h.".into()),
        Event::TextDelta("6:35pm, 240 km out".into()),
    ];
    // 18 bytes -> ceil(18 / 4) = 5 visible tokens.
    assert_eq!(
        ReasoningEstimate::from_events(&events).estimate(&usage(294, None)),
        Some(289)
    );
}

#[test]
fn a_zero_published_beside_readable_thinking_is_estimated() {
    // A chat-completions passthrough of Claude bills thinking inside
    // `completion_tokens` and may fill `reasoning_tokens: 0` it never measured.
    let readable = ReasoningEstimate::from_events(&thinking_then_answer("No."));
    assert_eq!(readable.estimate(&usage(30, Some(0))), Some(29));
    let mut wire =
        json!({"completion_tokens": 30, "completion_tokens_details": {"reasoning_tokens": 0}});
    readable.annotate(
        &mut wire,
        Some(&usage(30, Some(0))),
        "completion_tokens_details",
    );
    assert_eq!(wire["completion_tokens_details"]["reasoning_tokens"], 29);
    assert_eq!(wire["estimated_token_details"], json!(["reasoning_tokens"]));

    // A zero beside signature-only thinking, or no thinking, stands.
    let signed = ReasoningEstimate::from_events(&[
        Event::ThinkingSignature {
            index: 0,
            signature: "sig".into(),
        },
        Event::TextDelta("ok".into()),
    ]);
    assert_eq!(signed.estimate(&usage(30, Some(0))), None);
    let plain = ReasoningEstimate::from_events(&[Event::TextDelta("ok".into())]);
    assert_eq!(plain.estimate(&usage(30, Some(0))), None);
    // Empty reasoning deltas are not readable thinking.
    let empty = ReasoningEstimate::from_events(&[
        Event::ReasoningTextDelta(String::new()),
        Event::TextDelta("ok".into()),
    ]);
    assert_eq!(empty.estimate(&usage(30, Some(0))), None);
}

#[test]
fn novita_claude_reasoning_content_surfaces_as_reasoning_text_and_estimates() {
    // The Novita Claude passthrough (pa/claude-opus-4-8-hs) streams the
    // summarized thinking as `reasoning_content` deltas when it is asked with
    // Anthropic's adaptive thinking fields, then the answer as `content`.
    use crate::dialects::{Dialect, Normalizer};
    use crate::sse::SseEvent;

    let frames = [
        json!({"choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "Two trains: the gap closes at 24 km/h."}}]}),
        json!({"choices": [{"index": 0, "delta": {"reasoning_content": " 60 km takes 2.5 h."}}]}),
        json!({"choices": [{"index": 0, "delta": {"content": "6:35pm"}}]}),
        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        json!({"choices": [], "usage": {"prompt_tokens": 48, "completion_tokens": 289, "total_tokens": 337, "completion_tokens_details": {"reasoning_tokens": 0}}}),
    ];
    let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
    let mut events = Vec::new();
    for frame in frames {
        events.extend(
            normalizer
                .feed(&SseEvent {
                    event: None,
                    data: frame.to_string(),
                })
                .expect("Novita frame decodes"),
        );
    }
    events.extend(
        normalizer
            .feed(&SseEvent {
                event: None,
                data: "[DONE]".into(),
            })
            .expect("terminal decodes"),
    );
    let reasoning: String = events
        .iter()
        .filter_map(|event| match event {
            Event::ReasoningTextDelta(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        reasoning,
        "Two trains: the gap closes at 24 km/h. 60 km takes 2.5 h."
    );
    assert_eq!(
        crate::reasoning_display::flattened(&events, false).as_deref(),
        Some("Two trains: the gap closes at 24 km/h. 60 km takes 2.5 h.")
    );
    let usage = normalizer.observed_usage().expect("usage decoded");
    assert_eq!(usage.output_tokens, Some(289));
    // "6:35pm" is 6 bytes -> 2 visible tokens.
    assert_eq!(
        ReasoningEstimate::from_events(&events).estimate(usage),
        Some(287)
    );
}
