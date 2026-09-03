//! Usage-mapper and count-parsing regressions for `events.rs` (moved out of
//! the module for its line budget).

use super::*;
use serde_json::json;

#[test]
fn output_tokens_lead_a_turn_but_control_frames_do_not() {
    // Content, reasoning, and tool-call deltas are the first visible output.
    assert!(Event::TextDelta("hi".to_string()).is_output_token());
    assert!(Event::RefusalDelta("no".to_string()).is_output_token());
    assert!(Event::ProviderTextDelta {
        output_index: 0,
        item_id: "msg_1".to_string(),
        delta: "hi".to_string(),
    }
    .is_output_token());
    assert!(Event::ThinkingDelta {
        index: 0,
        delta: "hmm".to_string(),
    }
    .is_output_token());
    // A tool-only turn's first token is the tool call itself.
    assert!(Event::ToolCallStarted {
        index: 0,
        call_id: "call_1".to_string(),
        name: "get".to_string(),
    }
    .is_output_token());
    // Usage, terminals, and opaque reasoning-carrier frames never lead.
    assert!(!Event::Usage(Usage::default()).is_output_token());
    assert!(!Event::Completed.is_output_token());
    assert!(!Event::Incomplete.is_output_token());
    assert!(!Event::ThinkingSignature {
        index: 0,
        signature: "sig".to_string(),
    }
    .is_output_token());
    // A Responses item-start reserves a slot before the first delta; it
    // must not stamp TTFT early -- the following delta is the real token.
    assert!(!Event::ProviderOutputItemStarted {
        output_index: 0,
        item_id: Some("msg_1".to_string()),
        kind: ProviderOutputItemKind::Message,
        status: None,
        phase: None,
    }
    .is_output_token());
    // An empty delta (role-establishing or empty refusal frame) carries no
    // visible token, so it must not stamp TTFT.
    assert!(!Event::TextDelta(String::new()).is_output_token());
    assert!(!Event::RefusalDelta(String::new()).is_output_token());
    assert!(!Event::ProviderTextDelta {
        output_index: 0,
        item_id: "msg_1".to_string(),
        delta: String::new(),
    }
    .is_output_token());
}

#[test]
fn openai_compatible_usage_counts_absent_fields_as_zero() {
    let usage = openai_compatible_usage(&json!({"prompt_tokens": 7}), false).expect("valid usage");
    assert_eq!(usage.input_tokens, Some(7));
    assert_eq!(usage.output_tokens, Some(0));
    assert_eq!(usage.cached_input_tokens, None);
    assert_eq!(usage.reasoning_tokens, None);
}

#[test]
fn openai_compatible_usage_rejects_malformed_counts() {
    assert!(openai_compatible_usage(&json!({"prompt_tokens": "7"}), false).is_err());
    assert!(
        openai_compatible_usage(&json!({"prompt_tokens": MAXIMUM_LEDGER_COUNT + 1}), false)
            .is_err()
    );
    assert!(openai_compatible_usage(&json!([1]), false).is_err());
    assert!(openai_compatible_usage(
        &json!({"prompt_tokens": 1, "completion_tokens": 1, "prompt_tokens_details": 3}),
        false
    )
    .is_err());
}

#[test]
fn parsed_counts_are_bounded_to_the_persistable_ledger_range() {
    let at_bound = json!({"count": MAXIMUM_LEDGER_COUNT});
    let over_bound = json!({"count": MAXIMUM_LEDGER_COUNT + 1});
    let at_object = at_bound.as_object().expect("object");
    let over_object = over_bound.as_object().expect("object");
    // Exactly i64::MAX is persistable and accepted; one past it is a
    // provider contract violation everywhere counts are parsed.
    assert_eq!(
        count_or_zero(at_object, "count", "count"),
        Ok(MAXIMUM_LEDGER_COUNT)
    );
    assert!(count_or_zero(over_object, "count", "count").is_err());
    assert_eq!(
        require_u64(at_object, "count", "count"),
        Ok(MAXIMUM_LEDGER_COUNT)
    );
    assert!(require_u64(over_object, "count", "count").is_err());
    assert!(openai_usage(Some(&json!({
        "input_tokens": 1,
        "output_tokens": 1,
        "output_tokens_details": {"reasoning_tokens": MAXIMUM_LEDGER_COUNT + 1},
    })))
    .is_err());
}

#[test]
fn bedrock_usage_folds_cache_legs_and_rejects_unrepresentable_totals() {
    let usage = bedrock_usage(Some(&json!({
        "inputTokens": 9,
        "outputTokens": 4,
        "cacheReadInputTokens": 2,
        "cacheWriteInputTokens": 1,
    })))
    .expect("valid usage");
    assert_eq!(usage.input_tokens, Some(12));
    assert_eq!(usage.cached_input_tokens, Some(2));
    // A leg beyond the persistable ledger range fails at the parser.
    assert!(bedrock_usage(Some(&json!({
        "inputTokens": MAXIMUM_LEDGER_COUNT + 1,
        "outputTokens": 1,
    })))
    .is_err());
    // Individually persistable legs whose folded total is not are a
    // provider contract violation, never a clamped or wrapped total.
    assert!(bedrock_usage(Some(&json!({
        "inputTokens": MAXIMUM_LEDGER_COUNT,
        "outputTokens": 1,
        "cacheReadInputTokens": 1,
    })))
    .is_err());
    assert!(bedrock_usage(Some(&json!(null))).is_err());
    assert!(bedrock_usage(None).is_err());
}

#[test]
fn openai_usage_treats_absent_and_null_objects_as_unknown() {
    assert!(openai_usage(None).expect("absent is unknown").is_none());
    assert!(openai_usage(Some(&serde_json::Value::Null))
        .expect("null is unknown")
        .is_none());
    let usage = openai_usage(Some(&json!({
        "input_tokens": 2,
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": 1},
    })))
    .expect("valid usage")
    .expect("usage present");
    assert_eq!(usage.reasoning_tokens, Some(1));
    assert_eq!(usage.cached_input_tokens, None);
}

// Usage-contract pins: every mapper emits reasoning_tokens as a SUBSET of
// output_tokens (see the module documentation), so settlement never prices a
// reasoning token at zero because a provider reported it outside its total.

#[test]
fn openai_responses_usage_forwards_the_documented_reasoning_subset_unchanged() {
    // OpenAI Responses: output_tokens_details.reasoning_tokens is a subset of
    // output_tokens by the provider's definition, so the counts pass through.
    let usage = openai_usage(Some(&json!({
        "input_tokens": 36,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 700,
        "output_tokens_details": {"reasoning_tokens": 690},
        "total_tokens": 736,
    })))
    .expect("valid usage")
    .expect("usage present");
    assert_eq!(usage.output_tokens, Some(700));
    assert_eq!(usage.reasoning_tokens, Some(690));
    assert!(usage.reasoning_tokens <= usage.output_tokens);
}

#[test]
fn openai_compatible_usage_leaves_a_folded_provider_untouched() {
    // OpenAI-shaped Chat Completions usage (reasoning inside completion_tokens)
    // is forwarded exactly; the equal case is a legal subset, not evidence of
    // an additive provider.
    for (completion, reasoning) in [(700u64, 690u64), (690, 690), (5, 0)] {
        let usage = openai_compatible_usage(
            &json!({
                "prompt_tokens": 36,
                "completion_tokens": completion,
                "total_tokens": 36 + completion,
                "completion_tokens_details": {"reasoning_tokens": reasoning},
            }),
            false,
        )
        .expect("valid usage");
        assert_eq!(usage.output_tokens, Some(completion));
        assert_eq!(usage.reasoning_tokens, Some(reasoning));
    }
    // OpenRouter normalizes upstream reasoning into completion_tokens and
    // reports the subset alongside (shape captured from openrouter.ai).
    let openrouter = openai_compatible_usage(
        &json!({
            "prompt_tokens": 14,
            "completion_tokens": 543,
            "total_tokens": 557,
            "cost": 0.0016,
            "is_byok": false,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "cost_details": {"upstream_inference_cost": null},
            "completion_tokens_details": {"reasoning_tokens": 480, "image_tokens": 0},
        }),
        false,
    )
    .expect("valid usage");
    assert_eq!(openrouter.output_tokens, Some(543));
    assert_eq!(openrouter.reasoning_tokens, Some(480));
}

#[test]
fn openai_compatible_usage_folds_an_additive_provider_into_output_tokens() {
    // Verbatim usage from Azure Foundry grok-4.3 (silen-resource, 2026-09-03,
    // non-streaming): xAI reports reasoning OUTSIDE completion_tokens, which
    // its own total_tokens confirms (14 + 7 + 1303 = 1324). A reasoning count
    // above completion_tokens is impossible under subset semantics, so the
    // mapper folds it and the subset stays reported.
    let usage = openai_compatible_usage(
        &json!({
            "prompt_tokens": 14,
            "completion_tokens": 7,
            "total_tokens": 1324,
            "audio_prompt_tokens": 0,
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "audio_tokens": 0,
                "text_tokens": 14,
                "image_tokens": 0,
            },
            "completion_tokens_details": {
                "reasoning_tokens": 1303,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "num_sources_used": 0,
        }),
        false,
    )
    .expect("valid usage");
    assert_eq!(usage.input_tokens, Some(14));
    assert_eq!(usage.output_tokens, Some(1310));
    assert_eq!(usage.reasoning_tokens, Some(1303));
    assert_eq!(usage.cached_input_tokens, Some(0));
    // The folded totals reproduce the provider's own total_tokens.
    assert_eq!(
        usage.input_tokens.unwrap() + usage.output_tokens.unwrap(),
        1324
    );
    // A fold whose total leaves the persistable range is a contract violation.
    assert!(openai_compatible_usage(
        &json!({
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "completion_tokens_details": {"reasoning_tokens": MAXIMUM_LEDGER_COUNT},
        }),
        false
    )
    .is_err());
}

#[test]
fn gemini_usage_folds_thoughts_into_output_tokens() {
    // Verbatim usageMetadata from gemini-3.7-flash (generateContent,
    // 2026-09-03), thinking on: Google's totalTokenCount is prompt +
    // candidates + thoughts (11 + 8 + 524 = 543), so thoughts are additive and
    // fold into output_tokens while reasoning_tokens names the subset.
    let thinking = gemini_usage(&json!({
        "promptTokenCount": 11,
        "candidatesTokenCount": 8,
        "totalTokenCount": 543,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 11}],
        "thoughtsTokenCount": 524,
        "serviceTier": "standard",
    }))
    .expect("valid usage");
    assert_eq!(thinking.input_tokens, Some(11));
    assert_eq!(thinking.output_tokens, Some(532));
    assert_eq!(thinking.reasoning_tokens, Some(524));
    assert_eq!(thinking.cached_input_tokens, Some(0));
    assert_eq!(
        thinking.input_tokens.unwrap() + thinking.output_tokens.unwrap(),
        543
    );

    // Verbatim from gemini-2.5-flash with thinkingBudget 0: no
    // thoughtsTokenCount at all, so the reasoning subset stays unknown and
    // output_tokens is the candidate count (11 + 9 = 20).
    let no_thinking = gemini_usage(&json!({
        "promptTokenCount": 11,
        "candidatesTokenCount": 9,
        "totalTokenCount": 20,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 11}],
        "serviceTier": "standard",
    }))
    .expect("valid usage");
    assert_eq!(no_thinking.output_tokens, Some(9));
    assert_eq!(no_thinking.reasoning_tokens, None);

    // Documented cached-content shape (cachedContentTokenCount plus
    // cacheTokensDetails): the cache leg stays an input subset while thoughts
    // still fold into output.
    let cached = gemini_usage(&json!({
        "promptTokenCount": 3582,
        "candidatesTokenCount": 7,
        "totalTokenCount": 3806,
        "cachedContentTokenCount": 3072,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 3582}],
        "cacheTokensDetails": [{"modality": "TEXT", "tokenCount": 3072}],
        "thoughtsTokenCount": 217,
    }))
    .expect("valid usage");
    assert_eq!(cached.input_tokens, Some(3582));
    assert_eq!(cached.cached_input_tokens, Some(3072));
    assert_eq!(cached.output_tokens, Some(224));
    assert_eq!(cached.reasoning_tokens, Some(217));

    // A fold whose total leaves the persistable range is a contract violation.
    assert!(gemini_usage(&json!({
        "promptTokenCount": 1,
        "candidatesTokenCount": MAXIMUM_LEDGER_COUNT,
        "thoughtsTokenCount": 1,
    }))
    .is_err());
}

#[test]
fn openai_compatible_usage_honours_the_deployment_declaration() {
    // A declared-additive deployment folds even when reasoning is no larger
    // than the visible answer (the case the heuristic alone cannot see), and
    // still has nothing to fold when the provider publishes no reasoning count.
    let usage = openai_compatible_usage(
        &json!({
            "prompt_tokens": 30,
            "completion_tokens": 40,
            "completion_tokens_details": {"reasoning_tokens": 5},
        }),
        true,
    )
    .expect("valid usage");
    assert_eq!(usage.output_tokens, Some(45));
    assert_eq!(usage.reasoning_tokens, Some(5));
    let undetailed =
        openai_compatible_usage(&json!({"prompt_tokens": 30, "completion_tokens": 40}), true)
            .expect("valid usage");
    assert_eq!(undetailed.output_tokens, Some(40));
    assert_eq!(undetailed.reasoning_tokens, None);
}
