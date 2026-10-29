//! Usage-mapper and count-parsing regressions for `events.rs`.

use super::*;
use serde_json::json;

#[test]
fn coherent_additive_sample_corrects_provisional_subset_accounting() {
    check_coherent_reasoning_sample(true);
}

#[test]
fn coherent_subset_sample_corrects_provisional_additive_accounting() {
    check_coherent_reasoning_sample(false);
}

fn check_coherent_reasoning_sample(additive: bool) {
    for chat in [false, true] {
        let (input, output, details) = if chat {
            (
                "prompt_tokens",
                "completion_tokens",
                "completion_tokens_details",
            )
        } else {
            ("input_tokens", "output_tokens", "output_tokens_details")
        };
        let mut accumulator = OpenAiUsageAccumulator::default();
        for report in [
            json!({input: 100, output: 10, "total_tokens": if additive { 110 } else { 115 }}),
            json!({details: {"reasoning_tokens": 5}}),
        ] {
            if chat {
                accumulator.update_chat(&report).unwrap();
            } else {
                accumulator.update_responses(Some(&report)).unwrap();
            }
        }
        let final_report = json!({
            input: if additive { 100 } else { 105 }, output: 10,
            details: {"reasoning_tokens": 5}, "total_tokens": 115
        });
        let observed = if chat {
            accumulator.update_chat(&final_report).unwrap()
        } else {
            accumulator
                .update_responses(Some(&final_report))
                .unwrap()
                .unwrap()
        };
        assert_eq!(observed.output_tokens, Some(if additive { 15 } else { 10 }));
        assert_eq!(observed.reasoning_tokens, Some(5));
    }
}

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
        custom: false,
        namespace: None,
        caller: None,
        index: 0,
        call_id: "call_1".to_string(),
        name: "get".to_string(),
        thought_signature: None,
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
fn openai_compatible_usage_preserves_unknown_primary_counts() {
    let usage = openai_compatible_usage(&json!({"prompt_tokens": 7})).expect("valid usage");
    assert_eq!(usage.input_tokens, Some(7));
    assert_eq!(usage.output_tokens, None);
    assert_eq!(usage.cached_input_tokens, None);
    assert_eq!(usage.reasoning_tokens, None);
}

#[test]
fn openai_and_bedrock_usage_preserve_primary_absence_and_explicit_zero() {
    for usage in [
        openai_usage(Some(&json!({}))).unwrap().unwrap(),
        openai_compatible_usage(&json!({})).unwrap(),
        bedrock_usage(Some(&json!({}))).unwrap(),
    ] {
        assert_eq!(usage.input_tokens, None);
        assert_eq!(usage.output_tokens, None);
    }
    for usage in [
        openai_usage(Some(&json!({"output_tokens":0})))
            .unwrap()
            .unwrap(),
        openai_compatible_usage(&json!({"completion_tokens":0})).unwrap(),
        bedrock_usage(Some(&json!({"outputTokens":0}))).unwrap(),
    ] {
        assert_eq!(usage.input_tokens, None);
        assert_eq!(usage.output_tokens, Some(0));
    }
    assert_eq!(
        bedrock_usage(Some(&json!({"cacheReadInputTokens":5})))
            .unwrap()
            .input_tokens,
        None
    );
    assert_eq!(
        openai_usage(Some(
            &json!({"input_tokens_details":{"cache_write_tokens":5}})
        ))
        .unwrap()
        .unwrap()
        .input_tokens,
        None
    );
}

#[test]
fn observed_usage_merge_is_cumulative_and_clears_unknown_cache_ttl() {
    let mut usage = Usage {
        input_tokens: Some(13),
        output_tokens: Some(7),
        cache_creation_input_tokens: Some(5),
        cache_creation_1h_input_tokens: Some(3),
        ..Usage::default()
    };
    usage.merge_observed(&Usage::default());
    assert_eq!(usage.output_tokens, Some(7));
    assert_eq!(usage.cache_creation_1h_input_tokens, Some(3));
    usage.merge_observed(&Usage {
        output_tokens: Some(2),
        cache_creation_input_tokens: Some(6),
        ..Usage::default()
    });
    assert_eq!(usage.output_tokens, Some(7));
    assert_eq!(usage.cache_creation_input_tokens, Some(6));
    assert_eq!(usage.cache_creation_1h_input_tokens, None);
}

#[test]
fn observed_ttl_only_tracks_the_current_covering_write_total() {
    let mut usage = Usage {
        cache_creation_input_tokens: Some(20),
        ..Usage::default()
    };
    // A stale snapshot cannot restore a breakdown of only the old allocation.
    usage.merge_observed(&Usage {
        cache_creation_input_tokens: Some(10),
        cache_creation_1h_input_tokens: Some(5),
        ..Usage::default()
    });
    assert_eq!(usage.cache_creation_input_tokens, Some(20));
    assert_eq!(usage.cache_creation_1h_input_tokens, None);
    // An equal total can refine its previously unknown breakdown.
    usage.merge_observed(&Usage {
        cache_creation_input_tokens: Some(20),
        cache_creation_1h_input_tokens: Some(7),
        ..Usage::default()
    });
    usage.merge_observed(&Usage {
        cache_creation_input_tokens: Some(20),
        ..Usage::default()
    });
    assert_eq!(usage.cache_creation_1h_input_tokens, Some(7));
    // Neither a lower total nor its larger TTL subset revises current facts.
    usage.merge_observed(&Usage {
        cache_creation_input_tokens: Some(10),
        cache_creation_1h_input_tokens: Some(9),
        ..Usage::default()
    });
    assert_eq!(usage.cache_creation_1h_input_tokens, Some(7));
    usage.merge_observed(&Usage {
        cache_creation_input_tokens: Some(30),
        ..Usage::default()
    });
    assert_eq!(usage.cache_creation_1h_input_tokens, None);
    // A TTL-only refinement needs a known covering total and must fit it.
    for (total, hour, expected) in [
        (Some(30), 12, Some(12)),
        (Some(30), 31, None),
        (None, 12, None),
    ] {
        let mut partial = Usage {
            cache_creation_input_tokens: total,
            ..Usage::default()
        };
        partial.merge_observed(&Usage {
            cache_creation_1h_input_tokens: Some(hour),
            ..Usage::default()
        });
        assert_eq!(partial.cache_creation_1h_input_tokens, expected);
    }
}

#[test]
fn openai_compatible_usage_rejects_malformed_counts() {
    assert!(openai_compatible_usage(&json!({"prompt_tokens": "7"})).is_err());
    assert!(openai_compatible_usage(&json!({"prompt_tokens": MAXIMUM_LEDGER_COUNT + 1})).is_err());
    assert!(openai_compatible_usage(&json!([1])).is_err());
    assert!(openai_compatible_usage(
        &json!({"prompt_tokens": 1, "completion_tokens": 1, "prompt_tokens_details": 3})
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
fn bedrock_usage_reports_the_one_hour_write_subset_from_cache_details() {
    let usage = |details: serde_json::Value| {
        bedrock_usage(Some(&json!({
            "inputTokens": 10, "outputTokens": 2, "cacheWriteInputTokens": 30,
            "cacheDetails": details,
        })))
    };
    let complete = json!([{"ttl": "1h", "inputTokens": 20}, {"ttl": "5m", "inputTokens": 10}]);
    assert_eq!(
        usage(complete).unwrap().cache_creation_1h_input_tokens,
        Some(20)
    );
    assert_eq!(
        usage(json!([{"ttl": "5m", "inputTokens": 30}]))
            .unwrap()
            .cache_creation_1h_input_tokens,
        Some(0)
    );
    for unknown in [
        json!(null),
        json!([]),
        json!([{"ttl": "1h", "inputTokens": 20}]),
        json!([{"ttl": "24h", "inputTokens": 30}]),
    ] {
        assert_eq!(usage(unknown).unwrap().cache_creation_1h_input_tokens, None);
    }
    for malformed in [
        json!({"ttl": "1h"}),
        json!([{"ttl": "1h"}]),
        json!([{"ttl": "1h", "inputTokens": 31}]),
    ] {
        assert!(usage(malformed).is_err());
    }
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
    assert_eq!(usage.cache_creation_input_tokens, Some(1));
    assert_eq!(usage.cache_creation_1h_input_tokens, None);
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
fn openai_responses_usage_folds_reasoning_only_when_the_total_shows_it_additive() {
    // OpenAI Responses: output_tokens_details.reasoning_tokens is a subset of
    // output_tokens (total_tokens = input + output), so the counts pass through.
    let subset = openai_usage(Some(&json!({
        "input_tokens": 36,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 700,
        "output_tokens_details": {"reasoning_tokens": 690},
        "total_tokens": 736,
    })))
    .expect("valid usage")
    .expect("usage present");
    assert_eq!(subset.output_tokens, Some(700));
    assert_eq!(subset.reasoning_tokens, Some(690));
    // xAI's documented Responses usage (docs.x.ai) reports reasoning outside
    // output_tokens, and its total_tokens (32 + 9 + 110 = 151) says so.
    let additive = openai_usage(Some(&json!({
        "input_tokens": 32,
        "output_tokens": 9,
        "output_tokens_details": {"reasoning_tokens": 110},
        "total_tokens": 151,
    })))
    .expect("valid usage")
    .expect("usage present");
    assert_eq!(additive.output_tokens, Some(119));
    assert_eq!(additive.reasoning_tokens, Some(110));
}

#[test]
fn openai_compatible_usage_leaves_a_folded_provider_untouched() {
    // OpenAI-shaped Chat Completions usage (reasoning inside completion_tokens)
    // is forwarded exactly; the equal case is a legal subset, not evidence of
    // an additive provider.
    for (completion, reasoning) in [(700u64, 690u64), (690, 690), (5, 0)] {
        let usage = openai_compatible_usage(&json!({
            "prompt_tokens": 36,
            "completion_tokens": completion,
            "total_tokens": 36 + completion,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        }))
        .expect("valid usage");
        assert_eq!(usage.output_tokens, Some(completion));
        assert_eq!(usage.reasoning_tokens, Some(reasoning));
    }
    // A total_tokens that names the subset shape is authoritative even when
    // the reasoning count exceeds the output total; settlement clamps that
    // provider inconsistency instead of the mapper inventing output.
    let subset_total = openai_compatible_usage(&json!({
        "prompt_tokens": 36,
        "completion_tokens": 5,
        "total_tokens": 41,
        "completion_tokens_details": {"reasoning_tokens": 9},
    }))
    .expect("valid usage");
    assert_eq!(subset_total.output_tokens, Some(5));
    assert_eq!(subset_total.reasoning_tokens, Some(9));
    // OpenRouter normalizes upstream reasoning into completion_tokens and
    // reports the subset alongside (shape captured from openrouter.ai).
    let openrouter = openai_compatible_usage(&json!({
        "prompt_tokens": 14,
        "completion_tokens": 543,
        "total_tokens": 557,
        "cost": 0.0016,
        "is_byok": false,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        "cost_details": {"upstream_inference_cost": null},
        "completion_tokens_details": {"reasoning_tokens": 480, "image_tokens": 0},
    }))
    .expect("valid usage");
    assert_eq!(openrouter.output_tokens, Some(543));
    assert_eq!(openrouter.reasoning_tokens, Some(480));
}

#[test]
fn openai_compatible_usage_folds_an_additive_provider_into_output_tokens() {
    // Verbatim usage from Azure Foundry grok-4.3 (silen-resource, 2026-09-03,
    // non-streaming): xAI reports reasoning OUTSIDE completion_tokens, and its
    // own total_tokens identifies the additive shape (14 + 7 + 1303 = 1324),
    // so the mapper folds it and the subset stays reported.
    let usage = openai_compatible_usage(&json!({
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
    }))
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
    // The total decides regardless of magnitude: a long answer after a short
    // think (reasoning below completion_tokens) still folds when total_tokens
    // names the additive shape (14 + 900 + 300 = 1214).
    let short_think = openai_compatible_usage(&json!({
        "prompt_tokens": 14,
        "completion_tokens": 900,
        "total_tokens": 1214,
        "completion_tokens_details": {"reasoning_tokens": 300},
    }))
    .expect("valid usage");
    assert_eq!(short_think.output_tokens, Some(1200));
    assert_eq!(short_think.reasoning_tokens, Some(300));
    // Without a decisive total, only a reasoning count above completion_tokens
    // proves the additive shape; a smaller one is indistinguishable from a
    // subset and passes through.
    let no_total_above = openai_compatible_usage(&json!({
        "prompt_tokens": 14,
        "completion_tokens": 8,
        "completion_tokens_details": {"reasoning_tokens": 655},
    }))
    .expect("valid usage");
    assert_eq!(no_total_above.output_tokens, Some(663));
    let no_total_below = openai_compatible_usage(&json!({
        "prompt_tokens": 14,
        "completion_tokens": 900,
        "completion_tokens_details": {"reasoning_tokens": 300},
    }))
    .expect("valid usage");
    assert_eq!(no_total_below.output_tokens, Some(900));
    // A fold whose total leaves the persistable range is a contract violation.
    assert!(openai_compatible_usage(&json!({
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "completion_tokens_details": {"reasoning_tokens": MAXIMUM_LEDGER_COUNT},
    }))
    .is_err());
}

#[test]
fn openai_compatible_usage_reads_a_top_level_reasoning_count() {
    // Verbatim streamed usage from Azure Foundry DeepSeek-V4.1-Flash
    // (2026-10-11, reasoning_effort high): the reasoning count sits at the
    // top level of usage, inside completion_tokens (40 + 20 = 60).
    let usage = openai_compatible_usage(&json!({
        "total_tokens": 60,
        "completion_tokens": 20,
        "prompt_tokens": 40,
        "prompt_tokens_details": null,
        "reasoning_tokens": 18,
    }))
    .expect("valid usage");
    assert_eq!(usage.output_tokens, Some(20));
    assert_eq!(usage.reasoning_tokens, Some(18));
    // The nested OpenAI placement wins when a wire carries both.
    let both = openai_compatible_usage(&json!({
        "prompt_tokens": 40,
        "completion_tokens": 20,
        "total_tokens": 60,
        "completion_tokens_details": {"reasoning_tokens": 12},
        "reasoning_tokens": 18,
    }))
    .expect("valid usage");
    assert_eq!(both.reasoning_tokens, Some(12));
    // A relay's unknown-meter marker keeps a zero placeholder unknown and
    // refuses a positive top-level count, as it does a nested one.
    let unreported = openai_compatible_usage(&json!({
        "prompt_tokens": 40,
        "completion_tokens": 20,
        "reasoning_tokens": 0,
        "unreported_token_details": ["reasoning_tokens"],
    }))
    .expect("valid usage");
    assert_eq!(unreported.reasoning_tokens, None);
    assert!(openai_compatible_usage(&json!({
        "prompt_tokens": 40,
        "completion_tokens": 20,
        "reasoning_tokens": 18,
        "unreported_token_details": ["reasoning_tokens"],
    }))
    .is_err());
    // A malformed top-level count is a contract violation, as a nested one is.
    assert!(openai_compatible_usage(&json!({
        "prompt_tokens": 40,
        "completion_tokens": 20,
        "reasoning_tokens": "18",
    }))
    .is_err());
    // Responses usage keeps reading only output_tokens_details.
    let responses = openai_usage(Some(&json!({
        "input_tokens": 40,
        "output_tokens": 20,
        "reasoning_tokens": 18,
    })))
    .expect("valid usage")
    .expect("usage present");
    assert_eq!(responses.reasoning_tokens, None);
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
    // thoughtsTokenCount at all, so proto3's implicit scalar value is zero and
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
    assert_eq!(no_thinking.reasoning_tokens, Some(0));

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
fn tool_id_bound_counts_characters_and_preserves_opaque_signatures() {
    for id in [
        format!("toolu_synthetic~sig1:{}", "YWJj+/=".repeat(110)),
        format!("toolu_synthetic_sig1_{}", "YWJj+/=".repeat(110)),
        "é".repeat(65_536),
    ] {
        let mut tool = ToolAccumulator::new(id.clone(), "terminal".into());
        tool.raw_arguments = "{}".into();
        assert_eq!(tool.complete().expect("bounded opaque id").call_id, id);
    }
    for id in [String::new(), "x".repeat(65_537)] {
        let mut tool = ToolAccumulator::new(id, "terminal".into());
        tool.raw_arguments = "{}".into();
        assert!(tool.complete().is_err());
    }
}

/// Push fragments through the hold-back path and return what a client would
/// have been shown.
fn push_all(tool: &mut ToolAccumulator, fragments: &[&str]) -> Vec<String> {
    fragments
        .iter()
        .filter_map(|fragment| tool.push_arguments(fragment))
        .collect()
}

#[test]
fn zero_argument_tail_of_empty_literals_is_dropped_after_the_object_closes() {
    // Azure Foundry's DeepSeek-V4-Flash shim (captured live 2026-09-10):
    // a zero-argument call streams `""` (arguments start), `{}`, then a
    // stray `""` delta. Verbatim concatenation is `{}""`, the exact
    // "trailing characters at line 1 column 3 (4 bytes)" seen on 222
    // production attempts in one day. The stray delta carries no argument
    // content, so it is withheld from the caller and the call completes.
    let mut tool = ToolAccumulator::new("call_1".into(), "view_agent_graph".into());
    let shown = push_all(&mut tool, &["", "{}", "\"\""]);
    // The empty opening delta is shown as before (unchanged wire behaviour);
    // only the bytes after the closing brace are withheld.
    assert_eq!(shown, vec![String::new(), "{}".to_string()]);
    assert_eq!(tool.withheld_tail, "\"\"");
    let call = tool.complete().expect("a content-free tail completes");
    assert_eq!(call.raw_arguments, "{}");
    // The same verdict for every empty literal, in any mix, with whitespace;
    // a bare `{}` after `{}` is first of all a duplicated whole value.
    for tail in ["[]", "\"\"{}", " \"\" \n[] {}"] {
        assert_eq!(
            redundant_tail("{}", tail),
            Some(RedundantTail::EmptyLiterals)
        );
    }
    assert_eq!(
        redundant_tail("{}", "{}"),
        Some(RedundantTail::DuplicateValue)
    );
}

#[test]
fn duplicated_whole_object_deltas_collapse_to_one_value() {
    // A delta re-sent whole (`{}{}`, or a non-empty object twice) adds no
    // information: the first copy is the call, the repetition is dropped.
    let mut tool = ToolAccumulator::new("call_1".into(), "lookup".into());
    assert_eq!(push_all(&mut tool, &["{}", "{}"]), vec!["{}".to_string()]);
    assert_eq!(tool.complete().expect("duplicate").raw_arguments, "{}");

    let mut tool = ToolAccumulator::new("call_2".into(), "lookup".into());
    // The repetition straddles a fragment boundary with the closing byte.
    let shown = push_all(&mut tool, &["{\"a\":", "1}{\"a\"", ":1}"]);
    assert_eq!(shown.concat(), "{\"a\":1}");
    assert_eq!(tool.withheld_tail, "{\"a\":1}");
    assert_eq!(
        tool.complete().expect("duplicate").raw_arguments,
        "{\"a\":1}"
    );
    assert_eq!(
        redundant_tail("{\"a\":1}", " {\"a\":1}\n{\"a\":1}"),
        Some(RedundantTail::DuplicateValue)
    );
}

#[test]
fn pretty_printed_arguments_with_a_trailing_newline_complete() {
    let mut tool = ToolAccumulator::new("call_1".into(), "lookup".into());
    let shown = push_all(
        &mut tool,
        &["{\n  \"a\": [1, 2],\n", "  \"b\": {}\n}", "\n"],
    );
    assert_eq!(shown.concat(), "{\n  \"a\": [1, 2],\n  \"b\": {}\n}\n");
    assert!(tool.withheld_tail.is_empty());
    assert_eq!(
        redundant_tail("{}", " \n\t"),
        Some(RedundantTail::Whitespace)
    );
    tool.complete()
        .expect("whitespace after the object is not content");
}

#[test]
fn valid_argument_whitespace_survives_every_fragment_boundary() {
    let original = " { \"n\": 1e2, \"path\": \"雪\" } \r\n\t";
    for split in (0..=original.len()).filter(|index| original.is_char_boundary(*index)) {
        let mut tool = ToolAccumulator::new("call".into(), "inspect".into());
        let shown = push_all(&mut tool, &[&original[..split], &original[split..]]);
        assert_eq!(shown.concat(), original);
        assert_eq!(tool.complete().unwrap().raw_arguments, original);
        assert!(tool.withheld_tail.is_empty());
    }
    let mut tool = ToolAccumulator::new("call".into(), "inspect".into());
    assert_eq!(push_all(&mut tool, &["{} ", "{}", " \n"]).concat(), "{} ");
    assert_eq!(tool.withheld_tail, "{} \n");
    assert_eq!(tool.complete().unwrap().raw_arguments, "{} ");
}

#[test]
fn a_tail_carrying_content_or_a_bare_suffix_stays_malformed() {
    // Anything whose removal would pick one parse over another fails closed
    // with the parse position of the bytes the provider actually sent.
    for (fragments, position) in [
        (vec!["{\"a\":1}", "{\"b\":2}"], "line 1 column 8"),
        // A bare `}` is also what a dropped inner delta leaves behind.
        (vec!["{\"a\":1}", "}"], "line 1 column 8"),
        (vec!["{}", "\"x\""], "line 1 column 3"),
        (vec!["{}", "null"], "line 1 column 3"),
        (vec!["{\"a\":1}", "\"\""], "line 1 column 8"),
    ] {
        let mut tool = ToolAccumulator::new("call_1".into(), "lookup".into());
        push_all(&mut tool, &fragments);
        let error = tool
            .complete()
            .expect_err("content after the object is malformed");
        assert!(
            error
                .starts_with("streamed tool arguments are not valid JSON: trailing characters at ")
                && error.contains(position),
            "{fragments:?} -> {error}"
        );
    }
    assert_eq!(redundant_tail("{\"a\":1}", "}"), None);
    assert_eq!(redundant_tail("{\"a\":1}", "\"\""), None);
    assert_eq!(redundant_tail("{}", "{\"a\":1}"), None);
}

#[test]
fn the_scan_ignores_structural_bytes_inside_strings_and_never_closes_a_scalar() {
    let mut tool = ToolAccumulator::new("call_1".into(), "terminal".into());
    let shown = push_all(
        &mut tool,
        &[
            "{\"command\":\"echo }\\\"{ ]\"",
            ", \"n\": [1, {\"x\": \"}\"}]}",
            "{}",
        ],
    );
    assert_eq!(
        shown.concat(),
        "{\"command\":\"echo }\\\"{ ]\", \"n\": [1, {\"x\": \"}\"}]}"
    );
    assert_eq!(tool.withheld_tail, "{}");
    // `{}` after a non-empty object is content-bearing ambiguity, not noise.
    assert!(tool.complete().is_err());

    // A top-level scalar never closes, so every byte is shown and the strict
    // object contract rejects it at completion exactly as before.
    let mut tool = ToolAccumulator::new("call_2".into(), "lookup".into());
    assert_eq!(
        push_all(&mut tool, &["\"just", " text\""]).concat(),
        "\"just text\""
    );
    assert!(tool.withheld_tail.is_empty());
    assert_eq!(
        tool.complete().expect_err("a string is not an object"),
        "streamed tool arguments must decode to an object"
    );

    // An open object stays open: no tail, and the usual EOF parse error.
    let mut tool = ToolAccumulator::new("call_3".into(), "lookup".into());
    push_all(&mut tool, &["{\"a\": [1, 2"]);
    assert!(tool.withheld_tail.is_empty());
    assert!(tool.complete().is_err());
}

#[test]
fn custom_tool_input_passes_through_the_hold_back_untouched() {
    let mut tool = ToolAccumulator::new("call_1".into(), "shell".into());
    tool.custom = true;
    let shown = push_all(&mut tool, &["{}", "\"\"", " ls -la"]);
    assert_eq!(shown.concat(), "{}\"\" ls -la");
    assert!(tool.withheld_tail.is_empty());
    assert_eq!(
        tool.complete().expect("freeform").raw_arguments,
        "{}\"\" ls -la"
    );
}

#[test]
fn openai_cache_write_subsets_survive_normalization() {
    let chat = openai_compatible_usage(&json!({
        "prompt_tokens": 105, "completion_tokens": 3,
        "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 100}
    }))
    .expect("valid cache write");
    assert_eq!(chat.input_tokens, Some(105));
    assert_eq!(chat.cache_creation_input_tokens, Some(100));
    let responses = openai_usage(Some(&json!({
        "input_tokens": 105, "output_tokens": 3,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 100}
    })))
    .expect("valid cache write")
    .expect("usage");
    assert_eq!(responses.input_tokens, chat.input_tokens);
    assert_eq!(responses.cache_creation_input_tokens, Some(100));
    assert!(openai_compatible_usage(&json!({
        "prompt_tokens": 105, "prompt_tokens_details": {"cache_write_tokens": -1}
    }))
    .is_err());
}

#[test]
fn cache_subsets_above_openai_total_input_raise_the_input_total() {
    // Reads plus writes above the reported input keep every reported cache
    // token in its own leg and raise the input total to cover them, with no
    // fresh remainder, so nothing the provider reported is dropped.
    for (reads, writes, settled) in [(0, 11, 11), (6, 5, 11), (11, 0, 11)] {
        let chat = openai_compatible_usage(&json!({
            "prompt_tokens": 10, "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": reads, "cache_write_tokens": writes}
        }))
        .expect("an overlapping report settles");
        let responses = openai_usage(Some(&json!({
            "input_tokens": 10, "output_tokens": 1,
            "input_tokens_details": {"cached_tokens": reads, "cache_write_tokens": writes}
        })))
        .expect("an overlapping report settles")
        .expect("usage");
        for usage in [chat, responses] {
            assert_eq!(usage.input_tokens, Some(settled));
            assert_eq!(usage.cached_input_tokens, Some(reads));
            assert_eq!(usage.cache_creation_input_tokens, Some(writes));
            assert_eq!(usage.output_tokens, Some(1));
        }
    }
    let usage = openai_compatible_usage(&json!({
        "prompt_tokens": 10, "completion_tokens": 1,
        "prompt_tokens_details": {"cached_tokens": 6, "cache_write_tokens": 4}
    }))
    .unwrap();
    assert_eq!(usage.input_tokens, Some(10));
    assert_eq!(usage.cached_input_tokens, Some(6));
    assert_eq!(usage.cache_creation_input_tokens, Some(4));
    // An unpersistable subset sum stays a contract violation.
    assert!(openai_compatible_usage(&json!({
        "prompt_tokens": 10, "completion_tokens": 1,
        "prompt_tokens_details": {"cached_tokens": i64::MAX as u64, "cache_write_tokens": 1}
    }))
    .is_err());
}

#[test]
fn novita_kimi_cache_legs_above_the_prompt_total_settle_instead_of_failing() {
    // The Moonshot-style Kimi report Novita relays: `prompt_tokens` is
    // inclusive of both cache legs, and on some prompts the read and write
    // pages together overshoot it (kimi-k3, 2026-10-22, 37 malformed
    // attempts). The generation is finished and billed, so it settles.
    let usage = openai_compatible_usage(&json!({
        "prompt_tokens": 7588, "completion_tokens": 320, "total_tokens": 7908,
        "prompt_tokens_details": {"cached_tokens": 7398, "cache_write_tokens": 462},
        "completion_tokens_details": {"reasoning_tokens": 180}
    }))
    .expect("a finished generation settles");
    assert_eq!(usage.input_tokens, Some(7860));
    assert_eq!(usage.cached_input_tokens, Some(7398));
    assert_eq!(usage.cache_creation_input_tokens, Some(462));
    assert_eq!(usage.output_tokens, Some(320));
    assert_eq!(usage.reasoning_tokens, Some(180));
    // The ordinary inclusive Novita shape is untouched.
    let usage = openai_compatible_usage(&json!({
        "prompt_tokens": 2259, "completion_tokens": 134, "total_tokens": 2393,
        "prompt_tokens_details": {"cached_tokens": 974, "cache_write_tokens": 1230}
    }))
    .unwrap();
    assert_eq!(usage.input_tokens, Some(2259));
    assert_eq!(usage.cached_input_tokens, Some(974));
    assert_eq!(usage.cache_creation_input_tokens, Some(1230));
}

#[test]
fn image_output_tokens_are_read_from_gemini_modalities_and_openrouter_details() {
    // Gemini: the IMAGE modality inside candidatesTokenCount (gemini-3.1-flash-image, live).
    let gemini = gemini_usage(&json!({
        "promptTokenCount": 17,
        "candidatesTokenCount": 1278,
        "candidatesTokensDetails": [
            {"modality": "TEXT", "tokenCount": 158},
            {"modality": "IMAGE", "tokenCount": 1120},
        ],
    }))
    .unwrap();
    assert_eq!(gemini.output_tokens, Some(1278));
    assert_eq!(gemini.image_output_tokens, Some(1120));
    // No breakdown leaves the subset unknown; a text-only breakdown is a real 0.
    let unreported = json!({"promptTokenCount": 3, "candidatesTokenCount": 4});
    assert_eq!(gemini_usage(&unreported).unwrap().image_output_tokens, None);
    let text = json!({"candidatesTokenCount": 4, "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 4}]});
    assert_eq!(gemini_usage(&text).unwrap().image_output_tokens, Some(0));
    // A malformed breakdown fails the meter.
    assert!(gemini_usage(&json!({"candidatesTokensDetails": "IMAGE"})).is_err());

    // OpenRouter: completion_tokens_details.image_tokens; a reported 0 is a
    // real count (billed at the text rate), an absent key is unknown.
    let routed = openai_compatible_usage(&json!({
        "prompt_tokens": 10,
        "completion_tokens": 1300,
        "completion_tokens_details": {"reasoning_tokens": 0, "image_tokens": 1290},
    }))
    .unwrap();
    assert_eq!(routed.image_output_tokens, Some(1290));
    let text_only = openai_compatible_usage(&json!({
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "completion_tokens_details": {"image_tokens": 0},
    }))
    .unwrap();
    assert_eq!(text_only.image_output_tokens, Some(0));
    let absent =
        openai_compatible_usage(&json!({"prompt_tokens": 10, "completion_tokens": 3})).unwrap();
    assert_eq!(absent.image_output_tokens, None);
}

#[test]
fn an_image_reported_outside_the_gemini_totals_joins_the_output() {
    // Gemini 3 Pro Image: IMAGE 1120 beside candidates 55 and thoughts 100.
    // The image is added so the subset stays inside the output and is billed.
    let usage = gemini_usage(&json!({
        "promptTokenCount": 9,
        "candidatesTokenCount": 55,
        "thoughtsTokenCount": 100,
        "candidatesTokensDetails": [{"modality": "IMAGE", "tokenCount": 1120}],
    }))
    .unwrap();
    assert_eq!(usage.output_tokens, Some(1275));
    assert_eq!(usage.image_output_tokens, Some(1120));
    assert_eq!(usage.reasoning_tokens, Some(100));
}
