//! Usage-mapper and count-parsing regressions for `events.rs`.

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
        namespace: None,
        caller: None,
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
    let usage = openai_compatible_usage(&json!({"prompt_tokens": 7})).expect("valid usage");
    assert_eq!(usage.input_tokens, Some(7));
    assert_eq!(usage.output_tokens, Some(0));
    assert_eq!(usage.cached_input_tokens, None);
    assert_eq!(usage.reasoning_tokens, None);
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
    assert_eq!(shown.concat(), "{\n  \"a\": [1, 2],\n  \"b\": {}\n}");
    assert_eq!(tool.withheld_tail, "\n");
    assert_eq!(
        redundant_tail("{}", " \n\t"),
        Some(RedundantTail::Whitespace)
    );
    tool.complete()
        .expect("whitespace after the object is not content");
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
