//! Whole-sequence alignment of capture-only probabilities with emitted text.
use super::*;
use serde_json::json;

fn record(token: &str) -> serde_json::Value {
    json!({"token": token, "logprob": -1.0, "bytes": token.as_bytes(), "top_logprobs": []})
}

fn frame(reasoning: &str, content: &str, finishes: bool) -> FrameOutput {
    FrameOutput {
        content: content.into(),
        reasoning: reasoning.into(),
        finishes,
        ..Default::default()
    }
}

#[test]
fn a_reasoning_model_keeps_its_whole_token_sequence() {
    // vLLM with a reasoning parser: reasoning tokens' records on `content`,
    // the end-of-thinking delimiter in a frame of its own, then the answer,
    // then a stop token on the finishing frame.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("Think")]})),
        0,
        &frame("Think", "", false),
    );
    probabilities.retain(
        Some(&json!({"content":[record(" more")]})),
        0,
        &frame(" more", "", false),
    );
    probabilities.retain(
        Some(&json!({"content":[record("</think>")]})),
        0,
        &frame("", "", false),
    );
    probabilities.retain(
        Some(&json!({"content":[record("Hi")]})),
        0,
        &frame("", "Hi", false),
    );
    probabilities.retain(
        Some(&json!({"content":[record("!"), record("<|user|>")]})),
        0,
        &frame("", "!", true),
    );
    let (taken, truncated) = probabilities.take().unwrap();
    assert!(!truncated);
    let tokens: Vec<String> = taken
        .iter()
        .flat_map(|delta| delta.logprobs.as_ref().unwrap().content.clone().unwrap())
        .map(|record| record.token)
        .collect();
    assert_eq!(
        tokens,
        ["Think", " more", "</think>", "Hi", "!", "<|user|>"]
    );
}

#[test]
fn speculative_frames_spanning_reasoning_delimiter_and_answer_align() {
    // MTP / speculative decoding emits several tokens per frame.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("ok"), record("</think>"), record("Hel"), record("lo")]})),
        0,
        &frame("ok", "Hello", false),
    );
    let (taken, truncated) = probabilities.take().unwrap();
    assert_eq!(taken.len(), 1);
    assert!(!truncated);
}

#[test]
fn silent_tokens_anywhere_else_end_retention() {
    // A silent token after the answer started is a hole once text follows.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("Hi")]})),
        0,
        &frame("", "Hi", false),
    );
    probabilities.retain(
        Some(&json!({"content":[record("<x>")]})),
        0,
        &frame("", "", false),
    );
    probabilities.retain(
        Some(&json!({"content":[record("!")]})),
        0,
        &frame("", "!", false),
    );
    let (taken, truncated) = probabilities.take().unwrap();
    assert_eq!(taken.len(), 2);
    assert!(truncated);
    // One silent delimiter before the answer, not two.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("a"), record("</think>")]})),
        0,
        &frame("a", "", false),
    );
    probabilities.retain(
        Some(&json!({"content":[record("<y>"), record("b")]})),
        0,
        &frame("", "b", false),
    );
    let (taken, truncated) = probabilities.take().unwrap();
    assert_eq!(taken.len(), 1);
    assert!(truncated);
    // Records leading the text past the bound are a hole.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    let long = "x".repeat(MAXIMUM_RECORD_LEAD_BYTES + 1);
    probabilities.retain(
        Some(&json!({"content":[record(&long)]})),
        0,
        &frame("", "", false),
    );
    assert!(probabilities.take().unwrap().1);
    // Records beside the first reasoning that do not spell it are read as
    // leading the answer, and are a hole when the answer does not spell them.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("x"), record("y")]})),
        0,
        &frame("a", "", false),
    );
    probabilities.retain(
        Some(&json!({"content":[record("b")]})),
        0,
        &frame("", "b", true),
    );
    assert!(probabilities.take().unwrap().1);
}

#[test]
fn reasoning_without_records_still_keeps_the_answer() {
    // A provider that prices only the answer (records on answer frames only).
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(None, 0, &frame("thinking", "", false));
    probabilities.retain(
        Some(&json!({"content":[record("Hi")]})),
        0,
        &frame("", "Hi", true),
    );
    let (taken, truncated) = probabilities.take().unwrap();
    assert_eq!(taken.len(), 1);
    assert!(!truncated);
}

#[test]
fn tool_call_frames_with_records_stay_a_hole() {
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("{")]})),
        0,
        &FrameOutput {
            opaque: true,
            ..Default::default()
        },
    );
    assert!(probabilities.take().unwrap().1);
}

/// Replay one recorded vLLM stream (prompt "Say hi in 3 words.", `logprobs:
/// true` as injection sends it) through the
/// OpenAI-compatible normalizer in capture mode.
fn replay(stream: &str) -> (usize, bool, String) {
    let mut normalizer =
        crate::dialects::Normalizer::new(crate::dialects::Dialect::OpenAiCompatible);
    normalizer.enable_logprobs_capture();
    for line in stream.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data.starts_with('{') {
                normalizer
                    .feed(&crate::sse::SseEvent {
                        event: None,
                        data: data.to_owned(),
                    })
                    .unwrap();
            }
        }
    }
    let (deltas, truncated) = normalizer.take_captured_logprobs().unwrap();
    let tokens: String = deltas
        .iter()
        .flat_map(|delta| delta.logprobs.as_ref().unwrap().content.clone().unwrap())
        .map(|record| record.token)
        .collect();
    (deltas.len(), truncated, tokens)
}

#[test]
fn a_recorded_glm_mtp_stream_is_captured_whole() {
    let (frames, truncated, tokens) = replay(include_str!("fixtures/vllm_glm53_mtp_logprobs.sse"));
    assert!(!truncated);
    assert_eq!(frames, 3);
    assert_eq!(tokens, "</think>Hi there, friend!<|user|>");
}

#[test]
fn a_recorded_deepseek_stream_with_held_back_text_is_captured_whole() {
    // vLLM's incremental detokenizer emits some tokens' text a frame late, so
    // records run up to a few bytes ahead of the reasoning text.
    let (frames, truncated, tokens) =
        replay(include_str!("fixtures/vllm_dsv41_dspark_logprobs.sse"));
    assert!(!truncated);
    assert_eq!(frames, 21);
    assert!(tokens.starts_with("We need answer exactly"));
}

#[test]
fn an_answer_opening_with_a_prefix_of_the_delimiter_still_aligns() {
    // `<` is a strict prefix of `</think>`: the delimiter is still the silent
    // token, not a token whose text is only partly emitted.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("</think>")]})),
        0,
        &frame("", "", false),
    );
    probabilities.retain(
        Some(&json!({"content":[record("<")]})),
        0,
        &frame("", "<", false),
    );
    probabilities.retain(
        Some(&json!({"content":[record("div")]})),
        0,
        &frame("", "div", true),
    );
    let (taken, truncated) = probabilities.take().unwrap();
    assert_eq!(taken.len(), 3);
    assert!(!truncated);
}

#[test]
fn reasoning_records_ahead_of_their_text_price_the_reasoning() {
    // The first reasoning token's text is held back a frame: the records that
    // arrived first still mark the reasoning as priced.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("Th")]})),
        0,
        &frame("", "", false),
    );
    probabilities.retain(None, 0, &frame("Th", "", false));
    probabilities.retain(
        Some(&json!({"content":[record("</think>"), record("Hi")]})),
        0,
        &frame("", "Hi", true),
    );
    let (taken, truncated) = probabilities.take().unwrap();
    assert_eq!(taken.len(), 2);
    assert!(!truncated);
}

#[test]
fn reasoning_records_after_unpriced_reasoning_are_a_hole() {
    // The first reasoning frame carried no records, so the reasoning before a
    // later record was never matched: the sequence cannot be shown whole.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(None, 0, &frame("Think", "", false));
    probabilities.retain(
        Some(&json!({"content":[record(" more")]})),
        0,
        &frame(" more", "", false),
    );
    probabilities.retain(
        Some(&json!({"content":[record("Hi")]})),
        0,
        &frame("", "Hi", true),
    );
    let (taken, truncated) = probabilities.take().unwrap();
    assert!(taken.is_empty());
    assert!(truncated);
}

#[test]
fn records_waiting_for_text_before_a_tool_call_are_a_hole() {
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("Hi")]})),
        0,
        &frame("", "Hi", false),
    );
    probabilities.retain(
        Some(&json!({"content":[record("{\"")]})),
        0,
        &frame("", "", false),
    );
    probabilities.retain(
        None,
        0,
        &FrameOutput {
            opaque: true,
            ..Default::default()
        },
    );
    probabilities.retain(None, 0, &frame("", "", true));
    assert!(probabilities.take().unwrap().1);
}

#[test]
fn earlier_unmatched_records_at_an_empty_finish_are_a_hole() {
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("Hi")]})),
        0,
        &frame("", "Hi", false),
    );
    probabilities.retain(
        Some(&json!({"content":[record(" there")]})),
        0,
        &frame("", "", false),
    );
    probabilities.retain(None, 0, &frame("", "", true));
    let (taken, truncated) = probabilities.take().unwrap();
    assert_eq!(taken.len(), 2);
    assert!(truncated);
    // Stop tokens on the finishing frame itself stay silent.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("Hi")]})),
        0,
        &frame("", "Hi", false),
    );
    probabilities.retain(
        Some(&json!({"content":[record("<|user|>")]})),
        0,
        &frame("", "", true),
    );
    assert!(!probabilities.take().unwrap().1);
    // A delimiter no answer followed is still the one silent token.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("ok")]})),
        0,
        &frame("ok", "", false),
    );
    probabilities.retain(
        Some(&json!({"content":[record("</think>")]})),
        0,
        &frame("", "", false),
    );
    probabilities.retain(None, 0, &frame("", "", true));
    assert!(!probabilities.take().unwrap().1);
}

#[test]
fn typed_thinking_parts_are_aligned_as_reasoning() {
    let stream = concat!(
        r#"data: {"choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"Plan"}]}]},"logprobs":{"content":[{"token":"Plan","logprob":-1.0,"bytes":[80,108,97,110],"top_logprobs":[]}]}}]}"#,
        "\n",
        r#"data: {"choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":" it"}]},{"type":"text","text":"Hi"}]},"logprobs":{"content":[{"token":" it","logprob":-1.0,"bytes":[32,105,116],"top_logprobs":[]},{"token":"Hi","logprob":-1.0,"bytes":[72,105],"top_logprobs":[]}]},"finish_reason":"stop"}]}"#,
        "\n",
    );
    let (frames, truncated, tokens) = replay(stream);
    assert!(!truncated);
    assert_eq!(frames, 2);
    assert_eq!(tokens, "Plan itHi");
    // Thinking after answer text in one frame cannot be ordered: a hole.
    let stream = concat!(
        r#"data: {"choices":[{"index":0,"delta":{"content":[{"type":"text","text":"Hi"},{"type":"thinking","thinking":"x"}]},"logprobs":{"content":[{"token":"Hi","logprob":-1.0,"bytes":[72,105],"top_logprobs":[]},{"token":"x","logprob":-1.0,"bytes":[120],"top_logprobs":[]}]},"finish_reason":"stop"}]}"#,
        "\n",
    );
    let (frames, truncated, _) = replay(stream);
    assert_eq!(frames, 0);
    assert!(truncated);
}

#[test]
fn queued_records_after_unpriced_reasoning_are_a_hole() {
    // The record runs a frame ahead of its reasoning text.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(None, 0, &frame("Think", "", false));
    probabilities.retain(
        Some(&json!({"content":[record(" more")]})),
        0,
        &frame("", "", false),
    );
    probabilities.retain(None, 0, &frame(" more", "", false));
    probabilities.retain(
        Some(&json!({"content":[record("Hi")]})),
        0,
        &frame("", "Hi", true),
    );
    assert!(probabilities.take().unwrap().1);
}

#[test]
fn priced_reasoning_after_the_answer_is_a_hole() {
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("a"), record("Hi")]})),
        0,
        &frame("a", "Hi", false),
    );
    probabilities.retain(
        Some(&json!({"content":[record("b")]})),
        0,
        &frame("b", "", true),
    );
    let (taken, truncated) = probabilities.take().unwrap();
    assert_eq!(taken.len(), 1);
    assert!(truncated);
}

#[test]
fn frames_after_a_hole_are_not_aligned() {
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("x")]})),
        0,
        &frame("", "y", false),
    );
    for _ in 0..8 {
        probabilities.retain(
            Some(&json!({"content":[record(&"z".repeat(512))]})),
            0,
            &frame("", "w", false),
        );
    }
    let ChatProbabilities::Capture(buffer) = &probabilities else {
        unreachable!()
    };
    assert!(buffer.truncated);
    assert!(buffer
        .readings
        .0
        .iter()
        .all(|reading| reading.pending_bytes <= MAXIMUM_RECORD_LEAD_BYTES));
}

#[test]
fn a_zero_byte_record_is_a_silent_token_in_place() {
    let silent = json!({"token": "", "logprob": -1.0, "bytes": [], "top_logprobs": []});
    // After the answer started, a silent token off the finishing frame is a hole.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("Hi")]})),
        0,
        &frame("", "Hi", false),
    );
    probabilities.retain(
        Some(&json!({"content":[silent.clone()]})),
        0,
        &frame("", "", false),
    );
    probabilities.retain(
        Some(&json!({"content":[record("!")]})),
        0,
        &frame("", "!", true),
    );
    assert!(probabilities.take().unwrap().1);
    // Before the answer it is the delimiter; on the finishing frame, a stop token.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[silent.clone()]})),
        0,
        &frame("", "", false),
    );
    probabilities.retain(
        Some(&json!({"content":[record("Hi"), silent]})),
        0,
        &frame("", "Hi", true),
    );
    assert!(!probabilities.take().unwrap().1);
}

#[test]
fn a_token_partly_emitted_at_the_finish_is_a_hole() {
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("Hello")]})),
        0,
        &frame("", "Hel", true),
    );
    assert!(probabilities.take().unwrap().1);
}

#[test]
fn a_mixed_first_frame_with_answer_only_records_aligns() {
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("Hi")]})),
        0,
        &frame("thinking", "Hi", false),
    );
    probabilities.retain(
        Some(&json!({"content":[record("!")]})),
        0,
        &frame("", "!", true),
    );
    let (taken, truncated) = probabilities.take().unwrap();
    assert_eq!(taken.len(), 2);
    assert!(!truncated);
    // The same frame priced whole still takes the priced reading.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("ok"), record("Hi")]})),
        0,
        &frame("ok", "Hi", true),
    );
    assert!(!probabilities.take().unwrap().1);
}

#[test]
fn a_mixed_first_frame_after_queued_answer_records_aligns() {
    // The answer's record arrives first (its text held back), then the frame
    // that carries both texts and no records of its own.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("Hi")]})),
        0,
        &frame("", "", false),
    );
    probabilities.retain(None, 0, &frame("think", "Hi", true));
    let (taken, truncated) = probabilities.take().unwrap();
    assert_eq!(taken.len(), 1);
    assert!(!truncated);
}

#[test]
fn records_still_waiting_when_the_stream_ends_are_incomplete() {
    let stream = concat!(
        r#"data: {"choices":[{"index":0,"delta":{"content":"Hi"},"logprobs":{"content":[{"token":"Hi","logprob":-1.0,"bytes":[72,105],"top_logprobs":[]},{"token":"!","logprob":-1.0,"bytes":[33],"top_logprobs":[]}]}}]}"#,
        "\n",
        "data: [DONE]\n",
    );
    let mut normalizer =
        crate::dialects::Normalizer::new(crate::dialects::Dialect::OpenAiCompatible);
    normalizer.enable_logprobs_capture();
    for line in stream.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            let _ = normalizer.feed(&crate::sse::SseEvent {
                event: None,
                data: data.to_owned(),
            });
        }
    }
    assert!(normalizer.take_captured_logprobs().unwrap().1);
}

#[test]
fn silent_tokens_and_late_reasoning_inside_a_frame_are_holes() {
    let silent = json!({"token": "", "logprob": -1.0, "bytes": [], "top_logprobs": []});
    // A silent token ahead of the finishing frame's text, after the answer began.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("Hi")]})),
        0,
        &frame("", "Hi", false),
    );
    probabilities.retain(
        Some(&json!({"content":[silent.clone(), record("!")]})),
        0,
        &frame("", "!", true),
    );
    assert!(probabilities.take().unwrap().1);
    // A silent token between answer tokens on a non-finishing frame.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("H"), silent, record("i")]})),
        0,
        &frame("", "Hi", false),
    );
    assert!(probabilities.take().unwrap().1);
    // Priced reasoning on a mixed frame after the answer started.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("a"), record("Hi")]})),
        0,
        &frame("a", "Hi", false),
    );
    probabilities.retain(
        Some(&json!({"content":[record("x"), record("!")]})),
        0,
        &frame("x", "!", true),
    );
    assert!(probabilities.take().unwrap().1);
}

#[test]
fn an_answer_record_queued_ahead_of_unpriced_reasoning_aligns() {
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("Hi")]})),
        0,
        &frame("", "", false),
    );
    probabilities.retain(None, 0, &frame("think", "", false));
    probabilities.retain(None, 0, &frame("", "Hi", true));
    let (taken, truncated) = probabilities.take().unwrap();
    assert_eq!(taken.len(), 1);
    assert!(!truncated);
}

#[test]
fn image_frames_and_unemitted_reasoning_shapes_are_not_spelled() {
    // An image frame with records: output the text check cannot spell.
    let stream = concat!(
        r#"data: {"choices":[{"index":0,"delta":{"images":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]},"logprobs":{"content":[{"token":"x","logprob":-1.0,"bytes":[120],"top_logprobs":[]}]},"finish_reason":"stop"}]}"#,
        "\n",
    );
    // The capture check runs before the decoder judges the image itself.
    let mut normalizer =
        crate::dialects::Normalizer::new(crate::dialects::Dialect::OpenAiCompatible);
    normalizer.enable_logprobs_capture();
    let data = stream.trim().strip_prefix("data: ").unwrap();
    let _ = normalizer.feed(&crate::sse::SseEvent {
        event: None,
        data: data.to_owned(),
    });
    let (deltas, truncated) = normalizer.take_captured_logprobs().unwrap();
    assert!(deltas.is_empty());
    assert!(truncated);
    // A reasoning array the decoder never emits is not reasoning text.
    let stream = concat!(
        r#"data: {"choices":[{"index":0,"delta":{"reasoning":[{"type":"text","text":"ok"}]},"logprobs":{"content":[{"token":"o","logprob":-1.0,"bytes":[111],"top_logprobs":[]},{"token":"k","logprob":-1.0,"bytes":[107],"top_logprobs":[]}]}}]}"#,
        "\n",
        r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        "\n",
    );
    let (_, truncated, _) = replay(stream);
    assert!(truncated);
}

#[test]
fn opaque_late_and_answer_leading_records_are_placed() {
    // An opaque frame after an unpriced verdict with records still queued.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("Hi")]})),
        0,
        &frame("", "", false),
    );
    probabilities.retain(
        None,
        0,
        &FrameOutput {
            reasoning: "think".into(),
            opaque: true,
            ..Default::default()
        },
    );
    assert!(probabilities.take().unwrap().1);
    // A later mixed frame after an unpriced verdict cannot spend the delimiter
    // on the reasoning's own token.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(None, 0, &frame("a", "", false));
    probabilities.retain(
        Some(&json!({"content":[record("b"), record("Hi")]})),
        0,
        &frame("b", "Hi", true),
    );
    assert!(probabilities.take().unwrap().1);
    // Answer records on the first reasoning-only frame lead the answer.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("Hi")]})),
        0,
        &frame("think", "", false),
    );
    probabilities.retain(None, 0, &frame("", "Hi", true));
    let (taken, truncated) = probabilities.take().unwrap();
    assert_eq!(taken.len(), 1);
    assert!(!truncated);
    // Output after the finishing frame.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("Hi"), record("<stop>")]})),
        0,
        &frame("", "Hi", true),
    );
    probabilities.retain(
        Some(&json!({"content":[record("!")]})),
        0,
        &frame("", "!", false),
    );
    let (taken, truncated) = probabilities.take().unwrap();
    assert_eq!(taken.len(), 1);
    assert!(truncated);
}

#[test]
fn records_still_waiting_at_a_clean_end_after_a_finish_are_incomplete() {
    let mut normalizer =
        crate::dialects::Normalizer::new(crate::dialects::Dialect::OpenAiCompatible);
    normalizer.enable_logprobs_capture();
    for data in [
        r#"{"choices":[{"index":0,"delta":{"content":"Hi"},"logprobs":{"content":[{"token":"Hi","logprob":-1.0,"bytes":[72,105],"top_logprobs":[]}]},"finish_reason":"stop"}]}"#,
        r#"{"choices":[{"index":0,"delta":{},"logprobs":{"content":[{"token":"x","logprob":-1.0,"bytes":[120],"top_logprobs":[]}]}}]}"#,
    ] {
        let _ = normalizer.feed(&crate::sse::SseEvent {
            event: None,
            data: data.to_owned(),
        });
    }
    let _ = normalizer.on_stream_end();
    assert!(normalizer.take_captured_logprobs().unwrap().1);
}

#[test]
fn answer_leading_records_that_no_answer_spells_are_a_hole() {
    // The unpriced reading's records on a finishing frame spell nothing.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("x")]})),
        0,
        &frame("think", "", true),
    );
    assert!(probabilities.take().unwrap().1);
    // Nor may a queued answer-leading record pass as the delimiter at an
    // empty finish.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("x")]})),
        0,
        &frame("think", "", false),
    );
    probabilities.retain(None, 0, &frame("", "", true));
    assert!(probabilities.take().unwrap().1);
}

#[test]
fn opaque_output_after_the_finishing_frame_is_a_hole() {
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("Hi")]})),
        0,
        &frame("", "Hi", true),
    );
    probabilities.retain(
        None,
        0,
        &FrameOutput {
            opaque: true,
            ..Default::default()
        },
    );
    let (taken, truncated) = probabilities.take().unwrap();
    assert_eq!(taken.len(), 1);
    assert!(truncated);
}

#[test]
fn an_answer_record_equal_to_the_reasoning_keeps_both_readings() {
    // The queued record could be the reasoning token or the answer's leading
    // one; the answer frame that carries no record decides.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("Hi")]})),
        0,
        &frame("", "", false),
    );
    probabilities.retain(None, 0, &frame("Hi", "", false));
    probabilities.retain(None, 0, &frame("", "Hi", true));
    let (taken, truncated) = probabilities.take().unwrap();
    assert_eq!(taken.len(), 1);
    assert!(!truncated);
}

#[test]
fn an_answer_token_emitted_across_frames_is_not_taken_for_the_delimiter() {
    // "Hel" is a prefix of the front token and the next token repeats it: the
    // front token is the answer's, emitted over two frames.
    let mut probabilities = ChatProbabilities::Capture(Buffer::default());
    probabilities.retain(
        Some(&json!({"content":[record("Hello"), record("Hel")]})),
        0,
        &frame("", "Hel", false),
    );
    probabilities.retain(None, 0, &frame("", "loHel", true));
    let (taken, truncated) = probabilities.take().unwrap();
    assert_eq!(taken.len(), 1);
    assert!(!truncated);
}
