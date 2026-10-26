//! Gemini `streamGenerateContent` golden-fixture and thought-signature tests.

use super::super::{drain_stream_fixture, Dialect};
use super::*;
use crate::errors::RefusalReason;
use serde_json::json;

fn run_stream(dialect: Dialect, chunks: &[&[u8]]) -> (Vec<Value>, Option<Failure>) {
    let owned: Vec<Vec<u8>> = chunks.iter().map(|chunk| chunk.to_vec()).collect();
    drain_stream_fixture(dialect, &owned)
}

fn sse(payload: &Value) -> Vec<u8> {
    format!("data: {payload}\n\n").into_bytes()
}

#[test]
fn thought_summaries_and_signatures_keep_their_exact_part_association() {
    let parts = vec![
        json!({"thought":true,"text":"Summary, not full CoT. 雪"}),
        json!({"functionCall":{"name":"lookup","args":{"id":"a"}},"thoughtSignature":"tool-signature=="}),
        json!({"text":"answer","thoughtSignature":"text-signature=="}),
        json!({"thoughtSignature":"standalone=="}),
    ];
    let chunk = sse(&json!({"candidates":[{"content":{"parts":parts},"finishReason":"STOP"}]}));
    let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &[chunk.as_slice()]);
    assert!(failure.is_none());
    let captured: Vec<_> = events
        .iter()
        .filter(|event| event["kind"] == "gemini_thought_part")
        .map(|event| event["part"].clone())
        .collect();
    // Signed parts are retained whole; the unsigned summary is display
    // text only.
    assert_eq!(captured, parts[1..]);
    assert!(!events
        .iter()
        .any(|event| event["kind"] == "reasoning_content_delta"));
    // Only the thought part's text is displayable reasoning; signed answer
    // and tool parts never are.
    let displayed: Vec<_> = events
        .iter()
        .filter(|event| event["kind"] == "reasoning_text_delta")
        .collect();
    assert_eq!(
        displayed,
        vec![&json!({"kind":"reasoning_text_delta","text":"Summary, not full CoT. 雪"})]
    );
    let visible: Vec<_> = events
        .iter()
        .filter(|event| event["kind"] == "text_delta")
        .collect();
    assert_eq!(visible, vec![&json!({"kind":"text_delta","text":"answer"})]);
    let private = Event::GeminiThoughtPart(std::sync::Arc::new(parts[0].clone()));
    assert!(!private.is_output_token());
    assert!(private.is_generation_progress());
}

#[test]
fn signed_generated_image_over_the_evidence_bound_reaches_the_caller_intact() {
    // Gemini image models sign the image part itself. A real PNG padded past
    // the 1 MiB evidence bound still decodes, as large generated images do.
    use base64::Engine;
    let png = base64::engine::general_purpose::STANDARD
        .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC")
        .unwrap();
    let mut bytes = png;
    bytes.resize(1_500_000, 0);
    let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
    let part = json!({
        "inlineData": {"mimeType": "image/png", "data": data},
        "thoughtSignature": "image-signature==",
    });
    let chunk = sse(&json!({"candidates":[{"content":{"parts":[part]},"finishReason":"STOP"}]}));
    let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &[chunk.as_slice()]);
    assert!(failure.is_none(), "{failure:?}");
    let images: Vec<_> = events
        .iter()
        .filter(|event| event["kind"] == "image")
        .collect();
    assert_eq!(images.len(), 1);
    assert_eq!(
        images[0]["url"],
        json!(format!("data:image/png;base64,{data}"))
    );
    let captured: Vec<_> = events
        .iter()
        .filter(|event| event["kind"] == "gemini_thought_part")
        .map(|event| event["part"].clone())
        .collect();
    assert_eq!(
        captured,
        vec![json!({
            "inlineData": {"mimeType": "image/png", "dataBytes": data.len()},
            "thoughtSignature": "image-signature==",
        })]
    );
}

#[test]
fn gemini_golden_stream_normalizes_text_tools_usage_and_completion() {
    // Golden fixture: raw provider bytes in, exact canonical events out.
    let chunks = [
        sse(&json!({"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]})),
        sse(&json!({"candidates": [{"content": {"parts": [
            {"thought": true, "text": "hidden reasoning"},
            {"text": "lo"},
        ]}}]})),
        sse(&json!({"candidates": [{"content": {"parts": [{
            "functionCall": {
                "id": "call-1",
                "name": "lookup",
                "args": {"city": "Zürich", "count": 2},
            }
        }]}}]})),
        sse(&json!({
            "candidates": [{"finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 5,
                "cachedContentTokenCount": 2,
                "thoughtsTokenCount": 3,
            },
        })),
    ];
    let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
    let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
    assert!(failure.is_none());
    let raw_arguments = "{\"city\":\"Zürich\",\"count\":2}";
    assert_eq!(
        events,
        vec![
            json!({"kind": "text_delta", "text": "Hel"}),
            json!({"kind": "reasoning_text_delta", "text": "hidden reasoning"}),
            json!({"kind": "text_delta", "text": "lo"}),
            json!({"kind": "tool_call_started", "index": 0, "call_id": "call-1", "name": "lookup"}),
            json!({"kind": "tool_arguments_delta", "index": 0, "text": raw_arguments}),
            json!({
                "kind": "tool_call_completed",
                "index": 0,
                "call_id": "call-1",
                "name": "lookup",
                "raw_arguments": raw_arguments,
            }),
            // thoughtsTokenCount is additive on the Gemini wire, so the
            // output total carries the folded 5 + 3 and reasoning names
            // the subset.
            json!({
                "kind": "usage",
                "input_tokens": 11,
                "output_tokens": 8,
                "cached_input_tokens": 2,
                "reasoning_tokens": 3,
            }),
            json!({"kind": "completed"}),
        ]
    );
}

#[test]
fn gemini_thinking_stream_folds_thoughts_into_the_terminal_usage() {
    // Frame shapes from gemini-3.7-flash streamGenerateContent?alt=sse
    // (2026-09-03): every chunk carries usageMetadata, the final chunk
    // adds the thoughtSignature part and STOP. The terminal chunk's counts
    // win over the earlier partial ones, and Google's totalTokenCount
    // (11 + 7 + 654 = 672) shows thoughts are additive, so the normalized
    // output total is 661 with 654 as the reasoning subset.
    let chunks = [
        sse(&json!({
            "candidates": [{"content": {"parts": [{"text": "Sunlight scatters off air "}], "role": "model"}, "index": 0}],
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 3,
                "totalTokenCount": 668,
                "thoughtsTokenCount": 654,
            },
            "modelVersion": "gemini-3.7-flash",
            "responseId": "resp-1",
        })),
        sse(&json!({
            "candidates": [{
                "content": {"parts": [{"text": "molecules.", "thoughtSignature": "CikB"}], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
            }],
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 7,
                "totalTokenCount": 672,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 11}],
                "thoughtsTokenCount": 654,
                "serviceTier": "standard",
            },
            "modelVersion": "gemini-3.7-flash",
            "responseId": "resp-1",
        })),
    ];
    let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
    let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
    assert!(failure.is_none());
    assert_eq!(
        events,
        vec![
            json!({"kind": "text_delta", "text": "Sunlight scatters off air "}),
            json!({"kind": "gemini_thought_part", "part": {"text":"molecules.","thoughtSignature":"CikB"}}),
            json!({"kind": "text_delta", "text": "molecules."}),
            json!({
                "kind": "usage",
                "input_tokens": 11,
                "output_tokens": 661,
                "cached_input_tokens": 0,
                "reasoning_tokens": 654,
            }),
            json!({"kind": "completed"}),
        ]
    );
}

#[test]
fn gemini_missing_call_id_uses_the_deterministic_local_fallback() {
    let chunks = [
        sse(&json!({"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "first", "args": {}}},
            {"functionCall": {"name": "second"}},
        ]}}]})),
        sse(&json!({
            "candidates": [{"finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
        })),
    ];
    let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
    let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
    assert!(failure.is_none());
    assert_eq!(
        events[0],
        json!({"kind": "tool_call_started", "index": 0, "call_id": "gemini-call-0", "name": "first"})
    );
    assert_eq!(events[1]["text"], "{}");
    assert_eq!(
        events[3],
        json!({"kind": "tool_call_started", "index": 1, "call_id": "gemini-call-1", "name": "second"})
    );
    // Absent usage counts are zero (require_integer parity), and an
    // omitted thoughtsTokenCount stays unknown.
    assert_eq!(
        events[6],
        json!({
            "kind": "usage",
            "input_tokens": 1,
            "output_tokens": 1,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
        })
    );
    assert_eq!(events[7], json!({"kind": "completed"}));
}

#[test]
fn gemini_safety_finish_maps_to_a_categorized_refusal() {
    // Each finish token names the bounded category the caller sees, with
    // the fixed phrase in the message and never any provider prose.
    for (reason, category, phrase) in [
        ("SAFETY", "content_policy", "content policy"),
        ("PROHIBITED_CONTENT", "content_policy", "content policy"),
        ("BLOCKLIST", "content_policy", "content policy"),
        (
            "RECITATION",
            "recitation",
            "recitation of copyrighted material",
        ),
        ("SPII", "data_inspection", "data inspection"),
    ] {
        let chunks = [sse(&json!({"candidates": [{"finishReason": reason}]}))];
        let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(failure.is_none());
        assert_eq!(
            events,
            vec![json!({
                "kind": "failed",
                "failure_class": "refusal",
                "safe_message": format!("provider refused the request: {phrase}"),
                "refusal_reason": category,
            })],
            "{reason}"
        );
    }
}

#[test]
fn thinking_only_truncation_counts_thoughts_when_zero_candidates_are_omitted() {
    // UsageMetadata's proto3 implicit-presence int32 fields omit zero:
    // https://github.com/googleapis/googleapis/blob/master/google/ai/generativelanguage/v1beta/generative_service.proto
    // https://protobuf.dev/programming-guides/json/#presence-and-default-values
    let candidate = json!({"candidates":[{"content":{"parts":[]},"finishReason":"MAX_TOKENS"}]});
    for usage in [
        Some(json!({"promptTokenCount":42,"thoughtsTokenCount":7,"totalTokenCount":49})),
        None,
    ] {
        let mut payload = candidate.clone();
        if let Some(usage) = usage.clone() {
            payload["usageMetadata"] = usage;
        }
        let chunks = [sse(&payload)];
        let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(failure.is_none(), "{failure:?}");
        assert_eq!(events.last().unwrap()["kind"], "incomplete");
        let metered = events.iter().find(|event| event["kind"] == "usage");
        if usage.is_some() {
            let metered = metered.unwrap();
            assert_eq!(metered["input_tokens"], 42);
            assert_eq!(metered["output_tokens"], 7);
            assert_eq!(metered["reasoning_tokens"], 7);
        } else {
            assert!(metered.is_none(), "an absent whole object is unknown");
        }
    }
}

#[test]
fn gemini_prompt_block_is_a_content_free_refusal_with_its_usage() {
    // The production shape (2026-09-04, gemini-3.7-flash and 3.8-flash
    // via streamGenerateContent?alt=sse): one frame, no candidates, the
    // block named on promptFeedback, and usageMetadata counting the
    // prompt Google processed. It must terminate the stream as a refusal
    // (400, no retry, no failover), never as a malformed stream end.
    for (reason, category, message) in [
        (
            "SAFETY",
            "content_policy",
            "provider refused the request: content policy",
        ),
        (
            "PROHIBITED_CONTENT",
            "content_policy",
            "provider refused the request: content policy",
        ),
        (
            "BLOCKLIST",
            "content_policy",
            "provider refused the request: content policy",
        ),
        // A block reason the vocabulary does not know is an unnamed refusal.
        ("OTHER", "unspecified", "provider refused the request"),
        (
            "IMAGE_SAFETY",
            "content_policy",
            "provider refused the request: content policy",
        ),
    ] {
        let chunks = [sse(&json!({
            "promptFeedback": {
                "blockReason": reason,
                "safetyRatings": [
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "HIGH"}
                ],
            },
            "usageMetadata": {"promptTokenCount": 42, "totalTokenCount": 42},
        }))];
        let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(
            failure.is_none(),
            "{reason}: a prompt block is a terminal, not a failure"
        );
        assert_eq!(
            events,
            vec![
                json!({
                    "kind": "usage",
                    "input_tokens": 42,
                    "output_tokens": 0,
                    "cached_input_tokens": 0,
                    "reasoning_tokens": 0,
                }),
                json!({
                    "kind": "failed",
                    "failure_class": "refusal",
                    "safe_message": message,
                    "refusal_reason": category,
                }),
            ],
            "{reason}"
        );
    }
}

#[test]
fn gemini_refusals_keep_the_raw_verdict_token_for_the_ledger_only() {
    // gemini-4-argon, 2026-10-22: Claude Code turns blocked on the prompt
    // settled as a bare "provider refused the request" with no category and
    // no detail. The block reason Google named must reach the ledger (as the
    // provider detail) while the caller still sees only the bounded category.
    for (frame, category, detail) in [
        (
            json!({"promptFeedback": {"blockReason": "OTHER"}}),
            RefusalReason::Unspecified,
            "blockReason OTHER",
        ),
        (
            json!({"promptFeedback": {"blockReason": "JAILBREAK"}}),
            RefusalReason::Unspecified,
            "blockReason JAILBREAK",
        ),
        (
            json!({"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}),
            RefusalReason::ContentPolicy,
            "blockReason PROHIBITED_CONTENT",
        ),
        (
            json!({"promptFeedback": {"blockReason": "not a token, with prose"}}),
            RefusalReason::Unspecified,
            "blockReason non-identifier",
        ),
    ] {
        let mut normalizer = super::super::Normalizer::new(Dialect::GeminiGenerateContent);
        let events = normalizer
            .feed(&crate::sse::SseEvent {
                event: None,
                data: frame.to_string(),
            })
            .expect("a prompt block is a terminal event");
        let [Event::Failed(failure)] = events.as_slice() else {
            panic!("{frame}: expected one refusal, got {events:?}");
        };
        assert_eq!(failure.failure_class, FailureClass::Refusal, "{frame}");
        assert_eq!(failure.refusal_reason, Some(category), "{frame}");
        assert_eq!(failure.provider_detail.as_deref(), Some(detail), "{frame}");
        let body = failure.public_error().json_body();
        assert!(!body.to_string().contains(detail), "{frame}: {body}");
        assert!(!failure.retryable_same_deployment && !failure.failover_eligible);
    }
    // A candidate-level refusal finish keeps its token the same way; it is
    // emitted once transport ends.
    let mut normalizer = super::super::Normalizer::new(Dialect::GeminiGenerateContent);
    normalizer
        .feed(&crate::sse::SseEvent {
            event: None,
            data: json!({"candidates": [{"finishReason": "SAFETY"}]}).to_string(),
        })
        .expect("terminal candidate");
    let events = normalizer.on_stream_end().expect("terminal");
    let [Event::Failed(failure)] = events.as_slice() else {
        panic!("expected one refusal, got {events:?}");
    };
    assert_eq!(failure.refusal_reason, Some(RefusalReason::ContentPolicy));
    assert_eq!(
        failure.provider_detail.as_deref(),
        Some("finishReason SAFETY")
    );
}

#[test]
fn gemini_prompt_block_without_usage_is_still_a_refusal() {
    let chunks = [sse(&json!({"promptFeedback": {"blockReason": "SAFETY"}}))];
    let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
    let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
    assert!(failure.is_none());
    assert_eq!(
        events,
        vec![json!({
            "kind": "failed",
            "failure_class": "refusal",
            "safe_message": "provider refused the request: content policy",
            "refusal_reason": "content_policy",
        })]
    );
    // A refusal is the model's verdict on the prompt: neither redialed on
    // the same deployment nor failed over to a sibling lane.
    let refusal = Failure::refusal(crate::errors::RefusalReason::ContentPolicy);
    assert!(!refusal.retryable_same_deployment);
    assert!(!refusal.failover_eligible);
}

#[test]
fn gemini_unspecified_block_reason_and_ratings_only_feedback_are_not_blocks() {
    // Ordinary answers carry promptFeedback.safetyRatings (and some carry
    // the enum default) alongside real candidates; those must keep
    // flowing as content.
    for feedback in [
        json!({"safetyRatings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE"}]}),
        json!({"blockReason": "BLOCK_REASON_UNSPECIFIED"}),
    ] {
        let chunks = [
            sse(&json!({
                "promptFeedback": feedback,
                "candidates": [{"content": {"parts": [{"text": "fine"}]}}],
            })),
            sse(&json!({"candidates": [{"finishReason": "STOP"}]})),
        ];
        let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(failure.is_none());
        assert_eq!(
            events,
            vec![
                json!({"kind": "text_delta", "text": "fine"}),
                json!({"kind": "completed"}),
            ]
        );
    }
}

#[test]
fn gemini_error_envelope_is_classified_by_what_google_said() {
    // Google's own error envelope on the stream (verified shape for a
    // 503 UNAVAILABLE "overloaded"): a provider-declared failure. It is
    // classified by its content like every other dialect's: an overloaded
    // model is a THROTTLE (fail over, advertise Retry-After; redialing the
    // same saturated rung buys nothing), never a malformed stream end, and
    // never a completion when output preceded it.
    let envelope = json!({
        "error": {
            "code": 503,
            "message": "The model is overloaded. Please try again later.",
            "status": "UNAVAILABLE",
        }
    });
    let failed = json!({
        "kind": "failed",
        "failure_class": "throttled",
        "safe_message": "provider throttled the request; retry after the delay in the Retry-After header",
    });
    let alone = [sse(&envelope)];
    let refs: Vec<&[u8]> = alone.iter().map(Vec::as_slice).collect();
    let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
    assert!(failure.is_none());
    assert_eq!(events, vec![failed.clone()]);
    let after_output = [
        sse(&json!({"candidates": [{"content": {"parts": [{"text": "partial"}]}}]})),
        sse(&envelope),
    ];
    let refs: Vec<&[u8]> = after_output.iter().map(Vec::as_slice).collect();
    let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
    assert!(failure.is_none());
    assert_eq!(
        events,
        vec![json!({"kind": "text_delta", "text": "partial"}), failed]
    );
    // A genuine provider fault keeps the retry-then-failover shape.
    let internal = json!({
        "error": {"code": 500, "message": "Internal error encountered.", "status": "INTERNAL"}
    });
    let refs = [sse(&internal)];
    let refs: Vec<&[u8]> = refs.iter().map(Vec::as_slice).collect();
    let (events, _failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
    assert_eq!(
        events,
        vec![json!({
            "kind": "failed",
            "failure_class": "provider_internal",
            "safe_message": "provider stream failed",
        })]
    );
    let classified = crate::stream_errors::stream_failure(
        crate::stream_errors::StreamErrorKind::ProviderInternal,
        None,
    );
    assert_eq!(classified.failure_class, FailureClass::ProviderInternal);
    assert!(classified.retryable_same_deployment);
    assert!(classified.failover_eligible);
}

#[test]
fn gemini_max_tokens_is_incomplete_and_unknown_reasons_fail() {
    let incomplete = [sse(
        &json!({"candidates": [{"finishReason": "MAX_TOKENS"}]}),
    )];
    let refs: Vec<&[u8]> = incomplete.iter().map(Vec::as_slice).collect();
    let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
    assert!(failure.is_none());
    assert_eq!(events, vec![json!({"kind": "incomplete"})]);

    let unknown = [sse(
        &json!({"candidates": [{"finishReason": "MALFUNCTION"}]}),
    )];
    let refs: Vec<&[u8]> = unknown.iter().map(Vec::as_slice).collect();
    let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
    assert!(failure.is_none());
    assert_eq!(
        events,
        vec![json!({
            "kind": "failed",
            "failure_class": "provider_internal",
            "safe_message": "provider ended the stream unexpectedly",
        })]
    );
}

#[test]
fn gemini_malformed_frame_before_content_is_a_retryable_abnormal_end() {
    // A malformed frame arriving before any content is a Gemini abnormal
    // end with nothing to salvage: there is no partial answer, so it is
    // reclassified from a hard malformed reject to a retryable transport
    // failure (retry the lane, then fail over) rather than being accepted.
    // The frames are still rejected: none is silently taken as content.
    let cases: [Value; 3] = [
        // A non-text text part (python: parts.text must be a string).
        json!({"candidates": [{"content": {"parts": [{"text": 5}]}}]}),
        // Null function-call args (python: args must decode to a dict).
        json!({"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "x", "args": null}}
        ]}}]}),
        // Two candidates in one frame.
        json!({"candidates": [{}, {}]}),
    ];
    for payload in &cases {
        let chunk = sse(payload);
        let refs: Vec<&[u8]> = [chunk.as_slice()].to_vec();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(events.is_empty(), "a rejected frame emits no content");
        let failure = failure.expect("a malformed pre-content frame must not succeed");
        assert_eq!(failure.failure_class, FailureClass::Transport);
        assert!(failure.retryable_same_deployment);
        assert!(failure.failover_eligible);
    }
}

#[test]
fn gemini_usage_only_close_without_content_still_fails_malformed() {
    // A stream that closes having emitted NO content at all (here only a
    // usage-only trailer, then a clean EOF) still fails malformed: there is
    // no real answer to complete, so the clean-end tolerance does not apply
    // and the abnormal-end recovery never engages (nothing terminated it).
    let empty = [sse(&json!({
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 0},
    }))];
    let refs: Vec<&[u8]> = empty.iter().map(Vec::as_slice).collect();
    let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
    assert!(events.is_empty());
    assert_eq!(
        failure.expect("must fail").failure_class,
        FailureClass::MalformedResponse
    );
}

#[test]
fn gemini_malformed_frame_after_content_recovers_as_incomplete() {
    // Content emitted, then a structurally malformed frame: the real answer
    // is preserved and the turn ends `incomplete` (an early-termination
    // finish reason), never discarded as malformed. Last-seen usage is
    // folded so delivered tokens still bill at settle.
    let chunks = [
        sse(&json!({"candidates": [{"content": {"parts": [{"text": "partial"}]}}]})),
        sse(&json!({"usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 3}})),
        // A non-text text part after content: malformed mid-stream frame.
        sse(&json!({"candidates": [{"content": {"parts": [{"text": 5}]}}]})),
    ];
    let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
    let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
    assert!(
        failure.is_none(),
        "a partial answer is recovered, not failed"
    );
    assert_eq!(
        events,
        vec![
            json!({"kind": "text_delta", "text": "partial"}),
            json!({
                "kind": "usage",
                "input_tokens": 9,
                "output_tokens": 3,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
            }),
            json!({"kind": "incomplete"}),
        ]
    );
}

#[test]
fn gemini_usage_only_trailer_frame_folds_usage_without_failing() {
    // A trailer frame that carries only usageMetadata (no candidates array)
    // must not fail: its usage is folded and the stream continues to its
    // real terminal candidate, which flushes the last-seen usage.
    let chunks = [
        sse(&json!({"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})),
        sse(&json!({"usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2}})),
        sse(&json!({"candidates": [{"finishReason": "STOP"}]})),
    ];
    let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
    let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
    assert!(failure.is_none());
    assert_eq!(
        events,
        vec![
            json!({"kind": "text_delta", "text": "hi"}),
            json!({
                "kind": "usage",
                "input_tokens": 7,
                "output_tokens": 2,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
            }),
            json!({"kind": "completed"}),
        ]
    );
}

#[test]
fn gemini_clean_end_after_content_completes_with_folded_usage() {
    // Gemini sometimes ends the stream right after its last content frame
    // with no finishReason frame; the relay/drain closes it as a normal
    // completion (folding the last-seen usage) rather than throwing the
    // real answer away as malformed.
    let chunks = [
        sse(&json!({"candidates": [{"content": {"parts": [{"text": "done"}]}}]})),
        sse(&json!({"usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 1}})),
    ];
    let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
    let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
    assert!(failure.is_none());
    assert_eq!(
        events,
        vec![
            json!({"kind": "text_delta", "text": "done"}),
            json!({
                "kind": "usage",
                "input_tokens": 4,
                "output_tokens": 1,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
            }),
            json!({"kind": "completed"}),
        ]
    );
}

#[test]
fn gemini_frames_after_the_terminal_candidate_are_ignored() {
    let mut normalizer = Normalizer::new(Dialect::GeminiGenerateContent);
    let terminal = crate::sse::SseEvent {
        event: None,
        data: json!({"candidates": [{"finishReason": "STOP"}]}).to_string(),
    };
    let trailing = crate::sse::SseEvent {
        event: None,
        data: json!({"candidates": [{"content": {"parts": [{"text": "late"}]}}]}).to_string(),
    };
    let events = normalizer.feed(&terminal).expect("terminal frame");
    assert!(!events.iter().any(Event::is_terminal));
    assert!(!normalizer.saw_terminal());
    assert!(normalizer.metadata_drain_started().is_some());
    assert!(normalizer.feed(&trailing).expect("ignored").is_empty());
    assert!(matches!(
        normalizer.on_stream_end().unwrap().as_slice(),
        [Event::Completed]
    ));
    assert!(normalizer.saw_terminal());
}

#[test]
fn a_function_call_signature_rides_its_tool_call_and_only_that_one() {
    // Gemini signs the first call of a parallel batch; the second is unsigned.
    let parts = vec![
        json!({"functionCall":{"name":"get_weather","args":{"city":"Paris"},"id":"call_1"},"thoughtSignature":"c2lnbmF0dXJlLWE="}),
        json!({"functionCall":{"name":"get_weather","args":{"city":"Tokyo"},"id":"call_2"}}),
    ];
    let chunk = sse(&json!({"candidates":[{"content":{"parts":parts},"finishReason":"STOP"}]}));
    let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &[chunk.as_slice()]);
    assert!(failure.is_none());
    let signatures = |kind: &str| -> Vec<Value> {
        events
            .iter()
            .filter(|event| event["kind"] == kind)
            .map(|event| {
                event
                    .get("thought_signature")
                    .cloned()
                    .unwrap_or(Value::Null)
            })
            .collect()
    };
    let expected = vec![json!("c2lnbmF0dXJlLWE="), Value::Null];
    assert_eq!(signatures("tool_call_started"), expected);
    assert_eq!(signatures("tool_call_completed"), expected);
    // The signed part stays capture evidence too, paired with its call.
    let captured: Vec<_> = events
        .iter()
        .filter(|event| event["kind"] == "gemini_thought_part")
        .map(|event| event["part"].clone())
        .collect();
    assert_eq!(captured, parts[..1]);
}

#[test]
fn a_non_text_function_call_signature_is_malformed() {
    for signature in [json!(7), json!(""), json!({"bytes": "x"})] {
        let part = json!({"functionCall":{"name":"lookup","args":{}},"thoughtSignature":signature});
        let chunk =
            sse(&json!({"candidates":[{"content":{"parts":[part]},"finishReason":"STOP"}]}));
        let (_, failure) = run_stream(Dialect::GeminiGenerateContent, &[chunk.as_slice()]);
        assert!(failure.is_some(), "{signature}");
    }
}

#[test]
fn a_signed_image_past_one_megabyte_of_signature_is_served_not_overflowed() {
    // Gemini image models sign each generated image; the signature alone
    // measured 0.9 to 1.1 MB live (gemini-3.1-flash-image, 2026-10-19).
    use base64::Engine;
    let png = base64::engine::general_purpose::STANDARD
        .decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC")
        .unwrap();
    let data = base64::engine::general_purpose::STANDARD.encode(&png);
    let signature = "A".repeat(1_200_000);
    let part = json!({
        "inlineData": {"mimeType": "image/png", "data": data},
        "thoughtSignature": signature,
    });
    let chunk = sse(&json!({"candidates":[{"content":{"parts":[part]},"finishReason":"STOP"}]}));
    let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &[chunk.as_slice()]);
    assert!(failure.is_none(), "{failure:?}");
    assert_eq!(
        events
            .iter()
            .filter(|event| event["kind"] == "image")
            .count(),
        1
    );
    let retained =
        Event::GeminiThoughtPart(std::sync::Arc::new(json!({"thoughtSignature": signature})));
    assert!(crate::relay::event_retained_bytes(&retained) > 1_200_000);
}

#[test]
fn malformed_function_call_and_response_finishes_redial_then_fail_over() {
    for reason in ["MALFORMED_FUNCTION_CALL", "MALFORMED_RESPONSE"] {
        let mut normalizer = super::super::Normalizer::new(Dialect::GeminiGenerateContent);
        let frame = crate::sse::SseEvent {
            event: None,
            data: json!({"candidates":[{"finishReason": reason}]}).to_string(),
        };
        normalizer.feed(&frame).unwrap();
        let events = normalizer.finish_metadata_drain();
        let Some(Event::Failed(failure)) = events.last() else {
            panic!("{reason} must end the stream failed: {events:?}");
        };
        assert_eq!(
            failure.failure_class,
            FailureClass::MalformedResponse,
            "{reason}"
        );
        assert!(
            failure.retryable_same_deployment && failure.failover_eligible,
            "{reason}"
        );
    }
}

#[test]
fn a_blocked_image_turn_with_part_less_content_is_a_refusal() {
    // gemini-3-pro-image, live 2026-10-19: `"content": {}` beside IMAGE_RECITATION.
    for reason in ["IMAGE_RECITATION", "IMAGE_PROHIBITED_CONTENT"] {
        let chunk = sse(&json!({"candidates":[{"content":{},"finishReason":reason}]}));
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &[chunk.as_slice()]);
        assert!(failure.is_none(), "{reason}: {failure:?}");
        let last = events.last().unwrap();
        assert_eq!(last["kind"], "failed", "{reason}");
        assert_eq!(last["failure_class"], "refusal", "{reason}");
    }
}
