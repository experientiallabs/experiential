//! Inline tests for `encode_responses`, split into a submodule file so the
//! implementation stays within the repository line budget.

use super::*;
#[test]
fn ignored_generation_controls_are_disclosed_by_responses_encoder() {
    let envelope = ResponsesEnvelope {
        ignored_parameters: vec!["top_k".to_string()],
        ..ResponsesEnvelope::default()
    };
    let mut encoder =
        ResponsesSseEncoder::new("request-1", "coding", 1_700_000_000.0, envelope.clone());
    let frames = encoder.start().expect("stream start must encode");
    assert!(frames[0].contains("\"x-experiential-ignored-parameters\":[\"top_k\"]"));

    let completed = completed_responses_body(
        "request-1",
        "coding",
        1_700_000_000.0,
        envelope,
        &[Event::Completed],
    )
    .expect("completed body must encode");
    assert_eq!(
        completed.body["x-experiential-ignored-parameters"],
        json!(["top_k"])
    );
}

#[test]
fn thinking_deltas_project_onto_reasoning_summary_parts() {
    let mut encoder = ResponsesSseEncoder::new(
        "request-1",
        "coding",
        1_700_000_000.0,
        ResponsesEnvelope::default(),
    );
    encoder.start().expect("stream start must encode");
    let frames = encoder
        .feed(&Event::ThinkingDelta {
            index: 0,
            delta: "step one".to_string(),
        })
        .expect("thinking delta projects");
    assert!(frames[0].contains("response.output_item.added"));
    assert!(frames.iter().any(
        |frame| frame.contains("response.reasoning_summary_text.delta")
            && frame.contains("\"delta\":\"step one\"")
    ));
    // Signatures cannot round-trip on this surface and stay silent.
    assert!(encoder
        .feed(&Event::ThinkingSignature {
            index: 0,
            signature: "sig==".to_string(),
        })
        .expect("signature is dropped")
        .is_empty());
}

fn fireworks_tool_events() -> Vec<Event> {
    vec![
        Event::ReasoningContentDelta {
            route_sha256: "a".repeat(64),
            delta: "hidden provider reasoning".to_string(),
        },
        Event::ToolCallStarted {
            index: 0,
            call_id: "call-one".to_string(),
            name: "weather".to_string(),
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: "{}".to_string(),
        },
        Event::ToolCallCompleted {
            index: 0,
            call: CompletedToolCall {
                call_id: "call-one".to_string(),
                name: "weather".to_string(),
                raw_arguments: "{}".to_string(),
            },
        },
        Event::Completed,
    ]
}

#[test]
fn fireworks_reasoning_round_trips_as_encrypted_content() {
    let events = fireworks_tool_events();
    let envelope = ResponsesEnvelope {
        include_encrypted_reasoning: true,
        ..ResponsesEnvelope::default()
    };
    let mut encoder =
        ResponsesSseEncoder::new("request-1", "coding", 1_700_000_000.0, envelope.clone());
    encoder.start().expect("stream start must encode");
    encoder
        .set_reasoning_content_carrier("authenticated-carrier-v2".to_string())
        .expect("carrier must attach");
    let mut frames = Vec::new();
    for event in &events {
        frames.extend(encoder.feed(event).expect("Responses event must encode"));
    }
    let public = frames.join("");
    assert!(!public.contains("hidden provider reasoning"));
    assert!(public.contains("\"encrypted_content\":\"authenticated-carrier-v2\""));

    let completed = completed_responses_body_with_carrier(
        "request-1",
        "coding",
        1_700_000_000.0,
        envelope,
        &events,
        Some("authenticated-carrier-v2"),
    )
    .expect("completed body must preserve carrier");
    assert_eq!(
        completed.body["output"][0]["encrypted_content"],
        json!("authenticated-carrier-v2")
    );
    assert!(!completed
        .body
        .to_string()
        .contains("hidden provider reasoning"));
}

#[test]
fn fireworks_reasoning_fails_closed_without_a_sealed_carrier() {
    assert!(completed_responses_body(
        "request-1",
        "coding",
        1_700_000_000.0,
        ResponsesEnvelope::default(),
        &fireworks_tool_events(),
    )
    .is_err());
}

#[test]
fn encrypted_reasoning_lands_on_the_completed_reasoning_item() {
    let public_envelope = ResponsesEnvelope {
        include_encrypted_reasoning: true,
        ..ResponsesEnvelope::default()
    };
    let completed = completed_responses_body(
        "request-1",
        "coding",
        1_700_000_000.0,
        public_envelope.clone(),
        &[
            Event::ReasoningSummaryDelta {
                output_index: 0,
                summary_index: 0,
                delta: "planned".to_string(),
            },
            Event::EncryptedReasoning {
                output_index: 0,
                encrypted_content: "blob==".to_string(),
            },
            Event::TextDelta("answer".to_string()),
            Event::Completed,
        ],
    )
    .expect("completed body must encode");
    let reasoning = &completed.body["output"][0];
    assert_eq!(reasoning["type"], json!("reasoning"));
    assert_eq!(reasoning["encrypted_content"], json!("blob=="));
    assert_eq!(
        reasoning["summary"],
        json!([{"type": "summary_text", "text": "planned"}])
    );
    assert_eq!(completed.body["output"][1]["type"], json!("message"));

    // An encrypted payload with no summary still creates its output item.
    let bare = completed_responses_body(
        "request-2",
        "coding",
        1_700_000_000.0,
        public_envelope,
        &[
            Event::EncryptedReasoning {
                output_index: 0,
                encrypted_content: "blob==".to_string(),
            },
            Event::TextDelta("answer".to_string()),
            Event::Completed,
        ],
    )
    .expect("completed body must encode");
    assert_eq!(bare.body["output"][0]["type"], json!("reasoning"));
    assert_eq!(bare.body["output"][0]["encrypted_content"], json!("blob=="));
    assert_eq!(bare.body["output"][0]["summary"], json!([]));

    let hidden = completed_responses_body(
        "request-3",
        "coding",
        1_700_000_000.0,
        ResponsesEnvelope::default(),
        &[
            Event::EncryptedReasoning {
                output_index: 0,
                encrypted_content: "internal-only".to_string(),
            },
            Event::TextDelta("answer".to_string()),
            Event::Completed,
        ],
    )
    .expect("internal reasoning must not break public encoding");
    assert_eq!(hidden.body["output"][0]["type"], json!("reasoning"));
    assert_eq!(hidden.body["output"][0]["encrypted_content"], Value::Null);
}
