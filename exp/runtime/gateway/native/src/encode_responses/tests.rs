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

#[test]
fn encrypted_reasoning_lands_on_the_completed_reasoning_item() {
    let completed = completed_responses_body(
        "request-1",
        "coding",
        1_700_000_000.0,
        ResponsesEnvelope::default(),
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
        ResponsesEnvelope::default(),
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
}
