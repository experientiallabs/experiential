//! Inline tests for `encode_responses`, split into a submodule file so the
//! implementation stays within the repository line budget.

use super::*;
use crate::encode::reasoning_carrier_candidate;
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
            name: "lookup".to_string(),
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: "{}".to_string(),
        },
        Event::ToolCallCompleted {
            index: 0,
            call: CompletedToolCall {
                call_id: "call-one".to_string(),
                name: "lookup".to_string(),
                raw_arguments: "{}".to_string(),
                provider_item_id: None,
                provider_status: None,
            },
        },
        Event::Completed,
    ]
}

#[test]
fn fireworks_responses_reasoning_round_trips_as_encrypted_content() {
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
fn fireworks_responses_reasoning_fails_closed_without_a_sealed_carrier() {
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
fn fireworks_parallel_tools_share_public_and_carrier_order() {
    let events = vec![
        Event::ReasoningContentDelta {
            route_sha256: "a".repeat(64),
            delta: "hidden".to_string(),
        },
        Event::ToolCallStarted {
            index: 1,
            call_id: "call-one".to_string(),
            name: "first".to_string(),
        },
        Event::ToolCallStarted {
            index: 0,
            call_id: "call-zero".to_string(),
            name: "second".to_string(),
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: "{\"order\":0}".to_string(),
        },
        Event::ToolArgumentsDelta {
            index: 1,
            delta: "{\"order\":1}".to_string(),
        },
        Event::ToolCallCompleted {
            index: 0,
            call: CompletedToolCall {
                call_id: "call-zero".to_string(),
                name: "second".to_string(),
                raw_arguments: "{\"order\":0}".to_string(),
                provider_item_id: None,
                provider_status: None,
            },
        },
        Event::ToolCallCompleted {
            index: 1,
            call: CompletedToolCall {
                call_id: "call-one".to_string(),
                name: "first".to_string(),
                raw_arguments: "{\"order\":1}".to_string(),
                provider_item_id: None,
                provider_status: None,
            },
        },
        Event::Completed,
    ];
    let candidate = reasoning_carrier_candidate(&events)
        .expect("provider events must validate")
        .expect("reasoning plus tools must produce a carrier");
    let completed = completed_responses_body_with_carrier(
        "request-1",
        "coding",
        1_700_000_000.0,
        ResponsesEnvelope::default(),
        &events,
        Some("authenticated-carrier-v2"),
    )
    .expect("Responses output must encode");
    let output = completed.body["output"]
        .as_array()
        .expect("Responses output must be an array");
    let public_calls = output
        .iter()
        .filter(|item| item["type"] == "function_call")
        .map(|item| item["call_id"].as_str().expect("call ID must be text"))
        .collect::<Vec<_>>();
    let carrier_calls = candidate
        .tool_calls
        .iter()
        .map(|call| call.call_id.as_str())
        .collect::<Vec<_>>();

    assert_eq!(public_calls, vec!["call-one", "call-zero"]);
    assert_eq!(carrier_calls, public_calls);
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
                item_id: "rs-1".to_string(),
                delta: "planned".to_string(),
            },
            Event::EncryptedReasoning {
                output_index: 0,
                item_id: "rs-1".to_string(),
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
                item_id: "rs-2".to_string(),
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
                item_id: "rs-3".to_string(),
                encrypted_content: "internal-only".to_string(),
            },
            Event::TextDelta("answer".to_string()),
            Event::Completed,
        ],
    )
    .expect("internal reasoning must not break public encoding");
    assert_eq!(hidden.body["output"][0]["type"], json!("reasoning"));
    assert_eq!(hidden.body["output"][0]["encrypted_content"], Value::Null);

    for (include_encrypted_reasoning, expected) in [(false, None), (true, Some("stream-opaque"))] {
        let mut encoder = ResponsesSseEncoder::new(
            "request-stream",
            "coding",
            1_700_000_000.0,
            ResponsesEnvelope {
                include_encrypted_reasoning,
                ..ResponsesEnvelope::default()
            },
        );
        encoder.start().expect("stream start must encode");
        let mut frames = encoder
            .feed(&Event::EncryptedReasoning {
                output_index: 0,
                item_id: "rs-stream".to_string(),
                encrypted_content: "stream-opaque".to_string(),
            })
            .expect("encrypted reasoning must encode");
        frames.extend(
            encoder
                .feed(&Event::Completed)
                .expect("terminal must encode"),
        );
        let payloads: Vec<Value> = frames
            .iter()
            .filter_map(|frame| frame.split_once("data: ").map(|(_, data)| data.trim()))
            .map(|data| serde_json::from_str(data).expect("frame data must be JSON"))
            .collect();
        let done = payloads
            .iter()
            .find(|payload| payload["type"] == json!("response.output_item.done"))
            .expect("reasoning item must complete");
        let terminal = payloads.last().expect("response must terminate");
        for item in [&done["item"], &terminal["response"]["output"][0]] {
            match expected {
                Some(value) => assert_eq!(item["encrypted_content"], json!(value)),
                None => assert!(item.get("encrypted_content").is_none()),
            }
        }
    }
}

#[test]
fn provider_item_starts_preserve_reasoning_tool_order_and_identity() {
    let events = vec![
        Event::ProviderOutputItemStarted {
            output_index: 0,
            item_id: Some("rs-provider-0".to_string()),
            kind: ProviderOutputItemKind::Reasoning,
            status: None,
            phase: None,
        },
        Event::ProviderOutputItemStarted {
            output_index: 1,
            item_id: Some("fc-provider-1".to_string()),
            kind: ProviderOutputItemKind::FunctionCall,
            status: None,
            phase: None,
        },
        Event::ToolCallStarted {
            index: 1,
            call_id: "call-1".to_string(),
            name: "lookup".to_string(),
        },
        Event::ToolArgumentsDelta {
            index: 1,
            delta: "{}".to_string(),
        },
        Event::EncryptedReasoning {
            output_index: 0,
            item_id: "rs-provider-0".to_string(),
            encrypted_content: "opaque".to_string(),
        },
        Event::ToolCallCompleted {
            index: 1,
            call: CompletedToolCall {
                call_id: "call-1".to_string(),
                name: "lookup".to_string(),
                provider_item_id: Some("fc-provider-1".to_string()),
                provider_status: None,
                raw_arguments: "{}".to_string(),
            },
        },
        Event::Completed,
    ];
    let completed = completed_responses_body(
        "request-provider-order",
        "coding",
        1_700_000_000.0,
        ResponsesEnvelope::default(),
        &events,
    )
    .expect("provider order must encode");
    let output = completed.body["output"].as_array().expect("output array");
    assert_eq!(output[0]["type"], json!("reasoning"));
    assert_eq!(output[0]["id"], json!("rs-provider-0"));
    assert_eq!(output[1]["type"], json!("function_call"));
    assert_eq!(output[1]["id"], json!("fc-provider-1"));

    let mut encoder = ResponsesSseEncoder::new(
        "request-provider-order",
        "coding",
        1_700_000_000.0,
        ResponsesEnvelope::default(),
    );
    encoder.start().expect("stream starts");
    let mut frames = Vec::new();
    for event in &events {
        frames.extend(encoder.feed(event).expect("event must encode"));
    }
    let added: Vec<Value> = frames
        .iter()
        .filter_map(|frame| frame.split_once("data: ").map(|(_, data)| data.trim()))
        .map(|data| serde_json::from_str(data).expect("frame data must be JSON"))
        .filter(|payload: &Value| payload["type"] == json!("response.output_item.added"))
        .collect();
    assert_eq!(added[0]["item"]["id"], json!("rs-provider-0"));
    assert_eq!(added[0]["output_index"], json!(0));
    assert_eq!(added[1]["item"]["id"], json!("fc-provider-1"));
    assert_eq!(added[1]["output_index"], json!(1));
}

#[test]
fn provider_items_preserve_multiple_messages_status_phase_and_idless_call() {
    let events = vec![
        Event::ProviderOutputItemStarted {
            output_index: 0,
            item_id: Some("rs-0".to_string()),
            kind: ProviderOutputItemKind::Reasoning,
            status: Some(ProviderOutputItemStatus::InProgress),
            phase: None,
        },
        Event::EncryptedReasoning {
            output_index: 0,
            item_id: "rs-0".to_string(),
            encrypted_content: "opaque".to_string(),
        },
        Event::ProviderOutputItemCompleted {
            output_index: 0,
            item_id: Some("rs-0".to_string()),
            kind: ProviderOutputItemKind::Reasoning,
            status: Some(ProviderOutputItemStatus::Incomplete),
            phase: None,
        },
        Event::ProviderOutputItemStarted {
            output_index: 1,
            item_id: Some("msg-commentary".to_string()),
            kind: ProviderOutputItemKind::Message,
            status: Some(ProviderOutputItemStatus::InProgress),
            phase: Some(ProviderAssistantMessagePhase::Commentary),
        },
        Event::ProviderTextDelta {
            output_index: 1,
            item_id: "msg-commentary".to_string(),
            delta: "Checking.".to_string(),
        },
        Event::ProviderOutputItemCompleted {
            output_index: 1,
            item_id: Some("msg-commentary".to_string()),
            kind: ProviderOutputItemKind::Message,
            status: Some(ProviderOutputItemStatus::Incomplete),
            phase: Some(ProviderAssistantMessagePhase::Commentary),
        },
        Event::ProviderOutputItemStarted {
            output_index: 2,
            item_id: None,
            kind: ProviderOutputItemKind::FunctionCall,
            status: Some(ProviderOutputItemStatus::InProgress),
            phase: None,
        },
        Event::ToolCallStarted {
            index: 2,
            call_id: "call-required".to_string(),
            name: "lookup".to_string(),
        },
        Event::ToolArgumentsDelta {
            index: 2,
            delta: "{}".to_string(),
        },
        Event::ProviderOutputItemCompleted {
            output_index: 2,
            item_id: None,
            kind: ProviderOutputItemKind::FunctionCall,
            status: Some(ProviderOutputItemStatus::Incomplete),
            phase: None,
        },
        Event::ToolCallCompleted {
            index: 2,
            call: CompletedToolCall {
                call_id: "call-required".to_string(),
                name: "lookup".to_string(),
                provider_item_id: None,
                provider_status: Some(ProviderOutputItemStatus::Incomplete),
                raw_arguments: "{}".to_string(),
            },
        },
        Event::ProviderOutputItemStarted {
            output_index: 3,
            item_id: Some("msg-final".to_string()),
            kind: ProviderOutputItemKind::Message,
            status: Some(ProviderOutputItemStatus::InProgress),
            phase: Some(ProviderAssistantMessagePhase::FinalAnswer),
        },
        Event::ProviderTextDelta {
            output_index: 3,
            item_id: "msg-final".to_string(),
            delta: "Done.".to_string(),
        },
        Event::ProviderOutputItemCompleted {
            output_index: 3,
            item_id: Some("msg-final".to_string()),
            kind: ProviderOutputItemKind::Message,
            status: Some(ProviderOutputItemStatus::Completed),
            phase: Some(ProviderAssistantMessagePhase::FinalAnswer),
        },
        Event::Incomplete,
    ];
    let completed = completed_responses_body(
        "request-provider-fields",
        "coding",
        1_700_000_000.0,
        ResponsesEnvelope {
            include_encrypted_reasoning: true,
            ..ResponsesEnvelope::default()
        },
        &events,
    )
    .expect("provider fields must encode");
    let output = completed.body["output"].as_array().expect("output array");
    assert_eq!(output.len(), 4);
    assert_eq!(output[0]["status"], json!("incomplete"));
    assert_eq!(output[1]["phase"], json!("commentary"));
    assert_eq!(output[1]["status"], json!("incomplete"));
    assert!(output[2].get("id").is_none());
    assert_eq!(output[2]["call_id"], json!("call-required"));
    assert_eq!(output[2]["status"], json!("incomplete"));
    assert_eq!(output[3]["phase"], json!("final_answer"));
    assert_eq!(output[3]["status"], json!("completed"));

    let mut encoder = ResponsesSseEncoder::new(
        "request-provider-fields",
        "coding",
        1_700_000_000.0,
        ResponsesEnvelope::default(),
    );
    let mut frames = encoder.start().expect("stream starts");
    for event in &events {
        frames.extend(encoder.feed(event).expect("event encodes"));
    }
    assert!(!frames
        .iter()
        .any(|frame| frame.contains("response.function_call_arguments")));
}
