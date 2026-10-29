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
        ResponsesSseEncoder::new("request-1", "coding", 1_700_000_000, envelope.clone());
    let frames = encoder.start().expect("stream start must encode");
    assert!(frames[0].contains("\"x-experiential-ignored-parameters\":[\"top_k\"]"));

    let completed = completed_responses_body(
        "request-1",
        "coding",
        1_700_000_000,
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
        1_700_000_000,
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
            custom: false,
            namespace: None,
            caller: None,
            index: 0,
            call_id: "call-one".to_string(),
            name: "lookup".to_string(),
            thought_signature: None,
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: "{}".to_string(),
        },
        Event::ToolCallCompleted {
            index: 0,
            call: CompletedToolCall {
                namespace: None,
                caller: None,
                call_id: "call-one".to_string(),
                name: "lookup".to_string(),
                raw_arguments: "{}".to_string(),
                provider_item_id: None,
                provider_status: None,
                custom: false,
                thought_signature: None,
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
        ResponsesSseEncoder::new("request-1", "coding", 1_700_000_000, envelope.clone());
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
        1_700_000_000,
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
        1_700_000_000,
        ResponsesEnvelope::default(),
        &fireworks_tool_events(),
    )
    .is_err());
}

/// The provider events of a Codex `apply_patch` turn after the native-tool
/// inverter restored the custom call: freeform input, no JSON object.
fn fireworks_custom_tool_events() -> Vec<Event> {
    let patch = "*** Begin Patch\n*** End Patch\n";
    vec![
        Event::ReasoningContentDelta {
            route_sha256: "a".repeat(64),
            delta: "hidden provider reasoning".to_string(),
        },
        Event::ToolCallStarted {
            custom: true,
            namespace: None,
            caller: None,
            index: 0,
            call_id: "call-patch".to_string(),
            name: "apply_patch".to_string(),
            thought_signature: None,
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: patch.to_string(),
        },
        Event::ToolCallCompleted {
            index: 0,
            call: CompletedToolCall {
                namespace: None,
                caller: None,
                call_id: "call-patch".to_string(),
                name: "apply_patch".to_string(),
                raw_arguments: patch.to_string(),
                provider_item_id: None,
                provider_status: None,
                custom: true,
                thought_signature: None,
            },
        },
        Event::Completed,
    ]
}

#[test]
fn fireworks_custom_tool_turn_serves_without_a_carrier() {
    // Production 2026-10-19: every completed Codex apply_patch turn on an
    // Experiential Cloud reasoning lane failed at the terminal because the
    // carrier sealer tried to digest the freeform patch as JSON arguments.
    let events = fireworks_custom_tool_events();
    assert!(reasoning_carrier_candidate(&events)
        .expect("provider events must validate")
        .is_none());
    let envelope = ResponsesEnvelope {
        include_encrypted_reasoning: true,
        ..ResponsesEnvelope::default()
    };
    let mut encoder =
        ResponsesSseEncoder::new("request-1", "coding", 1_700_000_000, envelope.clone());
    encoder.start().expect("stream start must encode");
    let mut frames = Vec::new();
    for event in &events {
        frames.extend(encoder.feed(event).expect("Responses event must encode"));
    }
    let public = frames.join("");
    assert!(public.contains("response.completed"));
    assert!(public.contains("response.custom_tool_call_input.done"));
    assert!(!public.contains("hidden provider reasoning"));
    assert!(!public.contains("encrypted_content\":\""));
    assert!(
        completed_responses_body("request-1", "coding", 1_700_000_000, envelope, &events).is_ok()
    );
}

#[test]
fn fireworks_parallel_tools_share_public_and_carrier_order() {
    let events = vec![
        Event::ReasoningContentDelta {
            route_sha256: "a".repeat(64),
            delta: "hidden".to_string(),
        },
        Event::ToolCallStarted {
            custom: false,
            namespace: None,
            caller: None,
            index: 1,
            call_id: "call-one".to_string(),
            name: "first".to_string(),
            thought_signature: None,
        },
        Event::ToolCallStarted {
            custom: false,
            namespace: None,
            caller: None,
            index: 0,
            call_id: "call-zero".to_string(),
            name: "second".to_string(),
            thought_signature: None,
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
                namespace: None,
                caller: None,
                call_id: "call-zero".to_string(),
                name: "second".to_string(),
                raw_arguments: "{\"order\":0}".to_string(),
                provider_item_id: None,
                provider_status: None,
                custom: false,
                thought_signature: None,
            },
        },
        Event::ToolCallCompleted {
            index: 1,
            call: CompletedToolCall {
                namespace: None,
                caller: None,
                call_id: "call-one".to_string(),
                name: "first".to_string(),
                raw_arguments: "{\"order\":1}".to_string(),
                provider_item_id: None,
                provider_status: None,
                custom: false,
                thought_signature: None,
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
        1_700_000_000,
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
        1_700_000_000,
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
        1_700_000_000,
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
        1_700_000_000,
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
            1_700_000_000,
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
            custom: false,
            namespace: None,
            caller: None,
            index: 1,
            call_id: "call-1".to_string(),
            name: "lookup".to_string(),
            thought_signature: None,
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
                namespace: None,
                caller: None,
                call_id: "call-1".to_string(),
                name: "lookup".to_string(),
                provider_item_id: Some("fc-provider-1".to_string()),
                provider_status: None,
                raw_arguments: "{}".to_string(),
                custom: false,
                thought_signature: None,
            },
        },
        Event::Completed,
    ];
    let completed = completed_responses_body(
        "request-provider-order",
        "coding",
        1_700_000_000,
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
        1_700_000_000,
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
            custom: false,
            namespace: None,
            caller: None,
            index: 2,
            call_id: "call-required".to_string(),
            name: "lookup".to_string(),
            thought_signature: None,
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
                namespace: None,
                caller: None,
                call_id: "call-required".to_string(),
                name: "lookup".to_string(),
                provider_item_id: None,
                provider_status: Some(ProviderOutputItemStatus::Incomplete),
                raw_arguments: "{}".to_string(),
                custom: false,
                thought_signature: None,
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
        1_700_000_000,
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
        1_700_000_000,
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

#[test]
fn hosted_tool_items_reemit_verbatim_at_remapped_indexes() {
    let mut encoder = ResponsesSseEncoder::new(
        "request-1",
        "coding",
        1_700_000_000,
        ResponsesEnvelope::default(),
    );
    encoder.start().expect("stream start must encode");
    // The provider's own index is 4; the public stream re-numbers from 0.
    let started = encoder
        .feed(&Event::HostedToolItemStarted {
            output_index: 4,
            item_id: "ws_1".to_string(),
            item_type: "web_search_call".to_string(),
            item: "{\"id\":\"ws_1\",\"type\":\"web_search_call\",\"status\":\"in_progress\"}"
                .to_string(),
        })
        .expect("hosted start encodes");
    assert!(started[0].contains("response.output_item.added"));
    assert!(started[0].contains("\"output_index\":0"));
    assert!(started[0].contains("\"type\":\"web_search_call\""));
    let progress = encoder
        .feed(&Event::HostedToolItemProgress {
            output_index: 4,
            item_id: "ws_1".to_string(),
            event_type: "response.web_search_call.searching".to_string(),
            payload: "{\"type\":\"response.web_search_call.searching\",\"item_id\":\"ws_1\",\
                      \"output_index\":4,\"sequence_number\":9}"
                .to_string(),
        })
        .expect("hosted progress encodes");
    assert!(progress[0].starts_with("event: response.web_search_call.searching\n"));
    assert!(progress[0].contains("\"output_index\":0"));
    assert!(progress[0].contains("\"item_id\":\"ws_1\""));
    assert!(
        !progress[0].contains("\"sequence_number\":9"),
        "the provider's sequence number must be re-stamped: {}",
        progress[0]
    );
    let done = encoder
        .feed(&Event::HostedToolItemCompleted {
            output_index: 4,
            item_id: "ws_1".to_string(),
            item_type: "web_search_call".to_string(),
            item: "{\"id\":\"ws_1\",\"type\":\"web_search_call\",\"status\":\"completed\",\
                   \"action\":{\"type\":\"search\",\"query\":\"pi\"}}"
                .to_string(),
        })
        .expect("hosted completion encodes");
    assert!(done[0].contains("response.output_item.done"));
    assert!(done[0].contains("\"query\":\"pi\""));
    let terminal = encoder.feed(&Event::Completed).expect("terminal encodes");
    let last = terminal.last().expect("terminal frame");
    assert!(last.contains("response.completed"));
    assert!(last.contains("\"query\":\"pi\""));
}

/// The non-streaming body carries hosted items in slot order, and hosted
/// invocations join the ledger's tool names by item type.
#[test]
fn completed_body_serves_hosted_items_in_order() {
    let events = vec![
        Event::HostedToolItemStarted {
            output_index: 0,
            item_id: "mcp_1".to_string(),
            item_type: "mcp_call".to_string(),
            item: "{\"id\":\"mcp_1\",\"type\":\"mcp_call\",\"server_label\":\"wiki\",\
                   \"name\":\"ask\",\"arguments\":\"\"}"
                .to_string(),
        },
        Event::HostedToolItemCompleted {
            output_index: 0,
            item_id: "mcp_1".to_string(),
            item_type: "mcp_call".to_string(),
            item: "{\"id\":\"mcp_1\",\"type\":\"mcp_call\",\"server_label\":\"wiki\",\
                   \"name\":\"ask\",\"arguments\":\"{}\",\"output\":\"42\"}"
                .to_string(),
        },
        Event::ProviderOutputItemStarted {
            output_index: 1,
            item_id: Some("msg_1".to_string()),
            kind: ProviderOutputItemKind::Message,
            status: None,
            phase: None,
        },
        Event::ProviderTextDelta {
            output_index: 1,
            item_id: "msg_1".to_string(),
            delta: "The answer is 42.".to_string(),
        },
        Event::Completed,
    ];
    let aggregated = completed_responses_body(
        "request-1",
        "coding",
        1_700_000_000,
        ResponsesEnvelope::default(),
        &events,
    )
    .expect("completed body must encode");
    let output = aggregated.body["output"].as_array().expect("output array");
    assert_eq!(output[0]["type"], "mcp_call");
    assert_eq!(output[0]["output"], "42");
    assert_eq!(output[1]["type"], "message");
    assert_eq!(aggregated.tool_names, vec!["mcp_call".to_string()]);
}

/// Provider text annotations attach to the message's text part on both the
/// streaming and final shapes.
#[test]
fn provider_annotations_attach_to_the_message_text_part() {
    let mut encoder = ResponsesSseEncoder::new(
        "request-1",
        "coding",
        1_700_000_000,
        ResponsesEnvelope::default(),
    );
    encoder.start().expect("stream start must encode");
    encoder
        .feed(&Event::ProviderTextDelta {
            output_index: 0,
            item_id: "msg_1".to_string(),
            delta: "Cited answer.".to_string(),
        })
        .expect("text encodes");
    let frames = encoder
        .feed(&Event::ProviderTextAnnotation {
            output_index: 0,
            item_id: "msg_1".to_string(),
            annotation: "{\"type\":\"url_citation\",\"url\":\"https://example.com\",\
                         \"title\":\"Example\",\"start_index\":0,\"end_index\":13}"
                .to_string(),
        })
        .expect("annotation encodes");
    assert!(frames[0].contains("response.output_text.annotation.added"));
    assert!(frames[0].contains("\"annotation_index\":0"));
    assert!(frames[0].contains("https://example.com"));
    let terminal = encoder.feed(&Event::Completed).expect("terminal encodes");
    let last = terminal.last().expect("terminal frame");
    assert!(
        last.contains("\"annotations\":[{\"type\":\"url_citation\""),
        "the final message part must keep its annotations: {last}"
    );
}

#[test]
fn terminal_probability_observation_is_included_once_in_aggregate_output() {
    let body = completed_responses_body(
        "request-probability",
        "coding",
        1_700_000_000,
        ResponsesEnvelope::default(),
        &[
            Event::ProviderOutputItemStarted {
                output_index: 0,
                item_id: Some("msg_1".to_string()),
                kind: ProviderOutputItemKind::Message,
                status: None,
                phase: None,
            },
            Event::ProviderTextDelta {
                output_index: 0,
                item_id: "msg_1".to_string(),
                delta: "OK".to_string(),
            },
            Event::ProviderResponsesLogprobs {
                output_index: 0,
                item_id: "msg_1".to_string(),
                content_index: 0,
                phase: "terminal".to_string(),
                records: json!([{"token":"OK","logprob":-0.125,"bytes":[79,75]}]),
            },
            Event::Completed,
        ],
    )
    .expect("aggregate output encodes");
    let content = &body.body["output"][0]["content"][0];
    assert_eq!(content["logprobs"][0]["token"], "OK");
    assert_eq!(content["logprobs"][0]["bytes"], json!([79, 75]));
}

/// Results, approvals, and opaque conversation items are served in the
/// output but never recorded as invoked tools: only `*_call` item types
/// name an invocation that actually occurred.
#[test]
fn hosted_non_call_items_never_join_the_ledger_tool_names() {
    let events = vec![
        Event::HostedToolItemStarted {
            output_index: 0,
            item_id: "cmp_1".to_string(),
            item_type: "compaction".to_string(),
            item: "{\"id\":\"cmp_1\",\"type\":\"compaction\",\"encrypted_content\":\"opaque\"}"
                .to_string(),
        },
        Event::HostedToolItemCompleted {
            output_index: 0,
            item_id: "cmp_1".to_string(),
            item_type: "compaction".to_string(),
            item: "{\"id\":\"cmp_1\",\"type\":\"compaction\",\"encrypted_content\":\"opaque\"}"
                .to_string(),
        },
        Event::HostedToolItemStarted {
            output_index: 1,
            item_id: "ws_1".to_string(),
            item_type: "web_search_call".to_string(),
            item: "{\"id\":\"ws_1\",\"type\":\"web_search_call\",\"status\":\"completed\"}"
                .to_string(),
        },
        Event::HostedToolItemCompleted {
            output_index: 1,
            item_id: "ws_1".to_string(),
            item_type: "web_search_call".to_string(),
            item: "{\"id\":\"ws_1\",\"type\":\"web_search_call\",\"status\":\"completed\"}"
                .to_string(),
        },
        Event::Completed,
    ];
    let aggregated = completed_responses_body(
        "request-1",
        "coding",
        1_700_000_000,
        ResponsesEnvelope::default(),
        &events,
    )
    .expect("completed body must encode");
    assert_eq!(aggregated.tool_names, vec!["web_search_call".to_string()]);
    let mut usage = None;
    let mut tool_names = Vec::new();
    for event in &events {
        crate::relay::track_event(event, &mut usage, &mut tool_names);
    }
    assert_eq!(tool_names, vec!["web_search_call".to_string()]);
}

#[path = "probabilities_tests.rs"]
mod probabilities_tests;

#[path = "tool_attribution_tests.rs"]
mod tool_attribution_tests;

/// Azure Foundry's DeepSeek output filter streams text and then ends the
/// turn with `finish_reason: "content_filter"`, which the compatible dialect
/// reads as an empty refusal after the text (live 2026-10-20). The Responses
/// surface serves it as chat does, the text part then a refusal part, never
/// as a malformed-stream 502.
#[test]
fn refusal_after_text_opens_a_second_content_part() {
    let mut events = vec![
        Event::TextDelta("{\"page_type\": \"st".to_string()),
        Event::RefusalDelta(String::new()),
        Event::Failed(crate::errors::Failure::refusal(
            crate::errors::RefusalReason::ContentPolicy,
        )),
    ];
    let mut encoder = ResponsesSseEncoder::new(
        "request-1",
        "coding",
        1_700_000_000,
        ResponsesEnvelope::default(),
    );
    encoder.start().expect("stream start must encode");
    encoder.feed(&events[0]).expect("text delta encodes");
    let frames = encoder
        .feed(&events[1])
        .expect("a refusal after text is served, not refused");
    assert!(frames
        .iter()
        .any(|frame| frame.contains("response.content_part.added")
            && frame.contains("\"content_index\":1")
            && frame.contains("\"type\":\"refusal\"")));
    assert!(frames
        .iter()
        .any(|frame| frame.contains("response.refusal.delta")
            && frame.contains("\"content_index\":1")));

    let refusal = crate::respond::complete_visible_refusal(&mut events);
    assert!(refusal.is_some(), "the ledger still records the refusal");
    let completed = completed_responses_body(
        "request-1",
        "coding",
        1_700_000_000,
        ResponsesEnvelope::default(),
        &events,
    )
    .expect("completed body must encode");
    assert!(completed.failure.is_none());
    assert_eq!(
        completed.body["output"][0]["content"],
        json!([
            {"type": "output_text", "text": "{\"page_type\": \"st", "annotations": []},
            {"type": "refusal", "refusal": ""},
        ])
    );
}

#[test]
fn text_after_a_refusal_still_fails_closed() {
    let mut encoder = ResponsesSseEncoder::new(
        "request-1",
        "coding",
        1_700_000_000,
        ResponsesEnvelope::default(),
    );
    encoder.start().expect("stream start must encode");
    encoder
        .feed(&Event::RefusalDelta("I can't help with that.".to_string()))
        .expect("refusal delta encodes");
    assert!(encoder.feed(&Event::TextDelta("Sure".to_string())).is_err());
}

#[path = "item_order_tests.rs"]
mod item_order_tests;
