use super::*;
use crate::capture::record::{Protocol, Record, Request, Response, Scope};
use serde_json::{json, Value};
use std::sync::Arc;

fn record(context: Value, response: Option<Response>, transport: Option<Value>) -> Record {
    Record {
        checkpointed: false,
        schema_version: 1,
        request: Request {
            request_id: "request-1".into(),
            scope: Scope {
                organization_id: "org-1".into(),
                identity_id: "identity".into(),
                application_id: "alias".into(),
            },
            protocol: Protocol::ChatCompletions,
            model_id: Some("model-a".into()),
            context: Arc::new(context),
        },
        response,
        transport,
        provider_reasoning: None,
        provider_reasoning_source_json: None,
        provider_tool_calls_json: None,
        deployment_id: Some("deployment-a".into()),
        metrics: None,
        gemini_thought_parts: Vec::new(),
        gemini_thought_parts_source_json: None,
        provider_logprobs: None,
        captured_at: 1_700_000_000.5,
        transport_body: None,
        messages_part: None,
    }
}

fn sse() -> Response {
    Response::Sse {
        status: 200,
        frames: vec![json!({"choices": [{"delta": {"content": "雪 \"quoted\""}}]})],
        truncated: true,
        client_disconnected: false,
        source_json: None,
    }
}

/// Apply a storage plan by splicing spans (the destination's cheap path).
fn splice(bytes: &str, spans: &Spans, keep_prompt: bool, keep_response: bool) -> String {
    let mut edits: Vec<([usize; 2], &str)> = Vec::new();
    if !keep_prompt {
        edits.extend(spans.messages.map(|span| (span, "[]")));
        for span in [
            spans.source_json,
            spans.provider_context,
            spans.wire_request,
        ]
        .into_iter()
        .flatten()
        {
            edits.push((span, ""));
        }
    }
    if !keep_response {
        edits.extend(spans.response.map(|span| (span, "null")));
    }
    edits.sort_by_key(|(span, _)| std::cmp::Reverse(span[0]));
    let mut out = bytes.to_owned();
    for ([start, end], replacement) in edits {
        out.replace_range(start..end, replacement);
    }
    out
}

/// Apply the same plan by decoding and rewriting the record.
fn rewrite(bytes: &str, keep_prompt: bool, keep_response: bool) -> Value {
    let mut record: Value = serde_json::from_str(bytes).unwrap();
    if !keep_response {
        record["response"] = Value::Null;
    }
    if !keep_prompt {
        let context = record["request"]["context"].as_object_mut().unwrap();
        context.remove("source_json");
        context.remove("provider_context");
        record["request"]["context"]["request"]["messages"] = json!([]);
        if let Some(transport) = record.get_mut("transport").and_then(Value::as_object_mut) {
            transport.remove("wire_request");
        }
    }
    record
}

fn contexts() -> Vec<Value> {
    vec![
        json!({"schema_version": 1, "request": {"messages": [{"role": "user", "content": "hi"}]}}),
        // The repeated members first, in the middle, and last in their object.
        json!({"source_json": "{\"a\":1}", "schema_version": 1,
               "request": {"model": "m", "messages": [{"role": "user", "content": "雪,}"}], "n": 1},
               "provider_context": {"reasoning": "r"}}),
        json!({"provider_context": [1, 2], "request": {"messages": []}, "schema_version": 1,
               "source_json": "x"}),
        json!({"schema_version": 1, "provider_context": null, "source_json": "only",
               "request": {"messages": "not-a-list", "tools": []}}),
    ]
}

fn transports() -> Vec<Option<Value>> {
    vec![
        None,
        Some(json!({"wire_request": {"body": {"k": "v"}}})),
        Some(json!({"wire_request": {"body": "b"}, "timing": {"total_ms": 3}})),
        Some(json!({"headers": {"a": "b"}, "wire_request": {"body": "b"}})),
    ]
}

#[test]
fn encoded_bytes_are_identical_to_the_plain_encoding() {
    for context in contexts() {
        for transport in transports() {
            for response in [None, Some(sse())] {
                let record = record(context.clone(), response, transport.clone());
                let (encoded, _) = encode_with_facts(&record, 1 << 20, false).unwrap();
                assert_eq!(encoded, record.encode(1 << 20).unwrap());
            }
        }
    }
}

#[test]
fn splicing_spans_matches_a_decoded_rewrite_for_every_plan() {
    for context in contexts() {
        for transport in transports() {
            for response in [None, Some(sse())] {
                let record = record(context.clone(), response, transport.clone());
                let (encoded, facts) = encode_with_facts(&record, 1 << 20, false).unwrap();
                let facts: Value = serde_json::from_slice(&facts).unwrap();
                let spans: Spans = Spans {
                    messages: serde_json::from_value(facts["spans"]["messages"].clone()).unwrap(),
                    response: serde_json::from_value(facts["spans"]["response"].clone()).unwrap(),
                    source_json: serde_json::from_value(facts["spans"]["source_json"].clone())
                        .unwrap(),
                    provider_context: serde_json::from_value(
                        facts["spans"]["provider_context"].clone(),
                    )
                    .unwrap(),
                    wire_request: serde_json::from_value(facts["spans"]["wire_request"].clone())
                        .unwrap(),
                };
                for (keep_prompt, keep_response) in
                    [(true, true), (false, true), (true, false), (false, false)]
                {
                    let spliced = splice(&encoded, &spans, keep_prompt, keep_response);
                    let parsed: Value = serde_json::from_str(&spliced)
                        .unwrap_or_else(|error| panic!("{error}: {spliced}"));
                    assert_eq!(parsed, rewrite(&encoded, keep_prompt, keep_response));
                }
            }
        }
    }
}

#[test]
fn spans_point_at_exactly_the_values_they_name() {
    let record = record(contexts()[1].clone(), Some(sse()), transports()[2].clone());
    let (encoded, facts) = encode_with_facts(&record, 1 << 20, false).unwrap();
    let facts: Value = serde_json::from_slice(&facts).unwrap();
    let slice = |name: &str| {
        let span = &facts["spans"][name];
        let (start, end) = (span[0].as_u64().unwrap(), span[1].as_u64().unwrap());
        &encoded[start as usize..end as usize]
    };
    assert_eq!(slice("messages"), r#"[{"role":"user","content":"雪,}"}]"#);
    let response: Value = serde_json::from_str(slice("response")).unwrap();
    assert_eq!(response["kind"], "sse");
    assert_eq!(slice("source_json"), r#""source_json":"{\"a\":1}","#);
    assert_eq!(
        slice("provider_context"),
        r#","provider_context":{"reasoning":"r"}"#
    );
    assert_eq!(slice("wire_request"), r#""wire_request":{"body":"b"},"#);
}

#[test]
fn facts_carry_identity_and_response_flags_without_content() {
    let record = record(contexts()[0].clone(), Some(sse()), None);
    let (encoded, facts) = encode_with_facts(&record, 1 << 20, false).unwrap();
    let facts: Value = serde_json::from_slice(&facts).unwrap();
    assert_eq!(facts["facts_version"], FACTS_VERSION);
    assert_eq!(facts["schema_version"], 1);
    assert_eq!(facts["request_id"], "request-1");
    assert_eq!(
        facts["scope"],
        json!({"organization_id": "org-1", "identity_id": "identity", "application_id": "alias"})
    );
    assert_eq!(facts["protocol"], "chat_completions");
    assert_eq!(facts["model_id"], "model-a");
    assert_eq!(facts["deployment_id"], "deployment-a");
    assert_eq!(facts["captured_at"], 1_700_000_000.5);
    assert_eq!(facts["complete"], true);
    assert_eq!(
        facts["response"],
        json!({"kind": "sse", "status": 200, "truncated": true, "client_disconnected": false})
    );
    assert_eq!(facts["split"], false);
    assert_eq!(facts["record_bytes"], encoded.len());
    assert!(!facts.to_string().contains("雪"));
    assert!(!facts.to_string().contains("hi"));
}

#[test]
fn a_prompt_only_record_has_no_response_facts_or_span() {
    let record = record(contexts()[0].clone(), None, None);
    let (_, facts) = encode_with_facts(&record, 1 << 20, false).unwrap();
    let facts: Value = serde_json::from_slice(&facts).unwrap();
    assert_eq!(facts["complete"], false);
    assert_eq!(facts["response"], Value::Null);
    assert_eq!(facts["spans"]["response"], Value::Null);
    assert!(facts["spans"]["messages"].is_array());
}

#[test]
fn a_completion_reference_matches_its_update_encoding_and_has_no_prompt_spans() {
    let mut record = record(contexts()[1].clone(), Some(sse()), transports()[1].clone());
    record.checkpointed = true;
    let (encoded, facts) = encode_with_facts(&record, 1 << 20, true).unwrap();
    assert_eq!(encoded, record.encode_update(1 << 20).unwrap());
    let facts: Value = serde_json::from_slice(&facts).unwrap();
    assert_eq!(facts["schema_version"], 2);
    assert_eq!(facts["spans"]["messages"], Value::Null);
    assert_eq!(facts["spans"]["source_json"], Value::Null);
    assert!(facts["spans"]["response"].is_array());
    assert!(facts["spans"]["wire_request"].is_array());
    // Without completion references the same record is a whole schema 1 record.
    let (whole, _) = encode_with_facts(&record, 1 << 20, false).unwrap();
    assert_eq!(whole, record.encode(1 << 20).unwrap());
}

#[test]
fn the_record_bound_and_record_validity_still_apply() {
    let record = record(contexts()[0].clone(), Some(sse()), None);
    let plain = record.encode(1 << 20).unwrap();
    assert!(encode_with_facts(&record, plain.len() - 1, false).is_none());
    assert!(encode_with_facts(&record, plain.len(), false).is_some());
    let mut invalid = record;
    invalid.request.request_id = " ".into();
    assert!(encode_with_facts(&invalid, 1 << 20, false).is_none());
}
