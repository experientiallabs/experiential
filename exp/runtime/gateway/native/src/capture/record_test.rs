use super::*;
use serde_json::json;
use std::cell::Cell;

fn record<R>(response: R) -> Record<R> {
    Record {
        checkpointed: false,
        schema_version: 1,
        request: Request {
            request_id: "request".into(),
            scope: Scope {
                organization_id: "org".into(),
                identity_id: "identity".into(),
                application_id: "alias".into(),
            },
            protocol: Protocol::ChatCompletions,
            model_id: None,
            context: Arc::new(json!({"schema_version":1,"request":{"messages":[]}})),
        },
        response: Some(response),
        transport: None,
        provider_reasoning: None,
        provider_reasoning_source_json: None,
        provider_tool_calls_json: None,
        deployment_id: None,
        metrics: None,
        gemini_thought_parts: Vec::new(),
        gemini_thought_parts_source_json: None,
        captured_at: 1.0,
    }
}

#[test]
fn structured_response_is_encoded_only_at_the_destination() {
    let response = Response::Json {
        status: 200,
        body: json!({"text":"雪"}),
        source_json: None,
    };
    assert_eq!(
        response.json_bytes(),
        serde_json::to_string(&response).unwrap().len()
    );
    let record = record(response);
    let first = record.encode(4096).unwrap();
    let second = record.encode(4096).unwrap();
    assert_eq!(first, second);
    let decoded: Record = serde_json::from_str(&first).unwrap();
    let Response::Json { body, .. } = decoded.response.unwrap() else {
        panic!()
    };
    assert_eq!(body["text"], "雪");
    assert!(record.encode(first.len() - 1).is_none());
    assert_eq!(record.encode(first.len()).unwrap(), first);
}

#[test]
fn cloned_request_shares_content_and_charges_spare_capacity() {
    let mut input = record(Response::Json {
        status: 200,
        body: json!({}),
        source_json: None,
    });
    let copy = input.request.clone();
    assert!(Arc::ptr_eq(&copy.context, &input.request.context));
    let initial = input.heap_bytes();
    let mut text = String::with_capacity(32768);
    text.push('x');
    input.provider_reasoning = Some(text);
    assert!(input.heap_bytes() >= initial + 32768);
}

#[test]
fn hosted_completion_references_checkpoint_without_encoding_prompt_again() {
    let calls = Cell::new(0);
    let mut item = record(SerializeOnce(&calls));
    item.request.context = Arc::new(json!({"schema_version":1,"request":{
        "messages":[{"role":"user","content":"x".repeat(512 * 1024)}],
        "tools":[{"name":"search","parameters":{"type":"object"}}]
    }}));
    let full = item.encode_update(1024 * 1024).unwrap();
    assert!(full.len() > 512 * 1024);
    item.checkpointed = true;
    let encoded = item.encode_update(4096).unwrap();
    let value: Value = serde_json::from_str(&encoded).unwrap();
    assert_eq!(value["schema_version"], 2);
    assert_eq!(value["request"]["request_id"], "request");
    assert!(value["request"].get("context").is_none());
    assert_eq!(calls.get(), 2);
    // Complete-record destinations retain the full input contract.
    assert!(item.encode(1024 * 1024).unwrap().len() > 512 * 1024);
}

#[test]
fn all_protocol_and_response_sizes_match_final_json() {
    let mut item = record(Response::Json {
        status: 200,
        body: json!({"control":"\0\n\r\t\u{8}\u{c}", "unicode":"雪😀"}),
        source_json: Some("\\u0000\"".into()),
    });
    for protocol in [
        Protocol::ChatCompletions,
        Protocol::Responses,
        Protocol::Messages,
    ] {
        item.request.protocol = protocol;
        for model in [None, Some("snow-雪-\"quoted\"".to_owned())] {
            item.request.model_id = model;
            assert_eq!(
                item.request.json_bytes(),
                serde_json::to_vec(&item.request).unwrap().len()
            );
        }
    }
    let response = item.response.unwrap();
    assert_eq!(
        response.json_bytes(),
        serde_json::to_vec(&response).unwrap().len()
    );
    for truncated in [true, false] {
        for client_disconnected in [true, false] {
            for frames in [vec![], vec![json!({"x":"\0雪"}), json!("[DONE]")]] {
                let response = Response::Sse {
                    status: 200,
                    frames,
                    truncated,
                    client_disconnected,
                    source_json: None,
                };
                assert_eq!(
                    response.json_bytes(),
                    serde_json::to_vec(&response).unwrap().len()
                );
            }
        }
    }
}

struct SerializeOnce<'a>(&'a Cell<usize>);

impl Serialize for SerializeOnce<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.set(self.0.get() + 1);
        serializer.serialize_none()
    }
}

#[test]
fn exceptional_reasoning_borrows_the_record_and_serializes_the_response_once() {
    let calls = Cell::new(0);
    // The response deliberately does not implement Clone.
    let mut record = record(SerializeOnce(&calls));
    record.provider_reasoning = Some("first\0second雪".into());
    let encoded = record.encode(4096).unwrap();
    assert_eq!(calls.get(), 1);
    assert_eq!(
        record.provider_reasoning.as_deref(),
        Some("first\0second雪")
    );
    let decoded: Record = serde_json::from_str(&encoded).unwrap();
    assert_eq!(
        decoded.provider_reasoning.as_deref(),
        Some("first\u{fffd}second雪")
    );
    let source: String =
        serde_json::from_str(decoded.provider_reasoning_source_json.as_deref().unwrap()).unwrap();
    assert_eq!(source, "first\0second雪");
}
