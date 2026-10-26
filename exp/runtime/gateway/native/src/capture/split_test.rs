use super::*;
use crate::capture::record::{Protocol, Request, Scope};
use crate::capture::spill::tests::store;
use serde_json::json;
use std::sync::Arc;

pub(crate) fn record(context: Value, response: Option<Response>) -> Record {
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
            model_id: Some("model".into()),
            context: Arc::new(context),
        },
        response,
        transport: None,
        provider_reasoning: None,
        provider_reasoning_source_json: None,
        provider_tool_calls_json: None,
        deployment_id: Some("mp-row".into()),
        metrics: None,
        gemini_thought_parts: Vec::new(),
        gemini_thought_parts_source_json: None,
        captured_at: 1.0,
        provider_logprobs: None,
        transport_body: None,
        messages_part: None,
    }
}

/// Splice every reference back in, the way a destination does.
pub(crate) fn reassemble(split: &SplitRecord) -> Value {
    let mut skeleton: Value = serde_json::from_str(&split.skeleton).unwrap();
    resolve(&mut skeleton, split);
    skeleton
}

fn resolve(value: &mut Value, split: &SplitRecord) {
    let reference = value
        .as_object()
        .filter(|object| object.len() == 1)
        .and_then(|object| object.get(&split.marker))
        .and_then(Value::as_u64);
    if let Some(index) = reference {
        let part = &split.parts[index as usize];
        let text = std::fs::read_to_string(part.file.path()).unwrap();
        *value = match part.kind {
            PartKind::Json => serde_json::from_str(&text).unwrap(),
            PartKind::Frames => Value::Array(
                text.lines()
                    .map(|line| serde_json::from_str(line).unwrap())
                    .collect(),
            ),
        };
        resolve(value, split);
        return;
    }
    match value {
        Value::Array(items) => items.iter_mut().for_each(|item| resolve(item, split)),
        Value::Object(items) => items.iter_mut().for_each(|(_, item)| resolve(item, split)),
        _ => {}
    }
}

#[test]
fn messages_and_long_strings_leave_the_skeleton_and_reassemble_exactly() {
    let store = store("split-messages", 64 * 1024 * 1024);
    let image = "A".repeat(1_000_000);
    let context = json!({
        "schema_version": 1,
        "request": {
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{image}")}},
            ]}],
            "tools": [{"name": "search", "description": "x".repeat(10_000)}],
        },
    });
    let original = record(context, None);
    let split = split(&original, &store, 64 * 1024).unwrap();
    assert!(split.skeleton.len() < 8 * 1024, "{}", split.skeleton.len());
    assert!(split.marker.starts_with("$experiential_capture_part_"));
    // The messages part is the exact JSON of the list; the tool text is its own part.
    assert_eq!(split.parts[0].kind, PartKind::Json);
    let messages: Value =
        serde_json::from_slice(&std::fs::read(split.parts[0].file.path()).unwrap()).unwrap();
    assert_eq!(messages, original.request.context["request"]["messages"]);
    let expected = serde_json::to_value(&original).unwrap();
    assert_eq!(reassemble(&split), expected);
}

#[test]
fn a_spilled_stream_becomes_one_frame_per_line_without_trimming() {
    let store = store("split-frames", 512 * 1024 * 1024);
    let mut stream = store.open_stream().unwrap();
    let mut expected = Vec::new();
    for index in 0..2000 {
        let frame = json!({"id": "c", "choices": [{"delta": {"content": format!("{index}-{}", "y".repeat(5000))}}]});
        stream.push(bytes::Bytes::from(format!("data: {frame}\n\n")));
        expected.push(frame);
    }
    stream.push(bytes::Bytes::from_static(b"data: [DONE]\n\n"));
    expected.push(Value::String("[DONE]".into()));
    let file = stream.finish();
    let original = record(
        json!({"schema_version": 1, "request": {"messages": []}}),
        Some(Response::Spilled(SpilledResponse {
            status: 200,
            sse: true,
            truncated: false,
            client_disconnected: false,
            file,
        })),
    );
    let split = split(&original, &store, 64 * 1024).unwrap();
    let rebuilt = reassemble(&split);
    assert_eq!(rebuilt["response"]["kind"], "sse");
    assert_eq!(rebuilt["response"]["truncated"], false);
    assert_eq!(rebuilt["response"]["frames"], Value::Array(expected));
}

#[test]
fn a_spilled_json_body_keeps_its_structure_with_long_strings_as_parts() {
    let store = store("split-json", 512 * 1024 * 1024);
    let body = json!({
        "id": "c",
        "data": [{"b64_json": "Z".repeat(3_000_000)}],
        "usage": {"total_tokens": 5},
    });
    let mut stream = store.open_stream().unwrap();
    stream.push(bytes::Bytes::from(body.to_string()));
    let original = record(
        json!({"schema_version": 1, "request": {"messages": []}}),
        Some(Response::Spilled(SpilledResponse {
            status: 200,
            sse: false,
            truncated: false,
            client_disconnected: false,
            file: stream.finish(),
        })),
    );
    let split = split(&original, &store, 64 * 1024).unwrap();
    // The destination annotates usage on the skeleton: it is still there.
    let skeleton: Value = serde_json::from_str(&split.skeleton).unwrap();
    assert_eq!(skeleton["response"]["body"]["usage"]["total_tokens"], 5);
    assert_eq!(reassemble(&split)["response"]["body"], body);
}

#[test]
fn a_truncated_spilled_json_body_keeps_no_response() {
    let store = store("split-truncated", 64 * 1024 * 1024);
    let mut stream = store.open_stream().unwrap();
    stream.push(bytes::Bytes::from_static(b"{\"id\":"));
    let mut original = record(
        json!({"schema_version": 1, "request": {"messages": []}}),
        Some(Response::Spilled(SpilledResponse {
            status: 200,
            sse: false,
            truncated: true,
            client_disconnected: false,
            file: stream.finish(),
        })),
    );
    original.provider_reasoning = Some("thought".into());
    let rebuilt = reassemble(&split(&original, &store, 64 * 1024).unwrap());
    assert_eq!(rebuilt["response"], Value::Null);
    assert_eq!(rebuilt["provider_reasoning"], Value::Null);
}

#[test]
fn a_large_caller_body_becomes_the_wire_request_body() {
    let store = store("split-wire", 64 * 1024 * 1024);
    let mut original = record(
        json!({"schema_version": 1, "request": {"messages": []}}),
        None,
    );
    original.transport =
        Some(json!({"wire_request": {"method": "POST", "body_kind": "json", "body": null}}));
    let body =
        json!({"model": "m", "messages": [{"role": "user", "content": "q".repeat(500_000)}]});
    original.transport_body = Some(body.to_string().into_bytes());
    let rebuilt = reassemble(&split(&original, &store, 64 * 1024).unwrap());
    assert_eq!(rebuilt["transport"]["wire_request"]["body"], body);
    assert_eq!(rebuilt["transport"]["wire_request"]["body_kind"], "json");
    original.transport_body = Some(b"not json".to_vec());
    let rebuilt = reassemble(&split(&original, &store, 64 * 1024).unwrap());
    assert_eq!(rebuilt["transport"]["wire_request"]["body"], "not json");
    assert_eq!(rebuilt["transport"]["wire_request"]["body_kind"], "text");
}

#[test]
fn caller_content_cannot_forge_a_part_reference() {
    let store = store("split-forge", 64 * 1024 * 1024);
    let first = split(
        &record(
            json!({"schema_version": 1, "request": {"messages": []}}),
            None,
        ),
        &store,
        64 * 1024,
    )
    .unwrap();
    let second = split(
        &record(
            json!({"schema_version": 1, "request": {"messages": []}}),
            None,
        ),
        &store,
        64 * 1024,
    )
    .unwrap();
    assert_ne!(first.marker, second.marker);
}

#[test]
fn an_exhausted_disk_budget_refuses_the_record_rather_than_trimming_it() {
    let store = store("split-budget", 1024 * 1024);
    let original = record(
        json!({"schema_version": 1, "request": {"messages": [{"role": "user", "content": "w".repeat(2_000_000)}]}}),
        None,
    );
    assert!(split(&original, &store, 64 * 1024).is_none());
}
