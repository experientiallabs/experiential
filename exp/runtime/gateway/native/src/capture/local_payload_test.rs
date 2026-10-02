use super::*;
use serde_json::json;

fn record() -> Record {
    serde_json::from_value(json!({
        "schema_version":1,"request":{"request_id":"request",
            "scope":{"organization_id":"org","identity_id":"user","application_id":"app"},
            "protocol":"responses","model_id":"model",
            "context":{"schema_version":1,"request":{"messages":[
                {"role":"user","content":"prompt 雪"},
                {"role":"tool","content":" environment ","tool_call_id":"call"}
            ],"tools":[{"name":"lookup","parameters":{"type":"object"}}],
                "previous_response_id":"parent","metadata":{"conversation_id":"episode"}}}},
        "response":{"kind":"json","status":200,"body":{
            "id":"response","status":"completed","output":[]},"source_json":null},
        "provider_reasoning":"first\0second雪", "provider_reasoning_source_json":null,
        "provider_tool_calls_json":"[{\"raw_arguments\":\"{  }\"}]",
        "metrics":null,"gemini_thought_parts":[{"thoughtSignature":"opaque=="}],
        "gemini_thought_parts_source_json":null,
        "deployment_id":"deployment","captured_at":1.0
    }))
    .unwrap()
}

#[test]
fn borrowed_payload_preserves_schema_content_sidecars_and_exact_limit() {
    let record = record();
    let response = super::super::projection::CapturedResponse::completed_record(&record).unwrap();
    assert!(matches!(response, Cow::Borrowed(_)));
    let payload = encode(&record, &response, "experience-id", 8192).unwrap();
    let actual: Value = serde_json::from_str(&payload).unwrap();
    assert_eq!(
        actual,
        json!({
            "schema_version":1,"experience_id":"experience-id","response_id":"response",
            "episode_id":"episode","parent_response_id":"parent",
            "scope":{"user_id":"user","application_id":"app"},
            "protocol":"responses","captured_at":1.0,
            "request":{
                "exp_context":record.request.context,
                "exp_capture_output":{
                    "response":record.response,"provider_reasoning":"first�second雪",
                    "provider_reasoning_source_json":"\"first\\u0000second雪\"",
                    "provider_tool_calls_json":record.provider_tool_calls_json,
                    "metrics":record.metrics,
                    "gemini_thought_parts":record.gemini_thought_parts,
                    "gemini_thought_parts_source_json":null,
                },"previous_response_id":"parent"
            },"response":response,
            "provenance":{"source_id":"request","model_id":"model",
                "deployment_id":"deployment"},
        })
    );
    assert_eq!(
        encode(&record, &response, "experience-id", payload.len()),
        Some(payload.clone())
    );
    assert!(encode(&record, &response, "experience-id", payload.len() - 1).is_none());
    assert_eq!(
        record.provider_reasoning.as_deref(),
        Some("first\0second雪")
    );
}

#[test]
fn completed_responses_event_is_borrowed_from_retained_frames() {
    let mut record = record();
    record.response = Some(Response::Sse {
        status: 200,
        frames: vec![json!({"type":"response.completed", "response":{
            "id":"response","status":"completed","output":[]}})],
        truncated: false,
        client_disconnected: false,
        source_json: None,
    });
    let response = super::super::projection::CapturedResponse::completed_record(&record).unwrap();
    assert!(matches!(response, Cow::Borrowed(_)));
    assert_eq!(response["id"], "response");
}

#[test]
fn episode_uses_explicit_body_identity_then_optional_session_header() {
    let mut record = record();
    record.provider_reasoning = None;
    let context = Arc::make_mut(&mut record.request.context);
    context["session_id"] = json!("harness-session");
    let response = json!({"id":"response","status":"completed","output":[]});
    let payload = encode(&record, &response, "experience-id", 8192).unwrap();
    let actual: Value = serde_json::from_str(&payload).unwrap();
    assert_eq!(actual["episode_id"], "episode");
    assert!(actual["request"]["exp_capture_output"]["provider_reasoning"].is_null());
    Arc::make_mut(&mut record.request.context)["request"]["metadata"] = json!({});
    let payload = encode(&record, &response, "experience-id", 8192).unwrap();
    let actual: Value = serde_json::from_str(&payload).unwrap();
    assert_eq!(actual["episode_id"], "harness-session");
    Arc::make_mut(&mut record.request.context)["session_id"] = Value::Null;
    let payload = encode(&record, &response, "experience-id", 8192).unwrap();
    let actual: Value = serde_json::from_str(&payload).unwrap();
    assert!(actual["episode_id"].is_null());
}
