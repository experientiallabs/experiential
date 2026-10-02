//! Exercise translated tool calls through the real upstream relay and both
//! public Responses encodings, including JSON escapes split across frames.
use super::*;
use crate::encode_responses::{completed_responses_body, ResponsesEnvelope, ResponsesSseEncoder};
use futures_util::stream;
use serde_json::{json, Value};

async fn relayed_tool_result(
    name: &str,
    raw: &str,
    translate: bool,
) -> (Vec<Event>, Result<(), Failure>) {
    let mut chunks = vec![json!({"choices": [{"delta": {"tool_calls": [{
        "index": 0, "id": "call-one", "type": "function",
        "function": {"name": name, "arguments": ""}
    }]}}]})];
    for ch in raw.chars() {
        chunks.push(json!({"choices": [{"delta": {"tool_calls": [{
            "index": 0, "function": {"arguments": ch.to_string()}
        }]}}]}));
    }
    chunks.push(json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}));
    let mut frames: Vec<_> = chunks
        .into_iter()
        .map(|chunk| Ok::<_, reqwest::Error>(Bytes::from(format!("data: {chunk}\n\n"))))
        .collect();
    frames.push(Ok(Bytes::from("data: [DONE]\n\n")));
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut relay = UpstreamRelay::from_stream(
        stream::iter(frames).boxed(),
        Dialect::OpenAiCompatible,
        deadline,
    );
    if translate {
        relay.set_native_tool_translation(NativeToolTranslation::from([
            (
                "ns__patch".into(),
                ("patch".into(), Some("ns".into()), true),
            ),
            (
                "ns__lookup".into(),
                ("lookup".into(), Some("ns".into()), false),
            ),
        ]));
    }
    let mut events = Vec::new();
    loop {
        match relay
            .next_event(deadline, Duration::from_secs(5), Instant::now())
            .await
        {
            Ok(Some(event)) => {
                let terminal = event.is_terminal();
                events.push(event);
                if terminal {
                    return (events, Ok(()));
                }
            }
            Ok(None) => return (events, Ok(())),
            Err(failure) => return (events, Err(failure)),
        }
    }
}

async fn relayed_tool(name: &str, raw: &str, translate: bool) -> Vec<Event> {
    let (events, result) = relayed_tool_result(name, raw, translate).await;
    result.unwrap();
    assert!(matches!(events.last(), Some(Event::Completed)));
    events
}

fn assert_encoded(events: &[Event], name: &str, custom: bool, expected: &str) {
    let envelope = ResponsesEnvelope::default();
    let mut encoder = ResponsesSseEncoder::new("r", "m", 1, envelope.clone());
    let mut frames = encoder.start().unwrap();
    for event in events {
        frames.extend(encoder.feed(event).unwrap());
    }
    let payloads: Vec<Value> = frames
        .iter()
        .flat_map(|frame| frame.lines())
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    let item_type = if custom {
        "custom_tool_call"
    } else {
        "function_call"
    };
    let key = if custom { "input" } else { "arguments" };
    let added = payloads
        .iter()
        .find(|v| v["type"] == "response.output_item.added")
        .unwrap();
    assert_eq!(added["item"]["type"], item_type);
    assert_eq!(added["item"][key], "");
    let done = payloads
        .iter()
        .find(|v| v["type"] == "response.output_item.done")
        .unwrap();
    assert_eq!(done["item"]["type"], item_type);
    assert_eq!(done["item"]["name"], name);
    assert_eq!(done["item"][key], expected);
    let delta_type = if custom {
        "response.custom_tool_call_input.delta"
    } else {
        "response.function_call_arguments.delta"
    };
    let delta: String = payloads
        .iter()
        .filter(|v| v["type"] == delta_type)
        .map(|v| v["delta"].as_str().unwrap())
        .collect();
    assert_eq!(delta, expected);
    let body = completed_responses_body("r", "m", 1, envelope, events).unwrap();
    assert!(body.failure.is_none());
    assert_eq!(body.body["output"][0], done["item"]);
}

#[tokio::test]
async fn custom_tool_json_fragments_become_consistent_freeform_input() {
    let input = "*** Begin Patch\n+\"hello 🌍\"\\path\n*** End Patch";
    let raw = serde_json::to_string(&json!({"input": input})).unwrap();
    let events = relayed_tool("ns__patch", &raw, true).await;
    assert_encoded(&events, "patch", true, input);
    assert!(events
        .iter()
        .any(|event| matches!(event, Event::ToolCallStarted {
        namespace: Some(ns), custom: true, ..
    } if ns == "ns")));
}

#[tokio::test]
async fn invalid_translated_custom_input_fails_after_committed_start() {
    for raw in [r#"{"input":17}"#, r#"{"other":"x"}"#] {
        let (events, result) = relayed_tool_result("ns__patch", raw, true).await;
        let failure = result.unwrap_err();
        assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
        assert!(!failure.retryable_same_deployment && !failure.failover_eligible);
        assert!(events.iter().any(|event| matches!(
            event,
            Event::ToolCallStarted { name, namespace: Some(namespace), custom: true, .. }
                if name == "patch" && namespace == "ns"
        )));
        assert!(!events.iter().any(|event| matches!(
            event,
            Event::ToolArgumentsDelta { .. } | Event::ToolCallCompleted { .. } | Event::Completed
        )));
    }
}

#[tokio::test]
async fn empty_translated_custom_input_is_valid_on_both_encodings() {
    let events = relayed_tool("ns__patch", r#"{"input":""}"#, true).await;
    assert_encoded(&events, "patch", true, "");
}

#[tokio::test]
async fn ordinary_and_namespaced_function_tools_keep_streamed_arguments() {
    let raw = r#"{"query":"hello"}"#;
    let ordinary = relayed_tool("lookup", raw, true).await;
    let untranslated = relayed_tool("lookup", raw, false).await;
    assert_eq!(format!("{ordinary:?}"), format!("{untranslated:?}"));
    assert_encoded(&ordinary, "lookup", false, raw);
    let namespaced = relayed_tool("ns__lookup", raw, true).await;
    assert_encoded(&namespaced, "lookup", false, raw);
    assert_eq!(
        namespaced
            .iter()
            .filter(|e| matches!(e, Event::ToolArgumentsDelta { delta, .. } if !delta.is_empty()))
            .count(),
        raw.len()
    );
}
