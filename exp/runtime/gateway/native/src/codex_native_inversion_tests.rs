//! Actual relay and Responses encoder coverage for translated native tools.

use super::*;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures_util::{stream, StreamExt};
use serde_json::json;

use crate::dialects::Dialect;
use crate::encode_responses::{completed_responses_body, ResponsesEnvelope, ResponsesSseEncoder};
use crate::relay::UpstreamRelay;

fn translation() -> NativeToolTranslation {
    HashMap::from([
        ("apply_patch".into(), ("apply_patch".into(), None, true)),
        (
            "agents__close".into(),
            ("close".into(), Some("agents".into()), false),
        ),
    ])
}

fn chunk(index: u32, name: Option<&str>, arguments: &str) -> Value {
    let mut tool = json!({"index":index,"function":{"arguments":arguments}});
    if let Some(name) = name {
        tool["id"] = json!(format!("call-{index}"));
        tool["function"]["name"] = json!(name);
    }
    json!({"choices":[{"index":0,"delta":{"tool_calls":[tool]}}]})
}

fn relay(frames: Vec<Value>, mapping: NativeToolTranslation) -> UpstreamRelay {
    // Split every transport byte, including UTF-8 sequences and JSON escapes.
    let wire = frames
        .into_iter()
        .map(|frame| format!("data: {frame}\n\n"))
        .collect::<String>();
    let chunks: Vec<_> = wire
        .into_bytes()
        .into_iter()
        .map(|byte| Ok::<_, reqwest::Error>(Bytes::from(vec![byte])))
        .collect();
    let mut relay = UpstreamRelay::from_stream(
        stream::iter(chunks).boxed(),
        Dialect::OpenAiCompatible,
        Instant::now() + Duration::from_secs(10),
    );
    relay.set_native_tool_translation(mapping);
    relay
}

async fn next(relay: &mut UpstreamRelay) -> Result<Option<Event>, Failure> {
    relay
        .next_event(
            Instant::now() + Duration::from_secs(10),
            Duration::from_secs(5),
            Instant::now(),
        )
        .await
}

fn decoded(frames: &[String]) -> Vec<Value> {
    frames
        .iter()
        .flat_map(|frame| frame.lines())
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

#[tokio::test]
async fn custom_input_is_deferred_but_start_and_other_functions_remain_incremental() {
    for input in ["", "patch\n\"quoted\"\\path café 😀"] {
        let raw = serde_json::to_string(&json!({"input":input})).unwrap();
        let escaped = raw.replace("😀", "\\ud83d\\ude00");
        let mut frames = vec![
            chunk(0, Some("apply_patch"), ""),
            chunk(1, Some("agents__close"), "{\"id\":"),
            chunk(2, Some("plain"), "{\"x\":"),
            chunk(3, Some("apply_patch"), "{\"input\":\"second\"}"),
        ];
        frames.extend(escaped.chars().map(|c| chunk(0, None, &c.to_string())));
        frames.extend([
            chunk(1, None, "\"a\"}"),
            chunk(2, None, "1}"),
            json!({"choices":[{"delta":{},"finish_reason":"tool_calls"}]}),
        ]);
        let mut relay = relay(frames, translation());
        let mut encoder = ResponsesSseEncoder::new("r", "alias", 1, ResponsesEnvelope::default());
        let mut public = encoder.start().unwrap();
        let mut events = Vec::new();
        while let Some(event) = next(&mut relay).await.unwrap() {
            public.extend(encoder.feed(&event).unwrap());
            events.push(event);
        }
        assert!(relay.first_token_at().is_some());
        assert!(
            matches!(&events[0], Event::ToolCallStarted {custom:true, name, ..} if name=="apply_patch")
        );
        assert!(
            matches!(&events[1], Event::ToolCallStarted {custom:false, name, namespace, ..} if name=="close" && namespace.as_deref()==Some("agents"))
        );
        assert!(
            matches!(&events[2], Event::ToolArgumentsDelta {index:1, delta} if delta=="{\"id\":")
        );
        let custom_deltas: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::ToolArgumentsDelta { index: 0, delta } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(custom_deltas, [input]);
        let position = events
            .iter()
            .position(|event| matches!(event, Event::ToolArgumentsDelta { index: 0, .. }))
            .unwrap();
        assert!(
            matches!(&events[position+1], Event::ToolCallCompleted {index:0,call} if call.custom && call.raw_arguments==input)
        );
        let buffered =
            completed_responses_body("r", "alias", 1, ResponsesEnvelope::default(), &events)
                .unwrap();
        let public = decoded(&public);
        let completed = public
            .iter()
            .find(|event| event["type"] == "response.completed")
            .unwrap();
        assert_eq!(completed["response"], buffered.body);
        let output = buffered.body["output"].as_array().unwrap();
        assert_eq!(output.len(), 4);
        assert_eq!(output[0]["type"], "custom_tool_call");
        assert_eq!(output[0]["input"], input);
        assert_eq!(output[0]["call_id"], "call-0");
        assert_eq!(output[1]["name"], "close");
        assert_eq!(output[1]["namespace"], "agents");
        assert_eq!(output[2]["name"], "plain");
        assert_eq!(output[2]["arguments"], "{\"x\":1}");
        assert_eq!(output[3]["input"], "second");
        let added = public
            .iter()
            .find(|event| event["type"] == "response.output_item.added")
            .unwrap();
        assert_eq!(added["item"]["type"], "custom_tool_call");
        assert_eq!(added["item"]["id"], output[0]["id"]);
        let delta = public
            .iter()
            .find(|event| event["type"] == "response.custom_tool_call_input.delta")
            .unwrap();
        assert_eq!(delta["delta"], input);
        assert_eq!(delta["item_id"], output[0]["id"]);
    }
}

#[tokio::test]
async fn invalid_custom_wrappers_fail_after_start_without_flushing_raw_bytes() {
    for raw in [
        "not json",
        "{}",
        "{\"input\":1}",
        "{\"input\":\"x\",\"extra\":true}",
    ] {
        let mut relay = relay(
            vec![
                chunk(0, Some("apply_patch"), raw),
                json!({"choices":[{"delta":{},"finish_reason":"tool_calls"}]}),
            ],
            translation(),
        );
        assert!(matches!(
            next(&mut relay).await.unwrap(),
            Some(Event::ToolCallStarted { custom: true, .. })
        ));
        assert!(relay.first_token_at().is_some());
        let failure = next(&mut relay)
            .await
            .expect_err("wrapper must not become custom input");
        assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
    }
}

#[tokio::test]
async fn provider_truncation_keeps_custom_item_incomplete_and_never_flushes_partial_wrapper() {
    let mut relay = relay(
        vec![
            chunk(0, Some("apply_patch"), "{\"input\":\"partial"),
            json!({"choices":[{"delta":{},"finish_reason":"length"}]}),
        ],
        translation(),
    );
    let mut events = Vec::new();
    while let Some(event) = next(&mut relay).await.unwrap() {
        events.push(event);
    }
    assert!(matches!(
        &events[0],
        Event::ToolCallStarted { custom: true, .. }
    ));
    assert!(events.iter().all(|event| !matches!(
        event,
        Event::ToolArgumentsDelta { .. } | Event::ToolCallCompleted { .. }
    )));
    let buffered =
        completed_responses_body("r", "alias", 1, ResponsesEnvelope::default(), &events).unwrap();
    assert_eq!(buffered.body["status"], "incomplete");
    assert_eq!(buffered.body["output"][0]["status"], "incomplete");
    assert_eq!(buffered.body["output"][0]["input"], "");
}

#[tokio::test]
async fn a_new_attempt_without_mapping_never_inherits_custom_state() {
    let frames = vec![
        chunk(0, Some("apply_patch"), "{\"input\":\"x\"}"),
        json!({"choices":[{"delta":{},"finish_reason":"tool_calls"}]}),
    ];
    let mut first = relay(frames.clone(), translation());
    assert!(matches!(
        next(&mut first).await.unwrap(),
        Some(Event::ToolCallStarted { custom: true, .. })
    ));
    drop(first); // Cancellation discards only this attempt's deferred indices.
    let mut second = relay(frames, HashMap::new());
    assert!(matches!(
        next(&mut second).await.unwrap(),
        Some(Event::ToolCallStarted { custom: false, .. })
    ));
    assert!(
        matches!(next(&mut second).await.unwrap(),Some(Event::ToolArgumentsDelta {delta,..}) if delta=="{\"input\":\"x\"}")
    );
}

#[tokio::test]
async fn abrupt_eof_does_not_fabricate_a_completed_custom_input() {
    let mut relay = relay(
        vec![chunk(0, Some("apply_patch"), "{\"input\":\"partial")],
        translation(),
    );
    assert!(matches!(
        next(&mut relay).await.unwrap(),
        Some(Event::ToolCallStarted { custom: true, .. })
    ));
    // Compatible-wire EOF recovery already classifies a cut call as incomplete.
    assert!(matches!(
        next(&mut relay).await.unwrap(),
        Some(Event::Incomplete)
    ));
    assert!(next(&mut relay).await.unwrap().is_none());
}

#[tokio::test]
async fn suppressed_custom_deltas_still_obey_the_normalizer_byte_bound() {
    use crate::dialects::{MAXIMUM_RETAINED_OUTPUT_BYTES, OUTPUT_OVERFLOW_MESSAGE};
    let block = "x".repeat(1024 * 1024);
    let frames = std::iter::once(chunk(0, Some("apply_patch"), "{\"input\":\""))
        .chain((0..=MAXIMUM_RETAINED_OUTPUT_BYTES / block.len()).map(|_| chunk(0, None, &block)));
    let wire =
        frames.map(|frame| Ok::<_, reqwest::Error>(Bytes::from(format!("data: {frame}\n\n"))));
    let chunks: Vec<_> = wire.collect();
    let mut relay = UpstreamRelay::from_stream(
        stream::iter(chunks).boxed(),
        Dialect::OpenAiCompatible,
        Instant::now() + Duration::from_secs(30),
    );
    relay.set_native_tool_translation(translation());
    assert!(matches!(
        next(&mut relay).await.unwrap(),
        Some(Event::ToolCallStarted { custom: true, .. })
    ));
    let failure = next(&mut relay)
        .await
        .expect_err("suppression cannot bypass retained output bound");
    assert_eq!(failure.safe_message, OUTPUT_OVERFLOW_MESSAGE);
}

#[test]
fn native_custom_events_remain_incremental_with_an_empty_mapping() {
    use crate::events::{CompletedToolCall, ProviderOutputItemKind, ProviderOutputItemStatus};
    let mut inversion = NativeToolInverter::default();
    let mut encoder = ResponsesSseEncoder::new("r", "alias", 1, ResponsesEnvelope::default());
    encoder.start().unwrap();
    let mut ready = VecDeque::new();
    let events = [
        Event::ProviderOutputItemStarted {
            output_index: 0,
            item_id: Some("ctc-native".into()),
            kind: ProviderOutputItemKind::CustomToolCall,
            status: Some(ProviderOutputItemStatus::InProgress),
            phase: None,
        },
        Event::ToolCallStarted {
            index: 0,
            call_id: "call-native".into(),
            name: "apply_patch".into(),
            namespace: None,
            caller: None,
            custom: true,
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: "not ".into(),
        },
        Event::ToolArgumentsDelta {
            index: 0,
            delta: "JSON 😀".into(),
        },
        Event::ProviderOutputItemCompleted {
            output_index: 0,
            item_id: Some("ctc-native".into()),
            kind: ProviderOutputItemKind::CustomToolCall,
            status: Some(ProviderOutputItemStatus::Completed),
            phase: None,
        },
        Event::ToolCallCompleted {
            index: 0,
            call: CompletedToolCall {
                call_id: "call-native".into(),
                name: "apply_patch".into(),
                namespace: None,
                caller: None,
                provider_item_id: Some("ctc-native".into()),
                provider_status: Some(ProviderOutputItemStatus::Completed),
                raw_arguments: "not JSON 😀".into(),
                custom: true,
            },
        },
        Event::Completed,
    ];
    let mut encoded = Vec::new();
    for event in events {
        ready.extend(inversion.filter(event).unwrap());
        assert_eq!(ready.len(), 1);
        encoded.extend(encoder.feed(&ready.pop_front().unwrap()).unwrap());
    }
    let encoded = decoded(&encoded);
    let deltas: Vec<_> = encoded
        .iter()
        .filter(|event| event["type"] == "response.custom_tool_call_input.delta")
        .collect();
    assert_eq!(deltas.len(), 2);
    assert_eq!(deltas[0]["delta"], "not ");
    assert_eq!(deltas[1]["delta"], "JSON 😀");
    assert_eq!(deltas[0]["item_id"], "ctc-native");
}
