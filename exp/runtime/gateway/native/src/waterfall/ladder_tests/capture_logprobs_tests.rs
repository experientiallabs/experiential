//! Gateway-requested (capture-only) probabilities never change the caller's answer.
use super::*;
use crate::encode::{completed_chat_body_with_ignored, ChatSseEncoder};
use crate::encode_responses::{ResponsesEnvelope, ResponsesSseEncoder};

const PLAIN: &[&str] = &[
    r#"{"id":"x","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}"#,
    r#"{"id":"x","choices":[{"index":0,"delta":{"content":"lo"}}]}"#,
    r#"{"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
    r#"{"id":"x","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}"#,
];
const WITH_LOGPROBS: &[&str] = &[
    r#"{"id":"x","choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"},"logprobs":{"content":[{"token":"Hel","logprob":-0.25,"bytes":[72,101,108],"top_logprobs":[]}]}}]}"#,
    r#"{"id":"x","choices":[{"index":0,"delta":{"content":"lo"},"logprobs":{"content":[{"token":"lo","logprob":-0.5,"bytes":[108,111],"top_logprobs":[]}]}}]}"#,
    r#"{"id":"x","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}"#,
    r#"{"id":"x","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}"#,
];
const REFUSED_FIELD: &str = r#"{"error":{"message":"logprobs is not supported","type":"invalid_request_error","param":"logprobs"}}"#;

async fn run_capture(
    harness: &Harness,
    route: &[DeploymentWire],
    capture_logprobs: bool,
) -> (Won, AttemptGuard) {
    let mut guard = AttemptGuard::new(
        harness.bridge.clone(),
        Arc::new(AtomicUsize::new(0)),
        "capture-logprobs".to_string(),
        Instant::now(),
    );
    let context = WaterfallContext {
        bridge: &harness.bridge,
        http: &harness.http,
        request_id: "capture-logprobs",
        raw_key: "key",
        caller_scope: None,
        route,
        policy: RoutePolicy {
            maximum_total_attempts: 4,
            maximum_same_deployment_attempts: 1,
            refusal_failover: false,
            throttle_redial: None,
            physical_route_cap: None,
            backoff: None,
        },
        deadline: Instant::now() + Duration::from_secs(10),
        time_to_first_byte: Duration::from_secs(2),
        time_to_first_byte_slope_seconds_per_million_input_tokens: 0.0,
        time_to_first_token: Duration::from_secs(120),
        approximate_input_tokens: 1.0,
        chat_logprobs: true,
        capture_logprobs,
        output_less_retention: None,
        output_token_cap: None,
        tool_search: None,
        output_guardrails: None,
    };
    (acquire_attempt(&context, &mut guard).await, guard)
}

fn capture_wire(id: &str, url: &str) -> DeploymentWire {
    let mut wire = wire(id, url, 0);
    wire.capture_logprobs = true;
    wire
}

/// Every caller-visible event of the committed attempt, plus the relay.
async fn drain(won: Won) -> (Vec<Event>, Box<CommittedAttempt>) {
    let Won::Committed(mut committed) = won else {
        panic!("the rung must commit");
    };
    let mut events = committed.prefix.clone();
    while let Some(event) = committed
        .relay
        .next_event(
            Instant::now() + Duration::from_secs(2),
            Duration::from_secs(2),
            Instant::now(),
        )
        .await
        .expect("stream drains")
    {
        events.push(event);
    }
    (events, committed)
}

/// The three caller encodings: Chat SSE, the Chat body and Responses SSE.
fn encodings(events: &[Event]) -> (Vec<String>, String, Vec<String>) {
    let mut chat = ChatSseEncoder::new_with_ignored("r", "m", 1, true, Vec::new());
    let mut chat_frames = chat.start().expect("starts");
    for event in events {
        chat_frames.extend(chat.feed(event).expect("encodes"));
    }
    let body = completed_chat_body_with_ignored("r", "m", 1, events, &[], false)
        .expect("aggregates")
        .body
        .to_string();
    let mut responses = ResponsesSseEncoder::new("r", "m", 1, ResponsesEnvelope::default());
    let mut responses_frames = responses.start().expect("starts");
    for event in events {
        responses_frames.extend(responses.feed(event).expect("encodes"));
    }
    (chat_frames, body, responses_frames)
}

fn usage(events: &[Event]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Usage(usage) => Some(format!("{usage:?}")),
            _ => None,
        })
        .collect()
}

#[test]
fn injected_probabilities_leave_every_caller_encoding_byte_identical() {
    block_on(async {
        let harness = Harness::new();
        let plain = spawn_rung(vec![Answer::Stream(PLAIN)]).await;
        let (won, guard) = run_capture(&harness, &[wire("plain-a", &plain.url, 0)], false).await;
        let (baseline, _) = drain(finish(guard, won).await).await;

        let rich = spawn_rung(vec![Answer::Stream(WITH_LOGPROBS)]).await;
        let (won, guard) =
            run_capture(&harness, &[capture_wire("inject-a", &rich.url)], true).await;
        let (injected, committed) = drain(finish(guard, won).await).await;

        let sent: Value = serde_json::from_str(&rich.bodies.lock().unwrap()[0]).unwrap();
        assert_eq!(sent["logprobs"], json!(true));
        assert!(sent.get("top_logprobs").is_none());
        assert!(committed.relay.logprobs.injected);
        assert_eq!(encodings(&injected), encodings(&baseline));
        assert_eq!(usage(&injected), usage(&baseline));
        assert!(!injected
            .iter()
            .any(|event| matches!(event, Event::ChoiceLogprobsDelta(_))));
        let held = &committed.relay.logprobs.pending;
        let tokens: Vec<&str> = held
            .iter()
            .flat_map(|delta| delta.logprobs.as_ref().unwrap().content.as_ref().unwrap())
            .map(|record| record.token.as_str())
            .collect();
        assert_eq!(tokens, ["Hel", "lo"]);
    });
}

#[test]
fn injection_requires_the_request_capture_and_the_rung_eligibility() {
    for (capture, eligible) in [(false, true), (true, false)] {
        block_on(async {
            let harness = Harness::new();
            let rung = spawn_rung(vec![Answer::Stream(PLAIN)]).await;
            let mut wire = capture_wire(&format!("gate-{capture}-{eligible}"), &rung.url);
            wire.capture_logprobs = eligible;
            let (won, guard) = run_capture(&harness, &[wire], capture).await;
            let (_, committed) = drain(finish(guard, won).await).await;
            let sent: Value = serde_json::from_str(&rung.bodies.lock().unwrap()[0]).unwrap();
            assert!(sent.get("logprobs").is_none());
            assert!(!committed.relay.logprobs.injected);
        });
    }
}

#[test]
fn customer_managed_rungs_and_caller_controls_are_never_injected() {
    block_on(async {
        let harness = Harness::new();
        let rung = spawn_rung(vec![Answer::Stream(PLAIN)]).await;
        let mut wire = capture_wire("byok-a", &rung.url);
        wire.billing_customer_managed = true;
        let (won, guard) = run_capture(&harness, &[wire], true).await;
        let (_, committed) = drain(finish(guard, won).await).await;
        assert!(!committed.relay.logprobs.injected);
        let sent: Value = serde_json::from_str(&rung.bodies.lock().unwrap()[0]).unwrap();
        assert!(sent.get("logprobs").is_none());
    });
    assert!(crate::capture::logprobs::with_logprobs(&json!({"top_logprobs": 2})).is_none());
    assert!(crate::capture::logprobs::with_logprobs(&json!({"logprobs": false})).is_none());
}

#[test]
fn a_refusal_redials_plain_inside_one_attempt_and_is_remembered() {
    block_on(async {
        let harness = Harness::new();
        let rung = spawn_rung(vec![
            Answer::Rejected(REFUSED_FIELD),
            Answer::Stream(PLAIN),
            Answer::Stream(PLAIN),
        ])
        .await;
        let route = [capture_wire("refuses-a", &rung.url)];
        let (won, guard) = run_capture(&harness, &route, true).await;
        let (events, committed) = drain(finish(guard, won).await).await;
        assert!(!committed.relay.logprobs.injected);
        assert!(matches!(events.first(), Some(Event::TextDelta(text)) if text == "Hel"));
        {
            let bodies = rung.bodies.lock().unwrap();
            let first: Value = serde_json::from_str(&bodies[0]).unwrap();
            let second: Value = serde_json::from_str(&bodies[1]).unwrap();
            assert_eq!(first["logprobs"], json!(true));
            assert!(second.get("logprobs").is_none());
            // The injected body never shares the plain payload's key.
            let keys = rung.keys.lock().unwrap();
            assert_eq!(keys[0].as_deref(), Some("op-refuses-a.logprobs"));
            assert_eq!(keys[1].as_deref(), Some("op-refuses-a"));
        }
        // One physical attempt reserved for the caller, not two.
        let story = harness.story().await;
        assert_eq!(story["starts"].as_array().unwrap().len(), 1);
        assert!(crate::capture::logprobs::refused(
            &crate::capture::logprobs::rung_key(&route[0])
        ));
        // Another catalog's rung under the same local id is still asked.
        let mut elsewhere = route[0].clone();
        elsewhere.url = "http://elsewhere.invalid/v1/chat/completions".into();
        assert!(!crate::capture::logprobs::refused(
            &crate::capture::logprobs::rung_key(&elsewhere)
        ));
        // The worker never injects on that rung again.
        let (won, guard) = run_capture(&harness, &route, true).await;
        let (_, committed) = drain(finish(guard, won).await).await;
        assert!(!committed.relay.logprobs.injected);
        let third: Value = serde_json::from_str(&rung.bodies.lock().unwrap()[2]).unwrap();
        assert!(third.get("logprobs").is_none());
    });
}

#[test]
fn a_request_refused_either_way_fails_as_without_injection_and_is_not_remembered() {
    block_on(async {
        let harness = Harness::new();
        let plain = spawn_rung(vec![Answer::Rejected(REFUSED_FIELD)]).await;
        let (baseline, guard) =
            run_capture(&harness, &[wire("both-plain", &plain.url, 0)], false).await;
        let baseline = finish(guard, baseline).await;
        let rung = spawn_rung(vec![
            Answer::Rejected(REFUSED_FIELD),
            Answer::Rejected(REFUSED_FIELD),
        ])
        .await;
        let (won, guard) = run_capture(&harness, &[capture_wire("both-a", &rung.url)], true).await;
        let won = finish(guard, won).await;
        let (Won::Failed(expected), Won::Failed(actual)) = (baseline, won) else {
            panic!("both runs must fail");
        };
        assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
        assert_eq!(rung.bodies.lock().unwrap().len(), 2);
        assert!(!crate::capture::logprobs::refused(
            &crate::capture::logprobs::rung_key(&capture_wire("both-a", &rung.url))
        ));
    });
}

const REFUSED_IN_STREAM: &str = r#"{"error":{"message":"logprobs is not supported for this model","type":"invalid_request_error","code":"invalid_request_error","param":"logprobs"}}"#;

#[test]
fn a_refusal_under_http_200_redials_plain_before_any_output() {
    block_on(async {
        let harness = Harness::new();
        let plain = spawn_rung(vec![Answer::Stream(PLAIN)]).await;
        let (won, guard) =
            run_capture(&harness, &[wire("ok200-plain", &plain.url, 0)], false).await;
        let (baseline, _) = drain(finish(guard, won).await).await;

        let rung = spawn_rung(vec![
            Answer::Stream(&[REFUSED_IN_STREAM]),
            Answer::Stream(PLAIN),
        ])
        .await;
        let route = [capture_wire("ok200-a", &rung.url)];
        let (won, guard) = run_capture(&harness, &route, true).await;
        let (events, committed) = drain(finish(guard, won).await).await;
        assert_eq!(encodings(&events), encodings(&baseline));
        assert!(!committed.relay.logprobs.injected);
        let keys = rung.keys.lock().unwrap().clone();
        assert_eq!(keys[0].as_deref(), Some("op-ok200-a.logprobs"));
        assert_eq!(keys[1].as_deref(), Some("op-ok200-a"));
        // One reserved attempt per request: the baseline's and this one's.
        assert_eq!(harness.story().await["starts"].as_array().unwrap().len(), 2);
        assert!(crate::capture::logprobs::refused(
            &crate::capture::logprobs::rung_key(&route[0])
        ));
    });
}

#[test]
fn a_refusal_after_reported_usage_is_never_redialed() {
    block_on(async {
        let harness = Harness::new();
        let rung = spawn_rung(vec![Answer::Stream(&[
            r#"{"id":"x","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}"#,
            REFUSED_IN_STREAM,
        ]),
            Answer::Stream(PLAIN),
        ])
        .await;
        let (won, guard) =
            run_capture(&harness, &[capture_wire("metered-a", &rung.url)], true).await;
        let won = finish(guard, won).await;
        assert!(!matches!(won, Won::Committed(_)));
        // The metered dial is the attempt's answer: no second provider call.
        assert_eq!(rung.bodies.lock().unwrap().len(), 1);
        assert!(!crate::capture::logprobs::refused(
            &crate::capture::logprobs::rung_key(&capture_wire("metered-a", &rung.url))
        ));
    });
}

#[test]
fn a_refusal_after_generated_reasoning_is_never_redialed() {
    block_on(async {
        let harness = Harness::new();
        let rung = spawn_rung(vec![
            Answer::Stream(&[
                r#"{"id":"x","choices":[{"index":0,"delta":{"reasoning":"thinking"}}]}"#,
                REFUSED_IN_STREAM,
            ]),
            Answer::Stream(PLAIN),
        ])
        .await;
        let (won, guard) =
            run_capture(&harness, &[capture_wire("reasoned-a", &rung.url)], true).await;
        let won = finish(guard, won).await;
        assert!(!matches!(won, Won::Committed(_)));
        assert_eq!(rung.bodies.lock().unwrap().len(), 1);
    });
}
