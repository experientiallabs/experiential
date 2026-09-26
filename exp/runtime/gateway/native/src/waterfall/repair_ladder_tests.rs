//! Ladder tests for the encrypted-reasoning replay repair, on the same
//! scripted control plane and local rungs as `ladder_tests`: a refusal before
//! the stream (OpenAI's 400) or inside it (OpenRouter's 200 then
//! `response.failed`) strips the refused payloads and re-dials the same rung
//! through a new reservation; the per-request and per-caller memories spare the
//! refused dial afterwards; unrelated failures keep their verdict.

use std::time::Duration;

use serde_json::{json, Value};

use super::ladder_tests::{
    block_on, finish, responses_wire, spawn_rung, Answer, Harness, INVALID_ENCRYPTED_CONTENT_BODY,
    RESPONSES_FAILED_ENCRYPTED_FRAME, RESPONSES_FAILED_OTHER_FRAME, RESPONSES_TEXT_FRAME, SCHEDULE,
};
use super::*;

#[test]
fn explicit_one_blocks_every_repair_and_operator_throttle_redial() {
    block_on(async {
        for per_route in [false, true] {
            for in_stream in [false, true] {
                let harness = Harness::new();
                let rung = spawn_rung(vec![if in_stream {
                    Answer::ResponsesFailed(RESPONSES_FAILED_ENCRYPTED_FRAME)
                } else {
                    Answer::Rejected(INVALID_ENCRYPTED_CONTENT_BODY)
                }])
                .await;
                let token = format!("rsn_cap_{per_route}_{in_stream}_hA==");
                let route = [responses_wire("a", &rung.url, &[&token])];
                let policy = RoutePolicy {
                    maximum_total_attempts: if per_route { 8 } else { 1 },
                    maximum_same_deployment_attempts: 2,
                    physical_route_cap: per_route.then_some(1),
                    refusal_failover: false,
                    throttle_redial: Some(SCHEDULE),
                    backoff: None,
                };
                let (won, guard) = harness
                    .run_with_policy("key", None, &route, policy, Duration::from_secs(30))
                    .await;
                assert!(matches!(finish(guard, won).await, Won::Failed(_)));
                assert_eq!(rung.accepted.lock().unwrap().len(), 1);
                let story = harness.story().await;
                assert_eq!(story["starts"].as_array().unwrap().len(), 1);
                assert_eq!(story["settles"].as_array().unwrap().len(), 1);
                if in_stream {
                    assert_eq!(story["settles"][0]["usage"]["input_tokens"], 30);
                }
            }
            let harness = Harness::new();
            let rung = spawn_rung(vec![Answer::Throttle(Some(1))]).await;
            let route = [super::ladder_tests::wire("a", &rung.url, 2)];
            let policy = RoutePolicy {
                maximum_total_attempts: if per_route { 8 } else { 1 },
                maximum_same_deployment_attempts: 2,
                physical_route_cap: per_route.then_some(1),
                refusal_failover: false,
                throttle_redial: Some(SCHEDULE),
                backoff: None,
            };
            let (won, guard) = harness
                .run_with_policy("key", None, &route, policy, Duration::from_secs(30))
                .await;
            assert!(matches!(finish(guard, won).await, Won::Failed(_)));
            assert_eq!(rung.accepted.lock().unwrap().len(), 1);
        }
    });
}

#[test]
fn a_refused_encrypted_reasoning_item_is_stripped_and_the_same_rung_redialed() {
    block_on(async {
        let harness = Harness::new();
        let before = METRICS.snapshot()["encrypted_reasoning_stripped"]
            .as_u64()
            .unwrap();
        // The rung refuses the sealed-elsewhere payload, then serves the
        // stripped replay on its second connection.
        let rung_a = spawn_rung(vec![
            Answer::Rejected(INVALID_ENCRYPTED_CONTENT_BODY),
            Answer::ResponsesStream(&[RESPONSES_TEXT_FRAME]),
        ])
        .await;
        let rung_b = spawn_rung(vec![Answer::ResponsesStream(&[RESPONSES_TEXT_FRAME])]).await;
        let route = [
            responses_wire("a", &rung_a.url, &["rsn_a_refused_encrypted_reasoning_item_is_stripped_and_the_same_rung_redialed_hA=="]),
            responses_wire("b", &rung_b.url, &["rsn_a_refused_encrypted_reasoning_item_is_stripped_and_the_same_rung_redialed_hA=="]),
        ];
        let (won, guard) = harness.run(&route, None, Duration::from_secs(60)).await;
        let won = finish(guard, won).await;
        let Won::Committed(committed) = won else {
            panic!("the same rung serves the stripped replay");
        };
        assert_eq!(committed.depth, 0);
        assert!(committed.encrypted_reasoning_stripped);
        assert!(matches!(
            committed.prefix.first(),
            Some(Event::ProviderOutputItemStarted { .. })
        ));
        drop(committed);

        // Two dials of the same rung: the replay as sent, then without the
        // refused item; the visible turns and tool items travel both times.
        let bodies = rung_a.bodies.lock().expect("lock").clone();
        assert_eq!(bodies.len(), 2);
        let sent: Value = serde_json::from_str(&bodies[0]).expect("first body");
        let repaired: Value = serde_json::from_str(&bodies[1]).expect("second body");
        assert_eq!(sent["input"].as_array().expect("input").len(), 5);
        let repaired_input = repaired["input"].as_array().expect("input");
        assert_eq!(repaired_input.len(), 4);
        assert!(repaired_input
            .iter()
            .all(|item| item.get("encrypted_content").is_none()));
        // The call the stripped reasoning governed replays id-less (the
        // provider would demand the reasoning item back for `fc_turn_1`).
        assert_eq!(sent["input"][2]["id"], "fc_turn_1");
        assert_eq!(repaired_input[1]["type"], "function_call");
        assert!(repaired_input[1].get("id").is_none());
        assert_eq!(repaired_input[1]["call_id"], "call_1");
        assert_eq!(repaired_input[3]["content"], "now apply it");
        assert_eq!(repaired["include"], json!(["reasoning.encrypted_content"]));
        assert!(rung_b.accepted.lock().expect("lock").is_empty());

        // Each physical dial owns exactly one reservation and settlement.
        let story = harness.story().await;
        assert_eq!(story["starts"].as_array().expect("starts").len(), 2);
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles.len(), 2);
        assert_eq!(settles[0]["outcome"], "failed");
        assert_eq!(settles[1]["outcome"], "completed");
        assert_eq!(story["counts"], json!([2, 0]));
        assert!(
            METRICS.snapshot()["encrypted_reasoning_stripped"]
                .as_u64()
                .unwrap()
                > before
        );
    });
}

#[test]
fn a_second_refusal_of_the_stripped_replay_surfaces_the_providers_400() {
    block_on(async {
        let harness = Harness::new();
        let rung_a = spawn_rung(vec![
            Answer::Rejected(INVALID_ENCRYPTED_CONTENT_BODY),
            Answer::Rejected(INVALID_ENCRYPTED_CONTENT_BODY),
        ])
        .await;
        let rung_b = spawn_rung(vec![Answer::ResponsesStream(&[RESPONSES_TEXT_FRAME])]).await;
        let route = [
            responses_wire(
                "a",
                &rung_a.url,
                &["rsn_a_second_refusal_of_the_stripped_replay_surfaces_the_providers_400_hA=="],
            ),
            responses_wire(
                "b",
                &rung_b.url,
                &["rsn_a_second_refusal_of_the_stripped_replay_surfaces_the_providers_400_hA=="],
            ),
        ];
        let (won, guard) = harness.run(&route, None, Duration::from_secs(60)).await;
        let won = finish(guard, won).await;
        let Won::Failed(error) = won else {
            panic!("a client error is the caller's, never walked down the ladder");
        };
        assert_eq!(error.status_code, 400, "{error:?}");
        // Exactly one repair: the rung was dialed twice and no other rung.
        assert_eq!(rung_a.accepted.lock().expect("lock").len(), 2);
        assert!(rung_b.accepted.lock().expect("lock").is_empty());
        // The two refused physical dials settle separately, with no fallback.
        let story = harness.story().await;
        assert_eq!(story["starts"].as_array().expect("starts").len(), 2);
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles.len(), 2);
        assert_eq!(settles[0]["outcome"], "failed");
        assert_eq!(settles[0]["failure"]["failure_class"], "invalid_request");
    });
}

#[test]
fn the_verdict_on_a_replay_with_nothing_to_strip_surfaces_at_once() {
    block_on(async {
        let harness = Harness::new();
        let rung_a = spawn_rung(vec![Answer::Rejected(INVALID_ENCRYPTED_CONTENT_BODY)]).await;
        let route = [responses_wire("a", &rung_a.url, &[])];
        let (won, guard) = harness.run(&route, None, Duration::from_secs(60)).await;
        let won = finish(guard, won).await;
        let Won::Failed(error) = won else {
            panic!("nothing to repair: the provider's 400 is the answer");
        };
        let story = harness.story().await;
        assert_eq!(error.status_code, 400, "{error:?} {story}");
        assert_eq!(rung_a.accepted.lock().expect("lock").len(), 1);
        assert_eq!(rung_a.bodies.lock().expect("lock").len(), 1);
    });
}

#[test]
fn a_remembered_repair_is_redialed_without_earning_the_refusal_again() {
    block_on(async {
        let harness = Harness::new();
        // The rung refuses the foreign payload, throttles the stripped
        // re-dial, and serves the post-backoff redial: that redial must
        // carry the stripped payload directly, not the refused original.
        let rung_a = spawn_rung(vec![
            Answer::Rejected(INVALID_ENCRYPTED_CONTENT_BODY),
            Answer::Throttle(None),
            Answer::ResponsesStream(&[RESPONSES_TEXT_FRAME]),
        ])
        .await;
        let rung_b = spawn_rung(vec![Answer::ResponsesStream(&[RESPONSES_TEXT_FRAME])]).await;
        let route = [
            DeploymentWire {
                native_tool_translation: Default::default(),
                throttle_redial_budget: 2,
                throttle_redial: None,
                ..responses_wire("a", &rung_a.url, &["rsn_a_remembered_repair_is_redialed_without_earning_the_refusal_again_hA=="])
            },
            responses_wire("b", &rung_b.url, &["rsn_a_remembered_repair_is_redialed_without_earning_the_refusal_again_hA=="]),
        ];
        let (won, guard) = harness
            .run(&route, Some(SCHEDULE), Duration::from_secs(60))
            .await;
        let won = finish(guard, won).await;
        let Won::Committed(committed) = won else {
            panic!("the redialed rung serves the remembered stripped payload");
        };
        assert_eq!(committed.depth, 0);
        assert!(committed.encrypted_reasoning_stripped);
        drop(committed);

        // Three dials: the replay as sent, the stripped re-dial, and the
        // post-backoff redial that starts from the stripped payload.
        let bodies = rung_a.bodies.lock().expect("lock").clone();
        assert_eq!(bodies.len(), 3);
        let has_encrypted = |body: &str| {
            let value: Value = serde_json::from_str(body).expect("body");
            value["input"]
                .as_array()
                .expect("input")
                .iter()
                .any(|item| item.get("encrypted_content").is_some())
        };
        assert!(has_encrypted(&bodies[0]));
        assert!(!has_encrypted(&bodies[1]));
        assert!(!has_encrypted(&bodies[2]));
        assert!(rung_b.accepted.lock().expect("lock").is_empty());

        // Refusal, repaired throttle and successful redial each own a row.
        let story = harness.story().await;
        let starts = story["starts"].as_array().expect("starts");
        assert_eq!(starts.len(), 3);
        assert_eq!(starts[1]["reasoning_repair"], true);
        assert_eq!(starts[2]["throttle_backoff"], true);
        assert_eq!(starts[2]["current_depth"], 0);
        assert_eq!(starts[2]["failure"]["failure_class"], "throttled");
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles.len(), 3);
        assert_eq!(settles[0]["outcome"], "failed");
        assert_eq!(settles[1]["outcome"], "failed");
        assert_eq!(settles[2]["outcome"], "completed");
    });
}

#[test]
fn a_remembered_refused_payload_is_stripped_before_the_first_dial() {
    block_on(async {
        // Every turn presents a FRESH bearer, as a hosted worker sees them
        // (the front exchanges the caller's key for an ephemeral token per
        // request); only the admitted caller identity is stable. Turn one:
        // the rung refuses the foreign payload (the verdict quotes its head
        // and tail), the stripped re-dial serves. The local payload of the
        // same conversation is not what was refused.
        let caller = Some("org-remembered:identity-a");
        let foreign = "rsn_a_remembered_refused_payload_hA==";
        let local = "gAAA_a_remembered_refused_payload_local==";
        let harness = Harness::new();
        let rung = spawn_rung(vec![
            Answer::Rejected(INVALID_ENCRYPTED_CONTENT_BODY),
            Answer::ResponsesStream(&[RESPONSES_TEXT_FRAME]),
        ])
        .await;
        let route = [responses_wire("a", &rung.url, &[foreign, local])];
        let (won, guard) = harness
            .run_as(
                "ephemeral-turn-1",
                caller,
                &route,
                None,
                Duration::from_secs(60),
            )
            .await;
        let Won::Committed(committed) = finish(guard, won).await else {
            panic!("the stripped re-dial serves turn one");
        };
        assert!(committed.encrypted_reasoning_stripped);
        drop(committed);
        assert_eq!(rung.bodies.lock().expect("lock").len(), 2);

        // Turn two, same caller identity under a different bearer, same
        // history: only the remembered payload is stripped, before any dial,
        // so the rung sees exactly one dial that still carries the local
        // payload; the disclosure holds.
        let later = Harness::new();
        let rung = spawn_rung(vec![Answer::ResponsesStream(&[RESPONSES_TEXT_FRAME])]).await;
        let route = [responses_wire("a", &rung.url, &[foreign, local])];
        let (won, guard) = later
            .run_as(
                "ephemeral-turn-2",
                caller,
                &route,
                None,
                Duration::from_secs(60),
            )
            .await;
        let Won::Committed(committed) = finish(guard, won).await else {
            panic!("the remembered strip serves turn two");
        };
        assert!(committed.encrypted_reasoning_stripped);
        drop(committed);
        let bodies = rung.bodies.lock().expect("lock").clone();
        assert_eq!(bodies.len(), 1);
        let sent: Value = serde_json::from_str(&bodies[0]).expect("body");
        let payloads: Vec<&str> = sent["input"]
            .as_array()
            .expect("input")
            .iter()
            .filter_map(|item| item.get("encrypted_content").and_then(Value::as_str))
            .collect();
        assert_eq!(payloads, vec![local]);
        let story = later.story().await;
        assert_eq!(story["starts"].as_array().expect("starts").len(), 1);
        assert_eq!(story["settles"][0]["outcome"], "completed");

        // Another caller identity replaying the same payload is not affected
        // by this caller's memory, even under the bearer turn one presented:
        // its first dial carries both payloads.
        let stranger = Harness::new();
        let rung = spawn_rung(vec![Answer::ResponsesStream(&[RESPONSES_TEXT_FRAME])]).await;
        let route = [responses_wire("a", &rung.url, &[foreign, local])];
        let (won, guard) = stranger
            .run_as(
                "ephemeral-turn-1",
                Some("org-remembered:identity-b"),
                &route,
                None,
                Duration::from_secs(60),
            )
            .await;
        let Won::Committed(committed) = finish(guard, won).await else {
            panic!("the stranger's replay serves as sent");
        };
        assert!(!committed.encrypted_reasoning_stripped);
        drop(committed);
        let body: Value =
            serde_json::from_str(&rung.bodies.lock().expect("lock")[0]).expect("body");
        assert_eq!(body["input"].as_array().expect("input").len(), 6);

        // An admission that names no caller identity (an older control
        // plane) repairs reactively and remembers nothing.
        let unscoped = Harness::new();
        let rung = spawn_rung(vec![
            Answer::Rejected(INVALID_ENCRYPTED_CONTENT_BODY),
            Answer::ResponsesStream(&[RESPONSES_TEXT_FRAME]),
        ])
        .await;
        let route = [responses_wire("a", &rung.url, &[foreign, local])];
        let (won, guard) = unscoped
            .run_as(
                "ephemeral-turn-3",
                None,
                &route,
                None,
                Duration::from_secs(60),
            )
            .await;
        let Won::Committed(committed) = finish(guard, won).await else {
            panic!("the reactive repair still serves without a caller scope");
        };
        assert!(committed.encrypted_reasoning_stripped);
        drop(committed);
        assert_eq!(rung.bodies.lock().expect("lock").len(), 2);
    });
}

#[test]
fn repaired_dial_open_failure_does_not_settle_prior_usage_as_the_full_total() {
    block_on(async {
        let harness = Harness::new();
        let rung = spawn_rung(vec![
            Answer::ResponsesFailed(RESPONSES_FAILED_ENCRYPTED_FRAME),
            Answer::Rejected(INVALID_ENCRYPTED_CONTENT_BODY),
        ])
        .await;
        let route = [responses_wire(
            "a",
            &rung.url,
            &["rsn_repaired_open_unknown_hA=="],
        )];
        let (won, guard) = harness.run(&route, None, Duration::from_secs(60)).await;
        assert!(matches!(finish(guard, won).await, Won::Failed(_)));
        assert_eq!(rung.bodies.lock().unwrap().len(), 2);
        let story = harness.story().await;
        let settles = story["settles"].as_array().unwrap();
        assert_eq!(settles.len(), 2);
        assert_eq!(settles[0]["usage"]["input_tokens"], 30);
        assert!(settles[1]["usage"]["input_tokens"].is_null());
        assert!(settles[1]["usage"]["output_tokens"].is_null());
        assert_eq!(settles[1]["usage_incomplete_due_to_disconnect"], false);
    });
}

#[test]
fn repaired_first_dial_without_usage_keeps_aggregate_unknown() {
    block_on(async {
        let harness = Harness::new();
        let rung = spawn_rung(vec![
            Answer::ResponsesFailed(concat!(
                "{\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":",
                "{\"code\":\"invalid_prompt\",\"message\":\"The encrypted content rsn_...hA== could not be verified. ",
                "Reason: Encrypted content could not be decrypted or parsed.\"}}}"
            )),
            Answer::ResponsesStream(&[RESPONSES_TEXT_FRAME]),
        ]).await;
        let route = [responses_wire(
            "a",
            &rung.url,
            &["rsn_unknown_first_meter_hA=="],
        )];
        let (won, mut guard) = harness.run(&route, None, Duration::from_secs(60)).await;
        let Won::Committed(mut committed) = won else {
            panic!("repair must serve")
        };
        crate::relay::collect_committed(
            &mut committed,
            std::time::Instant::now() + Duration::from_secs(5),
            Duration::from_secs(2),
            guard.started,
        )
        .await
        .unwrap();
        guard
            .settle(
                "completed",
                committed.usage.as_ref(),
                &committed.tool_names,
                None,
                true,
            )
            .await;
        let story = harness.story().await;
        let settles = story["settles"].as_array().unwrap();
        assert_eq!(settles.len(), 2);
        assert!(settles[0]["usage"]["input_tokens"].is_null(), "{story}");
        assert!(settles[0]["usage"]["output_tokens"].is_null(), "{story}");
        assert_eq!(settles[1]["usage"]["input_tokens"], 12);
        assert_eq!(settles[1]["usage"]["output_tokens"], 3);
        assert_eq!(settles[0]["usage_incomplete_due_to_disconnect"], false);
        assert_eq!(rung.bodies.lock().unwrap().len(), 2);
    });
}

#[test]
fn an_in_stream_refusal_of_encrypted_reasoning_is_repaired_like_a_pre_stream_one() {
    block_on(async {
        // The relay answers 200 and fails the stream on its first frame with
        // OpenAI's sentence under `invalid_prompt`; the stripped re-dial serves.
        let harness = Harness::new();
        let rung = spawn_rung(vec![
            Answer::ResponsesFailed(RESPONSES_FAILED_ENCRYPTED_FRAME),
            Answer::ResponsesStream(&[RESPONSES_TEXT_FRAME]),
        ])
        .await;
        let route = [responses_wire(
            "a",
            &rung.url,
            &["rsn_an_in_stream_refusal_hA=="],
        )];
        let (won, guard) = harness.run(&route, None, Duration::from_secs(60)).await;
        let Won::Committed(mut committed) = won else {
            panic!("the stripped re-dial serves after an in-stream refusal");
        };
        assert_eq!(committed.depth, 0);
        assert!(committed.encrypted_reasoning_stripped);
        // The served stream carries only its own physical attempt's meter.
        let far = std::time::Instant::now() + Duration::from_secs(30);
        let folded = loop {
            match committed
                .relay
                .next_event(far, Duration::from_secs(10), std::time::Instant::now())
                .await
            {
                Ok(Some(Event::Usage(usage))) => break usage,
                Ok(Some(_)) => continue,
                other => panic!("the served stream reports usage: {other:?}"),
            }
        };
        assert_eq!(folded.input_tokens, Some(12));
        assert_eq!(folded.output_tokens, Some(3));
        let won = finish(guard, Won::Committed(committed)).await;
        drop(won);
        let bodies = rung.bodies.lock().expect("lock").clone();
        assert_eq!(bodies.len(), 2);
        let sent: Value = serde_json::from_str(&bodies[0]).expect("first body");
        let repaired: Value = serde_json::from_str(&bodies[1]).expect("second body");
        assert_eq!(sent["input"].as_array().expect("input").len(), 5);
        let repaired_input = repaired["input"].as_array().expect("input");
        assert_eq!(repaired_input.len(), 4);
        assert!(repaired_input
            .iter()
            .all(|item| item.get("encrypted_content").is_none()));
        // The refusal's billed usage remains on its own failed attempt.
        let story = harness.story().await;
        assert_eq!(story["starts"].as_array().expect("starts").len(), 2);
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles.len(), 2);
        assert_eq!(settles[0]["outcome"], "failed");
        assert_eq!(settles[0]["usage"]["input_tokens"], 30);
        assert_eq!(settles[1]["outcome"], "completed");

        // Another in-stream failure keeps the ordinary verdict: no re-dial.
        let plain = Harness::new();
        let rung = spawn_rung(vec![Answer::ResponsesFailed(RESPONSES_FAILED_OTHER_FRAME)]).await;
        let route = [responses_wire(
            "a",
            &rung.url,
            &["rsn_an_in_stream_refusal_other_hA=="],
        )];
        let (won, guard) = plain.run(&route, None, Duration::from_secs(60)).await;
        let Won::Failed(error) = finish(guard, won).await else {
            panic!("an unrelated in-stream failure is not repaired");
        };
        assert_eq!(error.status_code, 400, "{error:?}");
        assert_eq!(rung.accepted.lock().expect("lock").len(), 1);
    });
}
