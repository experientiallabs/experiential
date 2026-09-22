//! Accelerated real loopback regressions for private progress and safe commitment.

use super::ladder_tests::{block_on, read_request_body, spawn_rung, wire, Answer, Harness};
use super::*;
use crate::relay::collect_committed;
use tokio::io::AsyncWriteExt;

const REASONING: &str =
    "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"private thought\"}}]}\n\n";
const TEXT: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n";
const DONE: &str = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":9}}\n\ndata: [DONE]\n\n";
const SECONDARY: &[&str] = &["{\"choices\":[{\"delta\":{\"content\":\"secondary\"}}]}"];

/// Stream a finite script, one frame per tick, then remain open past the idle bound.
async fn paced_rung(frames: Vec<&'static str>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        read_request_body(&mut socket).await;
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n",
            )
            .await
            .unwrap();
        for frame in frames {
            if socket.write_all(frame.as_bytes()).await.is_err() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(30)).await;
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    });
    format!("http://{address}/v1/chat/completions")
}

fn private_wire(url: &str) -> DeploymentWire {
    let mut rung = wire("private", url, 0);
    rung.fireworks_reasoning_route_sha256 = Some("a".repeat(64));
    rung.timeout_seconds = 0.15;
    rung.time_to_first_token_base_seconds = Some(0.2);
    rung
}

#[test]
fn hidden_reasoning_then_pings_fails_over_without_publishing_the_carrier() {
    block_on(async {
        let mut frames = vec![REASONING];
        frames.extend(std::iter::repeat_n(": ping\n\n", 20));
        let primary = paced_rung(frames).await;
        let secondary = spawn_rung(vec![Answer::Stream(SECONDARY)]).await;
        let harness = Harness::new();
        let start = Instant::now();
        let (won, mut guard) = harness
            .run(
                &[private_wire(&primary), wire("secondary", &secondary.url, 0)],
                None,
                Duration::from_secs(3),
            )
            .await;
        let Won::Committed(committed) = won else {
            panic!("secondary must answer")
        };
        assert_eq!(
            committed.depth, 1,
            "hidden thought must not commit the primary"
        );
        assert!(start.elapsed() < Duration::from_millis(600));
        assert!(committed
            .prefix
            .iter()
            .all(|event| !matches!(event, Event::ReasoningContentDelta { .. })));
        guard.settle("completed", None, &[], None, true).await;
        let story = harness.story().await;
        assert_eq!(story["counts"], json!([1, 1]));
        assert_eq!(story["settles"][0]["outcome"], "failed");
        assert_eq!(story["settles"][0]["opened"], true);
        assert_eq!(
            story["starts"][1]["failure"]["retryable_same_deployment"],
            false
        );
        assert_eq!(story["settles"].as_array().unwrap().len(), 2);
    });
}

#[test]
fn active_private_reasoning_survives_first_token_window_and_retains_the_winner() {
    block_on(async {
        let mut frames = vec![REASONING; 12];
        frames.extend([TEXT, DONE]);
        let primary = paced_rung(frames).await;
        let harness = Harness::new();
        let (won, mut guard) = harness
            .run(&[private_wire(&primary)], None, Duration::from_secs(3))
            .await;
        let Won::Committed(mut committed) = won else {
            panic!("active reasoning must finish")
        };
        assert_eq!(committed.depth, 0);
        let events = collect_committed(
            &mut committed,
            Instant::now() + Duration::from_secs(2),
            Duration::from_millis(150),
            guard.started,
        )
        .await
        .unwrap();
        let reasoning: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::ReasoningContentDelta { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning, vec!["private thought".repeat(12)]);
        assert!(events
            .iter()
            .any(|event| matches!(event, Event::TextDelta(text) if text == "answer")));
        assert!(matches!(events.last(), Some(Event::Completed)));
        assert_eq!(committed.usage.as_ref().unwrap().output_tokens, Some(9));
        guard
            .settle("completed", committed.usage.as_ref(), &[], None, true)
            .await;
        assert_eq!(harness.story().await["counts"], json!([1, 0]));
    });
}

#[test]
fn unexposed_compatible_reasoning_keeps_generation_alive_without_a_carrier() {
    block_on(async {
        for after_visible_output in [false, true] {
            let mut frames = Vec::new();
            if after_visible_output {
                frames.push(TEXT);
            }
            frames.extend(std::iter::repeat_n(REASONING, 12));
            frames.extend([TEXT, DONE]);
            let primary = paced_rung(frames).await;
            let harness = Harness::new();
            let mut rung = private_wire(&primary);
            rung.fireworks_reasoning_route_sha256 = None;
            let (won, mut guard) = harness.run(&[rung], None, Duration::from_secs(3)).await;
            let Won::Committed(mut committed) = won else {
                panic!("unexposed reasoning is generation progress before visible output")
            };
            let events = collect_committed(
                &mut committed,
                Instant::now() + Duration::from_secs(2),
                Duration::from_millis(150),
                guard.started,
            )
            .await
            .expect("unexposed reasoning is generation progress after visible output");
            assert!(matches!(events.last(), Some(Event::Completed)));
            assert!(!events
                .iter()
                .any(|event| matches!(event, Event::ReasoningContentDelta { .. })));
            let public = crate::encode::completed_chat_body_with_ignored(
                "request-private",
                "kimi",
                1,
                &events,
                &[],
                false,
            )
            .unwrap();
            assert!(!public.body.to_string().contains("private thought"));
            assert!(public.body["choices"][0]["message"]
                .get("reasoning_content")
                .is_none());
            let text: String = events
                .iter()
                .filter_map(|event| match event {
                    Event::TextDelta(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(
                text,
                "answer".repeat(if after_visible_output { 2 } else { 1 })
            );
            guard
                .settle("completed", committed.usage.as_ref(), &[], None, true)
                .await;
            assert_eq!(harness.story().await["counts"], json!([1, 0]));
        }
    });
}

#[test]
fn empty_unexposed_reasoning_and_usage_do_not_renew_progress() {
    block_on(async {
        let empty = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"\"}}]}\n\n";
        let usage =
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":9}}\n\n";
        let mut frames = vec![REASONING];
        frames.extend([empty, usage, ": ping\n\n"].repeat(7));
        let primary = paced_rung(frames).await;
        let secondary = spawn_rung(vec![Answer::Stream(SECONDARY)]).await;
        let harness = Harness::new();
        let mut rung = private_wire(&primary);
        rung.fireworks_reasoning_route_sha256 = None;
        let (won, _guard) = harness
            .run(
                &[rung, wire("secondary", &secondary.url, 0)],
                None,
                Duration::from_secs(3),
            )
            .await;
        let Won::Committed(committed) = won else {
            panic!("empty deltas must allow failover")
        };
        assert_eq!(committed.depth, 1);
        let story = harness.story().await;
        assert_eq!(story["settles"][0]["failure"]["failure_class"], "transport");
    });
}

#[test]
fn unexposed_reasoning_cannot_extend_the_total_request_deadline() {
    block_on(async {
        let primary = paced_rung(vec![REASONING; 20]).await;
        let harness = Harness::new();
        let mut rung = private_wire(&primary);
        rung.fireworks_reasoning_route_sha256 = None;
        let (won, _guard) = harness.run(&[rung], None, Duration::from_millis(180)).await;
        assert!(matches!(won, Won::Failed(_)));
        assert_eq!(
            harness.story().await["settles"][0]["failure"]["failure_class"],
            "timeout"
        );
    });
}

#[test]
fn observed_usage_settles_before_a_private_or_committed_stall() {
    block_on(async {
        for first in [REASONING, TEXT] {
            let primary = paced_rung(vec![first,
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":9}}\n\n",
                ": ping\n\n",
            ]).await;
            let harness = Harness::new();
            let (won, mut guard) = harness
                .run(&[private_wire(&primary)], None, Duration::from_secs(3))
                .await;
            if first == TEXT {
                let Won::Committed(mut committed) = won else {
                    panic!("text commits")
                };
                let failure = collect_committed(
                    &mut committed,
                    Instant::now() + Duration::from_secs(2),
                    Duration::from_millis(150),
                    guard.started,
                )
                .await
                .unwrap_err();
                guard
                    .settle(
                        "failed",
                        committed.usage.as_ref(),
                        &[],
                        Some(&failure),
                        true,
                    )
                    .await;
            } else {
                assert!(matches!(won, Won::Failed(_)));
            }
            let story = harness.story().await;
            assert_eq!(story["settles"].as_array().unwrap().len(), 1);
            assert_eq!(story["settles"][0]["usage"]["input_tokens"], 7);
            assert_eq!(story["settles"][0]["usage"]["output_tokens"], 9);
        }
    });
}

#[test]
fn reported_usage_survives_failed_events_and_terminal_less_eof() {
    block_on(async {
        for failed_event in [true, false] {
            let primary = spawn_rung(vec![Answer::ResponsesFailed(if failed_event {
                "{\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":9}}\n\ndata: {\"error\":{\"code\":\"server_error\",\"message\":\"failed\"}}"
            } else {
                "{\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":9}}"
            })]).await;
            let harness = Harness::new();
            let (_won, _guard) = harness
                .run(&[private_wire(&primary.url)], None, Duration::from_secs(3))
                .await;
            let story = harness.story().await;
            assert_eq!(story["settles"][0]["usage"]["input_tokens"], 7);
            assert_eq!(story["settles"][0]["usage"]["output_tokens"], 9);
        }
    });
}

#[test]
fn progress_failure_respects_timeout_only_successor_policy() {
    block_on(async {
        let primary = paced_rung(vec![REASONING, ": ping\n\n"]).await;
        let secondary = spawn_rung(vec![Answer::Stream(SECONDARY)]).await;
        let harness = Harness::new();
        harness
            .configure(json!({"rules": [null, ["timeout"]]}))
            .await;
        let mut successor = wire("secondary", &secondary.url, 0);
        successor.failover_only_on = Some(vec!["timeout".to_string()]);
        let (won, _guard) = harness
            .run(
                &[private_wire(&primary), successor],
                None,
                Duration::from_secs(3),
            )
            .await;
        assert!(
            matches!(won, Won::Failed(_)),
            "transport idle is not a timeout-policy match"
        );
        assert!(secondary.accepted.lock().unwrap().is_empty());
        let story = harness.story().await;
        assert_eq!(story["settles"][0]["failure"]["failure_class"], "transport");
        assert_eq!(story["settles"].as_array().unwrap().len(), 1);
    });
}

#[test]
fn exposed_reasoning_is_irreversible_output() {
    block_on(async {
        let primary = paced_rung(vec![REASONING, ": ping\n\n"]).await;
        let harness = Harness::new();
        let mut rung = private_wire(&primary);
        rung.reasoning_output_exposed = true;
        let (won, mut guard) = harness.run(&[rung], None, Duration::from_secs(3)).await;
        let Won::Committed(mut committed) = won else {
            panic!("exposed thought commits")
        };
        assert!(matches!(
            committed.prefix.as_slice(),
            [Event::ReasoningContentDelta { .. }]
        ));
        let failure = collect_committed(
            &mut committed,
            Instant::now() + Duration::from_secs(2),
            Duration::from_millis(150),
            guard.started,
        )
        .await
        .unwrap_err();
        guard
            .settle(
                "failed",
                committed.usage.as_ref(),
                &[],
                Some(&failure),
                true,
            )
            .await;
        assert_eq!(harness.story().await["counts"], json!([1, 0]));
    });
}

#[test]
fn server_tool_action_commits_even_before_any_answer_text() {
    block_on(async {
        let primary = paced_rung(vec![
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"server_tool_use\",\"id\":\"srvtoolu_1\",\"name\":\"web_search\",\"input\":{}}}\n\n",
            "event: ping\ndata: {\"type\":\"ping\"}\n\n",
        ]).await;
        let secondary = spawn_rung(vec![Answer::Stream(SECONDARY)]).await;
        let harness = Harness::new();
        let mut rung = private_wire(&primary);
        rung.dialect = "anthropic_messages".to_string();
        let (won, mut guard) = harness
            .run(
                &[rung, wire("secondary", &secondary.url, 0)],
                None,
                Duration::from_secs(3),
            )
            .await;
        let Won::Committed(mut committed) = won else {
            panic!("server action commits")
        };
        assert!(committed
            .prefix
            .iter()
            .any(|event| matches!(event, Event::ServerToolUseStarted { .. })));
        let failure = collect_committed(
            &mut committed,
            Instant::now() + Duration::from_secs(2),
            Duration::from_millis(150),
            guard.started,
        )
        .await
        .unwrap_err();
        guard
            .settle(
                "failed",
                committed.usage.as_ref(),
                &[],
                Some(&failure),
                true,
            )
            .await;
        assert!(secondary.accepted.lock().unwrap().is_empty());
    });
}

#[test]
fn cancelling_private_progress_settles_once_without_dispatching_a_successor() {
    block_on(async {
        let primary = paced_rung(vec![REASONING; 20]).await;
        let secondary = spawn_rung(vec![Answer::Stream(SECONDARY)]).await;
        let harness = Harness::new();
        let route = [private_wire(&primary), wire("secondary", &secondary.url, 0)];
        assert!(tokio::time::timeout(
            Duration::from_millis(100),
            harness.run(&route, None, Duration::from_secs(3))
        )
        .await
        .is_err());
        let until = Instant::now() + Duration::from_secs(2);
        let story = loop {
            let story = harness.story().await;
            if !story["settles"].as_array().unwrap().is_empty() {
                break story;
            }
            assert!(Instant::now() < until, "cancelled attempt must settle");
            tokio::time::sleep(Duration::from_millis(10)).await;
        };
        assert_eq!(story["settles"].as_array().unwrap().len(), 1);
        assert_eq!(story["settles"][0]["outcome"], "failed");
        assert_eq!(story["settles"][0]["failure"]["failure_class"], "cancelled");
        assert_eq!(story["settles"][0]["opened"], true);
        assert!(story["settles"][0]["first_token_at"].is_string());
        assert!(secondary.accepted.lock().unwrap().is_empty());
    });
}

#[test]
fn private_only_success_preserves_terminal_usage_and_responses_continuation() {
    block_on(async {
        let primary = paced_rung(vec![REASONING, DONE]).await;
        let harness = Harness::new();
        let (won, mut guard) = harness
            .run(&[private_wire(&primary)], None, Duration::from_secs(3))
            .await;
        let Won::Committed(mut committed) = won else {
            panic!("successful hidden-only turn retains its carrier path")
        };
        let events = collect_committed(
            &mut committed,
            Instant::now() + Duration::from_secs(2),
            Duration::from_millis(150),
            guard.started,
        )
        .await
        .unwrap();
        assert_eq!(committed.usage.as_ref().unwrap().output_tokens, Some(9));
        let body = crate::encode_responses::completed_responses_body_with_carrier(
            "req",
            "model",
            1,
            crate::encode_responses::ResponsesEnvelope {
                include_encrypted_reasoning: true,
                ..Default::default()
            },
            &events,
            Some("sealed-carrier"),
        )
        .unwrap();
        assert!(
            body.body["output"][0]["encrypted_content"].is_null(),
            "private-only turns without tools have no carrier under the existing contract"
        );
        assert_eq!(body.body["usage"]["output_tokens"], 9);
        assert!(!body.body.to_string().contains("private thought"));
        guard
            .settle("completed", committed.usage.as_ref(), &[], None, true)
            .await;
    });
}

#[test]
fn winning_tool_turn_keeps_exact_private_carrier_on_all_three_surfaces() {
    block_on(async {
        let primary = paced_rung(vec![
            REASONING, REASONING,
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_one\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{}\"}}]}}]}\n\n",
            DONE,
        ]).await;
        let harness = Harness::new();
        let (won, mut guard) = harness
            .run(&[private_wire(&primary)], None, Duration::from_secs(3))
            .await;
        let Won::Committed(mut committed) = won else {
            panic!("tool turn commits")
        };
        let events = collect_committed(
            &mut committed,
            Instant::now() + Duration::from_secs(2),
            Duration::from_millis(150),
            guard.started,
        )
        .await
        .unwrap();
        let candidate = crate::encode::reasoning_carrier_candidate(&events)
            .unwrap()
            .unwrap();
        assert_eq!(candidate.content, "private thoughtprivate thought");
        assert_eq!(candidate.tool_calls[0].call_id, "call_one");
        assert_eq!(candidate.tool_calls[0].raw_arguments, "{}");
        assert_eq!(candidate.route_sha256, "a".repeat(64));
        let chat = crate::encode::completed_chat_body_with_carrier(
            "req",
            "model",
            1,
            &events,
            &[],
            Some("sealed-carrier"),
            false,
        )
        .unwrap();
        assert!(!chat.body.to_string().contains("private thought"));
        assert!(chat.body.to_string().contains("sealed-carrier"));
        let responses = crate::encode_responses::completed_responses_body_with_carrier(
            "req",
            "model",
            1,
            crate::encode_responses::ResponsesEnvelope {
                include_encrypted_reasoning: true,
                ..Default::default()
            },
            &events,
            Some("sealed-carrier"),
        )
        .unwrap();
        assert!(!responses.body.to_string().contains("private thought"));
        assert!(responses.body.to_string().contains("sealed-carrier"));
        let mut messages = crate::encode_messages::MessagesSseEncoder::new("req", "model");
        messages.start().unwrap();
        messages.set_reasoning_content_carrier("sealed-carrier".to_string());
        let output = events
            .iter()
            .flat_map(|event| messages.feed(event).unwrap())
            .collect::<Vec<_>>()
            .join("");
        assert!(!output.contains("private thought"));
        assert!(output.contains("sealed-carrier"));
        guard
            .settle("completed", committed.usage.as_ref(), &[], None, true)
            .await;
    });
}

#[test]
fn visible_output_then_pings_is_terminal_not_a_second_dispatch() {
    block_on(async {
        let mut frames = vec![TEXT];
        frames.extend(std::iter::repeat_n(": ping\n\n", 20));
        let primary = paced_rung(frames).await;
        let secondary = spawn_rung(vec![Answer::Stream(SECONDARY)]).await;
        let harness = Harness::new();
        let (won, mut guard) = harness
            .run(
                &[private_wire(&primary), wire("secondary", &secondary.url, 0)],
                None,
                Duration::from_secs(3),
            )
            .await;
        let Won::Committed(mut committed) = won else {
            panic!("visible text commits")
        };
        assert_eq!(committed.depth, 0);
        let started = Instant::now();
        let failure = collect_committed(
            &mut committed,
            Instant::now() + Duration::from_secs(2),
            Duration::from_millis(150),
            guard.started,
        )
        .await
        .expect_err("pings never keep generation alive");
        assert!(started.elapsed() < Duration::from_millis(600));
        guard
            .settle(
                "failed",
                committed.usage.as_ref(),
                &[],
                Some(&failure),
                true,
            )
            .await;
        assert!(secondary.accepted.lock().unwrap().is_empty());
        assert_eq!(harness.story().await["counts"], json!([1, 0]));
    });
}
