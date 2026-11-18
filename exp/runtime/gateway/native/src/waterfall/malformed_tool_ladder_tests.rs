//! A streamed tool call whose arguments never parse, met before anything
//! reached the caller, is retried once: on an untried sibling first, else one
//! redial of the same rung. A turn that already showed the caller output keeps
//! its terminal error, and a second malformed call is the request's answer.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::json;
use tokio::io::AsyncWriteExt;

use super::ladder_tests::{
    block_on, finish, read_request, spawn_rung, wire, Answer, Harness, Rung, TEXT_FRAME,
};
use super::*;
use crate::relay::collect_committed;

/// One rung that sends the first `split` frames, pauses, then the rest and
/// `[DONE]`: the attempt sees the opening frames as their own read, the way
/// a provider's live stream reaches the gateway.
async fn spawn_paced_rung(frames: &'static [&'static str], split: usize) -> Rung {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    let accepted = Arc::new(Mutex::new(Vec::new()));
    let recorder = accepted.clone();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        recorder.lock().expect("lock").push(Instant::now());
        read_request(&mut socket).await;
        let render = |frames: &[&str]| -> String {
            frames
                .iter()
                .map(|frame| format!("data: {frame}\n\n"))
                .collect()
        };
        let head = render(&frames[..split]);
        let tail = format!("{}data: [DONE]\n\n", render(&frames[split..]));
        let preamble = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
             content-length: {}\r\nconnection: close\r\n\r\n",
            head.len() + tail.len(),
        );
        socket
            .write_all(format!("{preamble}{head}").as_bytes())
            .await
            .expect("write");
        socket.flush().await.expect("flush");
        tokio::time::sleep(Duration::from_millis(300)).await;
        socket.write_all(tail.as_bytes()).await.expect("write");
        let _ = socket.shutdown().await;
    });
    Rung {
        url: format!("http://{address}/v1/chat/completions"),
        accepted,
        bodies: Arc::new(Mutex::new(Vec::new())),
        keys: Arc::new(Mutex::new(Vec::new())),
    }
}

/// A tool call whose streamed arguments are not one JSON object (two adjacent
/// strings, the shape DeepSeek V4.1 Flash on Novita emitted live on
/// 2026-11-18: "expected `,` or `}` at line 1 column 18").
const BROKEN_CALL_FRAMES: &[&str] = &[
    concat!(
        "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",",
        "\"type\":\"function\",\"function\":{\"name\":\"read\",\"arguments\":\"\"}}]}}]}"
    ),
    concat!(
        "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,",
        "\"function\":{\"arguments\":\"{\\\"path\\\": \\\"a\\\" \"}}]}}]}"
    ),
    concat!(
        "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,",
        "\"function\":{\"arguments\":\"\\\"b\\\"}\"}}]}}]}"
    ),
    "{\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}",
];

/// The same call with valid arguments.
const VALID_CALL_FRAMES: &[&str] = &[
    concat!(
        "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",",
        "\"type\":\"function\",\"function\":{\"name\":\"read\",\"arguments\":\"\"}}]}}]}"
    ),
    concat!(
        "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,",
        "\"function\":{\"arguments\":\"{\\\"path\\\": \\\"a\\\"}\"}}]}}]}"
    ),
    "{\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}",
];

/// Text the caller sees, then the broken call.
const TEXT_THEN_BROKEN_CALL_FRAMES: &[&str] = &[
    TEXT_FRAME,
    concat!(
        "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",",
        "\"type\":\"function\",\"function\":{\"name\":\"read\",\"arguments\":\"\"}}]}}]}"
    ),
    concat!(
        "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,",
        "\"function\":{\"arguments\":\"{\\\"path\\\": \\\"a\\\" \\\"b\\\"}\"}}]}}]}"
    ),
    "{\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}",
];

fn completed_call(events: &[Event]) -> &str {
    events
        .iter()
        .find_map(|event| match event {
            Event::ToolCallCompleted { call, .. } => Some(call.raw_arguments.as_str()),
            _ => None,
        })
        .expect("the served turn completes its tool call")
}

#[test]
fn a_held_malformed_tool_call_is_served_by_the_sibling() {
    block_on(async {
        let harness = Harness::new();
        // The call opens in its own read, long before its arguments fail:
        // without the hold that opening would have committed the stream.
        let broken = spawn_paced_rung(BROKEN_CALL_FRAMES, 1).await;
        let sibling = spawn_rung(vec![Answer::Stream(VALID_CALL_FRAMES)]).await;
        let route = [
            wire("novita", &broken.url, 0),
            wire("azure", &sibling.url, 0),
        ];
        let (won, guard) = harness.run(&route, None, Duration::from_secs(30)).await;
        let Won::Committed(mut committed) = won else {
            panic!("the sibling serves the turn the broken call never reached the caller in");
        };
        assert_eq!(committed.depth, 1);
        // The served turn reaches the caller whole: the call opens, streams
        // its valid arguments, and completes.
        // The hold released on the turn's terminal: the whole answer is the
        // committed prefix, nothing is left to drain.
        let events = std::mem::take(&mut committed.prefix);
        assert!(matches!(
            events.first(),
            Some(Event::ToolCallStarted { .. })
        ));
        assert_eq!(completed_call(&events), "{\"path\": \"a\"}");
        assert!(matches!(events.last(), Some(Event::Completed)));
        finish(guard, Won::Committed(committed)).await;
        assert_eq!(broken.accepted.lock().expect("lock").len(), 1);
        assert_eq!(sibling.accepted.lock().expect("lock").len(), 1);
        let story = harness.story().await;
        let starts = story["starts"].as_array().expect("starts");
        assert_eq!(starts.len(), 2);
        let failure = &starts[1]["failure"];
        assert_eq!(failure["failure_class"], "malformed_response");
        assert_eq!(failure["malformed_tool_call"], true);
        assert_eq!(failure["failover_eligible"], true);
        assert!(failure["safe_message"]
            .as_str()
            .expect("message")
            .starts_with("streamed tool arguments are not valid JSON"));
        // The broken dial keeps its own failed attempt row, disclosed.
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles.len(), 2);
        assert_eq!(settles[0]["outcome"], "failed");
        assert_eq!(settles[0]["finalize"], false);
        assert_eq!(settles[0]["failure"]["malformed_tool_call"], true);
        assert_eq!(settles[1]["outcome"], "completed");
    });
}

#[test]
fn a_malformed_tool_call_after_visible_output_keeps_its_terminal_error() {
    block_on(async {
        let harness = Harness::new();
        // The text reaches the caller in its own read before the call opens.
        let broken = spawn_paced_rung(TEXT_THEN_BROKEN_CALL_FRAMES, 1).await;
        let sibling = spawn_rung(vec![Answer::Stream(VALID_CALL_FRAMES)]).await;
        let route = [
            wire("novita", &broken.url, 0),
            wire("azure", &sibling.url, 0),
        ];
        let (won, mut guard) = harness.run(&route, None, Duration::from_secs(30)).await;
        let Won::Committed(mut committed) = won else {
            panic!("the text delta commits the live stream");
        };
        assert_eq!(committed.depth, 0);
        // The surface drains the live relay and meets the malformed call
        // itself; it ends the stream with the protocol's error, as before.
        let failure = collect_committed(
            &mut committed,
            Instant::now() + Duration::from_secs(10),
            Duration::from_secs(10),
            Instant::now(),
        )
        .await
        .expect_err("the malformed call ends the stream it was already part of");
        assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
        assert!(failure.malformed_tool_call);
        guard
            .settle("failed", None, &[], Some(&failure), true)
            .await;
        assert!(sibling.accepted.lock().expect("lock").is_empty());
        let story = harness.story().await;
        assert_eq!(story["starts"].as_array().expect("starts").len(), 1);
    });
}

#[test]
fn a_single_lane_redials_once_after_a_malformed_tool_call() {
    block_on(async {
        let harness = Harness::new();
        harness
            .configure(json!({"rules": [null], "same_redial": 2}))
            .await;
        let lane = spawn_rung(vec![
            Answer::Stream(BROKEN_CALL_FRAMES),
            Answer::Stream(VALID_CALL_FRAMES),
        ])
        .await;
        let route = [wire("novita", &lane.url, 0)];
        let (won, guard) = harness.run(&route, None, Duration::from_secs(30)).await;
        let Won::Committed(mut committed) = won else {
            panic!("the redial's valid call serves the request");
        };
        assert_eq!(committed.depth, 0);
        // The hold released on the turn's terminal: the whole answer is the
        // committed prefix, nothing is left to drain.
        let events = std::mem::take(&mut committed.prefix);
        assert_eq!(completed_call(&events), "{\"path\": \"a\"}");
        finish(guard, Won::Committed(committed)).await;
        assert_eq!(lane.accepted.lock().expect("lock").len(), 2);
        // The resample is a new provider intent: a wire that deduplicates
        // the stable key must not replay the malformed answer.
        let keys = lane.keys.lock().expect("lock").clone();
        assert_eq!(keys[0].as_deref(), Some("op-novita"));
        assert_eq!(keys[1].as_deref(), Some("op-novita.resample"));
        let story = harness.story().await;
        let starts = story["starts"].as_array().expect("starts");
        assert_eq!(starts.len(), 2);
        assert_eq!(starts[1]["current_depth"], 0);
        assert_eq!(starts[1]["failure"]["malformed_tool_call"], true);
        assert_eq!(starts[1]["failure"]["retryable_same_deployment"], true);
    });
}

#[test]
fn a_second_malformed_tool_call_is_the_requests_answer() {
    block_on(async {
        let harness = Harness::new();
        harness
            .configure(json!({"rules": [null], "same_redial": 2}))
            .await;
        let lane = spawn_rung(vec![
            Answer::Stream(BROKEN_CALL_FRAMES),
            Answer::Stream(BROKEN_CALL_FRAMES),
            Answer::Stream(VALID_CALL_FRAMES),
        ])
        .await;
        let route = [wire("novita", &lane.url, 0)];
        let (won, _guard) = harness.run(&route, None, Duration::from_secs(30)).await;
        let Won::Failed(error) = won else {
            panic!("the one retry is spent: the second malformed call is the answer");
        };
        assert_eq!(error.status_code, 502);
        assert_eq!(lane.accepted.lock().expect("lock").len(), 2);
        let story = harness.story().await;
        assert_eq!(story["starts"].as_array().expect("starts").len(), 2);
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles.len(), 2);
        assert_eq!(settles[1]["finalize"], true);
        assert_eq!(settles[1]["failure"]["malformed_tool_call"], true);
    });
}

/// One rung answering `first`, then resetting a connection, then `last`.
async fn spawn_resetting_rung(
    first: &'static [&'static str],
    last: &'static [&'static str],
) -> Rung {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    let accepted = Arc::new(Mutex::new(Vec::new()));
    let keys = Arc::new(Mutex::new(Vec::new()));
    let (recorder, key_recorder) = (accepted.clone(), keys.clone());
    tokio::spawn(async move {
        for answer in [Some(first), None, Some(last)] {
            let (mut socket, _) = listener.accept().await.expect("accept");
            recorder.lock().expect("lock").push(Instant::now());
            let (key, _) = read_request(&mut socket).await;
            key_recorder.lock().expect("lock").push(key);
            let Some(frames) = answer else {
                drop(socket);
                continue;
            };
            let mut body: String = frames
                .iter()
                .map(|frame| format!("data: {frame}\n\n"))
                .collect();
            body.push_str("data: [DONE]\n\n");
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                 content-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len(),
            );
            socket.write_all(response.as_bytes()).await.expect("write");
            let _ = socket.shutdown().await;
        }
    });
    Rung {
        url: format!("http://{address}/v1/chat/completions"),
        accepted,
        bodies: Arc::new(Mutex::new(Vec::new())),
        keys,
    }
}

#[test]
fn a_resample_keeps_its_own_key_through_a_later_safe_retry() {
    block_on(async {
        let harness = Harness::new();
        harness
            .configure(json!({"rules": [null], "same_redial": 3}))
            .await;
        let lane = spawn_resetting_rung(BROKEN_CALL_FRAMES, VALID_CALL_FRAMES).await;
        let route = [wire("novita", &lane.url, 0)];
        let policy = RoutePolicy {
            maximum_total_attempts: 8,
            maximum_same_deployment_attempts: 3,
            refusal_failover: false,
            throttle_redial: None,
            physical_route_cap: None,
            backoff: None,
        };
        let (won, guard) = harness
            .run_with_policy("key", None, &route, policy, Duration::from_secs(30))
            .await;
        let Won::Committed(committed) = won else {
            panic!("the resample's safe retry serves the request");
        };
        finish(guard, Won::Committed(committed)).await;
        // The reset resample is retried under the resample's key, never the
        // original one a deduplicating wire would answer with the bad call.
        let keys = lane.keys.lock().expect("lock").clone();
        assert_eq!(
            keys,
            vec![
                Some("op-novita".to_string()),
                Some("op-novita.resample".to_string()),
                Some("op-novita.resample".to_string()),
            ]
        );
    });
}

/// DeepSeek DSML tool-call markup a self-hosted vLLM lane relayed as content
/// because its parser did not recognise it (a lost opening tag): the guard
/// holds it and fails the turn as a malformed tool call.
const BROKEN_DSML_FRAMES: &[&str] = &[
    "{\"choices\":[{\"delta\":{\"content\":\"<｜DSML｜invoke name=\\\"read\\\">\"}}]}",
    "{\"choices\":[{\"delta\":{\"content\":\"<｜DSML｜parameter name=\\\"path\\\" string=\\\"true\\\">a\"}}]}",
    "{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}",
];

#[test]
fn a_single_lane_redials_once_after_malformed_dsml_markup() {
    block_on(async {
        let harness = Harness::new();
        harness
            .configure(json!({"rules": [null], "same_redial": 2}))
            .await;
        let lane = spawn_rung(vec![
            Answer::Stream(BROKEN_DSML_FRAMES),
            Answer::Stream(VALID_CALL_FRAMES),
        ])
        .await;
        let route = [wire("vllm", &lane.url, 0)];
        let (won, guard) = harness.run(&route, None, Duration::from_secs(30)).await;
        let Won::Committed(mut committed) = won else {
            panic!("the redial's valid call serves the request");
        };
        assert_eq!(committed.depth, 0);
        let events = std::mem::take(&mut committed.prefix);
        assert_eq!(completed_call(&events), "{\"path\": \"a\"}");
        finish(guard, Won::Committed(committed)).await;
        assert_eq!(lane.accepted.lock().expect("lock").len(), 2);
        let story = harness.story().await;
        let starts = story["starts"].as_array().expect("starts");
        assert_eq!(starts.len(), 2);
        assert_eq!(starts[1]["failure"]["malformed_tool_call"], true);
        assert_eq!(starts[1]["failure"]["retryable_same_deployment"], true);
    });
}
