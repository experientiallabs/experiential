//! A buffered surface restarts a committed attempt whose connection breaks
//! mid-body: nothing reached the caller, so the ladder takes the failure and
//! the answer comes from the next rung. A streaming surface keeps the
//! attempt the caller already saw.

use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::io::AsyncWriteExt;

use super::ladder_tests::{
    block_on, finish, read_request, spawn_rung, wire, Answer, Harness, Rung, TEXT_FRAME,
};
use super::*;
use crate::relay::collect_committed;

/// One rung whose 200 event stream promises more body than it sends: one
/// content frame, then the connection closes mid-body (a provider reset after
/// the stream began).
async fn spawn_broken_rung() -> Rung {
    spawn_broken_rung_after(Duration::ZERO).await
}

/// [`spawn_broken_rung`], holding the connection open for `hold` after the
/// content frame before it breaks.
async fn spawn_broken_rung_after(hold: Duration) -> Rung {
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
        let body = format!("data: {TEXT_FRAME}\n\n");
        let head = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
             content-length: {}\r\nconnection: close\r\n\r\n",
            body.len() + 4_096,
        );
        socket
            .write_all(format!("{head}{body}").as_bytes())
            .await
            .expect("write");
        tokio::time::sleep(hold).await;
        let _ = socket.shutdown().await;
    });
    Rung {
        url: format!("http://{address}/v1/chat/completions"),
        accepted,
        bodies: Arc::new(Mutex::new(Vec::new())),
        keys: Arc::new(Mutex::new(Vec::new())),
    }
}

/// Run the waterfall with the per-depth `WaterfallContext::buffered` view a
/// surface that assembles its whole answer before sending hands the loop.
async fn run_buffered(
    harness: &Harness,
    route: &[DeploymentWire],
    buffered: &[bool],
) -> (Won, AttemptGuard) {
    run_buffered_with_budget(harness, route, buffered, None).await
}

/// [`run_buffered`] under a shared pre-first-byte budget ending at `budget`.
async fn run_buffered_with_budget(
    harness: &Harness,
    route: &[DeploymentWire],
    buffered: &[bool],
    budget: Option<Instant>,
) -> (Won, AttemptGuard) {
    let mut guard = AttemptGuard::new(
        harness.bridge.clone(),
        Arc::new(AtomicUsize::new(0)),
        "request-throttle".to_string(),
        Instant::now(),
    );
    let context = WaterfallContext {
        pre_first_byte_deadline: budget,
        bridge: &harness.bridge,
        http: &harness.http,
        request_id: "request-throttle",
        raw_key: "key",
        caller_scope: None,
        route,
        policy: POLICY,
        deadline: Instant::now() + Duration::from_secs(30),
        time_to_first_byte: Duration::from_secs(5),
        time_to_first_byte_slope_seconds_per_million_input_tokens: 0.0,
        time_to_first_token: Duration::from_secs(120),
        approximate_input_tokens: 10.0,
        chat_logprobs: false,
        capture_logprobs: false,
        output_less_retention: None,
        output_token_cap: None,
        tool_search: None,
        output_guardrails: None,
        buffered,
    };
    let won = acquire_attempt(&context, &mut guard).await;
    (won, guard)
}

const POLICY: RoutePolicy = RoutePolicy {
    maximum_total_attempts: 8,
    maximum_same_deployment_attempts: 2,
    refusal_failover: false,
    throttle_redial: None,
    physical_route_cap: None,
    backoff: None,
};

#[test]
fn a_buffered_answer_broken_mid_body_restarts_on_the_next_rung() {
    block_on(async {
        let harness = Harness::new();
        let rung_a = spawn_broken_rung().await;
        let rung_b = spawn_rung(vec![Answer::Stream(&[TEXT_FRAME])]).await;
        let route = [wire("a", &rung_a.url, 0), wire("b", &rung_b.url, 0)];
        let (won, guard) = run_buffered(&harness, &route, &[true, true]).await;
        let Won::Committed(committed) = &won else {
            panic!("the sibling serves the buffered answer");
        };
        assert_eq!(committed.depth, 1);
        // The whole answer is already drained, terminal last.
        assert!(committed.prefix.last().is_some_and(Event::is_terminal));
        assert!(committed
            .prefix
            .iter()
            .any(|event| matches!(event, Event::TextDelta(text) if text == "hi")));
        assert!(committed.collection_failure.is_none());
        finish(guard, won).await;
        let story = harness.story().await;
        let starts = story["starts"].as_array().expect("starts");
        assert_eq!(starts.len(), 2, "{starts:?}");
        assert_eq!(starts[1]["current_depth"], 0);
        assert_eq!(starts[1]["failure"]["failure_class"], "transport");
        // One attempt row per dial: the broken one settled failed and the
        // ladder stayed open for the sibling.
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles[0]["outcome"], "failed");
        assert_eq!(settles[0]["finalize"], false);
        assert_eq!(settles[0]["failure"]["failure_class"], "transport");
        assert_eq!(settles[1]["outcome"], "completed");
        assert_eq!(rung_a.accepted.lock().expect("lock").len(), 1);
        assert_eq!(rung_b.accepted.lock().expect("lock").len(), 1);
    });
}

#[test]
fn a_restart_after_commit_is_not_bounded_by_the_spent_first_byte_budget() {
    block_on(async {
        let harness = Harness::new();
        // The first byte lands inside the budget; the break comes after it
        // expired, which a pre-commit failure would read as exhaustion.
        let rung_a = spawn_broken_rung_after(Duration::from_millis(600)).await;
        let rung_b = spawn_rung(vec![Answer::Stream(&[TEXT_FRAME])]).await;
        let route = [wire("a", &rung_a.url, 0), wire("b", &rung_b.url, 0)];
        let budget = Some(Instant::now() + Duration::from_millis(300));
        let (won, guard) = run_buffered_with_budget(&harness, &route, &[true, true], budget).await;
        let Won::Committed(committed) = &won else {
            panic!("the sibling serves after the budget expired");
        };
        assert_eq!(committed.depth, 1);
        finish(guard, won).await;
        assert_eq!(rung_b.accepted.lock().expect("lock").len(), 1);
    });
}

#[test]
fn a_buffered_answer_broken_on_every_rung_fails_with_the_transport_error() {
    block_on(async {
        let harness = Harness::new();
        let rung_a = spawn_broken_rung().await;
        let rung_b = spawn_broken_rung().await;
        let route = [wire("a", &rung_a.url, 0), wire("b", &rung_b.url, 0)];
        let (won, _guard) = run_buffered(&harness, &route, &[true, true]).await;
        let Won::Failed(error) = won else {
            panic!("an exhausted ladder answers with its failure");
        };
        // The caller's answer is the transport failure itself, the public
        // error the surface gave before the drain moved into the loop.
        assert_eq!(
            error.message,
            Failure::new(
                FailureClass::Transport,
                "provider transport failed; retry the request"
            )
            .public_error()
            .message
        );
        let story = harness.story().await;
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles.len(), 2, "{settles:?}");
        assert_eq!(settles[0]["finalize"], false);
        for settle in settles {
            assert_eq!(settle["failure"]["failure_class"], "transport");
        }
    });
}

#[test]
fn a_streaming_surface_keeps_the_attempt_the_caller_already_saw() {
    block_on(async {
        let harness = Harness::new();
        let rung_a = spawn_broken_rung().await;
        let rung_b = spawn_rung(vec![Answer::Stream(&[TEXT_FRAME])]).await;
        let route = [wire("a", &rung_a.url, 0), wire("b", &rung_b.url, 0)];
        let (won, mut guard) = run_buffered(&harness, &route, &[false, false]).await;
        let Won::Committed(mut committed) = won else {
            panic!("the first delta commits the live stream");
        };
        assert_eq!(committed.depth, 0);
        // The surface drains the live relay and meets the break itself; it
        // then ends the stream with the protocol's error event.
        let failure = collect_committed(
            &mut committed,
            Instant::now() + Duration::from_secs(10),
            Duration::from_secs(10),
            Instant::now(),
        )
        .await
        .expect_err("the break reaches the surface");
        assert_eq!(failure.failure_class, FailureClass::Transport);
        guard
            .settle("failed", None, &[], Some(&failure), true)
            .await;
        let story = harness.story().await;
        assert_eq!(story["starts"].as_array().expect("starts").len(), 1);
        assert!(rung_b.accepted.lock().expect("lock").is_empty());
    });
}

#[test]
fn restartable_names_upstream_faults_and_never_gateway_or_caller_ones() {
    let transport =
        Failure::new(FailureClass::Transport, "provider transport failed").with_retry(true, true);
    assert!(buffered::restartable(&transport));
    let stalled = Failure::new(FailureClass::Timeout, "provider stalled").with_retry(false, true);
    assert!(buffered::restartable(&stalled));
    // Unflagged: the classifier said neither redial nor fail over.
    assert!(!buffered::restartable(&Failure::new(
        FailureClass::Transport,
        "provider transport failed",
    )));
    for class in [
        FailureClass::Internal,
        FailureClass::Unavailable,
        FailureClass::InvalidRequest,
        FailureClass::Refusal,
        FailureClass::Throttled,
    ] {
        let failure = Failure::new(class, "classified").with_retry(true, true);
        assert!(!buffered::restartable(&failure), "{class:?}");
    }
}

/// A vLLM lane's verdict when it serves fewer tokens than the model
/// advertises.
const CONTEXT_WINDOW_BODY: &str = concat!(
    "{\"object\":\"error\",\"message\":\"This model's maximum context length is 16384 ",
    "tokens. However, you requested 8192 output tokens and your prompt contains at least ",
    "8193 input tokens.\",\"type\":\"BadRequestError\",\"param\":null,\"code\":400}"
);

#[test]
fn a_context_window_rejection_advances_once_to_the_next_rung() {
    block_on(async {
        let harness = Harness::new();
        let rung_a = spawn_rung(vec![Answer::Rejected(CONTEXT_WINDOW_BODY)]).await;
        let rung_b = spawn_rung(vec![Answer::Stream(&[TEXT_FRAME])]).await;
        let route = [wire("a", &rung_a.url, 0), wire("b", &rung_b.url, 0)];
        let (won, guard) = harness
            .run_with_policy("key", None, &route, POLICY, Duration::from_secs(30))
            .await;
        let Won::Committed(committed) = &won else {
            panic!("the wider rung serves");
        };
        assert_eq!(committed.depth, 1);
        finish(guard, won).await;
        let story = harness.story().await;
        let failure = &story["starts"][1]["failure"];
        assert_eq!(failure["failure_class"], "invalid_request");
        assert_eq!(failure["failover_eligible"], true);
        assert_eq!(failure["retryable_same_deployment"], false);
        assert_eq!(failure["context_window_rejected"], true);
        assert_eq!(failure["context_window_limit_tokens"], 16_384);
    });
}

#[test]
fn a_second_context_window_rejection_is_the_requests_answer() {
    block_on(async {
        let harness = Harness::new();
        let rung_a = spawn_rung(vec![Answer::Rejected(CONTEXT_WINDOW_BODY)]).await;
        let rung_b = spawn_rung(vec![Answer::Rejected(CONTEXT_WINDOW_BODY)]).await;
        let route = [wire("a", &rung_a.url, 0), wire("b", &rung_b.url, 0)];
        let (won, _guard) = harness
            .run_with_policy("key", None, &route, POLICY, Duration::from_secs(30))
            .await;
        let Won::Failed(error) = won else {
            panic!("the second verdict ends the request");
        };
        // The caller still reads the provider's own 400.
        assert_eq!(error.status_code, 400);
        assert!(error.message.contains("16384"), "{}", error.message);
        let story = harness.story().await;
        let starts = story["starts"].as_array().expect("starts");
        // First dial, the one advance, and nothing after the second verdict:
        // the data plane settled it final without asking for a successor.
        assert_eq!(starts.len(), 2, "{starts:?}");
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles.last().expect("settle")["finalize"], true);
    });
}

#[test]
fn work_the_provider_already_ran_is_never_restarted() {
    // Client tool calls are text the caller executes: a restart repeats nothing.
    assert!(!buffered::provider_executed(&Event::TextDelta("hi".into())));
    let client_call = Event::ToolArgumentsDelta {
        index: 0,
        delta: "{}".into(),
    };
    assert!(!buffered::provider_executed(&client_call));
    // A hosted MCP call or a server tool may already have had effects.
    let hosted = Event::HostedToolItemStarted {
        output_index: 0,
        item_id: "mcp_1".into(),
        item_type: "mcp_call".into(),
        item: "{}".into(),
    };
    assert!(buffered::provider_executed(&hosted));
    let server = Event::ServerToolUseStarted {
        index: 0,
        call_id: "srvtoolu_1".into(),
        name: "web_fetch".into(),
    };
    assert!(buffered::provider_executed(&server));
}
