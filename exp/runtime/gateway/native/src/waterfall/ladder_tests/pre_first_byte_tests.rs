//! The pre-first-byte budget: a chain of slow dials ends inside the budget
//! with the gateway's own retryable timeout, never past the edge's origin limit.
use super::*;

/// A rung that accepts every connection and never answers (a stalled lane).
async fn spawn_stalled_rung() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    tokio::spawn(async move {
        let mut held = Vec::new();
        while let Ok((socket, _)) = listener.accept().await {
            held.push(socket);
        }
    });
    format!("http://{address}/v1/chat/completions")
}

async fn run_budgeted(
    harness: &Harness,
    route: &[DeploymentWire],
    budget: Option<Duration>,
) -> (Won, AttemptGuard) {
    let mut guard = AttemptGuard::new(
        harness.bridge.clone(),
        Arc::new(AtomicUsize::new(0)),
        "request-budget".to_string(),
        Instant::now(),
    );
    let started = Instant::now();
    let context = WaterfallContext {
        pre_first_byte_deadline: budget.map(|budget| started + budget),
        bridge: &harness.bridge,
        http: &harness.http,
        request_id: "request-budget",
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
        deadline: started + Duration::from_secs(30),
        time_to_first_byte: Duration::from_secs(2),
        time_to_first_byte_slope_seconds_per_million_input_tokens: 0.0,
        time_to_first_token: Duration::from_secs(120),
        approximate_input_tokens: 1.0,
        chat_logprobs: false,
        capture_logprobs: false,
        output_less_retention: None,
        output_token_cap: None,
        tool_search: None,
        output_guardrails: None,
    };
    (acquire_attempt(&context, &mut guard).await, guard)
}

#[test]
fn stalled_dials_stop_at_the_budget_with_a_retryable_timeout() {
    block_on(async {
        let harness = Harness::new();
        let route = [
            wire("a", &spawn_stalled_rung().await, 0),
            wire("b", &spawn_stalled_rung().await, 0),
            wire("c", &spawn_stalled_rung().await, 0),
        ];
        let started = Instant::now();
        let (won, guard) = run_budgeted(&harness, &route, Some(Duration::from_secs(3))).await;
        let elapsed = started.elapsed();
        let Won::Failed(error) = finish(guard, won).await else {
            panic!("a fully stalled chain must fail");
        };
        // Two 2 s header windows would take 4 s and a third 6 s; the budget
        // ends the chain at 3 s instead, as our own retryable 503.
        assert!(elapsed < Duration::from_millis(3_900), "{elapsed:?}");
        assert!(elapsed >= Duration::from_millis(2_900), "{elapsed:?}");
        assert_eq!(error.status_code, 503);
        assert_eq!(error.code, "deadline_exceeded");
        assert_eq!(error.retry_after_seconds, Some(2));
    });
}

#[test]
fn without_a_budget_each_stalled_dial_gets_its_own_window() {
    block_on(async {
        let harness = Harness::new();
        let route = [
            wire("a", &spawn_stalled_rung().await, 0),
            wire("b", &spawn_stalled_rung().await, 0),
        ];
        let started = Instant::now();
        let (won, guard) = run_budgeted(&harness, &route, None).await;
        let elapsed = started.elapsed();
        assert!(matches!(finish(guard, won).await, Won::Failed(_)));
        assert!(elapsed >= Duration::from_millis(3_900), "{elapsed:?}");
    });
}

/// A rung that holds its response headers for `delay`, then streams one frame.
async fn spawn_slow_header_rung(delay: Duration) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let _ = read_request(&mut socket).await;
        tokio::time::sleep(delay).await;
        let answer = render(&Answer::Stream(&[TEXT_FRAME]));
        socket.write_all(answer.as_bytes()).await.expect("write");
        let _ = socket.shutdown().await;
    });
    format!("http://{address}/v1/chat/completions")
}

#[test]
fn an_authored_header_allowance_is_not_cut_at_the_configured_budget() {
    block_on(async {
        // Scaled down from the GLM lane: the lead authors a header allowance
        // (3 s) longer than the configured budget (1 s) and answers its
        // headers at 2 s. The budget is raised to the authored allowance
        // plus the reserve, so the lead serves instead of timing out.
        let harness = Harness::new();
        let mut lead = wire(
            "lead",
            &spawn_slow_header_rung(Duration::from_secs(2)).await,
            0,
        );
        lead.time_to_first_byte_base_seconds = Some(3.0);
        let route = [lead, wire("backup", &spawn_stalled_rung().await, 0)];
        let budget = super::super::pre_first_byte_budget(
            Some(Duration::from_secs(1)),
            super::super::wire::PRE_FIRST_BYTE_RESERVE,
            None,
            &route,
            Duration::from_secs(2),
            0.0,
            1.0,
        );
        let (won, guard) = run_budgeted(&harness, &route, budget).await;
        let Won::Committed(committed) = finish(guard, won).await else {
            panic!("the lead's authored allowance must outlast the configured budget");
        };
        assert_eq!(committed.depth, 0);
    });
}

#[test]
fn only_a_lane_that_spent_its_own_allowance_takes_the_header_timeout_strike() {
    block_on(async {
        // [A hung, B healthy but slower than what the budget leaves]: A
        // stalls out its own 1 s allowance; B is cut by the shared 1.5 s
        // budget half a second into its 5 s allowance.
        let harness = Harness::new();
        let mut hung = wire("a", &spawn_stalled_rung().await, 0);
        hung.time_to_first_byte_base_seconds = Some(1.0);
        let mut cut = wire("b", &spawn_stalled_rung().await, 0);
        cut.time_to_first_byte_base_seconds = Some(5.0);
        let route = [hung, cut];
        let (won, guard) = run_budgeted(&harness, &route, Some(Duration::from_millis(1_500))).await;
        assert!(matches!(finish(guard, won).await, Won::Failed(_)));
        let story = harness.story().await;
        let settles = story["settles"].as_array().expect("settles");
        let phases: Vec<&Value> = settles
            .iter()
            .map(|settle| &settle["failure"]["header_timeout"])
            .collect();
        assert_eq!(
            phases,
            [&json!("allowance"), &json!("budget")],
            "{settles:?}"
        );
    });
}
