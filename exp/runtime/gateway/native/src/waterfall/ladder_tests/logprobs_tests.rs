//! Actual waterfall regression tests for probability commitment and refusal isolation.
use super::*;

const CONTENT: &str = r#"{"choices":[{"index":0,"delta":{},"logprobs":{"content":[{"token":"a","logprob":0,"bytes":[97],"top_logprobs":[]}]}}]}"#;
const EMPTY: &str =
    r#"{"choices":[{"index":0,"delta":{},"logprobs":{"content":[],"refusal":null}}]}"#;
const REFUSAL: &str = r#"{"choices":[{"index":0,"delta":{"refusal":"no"},"logprobs":{"refusal":[{"token":"no","logprob":-1,"bytes":null,"top_logprobs":[]}]}}]}"#;
const FINISH: &str = r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;

async fn run_probabilities(harness: &Harness, route: &[DeploymentWire]) -> (Won, AttemptGuard) {
    harness.configure(json!({"refusal_failover": true})).await;
    let mut guard = AttemptGuard::new(
        harness.bridge.clone(),
        Arc::new(AtomicUsize::new(0)),
        "probability-test".to_string(),
        Instant::now(),
    );
    let context = WaterfallContext {
        bridge: &harness.bridge,
        http: &harness.http,
        request_id: "probability-test",
        raw_key: "key",
        caller_scope: None,
        route,
        policy: RoutePolicy {
            maximum_total_attempts: 4,
            maximum_same_deployment_attempts: 1,
            refusal_failover: true,
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
        output_less_retention: None,
        output_token_cap: None,
        tool_search: None,
    };
    (acquire_attempt(&context, &mut guard).await, guard)
}

fn probability_wire(id: &str, url: &str) -> DeploymentWire {
    let mut wire = wire(id, url, 0);
    wire.upstream_payload["logprobs"] = json!(true);
    wire
}

#[test]
fn content_probabilities_commit_before_text_and_prevent_failover() {
    block_on(async {
        let harness = Harness::new();
        let first = spawn_rung(vec![Answer::Stream(&[CONTENT, THROTTLE_FRAME])]).await;
        let next = spawn_rung(vec![Answer::Stream(&[CONTENT, FINISH])]).await;
        let route = [
            probability_wire("a", &first.url),
            probability_wire("b", &next.url),
        ];
        let (won, guard) = run_probabilities(&harness, &route).await;
        let Won::Committed(mut committed) = finish(guard, won).await else {
            panic!("must commit");
        };
        assert_eq!(committed.depth, 0);
        assert!(committed.relay.first_token_at().is_none());
        assert!(matches!(committed.prefix[0], Event::ChoiceLogprobsDelta(_)));
        let terminal = committed
            .relay
            .next_event(
                Instant::now() + Duration::from_secs(2),
                Duration::from_secs(2),
                Instant::now(),
            )
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(terminal, Event::Failed(_)));
        assert!(next.accepted.lock().unwrap().is_empty());
    });
}

#[test]
fn refusal_probabilities_and_empty_metadata_do_not_escape_abandoned_attempts() {
    for frames in [&[REFUSAL, FINISH][..], &[EMPTY, THROTTLE_FRAME][..]] {
        block_on(async {
            let harness = Harness::new();
            let first = spawn_rung(vec![Answer::Stream(frames)]).await;
            let next = spawn_rung(vec![Answer::Stream(&[CONTENT, FINISH])]).await;
            let route = [
                probability_wire("a", &first.url),
                probability_wire("b", &next.url),
            ];
            let (won, guard) = run_probabilities(&harness, &route).await;
            let Won::Committed(committed) = finish(guard, won).await else {
                panic!("fallback must commit");
            };
            assert_eq!(committed.depth, 1);
            assert!(!committed.visible_refusal);
            assert_eq!(committed.prefix.len(), 1);
            assert!(
                matches!(&committed.prefix[0], Event::ChoiceLogprobsDelta(delta)
                if delta.logprobs.as_ref().unwrap().content.as_ref().unwrap()[0].token == "a")
            );
        });
    }
}

#[test]
fn empty_probability_prefix_is_bounded_without_committing() {
    const MANY_EMPTY: [&str; 257] = [EMPTY; 257];
    block_on(async {
        let harness = Harness::new();
        let first = spawn_rung(vec![Answer::Stream(&MANY_EMPTY)]).await;
        let next = spawn_rung(vec![Answer::Stream(&[EMPTY, CONTENT, FINISH])]).await;
        let route = [
            probability_wire("a", &first.url),
            probability_wire("b", &next.url),
        ];
        let (won, guard) = run_probabilities(&harness, &route).await;
        let Won::Committed(committed) = finish(guard, won).await else {
            panic!("fallback must commit");
        };
        assert_eq!(committed.depth, 1);
        assert_eq!(committed.prefix.len(), 2);
        assert!(
            matches!(&committed.prefix[0], Event::ChoiceLogprobsDelta(delta)
            if delta.logprobs.as_ref().unwrap().content.as_ref().unwrap().is_empty())
        );
        assert!(committed.relay.first_token_at().is_none());
    });
}
