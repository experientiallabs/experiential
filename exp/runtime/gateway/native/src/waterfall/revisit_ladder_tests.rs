//! A failover revisits a rung this request skipped: after the rung that was
//! dialed fails, the data plane asks the control plane for a successor even
//! when the only untried rung sits EARLIER in the route.

use super::ladder_tests::{block_on, Harness};
use super::*;

/// A loopback URL nothing listens on, so every dial is refused at connect.
async fn refused_url() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let address = listener.local_addr().expect("address");
    drop(listener);
    format!("http://{address}/v1/chat/completions")
}

#[test]
fn a_failure_on_a_later_rung_asks_for_the_skipped_earlier_one() {
    block_on(async {
        let harness = Harness::new();
        // The control plane skipped depth 0 at admission (full) and dialed
        // depth 1, whose box refuses connections.
        harness
            .configure(serde_json::json!({"first_depth": 1}))
            .await;
        let route = [
            super::ladder_tests::wire("a", &refused_url().await, 0),
            super::ladder_tests::wire("b", &refused_url().await, 0),
        ];
        let policy = RoutePolicy {
            maximum_total_attempts: 4,
            maximum_same_deployment_attempts: 1,
            refusal_failover: false,
            throttle_redial: None,
            physical_route_cap: None,
            backoff: None,
        };
        let (won, _guard) = harness
            .run_with_policy(
                "key",
                None,
                &route,
                policy,
                std::time::Duration::from_secs(30),
            )
            .await;
        assert!(matches!(won, Won::Failed(_)));
        let story = harness.story().await;
        let starts = story["starts"].as_array().expect("starts");
        // The refused depth 1 asked for a successor: depth 0 is untried.
        assert_eq!(starts.len(), 2, "{starts:?}");
        assert_eq!(starts[1]["current_depth"], 1);
        assert_eq!(starts[1]["failure"]["failover_eligible"], true);
        // The dial that failed settled non-final, leaving the ladder open.
        let settles = story["settles"].as_array().expect("settles");
        assert_eq!(settles[0]["finalize"], false);
    });
}
