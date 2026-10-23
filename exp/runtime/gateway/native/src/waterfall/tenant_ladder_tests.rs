//! Ladder tests for a tenant origin, on the same scripted control plane and
//! local rungs as `ladder_tests`: whatever an upstream does (a rejected
//! credential, a client-error sentence, a throttle, a refused connection), the
//! caller reads one opaque 502 while settlement keeps the precise class.

use std::time::Duration;

use super::ladder_tests::{block_on, finish, spawn_rung, wire, Answer, Harness};
use super::*;
use crate::errors::TENANT_ORIGIN_MESSAGE;

const FORBIDDEN_BODY: &str = r#"{"error":{"message":"Invalid API key","type":"auth"}}"#;
const REJECTED_BODY: &str =
    r#"{"error":{"message":"redis_version:7.2.4 role:master","type":"invalid_request_error"}}"#;

fn tenant(url: &str) -> DeploymentWire {
    DeploymentWire {
        tenant_origin: true,
        ..wire("tenant", url, 0)
    }
}

/// Run one single-rung route, dialed once, to its exhausted answer and the
/// settled class.
async fn exhausted(route: DeploymentWire) -> (PublicError, String) {
    let harness = Harness::new();
    let policy = RoutePolicy {
        maximum_total_attempts: 1,
        maximum_same_deployment_attempts: 1,
        refusal_failover: false,
        throttle_redial: None,
        physical_route_cap: None,
        backoff: None,
    };
    let (won, guard) = harness
        .run_with_policy(
            "key",
            Some("org:key-holder"),
            &[route],
            policy,
            Duration::from_secs(30),
        )
        .await;
    let Won::Failed(error) = finish(guard, won).await else {
        panic!("a single failing rung exhausts the ladder");
    };
    let story = harness.story().await;
    let settles = story["settles"].as_array().expect("settles");
    let class = settles.last().expect("one settle")["failure"]["failure_class"]
        .as_str()
        .expect("settled class")
        .to_string();
    (error, class)
}

fn assert_opaque(error: &PublicError) {
    assert_eq!(error.status_code, 502);
    assert_eq!(error.code, "custom_endpoint_failed");
    assert_eq!(error.message, TENANT_ORIGIN_MESSAGE);
    assert_eq!(error.param, None);
    assert_eq!(error.retry_after_seconds, None);
}

#[test]
fn every_upstream_answer_on_a_tenant_origin_reads_the_same_to_the_caller() {
    block_on(async {
        for (answer, settled) in [
            (Answer::Forbidden(FORBIDDEN_BODY), "provider_authentication"),
            (Answer::Rejected(REJECTED_BODY), "invalid_request"),
            (Answer::Throttle(Some(7)), "throttled"),
        ] {
            let rung = spawn_rung(vec![answer.clone()]).await;
            let (error, class) = exhausted(tenant(&rung.url)).await;
            assert_opaque(&error);
            assert!(!error.message.contains("redis"));
            // The ledger still names what actually happened.
            assert_eq!(class, settled);

            // The same answer on an ordinary rung keeps its distinct shape.
            let rung = spawn_rung(vec![answer]).await;
            let (error, _) = exhausted(wire("house", &rung.url, 0)).await;
            assert_ne!(error.message, TENANT_ORIGIN_MESSAGE);
        }
    });
}

#[test]
fn a_refused_connection_and_a_garbage_body_are_indistinguishable_on_a_tenant_origin() {
    block_on(async {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let closed = format!(
            "http://{}/v1/chat/completions",
            listener.local_addr().expect("address")
        );
        drop(listener);
        let (refused, class) = exhausted(tenant(&closed)).await;
        assert_opaque(&refused);
        assert_eq!(class, "transport");

        let rung = spawn_rung(vec![Answer::Stream(&["not json"])]).await;
        let (garbage, _) = exhausted(tenant(&rung.url)).await;
        assert_opaque(&garbage);
        assert_eq!(refused.json_body(), garbage.json_body());
    });
}
