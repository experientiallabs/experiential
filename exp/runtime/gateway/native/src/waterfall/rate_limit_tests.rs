//! Caller quota headers follow only the attempt whose empty answer is served.

use super::ladder_tests::{block_on, read_request_body, wire, Harness};
use super::*;
use tokio::io::AsyncWriteExt;

async fn empty_rung(remaining: &'static str) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        read_request_body(&mut socket).await;
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":9}}\n\n",
            "data: [DONE]\n\n",
        );
        socket
            .write_all(
                format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                     x-ratelimit-remaining-tokens: {remaining}\r\n\
                     retry-after: 9\r\nx-account-id: hidden\r\n\
                     content-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len(),
                )
                .as_bytes(),
            )
            .await
            .unwrap();
    });
    format!("http://{address}/v1/chat/completions")
}

#[test]
fn empty_completion_preserves_only_the_served_customer_quota_headers() {
    block_on(async {
        for customer_managed in [true, false] {
            let harness = Harness::new();
            let first_url = empty_rung("111").await;
            let last_url = empty_rung("222").await;
            let mut first = wire("first", &first_url, 0);
            first.billing_customer_managed = true;
            let mut last = wire("last", &last_url, 0);
            last.billing_customer_managed = customer_managed;
            let (won, _guard) = harness
                .run_with_policy(
                    "key",
                    Some("org:key-holder"),
                    &[first, last],
                    RoutePolicy {
                        maximum_total_attempts: 2,
                        maximum_same_deployment_attempts: 1,
                        refusal_failover: false,
                        throttle_redial: None,
                        physical_route_cap: None,
                        backoff: None,
                    },
                    Duration::from_secs(2),
                )
                .await;
            let Won::Settled(settled) = won else {
                panic!("the exhausted empty completion must be served")
            };
            assert_eq!(settled.depth, 1);
            assert!(settled.empty_completion);
            let expected = if customer_managed {
                vec![(
                    "x-ratelimit-remaining-tokens".to_string(),
                    "222".to_string(),
                )]
            } else {
                Vec::new()
            };
            assert_eq!(settled.rate_limit_headers, expected);
            assert_eq!(harness.story().await["counts"], serde_json::json!([1, 1]));
        }
    });
}
