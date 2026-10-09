//! Host-granted renewal of a reused cache: the PATCH, its report, and its bounds.

use super::tests::{answer, execute_test, ready, server, timestamp, wire, Host};
use super::*;

fn ready_with_renewal(name: &str, until: &(f64, String)) -> Value {
    let mut ready = ready(name);
    ready["renew"] =
        json!({"operation_id": "cache-op", "expires_at": until.0, "expire_time": until.1});
    ready
}

#[tokio::test]
async fn a_granted_renewal_patches_expire_time_and_reports_the_provider_answer() {
    let until = timestamp(epoch_now().floor() as u64 + 900);
    let (base, task) = server(vec![answer(
        200,
        &json!({"name": "cachedContents/reused", "expireTime": until.1}).to_string(),
    )])
    .await;
    let original = wire(&format!(
        "{base}/v1beta/models/gemini-test:streamGenerateContent?alt=sse"
    ));
    let host = Host::new(vec![
        ready_with_renewal("cachedContents/reused", &until),
        json!({"state": "recorded"}),
    ]);
    let overlaid = execute_test(&original, &host, EndpointPolicy::Loopback)
        .await
        .unwrap()
        .unwrap();
    // The generation reuses the resource exactly as an unrenewed hit does.
    assert_eq!(
        overlaid.upstream_payload["cachedContent"],
        "cachedContents/reused"
    );
    assert_eq!(
        host.calls()[1],
        (
            "renew_explicit_cache",
            json!({"request_id": "request", "deployment_id": "deployment",
                   "operation_id": "cache-op", "outcome": "renewed", "expire_time": until.1})
        )
    );
    let requests = task.await.unwrap();
    let head = requests[0].split("\r\n\r\n").next().unwrap().to_lowercase();
    assert!(head.starts_with("patch /v1beta/cachedcontents/reused?updatemask=expiretime http/1.1"));
    assert!(head.contains("x-goog-api-key: test-secret"));
    assert!(!head.contains("idempotency-key"));
    assert!(requests[0].ends_with(&json!({"expireTime": until.1}).to_string()));
}

#[tokio::test]
async fn a_refused_or_unsafe_renewal_is_reported_and_never_blocks_the_hit() {
    // A 4xx proves no extension happened: the host releases the storage hold.
    let until = timestamp(epoch_now().floor() as u64 + 900);
    let (base, task) = server(vec![answer(404, "{}")]).await;
    let original = wire(&format!(
        "{base}/v1beta/models/gemini-test:streamGenerateContent?alt=sse"
    ));
    let host = Host::new(vec![
        ready_with_renewal("cachedContents/reused", &until),
        json!({"state": "recorded"}),
    ]);
    let overlaid = execute_test(&original, &host, EndpointPolicy::Loopback)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        overlaid.upstream_payload["cachedContent"],
        "cachedContents/reused"
    );
    assert_eq!(host.calls()[1].1["outcome"], "failed");
    assert_eq!(task.await.unwrap().len(), 1);

    // A grant beyond the renewal ceiling sends nothing and still serves the hit,
    // even when the host fails to acknowledge the report.
    let far = timestamp(epoch_now().floor() as u64 + 7200);
    let host = Host::new(vec![ready_with_renewal("cachedContents/reused", &far)]);
    host.push_error();
    let overlaid = execute_test(&original, &host, EndpointPolicy::Loopback)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        overlaid.upstream_payload["cachedContent"],
        "cachedContents/reused"
    );
    assert_eq!(host.calls()[1].1["outcome"], "failed");
    assert!(host.calls()[1].1.get("expire_time").is_none());
}

#[tokio::test]
async fn an_answer_that_does_not_move_the_expiry_later_is_unknown() {
    // A stale expiry inside the lease never proves an extension happened, so
    // the host keeps its storage hold instead of settling it as renewed.
    let until = timestamp(epoch_now().floor() as u64 + 900);
    let current = super::tests::expiry();
    let (base, task) = server(vec![answer(
        200,
        &json!({"name": "cachedContents/reused", "expireTime": current.1}).to_string(),
    )])
    .await;
    let original = wire(&format!(
        "{base}/v1beta/models/gemini-test:streamGenerateContent?alt=sse"
    ));
    let host = Host::new(vec![
        ready_with_renewal("cachedContents/reused", &until),
        json!({"state": "recorded"}),
    ]);
    let overlaid = execute_test(&original, &host, EndpointPolicy::Loopback)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        overlaid.upstream_payload["cachedContent"],
        "cachedContents/reused"
    );
    assert_eq!(host.calls()[1].1["outcome"], "unknown");
    assert_eq!(task.await.unwrap().len(), 1);
}

#[tokio::test]
async fn a_lease_on_a_resource_that_went_stale_is_reported_failed_without_a_patch() {
    // No server: a PATCH would fail the test by finding nothing to connect to.
    let until = timestamp(epoch_now().floor() as u64 + 900);
    let original =
        wire("http://127.0.0.1:9/v1beta/models/gemini-test:streamGenerateContent?alt=sse");
    let mut stale = ready_with_renewal("cachedContents/reused", &until);
    stale["expires_at"] = json!(epoch_now() + 2.0);
    let host = Host::new(vec![stale, json!({"state": "recorded"})]);
    let result = execute_test(&original, &host, EndpointPolicy::Loopback)
        .await
        .unwrap();
    assert!(result.is_none(), "a stale resource is never reused");
    assert_eq!(host.calls()[1].0, "renew_explicit_cache");
    assert_eq!(host.calls()[1].1["outcome"], "failed");
}
