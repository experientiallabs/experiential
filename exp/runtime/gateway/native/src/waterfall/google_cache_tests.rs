//! Scripted host callbacks and real loopback HTTP, without provider or database access.

use super::*;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Clone)]
pub(super) struct Host {
    replies: Arc<Mutex<VecDeque<Result<Value, PublicError>>>>,
    calls: Arc<Mutex<Vec<(&'static str, Value)>>>,
}

impl Host {
    pub(super) fn new(replies: Vec<Value>) -> Self {
        Self {
            replies: Arc::new(Mutex::new(replies.into_iter().map(Ok).collect())),
            calls: Default::default(),
        }
    }

    async fn call(&self, method: &'static str, argument: String) -> Result<String, PublicError> {
        self.calls
            .lock()
            .unwrap()
            .push((method, serde_json::from_str(&argument).unwrap()));
        self.replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected callback")
            .map(|value| value.to_string())
    }

    pub(super) fn calls(&self) -> Vec<(&'static str, Value)> {
        self.calls.lock().unwrap().clone()
    }
}

pub(super) fn wire(url: &str) -> DeploymentWire {
    serde_json::from_value(json!({
        "provider": "google", "deployment_id": "deployment", "dialect": "gemini_generate_content",
        "url": url, "headers": {"x-goog-api-key": "test-secret", "Idempotency-Key": "generation-only"},
        "timeout_seconds": 1.0, "idempotency_key": "generation-op", "explicit_cache": true,
        "upstream_payload": {"contents": [{"role": "user", "parts": [{"text": "prefix and suffix"}]}]},
    })).unwrap()
}

fn http() -> reqwest::Client {
    crate::upstream::build_client(Duration::from_secs(1)).unwrap()
}

pub(super) fn expiry() -> (f64, String) {
    let epoch = epoch_now().floor() as u64 + 250;
    // httpdate owns calendar formatting; turn its UTC components into Google's spelling.
    let date = httpdate::fmt_http_date(UNIX_EPOCH + Duration::from_secs(epoch));
    let fields: Vec<&str> = date.split_whitespace().collect();
    let month = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    .iter()
    .position(|month| *month == fields[2])
    .unwrap()
        + 1;
    (
        epoch as f64,
        format!("{}-{:02}-{}T{}Z", fields[3], month, fields[1], fields[4]),
    )
}

pub(super) fn ready(name: &str) -> Value {
    json!({"state": "ready", "resource_name": name, "resource_prefix": "cachedContents/", "expires_at": expiry().0, "payload": {
        "cachedContent": name,
        "contents": [{"role": "user", "parts": [{"text": "suffix"}]}],
        "generationConfig": {"maxOutputTokens": 10},
    }})
}

pub(super) fn claim(url: &str, expires: &(f64, String)) -> Value {
    json!({"state": "create", "operation_id": "cache-op", "url": url, "resource_prefix": "cachedContents/",
        "payload": {"model": "models/gemini-test", "expireTime": expires.1,
            "contents": [{"role": "user", "parts": [{"text": "private prefix"}]}]},
        "expires_at": expires.0,
    })
}

pub(super) fn response(expires: &(f64, String)) -> Value {
    json!({"name": "cachedContents/test-cache", "expireTime": expires.1,
        "usageMetadata": {"totalTokenCount": 4096},
        "contents": "private provider text never forwarded", "other": "ignored"})
}

pub(super) async fn execute_test(
    wire: &DeploymentWire,
    host: &Host,
    policy: EndpointPolicy,
) -> Result<Option<DeploymentWire>, Failure> {
    execute(
        &http(),
        wire,
        "request",
        Instant::now() + Duration::from_secs(5),
        false,
        policy,
        |method, argument| host.call(method, argument),
    )
    .await
}

/// Read the complete request to inspect auth, operation path, and exact dispatched JSON.
async fn request(socket: &mut tokio::net::TcpStream) -> String {
    let mut bytes = Vec::new();
    let mut chunk = [0; 4096];
    loop {
        if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..end]).to_lowercase();
            let length = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .unwrap_or("0")
                .trim()
                .parse::<usize>()
                .unwrap();
            if bytes.len() >= end + 4 + length {
                return String::from_utf8(bytes).unwrap();
            }
        }
        let length = socket.read(&mut chunk).await.unwrap();
        assert_ne!(length, 0);
        bytes.extend_from_slice(&chunk[..length]);
    }
}

/// Serve an exact sequence; any implicit retry or redirect would consume another response.
pub(super) async fn server(
    responses: Vec<String>,
) -> (String, tokio::task::JoinHandle<Vec<String>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        let mut requests = Vec::new();
        for response in responses {
            let (mut socket, _) = listener.accept().await.unwrap();
            requests.push(request(&mut socket).await);
            socket.write_all(response.as_bytes()).await.unwrap();
            let _ = socket.shutdown().await;
        }
        requests
    });
    (format!("http://{address}"), task)
}

pub(super) fn answer(status: u16, body: &str) -> String {
    format!("HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
}

#[tokio::test]
async fn default_off_and_ineligible_paths_make_no_callback_or_http() {
    let host = Host::new(vec![]);
    let mut original = wire("https://generativelanguage.googleapis.com/v1beta/models/gemini-test:streamGenerateContent?alt=sse");
    original.explicit_cache = false;
    assert!(execute_test(&original, &host, EndpointPolicy::Official)
        .await
        .unwrap()
        .is_none());
    let minimal: DeploymentWire = serde_json::from_value(json!({
        "provider":"google", "deployment_id":"deployment", "dialect":"gemini_generate_content",
        "url":"invalid", "headers":{}, "timeout_seconds":1.0, "idempotency_key":"op"
    }))
    .unwrap();
    assert!(!minimal.explicit_cache);
    original.explicit_cache = true;
    original.upstream_body = Some("signed".into());
    assert!(execute_test(&original, &host, EndpointPolicy::Official)
        .await
        .unwrap()
        .is_none());
    original.upstream_body = None;
    original.dialect = "openai_compatible".into();
    assert!(execute_test(&original, &host, EndpointPolicy::Official)
        .await
        .unwrap()
        .is_none());
    original.dialect = "gemini_generate_content".into();
    assert!(execute(
        &http(),
        &original,
        "request",
        Instant::now() + Duration::from_secs(5),
        true,
        EndpointPolicy::Official,
        |method, argument| host.call(method, argument)
    )
    .await
    .unwrap()
    .is_none());
    original.url = "http://127.0.0.1:1/v1beta/models/gemini-test:streamGenerateContent".into();
    assert!(execute_test(&original, &host, EndpointPolicy::Official)
        .await
        .unwrap()
        .is_none());
    assert!(host.calls().is_empty());
}

#[tokio::test]
async fn disabled_and_unavailable_keep_original_generation() {
    for state in ["disabled", "unavailable"] {
        let host = Host::new(vec![json!({"state":state})]);
        let original = wire(
            "https://generativelanguage.googleapis.com/v1/models/gemini-test:streamGenerateContent",
        );
        assert!(execute_test(&original, &host, EndpointPolicy::Official)
            .await
            .unwrap()
            .is_none());
        assert_eq!(
            host.calls(),
            vec![(
                "prepare_explicit_cache",
                json!({"request_id":"request", "deployment_id":"deployment"})
            )]
        );
    }
}

#[tokio::test]
async fn reused_payload_is_a_private_overlay_and_repair_sees_it() {
    let host = Host::new(vec![ready("cachedContents/reused")]);
    let original = wire(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:streamGenerateContent",
    );
    let overlaid = execute_test(&original, &host, EndpointPolicy::Official)
        .await
        .unwrap()
        .unwrap();
    assert!(original.upstream_payload.get("cachedContent").is_none());
    assert_eq!(overlaid.headers, original.headers);
    assert_eq!(overlaid.url, original.url);
    assert_eq!(overlaid.timeout_seconds, original.timeout_seconds);
    let mut repaired = None;
    let repair =
        crate::replay_repair::AttemptRepair::begin(&overlaid, None, &mut repaired, "request");
    assert_eq!(repair.payload()["cachedContent"], "cachedContents/reused");
    assert_eq!(host.calls().len(), 1);
}

#[tokio::test]
async fn creation_records_only_allowlisted_evidence_before_cached_generation() {
    let expires = expiry();
    let generated = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n";
    let (base, task) = server(vec![
        answer(200, &response(&expires).to_string()),
        answer(200, generated),
    ])
    .await;
    let original = wire(&format!(
        "{base}/v1beta/models/gemini-test:streamGenerateContent?alt=sse"
    ));
    let host = Host::new(vec![
        claim(&format!("{base}/v1beta/cachedContents"), &expires),
        ready("cachedContents/test-cache"),
    ]);
    let overlaid = execute_test(&original, &host, EndpointPolicy::Loopback)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        host.calls()[1],
        (
            "finish_explicit_cache",
            json!({
                "request_id":"request", "deployment_id":"deployment", "operation_id":"cache-op",
                "outcome":"ready", "http_status":200, "name":"cachedContents/test-cache",
                "expire_time":expires.1, "total_tokens":4096
            })
        )
    );
    let result = crate::upstream::open_stream(
        &http(),
        &overlaid.url,
        &overlaid.headers,
        &overlaid.idempotency_key,
        &overlaid.upstream_payload,
        None,
        Duration::from_secs(1),
        crate::dialects::Dialect::GeminiGenerateContent,
    )
    .await
    .unwrap();
    assert!(result.text().await.unwrap().contains("ok"));
    let requests = task.await.unwrap();
    assert_eq!(requests.len(), 2);
    let first_headers = requests[0].split("\r\n\r\n").next().unwrap().to_lowercase();
    assert!(first_headers.starts_with("post /v1beta/cachedcontents http/1.1"));
    assert!(first_headers.contains("x-goog-api-key: test-secret"));
    assert!(first_headers.contains("content-type: application/json"));
    assert!(!first_headers.contains("idempotency-key"));
    assert!(requests[0].contains("private prefix"));
    assert!(requests[0].contains(&expires.1));
    assert!(requests[1].contains("cachedContents/test-cache"));
    assert!(!requests[1].contains("private prefix"));
    assert!(original.upstream_payload.get("cachedContent").is_none());
}

#[tokio::test]
async fn prepare_ready_expiring_during_callback_keeps_original_generation() {
    let (base, task) = server(vec![answer(200, "original generation")]).await;
    let original = wire(&format!(
        "{base}/v1beta/models/gemini-test:streamGenerateContent"
    ));
    let mut prepared = ready("cachedContents/stale");
    prepared["expires_at"] = json!(epoch_now() + 5.05);
    let host = Host::new(vec![prepared]);
    let overlaid = execute(
        &http(),
        &original,
        "request",
        Instant::now() + Duration::from_secs(5),
        false,
        EndpointPolicy::Loopback,
        |method, argument| {
            let host = &host;
            async move {
                let reply = host.call(method, argument).await;
                tokio::time::sleep(Duration::from_millis(60)).await;
                reply
            }
        },
    )
    .await
    .unwrap();
    assert!(overlaid.is_none());
    assert_eq!(host.calls().len(), 1);
    assert_eq!(host.calls()[0].0, "prepare_explicit_cache");
    let response = crate::upstream::open_stream(
        &http(),
        &original.url,
        &original.headers,
        &original.idempotency_key,
        &original.upstream_payload,
        None,
        Duration::from_secs(1),
        crate::dialects::Dialect::GeminiGenerateContent,
    )
    .await
    .unwrap();
    assert_eq!(response.text().await.unwrap(), "original generation");
    let requests = task.await.unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].starts_with("POST /v1beta/models/"));
    assert!(requests[0].contains("prefix and suffix"));
    assert!(!requests[0].contains("cachedContent"));
}

#[tokio::test]
async fn finish_ready_with_stale_expiry_keeps_original_generation() {
    let expires = expiry();
    let (base, task) = server(vec![
        answer(200, &response(&expires).to_string()),
        answer(200, "original generation"),
    ])
    .await;
    let original = wire(&format!(
        "{base}/v1beta/models/gemini-test:streamGenerateContent"
    ));
    let mut finished = ready("cachedContents/test-cache");
    finished["expires_at"] = json!(epoch_now() - 1.0);
    let host = Host::new(vec![
        claim(&format!("{base}/v1beta/cachedContents"), &expires),
        finished,
    ]);
    assert!(execute_test(&original, &host, EndpointPolicy::Loopback)
        .await
        .unwrap()
        .is_none());
    assert_eq!(host.calls().len(), 2);
    assert_eq!(host.calls()[1].1["outcome"], "ready");
    let response = crate::upstream::open_stream(
        &http(),
        &original.url,
        &original.headers,
        &original.idempotency_key,
        &original.upstream_payload,
        None,
        Duration::from_secs(1),
        crate::dialects::Dialect::GeminiGenerateContent,
    )
    .await
    .unwrap();
    assert_eq!(response.text().await.unwrap(), "original generation");
    let requests = task.await.unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests[1].contains("prefix and suffix"));
    assert!(!requests[1].contains("cachedContent"));
}

#[tokio::test]
async fn expired_created_resource_is_accounted_before_uncached_generation() {
    let expires = expiry();
    let mut created = response(&expires);
    created["expireTime"] = json!("2000-01-01T00:00:00Z");
    let (base, task) = server(vec![
        answer(200, &created.to_string()),
        answer(200, "original generation"),
    ])
    .await;
    let original = wire(&format!(
        "{base}/v1beta/models/gemini-test:streamGenerateContent"
    ));
    let host = Host::new(vec![
        claim(&format!("{base}/v1beta/cachedContents"), &expires),
        json!({"state":"unavailable"}),
    ]);
    let overlay = execute_test(&original, &host, EndpointPolicy::Loopback)
        .await
        .unwrap();
    assert!(overlay.is_none());
    assert_eq!(
        host.calls()[1],
        (
            "finish_explicit_cache",
            json!({
                "request_id":"request", "deployment_id":"deployment", "operation_id":"cache-op",
                "outcome":"ready", "http_status":200, "name":"cachedContents/test-cache",
                "expire_time":"2000-01-01T00:00:00Z", "total_tokens":4096
            })
        )
    );
    let result = crate::upstream::open_stream(
        &http(),
        &original.url,
        &original.headers,
        &original.idempotency_key,
        &original.upstream_payload,
        None,
        Duration::from_secs(1),
        crate::dialects::Dialect::GeminiGenerateContent,
    )
    .await
    .unwrap();
    assert_eq!(result.text().await.unwrap(), "original generation");
    let requests = task.await.unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests[1].contains("prefix and suffix"));
    assert!(!requests[1].contains("cachedContent"));
    assert_eq!(host.calls().len(), 2);
}

#[tokio::test]
async fn bad_responses_and_rejections_are_unknown_once_without_retry() {
    let expires = expiry();
    let mut no_tokens = response(&expires);
    no_tokens["usageMetadata"]["totalTokenCount"] = json!(0);
    let mut wrong_scope = response(&expires);
    wrong_scope["name"] = json!("projects/other/locations/global/cachedContents/cache");
    let mut late_expiry = response(&expires);
    late_expiry["expireTime"] = json!("9999-01-01T00:00:00Z");
    let cases = vec![
        (200, "{}".to_string()),
        (200, "provider text not JSON".to_string()),
        (200, no_tokens.to_string()),
        (200, wrong_scope.to_string()),
        (200, late_expiry.to_string()),
        (200, "x".repeat(MAXIMUM_RESPONSE_BYTES + 1)),
        (401, "provider secret rejection".into()),
        (429, "rate limit raw text".into()),
        (500, "provider failure".into()),
    ];
    for (status, body) in cases {
        let (base, task) = server(vec![answer(status, &body)]).await;
        let host = Host::new(vec![
            claim(&format!("{base}/v1beta/cachedContents"), &expires),
            json!({"state":"unavailable"}),
        ]);
        let original = wire(&format!(
            "{base}/v1beta/models/gemini-test:streamGenerateContent"
        ));
        assert!(execute_test(&original, &host, EndpointPolicy::Loopback)
            .await
            .unwrap()
            .is_none());
        assert_eq!(task.await.unwrap().len(), 1);
        assert_eq!(
            host.calls()[1].1,
            json!({"request_id":"request", "deployment_id":"deployment",
            "operation_id":"cache-op", "outcome":"unknown", "http_status":status})
        );
        assert_eq!(host.calls().len(), 2);
    }
}

#[tokio::test]
async fn chunked_body_cap_and_redirect_policy_are_enforced() {
    let expires = expiry();
    let large = "x".repeat(MAXIMUM_RESPONSE_BYTES + 1);
    let chunked = format!(
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{}\r\n0\r\n\r\n",
        large.len(),
        large
    );
    for reply in [chunked, "HTTP/1.1 307 Redirect\r\nLocation: http://127.0.0.1:1/private\r\nContent-Length: 0\r\n\r\n".into()] {
        let (base, task) = server(vec![reply]).await;
        let host = Host::new(vec![claim(&format!("{base}/v1beta/cachedContents"), &expires), json!({"state":"unavailable"})]);
        let original = wire(&format!("{base}/v1beta/models/gemini-test:streamGenerateContent"));
        assert!(execute_test(&original, &host, EndpointPolicy::Loopback).await.unwrap().is_none());
        assert_eq!(task.await.unwrap().len(), 1);
        assert_eq!(host.calls()[1].1["outcome"], "unknown");
    }
}

#[tokio::test]
async fn create_transport_failure_and_timeout_acknowledge_unknown() {
    let expires = expiry();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let delayed = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        request(&mut socket).await;
        tokio::time::sleep(Duration::from_millis(150)).await;
    });
    let mut original = wire(&format!(
        "{base}/v1beta/models/gemini-test:streamGenerateContent"
    ));
    original.timeout_seconds = 0.025;
    let host = Host::new(vec![
        claim(&format!("{base}/v1beta/cachedContents"), &expires),
        json!({"state":"unavailable"}),
    ]);
    assert!(execute_test(&original, &host, EndpointPolicy::Loopback)
        .await
        .unwrap()
        .is_none());
    assert_eq!(host.calls()[1].1["outcome"], "unknown");
    assert!(host.calls()[1].1.get("http_status").is_none());
    delayed.await.unwrap();
    let host = Host::new(vec![
        claim(&format!("{base}/v1beta/cachedContents"), &expires),
        json!({"state":"unavailable"}),
    ]);
    assert!(execute_test(&original, &host, EndpointPolicy::Loopback)
        .await
        .unwrap()
        .is_none());
    assert_eq!(host.calls()[1].1["outcome"], "unknown");
}

#[tokio::test]
async fn accounting_failure_never_returns_generation_overlay_or_fallback() {
    let expires = expiry();
    let (base, task) = server(vec![answer(200, &response(&expires).to_string())]).await;
    let host = Host::new(vec![claim(
        &format!("{base}/v1beta/cachedContents"),
        &expires,
    )]);
    host.replies
        .lock()
        .unwrap()
        .push_back(Err(PublicError::internal()));
    let original = wire(&format!(
        "{base}/v1beta/models/gemini-test:streamGenerateContent"
    ));
    let failure = execute_test(&original, &host, EndpointPolicy::Loopback)
        .await
        .unwrap_err();
    assert_eq!(failure.failure_class, FailureClass::Internal);
    assert!(!failure.retryable_same_deployment && !failure.failover_eligible);
    assert_eq!(task.await.unwrap().len(), 1);
    assert_eq!(host.calls().len(), 2);
}

#[tokio::test]
async fn cancellation_after_create_dispatch_leaves_claim_unfinished() {
    let expires = expiry();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let (accepted, received) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        request(&mut socket).await;
        accepted.send(()).unwrap();
        std::future::pending::<()>().await;
    });
    let host = Host::new(vec![claim(
        &format!("{base}/v1beta/cachedContents"),
        &expires,
    )]);
    let task_host = host.clone();
    let original = wire(&format!(
        "{base}/v1beta/models/gemini-test:streamGenerateContent"
    ));
    let executing =
        tokio::spawn(
            async move { execute_test(&original, &task_host, EndpointPolicy::Loopback).await },
        );
    tokio::time::timeout(Duration::from_secs(2), received)
        .await
        .unwrap()
        .unwrap();
    executing.abort();
    assert!(executing.await.unwrap_err().is_cancelled());
    assert_eq!(host.calls().len(), 1);
    assert_eq!(host.calls()[0].0, "prepare_explicit_cache");
    server.abort();
}

#[test]
fn create_expiration_never_exceeds_or_drifts_from_reserved_horizon() {
    let expires = expiry();
    let url = "https://generativelanguage.googleapis.com/v1beta/cachedContents";
    let endpoint = cache_endpoint(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:streamGenerateContent",
        EndpointPolicy::Official,
    )
    .unwrap();
    let mut payload = claim(url, &expires)["payload"].clone();
    payload["expireTime"] = json!(expires.1.replace('Z', ".0005Z"));
    assert!(!valid_create(&endpoint, url, &payload, expires.0));
}

#[tokio::test]
async fn provider_submillisecond_expiry_extension_records_unknown() {
    let expires = expiry();
    let mut created = response(&expires);
    created["expireTime"] = json!(expires.1.replace('Z', ".0005Z"));
    let (base, task) = server(vec![answer(200, &created.to_string())]).await;
    let original = wire(&format!(
        "{base}/v1beta/models/gemini-test:streamGenerateContent"
    ));
    let host = Host::new(vec![
        claim(&format!("{base}/v1beta/cachedContents"), &expires),
        json!({"state":"unavailable"}),
    ]);
    assert!(execute_test(&original, &host, EndpointPolicy::Loopback)
        .await
        .unwrap()
        .is_none());
    assert_eq!(task.await.unwrap().len(), 1);
    assert_eq!(host.calls()[1].1["outcome"], "unknown");
    assert!(host.calls()[1].1.get("expire_time").is_none());
}

#[tokio::test]
async fn unsafe_claims_acknowledge_unknown_without_http() {
    let expires = expiry();
    let original = wire(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:streamGenerateContent",
    );
    let endpoint = "https://generativelanguage.googleapis.com/v1beta/cachedContents";
    let mut cases = Vec::new();
    for url in [
        "https://attacker.test/v1beta/cachedContents",
        "http://generativelanguage.googleapis.com/v1beta/cachedContents",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:streamGenerateContent",
        "https://generativelanguage.googleapis.com/v1beta/cachedContents?key=secret",
    ] {
        cases.push(claim(url, &expires));
    }
    let mut ttl = claim(endpoint, &expires);
    ttl["payload"]["ttl"] = json!("300s");
    cases.push(ttl);
    let mut mismatch = claim(endpoint, &expires);
    mismatch["expires_at"] = json!(expires.0 + 1.0);
    cases.push(mismatch);
    let mut too_long = claim(endpoint, &expires);
    too_long["expires_at"] = json!(epoch_now() + 301.0);
    cases.push(too_long);
    let mut expired = claim(endpoint, &expires);
    expired["expires_at"] = json!(epoch_now());
    cases.push(expired);
    for claim in cases {
        let host = Host::new(vec![claim, json!({"state":"unavailable"})]);
        assert!(execute_test(&original, &host, EndpointPolicy::Official)
            .await
            .unwrap()
            .is_none());
        assert_eq!(
            host.calls()[1].1,
            json!({"request_id":"request", "deployment_id":"deployment",
            "operation_id":"cache-op", "outcome":"unknown"})
        );
    }
}

#[tokio::test]
async fn malformed_host_reply_and_ready_after_unknown_fail_closed() {
    let original = wire(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:streamGenerateContent",
    );
    for reply in [
        json!({}),
        ready("cachedContents/../bad"),
        json!({"state":"ready", "resource_name":"cachedContents/cache", "payload":{}}),
    ] {
        let host = Host::new(vec![reply]);
        assert_eq!(
            execute_test(&original, &host, EndpointPolicy::Official)
                .await
                .unwrap_err()
                .failure_class,
            FailureClass::Internal
        );
    }
    let host = Host::new(vec![
        claim("https://attacker.test/cache", &expiry()),
        ready("cachedContents/cache"),
    ]);
    assert_eq!(
        execute_test(&original, &host, EndpointPolicy::Official)
            .await
            .unwrap_err()
            .failure_class,
        FailureClass::Internal
    );
}

#[tokio::test]
async fn malformed_ready_expiry_fails_closed_in_both_callbacks() {
    let url =
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:streamGenerateContent";
    for invalid in [Value::Null, json!("NaN"), json!("999999999999999999999")] {
        let mut reply = ready("cachedContents/test-cache");
        reply["expires_at"] = invalid;
        let host = Host::new(vec![reply.clone()]);
        assert_eq!(
            execute_test(&wire(url), &host, EndpointPolicy::Official)
                .await
                .unwrap_err()
                .failure_class,
            FailureClass::Internal
        );
        let expires = expiry();
        let (base, task) = server(vec![answer(200, &response(&expires).to_string())]).await;
        let host = Host::new(vec![
            claim(&format!("{base}/v1beta/cachedContents"), &expires),
            reply,
        ]);
        let original = wire(&format!(
            "{base}/v1beta/models/gemini-test:streamGenerateContent"
        ));
        assert_eq!(
            execute_test(&original, &host, EndpointPolicy::Loopback)
                .await
                .unwrap_err()
                .failure_class,
            FailureClass::Internal
        );
        assert_eq!(task.await.unwrap().len(), 1);
    }
    let endpoint = cache_endpoint(url, EndpointPolicy::Official).unwrap();
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(overlay(
            &wire(url),
            ready("cachedContents/test-cache")["payload"].clone(),
            "cachedContents/test-cache",
            &endpoint,
            invalid
        )
        .is_err());
    }
}

#[tokio::test]
async fn expired_request_cannot_continue_generation_after_host_ack() {
    let original = wire(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:streamGenerateContent",
    );
    let host = Host::new(vec![json!({"state":"unavailable"})]);
    let failure = execute(
        &http(),
        &original,
        "request",
        Instant::now() - Duration::from_secs(1),
        false,
        EndpointPolicy::Official,
        |method, argument| host.call(method, argument),
    )
    .await
    .unwrap_err();
    assert!(matches!(
        failure.failure_class,
        FailureClass::Timeout | FailureClass::Internal
    ));
}

#[test]
fn official_endpoint_and_resource_scopes_are_exact() {
    for (generation, expected, prefix) in [
        ("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
         "https://generativelanguage.googleapis.com/v1beta/cachedContents", "cachedContents/"),
        ("https://us-central1-aiplatform.googleapis.com/v1/projects/project/locations/us-central1/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
         "https://us-central1-aiplatform.googleapis.com/v1/projects/project/locations/us-central1/cachedContents", "projects/project/locations/us-central1/cachedContents/"),
        ("https://aiplatform.googleapis.com/v1beta1/projects/123/locations/global/publishers/google/models/gemini-test:streamGenerateContent",
         "https://aiplatform.googleapis.com/v1beta1/projects/123/locations/global/cachedContents", "projects/123/locations/global/cachedContents/"),
    ] {
        let endpoint = cache_endpoint(generation, EndpointPolicy::Official).unwrap();
        assert_eq!(endpoint.url.as_str(), expected);
        assert_eq!(endpoint.resource_prefix, prefix);
        assert!(valid_resource(&endpoint, &format!("{prefix}cache-123")));
        assert!(!valid_resource(&endpoint, &format!("{prefix}../other")));
        assert!(!valid_resource(&endpoint, "https://attacker.test/cache"));
    }
    for url in [
        "https://generativelanguage.googleapis.com.attacker.test/v1beta/models/model:streamGenerateContent",
        "https://user@generativelanguage.googleapis.com/v1beta/models/model:streamGenerateContent",
        "https://generativelanguage.googleapis.com:444/v1beta/models/model:streamGenerateContent",
        "http://generativelanguage.googleapis.com/v1beta/models/model:streamGenerateContent",
        "https://generativelanguage.googleapis.com/v1beta/models/model:generateContent",
        "https://aiplatform.googleapis.com/v1/projects/p/locations/l/publishers/other/models/m:streamGenerateContent",
        "https://evil.aiplatform.googleapis.com/v1/projects/p/locations/l/publishers/google/models/m:streamGenerateContent",
        "https://generativelanguage.googleapis.com/v1beta/models/m%2fn:streamGenerateContent",
        "https://generativelanguage.googleapis.com/v1beta/models/model:streamGenerateContent#fragment",
    ] {
        assert!(cache_endpoint(url, EndpointPolicy::Official).is_none(), "{url}");
    }
}

#[test]
fn query_auth_comes_only_from_admitted_studio_url() {
    let generation = "https://generativelanguage.googleapis.com/v1beta/models/m:streamGenerateContent?key=test%20key&alt=sse";
    let endpoint = cache_endpoint(generation, EndpointPolicy::Official).unwrap();
    assert_eq!(
        endpoint.url.query_pairs().collect::<Vec<_>>(),
        vec![("key".into(), "test key".into())]
    );
    assert!(matching_create_url(
        &endpoint.url,
        "https://generativelanguage.googleapis.com/v1beta/cachedContents?key=test%20key"
    ));
    assert!(!matching_create_url(
        &endpoint.url,
        "https://generativelanguage.googleapis.com/v1beta/cachedContents?key=other"
    ));
    assert!(!matching_create_url(
        &endpoint.url,
        "https://generativelanguage.googleapis.com/v1beta/cachedContents?key=test+key&alt=sse"
    ));
    assert!(!matching_create_url(
        &endpoint.url,
        "https://generativelanguage.googleapis.com/v1beta/cachedContents"
    ));
    for generation in [
        "https://generativelanguage.googleapis.com/v1beta/models/m:streamGenerateContent?key=a&key=b",
        "https://generativelanguage.googleapis.com/v1beta/models/m:streamGenerateContent?key=",
        "https://generativelanguage.googleapis.com/v1beta/models/m:streamGenerateContent?alt=json",
        "https://generativelanguage.googleapis.com/v1beta/models/m:streamGenerateContent?other=secret",
        "https://aiplatform.googleapis.com/v1/projects/p/locations/global/publishers/google/models/m:streamGenerateContent?key=secret",
    ] {
        assert!(cache_endpoint(generation, EndpointPolicy::Official).is_none());
    }
}

#[tokio::test]
async fn loopback_create_preserves_admitted_query_key_without_bridging_it() {
    let expires = expiry();
    let (base, task) = server(vec![answer(200, &response(&expires).to_string())]).await;
    let mut original = wire(&format!(
        "{base}/v1beta/models/gemini-test:streamGenerateContent?alt=sse&key=test%20key"
    ));
    original.headers.remove("x-goog-api-key");
    let host = Host::new(vec![
        claim(
            &format!("{base}/v1beta/cachedContents?key=test%20key"),
            &expires,
        ),
        ready("cachedContents/test-cache"),
    ]);
    assert!(execute_test(&original, &host, EndpointPolicy::Loopback)
        .await
        .unwrap()
        .is_some());
    assert!(task.await.unwrap()[0].starts_with("POST /v1beta/cachedContents?key=test+key HTTP/1.1"));
    assert!(!serde_json::to_string(&host.calls())
        .unwrap()
        .contains("test key"));
    assert!(!serde_json::to_string(&host.calls())
        .unwrap()
        .contains("test%20key"));
}

#[test]
fn absolute_expiry_parser_rejects_invalid_dates_and_relative_ttl() {
    assert_eq!(expiry_epoch("1970-01-01T00:00:00Z"), Some(0.0));
    assert_eq!(
        expiry_epoch("2023-11-14T22:13:20.500+00:00"),
        Some(1_700_000_000.5)
    );
    assert_eq!(expiry_epoch("2024-02-29T00:00:00Z"), Some(1_709_164_800.0));
    for invalid in [
        "300s",
        "2023-02-29T00:00:00Z",
        "2024-13-01T00:00:00Z",
        "2024-01-01T24:00:00Z",
        "2024-01-01T00:00:60Z",
        "2024-01-01T00:00:00.1234567890Z",
        "2024-01-01T00:00:00+01:00",
        "2024-01-01T00:00:00.Z",
        "2024-01-01T00:00:00NaNZ",
        "éééééééééééééééééééé",
    ] {
        assert!(expiry_epoch(invalid).is_none(), "{invalid}");
    }
}
