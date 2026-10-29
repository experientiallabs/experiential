//! Reseller and account-verdict tests for the pre-stream 4xx read: Novita's
//! flat envelope (sentence relay, lane limitation, MODEL_NOT_FOUND policy,
//! NOT_ENOUGH_BALANCE under 403) and the 429 body read (OpenAI
//! insufficient_quota, a throttle's token as ledger detail). Split from
//! `upstream.rs` for the module line budget; `open_against_body` is the
//! parent test module's loopback harness.

use super::tests::open_against_body;
use super::*;

#[tokio::test]
async fn a_resellers_flat_400_relays_its_sentence() {
    // Novita's envelope: no `error` object (live shape, 2026-09-15). The
    // caller must see the provider's sentence, and the ledger must record
    // the attempt as detailed, exactly as it does for an OpenAI body.
    let failure = open_against_body(
        "400 Bad Request",
        "{\"code\":400,\"reason\":\"INVALID_REQUEST_BODY\",\
         \"message\":\"max_tokens must be less than or equal to 131072\",\"metadata\":{}}",
        "deepseek/deepseek-v4.1-flash",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
    assert!(!failure.failover_eligible);
    assert_eq!(
        failure.provider_detail.as_deref(),
        Some("max_tokens must be less than or equal to 131072")
    );
    assert_eq!(
        failure.public_error().message,
        "provider rejected the request: max_tokens must be less than or equal to 131072"
    );
}

#[tokio::test]
async fn a_resellers_flat_lane_limitation_400_fails_over() {
    let failure = open_against_body(
        "400 Bad Request",
        "{\"code\":400,\"reason\":\"INVALID_REQUEST_BODY\",\
         \"message\":\"System message must be at the beginning.\",\"metadata\":{}}",
        "pa/gpt-5.6-luna",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
    assert!(
        failure.failover_eligible,
        "another rung can carry the request"
    );
    assert!(failure
        .provider_detail
        .as_deref()
        .is_some_and(|detail| detail.contains("System message must be at the beginning")));
}

#[tokio::test]
async fn a_resellers_flat_403_balance_verdict_is_provider_quota() {
    // Novita answers an unfunded account with 403 NOT_ENOUGH_BALANCE; a
    // status-only read filed it as a credential failure, which the house
    // exhaustion sweep (reading provider_quota) never sees.
    let failure = open_against_body(
        "403 Forbidden",
        "{\"code\":403,\"reason\":\"NOT_ENOUGH_BALANCE\",\
         \"message\":\"Insufficient balance\",\"metadata\":{}}",
        "m",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::ProviderQuota);
    assert!(failure.failover_eligible);
    assert_eq!(
        failure.provider_detail.as_deref(),
        Some("http 403: NOT_ENOUGH_BALANCE"),
        "an account-state failure keeps only the status and token"
    );
    // Its credential siblings keep the credential class.
    let denied = open_against_body(
        "403 Forbidden",
        "{\"code\":403,\"reason\":\"ACCESS_DENY\",\"message\":\"Access denied\",\"metadata\":{}}",
        "m",
    )
    .await;
    assert_eq!(denied.failure_class, FailureClass::ProviderAuthentication);
}

#[tokio::test]
async fn a_429_naming_an_exhausted_account_is_provider_quota_and_a_throttle_keeps_its_code() {
    // OpenAI's live shape for an unfunded account: HTTP 429, code
    // insufficient_quota (docs, 2026-09). The status says throttle; the
    // body says the account is dead.
    let quota = open_against_body(
        "429 Too Many Requests",
        "{\"error\":{\"message\":\"You exceeded your current quota, please check your \
         plan and billing details.\",\"type\":\"insufficient_quota\",\"param\":null,\
         \"code\":\"insufficient_quota\"}}",
        "m",
    )
    .await;
    assert_eq!(quota.failure_class, FailureClass::ProviderQuota);
    assert!(quota.failover_eligible);
    assert_eq!(
        quota.provider_detail.as_deref(),
        Some("http 429: insufficient_quota")
    );
    // A genuine throttle keeps its class and the code token rides into
    // the ledger only; the public error stays the generic throttle text.
    let throttled = open_against_body(
        "429 Too Many Requests",
        "{\"code\":429,\"reason\":\"TOKEN_LIMIT_EXCEEDED\",\
         \"message\":\"Token limit exceeded, please try again later\",\"metadata\":{}}",
        "m",
    )
    .await;
    assert_eq!(throttled.failure_class, FailureClass::Throttled);
    assert!(throttled.failover_eligible);
    assert_eq!(
        throttled.provider_detail.as_deref(),
        Some("http 429: TOKEN_LIMIT_EXCEEDED")
    );
    assert_eq!(
        throttled.public_error().message,
        "provider throttled the request; retry after the delay in the Retry-After header"
    );
    // No body leaves the throttle with the status-only detail.
    let bare = open_against_body("429 Too Many Requests", "", "m").await;
    assert_eq!(bare.failure_class, FailureClass::Throttled);
    assert_eq!(bare.provider_detail.as_deref(), Some("http 429"));
}

#[tokio::test]
async fn a_resellers_flat_model_not_found_400_takes_the_lane_policy() {
    let failure = open_against_body(
        "400 Bad Request",
        "{\"code\":400,\"reason\":\"MODEL_NOT_FOUND\",\"message\":\"Model not found\",\"metadata\":{}}",
        "m",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::ProviderNotFound);
    assert!(failure.failover_eligible);
}

#[tokio::test]
async fn a_429_whose_body_stalls_never_outlives_the_header_phase_budget() {
    // The throttle-body read is bounded twice over: its own 250 ms budget and
    // what is left of the rung's header-phase window, which the waterfall
    // already sizes to the caller's remaining deadline (`open_bound`). A
    // provider that answers 429 and then never sends its body must fail over
    // as a content-free throttle within that window, never after the stall.
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let mut buffer = [0u8; 8192];
        let _ = socket.read(&mut buffer).await;
        // Headers promise a body that never comes.
        socket
            .write_all(
                b"HTTP/1.1 429 Too Many Requests\r\ncontent-type: application/json\r\n\
                  content-length: 200\r\n\r\n",
            )
            .await
            .expect("write");
        tokio::time::sleep(Duration::from_secs(5)).await;
    });
    let client = build_client(Duration::from_secs(2), false).expect("client");
    let started = Instant::now();
    let failure = open_stream(
        &client,
        &format!("http://{addr}/v1/chat/completions"),
        &HashMap::new(),
        "idem-429-stall",
        &serde_json::json!({"model": "m", "messages": []}),
        None,
        Duration::from_millis(100),
        Dialect::OpenAiCompatible,
    )
    .await
    .expect_err("a 429 is a failure");
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(1),
        "the stalled body must not be waited out: {elapsed:?}"
    );
    assert_eq!(failure.failure_class, FailureClass::Throttled);
    assert!(failure.failover_eligible);
    assert_eq!(failure.provider_detail.as_deref(), Some("http 429"));
}

#[tokio::test]
async fn a_resellers_400_naming_an_exhausted_account_is_provider_quota() {
    // Live Novita sentence (gpt-5.6-sol, 2026-09-16 05:28Z) under HTTP 400 in
    // the flat `{code:0, message, type}` envelope: the account's prepaid
    // balance is gone. A status-only read filed it as the caller's 400.
    let failure = open_against_body(
        "400 Bad Request",
        "{\"code\":0,\"message\":\"Insufficient quota available for instant inference. \
         trace_id: 92913336280c9c28f727ac9bfefbd89c\",\"type\":\"invalid_request_error\"}",
        "pa/gpt-5.6-sol",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::ProviderQuota);
    assert!(
        failure.failover_eligible,
        "another rung must serve the request"
    );
    assert!(!failure.retryable_same_deployment);
    let detail = failure.provider_detail.as_deref().expect("ledger detail");
    assert!(
        detail.starts_with("http 400: Insufficient quota available for instant inference"),
        "{detail}"
    );
    assert!(
        !detail.contains("92913336280c9c28f727ac9bfefbd89c"),
        "the trace id is masked: {detail}"
    );
    assert!(
        !failure
            .public_error()
            .message
            .contains("Insufficient quota"),
        "an account-state failure never relays its sentence to the caller"
    );
    // An ordinary caller sentence that merely mentions a quota WORD elsewhere
    // does not qualify: only the funding phrases do.
    let caller = open_against_body(
        "400 Bad Request",
        "{\"error\":{\"message\":\"max_tokens must be less than or equal to 131072\",\
         \"type\":\"invalid_request_error\"}}",
        "m",
    )
    .await;
    assert_eq!(caller.failure_class, FailureClass::InvalidRequest);
}

#[tokio::test]
async fn a_relays_decode_failure_relays_the_upstream_error_it_embeds() {
    // Live Novita sentence (gpt-5.6-luna on the Responses wire, 2026-09-16
    // 05:31Z): the relay's Go decoder choked on a numeric upstream
    // `error.code` and answered its own 400 with the upstream document after
    // `raw: `. The caller must see the UPSTREAM sentence (their own image
    // limit), never the decoder's noise.
    let failure = open_against_body(
        "400 Bad Request",
        "{\"error\":{\"code\":0,\"message\":\"failed to decode error response: json: cannot \
         unmarshal number into Go struct field ResponseError.error.code of type string, raw: \
         {\\\"error\\\":{\\\"code\\\":0,\\\"message\\\":\\\"Exceeded maximum number of images (50) \
         allowed in the request.\\\"}} trace_id: 92913336280c9c28f727ac9bfefbd89c\",\
         \"type\":\"invalid_request_error\"}}",
        "pa/gpt-5.6-luna",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
    assert!(!failure.failover_eligible);
    assert_eq!(
        failure.provider_detail.as_deref(),
        Some("Exceeded maximum number of images (50) allowed in the request.")
    );
    assert_eq!(
        failure.public_error().message,
        "provider rejected the request: Exceeded maximum number of images (50) allowed in the request."
    );
    // A relayed THROTTLE takes the throttle class and fails over; the
    // upstream sentence stays ledger-only under the relay's status.
    let throttled = open_against_body(
        "400 Bad Request",
        "{\"error\":{\"code\":0,\"message\":\"failed to decode error response: json: cannot \
         unmarshal number into Go struct field ResponseError.error.code of type string, raw: \
         {\\\"error\\\":{\\\"code\\\":429,\\\"message\\\":\\\"Rate limit exceeded, please retry later.\\\"}}\",\
         \"type\":\"invalid_request_error\"}}",
        "pa/gpt-5.6-luna",
    )
    .await;
    assert_eq!(throttled.failure_class, FailureClass::Throttled);
    assert!(throttled.failover_eligible);
    assert_eq!(
        throttled.provider_detail.as_deref(),
        Some("http 400: Rate limit exceeded, please retry later.")
    );
    assert!(!throttled
        .public_error()
        .message
        .contains("Rate limit exceeded, please retry"));
    // A truncated upstream document (the relay cuts long bodies) still yields
    // its sentence by the bounded scan.
    let cut = open_against_body(
        "400 Bad Request",
        "{\"error\":{\"code\":0,\"message\":\"failed to decode error response: json: cannot \
         unmarshal number into Go struct field ResponseError.error.code of type string, raw: \
         {\\\"error\\\":{\\\"code\\\":0,\\\"message\\\":\\\"Exceeded maximum number of images (50) \
         allowed in the request. trace_id: 92913336280c9c28f727ac9bfefbd89c\",\
         \"type\":\"invalid_request_error\"}}",
        "pa/gpt-5.6-luna",
    )
    .await;
    assert_eq!(cut.failure_class, FailureClass::InvalidRequest);
    assert!(cut
        .provider_detail
        .as_deref()
        .is_some_and(|detail| detail.starts_with("Exceeded maximum number of images (50)")));
}
