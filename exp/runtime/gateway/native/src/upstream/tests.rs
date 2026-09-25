//! Unit tests for the upstream transport classification.

use super::*;

#[tokio::test]
async fn h2_refused_stream_never_redials_beneath_the_waterfall() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let recorded = calls.clone();
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let mut connection = h2::server::handshake(socket).await.unwrap();
        while let Some(Ok((_request, mut response))) = connection.accept().await {
            recorded.fetch_add(1, Ordering::SeqCst);
            response.send_reset(h2::Reason::REFUSED_STREAM);
        }
    });
    let client = client_builder(Duration::from_secs(1))
        .http2_prior_knowledge()
        .build()
        .unwrap();
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        client
            .post(format!("http://{address}/v1/chat/completions"))
            .body("{}")
            .send(),
    )
    .await
    .unwrap();
    assert!(result.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    server.abort();
}

#[tokio::test]
async fn decision_wire_omits_all_idempotency_keys_without_changing_chat_headers() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    for dialect in [Dialect::TypesafeSystemone, Dialect::OpenAiCompatible] {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        let received = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut request = Vec::new();
            loop {
                let mut chunk = [0u8; 1024];
                let n = socket.read(&mut chunk).await.expect("read");
                assert!(n > 0, "complete request headers required");
                request.extend_from_slice(&chunk[..n]);
                assert!(request.len() < 8192);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            socket
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}")
                .await
                .expect("write");
            String::from_utf8(request).expect("HTTP headers")
        });
        let client = build_client(Duration::from_secs(2)).expect("client");
        open_stream(
            &client,
            &format!("http://{addr}/v1/systemone"),
            &HashMap::from([
                ("iDeMpOtEnCy-KeY".into(), "configured-key".into()),
                ("Authorization".into(), "Bearer test-key".into()),
            ]),
            "synthesized-key",
            &serde_json::json!({"model": "jev-latest"}),
            None,
            Duration::from_secs(2),
            dialect,
        )
        .await
        .expect("success");
        let headers = received.await.expect("server task").to_ascii_lowercase();
        assert!(headers.contains("authorization: bearer test-key\r\n"));
        assert!(!headers.contains("configured-key"));
        match dialect {
            Dialect::TypesafeSystemone => assert!(!headers.contains("idempotency-key")),
            _ => assert!(headers.contains("idempotency-key: synthesized-key\r\n")),
        }
    }
}

#[test]
fn transport_failure_flags_mirror_the_python_taxonomy() {
    // (status, retryable_same_deployment, failover_eligible)
    let table = [
        (Some(401), false, true),
        (Some(403), false, true),
        (Some(404), false, true),
        (Some(429), false, true),
        (Some(402), false, true),
        (Some(408), true, true),
        (Some(500), true, true),
        (Some(503), true, true),
        (Some(409), true, true),
        (Some(425), true, true),
        (Some(400), false, false),
        (Some(422), false, false),
        (Some(301), false, true),
        (None, true, true),
    ];
    for (status, retryable, failover) in table {
        let failure = transport_failure(status);
        assert_eq!(
            failure.retryable_same_deployment, retryable,
            "retryable for {status:?}"
        );
        assert_eq!(
            failure.failover_eligible, failover,
            "failover for {status:?}"
        );
    }
}

#[tokio::test]
async fn a_402_with_the_literal_tokenhub_body_classes_provider_quota() {
    // TokenHub (the Tencent relay) answers HTTP 402 with provider code
    // 401008 on EVERY request shape once the account's free trial is
    // exhausted and postpaid billing is off. Classing that invalid_request
    // blamed 652 callers' request fields for the provider's billing state
    // (2026-09 incident): the class must be the provider-side quota family,
    // the rung must fail over, and the body must never be relayed.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let (mut socket, _) = listener.accept().await.expect("accept");
        let mut buffer = [0u8; 8192];
        let _ = socket.read(&mut buffer).await;
        let body = "{\"error\":{\"code\":401008,\"message\":\"free trial quota exhausted \
                    and postpaid billing is not enabled - enable in Console > Online \
                    Inference Service\",\"type\":\"payment_required\"}}";
        let response = format!(
            "HTTP/1.1 402 Payment Required\r\ncontent-type: application/json\r\n\
             content-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body,
        );
        socket.write_all(response.as_bytes()).await.expect("write");
    });
    let client = build_client(Duration::from_secs(2)).expect("client");
    let failure = open_stream(
        &client,
        &format!("http://{addr}/v1/chat/completions"),
        &HashMap::new(),
        "idem-402",
        &serde_json::json!({"model": "m", "messages": []}),
        None,
        Duration::from_secs(5),
        Dialect::OpenAiCompatible,
    )
    .await
    .expect_err("a 402 must classify as a failure");
    assert_eq!(failure.failure_class, FailureClass::ProviderQuota);
    assert!(failure.failover_eligible, "an unfunded rung must fail over");
    assert!(!failure.retryable_same_deployment);
    assert!(
        failure.rejected_parameter.is_none(),
        "a billing failure must stay content-free"
    );
    // The only detail is the engine's own status token, never body text.
    assert_eq!(failure.provider_detail.as_deref(), Some("http 402"));
    assert!(!failure.public_error().message.contains("402"));
    assert!(
        !failure.safe_message.contains("request fields"),
        "the caller must never be told to fix their fields for a provider billing state"
    );
}

pub(super) async fn open_against_body(status_line: &str, body: &str, model: &str) -> Failure {
    open_dialect_against_body(status_line, body, model, Dialect::OpenAiCompatible).await
}

async fn open_dialect_against_body(
    status_line: &str,
    body: &str,
    model: &str,
    dialect: Dialect,
) -> Failure {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    let status_line = status_line.to_string();
    let body = body.to_string();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let mut buffer = [0u8; 8192];
        let _ = socket.read(&mut buffer).await;
        let response = format!(
            "HTTP/1.1 {status_line}\r\ncontent-type: application/json\r\n\
             content-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body,
        );
        socket.write_all(response.as_bytes()).await.expect("write");
    });
    let client = build_client(Duration::from_secs(2)).expect("client");
    open_stream(
        &client,
        &format!("http://{addr}/v1/chat/completions"),
        &HashMap::new(),
        "idem-4xx",
        &serde_json::json!({"model": model, "messages": []}),
        None,
        Duration::from_secs(5),
        dialect,
    )
    .await
    .expect_err("a 4xx must classify as a failure")
}

#[tokio::test]
async fn a_dropped_provider_sentence_still_relays_the_provider_code() {
    // The sentence names an account handle: the handle is masked and the
    // sentence around it still reaches the caller.
    let failure = open_against_body(
        "400 Bad Request",
        "{\"error\":{\"code\":\"invalid_value\",\"type\":\"invalid_request_error\",\
         \"message\":\"Invalid value for organization org_a1b2c3d4e5f6: not allowed\"}}",
        "m",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
    assert_eq!(
        failure.provider_detail.as_deref(),
        Some("Invalid value for organization [redacted]: not allowed")
    );
    assert_eq!(
        failure.public_error().message,
        "provider rejected the request: Invalid value for organization [redacted]: not allowed"
    );
}

#[tokio::test]
async fn a_404_for_a_callers_dangling_item_reference_is_the_callers_400() {
    let failure = open_against_body(
        "404 Not Found",
        "{\"error\":{\"message\":\"Item with id 'rs_0000' not found. Items are not \
         persisted when `store` is set to false.\",\"type\":\"invalid_request_error\",\
         \"param\":\"input\",\"code\":null}}",
        "gpt-6-astra",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
    assert!(
        !failure.failover_eligible,
        "every rung would answer the same"
    );
    assert_eq!(failure.public_error().status_code, 400);
    assert!(failure
        .provider_detail
        .as_deref()
        .is_some_and(|detail| detail.starts_with("Item with id 'rs_0000' not found")));
}

#[tokio::test]
async fn a_vllm_flat_400_with_trailing_help_text_relays_its_sentence() {
    // Exact body captured live from an Azure Foundry DeepSeek deployment
    // (2026-09-15): 438 such 400s in 48h had settled with no detail
    // because the body is a flat vLLM object followed by a help line.
    let failure = open_against_body(
        "400 Bad Request",
        "{\"object\":\"error\",\"message\":\"Tool 'g' not found in tools list.\",\
         \"type\":\"BadRequestError\",\"param\":null,\"code\":400}\n\
         Please check this guide to understand why this error code might have been returned \n\
         https://docs.microsoft.com/en-us/azure/machine-learning/how-to-troubleshoot-online-endpoints#http-status-codes\n",
        "DeepSeek-V4-Flash",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
    assert_eq!(
        failure.provider_detail.as_deref(),
        Some("Tool 'g' not found in tools list.")
    );
    assert_eq!(
        failure.public_error().message,
        "provider rejected the request: Tool 'g' not found in tools list."
    );
}

#[tokio::test]
async fn xai_and_novita_envelopes_relay_their_sentences() {
    // xAI spells the sentence as a string `error` beside a `code`
    // (captured live 2026-09-15).
    let xai = open_against_body(
        "400 Bad Request",
        "{\"code\":\"invalid-argument\",\"error\":\"Argument not supported on this \
         model: presencePenalty\"}",
        "grok-4.20-multi-agent",
    )
    .await;
    assert_eq!(xai.failure_class, FailureClass::InvalidRequest);
    assert_eq!(
        xai.provider_detail.as_deref(),
        Some("Argument not supported on this model: presencePenalty")
    );
    // Novita answers a flat gRPC-style object whose `reason` is the token
    // (captured live 2026-09-15).
    let novita = open_against_body(
        "400 Bad Request",
        "{\"code\":400,\"reason\":\"INVALID_PARAMETER\",\"message\":\"tools is not \
         supported by this model\",\"metadata\":{}}",
        "deepseek/deepseek-v4.1-flash",
    )
    .await;
    assert_eq!(novita.failure_class, FailureClass::InvalidRequest);
    assert_eq!(
        novita.provider_detail.as_deref(),
        Some("tools is not supported by this model")
    );
}

#[tokio::test]
async fn status_only_classifications_carry_the_status_as_ledger_detail() {
    // A 503 relay fault and a 500 model fault were indistinguishable on
    // the ledger; the status token now rides as the detail, ledger-only.
    let failure = open_against_body(
        "503 Service Unavailable",
        "{\"error\":{\"message\":\"upstream connect error to 10.0.0.7\"}}",
        "m",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::ProviderInternal);
    assert_eq!(failure.provider_detail.as_deref(), Some("http 503"));
    assert_eq!(
        failure.public_error().message,
        "provider service failed; retry after a short delay",
        "a server-side class never relays detail to the caller"
    );
    let throttled = open_against_body("429 Too Many Requests", "", "m").await;
    assert_eq!(throttled.failure_class, FailureClass::Throttled);
    assert_eq!(throttled.provider_detail.as_deref(), Some("http 429"));
}

#[tokio::test]
async fn a_refused_connection_names_the_transport_fault_for_the_ledger() {
    // Bind then drop the listener so the port refuses the connect.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    drop(listener);
    let client = build_client(Duration::from_secs(2)).expect("client");
    let failure = open_stream(
        &client,
        &format!("http://{addr}/v1/chat/completions"),
        &HashMap::new(),
        "idem-refused",
        &serde_json::json!({"model": "m", "messages": []}),
        None,
        Duration::from_secs(5),
        Dialect::OpenAiCompatible,
    )
    .await
    .expect_err("a refused connect is a failure");
    assert_eq!(failure.failure_class, FailureClass::Transport);
    let detail = failure
        .provider_detail
        .as_deref()
        .expect("transport detail");
    assert!(detail.starts_with("open connect failed"), "{detail}");
    assert!(
        !detail.contains("127.0.0.1") && !detail.contains(&addr.port().to_string()),
        "the socket address never rides in the detail: {detail}"
    );
    assert_eq!(
        failure.public_error().message,
        "provider transport failed; retry the request"
    );
}

#[tokio::test]
async fn a_404_naming_a_missing_model_keeps_the_lane_policy() {
    let failure = open_against_body(
        "404 Not Found",
        "{\"error\":{\"message\":\"The model `x` does not exist or you do not have \
         access to it.\",\"type\":\"invalid_request_error\",\"param\":\"model\",\
         \"code\":\"model_not_found\"}}",
        "x",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::ProviderNotFound);
    assert!(failure.failover_eligible);
}

#[tokio::test]
async fn a_bodiless_404_keeps_the_lane_policy() {
    let failure = open_against_body("404 Not Found", "", "m").await;
    assert_eq!(failure.failure_class, FailureClass::ProviderNotFound);
    assert!(failure.failover_eligible);
}

#[tokio::test]
async fn a_blocked_by_sentence_without_a_refusal_code_stays_a_request_error() {
    let failure = open_against_body(
        "400 Bad Request",
        "{\"error\":{\"code\":\"invalid_value\",\"message\":\"Request blocked by the \
         organization policy for this parameter.\"}}",
        "m",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
}

#[tokio::test]
async fn a_content_filter_4xx_is_a_refusal_not_a_request_shape_error() {
    let failure = open_against_body(
        "400 Bad Request",
        "{\"error\":{\"code\":\"content_filter\",\"message\":\"The response was \
         filtered due to the prompt triggering the content management policy.\"}}",
        "m",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::Refusal);
    assert_eq!(failure.public_error().status_code, 400);
    assert_eq!(failure.public_error().code, "refusal");
    // The content_filter code names the content-policy category.
    assert_eq!(
        failure.refusal_reason,
        Some(crate::errors::RefusalReason::ContentPolicy)
    );
    // The sanitized sentence (or the code token when it must drop) rides
    // to the ledger; a refusal never relays it to the caller.
    assert!(failure.provider_detail.is_some());
    assert_eq!(
        failure.public_error().message,
        "provider refused the request: content policy"
    );
}

#[tokio::test]
async fn an_azure_completion_finished_content_filter_under_a_400_is_a_refusal() {
    // Captured live from Azure AI Foundry (DeepSeek-V4-Flash, 2026-09-15):
    // a 400 carrying a chat.completion body and no error envelope.
    let failure = open_against_body(
        "400 Bad Request",
        "{\"id\":\"chatcmpl-802d5a802bf84292896e446052595\",\"model\":\"\",\"choices\":\
         [{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"\"},\
         \"finish_reason\":\"content_filter\",\"content_filter_results\":{\"error\":\
         {\"code\":\"content_filter\",\"message\":\"Response content blocked by label \
         'MultiSeverity_ViolenceScore'.\"}}}],\"usage\":{\"prompt_tokens\":55,\
         \"total_tokens\":55},\"created\":1789466192,\"object\":\"chat.completion\",\
         \"prompt_filter_results\":null}",
        "DeepSeek-V4-Flash",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::Refusal);
    assert_eq!(
        failure.refusal_reason,
        Some(crate::errors::RefusalReason::ContentPolicy)
    );
    assert!(!failure.failover_eligible);
    assert!(!failure.retryable_same_deployment);
    // The quoted label is caller-visible vocabulary, so the sentence
    // survives the identifier screen into the ledger; the caller still
    // gets only the bounded refusal.
    assert_eq!(
        failure.provider_detail.as_deref(),
        Some("Response content blocked by label 'MultiSeverity_ViolenceScore'.")
    );
    assert_eq!(failure.public_error().status_code, 400);
    assert_eq!(failure.public_error().code, "refusal");
}

#[tokio::test]
async fn an_aggregator_routing_gate_403_is_not_a_credential_failure() {
    let failure = open_against_body(
        "403 Forbidden",
        "{\"error\":{\"message\":\"thinkingmachines/inkling:free is only available \
         on agentic harnesses.\",\"code\":403,\"metadata\":{\"routing_funnel\":\
         [{\"step\":\"Initial Endpoints\",\"endpoint_count\":1}],\
         \"failed_routing_step\":\"Gate Free Endpoints by Agentic Harness\"}}}",
        "thinkingmachines/inkling:free",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::ProviderNotFound);
    assert!(failure.failover_eligible);
    assert!(!failure.retryable_same_deployment);
    assert!(failure.provider_detail.is_none());
}

#[test]
fn gemini_relay_refusal_classifies_the_exact_403_envelope() {
    assert_eq!(
        crate::stream_errors::classify_stream_error(
            Some("403"),
            Some("Gemini blocked the request: PROHIBITED_CONTENT"),
        ),
        crate::stream_errors::StreamErrorKind::Refusal(crate::errors::RefusalReason::ContentPolicy),
    );
}

#[test]
fn gemini_relay_refusal_never_overrides_other_codes_or_incidental_text() {
    use crate::stream_errors::{classify_stream_error, StreamErrorKind};
    let message = Some("Gemini blocked the request: PROHIBITED_CONTENT");
    for (code, expected) in [
        ("401", StreamErrorKind::ProviderAuthentication),
        ("permission_denied", StreamErrorKind::ProviderAuthentication),
        (
            "authentication_error",
            StreamErrorKind::ProviderAuthentication,
        ),
        ("invalid_api_key", StreamErrorKind::ProviderAuthentication),
        ("402", StreamErrorKind::ProviderQuota),
        ("insufficient_quota", StreamErrorKind::ProviderQuota),
        ("billing_hard_limit_reached", StreamErrorKind::ProviderQuota),
        ("not_enough_balance", StreamErrorKind::ProviderQuota),
        ("429", StreamErrorKind::Throttled),
        ("rate_limit_exceeded", StreamErrorKind::Throttled),
        ("404", StreamErrorKind::ProviderNotFound),
    ] {
        assert_eq!(
            classify_stream_error(Some(code), message),
            expected,
            "{code}"
        );
    }
    for message in [
        "PROHIBITED_CONTENT",
        "Forbidden",
        "Gemini blocked the request: PROHIBITED_CONTENT_EXTRA",
        "Gemini blocked the request: PROHIBITED_CONTENT; authentication failed",
        "Gemini blocked the request: PROHIBITED_CONTENT due to billing",
        "Gemini blocked the request: PROHIBITED_CONTENT due to rate limit",
        "Prompt included Gemini blocked the request: PROHIBITED_CONTENT",
        "\"Gemini blocked the request: PROHIBITED_CONTENT\"",
        "Gemini blocked the request: UNKNOWN_REASON",
    ] {
        assert_eq!(
            classify_stream_error(Some("403"), Some(message)),
            StreamErrorKind::ProviderAuthentication,
            "{message}",
        );
    }
}

#[tokio::test]
async fn gemini_relay_refusal_http_403_keeps_the_bounded_public_contract() {
    for dialect in [Dialect::OpenAiCompatible, Dialect::OpenAiResponses] {
        let failure = open_dialect_against_body(
            "403 Forbidden",
            r#"{"error":{"code":403,"message":"Gemini blocked the request: PROHIBITED_CONTENT"}}"#,
            "google/gemini-test",
            dialect,
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::Refusal);
        assert_eq!(
            failure.refusal_reason,
            Some(crate::errors::RefusalReason::ContentPolicy)
        );
        assert!(!failure.retryable_same_deployment && !failure.failover_eligible);
        assert!(failure.rejected_parameter.is_none());
        assert_eq!(
            failure.provider_detail.as_deref(),
            Some("Gemini blocked the request: PROHIBITED_CONTENT"),
        );
        let public = failure.public_error();
        assert_eq!(public.status_code, 400);
        assert_eq!(public.code, "refusal");
        assert_eq!(public.error_type, "invalid_request_error");
        assert_eq!(
            public.json_body()["error"]["refusal_reason"],
            "content_policy"
        );
        assert_eq!(
            public.message,
            "provider refused the request: content policy"
        );
        assert!(!public
            .json_body()
            .to_string()
            .contains("PROHIBITED_CONTENT"));
    }
}

#[tokio::test]
async fn gemini_relay_refusal_http_does_not_read_quotes_echoes_or_conflicting_codes() {
    let message = "Gemini blocked the request: PROHIBITED_CONTENT";
    let mut cases = vec![
        (
            "401 Unauthorized",
            serde_json::json!({"error": {"code": 403, "message": message}}),
            FailureClass::ProviderAuthentication,
        ),
        (
            "402 Payment Required",
            serde_json::json!({"error": {"code": 403, "message": message}}),
            FailureClass::ProviderQuota,
        ),
        (
            "429 Too Many Requests",
            serde_json::json!({"error": {"code": 403, "message": message}}),
            FailureClass::Throttled,
        ),
        (
            "403 Forbidden",
            serde_json::json!({"error": {"code": "insufficient_quota", "message": message}}),
            FailureClass::ProviderQuota,
        ),
        (
            "403 Forbidden",
            serde_json::json!({"error": {"code": "authentication_error", "message": message}}),
            FailureClass::ProviderAuthentication,
        ),
        (
            "403 Forbidden",
            serde_json::json!({"error": {"code": "rate_limit_exceeded", "message": message}}),
            FailureClass::ProviderAuthentication,
        ),
        (
            "403 Forbidden",
            serde_json::json!({"error": {"code": 403, "message": "Forbidden", "echo": message}, "prompt": message}),
            FailureClass::ProviderAuthentication,
        ),
        (
            "403 Forbidden",
            serde_json::json!({"error": {"code": 403, "message": "Forbidden", "metadata": {"raw": message}}}),
            FailureClass::ProviderAuthentication,
        ),
    ];
    for incidental in [
        format!("Prompt included {message}"),
        format!("\"{message}\""),
        format!("{message}; authentication failed"),
        format!("{message} due to billing"),
        format!("{message} due to rate limit"),
        format!("{message}_EXTRA"),
    ] {
        cases.push((
            "403 Forbidden",
            serde_json::json!({"error": {"code": 403, "message": incidental}}),
            FailureClass::ProviderAuthentication,
        ));
    }
    for (status, body, expected) in cases {
        for dialect in [Dialect::OpenAiCompatible, Dialect::OpenAiResponses] {
            let failure =
                open_dialect_against_body(status, &body.to_string(), "google/gemini-test", dialect)
                    .await;
            assert_eq!(
                failure.failure_class, expected,
                "{dialect:?} {status} {body}"
            );
            assert!(failure.failover_eligible && !failure.retryable_same_deployment);
            assert!(failure.refusal_reason.is_none());
        }
    }
}

#[tokio::test]
async fn a_plain_403_stays_a_credential_failure() {
    let failure = open_against_body(
        "403 Forbidden",
        "{\"error\":{\"message\":\"Forbidden\",\"code\":403}}",
        "m",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::ProviderAuthentication);
    assert!(failure.failover_eligible);
}

#[tokio::test]
async fn a_lane_limitation_400_keeps_the_class_but_fails_over() {
    let failure = open_against_body(
        "400 Bad Request",
        "{\"error\":{\"message\":\"System message must be at the beginning.\",\
         \"type\":\"invalid_request_error\"}}",
        "qwen3.8-27b",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
    assert!(
        failure.failover_eligible,
        "another rung can carry the request"
    );
    assert!(!failure.retryable_same_deployment);
    assert!(failure
        .provider_detail
        .as_deref()
        .is_some_and(|detail| detail.contains("System message must be at the beginning")));
}

#[tokio::test]
async fn a_lane_limitation_phrase_echoed_outside_the_message_does_not_fail_over() {
    let failure = open_against_body(
        "400 Bad Request",
        "{\"error\":{\"message\":\"Invalid value for temperature.\",\
         \"type\":\"invalid_request_error\",\"param\":\"temperature\",\
         \"echo\":\"System message must be at the beginning\"}}",
        "m",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
    assert!(!failure.failover_eligible);
}

#[tokio::test]
async fn a_403_naming_a_step_without_a_walked_funnel_stays_a_credential_failure() {
    let failure = open_against_body(
        "403 Forbidden",
        "{\"error\":{\"message\":\"Forbidden\",\"code\":403,\
         \"metadata\":{\"failed_routing_step\":\"Authenticate\"}}}",
        "m",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::ProviderAuthentication);
}

#[tokio::test]
async fn an_ordinary_400_does_not_fail_over() {
    let failure = open_against_body(
        "400 Bad Request",
        "{\"error\":{\"message\":\"Invalid value for temperature.\",\
         \"type\":\"invalid_request_error\"}}",
        "m",
    )
    .await;
    assert_eq!(failure.failure_class, FailureClass::InvalidRequest);
    assert!(!failure.failover_eligible);
}

#[tokio::test]
async fn a_zai_429_is_quota_for_code_1113_and_a_throttle_otherwise() {
    // Z.ai: 429 + code 1113 is an empty balance (quota, fails over); 1302 a rate limit.
    let quota = open_against_body(
        "429 Too Many Requests",
        "{\"error\":{\"code\":\"1113\",\"message\":\"Insufficient balance or no \
         resource package. Please recharge.\"}}",
        "glm-4.6",
    )
    .await;
    assert_eq!(quota.failure_class, FailureClass::ProviderQuota);
    assert!(quota.failover_eligible && !quota.retryable_same_deployment);
    assert!(quota.provider_detail.is_none() && quota.rejected_parameter.is_none());
    let limit = open_against_body(
        "429 Too Many Requests",
        "{\"error\":{\"code\":\"1302\",\"message\":\"Rate limit reached for requests\"}}",
        "glm-4.6",
    )
    .await;
    assert_eq!(limit.failure_class, FailureClass::Throttled);
}

#[test]
fn header_phase_timeout_fails_over_without_a_same_deployment_redial() {
    // A lead that connects but never completes the response-header phase must
    // skip straight to the next rung (failover-eligible) instead of redialing
    // the same stalled deployment for another full header-timeout window.
    let failure = open_timeout_failure();
    assert_eq!(failure.failure_class, FailureClass::Timeout);
    assert!(
        !failure.retryable_same_deployment,
        "a header-phase stall must not redial the same deployment"
    );
    assert!(
        failure.failover_eligible,
        "a header-phase stall must fail over to the next certified rung"
    );
}
