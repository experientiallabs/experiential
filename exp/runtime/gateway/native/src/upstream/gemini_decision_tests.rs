//! Gemini decision rejections on the decisions surface: Google answers a
//! rejected API key under 400, which must fail over as a credential verdict.

use std::collections::HashMap;
use std::time::Duration;

use super::*;
#[tokio::test]
async fn gemini_decision_api_key_rejection_under_400_fails_over_as_a_credential_verdict() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    // (body, fails over, class)
    let invalid_key = r#"{"error":{"code":400,"message":"API key not valid. Please pass a valid API key.","status":"INVALID_ARGUMENT","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"API_KEY_INVALID","domain":"googleapis.com"}]}}"#;
    let bad_request = r#"{"error":{"code":400,"message":"Request contains an invalid argument.","status":"INVALID_ARGUMENT"}}"#;
    let table = [
        (invalid_key, true, FailureClass::ProviderAuthentication),
        (bad_request, false, FailureClass::InvalidRequest),
    ];
    for (body, failover, class) in table {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        let body = body.to_string();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept");
            let mut buffer = [0u8; 8192];
            let _ = socket.read(&mut buffer).await;
            let response = format!(
                "HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\n\
                 content-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body,
            );
            socket.write_all(response.as_bytes()).await.expect("write");
        });
        let client = build_client(Duration::from_secs(2), false).expect("client");
        let failure = open_decision_stream(
            &client,
            &format!("http://{addr}/v1beta/decisions"),
            &HashMap::new(),
            "idem-decision",
            &serde_json::json!({"model": "gemini-decision-1", "input": "x", "questions": {}}),
            Duration::from_secs(5),
            Dialect::GeminiGenerateContent,
        )
        .await
        .expect_err("a 400 must classify as a failure");
        assert!(failure.decision_provider_rejected);
        assert!(!failure.retryable_same_deployment);
        assert_eq!(failure.failover_eligible, failover);
        assert_eq!(failure.failure_class, class);
        if failover {
            assert_eq!(
                failure.provider_detail.as_deref(),
                Some("http 400: api_key_invalid")
            );
        }
    }
}
