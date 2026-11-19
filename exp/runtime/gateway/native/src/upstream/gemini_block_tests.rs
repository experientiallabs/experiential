//! A relay's HTTP 502 naming Gemini's output block is the content's refusal;
//! any other 502 keeps the provider-internal class.

use super::tests::open_dialect_against_body;
use super::*;
use crate::errors::RefusalReason;

/// The exact production body (OpenRouter, gemini-4-neon, 2026-11-19).
const OTHER_BLOCK: &str =
    r#"{"error":{"code":502,"message":"Gemini blocked the response: OTHER"}}"#;

#[tokio::test]
async fn a_relayed_gemini_response_block_under_502_is_a_refusal() {
    for dialect in [Dialect::OpenAiCompatible, Dialect::OpenAiResponses] {
        let failure = open_dialect_against_body(
            "502 Bad Gateway",
            OTHER_BLOCK,
            "google/gemini-test",
            dialect,
        )
        .await;
        assert_eq!(failure.failure_class, FailureClass::Refusal, "{dialect:?}");
        assert_eq!(failure.refusal_reason, Some(RefusalReason::Unspecified));
        assert!(!failure.retryable_same_deployment && !failure.failover_eligible);
        assert!(!failure.provider_content_filter);
        assert_eq!(
            failure.provider_detail.as_deref(),
            Some("Gemini blocked the response: OTHER")
        );
        let public = failure.public_error();
        assert_eq!(public.status_code, 400);
        assert_eq!(public.code, "refusal");
        assert_eq!(public.json_body()["error"]["refusal_reason"], "unspecified");
        assert!(!public.json_body().to_string().contains("OTHER"));
    }
    let safety = OTHER_BLOCK.replace("OTHER", "SAFETY");
    let failure = open_dialect_against_body(
        "502 Bad Gateway",
        &safety,
        "google/gemini-test",
        Dialect::OpenAiCompatible,
    )
    .await;
    assert_eq!(failure.refusal_reason, Some(RefusalReason::ContentPolicy));
    assert!(failure.provider_content_filter);
}

#[tokio::test]
async fn any_other_502_stays_a_provider_failure() {
    let cases = [
        (
            r#"{"error":{"code":502,"message":"Bad Gateway"}}"#,
            Dialect::OpenAiCompatible,
        ),
        ("502 Bad Gateway", Dialect::OpenAiCompatible),
        (
            r#"{"error":{"code":502,"message":"Bad Gateway","metadata":{"raw":"Gemini blocked the response: OTHER"}}}"#,
            Dialect::OpenAiCompatible,
        ),
        (
            r#"{"error":{"code":502,"message":"Gemini blocked the response: UNKNOWN"}}"#,
            Dialect::OpenAiCompatible,
        ),
        // Only an OpenAI-family relay's body is read under 502.
        (OTHER_BLOCK, Dialect::AnthropicMessages),
    ];
    for (body, dialect) in cases {
        let failure =
            open_dialect_against_body("502 Bad Gateway", body, "google/gemini-test", dialect).await;
        assert_eq!(
            failure.failure_class,
            FailureClass::ProviderInternal,
            "{body}"
        );
        assert!(failure.failover_eligible, "{body}");
        assert_eq!(
            failure.provider_detail.as_deref(),
            Some("http 502"),
            "{body}"
        );
    }
}
