//! Google account verdicts read off Gemini-wire error bodies.

use super::*;
use serde_json::json;

fn body(status: &str, code: u16, details: Value) -> String {
    json!({"error": {"code": code, "status": status, "message": "x", "details": details}})
        .to_string()
}

fn error_info(reason: &str, metadata: Value) -> Value {
    json!({
        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
        "reason": reason,
        "domain": "cloudcode-pa.googleapis.com",
        "metadata": metadata,
    })
}

const GEMINI: Dialect = Dialect::GeminiGenerateContent;

#[test]
fn an_exhausted_plan_quota_is_the_accounts_funding_state_with_its_reset() {
    for reason in ["QUOTA_EXHAUSTED", "INSUFFICIENT_G1_CREDITS_BALANCE"] {
        let text = body(
            "RESOURCE_EXHAUSTED",
            429,
            json!([
                error_info(reason, json!({"quotaResetDelay": "3h2m0.5s"})),
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "10921.5s"},
            ]),
        );
        let verdict = google_account_verdict(GEMINI, 429, Some(&text)).unwrap();
        assert_eq!(
            verdict,
            GoogleAccountVerdict::QuotaExhausted {
                reason: reason.to_string(),
                // RetryInfo wins over the ErrorInfo metadata, rounded up.
                reset_seconds: Some(10_922),
            }
        );
        let failure = verdict.failure("http 429");
        assert_eq!(failure.failure_class, FailureClass::ProviderQuota);
        assert!(!failure.retryable_same_deployment);
        assert!(failure.failover_eligible);
        assert_eq!(failure.retry_after_seconds, Some(10_922));
        assert_eq!(
            failure.provider_detail.as_deref(),
            Some(format!("http 429: {reason}").as_str())
        );
        assert!(!failure.account_suspended);
    }
}

#[test]
fn the_quota_reset_falls_back_to_the_error_info_metadata() {
    let text = body(
        "RESOURCE_EXHAUSTED",
        429,
        json!([error_info(
            "QUOTA_EXHAUSTED",
            json!({"quotaResetDelay": "373.8ms"})
        )]),
    );
    match google_account_verdict(GEMINI, 429, Some(&text)) {
        Some(GoogleAccountVerdict::QuotaExhausted { reset_seconds, .. }) => {
            assert_eq!(reset_seconds, Some(1));
        }
        other => panic!("unexpected verdict {other:?}"),
    }
}

#[test]
fn a_rate_limit_is_a_throttle_not_an_account_verdict_and_states_its_wait() {
    let text = body(
        "RESOURCE_EXHAUSTED",
        429,
        json!([
            error_info("RATE_LIMIT_EXCEEDED", json!({})),
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "2.1s"},
        ]),
    );
    assert_eq!(google_account_verdict(GEMINI, 429, Some(&text)), None);
    assert_eq!(google_body_retry_seconds(GEMINI, Some(&text)), Some(3));
}

#[test]
fn a_terms_of_service_suspension_is_a_marked_credential_failure_with_its_appeal() {
    let appeal = "https://support.google.com/appeal?form=antigravity";
    let text = body(
        "PERMISSION_DENIED",
        403,
        json!([error_info(
            "TOS_VIOLATION",
            json!({"appeal_url": appeal, "appeal_url_link_text": "Appeal"})
        )]),
    );
    let verdict = google_account_verdict(GEMINI, 403, Some(&text)).unwrap();
    assert_eq!(
        verdict,
        GoogleAccountVerdict::Suspended {
            appeal_url: Some(appeal.to_string())
        }
    );
    let failure = verdict.failure("http 403");
    assert_eq!(failure.failure_class, FailureClass::ProviderAuthentication);
    assert!(!failure.retryable_same_deployment);
    assert!(failure.failover_eligible);
    assert!(failure.account_suspended);
    assert_eq!(appeal_url_of(&failure), Some(appeal));
    assert_eq!(
        failure.provider_detail.as_deref(),
        Some(format!("http 403: TOS_VIOLATION; appeal_url={appeal}").as_str())
    );
    // On a customer's own plan connection the 400 names the suspension and the appeal.
    let owned = crate::stream_errors::customer_credential_failure(failure, "gemini");
    let public = owned.public_error();
    assert_eq!(public.status_code, 400);
    assert_eq!(public.code, "provider_account_suspended");
    assert!(public.message.contains(appeal), "{}", public.message);
}

#[test]
fn an_unusable_appeal_url_is_dropped_but_the_suspension_stands() {
    let long = format!(
        "https://example.com/{}",
        "a".repeat(MAXIMUM_APPEAL_URL_CHARS)
    );
    for url in [
        "http://insecure.example/appeal",
        "https://a b",
        "",
        long.as_str(),
    ] {
        let text = body(
            "PERMISSION_DENIED",
            403,
            json!([error_info("TOS_VIOLATION", json!({"appeal_url": url}))]),
        );
        assert_eq!(
            google_account_verdict(GEMINI, 403, Some(&text)),
            Some(GoogleAccountVerdict::Suspended { appeal_url: None })
        );
    }
}

#[test]
fn reasons_count_only_under_their_documented_status_and_wire() {
    let suspended = body(
        "PERMISSION_DENIED",
        403,
        json!([error_info("TOS_VIOLATION", json!({}))]),
    );
    let quota = body(
        "RESOURCE_EXHAUSTED",
        429,
        json!([error_info("QUOTA_EXHAUSTED", json!({}))]),
    );
    assert_eq!(google_account_verdict(GEMINI, 400, Some(&suspended)), None);
    assert_eq!(google_account_verdict(GEMINI, 403, Some(&quota)), None);
    assert_eq!(
        google_account_verdict(Dialect::OpenAiCompatible, 403, Some(&suspended)),
        None
    );
    assert_eq!(google_account_verdict(GEMINI, 403, None), None);
    assert_eq!(google_account_verdict(GEMINI, 403, Some("not json")), None);
    // A streaming endpoint's one-element array envelope reads the same.
    let wrapped = format!("[{suspended}]");
    assert!(google_account_verdict(GEMINI, 403, Some(&wrapped)).is_some());
}

#[test]
fn durations_parse_google_and_go_spellings() {
    assert_eq!(duration_seconds("0.847655010s"), Some(0.847655010));
    assert!((duration_seconds("373.8ms").unwrap() - 0.3738).abs() < 1e-9);
    assert_eq!(duration_seconds("1h2m3s"), Some(3_723.0));
    assert_eq!(duration_seconds("5x"), None);
    assert_eq!(duration_seconds(""), None);
    assert_eq!(duration_seconds("s"), None);
}

#[test]
fn the_suspension_crosses_the_bridge_under_the_python_field_name() {
    let failure = GoogleAccountVerdict::Suspended {
        appeal_url: Some("https://g.co/appeal".to_string()),
    }
    .failure("http 403");
    let wire = serde_json::to_value(&failure).unwrap();
    assert_eq!(wire["provider_account_suspended"], json!(true));
    let echoed: Failure = serde_json::from_value(wire).unwrap();
    assert!(echoed.account_suspended);
    assert_eq!(appeal_url_of(&echoed), Some("https://g.co/appeal"));
}
