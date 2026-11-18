//! A provider's block on this request's END USER for an earlier policy violation.
//!
//! OpenAI answers "Policy Violation: this user has been blocked for a previous
//! policy violation" once it has blocked the end-user identifier a request
//! carries (OpenRouter forwards it and relays the verdict as a flat 403, as a
//! generic 400 `invalid_prompt` with OpenAI's document in `metadata.raw`, or
//! in-stream). It is a verdict on that one end user, never on the account
//! credential: the same house account kept serving every other caller while
//! one organization's requests drew this answer (2026-11-18, 510 successes
//! beside the blocks). Filing it as `provider_authentication` let one blocked
//! end user feed the account-health and BYOK credential breakers and told the
//! caller authentication failed. It is a typed content-policy refusal instead:
//! never redialed, never a credential verdict, and (as a provider filter) it
//! may take one untried rung of another provider, which holds no such block.

use crate::errors::{Failure, RefusalReason};

/// The durable operator-facing sentence; `caller_copy` words it for callers.
pub(crate) const END_USER_BLOCK_MESSAGE: &str =
    "provider blocked the request's end user for a previous policy violation";

/// The ledger token appended to the status detail (`http 403: <token>`).
const END_USER_BLOCK_DETAIL: &str = "end_user_policy_block";

/// Whether one provider text names an end-user policy block.
///
/// Reads the raw text case-insensitively, so it also finds the sentence inside
/// an aggregator's JSON-string `metadata.raw` envelope. Both stable fragments
/// must appear; the full sentence is not required. The block must name the
/// USER: an "account (or organization) has been blocked" sentence is a verdict
/// on the credential and keeps the credential class.
pub(crate) fn names_end_user_policy_block(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("user has been blocked") && lower.contains("policy violation")
}

/// Whether any present text (an envelope message, its raw relay) names one.
pub(crate) fn names_any_end_user_policy_block(texts: &[Option<&str>]) -> bool {
    texts
        .iter()
        .flatten()
        .any(|text| names_end_user_policy_block(text))
}

/// The refusal for an end-user policy block, with `detail` as its ledger line.
pub(crate) fn end_user_block_failure(detail: Option<String>) -> Failure {
    let mut failure = Failure::refusal(RefusalReason::ContentPolicy);
    failure.safe_message = END_USER_BLOCK_MESSAGE.to_string();
    crate::content_filter::mark(failure).with_provider_detail(detail)
}

/// The ledger detail for a block answered under HTTP `status_detail`.
pub(crate) fn end_user_block_detail(status_detail: &str) -> String {
    format!("{status_detail}: {END_USER_BLOCK_DETAIL}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::FailureClass;

    const PRODUCTION_403: &str = r#"{"error":{"code":403,"message":"Policy Violation: this user has been blocked for a previous policy violation. Learn more: https://openrouter.ai/docs"}}"#;

    #[test]
    fn the_production_body_names_an_end_user_block() {
        assert!(names_end_user_policy_block(PRODUCTION_403));
        assert!(names_end_user_policy_block(
            "POLICY VIOLATION: THIS USER HAS BEEN BLOCKED"
        ));
    }

    #[test]
    fn one_fragment_alone_is_not_a_block() {
        for text in [
            r#"{"error":{"message":"Invalid API key","code":403}}"#,
            r#"{"error":{"message":"Your account has been blocked","code":403}}"#,
            r#"{"error":{"message":"This account has been blocked for a policy violation"}}"#,
            r#"{"error":{"message":"This user has been blocked","code":403}}"#,
            r#"{"error":{"message":"Invalid prompt: policy violation","code":"invalid_prompt"}}"#,
        ] {
            assert!(!names_end_user_policy_block(text), "{text}");
        }
    }

    #[test]
    fn the_failure_is_a_content_policy_refusal_not_a_credential_verdict() {
        let failure = end_user_block_failure(Some(end_user_block_detail("http 403")));
        assert_eq!(failure.failure_class, FailureClass::Refusal);
        assert_eq!(failure.refusal_reason, Some(RefusalReason::ContentPolicy));
        assert!(failure.provider_content_filter);
        assert!(!failure.retryable_same_deployment && !failure.failover_eligible);
        assert_eq!(
            failure.provider_detail.as_deref(),
            Some("http 403: end_user_policy_block")
        );
        let public = failure.public_error();
        assert_eq!(public.status_code, 400);
        assert!(public.message.contains("end user"), "{}", public.message);
        assert!(public.message.contains("policy violation"));
        assert!(!public
            .message
            .to_ascii_lowercase()
            .contains("authentication"));
    }
}
