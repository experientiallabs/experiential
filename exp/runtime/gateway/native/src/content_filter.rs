//! A provider's own content filter refusing a request or its output.
//!
//! Azure's content management system (and other account-side moderation
//! layers) can refuse content that another provider serving the same model
//! applies no filter to. That verdict is the provider's, not the model's: a
//! model declining in its own words (a `refusal` delta, refusal text) is
//! never one, and neither is the gateway's own guardrail class. Such a
//! refusal keeps its `refusal` class and caller sentence, but the waterfall
//! may take it ONCE per request to an untried rung of ANOTHER provider (the
//! same provider's filter would refuse again; python `native_content_filter`
//! chooses which rung), and only while nothing reached the caller.

use crate::errors::{Failure, FailureClass, RefusalReason};

/// Mark a typed provider content-policy refusal: never redialed (the same
/// rung's filter refuses the same content again). Eligibility to fail over is
/// granted by the waterfall (`fallback_rules::content_filter_successor`),
/// once per request, so the other ladders (images, embeddings, audio) keep
/// their answer unchanged. A failure that is not a content-policy refusal is
/// returned untouched.
pub(crate) fn mark(mut failure: Failure) -> Failure {
    if failure.failure_class == FailureClass::Refusal
        && failure.refusal_reason == Some(RefusalReason::ContentPolicy)
    {
        failure.provider_content_filter = true;
        failure.retryable_same_deployment = false;
    }
    failure
}

/// Whether one provider code names a content-policy verdict specifically
/// (`content_filter`, `safety`, ...), not a generic `refusal` whose category
/// would come from the sentence: only such a code marks a provider filter.
pub(crate) fn is_content_policy_code(code: Option<&str>) -> bool {
    code.is_some_and(|value| {
        crate::stream_errors::CONTENT_POLICY_CODES
            .contains(&value.trim().to_ascii_lowercase().as_str())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_content_policy_refusal_is_a_provider_filter() {
        let marked = mark(Failure::refusal(RefusalReason::ContentPolicy).with_retry(true, false));
        assert!(marked.provider_content_filter);
        assert!(!marked.retryable_same_deployment && !marked.failover_eligible);
        for reason in [
            RefusalReason::Unspecified,
            RefusalReason::CyberPolicy,
            RefusalReason::Cbrn,
            RefusalReason::Recitation,
        ] {
            assert!(!mark(Failure::refusal(reason)).provider_content_filter);
        }
        let guardrail = Failure::new(FailureClass::Guardrail, "blocked by guardrail");
        assert!(!mark(guardrail).provider_content_filter);
    }

    #[test]
    fn only_a_content_policy_code_is_a_typed_filter_verdict() {
        for code in ["content_filter", "SAFETY", "content_policy_violation"] {
            assert!(is_content_policy_code(Some(code)), "{code}");
        }
        // A generic refusal code takes its category from the sentence, so
        // it never marks a provider filter.
        for code in ["refusal", "cyber_policy", "rate_limit_exceeded"] {
            assert!(!is_content_policy_code(Some(code)), "{code}");
        }
        assert!(!is_content_policy_code(None));
    }
}
