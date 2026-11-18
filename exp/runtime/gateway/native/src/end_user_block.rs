//! A provider's block on this request's END USER for an earlier policy violation.
//!
//! OpenAI answers "Policy Violation: this user has been blocked for a previous
//! policy violation" once it has blocked the end-user identifier a request
//! carries (OpenRouter forwards it and relays the verdict as a flat 403, as a
//! generic 400 `invalid_prompt` with OpenAI's document in `metadata.raw`, or
//! in-stream). It is a verdict on that one end user, never on the account
//! credential: the same account keeps serving every other end user. Filing it
//! as `provider_authentication` let one blocked end user feed the account
//! health and BYOK credential breakers and told the caller authentication
//! failed. It is a typed content-policy refusal instead: never redialed,
//! never a credential verdict, and (as a provider filter) it may take one
//! untried rung of another provider, which holds no such block.

use crate::errors::{Failure, RefusalReason};

/// The durable operator-facing sentence; `caller_copy` words it for callers.
pub(crate) const END_USER_BLOCK_MESSAGE: &str =
    "provider blocked the request's end user for a previous policy violation";

/// The ledger token appended to the status detail (`http 403: <token>`).
const END_USER_BLOCK_DETAIL: &str = "end_user_policy_block";

/// How far into the verdict sentence "policy violation" may start: room for
/// an aggregator's "<provider name>: " prefix, never for quoted input.
const VERDICT_HEAD_WINDOW: usize = 48;

/// How far into the verdict sentence "user has been blocked" must start.
const BLOCK_CLAUSE_WINDOW: usize = 120;

/// Whether one provider ERROR SENTENCE is an end-user policy block.
///
/// Both stable fragments must appear (case-insensitive), the full sentence is
/// not required, and both must lead the sentence: "policy violation" within
/// [`VERDICT_HEAD_WINDOW`] characters (allowing a relay's provider-name
/// prefix) and "user has been blocked" within [`BLOCK_CLAUSE_WINDOW`], so a
/// sentence that merely quotes the words is not one. The block must name the
/// USER: an "account has been blocked" sentence is a verdict on the
/// credential and keeps the credential class.
pub(crate) fn names_end_user_policy_block(sentence: &str) -> bool {
    let lower = sentence.trim().to_ascii_lowercase();
    let head = lower.find("policy violation");
    let clause = lower.find("user has been blocked");
    matches!(
        (head, clause),
        (Some(head), Some(clause)) if head <= VERDICT_HEAD_WINDOW && clause <= BLOCK_CLAUSE_WINDOW
    )
}

/// Whether any present sentence (an envelope message, its raw relay) is one.
pub(crate) fn names_any_end_user_policy_block(sentences: &[Option<&str>]) -> bool {
    sentences
        .iter()
        .flatten()
        .any(|sentence| names_end_user_policy_block(sentence))
}

/// Whether one pre-stream error BODY carries an end-user policy block in its
/// verdict fields only: the envelope's own message, or the upstream sentence
/// an aggregator relayed in `metadata.raw` behind its generic message. Any
/// other field (an echoed prompt, a diagnostic) is never read.
pub(crate) fn body_names_end_user_policy_block(body: &str) -> bool {
    let Some(document) = crate::error_envelope::parse_error_document(body) else {
        return false;
    };
    let Some(envelope) = crate::error_envelope::openai_family_envelope(&document) else {
        return false;
    };
    let relayed = envelope
        .error_object
        .zip(envelope.message)
        .and_then(|(error, message)| {
            crate::rejection_shapes::upstream_relayed_message(error, message)
        });
    names_any_end_user_policy_block(&[envelope.message, relayed.as_deref()])
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

    const FLAT_403: &str = r#"{"error":{"code":403,"message":"Policy Violation: this user has been blocked for a previous policy violation. Learn more: https://openrouter.ai/docs"}}"#;

    const RELAYED_400: &str = r#"{"error":{"message":"Provider returned error","code":400,"metadata":{"raw":"{\"error\":{\"message\":\"Policy Violation: this user has been blocked for a previous policy violation\",\"code\":\"invalid_prompt\"}}","provider_name":"OpenAI"}}}"#;

    #[test]
    fn the_observed_bodies_name_an_end_user_block() {
        assert!(body_names_end_user_policy_block(FLAT_403));
        assert!(body_names_end_user_policy_block(RELAYED_400));
        assert!(names_end_user_policy_block(
            "POLICY VIOLATION: THIS USER HAS BEEN BLOCKED"
        ));
        // A relay's provider-name prefix in-stream.
        assert!(names_end_user_policy_block(
            "OpenAI: Policy Violation: this user has been blocked for a previous policy violation"
        ));
    }

    #[test]
    fn the_fragments_outside_the_verdict_fields_are_not_a_block() {
        // An echoed prompt beside a credential verdict.
        let echoed = r#"{"error":{"message":"Invalid API key","code":403,"prompt":"Policy Violation: this user has been blocked for a previous policy violation"}}"#;
        assert!(!body_names_end_user_policy_block(echoed));
        // A sentence quoting the words after its own verdict.
        assert!(!names_end_user_policy_block(
            "Invalid API key provided. Your request input was: 'please explain why a policy violation means this user has been blocked'"
        ));
        assert!(!body_names_end_user_policy_block(
            "not json: this user has been blocked for a policy violation"
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
            assert!(!body_names_end_user_policy_block(text), "{text}");
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
