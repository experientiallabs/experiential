//! A request whose model kept producing a broken function call.
//!
//! The model, not the provider, failed: Gemini finished the turn with
//! `MALFORMED_FUNCTION_CALL` (`malformed_function_call`), or a streamed client
//! tool call never became one valid call (`malformed_tool_call`). The ladder
//! still redials and fails over, because a fresh sample often parses. Once
//! it is exhausted after at least two such failures, though, the same conversation tends to
//! fail the same way on every sample, and the generic 502 `all_routes_failed`
//! is an answer every OpenAI and Anthropic SDK retries automatically (both
//! retry 408/409/429/5xx), so agent clients re-sent the identical long
//! conversation dozens of times. The caller instead receives a 400
//! `invalid_request_error`, the shape `empty_completion` uses, which no SDK
//! retries, with a sentence that names the remedy.
//!
//! The ledger keeps `malformed_response`: only the public answer changes.

use super::{Failure, FailureClass};

/// Public code of the exhausted broken-tool-call answer.
pub const BROKEN_TOOL_CALL_CODE: &str = "malformed_tool_call";

/// Caller sentence of the exhausted broken-tool-call answer.
pub const BROKEN_TOOL_CALL_MESSAGE: &str =
    "The model produced an invalid tool call for this conversation repeatedly. \
     Retrying the same request will likely fail again; shorten or restructure \
     the conversation, or try another model.";

impl Failure {
    /// Whether this failure is the model producing a broken function call.
    pub fn broken_tool_call(&self) -> bool {
        self.failure_class == FailureClass::MalformedResponse
            && (self.malformed_tool_call || self.malformed_function_call)
    }

    /// This failure without its broken-tool-call facts: the ladder saw only
    /// one broken call (no redial or failover was possible), so a client
    /// retry is still the conversation's next fresh sample.
    pub fn without_broken_tool_call_facts(mut self) -> Self {
        self.malformed_tool_call = false;
        self.malformed_function_call = false;
        self
    }

    /// The control plane's echo of the failure that exhausted the ladder,
    /// with the data plane's own broken-tool-call facts restored.
    ///
    /// The echo carries only the fields the control plane models, and it may
    /// replace the failure altogether (an unavailable route, a quota), so the
    /// facts transfer only when the echo is still the same malformed answer
    /// the data plane last saw.
    pub fn with_broken_tool_call_facts(mut self, last: Option<&Failure>) -> Self {
        if let Some(last) = last.filter(|last| last.broken_tool_call()) {
            if self.failure_class == FailureClass::MalformedResponse {
                self.malformed_tool_call |= last.malformed_tool_call;
                self.malformed_function_call |= last.malformed_function_call;
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gemini_malformed() -> Failure {
        let mut failure = Failure::new(
            FailureClass::MalformedResponse,
            "provider ended the stream with a malformed answer",
        )
        .with_retry(true, true);
        failure.validated_tool_redial = true;
        failure.malformed_function_call = true;
        failure
    }

    #[test]
    fn an_exhausted_broken_tool_call_is_a_400_no_sdk_retries() {
        for failure in [
            gemini_malformed(),
            Failure::new(FailureClass::MalformedResponse, "tool call never parsed")
                .with_malformed_tool_call(true),
        ] {
            let error = failure.boundary().public_error();
            assert_eq!(error.status_code, 400);
            assert_eq!(error.code, BROKEN_TOOL_CALL_CODE);
            assert_eq!(error.error_type, "invalid_request_error");
            assert_eq!(error.message, BROKEN_TOOL_CALL_MESSAGE);
            assert_eq!(error.retry_after_seconds, None);
        }
        assert!(!BROKEN_TOOL_CALL_MESSAGE.contains('\u{2014}'));
        assert!(!BROKEN_TOOL_CALL_MESSAGE.contains('\u{2013}'));
    }

    #[test]
    fn a_malformed_stream_that_is_not_a_tool_call_keeps_the_retryable_502() {
        let error = Failure::new(FailureClass::MalformedResponse, "bad frame")
            .boundary()
            .public_error();
        assert_eq!(error.status_code, 502);
        assert_eq!(error.code, "all_routes_failed");
        // A Gemini MALFORMED_RESPONSE finish redials validated but names no call.
        let mut answer = gemini_malformed();
        answer.malformed_function_call = false;
        assert_eq!(answer.boundary().public_error().status_code, 502);
    }

    #[test]
    fn the_boundary_keeps_the_gemini_fact_and_the_ledger_class() {
        let bounded = gemini_malformed().boundary();
        assert!(bounded.validated_tool_redial && bounded.malformed_function_call);
        assert_eq!(bounded.failure_class, FailureClass::MalformedResponse);
    }

    #[test]
    fn the_echo_regains_the_facts_only_while_it_is_the_same_malformed_answer() {
        let last = gemini_malformed();
        // The control plane's echo models no broken-tool-call fact.
        let echoed = Failure::new(
            FailureClass::MalformedResponse,
            "provider ended the stream with a malformed answer",
        );
        assert!(!echoed.broken_tool_call());
        assert!(echoed
            .clone()
            .with_broken_tool_call_facts(Some(&last))
            .broken_tool_call());
        // A replaced answer (the whole route unavailable) stays itself.
        let replaced = Failure::new(
            FailureClass::ProviderInternal,
            "all exact-model deployments are unavailable",
        )
        .with_broken_tool_call_facts(Some(&last));
        assert!(!replaced.malformed_function_call);
        assert_eq!(replaced.public_error().status_code, 502);
        // No broken call last: nothing to restore.
        let plain = Failure::new(FailureClass::MalformedResponse, "bad frame");
        assert!(!echoed
            .with_broken_tool_call_facts(Some(&plain))
            .broken_tool_call());
    }
}
