//! A provider's context-window refusal, read off its 4xx error envelope.
//!
//! A context-length 400 keeps the caller's class and sentence, but a lane can
//! serve less than the window the model advertises, so the waterfall may take
//! it once to a rung whose window could fit (python `native_context_window`
//! chooses which). The verdict and the window the provider states ride the
//! failure across the bridge.

use crate::dialects::Dialect;
use crate::error_envelope::parse_error_document;
use crate::errors::Failure;
use crate::param_attribution::error_message_field;

/// Sentences a provider answers when the request is longer than the window
/// THIS lane serves. Lowercased substrings of the dialect's message field.
const CONTEXT_WINDOW_PHRASES: &[&str] = &[
    "maximum context length is",
    "maximum context length of",
    "exceeds the maximum context window",
    "context length exceeded",
    "exceeds the context window",
];

/// The phrase that precedes the window a provider states (vLLM, OpenAI).
const CONTEXT_WINDOW_LIMIT_PREFIXES: &[&str] =
    &["maximum context length is ", "maximum context length of "];

/// Whether a 4xx body is the provider's context-window verdict: its code is
/// `context_length_exceeded`, or its message names the window
/// ([`CONTEXT_WINDOW_PHRASES`]). `Some(limit)` carries the window the
/// sentence states (`None` when it names none). Only the dialect's message
/// field is read, so request text echoed elsewhere cannot change routing.
pub(crate) fn rejected_context_window(
    dialect: Dialect,
    code: Option<&str>,
    body: &str,
) -> Option<Option<u32>> {
    let message = parse_error_document(body)
        .and_then(|value| error_message_field(dialect, &value).map(str::to_ascii_lowercase));
    let named = message.as_deref().is_some_and(|message| {
        CONTEXT_WINDOW_PHRASES
            .iter()
            .any(|phrase| message.contains(phrase))
    });
    if !named && code != Some("context_length_exceeded") {
        return None;
    }
    Some(message.as_deref().and_then(stated_context_window))
}

/// The window a context-length sentence states ("maximum context length is
/// 16384 tokens"), digits and thousands separators only.
fn stated_context_window(message: &str) -> Option<u32> {
    CONTEXT_WINDOW_LIMIT_PREFIXES.iter().find_map(|prefix| {
        let start = message.find(prefix)? + prefix.len();
        let digits: String = message[start..]
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == ',')
            .filter(char::is_ascii_digit)
            .collect();
        digits.parse::<u32>().ok().filter(|limit| *limit > 0)
    })
}

/// Mark `failure` as a context-window refusal: failover-eligible, never
/// redialed (the same rung refuses the same request again).
pub(crate) fn mark(mut failure: Failure, limit_tokens: Option<u32>) -> Failure {
    failure.context_window_rejected = true;
    failure.context_window_limit_tokens = limit_tokens;
    failure.failover_eligible = true;
    failure.retryable_same_deployment = false;
    failure
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_context_window_verdict_names_the_window_the_lane_serves() {
        // vLLM's context-length 400.
        let vllm = r#"{"object":"error","message":"This model's maximum context length is 16384 tokens. However, you requested 8192 output tokens and your prompt contains at least 8193 input tokens, for a total of at least 16385 tokens.","type":"BadRequestError","param":null,"code":400}"#;
        assert_eq!(
            rejected_context_window(Dialect::OpenAiCompatible, None, vllm),
            Some(Some(16_384))
        );
        let openai = r#"{"error":{"message":"This model's maximum context length is 128,000 tokens.","type":"invalid_request_error","param":"messages","code":"context_length_exceeded"}}"#;
        assert_eq!(
            rejected_context_window(
                Dialect::OpenAiCompatible,
                Some("context_length_exceeded"),
                openai
            ),
            Some(Some(128_000))
        );
        // The code alone decides; a sentence with no figure states no window.
        let bare = r#"{"error":{"message":"Input exceeds the maximum context window.","type":"invalid_request_error"}}"#;
        assert_eq!(
            rejected_context_window(Dialect::OpenAiCompatible, None, bare),
            Some(None)
        );
        assert_eq!(
            rejected_context_window(
                Dialect::OpenAiCompatible,
                Some("context_length_exceeded"),
                "{}"
            ),
            Some(None)
        );
        // An ordinary caller error is not one.
        let other = r#"{"error":{"message":"temperature must be at most 2","type":"invalid_request_error","param":"temperature"}}"#;
        assert_eq!(
            rejected_context_window(Dialect::OpenAiCompatible, None, other),
            None
        );
    }
}
