//! A relay (OpenRouter) passing on Gemini's own block verdict.
//!
//! OpenRouter forwards a Gemini prompt block as `403 "Gemini blocked the
//! request: <blockReason>"` and an output block as `502 "Gemini blocked the
//! response: <finishReason>"`, in the HTTP envelope or in-stream. Both are
//! Google's content verdict, not a credential failure and not a provider
//! outage: filed as `provider_internal`, one caller's content made a healthy
//! lane read as dead to lane health and could demote it for every other
//! organization. The category follows the native Gemini lane
//! (`dialects/gemini.rs`), which reads the same token through
//! `refusal_reason`, so one verdict is filed alike on either lane.
//!
//! The match is exact on the envelope's own message (never a sanitized,
//! relayed, or echoed sentence) and on the status the relay uses for that
//! verdict, so prompt text quoting the sentence, a longer sentence, or an
//! unknown token keeps its status class.

use serde_json::Value;

use crate::errors::RefusalReason;

/// The verdict sentences, each with the only status code the relay sends it under.
const VERDICTS: &[(&str, &str)] = &[
    ("403", "Gemini blocked the request: "),
    ("502", "Gemini blocked the response: "),
];

/// Gemini block and finish tokens the relay names: `BlockedReason` and every
/// blocking `FinishReason` the native lane files as a refusal
/// (`dialects/gemini.rs`).
const BLOCK_TOKENS: &[&str] = &[
    "OTHER",
    "PROHIBITED_CONTENT",
    "SAFETY",
    "BLOCKLIST",
    "SPII",
    "RECITATION",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_RECITATION",
];

/// The refusal a relayed Gemini block names, or `None` when the code and raw
/// envelope message are not exactly one verdict sentence.
pub(crate) fn relayed_gemini_refusal(
    code: Option<&str>,
    message: Option<&str>,
) -> Option<RefusalReason> {
    let code = code?.trim();
    let message = message?;
    let token = VERDICTS.iter().find_map(|(status, prefix)| {
        (code == *status)
            .then(|| message.strip_prefix(prefix))
            .flatten()
    })?;
    BLOCK_TOKENS
        .contains(&token)
        .then(|| crate::stream_errors::refusal_reason(Some(token), None))
}

/// The refusal and raw verdict sentence a relay's HTTP error body names, read
/// from the OpenAI-family envelope only.
pub(crate) fn relayed_gemini_refusal_body(body: &str) -> Option<(RefusalReason, String)> {
    let value: Value = crate::error_envelope::parse_error_document(body)?;
    let envelope = crate::error_envelope::openai_family_envelope(&value)?;
    let message = envelope.message?;
    let reason = relayed_gemini_refusal(envelope.code.as_deref(), Some(message))?;
    Some((reason, message.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_verdict_takes_the_native_lane_category() {
        for (code, message, reason) in [
            (
                "502",
                "Gemini blocked the response: OTHER",
                RefusalReason::Unspecified,
            ),
            (
                "502",
                "Gemini blocked the response: SAFETY",
                RefusalReason::ContentPolicy,
            ),
            (
                "502",
                "Gemini blocked the response: PROHIBITED_CONTENT",
                RefusalReason::ContentPolicy,
            ),
            (
                "502",
                "Gemini blocked the response: BLOCKLIST",
                RefusalReason::ContentPolicy,
            ),
            (
                "502",
                "Gemini blocked the response: IMAGE_SAFETY",
                RefusalReason::ContentPolicy,
            ),
            (
                "502",
                "Gemini blocked the response: RECITATION",
                RefusalReason::Recitation,
            ),
            (
                "502",
                "Gemini blocked the response: SPII",
                RefusalReason::DataInspection,
            ),
            (
                "502",
                "Gemini blocked the response: IMAGE_RECITATION",
                RefusalReason::Recitation,
            ),
            (
                "502",
                "Gemini blocked the response: IMAGE_PROHIBITED_CONTENT",
                crate::stream_errors::refusal_reason(Some("IMAGE_PROHIBITED_CONTENT"), None),
            ),
            (
                "403",
                "Gemini blocked the request: PROHIBITED_CONTENT",
                RefusalReason::ContentPolicy,
            ),
            (
                " 403 ",
                "Gemini blocked the request: OTHER",
                RefusalReason::Unspecified,
            ),
        ] {
            assert_eq!(
                relayed_gemini_refusal(Some(code), Some(message)),
                Some(reason),
                "{code} {message}"
            );
            // The native lane files the same token under the same category.
            let token = message.rsplit(' ').next().unwrap();
            assert_eq!(
                crate::stream_errors::refusal_reason(Some(token), None),
                reason
            );
        }
    }

    #[test]
    fn only_the_exact_sentence_under_its_own_status_is_a_verdict() {
        for (code, message) in [
            (Some("502"), Some("Bad Gateway")),
            (Some("502"), Some("Gemini blocked the response: UNKNOWN")),
            (Some("502"), Some("Gemini blocked the response: OTHER.")),
            (
                Some("502"),
                Some("Gemini blocked the response: OTHER; retry"),
            ),
            (Some("502"), Some("Gemini blocked the response:OTHER")),
            (Some("502"), Some("gemini blocked the response: OTHER")),
            (
                Some("502"),
                Some("Upstream said Gemini blocked the response: OTHER"),
            ),
            (Some("502"), Some("\"Gemini blocked the response: OTHER\"")),
            (Some("502"), Some("Gemini blocked the request: OTHER")),
            (Some("403"), Some("Gemini blocked the response: OTHER")),
            (Some("500"), Some("Gemini blocked the response: OTHER")),
            (
                Some("401"),
                Some("Gemini blocked the request: PROHIBITED_CONTENT"),
            ),
            (None, Some("Gemini blocked the response: OTHER")),
            (Some("502"), None),
        ] {
            assert_eq!(
                relayed_gemini_refusal(code, message),
                None,
                "{code:?} {message:?}"
            );
        }
    }

    #[test]
    fn the_body_reader_takes_only_the_envelope_message() {
        assert_eq!(
            relayed_gemini_refusal_body(
                r#"{"error":{"code":502,"message":"Gemini blocked the response: OTHER"}}"#
            ),
            Some((
                RefusalReason::Unspecified,
                "Gemini blocked the response: OTHER".to_string()
            ))
        );
        // The sentence elsewhere in the body (an echoed prompt, metadata)
        // never stands in for the envelope message.
        for body in [
            r#"{"error":{"code":502,"message":"Bad Gateway","metadata":{"raw":"Gemini blocked the response: OTHER"}}}"#,
            r#"{"error":{"code":502,"message":"Bad Gateway"},"prompt":"Gemini blocked the response: OTHER"}"#,
            "Gemini blocked the response: OTHER",
            "502 Bad Gateway",
        ] {
            assert_eq!(relayed_gemini_refusal_body(body), None, "{body}");
        }
    }

    /// One provider frame through the normalizer, expecting one failure.
    fn stream_failure(dialect: crate::dialects::Dialect, data: &str) -> crate::errors::Failure {
        let mut normalizer = crate::dialects::Normalizer::new(dialect);
        let events = normalizer
            .feed(&crate::sse::SseEvent {
                event: None,
                data: data.to_string(),
            })
            .expect("frame decodes");
        let [crate::events::Event::Failed(failure)] = events.as_slice() else {
            panic!("expected one failure for {data}");
        };
        failure.clone()
    }

    #[test]
    fn the_production_in_stream_block_is_a_refusal_and_a_plain_502_is_not() {
        use crate::dialects::Dialect;
        use crate::errors::FailureClass;
        // The exact production frame (OpenRouter, gemini-4-neon, 2026-11-19).
        let chat = r#"{"error":{"code":502,"message":"Gemini blocked the response: OTHER"},"choices":[{"index":0,"delta":{},"finish_reason":"error"}]}"#;
        let responses = r#"{"type":"response.failed","response":{"error":{"code":502,"message":"Gemini blocked the response: OTHER"}}}"#;
        for (dialect, data) in [
            (Dialect::OpenAiCompatible, chat),
            (Dialect::OpenAiResponses, responses),
        ] {
            let failure = stream_failure(dialect, data);
            assert_eq!(failure.failure_class, FailureClass::Refusal, "{data}");
            assert_eq!(failure.refusal_reason, Some(RefusalReason::Unspecified));
            assert!(!failure.retryable_same_deployment && !failure.failover_eligible);
            assert_eq!(
                failure.provider_detail.as_deref(),
                Some("502: Gemini blocked the response: OTHER")
            );
        }
        let safety = stream_failure(Dialect::OpenAiCompatible, &chat.replace("OTHER", "SAFETY"));
        assert_eq!(safety.refusal_reason, Some(RefusalReason::ContentPolicy));
        assert!(safety.provider_content_filter);
        for data in [
            r#"{"error":{"code":502,"message":"Bad Gateway"}}"#,
            r#"{"error":{"code":502,"message":"Provider returned error","metadata":{"raw":"Gemini blocked the response: OTHER"}}}"#,
        ] {
            let failure = stream_failure(Dialect::OpenAiCompatible, data);
            assert_eq!(
                failure.failure_class,
                FailureClass::ProviderInternal,
                "{data}"
            );
        }
    }
}
