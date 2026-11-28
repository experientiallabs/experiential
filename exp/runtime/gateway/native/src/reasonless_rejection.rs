//! A client-error body whose sentence gives no reason at all.
//!
//! Some reseller relays answer a 4xx with nothing but the family name of the
//! error and their own trace id (`{"code":0,"message":"invalid request error
//! trace_id: 9f3c","type":"invalid_request_error"}`). That sentence names no
//! field, no limit and no remedy, so it is not evidence that the CALLER'S
//! request is wrong: the same relay answers it for a request it served a
//! minute earlier and serves again a minute later. Two consequences follow:
//!
//! - It is not relayed as detail. "provider rejected the request: invalid
//!   request error trace_id: [redacted]" reads as an explanation and says
//!   nothing; the caller gets the fieldless rejection copy instead, which
//!   states that the provider gave no reason.
//! - It fails over. Another rung of the same model either serves the request
//!   or answers with its own sentence, which is strictly more useful to the
//!   caller than this one; a route with no other rung still surfaces the 400.
//!
//! Only the dialect's message field is read, so request text echoed elsewhere
//! in the body cannot change routing.

use crate::dialects::Dialect;
use crate::error_envelope::parse_error_document;
use crate::param_attribution::error_message_field;

/// Sentences that only restate that the request was rejected, compared after
/// lowercasing, dropping a trailing `trace_id` clause and trailing punctuation.
const REASONLESS_SENTENCES: &[&str] = &[
    "invalid request error",
    "invalid request",
    "invalid_request_error",
    "bad request",
];

/// Whether a 4xx body's sentence states no reason (see the module docs).
pub fn rejected_without_reason(dialect: Dialect, body: &str) -> bool {
    let Some(value) = parse_error_document(body) else {
        return false;
    };
    error_message_field(dialect, &value).is_some_and(reasonless_sentence)
}

/// Whether one provider sentence is a bare restatement of the rejection.
fn reasonless_sentence(message: &str) -> bool {
    let lowered = message.trim().to_ascii_lowercase();
    let head = lowered
        .split_once("trace_id")
        .map_or(lowered.as_str(), |(head, _)| head);
    let head =
        head.trim_end_matches(|c: char| c.is_whitespace() || matches!(c, '.' | ',' | ';' | ':'));
    REASONLESS_SENTENCES.contains(&head)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_relays_family_name_with_its_trace_id_is_reasonless() {
        // A reseller relay's flat envelope: the family name and its trace id.
        let flat = r#"{"code":0,"message":"invalid request error trace_id: 9f3c2b7a1d4e","type":"invalid_request_error"}"#;
        assert!(rejected_without_reason(Dialect::OpenAiCompatible, flat));
        let nested =
            r#"{"error":{"message":"Invalid request error.","type":"invalid_request_error"}}"#;
        assert!(rejected_without_reason(Dialect::OpenAiCompatible, nested));
        let bare =
            r#"{"code":400,"reason":"INVALID_REQUEST_BODY","message":"Bad Request","metadata":{}}"#;
        assert!(rejected_without_reason(Dialect::OpenAiCompatible, bare));
    }

    #[test]
    fn a_sentence_naming_a_field_or_limit_keeps_the_callers_class() {
        for message in [
            "max_tokens must be less than or equal to 131072",
            "Function tools with reasoning_effort are not supported. trace_id: 9f3c",
            "invalid request error: messages[2].content must not be empty trace_id: 9f3c",
            "invalid request: unknown parameter logit_bias",
        ] {
            let body = serde_json::json!({"code": 0, "message": message}).to_string();
            assert!(
                !rejected_without_reason(Dialect::OpenAiCompatible, &body),
                "{message}"
            );
        }
        // A body without a message field, or not JSON, is never read as one.
        assert!(!rejected_without_reason(
            Dialect::OpenAiCompatible,
            r#"{"error":{"type":"invalid_request_error"}}"#
        ));
        assert!(!rejected_without_reason(
            Dialect::OpenAiCompatible,
            "invalid request error"
        ));
    }

    #[test]
    fn request_text_outside_the_message_field_cannot_make_a_body_reasonless() {
        let echoed = r#"{"code":0,"message":"prompt too long","param":"invalid request error"}"#;
        assert!(!rejected_without_reason(Dialect::OpenAiCompatible, echoed));
    }
}
