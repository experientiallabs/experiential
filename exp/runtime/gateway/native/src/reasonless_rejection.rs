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
//! The sentence is read only from the dialect's message field, and a body
//! whose structured fields name anything (a `param` or `field`, a parameter
//! the dialect's own reader attributes, or a code token more specific than
//! the family type) is never reasonless, however terse its sentence. Only a
//! terminal correlation-id clause (`trace_id: 9f3c`, `(request_id=ab12)`) is
//! ignored; a sentence that is ABOUT such a field (`trace_id is required`)
//! keeps its reason.

use serde_json::Value;

use crate::dialects::Dialect;
use crate::error_envelope::parse_error_document;
use crate::param_attribution::{
    error_message_field, generic_error_code, rejected_code, rejected_parameter,
};

/// Sentences that only restate that the request was rejected, compared after
/// lowercasing, dropping a terminal correlation-id clause and trailing
/// punctuation.
const REASONLESS_SENTENCES: &[&str] = &[
    "invalid request error",
    "invalid request",
    "invalid request body",
    "invalid_request_error",
    "bad request",
];

/// Correlation-id labels a relay appends to its sentence.
const CORRELATION_LABELS: &[&str] = &["trace_id", "request_id"];

/// Shortest correlation value read as an id rather than a word.
const MINIMUM_CORRELATION_ID_LENGTH: usize = 4;

/// Whether a 4xx body's sentence states no reason (see the module docs).
pub fn rejected_without_reason(dialect: Dialect, body: &str) -> bool {
    let Some(value) = parse_error_document(body) else {
        return false;
    };
    // Gemini's streaming endpoints wrap the error in a one-element array.
    let value = match value {
        Value::Array(mut items) if items.len() == 1 && items[0].is_object() => items.remove(0),
        value => value,
    };
    let document = value.to_string();
    if names_a_structured_reason(dialect, &value, &document) {
        return false;
    }
    error_message_field(dialect, &value).is_some_and(reasonless_sentence)
}

/// Whether the body's structured fields say more than "rejected": a named
/// parameter or field, or ANY code-like token (`code`, `type`, `reason`,
/// `status`, at the top level or under `error`, or the dialect's own code
/// reader) beyond the generic family type. Every token is read, not only the
/// one an envelope reader prefers, so a specific `type` beside a numeric
/// `code` still counts.
fn names_a_structured_reason(dialect: Dialect, value: &Value, body: &str) -> bool {
    let objects = [
        Some(value),
        value.get("error").filter(|error| error.is_object()),
    ];
    let named_field = objects.iter().flatten().any(|object| {
        ["param", "field"].iter().any(|key| {
            object
                .get(*key)
                .and_then(Value::as_str)
                .is_some_and(|name| !name.trim().is_empty())
        })
    });
    let specific_token = objects.iter().flatten().any(|object| {
        ["code", "type", "reason", "status"].iter().any(|key| {
            object
                .get(*key)
                .and_then(Value::as_str)
                .is_some_and(specific_code)
        })
    });
    named_field
        || specific_token
        || rejected_parameter(dialect, body).is_some()
        || rejected_code(dialect, body).is_some_and(|code| specific_code(&code))
}

/// Family-type tokens that say only "the request was rejected", compared
/// lowercased with `_` and `-` removed (`invalid-argument`, `INVALID_ARGUMENT`,
/// vLLM's `BadRequestError`).
const FAMILY_TOKENS: &[&str] = &[
    "invalidrequesterror",
    "invalidrequest",
    "invalidrequestbody",
    "invalidargument",
    "badrequest",
    "badrequesterror",
    "error",
];

/// Whether one code token names more than the generic family type.
fn specific_code(token: &str) -> bool {
    let token = token.trim();
    let folded: String = token
        .chars()
        .filter(|c| !matches!(c, '_' | '-'))
        .collect::<String>()
        .to_ascii_lowercase();
    !token.is_empty() && !generic_error_code(token) && !FAMILY_TOKENS.contains(&folded.as_str())
}

/// Whether one provider sentence is a bare restatement of the rejection.
fn reasonless_sentence(message: &str) -> bool {
    let lowered = message.trim().to_ascii_lowercase();
    let head = without_correlation_suffix(&lowered);
    let head = head
        .trim_end_matches(|c: char| c.is_whitespace() || matches!(c, '.' | ',' | ';' | ':' | '!'));
    REASONLESS_SENTENCES.contains(&head)
}

/// `sentence` without a terminal correlation-id clause, or unchanged.
///
/// The clause is a label from [`CORRELATION_LABELS`] at a word boundary,
/// then `:` or `=`, then one id-shaped value (ASCII letters, digits, `-` or
/// `_`, at least [`MINIMUM_CORRELATION_ID_LENGTH`] long, containing a digit),
/// optionally closed by `)` and a final `.`, and nothing after it.
fn without_correlation_suffix(sentence: &str) -> &str {
    for label in CORRELATION_LABELS {
        let Some(at) = sentence.rfind(label) else {
            continue;
        };
        let before = &sentence[..at];
        if !before.is_empty() && !before.ends_with([' ', '(', ',', ';', '[']) {
            continue;
        }
        let rest = sentence[at + label.len()..].trim_start();
        let Some(rest) = rest.strip_prefix([':', '=']) else {
            continue;
        };
        let rest = rest.trim_start();
        let id_end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '_')))
            .unwrap_or(rest.len());
        let (id, tail) = rest.split_at(id_end);
        let tail = tail.trim_end();
        let tail = tail.strip_suffix('.').unwrap_or(tail);
        let tail = tail.strip_suffix([')', ']']).unwrap_or(tail);
        let id_shaped =
            id.len() >= MINIMUM_CORRELATION_ID_LENGTH && id.chars().any(|c| c.is_ascii_digit());
        if id_shaped && tail.trim().is_empty() {
            return before.trim_end_matches(['(', '[', ' ', ',', ';']);
        }
    }
    sentence
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reasonless(body: &str) -> bool {
        rejected_without_reason(Dialect::OpenAiCompatible, body)
    }

    #[test]
    fn a_relays_family_name_with_its_trace_id_is_reasonless() {
        // A reseller relay's flat envelope: the family name and its trace id.
        assert!(reasonless(
            r#"{"code":0,"message":"invalid request error trace_id: 9f3c2b7a1d4e","type":"invalid_request_error"}"#
        ));
        assert!(reasonless(
            r#"{"error":{"message":"Invalid request error.","type":"invalid_request_error"}}"#
        ));
        assert!(reasonless(
            r#"{"code":400,"reason":"INVALID_REQUEST_BODY","message":"Bad Request","metadata":{}}"#
        ));
        assert!(reasonless(
            r#"{"code":400,"message":"Bad Request (request_id=ab12cd34).","type":"invalid_request_error"}"#
        ));
        // xAI's documented envelope: a generic code spelled with a hyphen.
        assert!(reasonless(
            r#"{"code":"invalid-argument","error":"Bad Request"}"#
        ));
        assert!(reasonless(r#"{"code":400,"message":"Bad Request!"}"#));
        // vLLM's family class, API Management's flat sentence, and Gemini's
        // streaming array envelope.
        assert!(reasonless(
            r#"{"object":"error","message":"Bad Request","type":"BadRequestError","param":null,"code":400}"#
        ));
        assert!(reasonless(
            r#"{"statusCode":400,"message":"Invalid request body."}"#
        ));
        assert!(rejected_without_reason(
            Dialect::GeminiGenerateContent,
            r#"[{"error":{"code":400,"message":"Bad Request","status":"INVALID_ARGUMENT"}}]"#
        ));
    }

    #[test]
    fn a_terse_sentence_beside_a_named_field_or_specific_code_is_not_reasonless() {
        // The sentence restates the rejection, but the structured fields say
        // which parameter and why: that is the caller's error, not the lane's.
        for body in [
            r#"{"error":{"message":"Bad Request","type":"invalid_request_error","param":"temperature","code":"invalid_value"}}"#,
            r#"{"error":{"message":"Bad Request","type":"invalid_request_error","param":"temperature"}}"#,
            r#"{"error":{"message":"Bad Request","type":"invalid_request_error","code":"invalid_value"}}"#,
            r#"{"code":400,"reason":"INVALID_PARAMETER","message":"invalid request error trace_id: 9f3c"}"#,
            r#"{"code":400,"message":"Bad Request","type":"invalid_request_error","field":"messages[1].content"}"#,
            // A specific `type` beside a numeric `code` the envelope reader prefers.
            r#"{"code":400,"type":"invalid_value","message":"Bad Request"}"#,
        ] {
            assert!(!reasonless(body), "{body}");
        }
    }

    #[test]
    fn a_sentence_about_a_correlation_field_keeps_its_reason() {
        // Only a terminal id-shaped clause is dropped, never the first
        // occurrence of the label.
        for message in [
            "Bad Request: trace_id is required",
            "invalid request: trace_id must be a uuid",
            "invalid request error trace_id: missing",
            "Bad Request: request_id=abc1 is already in use",
        ] {
            let body =
                serde_json::json!({"code": 400, "message": message, "type": "invalid_request_error"})
                    .to_string();
            assert!(!reasonless(&body), "{message}");
        }
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
            assert!(!reasonless(&body), "{message}");
        }
        // A body without a message field, or not JSON, is never read as one.
        assert!(!reasonless(r#"{"error":{"type":"invalid_request_error"}}"#));
        assert!(!reasonless("invalid request error"));
    }

    #[test]
    fn request_text_outside_the_message_field_cannot_make_a_body_reasonless() {
        assert!(!reasonless(
            r#"{"code":0,"message":"prompt too long","param":"invalid request error"}"#
        ));
    }

    #[test]
    fn only_a_terminal_id_shaped_clause_is_stripped() {
        assert_eq!(
            without_correlation_suffix("invalid request error trace_id: 9f3c2b7a1d4e"),
            "invalid request error"
        );
        assert_eq!(
            without_correlation_suffix("bad request (trace_id=ab-12_cd)."),
            "bad request"
        );
        assert_eq!(
            without_correlation_suffix("bad request: trace_id is required"),
            "bad request: trace_id is required"
        );
        assert_eq!(
            without_correlation_suffix("bad request x_trace_id: 9f3c"),
            "bad request x_trace_id: 9f3c"
        );
    }
}
