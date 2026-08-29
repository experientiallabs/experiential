//! Provider-rejected parameter attribution for sanitized 400s.
//!
//! When a provider rejects a dispatched request with a client-error status,
//! the public failure stays sanitized: no provider prose or body content ever
//! crosses the boundary. The ONE fact this module may relay is the parameter
//! path the provider named, and only when it validates against the strict
//! path grammar below — anything else keeps today's content-free message.
//!
//! Extraction classification per dialect (a new [`Dialect`] variant fails to
//! compile until it is classified here, and the exhaustiveness test pins the
//! documented source):
//!
//! | dialect                 | source                                        |
//! |-------------------------|-----------------------------------------------|
//! | `OpenAiResponses`       | `error.param` field, verbatim JSON string      |
//! | `OpenAiCompatible`      | `error.param` field, verbatim JSON string      |
//! | `AnthropicMessages`     | leading `path: ` token of `error.message`      |
//! | `GeminiGenerateContent` | `fieldViolations[].field`, else `* path: ` msg |
//! | `BedrockConverseStream` | none — no machine-readable parameter contract  |

use serde_json::Value;

use crate::dialects::Dialect;

/// Longest parameter path relayed; anything longer is treated as prose.
const MAXIMUM_PATH_LENGTH: usize = 128;

/// Extract the provider-named parameter path from one client-error body.
///
/// Returns `Some(path)` only when the dialect's documented source yields a
/// string that passes [`valid_parameter_path`]; every other body — missing
/// fields, prose, oversized or non-path content, non-JSON — yields `None`
/// and the caller keeps the content-free sanitized message.
pub fn rejected_parameter(dialect: Dialect, body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    let candidate = match dialect {
        Dialect::OpenAiResponses | Dialect::OpenAiCompatible => value
            .get("error")?
            .get("param")?
            .as_str()
            .map(str::to_string),
        Dialect::AnthropicMessages => {
            let message = value.get("error")?.get("message")?.as_str()?;
            let (head, _rest) = message.split_once(": ")?;
            Some(head.to_string())
        }
        Dialect::GeminiGenerateContent => gemini_field_violation(&value),
        Dialect::BedrockConverseStream => None,
    }?;
    valid_parameter_path(&candidate).then_some(candidate)
}

/// `fieldViolations[].field` when a `google.rpc.BadRequest` detail exists,
/// else the leading `* <path>: ` token of the message (the shape the live
/// API returned for a generation-config violation, 2026-08-29).
fn gemini_field_violation(value: &Value) -> Option<String> {
    let error = value.get("error")?;
    if let Some(details) = error.get("details").and_then(Value::as_array) {
        for detail in details {
            let type_url = detail.get("@type").and_then(Value::as_str).unwrap_or("");
            if !type_url.ends_with("google.rpc.BadRequest") {
                continue;
            }
            if let Some(field) = detail
                .get("fieldViolations")
                .and_then(Value::as_array)
                .and_then(|violations| {
                    violations
                        .iter()
                        .find_map(|violation| violation.get("field").and_then(Value::as_str))
                })
            {
                return Some(field.to_string());
            }
        }
    }
    let message = error.get("message")?.as_str()?;
    let head = message.strip_prefix("* ")?;
    let (path, _rest) = head.split_once(": ")?;
    Some(path.to_string())
}

/// Whether one candidate is a parameter path and cannot be prose.
///
/// Grammar: ASCII segments of `[A-Za-z0-9_-]` joined by `.`, with optional
/// numeric `[N]` indexes; no whitespace, no empty segments, bounded length.
/// This is deliberately narrower than what providers could emit: a rejected
/// candidate costs only attribution, while an accepted one crosses the
/// sanitization boundary.
pub fn valid_parameter_path(candidate: &str) -> bool {
    if candidate.is_empty() || candidate.len() > MAXIMUM_PATH_LENGTH {
        return false;
    }
    if !candidate.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_') {
        return false;
    }
    let mut chars = candidate.chars().peekable();
    let mut segment_open = false;
    while let Some(c) = chars.next() {
        match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' => segment_open = true,
            '.' => {
                if !segment_open || chars.peek().is_none() {
                    return false;
                }
                segment_open = false;
            }
            '[' => {
                if !segment_open {
                    return false;
                }
                let mut digits = 0usize;
                loop {
                    match chars.next() {
                        Some(d) if d.is_ascii_digit() => digits += 1,
                        Some(']') if digits > 0 => break,
                        _ => return false,
                    }
                }
            }
            _ => return false,
        }
    }
    segment_open
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_param_field_is_relayed_when_it_is_a_path() {
        // Exact body captured live from api.openai.com (2026-08-29).
        let body = r#"{"error": {"message": "Unknown parameter: 'input[1].status'.",
            "type": "invalid_request_error", "param": "input[1].status",
            "code": "unknown_parameter"}}"#;
        assert_eq!(
            rejected_parameter(Dialect::OpenAiResponses, body).as_deref(),
            Some("input[1].status")
        );
        assert_eq!(
            rejected_parameter(Dialect::OpenAiCompatible, body).as_deref(),
            Some("input[1].status")
        );
    }

    #[test]
    fn anthropic_leading_message_token_is_relayed_when_it_is_a_path() {
        // Anthropic names the field as a leading `path: ` message token.
        let body = r#"{"type": "error", "error": {"type": "invalid_request_error",
            "message": "context_management: Extra inputs are not permitted"}}"#;
        assert_eq!(
            rejected_parameter(Dialect::AnthropicMessages, body).as_deref(),
            Some("context_management")
        );
        let nested = r#"{"type": "error", "error": {"type": "invalid_request_error",
            "message": "messages.1.content.0.text: Field required"}}"#;
        assert_eq!(
            rejected_parameter(Dialect::AnthropicMessages, nested).as_deref(),
            Some("messages.1.content.0.text")
        );
    }

    #[test]
    fn gemini_message_path_token_is_relayed_without_violation_details() {
        // Exact live shape from generativelanguage.googleapis.com (2026-08-29).
        let body = r#"{"error": {"code": 400, "status": "INVALID_ARGUMENT",
            "message": "* GenerateContentRequest.generation_config.temperature: temperature must be in the range [0.0, 2.0].\n"}}"#;
        assert_eq!(
            rejected_parameter(Dialect::GeminiGenerateContent, body).as_deref(),
            Some("GenerateContentRequest.generation_config.temperature")
        );
        // Prose-leading messages stay content-free.
        let prose = r#"{"error": {"code": 400, "message": "API key not valid: renew it"}}"#;
        assert_eq!(
            rejected_parameter(Dialect::GeminiGenerateContent, prose),
            None
        );
    }

    #[test]
    fn gemini_bad_request_field_violation_is_relayed() {
        let body = r#"{"error": {"code": 400, "status": "INVALID_ARGUMENT",
            "message": "Invalid JSON payload received.",
            "details": [{"@type": "type.googleapis.com/google.rpc.BadRequest",
                "fieldViolations": [{"field": "generation_config.temperature",
                    "description": "out of range"}]}]}}"#;
        assert_eq!(
            rejected_parameter(Dialect::GeminiGenerateContent, body).as_deref(),
            Some("generation_config.temperature")
        );
    }

    #[test]
    fn bedrock_never_attributes() {
        let body = r#"{"message": "Malformed input request: extraneous key [topK]"}"#;
        assert_eq!(
            rejected_parameter(Dialect::BedrockConverseStream, body),
            None
        );
    }

    #[test]
    fn provider_prose_never_crosses_the_boundary() {
        // A prose-bearing param field, prose-only Anthropic messages, and
        // adversarial path-shaped content all stay content-free.
        let prose_param = r#"{"error": {"param": "please contact support at example.com"}}"#;
        assert_eq!(
            rejected_parameter(Dialect::OpenAiResponses, prose_param),
            None
        );
        let no_path_message =
            r#"{"type": "error", "error": {"message": "Extra inputs are not permitted"}}"#;
        assert_eq!(
            rejected_parameter(Dialect::AnthropicMessages, no_path_message),
            None
        );
        let prose_head = r#"{"error": {"message": "Your credit balance is too low: top up"}}"#;
        assert_eq!(
            rejected_parameter(Dialect::AnthropicMessages, prose_head),
            None
        );
        let not_json = "upstream said no";
        assert_eq!(rejected_parameter(Dialect::OpenAiResponses, not_json), None);
    }

    #[test]
    fn the_path_grammar_is_strict() {
        for accepted in [
            "temperature",
            "input[1].status",
            "messages.1.content.0.text",
            "tools[0].function.name",
            "generation_config.top_k",
            "anthropic-beta",
        ] {
            assert!(valid_parameter_path(accepted), "{accepted}");
        }
        for rejected in [
            "",
            "has space",
            "trailing.",
            ".leading",
            "double..dot",
            "input[].status",
            "input[1.status",
            "input[a]",
            "9starts_with_digit",
            "unicode_ĸey",
            "a]b",
            "semi;colon",
            "path\nnewline",
        ] {
            assert!(!valid_parameter_path(rejected), "{rejected}");
        }
        assert!(!valid_parameter_path(&"x".repeat(129)));
    }

    /// The documented extraction source for one dialect.
    ///
    /// The match is exhaustive on purpose: adding a dialect without deciding
    /// its attribution contract fails this drift gate at compile time, and
    /// `rejected_parameter`'s own exhaustive match enforces the same in the
    /// production path.
    fn extraction_classification(dialect: Dialect) -> &'static str {
        match dialect {
            Dialect::OpenAiResponses | Dialect::OpenAiCompatible => "error.param field",
            Dialect::AnthropicMessages => "leading path token of error.message",
            Dialect::GeminiGenerateContent => {
                "google.rpc.BadRequest fieldViolations, else leading message path token"
            }
            Dialect::BedrockConverseStream => "none: no machine-readable parameter contract",
        }
    }

    #[test]
    fn every_dialect_carries_an_explicit_extraction_classification() {
        for dialect in [
            Dialect::OpenAiResponses,
            Dialect::AnthropicMessages,
            Dialect::OpenAiCompatible,
            Dialect::GeminiGenerateContent,
            Dialect::BedrockConverseStream,
        ] {
            assert!(!extraction_classification(dialect).is_empty());
        }
    }
}
