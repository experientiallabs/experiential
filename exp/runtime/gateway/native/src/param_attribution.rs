//! Provider-rejected parameter attribution for sanitized 400s.
//!
//! When a provider rejects a dispatched request with a client-error status,
//! two facts may reach the caller, and only for that class: the parameter
//! path the provider named, validated against the strict path grammar below,
//! and the provider's own one-sentence explanation, read from the documented
//! message field and sanitized by [`rejected_detail`]. Nothing else from the
//! body crosses the boundary, and no other failure class relays any of it.
//!
//! Extraction classification per dialect (a new [`Dialect`] variant fails to
//! compile until it is classified here, and the exhaustiveness test pins the
//! documented source):
//!
//! | dialect                 | source                                        |
//! |-------------------------|-----------------------------------------------|
//! | `OpenAiResponses`       | `error.param`, else fixed unknown-argument msg |
//! | `OpenAiCompatible`      | `error.param`, else fixed unknown-argument msg |
//! | `AnthropicMessages`     | leading `path: ` or `` `path` `` message token |
//! | `GeminiGenerateContent` | `fieldViolations[].field`, else `* path: ` msg |
//! | `BedrockConverseStream` | none — no machine-readable parameter contract  |
//!
//! The explanation relayed alongside it comes from `error.message` for every
//! dialect except Bedrock, which reports a bare top-level `message`.

use serde_json::Value;

use crate::dialects::Dialect;

/// Longest parameter path relayed; anything longer is treated as prose.
const MAXIMUM_PATH_LENGTH: usize = 128;

/// Longest provider explanation relayed; longer text is a body dump.
const MAXIMUM_DETAIL_LENGTH: usize = 240;

/// Fixed OpenAI-family prefix naming one unsupported argument.
const UNKNOWN_ARGUMENT_PREFIXES: [&str; 2] = [
    "Unrecognized request argument supplied: ",
    "Unknown parameter: ",
];

/// Extract the provider-named parameter path from one client-error body.
///
/// Returns `Some(path)` only when the dialect's documented source yields a
/// string that passes [`valid_parameter_path`]; every other body — missing
/// fields, prose, oversized or non-path content, non-JSON — yields `None`
/// and the caller keeps the content-free sanitized message.
pub fn rejected_parameter(dialect: Dialect, body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    let candidate = match dialect {
        Dialect::OpenAiResponses | Dialect::OpenAiCompatible => {
            let error = value.get("error")?;
            match error.get("param").and_then(Value::as_str) {
                Some(param) => Some(param.to_string()),
                None => unknown_argument_name(error.get("message")?.as_str()?),
            }
        }
        Dialect::AnthropicMessages => {
            let message = value.get("error")?.get("message")?.as_str()?;
            match message.split_once(": ") {
                Some((head, _rest)) if valid_parameter_path(head) => Some(head.to_string()),
                _ => quoted_leading_name(message),
            }
        }
        Dialect::GeminiGenerateContent => gemini_field_violation(&value),
        Dialect::BedrockConverseStream => None,
    }?;
    valid_parameter_path(&candidate).then_some(candidate)
}

/// Extract the provider's own explanation from one client-error body.
///
/// A client-error body explains what the caller got wrong, and the caller is
/// the only party who can act on it, so the sentence itself is worth more to
/// them than the gateway's generic wording. Only the dialect's documented
/// message field is read, and only after [`sanitized_detail`] proves it is
/// one bounded single-line sentence; a body dump, a stack trace, or a
/// multi-line payload yields `None` and the caller keeps the generic message.
///
/// This relays provider wording verbatim, so it is restricted at the call
/// site to the client-error class. Provider messages for authentication,
/// not-found, and server-side failures are operator-facing and can name
/// deployments or accounts, so they stay content-free.
pub fn rejected_detail(dialect: Dialect, body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    let message = match dialect {
        Dialect::OpenAiResponses
        | Dialect::OpenAiCompatible
        | Dialect::AnthropicMessages
        | Dialect::GeminiGenerateContent => value.get("error")?.get("message")?.as_str()?,
        // Bedrock reports a modeling error as a bare top-level `message`.
        Dialect::BedrockConverseStream => value.get("message")?.as_str()?,
    };
    sanitized_detail(message)
}

/// One provider sentence reduced to bounded, single-line, printable text.
///
/// Control characters end the candidate rather than being escaped: their
/// presence means the field carries a payload, not a sentence. Interior runs
/// of spaces and tabs collapse so the relayed text stays one readable line.
fn sanitized_detail(message: &str) -> Option<String> {
    let trimmed = message.trim();
    if trimmed.is_empty() || trimmed.chars().count() > MAXIMUM_DETAIL_LENGTH {
        return None;
    }
    if trimmed
        .chars()
        .any(|c| (c.is_control() && c != '\t') || (c.is_whitespace() && c != ' ' && c != '\t'))
    {
        return None;
    }
    let collapsed = trimmed.split_whitespace().collect::<Vec<_>>().join(" ");
    (!collapsed.is_empty()).then_some(collapsed)
}

/// The argument named after one fixed OpenAI-family unknown-argument prefix.
///
/// Azure's OpenAI surface reports an unsupported sampling field this way and
/// carries no `param`, so the name lives in an otherwise fixed sentence. Only
/// the trailing name is read, and only when the sentence matches exactly.
fn unknown_argument_name(message: &str) -> Option<String> {
    UNKNOWN_ARGUMENT_PREFIXES.iter().find_map(|prefix| {
        let name = message.strip_prefix(prefix)?.trim_end_matches('.');
        Some(name.trim_matches('\'').to_string())
    })
}

/// The backtick-quoted name one message opens with.
///
/// Anthropic reports a per-model sampling refusal as `` `top_p` is deprecated
/// for this model. ``, naming the field only inside the prose it otherwise
/// owns. Only the quoted leading token is read; the prose is discarded.
fn quoted_leading_name(message: &str) -> Option<String> {
    let rest = message.strip_prefix('`')?;
    let (name, _prose) = rest.split_once('`')?;
    Some(name.to_string())
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
    fn anthropic_backtick_quoted_leading_name_is_relayed() {
        // Exact body captured live from api.anthropic.com (2026-08-29): the
        // per-model sampling refusal names the field only inside prose.
        let body = r#"{"type": "error", "error": {"type": "invalid_request_error",
            "message": "`top_p` is deprecated for this model."}}"#;
        assert_eq!(
            rejected_parameter(Dialect::AnthropicMessages, body).as_deref(),
            Some("top_p")
        );
        // A quoted phrase is prose, not a path, so nothing is relayed.
        let quoted_prose = r#"{"type": "error", "error": {
            "message": "`contact support` for account 4821 to continue."}}"#;
        assert_eq!(
            rejected_parameter(Dialect::AnthropicMessages, quoted_prose),
            None
        );
    }

    #[test]
    fn openai_family_unknown_argument_sentence_is_relayed_without_a_param() {
        // Exact body captured live from Azure's OpenAI surface (2026-08-29).
        let body = r#"{"error": {"code": "unrecognized_request_argument",
            "message": "Unrecognized request argument supplied: top_k",
            "details": "Unrecognized request argument supplied: top_k"}}"#;
        assert_eq!(
            rejected_parameter(Dialect::OpenAiCompatible, body).as_deref(),
            Some("top_k")
        );
        let quoted = r#"{"error": {"message": "Unknown parameter: 'response_format'."}}"#;
        assert_eq!(
            rejected_parameter(Dialect::OpenAiResponses, quoted).as_deref(),
            Some("response_format")
        );
        // Any other message keeps the content-free failure.
        let other = r#"{"error": {"message": "This model is not available to your account."}}"#;
        assert_eq!(rejected_parameter(Dialect::OpenAiCompatible, other), None);
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
            Dialect::OpenAiResponses | Dialect::OpenAiCompatible => {
                "error.param field, else fixed unknown-argument message"
            }
            Dialect::AnthropicMessages => "leading path or backtick-quoted token of error.message",
            Dialect::GeminiGenerateContent => {
                "google.rpc.BadRequest fieldViolations, else leading message path token"
            }
            Dialect::BedrockConverseStream => "none: no machine-readable parameter contract",
        }
    }

    #[test]
    fn provider_explanation_is_relayed_for_every_dialect_message_field() {
        // Exact shapes captured live from each provider (2026-08-29).
        let openai = r#"{"error": {"message": "Unknown parameter: 'top_k'.",
            "type": "invalid_request_error"}}"#;
        assert_eq!(
            rejected_detail(Dialect::OpenAiCompatible, openai).as_deref(),
            Some("Unknown parameter: 'top_k'.")
        );
        let anthropic = r#"{"type": "error", "error": {"type": "invalid_request_error",
            "message": "`top_p` is deprecated for this model."}}"#;
        assert_eq!(
            rejected_detail(Dialect::AnthropicMessages, anthropic).as_deref(),
            Some("`top_p` is deprecated for this model.")
        );
        let bedrock = r#"{"message": "The provided model does not support tool use."}"#;
        assert_eq!(
            rejected_detail(Dialect::BedrockConverseStream, bedrock).as_deref(),
            Some("The provided model does not support tool use.")
        );
    }

    #[test]
    fn provider_explanation_is_dropped_when_it_is_not_one_bounded_sentence() {
        let multiline = r#"{"error": {"message": "failed\n  at deployment-7\n"}}"#;
        assert_eq!(rejected_detail(Dialect::OpenAiCompatible, multiline), None);
        let oversized = format!(
            r#"{{"error": {{"message": "{}"}}}}"#,
            "x".repeat(MAXIMUM_DETAIL_LENGTH + 1)
        );
        assert_eq!(rejected_detail(Dialect::OpenAiCompatible, &oversized), None);
        assert_eq!(rejected_detail(Dialect::OpenAiCompatible, "{}"), None);
        assert_eq!(rejected_detail(Dialect::OpenAiCompatible, "<html>"), None);
        let blank = r#"{"error": {"message": "   "}}"#;
        assert_eq!(rejected_detail(Dialect::OpenAiCompatible, blank), None);
    }

    #[test]
    fn relayed_explanation_collapses_interior_whitespace_runs() {
        let padded = r#"{"error": {"message": "  Unknown   parameter:\t'top_k'.  "}}"#;
        assert_eq!(
            rejected_detail(Dialect::OpenAiCompatible, padded).as_deref(),
            Some("Unknown parameter: 'top_k'.")
        );
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
