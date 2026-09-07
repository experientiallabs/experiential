//! Provider rejection SHAPES the pre-stream and in-stream failure paths read
//! off a client-error body: an aggregator's generic sentence hiding the
//! upstream's own, a lane limitation a caller cannot fix, an aggregator
//! routing gate that is not a credential verdict, and a 404 that refuses a
//! handle the caller sent rather than naming a missing model. Each reads only
//! the dialect's documented fields and never logs body text.

use serde_json::Value;

use crate::dialects::Dialect;
use crate::param_attribution::{error_message_field, rejected_model_not_found};

/// Sentences a provider answers with a 400 for a request shape the OpenAI
/// contract allows but THIS lane's serving stack cannot carry (a chat
/// template that only accepts a leading system turn). They are lane
/// limitations, not caller errors: the request fails over to the next rung and
/// only a route with no other rung surfaces the sentence.
const LANE_LIMITATION_PHRASES: &[&str] = &[
    "system message must be at the beginning",
    "system message should be at the beginning",
    "only the first message can be a system message",
];

/// Whether a 4xx body's error SENTENCE describes a limitation of the lane
/// rather than of the caller's request (see [`LANE_LIMITATION_PHRASES`]). Only
/// the dialect's message field is read, so request text echoed elsewhere in
/// the body cannot change routing.
pub fn rejected_by_lane_limitation(dialect: Dialect, body: &str) -> bool {
    let value: Value = match serde_json::from_str(body) {
        Ok(value) => value,
        Err(_) => return false,
    };
    error_message_field(dialect, &value).is_some_and(|message| {
        let lowered = message.to_ascii_lowercase();
        LANE_LIMITATION_PHRASES
            .iter()
            .any(|phrase| lowered.contains(phrase))
    })
}

/// Whether a 403 body is an aggregator ROUTING verdict rather than a
/// credential one. OpenRouter runs its routing funnel only AFTER the key has
/// authenticated, and reports the funnel it walked (`metadata.routing_funnel`)
/// plus the step that refused (`metadata.failed_routing_step`; "Gate Free
/// Endpoints by Agentic Harness" on its app-allow-listed free endpoints,
/// 2026-09-06). Both fields together mean the key is fine and this lane will
/// not serve this model for the gateway's account, so it takes the not-found
/// policy and the ladder advances. Only the OpenAI-compatible wire OpenRouter
/// speaks is read; other dialects keep the credential verdict.
pub fn rejected_by_routing_gate(dialect: Dialect, body: &str) -> bool {
    if dialect != Dialect::OpenAiCompatible {
        return false;
    }
    let value: Value = match serde_json::from_str(body) {
        Ok(value) => value,
        Err(_) => return false,
    };
    let Some(metadata) = value.get("error").and_then(|error| error.get("metadata")) else {
        return false;
    };
    let walked_funnel = metadata
        .get("routing_funnel")
        .and_then(Value::as_array)
        .is_some_and(|steps| !steps.is_empty());
    let failed_step = metadata
        .get("failed_routing_step")
        .and_then(Value::as_str)
        .is_some_and(|step| !step.trim().is_empty());
    walked_funnel && failed_step
}

/// Aggregator sentences that say nothing about what was refused.
const GENERIC_AGGREGATOR_MESSAGES: &[&str] = &["provider returned error", "provider error"];

/// The upstream provider's own sentence behind an aggregator's generic one.
///
/// OpenRouter answers a rejected relay with "Provider returned error" and
/// puts the upstream body in `error.metadata.raw` (a JSON document or plain
/// text) plus the upstream's name in `error.metadata.provider_name`. The
/// generic sentence leaves the caller nothing to act on (673 such 400s across
/// 137 orgs in the 24h to 2026-09-07 00:20 UTC), so when the aggregator's
/// message is generic the upstream sentence is read instead, prefixed with
/// the provider's name. Only the message field of a JSON `raw` is used; a
/// plain-text `raw` is taken whole. The result still passes the same
/// identifier screen and length bound as any relayed sentence.
pub fn upstream_relayed_message(error: &Value, message: &str) -> Option<String> {
    let lowered = message.trim().trim_end_matches('.').to_ascii_lowercase();
    if !GENERIC_AGGREGATOR_MESSAGES.contains(&lowered.as_str()) {
        return None;
    }
    let metadata = error.get("metadata")?;
    let raw = metadata.get("raw")?;
    let upstream = match raw {
        Value::String(text) => match serde_json::from_str::<Value>(text) {
            Ok(document) => json_error_sentence(&document)?,
            Err(_) => text.trim().to_string(),
        },
        Value::Object(_) => json_error_sentence(raw)?,
        _ => return None,
    };
    if upstream.is_empty() {
        return None;
    }
    let provider = metadata
        .get("provider_name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| {
            !name.is_empty()
                && name.chars().all(|c| {
                    c.is_ascii_alphanumeric() || c == ' ' || c == '-' || c == '_' || c == '.'
                })
        });
    Some(match provider {
        Some(name) => format!("{name}: {upstream}"),
        None => upstream,
    })
}

/// The human sentence of one upstream error document, whichever documented
/// spelling it uses (`error.message`, `message`, `detail`, or a bare string
/// `error`).
fn json_error_sentence(document: &Value) -> Option<String> {
    let error = document.get("error");
    let candidates = [
        error.and_then(|error| error.get("message")),
        document.get("message"),
        document.get("detail"),
        error.filter(|value| value.is_string()),
    ];
    candidates
        .into_iter()
        .flatten()
        .find_map(|value| value.as_str())
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
}

/// Whether a 404 body is the provider refusing a CALLER reference (an
/// `item_reference`, `conversation`, or similar handle the provider does not
/// hold) rather than a missing model. OpenAI answers those as HTTP 404 with
/// `type: invalid_request_error` and no `model_not_found` code (live
/// 2026-09-07: "Item with id 'rs_...' not found. Items are not persisted when
/// `store` is set to false." and "Conversation with id 'conv_...' not
/// found."). The catalog is fine and every other rung would answer the same,
/// so the request is the caller's 400, never a lane 404 that fails over: one
/// client replaying foreign item ids drove 361 attempts across the astra
/// ladder in three and a half hours.
pub fn rejected_caller_reference_not_found(dialect: Dialect, body: &str) -> bool {
    if !matches!(
        dialect,
        Dialect::OpenAiResponses | Dialect::OpenAiCompatible
    ) {
        return false;
    }
    let value: Value = match serde_json::from_str(body) {
        Ok(value) => value,
        Err(_) => return false,
    };
    let Some(error) = value.get("error") else {
        return false;
    };
    if error.get("type").and_then(Value::as_str) != Some("invalid_request_error")
        || rejected_model_not_found(dialect, body)
    {
        return false;
    }
    // Positive evidence only: the provider names the `input` field, or the
    // sentence opens with one of its reference-not-found shapes. A 404 that
    // names neither (a model or deployment reported without the documented
    // code) keeps the lane policy so another rung can still serve.
    let names_input = error.get("param").and_then(Value::as_str) == Some("input");
    let sentence = error.get("message").and_then(Value::as_str).unwrap_or("");
    names_input
        || CALLER_REFERENCE_SENTENCES
            .iter()
            .any(|prefix| sentence.starts_with(prefix))
}

/// How OpenAI opens a 404 about a handle the caller sent (live 2026-09-07).
const CALLER_REFERENCE_SENTENCES: &[&str] = &[
    "Item with id ",
    "Conversation with id ",
    "Previous response with id ",
    "Response with id ",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::param_attribution::rejected_detail;

    #[test]
    fn an_aggregators_generic_sentence_yields_to_the_upstream_message() {
        let body = r#"{"error":{"message":"Provider returned error","code":400,
            "metadata":{"raw":"{\"error\":{\"message\":\"Input exceeds the maximum context window.\",\"type\":\"invalid_request_error\"}}",
            "provider_name":"Relace"}}}"#;
        assert_eq!(
            rejected_detail(Dialect::OpenAiCompatible, body, &[]).as_deref(),
            Some("Relace: Input exceeds the maximum context window.")
        );
        // A plain-text raw is relayed whole; a specific aggregator sentence is kept.
        let plain = r#"{"error":{"message":"Provider returned error","code":400,
            "metadata":{"raw":"model is overloaded, try again","provider_name":"Fugu"}}}"#;
        assert_eq!(
            rejected_detail(Dialect::OpenAiCompatible, plain, &[]).as_deref(),
            Some("Fugu: model is overloaded, try again")
        );
        let specific = r#"{"error":{"message":"temperature must be <= 1","code":400,
            "metadata":{"raw":"{\"error\":{\"message\":\"ignored\"}}"}}}"#;
        assert_eq!(
            rejected_detail(Dialect::OpenAiCompatible, specific, &[]).as_deref(),
            Some("temperature must be <= 1")
        );
        // Without metadata the generic sentence stays what it was.
        let bare = r#"{"error":{"message":"Provider returned error","code":400}}"#;
        assert_eq!(
            rejected_detail(Dialect::OpenAiCompatible, bare, &[]).as_deref(),
            Some("Provider returned error")
        );
    }

    #[test]
    fn a_404_refusing_a_caller_reference_is_the_callers_error_not_a_missing_model() {
        let item = r#"{"error":{"message":"Item with id 'rs_0' not found. Items are not persisted when `store` is set to false.","type":"invalid_request_error","param":"input","code":null}}"#;
        assert!(rejected_caller_reference_not_found(
            Dialect::OpenAiResponses,
            item
        ));
        assert!(rejected_caller_reference_not_found(
            Dialect::OpenAiCompatible,
            item
        ));
        let conversation = r#"{"error":{"message":"Conversation with id 'conv_0' not found.","type":"invalid_request_error","param":null,"code":null}}"#;
        assert!(rejected_caller_reference_not_found(
            Dialect::OpenAiResponses,
            conversation
        ));
        let model = r#"{"error":{"message":"The model `x` does not exist.","type":"invalid_request_error","param":"model","code":"model_not_found"}}"#;
        assert!(!rejected_caller_reference_not_found(
            Dialect::OpenAiResponses,
            model
        ));
        // A model reported as a 404 WITHOUT the documented code, naming neither
        // `input` nor a reference shape, keeps the lane policy.
        let uncoded_model = r#"{"error":{"message":"The model `x` was not found.","type":"invalid_request_error","param":"model","code":null}}"#;
        assert!(!rejected_caller_reference_not_found(
            Dialect::OpenAiResponses,
            uncoded_model
        ));
        let anthropic =
            r#"{"type":"error","error":{"type":"not_found_error","message":"model: x"}}"#;
        assert!(!rejected_caller_reference_not_found(
            Dialect::AnthropicMessages,
            anthropic
        ));
        assert!(!rejected_caller_reference_not_found(
            Dialect::OpenAiResponses,
            "<html>"
        ));
    }
}
