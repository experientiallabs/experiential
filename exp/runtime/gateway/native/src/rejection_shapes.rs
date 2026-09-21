//! Provider rejection SHAPES the pre-stream and in-stream failure paths read
//! off a client-error body: an aggregator's generic sentence hiding the
//! upstream's own, a lane limitation a caller cannot fix, an aggregator
//! routing gate that is not a credential verdict, and a 404 that refuses a
//! handle the caller sent rather than naming a missing model, and a content
//! filter verdict stated inside a completion body rather than an error
//! envelope. Each reads only the dialect's documented fields and never logs
//! body text.

use serde_json::Value;

use crate::dialects::Dialect;
use crate::error_envelope::{openai_family_envelope, parse_error_document};
use crate::param_attribution::{error_message_field, rejected_model_not_found};
use crate::stream_errors::is_refusal_code;

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
    let Some(value) = parse_error_document(body) else {
        return false;
    };
    error_message_field(dialect, &value).is_some_and(|message| {
        let lowered = message.to_ascii_lowercase();
        LANE_LIMITATION_PHRASES
            .iter()
            .any(|phrase| lowered.contains(phrase))
    })
}

/// Sentences a provider answers under a client-error status when the
/// ACCOUNT, not the request, is what it refuses: the house credential is out
/// of prepaid balance or quota. Novita answers `400 "Insufficient quota
/// available for instant inference. trace_id: …"` on a drained account
/// (gpt-5.6-sol, 2026-09-16 05:28Z), which a status-only read filed as the
/// caller's `invalid_request`: no failover, no exhaustion-sweep signal, a
/// customer 400 for the operator's balance. Narrower than the in-stream
/// `QUOTA_PHRASES` on purpose (no bare "billing"): a pre-stream 4xx sentence
/// decides a class the ladder acts on, so only unambiguous funding wording
/// qualifies.
const ACCOUNT_QUOTA_PHRASES: &[&str] = &[
    "insufficient quota",
    "insufficient balance",
    "insufficient credits",
    "insufficient funds",
    "not enough balance",
    "exceeded your current quota",
];

/// Whether a 4xx body's error SENTENCE says the provider ACCOUNT cannot pay
/// (see [`ACCOUNT_QUOTA_PHRASES`]). Only the dialect's message field is read.
pub fn rejected_by_account_quota(dialect: Dialect, body: &str) -> bool {
    let Some(value) = parse_error_document(body) else {
        return false;
    };
    error_message_field(dialect, &value).is_some_and(|message| {
        let lowered = message.to_ascii_lowercase();
        ACCOUNT_QUOTA_PHRASES
            .iter()
            .any(|phrase| lowered.contains(phrase))
    })
}

/// The head of Novita's relay sentence when ITS decoder choked on the upstream
/// error it received: "failed to decode error response: json: cannot unmarshal
/// number into Go struct field ResponseError.error.code of type string, raw:
/// {"error":{"code":0,"message":"Exceeded maximum number of images (50)
/// allowed in the request."}} trace_id: …" (gpt-5.6-luna on the Responses
/// wire, 2026-09-16 04:58-05:31Z, six customer 400s). The relay's own
/// sentence says nothing; the UPSTREAM document after `raw: ` carries the
/// real code and sentence (a caller's image limit here, a 429 elsewhere).
const DECODE_FAILURE_HEAD: &str = "failed to decode error response:";
const DECODE_FAILURE_RAW_MARKER: &str = "raw: ";

/// The upstream `(code, sentence)` a relay's decode-failure sentence embeds,
/// or `None` for any other sentence. The embedded text is parsed as its first
/// JSON document (the relay appends its own trace id after it); a truncated
/// document still yields its `"message"` field by a bounded scan, because the
/// relay cuts long upstream bodies.
pub fn relayed_decode_failure(message: &str) -> Option<(Option<String>, String)> {
    let trimmed = message.trim_start();
    if !trimmed
        .get(..DECODE_FAILURE_HEAD.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(DECODE_FAILURE_HEAD))
    {
        return None;
    }
    let raw_start = trimmed.find(DECODE_FAILURE_RAW_MARKER)? + DECODE_FAILURE_RAW_MARKER.len();
    let raw = trimmed[raw_start..].trim_start();
    if let Some(document) = parse_error_document(raw) {
        // A zero code is the upstream's "no code" (it is what broke the relay's
        // decoder), never an HTTP status: it must not classify as one.
        let code = openai_family_envelope(&document)
            .and_then(|envelope| envelope.code)
            .filter(|code| code != "0");
        let sentence = json_error_sentence(&document)?;
        return Some((code, sentence));
    }
    Some((None, quoted_message_field(raw)?))
}

/// The `"message":"…"` value of one (possibly unterminated) JSON fragment.
fn quoted_message_field(raw: &str) -> Option<String> {
    const KEY: &str = "\"message\":\"";
    let start = raw.find(KEY)? + KEY.len();
    let rest = &raw[start..];
    let end = rest.find('"').unwrap_or(rest.len());
    let text = rest[..end].trim();
    (!text.is_empty()).then(|| text.to_string())
}

/// The upstream `(code, sentence)` behind a 4xx body whose sentence is a
/// relay decode failure (see [`relayed_decode_failure`]); only the dialect's
/// message field is read.
pub fn rejected_via_decode_failure(
    dialect: Dialect,
    body: &str,
) -> Option<(Option<String>, String)> {
    let value = parse_error_document(body)?;
    let message = error_message_field(dialect, &value)?;
    relayed_decode_failure(message)
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
    let Some(value) = parse_error_document(body) else {
        return false;
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

/// OpenAI-family `error.code` refusing a replayed reasoning item's
/// `encrypted_content`.
const INVALID_ENCRYPTED_CONTENT_CODE: &str = "invalid_encrypted_content";

/// The fixed head and verdict of OpenAI's sentence for that refusal ("The
/// encrypted content <blob> could not be verified. Reason: ..."). The reason
/// tail varies and a relay may append its own trace id, so only these two
/// fragments are matched.
const ENCRYPTED_CONTENT_SENTENCE_HEAD: &str = "The encrypted content ";
const ENCRYPTED_CONTENT_SENTENCE_VERDICT: &str = " could not be verified";

/// Whether a 4xx body is the native Responses wire refusing replayed encrypted
/// reasoning: a reasoning item's `encrypted_content` was sealed by another
/// organization or tenant, or is not a payload the provider issued at all.
///
/// The body is read through the shared OpenAI-family envelope reader, so
/// every spelling the Responses-wire lanes answer decides the same way:
/// OpenAI and Azure carry `code: invalid_encrypted_content`; Novita's relay
/// re-envelopes the refusal flat with the generic `type` and its own trace
/// id, keeping only OpenAI's sentence, so the sentence's fixed head and
/// verdict decide too; OpenRouter's relay wraps OpenAI's own document under
/// `error.metadata.raw`, which is read recursively. Other dialects have no
/// such item and keep the plain client-error verdict.
pub fn rejected_encrypted_reasoning(dialect: Dialect, body: &str) -> bool {
    if dialect != Dialect::OpenAiResponses {
        return false;
    }
    parse_error_document(body).is_some_and(|document| encrypted_reasoning_verdict(&document))
}

/// The verdict on one parsed error document, following an aggregator's
/// relayed upstream document one level down.
fn encrypted_reasoning_verdict(document: &Value) -> bool {
    let Some(envelope) = openai_family_envelope(document) else {
        return false;
    };
    refuses_encrypted_reasoning(envelope.code.as_deref(), envelope.message)
        || envelope
            .error_object
            .and_then(relayed_upstream_document)
            .is_some_and(|upstream| encrypted_reasoning_verdict(&upstream))
}

/// Whether one provider error, as its code token and sentence, refuses
/// replayed encrypted reasoning. Shared by the pre-stream body predicate and
/// the in-stream `response.failed` classification: OpenRouter's Responses
/// relay answers 200 and then fails the stream with OpenAI's sentence under
/// the code `invalid_prompt`, so the sentence decides there as well.
pub(crate) fn refuses_encrypted_reasoning(code: Option<&str>, message: Option<&str>) -> bool {
    code == Some(INVALID_ENCRYPTED_CONTENT_CODE)
        || message.is_some_and(names_refused_encrypted_content)
}

/// Whether one provider sentence is OpenAI's refusal of an encrypted payload.
fn names_refused_encrypted_content(message: &str) -> bool {
    let sentence = message.trim_start();
    sentence.starts_with(ENCRYPTED_CONTENT_SENTENCE_HEAD)
        && sentence.contains(ENCRYPTED_CONTENT_SENTENCE_VERDICT)
}

/// The upstream error DOCUMENT an aggregator carried in `metadata.raw`, when
/// that field holds JSON as a string or as an object.
fn relayed_upstream_document(error: &Value) -> Option<Value> {
    match error.get("metadata")?.get("raw")? {
        Value::String(text) => parse_error_document(text),
        raw @ Value::Object(_) => Some(raw.clone()),
        _ => None,
    }
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
    let Some(value) = parse_error_document(body) else {
        return false;
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

/// A content-filter verdict a provider states INSIDE a completion body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilteredCompletion {
    /// The provider's refusal code (`content_filter`), read from the choice's
    /// `content_filter_results.error.code`, else its finish reason.
    pub code: String,
    /// The provider's sentence naming what was blocked, when it gave one.
    pub message: Option<String>,
}

/// Read a 4xx body that is a CHAT COMPLETION carrying a content-filter finish
/// rather than an error envelope.
///
/// Azure AI Foundry's model-inference surface (DeepSeek-V4-Flash, captured
/// live 2026-09-15) answers a non-streaming request whose OUTPUT its content
/// safety layer blocked with HTTP 400 and a `chat.completion` object: no
/// top-level `error`, `choices[0].finish_reason = "content_filter"`,
/// `choices[0].message.content = ""`, and the verdict under
/// `choices[0].content_filter_results.error` (`{code: "content_filter",
/// message: "Response content blocked by label 'MultiSeverity_ViolenceScore'."}`).
/// The envelope readers see no error there, so 438 such refusals in 48h
/// settled as the generic request-shape 400 with no detail. The finish reason
/// is the authoritative verdict: a `content_filter`/`safety` finish on the
/// first choice is the model's answer to the content, never a request-shape
/// error, and the nested code names the category. Only the OpenAI-compatible
/// chat wire is read; other dialects state their verdicts in their own
/// envelopes.
pub fn content_filtered_completion(dialect: Dialect, body: &str) -> Option<FilteredCompletion> {
    if dialect != Dialect::OpenAiCompatible {
        return None;
    }
    let value = parse_error_document(body)?;
    let choice = value.get("choices")?.as_array()?.first()?.as_object()?;
    let finish = choice.get("finish_reason")?.as_str()?;
    if !matches!(finish, "content_filter" | "safety") {
        return None;
    }
    let verdict = choice
        .get("content_filter_results")
        .and_then(|results| results.get("error"))
        .and_then(Value::as_object);
    let nested_code = verdict
        .and_then(|error| error.get("code"))
        .and_then(Value::as_str)
        .filter(|code| is_refusal_code(Some(code)))
        .map(str::to_string);
    Some(FilteredCompletion {
        code: nested_code.unwrap_or_else(|| finish.to_string()),
        message: verdict
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
            .filter(|message| !message.trim().is_empty())
            .map(str::to_string),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::param_attribution::rejected_detail;

    #[test]
    fn a_resellers_flat_lane_limitation_sentence_is_read_on_the_compatible_dialect() {
        // Novita's envelope has no `error` object; the sentence still decides.
        let flat = r#"{"code":400,"reason":"INVALID_REQUEST_BODY","message":"System message must be at the beginning.","metadata":{}}"#;
        assert!(rejected_by_lane_limitation(Dialect::OpenAiCompatible, flat));
        assert!(!rejected_by_lane_limitation(
            Dialect::AnthropicMessages,
            flat
        ));
        let other =
            r#"{"code":400,"reason":"INVALID_REQUEST_BODY","message":"max_tokens too large"}"#;
        assert!(!rejected_by_lane_limitation(
            Dialect::OpenAiCompatible,
            other
        ));
    }

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

    /// The Azure Foundry DeepSeek body captured live on 2026-09-15 (key
    /// redacted, nothing else edited).
    const AZURE_FILTERED_COMPLETION: &str = r#"{"id":"chatcmpl-802d5a802bf84292896e446052595","model":"","choices":[{"index":0,"message":{"role":"assistant","content":""},"finish_reason":"content_filter","content_filter_results":{"error":{"code":"content_filter","message":"Response content blocked by label 'MultiSeverity_ViolenceScore'."}}}],"usage":{"prompt_tokens":55,"total_tokens":55},"created":1789466192,"object":"chat.completion","prompt_filter_results":null}"#;

    #[test]
    fn azure_filtered_completion_body_is_read_as_the_provider_verdict() {
        let filtered =
            content_filtered_completion(Dialect::OpenAiCompatible, AZURE_FILTERED_COMPLETION)
                .expect("a content_filter finish inside a completion is a verdict");
        assert_eq!(filtered.code, "content_filter");
        assert_eq!(
            filtered.message.as_deref(),
            Some("Response content blocked by label 'MultiSeverity_ViolenceScore'.")
        );
        // The verdict rides the finish reason even without the nested error.
        let bare = content_filtered_completion(
            Dialect::OpenAiCompatible,
            r#"{"choices":[{"index":0,"message":{"role":"assistant","content":""},"finish_reason":"content_filter"}]}"#,
        )
        .expect("finish reason alone names the verdict");
        assert_eq!(bare.code, "content_filter");
        assert_eq!(bare.message, None);
    }

    #[test]
    fn completions_that_are_not_filtered_and_other_dialects_are_not_verdicts() {
        let stopped = r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#;
        assert_eq!(
            content_filtered_completion(Dialect::OpenAiCompatible, stopped),
            None
        );
        assert_eq!(
            content_filtered_completion(Dialect::OpenAiResponses, AZURE_FILTERED_COMPLETION),
            None
        );
        assert_eq!(
            content_filtered_completion(
                Dialect::OpenAiCompatible,
                r#"{"error":{"code":"content_filter"}}"#
            ),
            None
        );
        assert_eq!(
            content_filtered_completion(Dialect::OpenAiCompatible, "not json"),
            None
        );
    }

    #[test]
    fn the_responses_wire_names_refused_encrypted_reasoning_by_code_alone() {
        // OpenAI's two documented reason sentences carry the same code.
        let foreign_organization = r#"{"error":{"message":"The encrypted content for item rs_0d09 could not be verified. Reason: Encrypted content organization_id did not match the target organization.","type":"invalid_request_error","param":null,"code":"invalid_encrypted_content"}}"#;
        let unparseable = r#"{"error":{"message":"The encrypted content rsn_...hA== could not be verified. Reason: Encrypted content could not be decrypted or parsed.","type":"invalid_request_error","param":null,"code":"invalid_encrypted_content"}}"#;
        assert!(rejected_encrypted_reasoning(
            Dialect::OpenAiResponses,
            foreign_organization
        ));
        assert!(rejected_encrypted_reasoning(
            Dialect::OpenAiResponses,
            unparseable
        ));
        // OpenAI's sentence decides even when a relay dropped the code, and
        // another sentence under another code does not.
        let no_code = r#"{"error":{"message":"The encrypted content rs_1 could not be verified.","type":"invalid_request_error","code":null}}"#;
        assert!(rejected_encrypted_reasoning(
            Dialect::OpenAiResponses,
            no_code
        ));
        let other_sentence = r#"{"error":{"message":"Invalid value for 'input[1].id': expected a value.","type":"invalid_request_error","code":"invalid_value"}}"#;
        assert!(!rejected_encrypted_reasoning(
            Dialect::OpenAiResponses,
            other_sentence
        ));
        let bare_message = r#"{"message":"The encrypted content rs_1 could not be verified."}"#;
        assert!(!rejected_encrypted_reasoning(
            Dialect::OpenAiResponses,
            bare_message
        ));
        // Only the Responses wire carries reasoning items.
        assert!(!rejected_encrypted_reasoning(
            Dialect::OpenAiCompatible,
            foreign_organization
        ));
        assert!(!rejected_encrypted_reasoning(
            Dialect::OpenAiResponses,
            "not json"
        ));
    }

    #[test]
    fn the_shared_predicate_reads_the_code_or_the_sentence() {
        // OpenRouter's Responses relay fails the stream under `invalid_prompt`
        // with OpenAI's sentence (live, gpt-5.6-sol, 2026-09-16 00:25Z).
        assert!(refuses_encrypted_reasoning(
            Some("invalid_prompt"),
            Some("The encrypted content rsn_...Ypi9 could not be verified. Reason: Encrypted content could not be decrypted or parsed.")
        ));
        assert!(refuses_encrypted_reasoning(
            Some("invalid_encrypted_content"),
            None
        ));
        assert!(!refuses_encrypted_reasoning(
            Some("invalid_prompt"),
            Some("Invalid prompt: we've limited access to this content.")
        ));
        assert!(!refuses_encrypted_reasoning(None, None));
    }

    #[test]
    fn a_relayed_encrypted_reasoning_verdict_is_read_through_the_aggregator_envelope() {
        // OpenRouter's Responses relay: numeric status, generic sentence, and
        // OpenAI's own document as a JSON string under metadata.raw.
        let openrouter = r#"{"error":{"message":"Provider returned error","code":400,
            "metadata":{"raw":"{\"error\":{\"message\":\"The encrypted content for item rs_0d09 could not be verified. Reason: Encrypted content organization_id did not match the target organization.\",\"type\":\"invalid_request_error\",\"param\":null,\"code\":\"invalid_encrypted_content\"}}",
            "provider_name":"OpenAI"}}}"#;
        assert!(rejected_encrypted_reasoning(
            Dialect::OpenAiResponses,
            openrouter
        ));
        // The same envelope carrying the document as an object.
        let object_raw = r#"{"error":{"message":"Provider returned error","code":400,
            "metadata":{"raw":{"error":{"message":"x","type":"invalid_request_error","code":"invalid_encrypted_content"}},
            "provider_name":"OpenAI"}}}"#;
        assert!(rejected_encrypted_reasoning(
            Dialect::OpenAiResponses,
            object_raw
        ));
        // Novita's Responses relay re-envelopes the refusal flat: the generic
        // `type` as its only token, `code: 0`, and OpenAI's sentence with the
        // relay's trace id appended (live, gpt-5.6-luna, 2026-09-15 16:49Z).
        let novita = r#"{"code":0,"message":"The encrypted content gAAA...WA== could not be verified. Reason: Encrypted content could not be decrypted or parsed. trace_id: 9f3c2b7a1d4e","type":"invalid_request_error"}"#;
        assert!(rejected_encrypted_reasoning(
            Dialect::OpenAiResponses,
            novita
        ));
        let novita_other = r#"{"code":0,"message":"Function tools with reasoning_effort are not supported. trace_id: 9f3c","type":"invalid_request_error"}"#;
        assert!(!rejected_encrypted_reasoning(
            Dialect::OpenAiResponses,
            novita_other
        ));
        // Azure OpenAI's v1 Responses endpoint answers OpenAI's shape verbatim.
        let azure = r#"{"error":{"message":"The encrypted content gAAA...ke== could not be verified. Reason: Encrypted content could not be decrypted or parsed.","type":"invalid_request_error","param":null,"code":"invalid_encrypted_content"}}"#;
        assert!(rejected_encrypted_reasoning(
            Dialect::OpenAiResponses,
            azure
        ));
        // A relayed document under another code, or raw text, is not the verdict.
        let other = r#"{"error":{"message":"Provider returned error","code":400,
            "metadata":{"raw":"{\"error\":{\"message\":\"bad\",\"code\":\"invalid_value\"}}"}}}"#;
        assert!(!rejected_encrypted_reasoning(
            Dialect::OpenAiResponses,
            other
        ));
        let text = r#"{"error":{"message":"Provider returned error","code":400,
            "metadata":{"raw":"invalid_encrypted_content"}}}"#;
        assert!(!rejected_encrypted_reasoning(
            Dialect::OpenAiResponses,
            text
        ));
    }
}
