//! One reader for the error envelopes the OpenAI-compatible family answers.
//!
//! The OpenAI wire documents exactly one error shape, `{error: {message,
//! type, code, param}}`, but the lanes that speak the compatible dialect are
//! resellers, self-hosted engines, and first-party clones, and each spells a
//! rejection its own way. Reading only the documented shape lost the
//! provider's sentence for whole lanes (2026-09-15 ledger, 24h: 470 Novita,
//! 438 Azure Foundry, 21 Experiential Cloud and 11 xAI client errors settled
//! with the generic "verify the request fields" and `error_detailed=false`),
//! so the caller had nothing to act on and the operator nothing to triage.
//!
//! Shapes read, each captured live:
//!
//! | provider                         | envelope                                          |
//! |----------------------------------|---------------------------------------------------|
//! | OpenAI, Azure OpenAI, OpenRouter | `{error: {message, type, code, param}}`            |
//! | xAI                              | `{code: "invalid-argument", error: "<sentence>"}`  |
//! | Novita                           | `{code: 400, reason: "INVALID_…", message, metadata}` |
//! | vLLM engines (Azure Foundry MaaS)| `{object: "error", message, type, param, code}`    |
//! | FastAPI origins                  | `{detail: "<sentence>"}` / `{detail: [{loc, msg, type}]}` |
//! | Azure API Management             | `{statusCode, message}`                            |
//!
//! Every reader takes only the envelope's documented fields (a sentence, a
//! vocabulary token, a parameter path), never echoed request data. The body
//! is parsed as its FIRST JSON document, so a provider that appends help text
//! after the JSON (Azure Foundry's "Please check this guide …" line) still
//! yields its envelope.

use serde_json::Value;

/// The facts one error envelope carries, whichever spelling it used.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct ErrorEnvelope<'a> {
    /// The provider's human sentence.
    pub message: Option<&'a str>,
    /// The provider's vocabulary token: `code` when it is a string or number,
    /// else `type`, else the flat `reason`.
    pub code: Option<String>,
    /// The parameter path the provider named.
    pub param: Option<String>,
    /// The `error` OBJECT when the envelope has one (aggregator metadata such
    /// as OpenRouter's `metadata.raw` lives inside it).
    pub error_object: Option<&'a Value>,
}

/// Parse one error body as its first JSON document, tolerating trailing text.
///
/// Azure Foundry's model-inference surface answers a vLLM error object
/// followed by a plain-text help line; strict parsing rejected the whole body
/// and dropped the sentence. Only the leading document is read.
pub(crate) fn parse_error_document(body: &str) -> Option<Value> {
    let mut documents = serde_json::Deserializer::from_str(body).into_iter::<Value>();
    documents.next()?.ok()
}

/// Read the OpenAI-family envelope out of one parsed error document.
///
/// Returns `None` when the document names no error sentence, code, or
/// parameter in any of the documented spellings.
pub(crate) fn openai_family_envelope(value: &Value) -> Option<ErrorEnvelope<'_>> {
    let error = value.get("error").filter(|error| !error.is_null());
    let envelope = match error {
        // `{error: {message, type, code, param}}`: the documented shape.
        Some(Value::Object(_)) => nested_envelope(error?),
        // `{code, error: "<sentence>"}`: xAI's shape.
        Some(Value::String(sentence)) => ErrorEnvelope {
            message: Some(sentence.as_str()),
            code: token(value.get("code")),
            param: None,
            error_object: None,
        },
        _ => flat_envelope(value),
    };
    let named = envelope.message.is_some() || envelope.code.is_some() || envelope.param.is_some();
    named.then_some(envelope)
}

fn nested_envelope(error: &Value) -> ErrorEnvelope<'_> {
    ErrorEnvelope {
        message: error.get("message").and_then(Value::as_str),
        code: token(error.get("code")).or_else(|| token(error.get("type"))),
        param: error
            .get("param")
            .and_then(Value::as_str)
            .map(str::to_string),
        error_object: Some(error),
    }
}

/// The flat spellings: vLLM's `{object: "error", message, type, param,
/// code}`, Novita's `{code, reason, message}`, API Management's
/// `{statusCode, message}`, and FastAPI's `{detail}`.
fn flat_envelope(value: &Value) -> ErrorEnvelope<'_> {
    if let Some(detail) = value.get("detail") {
        return fastapi_envelope(detail);
    }
    // A bare `{message}` beside no token or error marker is not an envelope
    // on this family (Bedrock's bare message is its own dialect's shape): a
    // flat body must also carry a `reason`, `code`, `type`, `statusCode`, or
    // `object: "error"` to be read.
    let declares_error = value.get("object").and_then(Value::as_str) == Some("error")
        || ["reason", "code", "type", "statusCode"]
            .iter()
            .any(|key| value.get(*key).is_some_and(|field| !field.is_null()));
    if !declares_error {
        return ErrorEnvelope::default();
    }
    ErrorEnvelope {
        message: value.get("message").and_then(Value::as_str),
        // Novita's `reason` is the vocabulary token (`INVALID_PARAMETER`);
        // its numeric `code` is only the status. vLLM carries the class in
        // `type` (`BadRequestError`) beside a numeric `code`.
        code: value
            .get("reason")
            .and_then(Value::as_str)
            .filter(|reason| !reason.is_empty())
            .map(str::to_string)
            .or_else(|| token(value.get("code")))
            .or_else(|| token(value.get("type")))
            .or_else(|| token(value.get("statusCode"))),
        param: value
            .get("param")
            .and_then(Value::as_str)
            .map(str::to_string),
        error_object: None,
    }
}

/// FastAPI's `detail`: a plain sentence, or the validation list whose first
/// entry carries `msg` (the sentence), `type` (the token), and `loc` (the
/// field path, rendered `messages[0].content` with the leading `body` root
/// dropped).
fn fastapi_envelope(detail: &Value) -> ErrorEnvelope<'_> {
    match detail {
        Value::String(sentence) => ErrorEnvelope {
            message: Some(sentence.as_str()),
            ..ErrorEnvelope::default()
        },
        Value::Array(entries) => {
            let first = entries.first();
            ErrorEnvelope {
                message: first
                    .and_then(|entry| entry.get("msg"))
                    .and_then(Value::as_str),
                code: first.and_then(|entry| token(entry.get("type"))),
                param: first
                    .and_then(|entry| entry.get("loc"))
                    .and_then(Value::as_array)
                    .and_then(|loc| location_path(loc)),
                error_object: None,
            }
        }
        _ => ErrorEnvelope::default(),
    }
}

/// Render one FastAPI `loc` array as a parameter path.
fn location_path(loc: &[Value]) -> Option<String> {
    let mut path = String::new();
    for (index, segment) in loc.iter().enumerate() {
        match segment {
            Value::String(name) => {
                if index == 0 && name == "body" {
                    continue;
                }
                if !path.is_empty() {
                    path.push('.');
                }
                path.push_str(name);
            }
            Value::Number(number) => {
                if path.is_empty() {
                    return None;
                }
                path.push('[');
                path.push_str(&number.to_string());
                path.push(']');
            }
            _ => return None,
        }
    }
    (!path.is_empty()).then_some(path)
}

/// A string or numeric field as its vocabulary token; anything else is not
/// a token.
fn token(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(text) if !text.is_empty() => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(body: &str) -> ErrorEnvelope<'static> {
        // Leak the parsed document so the borrowed envelope outlives the test body.
        let value: &'static Value =
            Box::leak(Box::new(parse_error_document(body).expect("body parses")));
        openai_family_envelope(value).expect("envelope recognized")
    }

    #[test]
    fn the_documented_openai_shape_reads_every_field() {
        let read = envelope(
            r#"{"error": {"message": "Unknown parameter: 'top_k'.", "type": "invalid_request_error",
                "param": "top_k", "code": "unknown_parameter"}}"#,
        );
        assert_eq!(read.message, Some("Unknown parameter: 'top_k'."));
        assert_eq!(read.code.as_deref(), Some("unknown_parameter"));
        assert_eq!(read.param.as_deref(), Some("top_k"));
        assert!(read.error_object.is_some());
        // A null code falls back to the family type.
        let typed = envelope(
            r#"{"error": {"code": null, "type": "invalid_request_error", "message": "x"}}"#,
        );
        assert_eq!(typed.code.as_deref(), Some("invalid_request_error"));
    }

    #[test]
    fn xai_spells_the_sentence_as_a_string_error_beside_a_code() {
        // Exact body captured live from api.x.ai (2026-09-15).
        let read = envelope(
            r#"{"code":"invalid-argument","error":"Incorrect API key provided. You can obtain an API key from https://console.x.ai."}"#,
        );
        assert_eq!(
            read.message,
            Some(
                "Incorrect API key provided. You can obtain an API key from https://console.x.ai."
            )
        );
        assert_eq!(read.code.as_deref(), Some("invalid-argument"));
        assert_eq!(read.param, None);
        assert!(read.error_object.is_none());
    }

    #[test]
    fn novita_spells_a_flat_envelope_whose_reason_is_the_token() {
        // Exact body captured live from api.novita.ai (2026-09-15).
        let read = envelope(
            r#"{"code":401,"reason":"FAILED_TO_AUTH","message":"failed to authenticate API key","metadata":{}}"#,
        );
        assert_eq!(read.message, Some("failed to authenticate API key"));
        assert_eq!(read.code.as_deref(), Some("FAILED_TO_AUTH"));
    }

    #[test]
    fn vllm_flat_envelope_survives_trailing_help_text() {
        // Exact body captured live from an Azure Foundry DeepSeek deployment
        // (2026-09-15): a vLLM error object, then a plain-text help line.
        let body = "{\"object\":\"error\",\"message\":\"Tool 'g' not found in tools list.\",\
                    \"type\":\"BadRequestError\",\"param\":null,\"code\":400}\n\
                    Please check this guide to understand why this error code might have been returned \n\
                    https://docs.microsoft.com/en-us/azure/machine-learning/how-to-troubleshoot-online-endpoints#http-status-codes\n";
        assert!(
            serde_json::from_str::<Value>(body).is_err(),
            "the fixture must reproduce the strict-parse failure"
        );
        let read = envelope(body);
        assert_eq!(read.message, Some("Tool 'g' not found in tools list."));
        // The numeric status is the token when no reason names the class.
        assert_eq!(read.code.as_deref(), Some("400"));
        assert_eq!(read.param, None);
    }

    #[test]
    fn fastapi_detail_reads_a_sentence_or_the_first_validation_entry() {
        let sentence = envelope(r#"{"detail": "An image input is required."}"#);
        assert_eq!(sentence.message, Some("An image input is required."));
        assert_eq!(sentence.code, None);
        let validation = envelope(
            r#"{"detail": [{"loc": ["body", "messages", 0, "content"], "msg": "field required",
                "type": "value_error.missing"}]}"#,
        );
        assert_eq!(validation.message, Some("field required"));
        assert_eq!(validation.code.as_deref(), Some("value_error.missing"));
        assert_eq!(validation.param.as_deref(), Some("messages[0].content"));
    }

    #[test]
    fn api_management_status_code_envelope_reads_its_message() {
        let read = envelope(r#"{"statusCode": 400, "message": "Invalid request body."}"#);
        assert_eq!(read.message, Some("Invalid request body."));
        assert_eq!(read.code.as_deref(), Some("400"));
    }

    #[test]
    fn a_body_naming_no_error_is_not_an_envelope() {
        for body in [
            r#"{}"#,
            r#"{"choices": []}"#,
            r#"{"error": null}"#,
            r#"{"detail": 5}"#,
            r#"{"message": "something happened"}"#,
        ] {
            let value = parse_error_document(body).expect("parses");
            assert!(openai_family_envelope(&value).is_none(), "{body}");
        }
        assert!(parse_error_document("<html>").is_none());
        assert!(parse_error_document("").is_none());
    }
}
