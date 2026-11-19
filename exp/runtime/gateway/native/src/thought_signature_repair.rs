//! Gemini's half of the replay repair (`replay_repair`): a thought signature
//! one Google backend minted is refused by another.
//!
//! Gemini signs a function call with an opaque `thoughtSignature`, and the
//! caller replays it on the next turn (Chat and Responses as
//! `extra_content.google.thought_signature`, generateContent as the part's
//! own field). The Gemini API and Vertex AI each verify only the signatures
//! they issued: a conversation whose earlier turn one served and whose later
//! turn routes to the other (failover, throttling, latency demotion or plain
//! ordering) is refused with HTTP 400 "Invalid thought signature." before any
//! output. The refusal reads as the caller's `invalid_request`, which never
//! fails over, so the turn failed outright (production, 2026-11-19, one model
//! served by both backends: 25 refused turns from 5 organizations in 35
//! minutes on its Vertex lane).
//!
//! The repair is the encrypted-reasoning one: the waterfall re-dials the SAME
//! rung once with every replayed signature replaced by Google's documented
//! bypass value ([`THOUGHT_SIGNATURE_BYPASS`], what the payload builder
//! already sends for a call whose signature was not retained), and the worker
//! remembers the refused signatures for that caller AND that rung, so a later
//! turn dialing the same rung replaces them before its first dial while a
//! turn dialing the backend that minted them still sends them verbatim.
//! Nothing changes for a signature its own backend receives.

use serde_json::Value;

use crate::dialects::Dialect;
use crate::error_envelope::parse_error_document;
use crate::errors::{Failure, FailureClass};
use crate::replay_repair::{
    encrypted_payloads, payload_digest, without_encrypted_content_where, MEMORY,
};
use crate::waterfall::DeploymentWire;

/// The wire dialect whose replayed history carries thought signatures.
pub const GEMINI_DIALECT: &str = "gemini_generate_content";

/// Google's documented stand-in for a replayed call whose real signature is
/// not available (the python payload builder's `GEMINI_THOUGHT_SIGNATURE_BYPASS`).
pub const THOUGHT_SIGNATURE_BYPASS: &str = "skip_thought_signature_validator";

/// Whether one provider sentence refuses a replayed thought signature
/// ("Invalid thought signature.", and the "Corrupted thought signature."
/// spelling of the same verdict). A MISSING signature is a different request
/// shape and is not matched.
pub(crate) fn refuses_thought_signature(message: &str) -> bool {
    let sentence = message.to_ascii_lowercase();
    (sentence.contains("thought signature") || sentence.contains("thought_signature"))
        && (sentence.contains("invalid") || sentence.contains("corrupt"))
}

/// Whether a 4xx body is the Gemini wire refusing a replayed thought
/// signature. Google's envelope is `{"error": {"message": ...}}`; a streaming
/// Vertex endpoint may wrap it in a one-element array, and a bare sentence is
/// read as itself.
pub fn rejected_thought_signature(dialect: Dialect, body: &str) -> bool {
    if dialect != Dialect::GeminiGenerateContent {
        return false;
    }
    let Some(document) = parse_error_document(body) else {
        return refuses_thought_signature(body.trim());
    };
    let envelope = match &document {
        Value::Array(items) => items.first(),
        other => Some(other),
    };
    envelope
        .and_then(|value| value.get("error"))
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .is_some_and(refuses_thought_signature)
}

/// Every replayed thought signature the repair can replace, in order (the
/// bypass value itself is not one).
pub fn thought_signatures(payload: &Value) -> Vec<&str> {
    let Some(contents) = payload.get("contents").and_then(Value::as_array) else {
        return Vec::new();
    };
    contents
        .iter()
        .filter_map(|content| content.get("parts").and_then(Value::as_array))
        .flatten()
        .filter_map(|part| part.get("thoughtSignature").and_then(Value::as_str))
        .filter(|signature| *signature != THOUGHT_SIGNATURE_BYPASS)
        .collect()
}

/// The generateContent payload with the replayed signatures `strip` selects
/// replaced by [`THOUGHT_SIGNATURE_BYPASS`], or `None` when it selects none.
/// Every part keeps its position and every other field; a call stays paired
/// with its response.
pub fn without_thought_signatures_where(
    payload: &Value,
    mut strip: impl FnMut(&str) -> bool,
) -> Option<Value> {
    let mut repaired = payload.clone();
    let contents = repaired.get_mut("contents")?.as_array_mut()?;
    let mut replaced = false;
    for content in contents {
        let Some(parts) = content.get_mut("parts").and_then(Value::as_array_mut) else {
            continue;
        };
        for part in parts {
            let Some(object) = part.as_object_mut() else {
                continue;
            };
            let selected = object
                .get("thoughtSignature")
                .and_then(Value::as_str)
                .is_some_and(|signature| signature != THOUGHT_SIGNATURE_BYPASS && strip(signature));
            if selected {
                object.insert(
                    "thoughtSignature".to_string(),
                    Value::String(THOUGHT_SIGNATURE_BYPASS.to_string()),
                );
                replaced = true;
            }
        }
    }
    replaced.then_some(repaired)
}

/// The replayed provider state one rung's wire carries, and so may strip:
/// OpenAI-family `encrypted_content`, or Gemini thought signatures
/// (`thought_signature_repair`), which are Gemini's encrypted reasoning state
/// and share the repair's header, counters and operator line.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReplayedState {
    EncryptedContent,
    ThoughtSignatures,
}

impl ReplayedState {
    pub(crate) fn of(wire: &DeploymentWire) -> Self {
        if wire.dialect == GEMINI_DIALECT {
            Self::ThoughtSignatures
        } else {
            Self::EncryptedContent
        }
    }

    /// Every replayed payload the strip can remove from this payload.
    pub(crate) fn payloads(self, payload: &Value) -> Vec<&str> {
        match self {
            Self::EncryptedContent => encrypted_payloads(payload),
            Self::ThoughtSignatures => thought_signatures(payload),
        }
    }

    /// The payload with the selected replayed payloads stripped, if any.
    pub(crate) fn strip_where(
        self,
        payload: &Value,
        strip: impl FnMut(&str) -> bool,
    ) -> Option<Value> {
        match self {
            Self::EncryptedContent => without_encrypted_content_where(payload, strip),
            Self::ThoughtSignatures => without_thought_signatures_where(payload, strip),
        }
    }

    /// The memory key of one refused payload. A thought signature is also
    /// keyed by the rung that refused it: the backend that minted it still
    /// verifies it, so a later turn dialing that rung must send it verbatim.
    pub(crate) fn memory_key(self, scope: &str, wire: &DeploymentWire, content: &str) -> [u8; 32] {
        match self {
            Self::EncryptedContent => payload_digest(scope, content),
            Self::ThoughtSignatures => {
                payload_digest(&format!("{scope}\0{}", wire.deployment_id), content)
            }
        }
    }
}

/// The caller's answer when a refused thought signature survives the repair.
pub(crate) fn unrepaired_thought_signature(failure: &Failure) -> Failure {
    Failure::new(
        FailureClass::InvalidRequest,
        "The conversation replays a Gemini thought signature this model's provider could not verify (it was issued by another Google backend). Resend the request without the thought_signature fields, or start a new conversation",
    )
    .with_provider_detail(failure.provider_detail.clone())
    .with_rate_limit_facts(failure.rate_limit_headers.as_deref().cloned(), None)
    .with_tenant_origin(failure.tenant_origin)
}

/// Whether this rung already refused one of this caller's replayed thought
/// signatures. The attempt then dials plain: a Google cache overlay would
/// carry the signature in its cached prefix, beyond the repair's reach.
pub(crate) fn remembers_refused_signature(wire: &DeploymentWire, scope: Option<&str>) -> bool {
    let (Some(scope), ReplayedState::ThoughtSignatures) = (scope, ReplayedState::of(wire)) else {
        return false;
    };
    thought_signatures(&wire.upstream_payload)
        .into_iter()
        .any(|signature| {
            MEMORY.recall(&ReplayedState::ThoughtSignatures.memory_key(scope, wire, signature))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The refusal as the Vertex lane answered it in production.
    const VERTEX_REFUSAL: &str = r#"{"error":{"code":400,"message":"Invalid thought signature.","status":"INVALID_ARGUMENT"}}"#;

    fn conversation(signatures: &[&str]) -> Value {
        let calls: Vec<Value> = signatures
            .iter()
            .enumerate()
            .map(|(index, signature)| {
                json!({
                    "functionCall": {"name": "lookup", "args": {"index": index}},
                    "thoughtSignature": signature,
                })
            })
            .collect();
        json!({
            "contents": [
                {"role": "user", "parts": [{"text": "look it up"}]},
                {"role": "model", "parts": calls},
                {"role": "user", "parts": [{"functionResponse": {"name": "lookup", "response": {"ok": true}}}]},
            ],
            "generationConfig": {"maxOutputTokens": 64},
        })
    }

    #[test]
    fn the_production_refusal_is_a_thought_signature_rejection() {
        let gemini = Dialect::GeminiGenerateContent;
        assert!(rejected_thought_signature(gemini, VERTEX_REFUSAL));
        assert!(rejected_thought_signature(
            gemini,
            &format!("[{VERTEX_REFUSAL}]")
        ));
        assert!(rejected_thought_signature(
            gemini,
            "Invalid thought signature."
        ));
        assert!(rejected_thought_signature(
            gemini,
            r#"{"error":{"code":400,"message":"Corrupted thought signature.","status":"INVALID_ARGUMENT"}}"#
        ));
        // Another wire's 400 with the same words is not Gemini's verdict.
        assert!(!rejected_thought_signature(
            Dialect::OpenAiCompatible,
            VERTEX_REFUSAL
        ));
        // A missing signature, or an unrelated 400, keeps its own verdict.
        assert!(!rejected_thought_signature(
            gemini,
            r#"{"error":{"code":400,"message":"Function call is missing a thought_signature in functionCall parts.","status":"INVALID_ARGUMENT"}}"#
        ));
        assert!(!rejected_thought_signature(
            gemini,
            r#"{"error":{"code":400,"message":"Request contains an invalid argument.","status":"INVALID_ARGUMENT"}}"#
        ));
    }

    #[test]
    fn stripping_replaces_signatures_with_the_bypass_and_keeps_every_part() {
        let payload = conversation(&["minted-on-gemini==", "second-call=="]);
        assert_eq!(
            thought_signatures(&payload),
            vec!["minted-on-gemini==", "second-call=="]
        );
        let repaired = without_thought_signatures_where(&payload, |_| true).expect("two to strip");
        let parts = repaired["contents"][1]["parts"].as_array().expect("parts");
        assert_eq!(parts.len(), 2);
        for (index, part) in parts.iter().enumerate() {
            assert_eq!(part["thoughtSignature"], THOUGHT_SIGNATURE_BYPASS);
            assert_eq!(
                part["functionCall"],
                payload["contents"][1]["parts"][index]["functionCall"]
            );
        }
        assert_eq!(repaired["contents"][0], payload["contents"][0]);
        assert_eq!(repaired["contents"][2], payload["contents"][2]);
        assert_eq!(repaired["generationConfig"], payload["generationConfig"]);
        // Nothing left to replace: the bypass is never counted or re-stripped.
        assert!(thought_signatures(&repaired).is_empty());
        assert!(without_thought_signatures_where(&repaired, |_| true).is_none());
    }

    #[test]
    fn selective_stripping_keeps_the_signatures_it_does_not_select() {
        let payload = conversation(&["foreign==", "native=="]);
        let repaired =
            without_thought_signatures_where(&payload, |signature| signature == "foreign==")
                .expect("one to strip");
        assert_eq!(thought_signatures(&repaired), vec!["native=="]);
        assert!(without_thought_signatures_where(&payload, |_| false).is_none());
    }
}
