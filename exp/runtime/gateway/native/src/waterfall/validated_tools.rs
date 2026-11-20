//! Schema-validated redials after a malformed Gemini function call.
//!
//! Gemini ends a turn with `finishReason: MALFORMED_FUNCTION_CALL` when the
//! function call it sampled does not parse. For short prompts a fresh sample
//! is usually valid, so the ladder retries it. Some long agent conversations
//! fail on every sample instead: production 2026-11-20 on Gemini 4 Neon had
//! one conversation per organization fail 4 of 4 dials across the Gemini API
//! and Vertex, request after request, each dial thinking about as long as the
//! last before the broken call. Retrying the same body cannot fix that.
//!
//! So every dial after such a failure asks Google for
//! `toolConfig.functionCallingConfig.mode = "VALIDATED"`, both backends'
//! documented mode that constrains a function call to its declared schema
//! while still letting the model answer in text. That is the caller's `AUTO`
//! behavior with the parse guaranteed. It applies only where it keeps the
//! caller's intent:
//!
//! - the body declares at least one function, and
//! - the caller's mode is absent, `AUTO` or `MODE_UNSPECIFIED`. `ANY` and
//!   `NONE` already constrain the turn more than `VALIDATED` does.
//!
//! A cached body is out of reach. Google refuses `toolConfig` beside
//! `cachedContent`, because the cache holds it. So a body that already names
//! a cache is left alone, and a validated dial of a cache-eligible rung goes
//! plain (no overlay), the same way a repaired dial does.

use serde_json::{json, Value};

use super::DeploymentWire;
use crate::thought_signature_repair::GEMINI_DIALECT;

/// The function-calling mode a validated redial sends.
pub(super) const VALIDATED_MODE: &str = "VALIDATED";

/// This rung's wire with schema-validated function calling requested, or
/// `None` when the redial must dial the rung as it is.
pub(super) fn validated(wire: &DeploymentWire) -> Option<DeploymentWire> {
    if wire.dialect != GEMINI_DIALECT || wire.upstream_body.is_some() {
        return None;
    }
    let payload = wire.upstream_payload.as_object()?;
    if payload.contains_key("cachedContent") || !declares_functions(payload.get("tools")) {
        return None;
    }
    match payload.get("toolConfig") {
        None | Some(Value::Null) => {}
        Some(Value::Object(config)) => match config.get("functionCallingConfig") {
            None | Some(Value::Null) => {}
            Some(Value::Object(calling)) => match calling.get("mode") {
                None | Some(Value::Null) => {}
                Some(Value::String(mode)) if mode == "AUTO" || mode == "MODE_UNSPECIFIED" => {}
                Some(_) => return None,
            },
            Some(_) => return None,
        },
        Some(_) => return None,
    }
    let mut result = wire.clone();
    result.explicit_cache = false;
    result.automatic_cache = false;
    let payload = result.upstream_payload.as_object_mut()?;
    let config = payload
        .entry("toolConfig")
        .and_modify(|config| {
            if config.is_null() {
                *config = json!({});
            }
        })
        .or_insert_with(|| json!({}))
        .as_object_mut()?;
    let calling = config
        .entry("functionCallingConfig")
        .and_modify(|calling| {
            if calling.is_null() {
                *calling = json!({});
            }
        })
        .or_insert_with(|| json!({}))
        .as_object_mut()?;
    calling.insert("mode".to_string(), Value::from(VALIDATED_MODE));
    Some(result)
}

/// Whether a generateContent `tools` list declares at least one function.
fn declares_functions(tools: Option<&Value>) -> bool {
    tools.and_then(Value::as_array).is_some_and(|tools| {
        tools.iter().any(|tool| {
            tool.get("functionDeclarations")
                .and_then(Value::as_array)
                .is_some_and(|declarations| !declarations.is_empty())
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::waterfall::ladder_tests::wire;

    fn gemini(payload: Value) -> DeploymentWire {
        let mut rung = wire("gemini-neon", "http://127.0.0.1:9/v1beta/models/neon", 0);
        rung.dialect = GEMINI_DIALECT.to_string();
        rung.explicit_cache = true;
        rung.automatic_cache = true;
        rung.upstream_payload = payload;
        rung
    }

    fn declared() -> Value {
        json!([{"functionDeclarations": [{"name": "read", "parametersJsonSchema": {"type": "object"}}]}])
    }

    #[test]
    fn an_auto_or_absent_mode_becomes_validated_and_dials_plain() {
        for tool_config in [
            None,
            Some(Value::Null),
            Some(json!({})),
            Some(json!({"functionCallingConfig": {"mode": "AUTO"}})),
            Some(json!({"functionCallingConfig": {"mode": "MODE_UNSPECIFIED"}})),
        ] {
            let mut payload = json!({"contents": [], "tools": declared()});
            if let Some(config) = tool_config {
                payload["toolConfig"] = config;
            }
            let validated = validated(&gemini(payload.clone())).expect("validated");
            assert_eq!(
                validated.upstream_payload["toolConfig"]["functionCallingConfig"]["mode"],
                VALIDATED_MODE
            );
            assert!(!validated.explicit_cache && !validated.automatic_cache);
            // Only the mode changes.
            let mut sent = validated.upstream_payload.clone();
            sent.as_object_mut().unwrap().remove("toolConfig");
            payload.as_object_mut().unwrap().remove("toolConfig");
            assert_eq!(sent, payload);
        }
    }

    #[test]
    fn allowed_function_names_survive_the_validated_mode() {
        let payload = json!({
            "tools": declared(),
            "toolConfig": {"functionCallingConfig": {"mode": "AUTO", "allowedFunctionNames": ["read"]}},
        });
        let validated = validated(&gemini(payload)).expect("validated");
        assert_eq!(
            validated.upstream_payload["toolConfig"],
            json!({"functionCallingConfig": {"mode": "VALIDATED", "allowedFunctionNames": ["read"]}})
        );
    }

    #[test]
    fn stricter_modes_caches_toolless_bodies_and_other_dialects_dial_unchanged() {
        for mode in ["ANY", "NONE", "VALIDATED"] {
            let payload = json!({"tools": declared(), "toolConfig": {"functionCallingConfig": {"mode": mode}}});
            assert!(validated(&gemini(payload)).is_none(), "{mode}");
        }
        assert!(validated(&gemini(
            json!({"tools": declared(), "cachedContent": "cachedContents/x"})
        ))
        .is_none());
        assert!(validated(&gemini(json!({"contents": []}))).is_none());
        assert!(validated(&gemini(json!({"tools": [{"googleSearch": {}}]}))).is_none());
        assert!(validated(&gemini(json!({"tools": [{"functionDeclarations": []}]}))).is_none());
        let mut openai = gemini(json!({"tools": declared()}));
        openai.dialect = "openai_chat".to_string();
        assert!(validated(&openai).is_none());
    }
}
