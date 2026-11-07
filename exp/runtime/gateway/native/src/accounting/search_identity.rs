//! Recompute recovery identity from the inspected prefix and the currently loaded tools.

use super::{
    canonical,
    search_subject::{items, present, Subject},
};
use crate::errors::PublicError;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn digest(value: &Value) -> Result<String, PublicError> {
    Ok(format!(
        "{:x}",
        Sha256::digest(canonical::bytes(value, false)?)
    ))
}

impl Subject {
    pub fn recovery_prefix(&self) -> Result<Option<String>, PublicError> {
        let messages = items(&self.request["messages"]);
        let mut count = 0;
        for m in messages {
            if matches!(m["role"].as_str(), Some("system" | "developer")) {
                count += 1;
            } else if count > 0 {
                break;
            } else if m["role"] == "user" {
                count = 1;
                break;
            } else {
                return Ok(None);
            }
        }
        if count == 0 {
            return Ok(None);
        }
        let mut request = self.visible.clone();
        request["messages"] = json!(messages[..count]
            .iter()
            .map(|m| self.project_message(m))
            .collect::<Vec<_>>());
        for key in [
            "prompt_cache_key",
            "provider_prompt_cache_key",
            "provider_session_id",
            "client_request_id",
            "idempotency_key",
            "previous_response_id",
            "user",
            "safety_identifier",
        ] {
            if request.get(key).is_some() {
                request[key] = Value::Null;
            }
        }
        if request.get("metadata").is_some() {
            request["metadata"] = json!({});
        }
        let request_hash = digest(&request)?;
        let mut envelope = self.recovery_identity["envelope"].clone();
        let mut replay = vec![];
        for (i, m) in messages[..count].iter().enumerate() {
            let entry = super::search_counter::message_replay(m, i)?;
            if entry.as_object().is_some_and(|v| v.len() > 1) {
                replay.push(entry);
            }
        }
        let mut tools = vec![];
        for (i, tool) in items(&self.request["tools"]).iter().enumerate() {
            let fields = [
                "eager_input_streaming",
                "defer_loading",
                "allowed_callers",
                "input_examples",
            ];
            if fields.iter().all(|k| tool[k].is_null()) {
                continue;
            }
            let mut retained = json!({"tool_index":i,"name":tool["name"]});
            for key in fields {
                if !tool[key].is_null() {
                    retained[key] = tool[key].clone();
                }
            }
            tools.push(retained);
        }
        if envelope.is_null() && (!replay.is_empty() || !tools.is_empty()) {
            envelope = json!({"provider_replay":[],"provider_thinking_config":null,"reasoning_context":null,
                "context_management":null,"provider_output_config":null});
        }
        if !envelope.is_null() {
            envelope["provider_replay"] = json!(replay);
            if !tools.is_empty() {
                envelope["tools"] = json!(tools);
            }
        }
        let policy = &self.recovery_identity["gateway"];
        if envelope.is_null() && !present(policy) {
            return Ok(Some(request_hash));
        }
        let mut identity = json!({"request_sha256":request_hash});
        if let Some(envelope) = envelope.as_object() {
            identity
                .as_object_mut()
                .ok_or_else(PublicError::internal)?
                .extend(envelope.clone());
        }
        if present(policy) {
            identity["gateway"] = policy.clone();
        }
        Ok(Some(digest(&identity)?))
    }
}
