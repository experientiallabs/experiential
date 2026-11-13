//! Native sealing of provider-scoped reasoning continuation carriers.

use std::collections::HashSet;

use base64::{
    engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
    Engine,
};
use ring::{
    aead,
    rand::{SecureRandom, SystemRandom},
};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::errors::PublicError;

use super::canonical;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Authority {
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub claims: Value,
    key: String,
    prefix: String,
    pub history_sha256: String,
}

fn hash(value: &Value, normalized: bool) -> Result<String, PublicError> {
    Ok(format!(
        "{:x}",
        Sha256::digest(canonical::bytes(value, normalized)?)
    ))
}

impl Authority {
    pub fn seal(&self, request: &str, depth: usize, data: &Value) -> Result<Value, PublicError> {
        let error = || {
            PublicError::new(
                502,
                "malformed_response",
                "the provider returned malformed reasoning continuation data",
                "api_error",
            )
        };
        self.seal_inner(request, depth, data).map_err(|_| error())
    }

    fn seal_inner(&self, request: &str, depth: usize, data: &Value) -> Result<Value, PublicError> {
        if data["route_depth"].as_u64() != Some(depth as u64)
            || data["route_sha256"].as_str().is_none()
            || data["route_sha256"] != self.claims["reasoning_route_sha256"]
            || self.claims["schema_version"] != 3
        {
            return Err(PublicError::internal());
        }
        let content = data["content"]
            .as_str()
            .filter(|s| !s.is_empty() && s.len() <= 8 * 1024 * 1024)
            .ok_or_else(PublicError::internal)?;
        let assistant = match &data["assistant_content"] {
            Value::Null => Value::Null,
            Value::String(s) if s.trim().is_empty() => Value::Null,
            Value::String(s) => json!(s),
            _ => return Err(PublicError::internal()),
        };
        let calls = data["tool_calls"]
            .as_array()
            .filter(|v| !v.is_empty())
            .ok_or_else(PublicError::internal)?;
        let mut ids = Vec::new();
        let mut seen = HashSet::new();
        let mut digests = Vec::new();
        for call in calls {
            let id = call["call_id"]
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or_else(PublicError::internal)?;
            let name = call["name"]
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or_else(PublicError::internal)?;
            if !seen.insert(id) {
                return Err(PublicError::internal());
            }
            let arguments = canonical::arguments_bytes(
                call["raw_arguments"]
                    .as_str()
                    .ok_or_else(PublicError::internal)?,
            )?;
            ids.push(id);
            digests.push(json!({"call_id": id, "name": name,
                "arguments_sha256": format!("{:x}", Sha256::digest(arguments))}));
        }
        let mut claims = self.claims.clone();
        claims["issuing_request_id"] = json!(request);
        claims["issuing_route_depth"] = json!(depth);
        claims["issuing_history_sha256"] = json!(self.history_sha256);
        claims["issuing_turn_sha256"] = json!(hash(
            &json!({"assistant_content": assistant, "tool_calls": digests}),
            false
        )?);
        claims["tool_call_ids"] = json!(ids);
        claims["content"] = json!(content);
        let mut plaintext = canonical::bytes(&claims, false)?;
        if plaintext.len() > 8 * 1024 * 1024 + 32 * 1024 {
            return Err(PublicError::internal());
        }
        let deployment = self.claims["deployment_id"]
            .as_str()
            .filter(|s| !s.is_empty() && s.len() <= 256)
            .ok_or_else(PublicError::internal)?;
        if !self.prefix.is_ascii() || self.prefix.len() > 256 {
            return Err(PublicError::internal());
        }
        let public = format!("{}{}", self.prefix, URL_SAFE_NO_PAD.encode(deployment));
        let key = STANDARD
            .decode(&self.key)
            .map_err(|_| PublicError::internal())?;
        let key = aead::LessSafeKey::new(
            aead::UnboundKey::new(&aead::AES_256_GCM, &key).map_err(|_| PublicError::internal())?,
        );
        let mut nonce = [0; 12];
        SystemRandom::new()
            .fill(&mut nonce)
            .map_err(|_| PublicError::internal())?;
        key.seal_in_place_append_tag(
            aead::Nonce::assume_unique_for_key(nonce),
            aead::Aad::from(public.as_bytes()),
            &mut plaintext,
        )
        .map_err(|_| PublicError::internal())?;
        let mut envelope = nonce.to_vec();
        envelope.extend(plaintext);
        Ok(json!({"carrier": format!("{public}:{}", URL_SAFE_NO_PAD.encode(envelope))}))
    }
}
