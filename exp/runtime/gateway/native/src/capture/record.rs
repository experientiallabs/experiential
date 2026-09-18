//! Versioned content records shared by local and hosted capture destinations.

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub(crate) const SCHEMA_VERSION: u32 = 1;

/// Authority-derived tenancy; none of these values comes from a caller's metadata.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Scope {
    pub organization_id: String,
    pub identity_id: String,
    pub application_id: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Protocol {
    ChatCompletions,
    Responses,
    Messages,
}

/// Effective input supplied by the authenticated, post-guardrail admission seam.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Request {
    pub request_id: String,
    pub scope: Scope,
    pub protocol: Protocol,
    pub model_id: Option<String>,
    pub context: Value,
}

/// Exact public content, distinguished from a successfully reconstructed completion.
/// A disconnected or truncated stream remains evidence, never a complete rollout.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Response {
    Json {
        status: u16,
        body: Value,
    },
    Sse {
        status: u16,
        frames: Vec<Value>,
        truncated: bool,
        client_disconnected: bool,
    },
}

/// One idempotent request update. A later update may supply its response.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Record {
    pub schema_version: u32,
    pub request: Request,
    pub response: Option<Response>,
    pub deployment_id: Option<String>,
    pub captured_at: f64,
}

impl Record {
    /// Validate both the version and the content budget before destination admission.
    pub(crate) fn encode(&self, maximum_bytes: usize) -> Option<String> {
        if self.schema_version != SCHEMA_VERSION
            || !self.captured_at.is_finite()
            || self.captured_at < 0.0
            || [
                &self.request.request_id,
                &self.request.scope.organization_id,
                &self.request.scope.identity_id,
                &self.request.scope.application_id,
            ]
            .iter()
            .any(|value| value.trim().is_empty() || value.len() > 512)
            || self
                .request
                .model_id
                .as_ref()
                .is_some_and(|value| value.trim().is_empty() || value.len() > 512)
            || self
                .request
                .context
                .get("schema_version")
                .and_then(Value::as_u64)
                != Some(1)
            || !self
                .request
                .context
                .get("request")
                .is_some_and(Value::is_object)
        {
            return None;
        }
        let encoded = serde_json::to_string(self).ok()?;
        (encoded.len() <= maximum_bytes).then_some(encoded)
    }
}
