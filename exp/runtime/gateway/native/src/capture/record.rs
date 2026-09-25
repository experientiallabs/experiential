//! Versioned content records shared by local and hosted capture destinations.

use std::borrow::Cow;
use std::sync::Arc;

use serde::ser::{Error, SerializeStruct};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::Value;

use super::budget::{self, json_bytes, optional_string_bytes, string_bytes};

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
    pub context: Arc<Value>,
}

impl Request {
    pub(crate) fn json_bytes(&self) -> usize {
        let protocol = match self.protocol {
            Protocol::ChatCompletions => "chat_completions",
            Protocol::Responses => "responses",
            Protocol::Messages => "messages",
        };
        r#"{"request_id":,"scope":{"organization_id":,"identity_id":,"application_id":},"protocol":,"model_id":,"context":}"#.len()
            + string_bytes(&self.request_id)
            + string_bytes(&self.scope.organization_id)
            + string_bytes(&self.scope.identity_id)
            + string_bytes(&self.scope.application_id)
            + string_bytes(protocol)
            + optional_string_bytes(self.model_id.as_deref())
            + json_bytes(&self.context)
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        budget::heap_bytes(&self.context)
            + std::mem::size_of::<Value>()
            + 32
            + self.request_id.capacity()
            + self.scope.organization_id.capacity()
            + self.scope.identity_id.capacity()
            + self.scope.application_id.capacity()
            + self.model_id.as_ref().map_or(0, String::capacity)
    }
}

/// Exact public content, distinguished from a successfully reconstructed completion.
/// A disconnected or truncated stream remains evidence, never a complete rollout.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Response {
    Json {
        status: u16,
        body: Value,
        /// Exact JSON when text or numeric values need a queryable projection.
        source_json: Option<String>,
    },
    Sse {
        status: u16,
        frames: Vec<Value>,
        truncated: bool,
        client_disconnected: bool,
        source_json: Option<String>,
    },
}

impl Response {
    pub(crate) fn json_bytes(&self) -> usize {
        match self {
            Self::Json { status, body, source_json } => {
                r#"{"kind":"json","status":,"body":,"source_json":}"#.len()
                    + status.to_string().len() + json_bytes(body)
                    + optional_string_bytes(source_json.as_deref())
            }
            Self::Sse { status, frames, truncated, client_disconnected, source_json } => {
                r#"{"kind":"sse","status":,"frames":,"truncated":,"client_disconnected":,"source_json":}"#.len()
                    + status.to_string().len()
                    + 2 + frames.len().saturating_sub(1)
                    + frames.iter().map(json_bytes).sum::<usize>()
                    + if *truncated { 4 } else { 5 }
                    + if *client_disconnected { 4 } else { 5 }
                    + optional_string_bytes(source_json.as_deref())
            }
        }
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        match self {
            Self::Json {
                body, source_json, ..
            } => budget::heap_bytes(body) + source_json.as_ref().map_or(0, String::capacity),
            Self::Sse {
                frames,
                source_json,
                ..
            } => {
                frames.capacity() * std::mem::size_of::<Value>()
                    + frames.iter().map(budget::heap_bytes).sum::<usize>()
                    + source_json.as_ref().map_or(0, String::capacity)
            }
        }
    }
}

/// One idempotent request update. A later update may supply its response.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Record<R = Response> {
    /// Internal delivery dependency, never part of a complete capture record.
    #[serde(skip)]
    pub checkpointed: bool,
    pub schema_version: u32,
    pub request: Request,
    pub response: Option<R>,
    /// Caller-facing headers, timing and optional redacted wire request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<Value>,
    /// Provider-returned plaintext from an explicitly exposure-enabled winning rung.
    pub provider_reasoning: Option<String>,
    pub provider_reasoning_source_json: Option<String>,
    /// Exact completed tool calls, escaped once so JSONB cannot alter their text.
    pub provider_tool_calls_json: Option<String>,
    pub deployment_id: Option<String>,
    pub metrics: Option<super::metrics::Metrics>,
    /// Provider parts in order; `thought: true` text is a summary, never full CoT.
    pub gemini_thought_parts: Vec<Arc<Value>>,
    pub gemini_thought_parts_source_json: Option<String>,
    pub captured_at: f64,
}

/// Borrow ordinary reasoning; materialize a lossless projection only for NUL text.
pub(crate) struct Reasoning<'a> {
    pub text: Option<Cow<'a, str>>,
    pub source_json: Option<Cow<'a, str>>,
}

impl<R> Record<R> {
    pub(crate) fn durable_gemini_parts(&self) -> (Cow<'_, [Arc<Value>]>, Option<Cow<'_, str>>) {
        if !self
            .gemini_thought_parts
            .iter()
            .any(|part| super::response::contains_nul(part))
        {
            return (
                Cow::Borrowed(&self.gemini_thought_parts),
                self.gemini_thought_parts_source_json
                    .as_deref()
                    .map(Cow::Borrowed),
            );
        }
        let mut value = Value::Array(
            self.gemini_thought_parts
                .iter()
                .map(|part| (**part).clone())
                .collect(),
        );
        let source = super::response::lossless_projection(&mut value);
        let Value::Array(parts) = value else {
            unreachable!()
        };
        (
            Cow::Owned(parts.into_iter().map(Arc::new).collect()),
            source.map(Cow::Owned),
        )
    }

    pub(crate) fn durable_reasoning(&self) -> Result<Reasoning<'_>, serde_json::Error> {
        if let Some(text) = self
            .provider_reasoning
            .as_ref()
            .filter(|text| text.contains('\0'))
        {
            return Ok(Reasoning {
                text: Some(Cow::Owned(text.replace('\0', "\u{fffd}"))),
                source_json: Some(Cow::Owned(serde_json::to_string(text)?)),
            });
        }
        Ok(Reasoning {
            text: self.provider_reasoning.as_deref().map(Cow::Borrowed),
            source_json: self
                .provider_reasoning_source_json
                .as_deref()
                .map(Cow::Borrowed),
        })
    }
}

impl<R: Serialize> Serialize for Record<R> {
    /// Project only exceptional reasoning text; never clone the request or response.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.serialize_update(serializer, false)
    }
}

struct CompletionUpdate<'a, R>(&'a Record<R>);

impl<R: Serialize> Serialize for CompletionUpdate<'_, R> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize_update(serializer, true)
    }
}

impl<R: Serialize> Record<R> {
    fn serialize_update<S: Serializer>(
        &self,
        serializer: S,
        completion: bool,
    ) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Reference<'a> {
            request_id: &'a str,
            scope: &'a Scope,
            protocol: Protocol,
            model_id: &'a Option<String>,
        }
        let reasoning = self.durable_reasoning().map_err(S::Error::custom)?;
        let (parts, parts_source) = self.durable_gemini_parts();
        let mut record = serializer.serialize_struct("Record", 11)?;
        record.serialize_field(
            "schema_version",
            &if completion { 2 } else { self.schema_version },
        )?;
        if completion {
            record.serialize_field(
                "request",
                &Reference {
                    request_id: &self.request.request_id,
                    scope: &self.request.scope,
                    protocol: self.request.protocol,
                    model_id: &self.request.model_id,
                },
            )?;
        } else {
            record.serialize_field("request", &self.request)?;
        }
        record.serialize_field("response", &self.response)?;
        if let Some(transport) = &self.transport {
            record.serialize_field("transport", transport)?;
        }
        record.serialize_field("provider_reasoning", &reasoning.text)?;
        record.serialize_field("provider_reasoning_source_json", &reasoning.source_json)?;
        record.serialize_field("provider_tool_calls_json", &self.provider_tool_calls_json)?;
        record.serialize_field("deployment_id", &self.deployment_id)?;
        record.serialize_field("metrics", &self.metrics)?;
        record.serialize_field("gemini_thought_parts", &parts)?;
        record.serialize_field("gemini_thought_parts_source_json", &parts_source)?;
        record.serialize_field("captured_at", &self.captured_at)?;
        record.end()
    }
}

impl<R: Serialize> Record<R> {
    pub(crate) fn valid(&self) -> bool {
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
            return false;
        }
        true
    }

    /// Only destinations encode records, after collection and policy have completed.
    pub(crate) fn encode(&self, maximum_bytes: usize) -> Option<String> {
        self.valid()
            .then(|| budget::encode(self, maximum_bytes))
            .flatten()
    }

    /// Hosted completion updates reference an already queued prompt checkpoint.
    /// A destination must retry schema 2 until that prompt is durably present.
    pub(crate) fn encode_update(&self, maximum_bytes: usize) -> Option<String> {
        if self.checkpointed && self.response.is_some() {
            self.valid()
                .then(|| budget::encode(&CompletionUpdate(self), maximum_bytes))
                .flatten()
        } else {
            self.encode(maximum_bytes)
        }
    }
}

impl Record {
    /// Shared request trees are conservatively charged in each owning queue.
    pub(crate) fn heap_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.request.heap_bytes()
            + self.response.as_ref().map_or(0, Response::heap_bytes)
            + self.transport.as_ref().map_or(0, budget::heap_bytes)
            + self.provider_reasoning.as_ref().map_or(0, String::capacity)
            + self
                .provider_reasoning_source_json
                .as_ref()
                .map_or(0, String::capacity)
            + self
                .provider_tool_calls_json
                .as_ref()
                .map_or(0, String::capacity)
            + self.deployment_id.as_ref().map_or(0, String::capacity)
            + self.gemini_thought_parts.capacity() * std::mem::size_of::<Arc<Value>>()
            + self
                .gemini_thought_parts
                .iter()
                .map(|part| budget::heap_bytes(part) + 64)
                .sum::<usize>()
            + self
                .gemini_thought_parts_source_json
                .as_ref()
                .map_or(0, String::capacity)
    }
}

#[cfg(test)]
#[path = "record_test.rs"]
mod tests;
