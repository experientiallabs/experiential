//! Borrowed database payload projection; only the final writer encodes content.
use super::record::{Protocol, Record, Response};
use serde::Serialize;
use serde_json::Value;
use std::borrow::Cow;
use std::sync::Arc;

#[derive(Serialize)]
struct Scope<'a> {
    user_id: &'a str,
    application_id: &'a str,
}

#[derive(Serialize)]
struct Output<'a> {
    response: &'a Option<Response>,
    provider_reasoning: Option<Cow<'a, str>>,
    provider_reasoning_source_json: Option<Cow<'a, str>>,
    provider_tool_calls_json: &'a Option<String>,
    metrics: &'a Option<super::metrics::Metrics>,
    canonical_model_id: &'a Option<String>,
    gemini_thought_parts_truncated: Option<bool>,
    gemini_thought_parts: Cow<'a, [Arc<Value>]>,
    gemini_thought_parts_source_json: Option<Cow<'a, str>>,
}

#[derive(Serialize)]
struct Request<'a> {
    exp_context: &'a Value,
    exp_capture_output: Output<'a>,
    previous_response_id: &'a Value,
}

#[derive(Serialize)]
struct Provenance<'a> {
    source_id: &'a str,
    model_id: &'a Option<String>,
    deployment_id: &'a Option<String>,
}

#[derive(Serialize)]
struct Experience<'a> {
    schema_version: u32,
    experience_id: &'a str,
    response_id: &'a str,
    episode_id: Option<&'a str>,
    parent_response_id: &'a Value,
    scope: Scope<'a>,
    protocol: Protocol,
    captured_at: f64,
    request: Request<'a>,
    response: &'a Value,
    provenance: Provenance<'a>,
}

pub(super) fn encode(
    record: &Record,
    response: &Value,
    experience_id: &str,
    maximum: usize,
) -> Option<String> {
    let context = &record.request.context;
    let scope = &record.request.scope;
    let reasoning = record.durable_reasoning().ok()?;
    let (gemini_thought_parts, gemini_thought_parts_source_json) = record.durable_gemini_parts();
    let parent = &context["request"]["previous_response_id"];
    let experience = Experience {
        schema_version: 1,
        experience_id,
        response_id: response["id"].as_str()?,
        episode_id: context["request"]["metadata"]["conversation_id"]
            .as_str()
            .filter(|value| !value.trim().is_empty() && value.len() <= 512)
            .or_else(|| context["session_id"].as_str()),
        parent_response_id: parent,
        scope: Scope {
            user_id: &scope.identity_id,
            application_id: &scope.application_id,
        },
        protocol: record.request.protocol,
        captured_at: record.captured_at,
        request: Request {
            exp_context: context,
            exp_capture_output: Output {
                response: &record.response,
                provider_reasoning: reasoning.text,
                provider_reasoning_source_json: reasoning.source_json,
                provider_tool_calls_json: &record.provider_tool_calls_json,
                metrics: &record.metrics,
                canonical_model_id: &record.canonical_model_id,
                gemini_thought_parts_truncated: record.gemini_thought_parts_truncated,
                gemini_thought_parts,
                gemini_thought_parts_source_json,
            },
            previous_response_id: parent,
        },
        response,
        provenance: Provenance {
            source_id: &record.request.request_id,
            model_id: &record.request.model_id,
            deployment_id: &record.deployment_id,
        },
    };
    super::budget::encode(&experience, maximum)
}

#[cfg(test)]
#[path = "local_payload_test.rs"]
mod tests;
