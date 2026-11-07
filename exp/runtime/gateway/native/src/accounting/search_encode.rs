//! Rebuild provider conversation fields from the current inspected canonical request.

use super::{search_history as history, search_subject::items};
use crate::errors::PublicError;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Options {
    pub model_id: String,
    pub reasoning_route: Option<String>,
    pub fireworks: bool,
    pub exposed: bool,
    pub deepseek: bool,
    pub leading_only: bool,
    pub cache_control: bool,
    pub cache_options: bool,
    pub strip_reasoning: bool,
    pub json_instruction: String,
    #[serde(default)]
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub shaping: Value,
}

pub(super) fn rebuild(
    payload: &mut Value,
    dialect: &str,
    request: &Value,
    options: &Options,
) -> Result<(), PublicError> {
    let mut messages = items(&request["messages"]).to_vec();
    super::search_shaping::prepare(&mut messages, dialect, request, &options.shaping)?;
    if options.strip_reasoning {
        let start = messages
            .iter()
            .rposition(|m| m["role"] == "user")
            .map_or(0, |i| i + 1);
        for m in &mut messages[start..] {
            m["provider_reasoning"] = json!(items(&m["provider_reasoning"])
                .iter()
                .filter(|b| !matches!(
                    b["kind"].as_str(),
                    Some("sealed_reasoning_content" | "reasoning_content")
                ))
                .collect::<Vec<_>>());
        }
    }
    match dialect {
        "openai_compatible" => super::search_openai::chat(payload, request, messages, options),
        "openai_responses" => super::search_responses::rebuild(
            payload,
            request,
            history::attachments(messages, false),
            options,
        ),
        "anthropic_messages" => {
            super::search_anthropic::rebuild(payload, request, &messages, options)
        }
        "gemini_generate_content" => super::search_gemini::rebuild(
            payload,
            request,
            history::instructions(history::attachments(messages, true), false, false),
            options,
        ),
        "bedrock_converse_stream" => super::search_bedrock::rebuild(
            payload,
            request,
            history::instructions(history::attachments(messages, false), false, false),
            options,
        ),
        _ => Err(PublicError::internal()),
    }?;
    super::search_shaping::finish(payload, dialect, request, &options.shaping)
}

pub(super) fn arguments(call: &Value) -> Result<Value, PublicError> {
    Ok(match call["raw_arguments"].as_str() {
        Some(raw) => json!(raw),
        None => json!(super::json_text::spaced(&call["arguments"])?),
    })
}

pub(super) fn remove(value: &mut Value, key: &str) {
    if let Some(object) = value.as_object_mut() {
        object.remove(key);
    }
}

pub(super) fn push(messages: &mut Vec<Value>, role: &str, mut blocks: Vec<Value>, key: &str) {
    if let Some(last) = messages.last_mut().filter(|m| m["role"] == role) {
        last[key]
            .as_array_mut()
            .expect("encoder owns array")
            .append(&mut blocks);
    } else {
        messages.push(json!({"role": role, key: blocks}));
    }
}

fn markers(m: &Value) -> Vec<bool> {
    let blocks = history::marked(items(&m["provider_text_blocks"]), false);
    let mut blocks = blocks.iter();
    items(&m["content_parts"])
        .iter()
        .map(|p| match p["kind"].as_str() {
            Some("text") => {
                blocks
                    .next()
                    .is_some_and(|b| b.get("cache_control").is_some())
                    || !p["cache_control"].is_null()
            }
            Some("image" | "document") => !p["cache_control"].is_null(),
            _ => false,
        })
        .collect()
}

pub(super) fn has_breakpoints(m: &Value) -> bool {
    !m["cache_control"].is_null()
        || items(&m["provider_text_blocks"])
            .iter()
            .any(|b| b.get("cache_control").is_some())
        || markers(m).iter().any(|b| *b)
}

pub(super) fn breakpoints(m: &Value, wire: &mut Value, responses: bool) {
    let key = if responses && m["role"] == "tool" {
        "output"
    } else {
        "content"
    };
    let text_type = if responses { "input_text" } else { "text" };
    if m["role"] == "tool" {
        if m["cache_control"].is_null() {
            return;
        }
        if wire[key].is_string() {
            wire[key] = json!([{"type": text_type, "text": wire[key], "prompt_cache_breakpoint": {"mode": "explicit"}}]);
        } else if let Some(last) = wire[key]
            .as_array_mut()
            .and_then(|a| a.last_mut())
            .filter(|v| v.is_object())
        {
            last["prompt_cache_breakpoint"] = json!({"mode": "explicit"});
        }
    } else if !items(&m["content_parts"]).is_empty() && wire[key].is_array() {
        let markers = markers(m);
        if let Some(parts) = wire[key]
            .as_array_mut()
            .filter(|a| a.len() == markers.len())
        {
            for (part, marked) in parts.iter_mut().zip(markers) {
                if marked && part.is_object() {
                    part["prompt_cache_breakpoint"] = json!({"mode": "explicit"});
                }
            }
        }
    } else if wire[key].is_string() {
        let blocks = history::marked(items(&m["provider_text_blocks"]), false);
        if blocks.iter().any(|b| b.get("cache_control").is_some()) {
            wire[key] = json!(blocks
                .iter()
                .map(|b| {
                    let mut part = json!({"type": text_type, "text": b["text"]});
                    if b.get("cache_control").is_some() {
                        part["prompt_cache_breakpoint"] = json!({"mode": "explicit"});
                    }
                    part
                })
                .collect::<Vec<_>>());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversation_rebuild_matches_python_provider_encoders() {
        for line in include_str!("../../tests/generated/search_encoders.jsonl").lines() {
            let group: Value = crate::lossless_json::from_str(line).unwrap();
            let subject: super::super::search_subject::Subject =
                crate::lossless_json::from_value(group["subject"].clone()).unwrap();
            assert_eq!(
                json!(subject.recovery_prefix().unwrap()),
                group["prefix"],
                "{} recovery identity",
                group["name"]
            );
            for case in group["encodings"].as_array().unwrap() {
                let options: Options =
                    crate::lossless_json::from_value(case["options"].clone()).unwrap();
                let mut payload = json!({});
                let result = rebuild(
                    &mut payload,
                    case["dialect"].as_str().unwrap(),
                    &subject.request,
                    &options,
                );
                let actual = if result.is_ok() { payload } else { Value::Null };
                assert_eq!(
                    actual, case["expected"],
                    "{} {} {} (reference error {:?})",
                    group["name"], case["name"], case["dialect"], case["error"]
                );
            }
        }
    }
}
