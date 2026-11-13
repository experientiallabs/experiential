//! Complete and classifier-visible subjects for evolving native conversations.

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::errors::PublicError;

use super::{canonical, search::Tool, search_round::Frame};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Subject {
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub request: Value,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub visible: Value,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub message_default: Value,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub call_default: Value,
    pub message_keys: Vec<String>,
    pub call_keys: Vec<String>,
    tool_keys: Vec<String>,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    definitions: HashMap<String, Value>,
    #[serde(default)]
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    pub recovery_identity: Value,
}

fn project(value: &Value, keys: &[String]) -> Value {
    Value::Object(
        keys.iter()
            .map(|key| (key.clone(), value[key].clone()))
            .collect(),
    )
}

pub(super) fn items(value: &Value) -> &[Value] {
    value.as_array().map(Vec::as_slice).unwrap_or_default()
}

pub(super) fn present(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Array(v) => !v.is_empty(),
        Value::Object(v) => !v.is_empty(),
        Value::String(v) => !v.is_empty(),
        Value::Bool(v) => *v,
        Value::Number(n) => n.as_f64() != Some(0.0),
    }
}

pub(super) fn folded(message: &Value) -> String {
    let text = message["content"].as_str().unwrap_or("");
    if message["tool_is_error"] == true {
        format!("[tool error] {text}")
    } else {
        text.to_owned()
    }
}

impl Subject {
    pub fn append(&mut self, tools: &[Tool], frame: &Frame) -> Result<(), PublicError> {
        let definitions = tools
            .iter()
            .map(|tool| {
                self.definitions
                    .get(&tool.key)
                    .cloned()
                    .ok_or_else(PublicError::internal)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.visible["tools"] = json!(definitions
            .iter()
            .map(|tool| project(tool, &self.tool_keys))
            .collect::<Vec<_>>());
        self.request["tools"] = json!(definitions);
        self.request["tool_choice"] = Value::Null;
        self.visible["tool_choice"] = Value::Null;
        let mut assistant = self.message_default.clone();
        assistant["role"] = json!("assistant");
        assistant["tool_calls"] = json!(frame
            .calls
            .iter()
            .map(|call| {
                let mut value = self.call_default.clone();
                value["call_id"] = json!(call.call_id);
                value["name"] = json!(call.name);
                value["arguments"] = call.arguments.clone();
                value["raw_arguments"] = json!(call.raw_arguments);
                value
            })
            .collect::<Vec<_>>());
        self.push(assistant)?;
        for (call, text) in frame.calls.iter().zip(&frame.results) {
            let mut result = self.message_default.clone();
            result["role"] = json!("tool");
            result["tool_call_id"] = json!(call.call_id);
            result["content"] = json!(text);
            self.push(result)?;
        }
        Ok(())
    }

    fn push(&mut self, message: Value) -> Result<(), PublicError> {
        let visible = self.project_message(&message);
        self.request["messages"]
            .as_array_mut()
            .ok_or_else(PublicError::internal)?
            .push(message);
        self.visible["messages"]
            .as_array_mut()
            .ok_or_else(PublicError::internal)?
            .push(visible);
        Ok(())
    }

    pub fn project_message(&self, message: &Value) -> Value {
        let mut visible = project(message, &self.message_keys);
        visible["tool_calls"] = json!(items(&message["tool_calls"])
            .iter()
            .map(|call| project(call, &self.call_keys))
            .collect::<Vec<_>>());
        if let Some(parts) = visible["content_parts"].as_array_mut() {
            for part in parts {
                if let Some(object) = part.as_object_mut() {
                    object.remove("cache_control");
                }
            }
        }
        visible
    }

    pub fn set_messages(&mut self, messages: Vec<Value>) {
        self.visible["messages"] = json!(messages
            .iter()
            .map(|m| self.project_message(m))
            .collect::<Vec<_>>());
        self.request["messages"] = json!(messages);
    }

    pub fn size(&self) -> Result<usize, PublicError> {
        Ok(canonical::bytes(&self.request, false)?.len())
    }

    /// The immutable context inspected before message text and argument strings.
    pub fn context(&self) -> Result<String, ()> {
        let r = &self.request;
        if [
            "context_management",
            "provider_output_config",
            "provider_thinking_config",
        ]
        .iter()
        .any(|k| !r[k].is_null())
            || [
                "safeguards",
                "provider_native_tools",
                "provider_server_tools",
            ]
            .iter()
            .any(|k| present(&r[k]))
            || items(&r["messages"]).iter().any(|m| {
                [
                    "content_parts",
                    "provider_reasoning",
                    "capture_only_reasoning",
                ]
                .iter()
                .any(|k| present(&m[k]))
                    || [
                        "provider_native_item",
                        "provider_anthropic_block",
                        "provider_anthropic_blocks",
                        "provider_tool_caller",
                    ]
                    .iter()
                    .any(|k| !m[k].is_null())
                    || items(&m["tool_calls"])
                        .iter()
                        .any(|c| !c["provider_caller"].is_null())
            })
        {
            return Err(());
        }
        let mut context = json!({});
        if present(&r["tools"]) {
            context["tools"] = json!(items(&r["tools"])
                .iter()
                .map(|tool| {
                    let mut v = project(tool, &self.tool_keys);
                    for key in [
                        "input_examples",
                        "allowed_callers",
                        "defer_loading",
                        "eager_input_streaming",
                    ] {
                        v[key] = tool[key].clone();
                    }
                    v
                })
                .collect::<Vec<_>>());
        }
        for key in [
            "structured_text",
            "tool_choice",
            "web_search",
            "tool_search",
        ] {
            if !r[key].is_null() {
                context[key] = self.visible[key].clone();
            }
        }
        let calls: Vec<_> = items(&r["messages"]).iter().flat_map(|m| items(&m["tool_calls"]))
            .map(|c| json!({"call_id": c["call_id"], "name": c["name"], "namespace": c["provider_namespace"]})).collect();
        if !calls.is_empty() {
            context["tool_calls"] = json!(calls);
        }
        let results: Vec<_> = items(&r["messages"]).iter().filter(|m| ["tool_call_id", "provider_tool_name", "provider_tool_namespace"].iter().any(|k| !m[k].is_null()))
            .map(|m| {
                let mut result = json!({"call_id": m["tool_call_id"], "name": m["provider_tool_name"], "namespace": m["provider_tool_namespace"]});
                if present(&m["tool_references"]) { result["tool_references"] = m["tool_references"].clone(); }
                result
            }).collect();
        if !results.is_empty() {
            context["tool_results"] = json!(results);
        }
        if context.as_object().unwrap().is_empty() {
            return Ok(String::new());
        }
        String::from_utf8(canonical::bytes(&context, false).map_err(|_| ())?).map_err(|_| ())
    }

    pub fn subjects(&self) -> Result<Vec<String>, ()> {
        let context = self.context()?;
        let mut result = Vec::new();
        if !context.is_empty() {
            result.push(context);
        }
        for message in items(&self.request["messages"]) {
            let text = folded(message);
            if !text.is_empty() {
                result.push(text);
            }
            for call in items(&message["tool_calls"]) {
                result.push(match call["raw_arguments"].as_str() {
                    Some(text) => text.to_owned(),
                    None => super::json_text::spaced(&call["arguments"]).map_err(|_| ())?,
                });
            }
        }
        Ok(result)
    }
}
