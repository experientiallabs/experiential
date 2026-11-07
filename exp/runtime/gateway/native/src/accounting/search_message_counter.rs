//! Recount canonical messages after a policy changes the conversation shape.

use super::*;
use crate::accounting::search_subject::items;

impl Counter {
    pub fn count_messages(
        &self,
        tools: &[Tool],
        messages: &[Value],
    ) -> Result<(u64, u64), PublicError> {
        let mut counter = self.clone();
        counter.messages = Parts::default();
        counter.message_count = messages.len();
        let mut retained = Vec::new();
        for (index, message) in messages.iter().enumerate() {
            counter.messages.fixed += self.message_framing as i64;
            for key in ["content", "tool_call_id"] {
                if let Some(text) = message[key].as_str() {
                    counter.messages.text(text);
                }
            }
            for call in items(&message["tool_calls"]) {
                for key in ["call_id", "name"] {
                    counter
                        .messages
                        .text(call[key].as_str().ok_or_else(PublicError::internal)?);
                }
                if call["raw_arguments"].is_null() {
                    counter.messages.json(&call["arguments"])?;
                }
            }
            for part in items(&message["content_parts"]) {
                counter.messages.part(part)?;
            }
            let replay = message_replay(message, index)?;
            if replay.as_object().is_some_and(|o| o.len() > 1) {
                retained.push(replay);
            }
        }
        counter.replay = if self.replay.is_null() {
            json!({"provider_replay": [],
            "provider_thinking_config": null, "reasoning_context": null,
            "context_management": null, "provider_output_config": null})
        } else {
            self.replay.clone()
        };
        counter.replay["provider_replay"] = json!(retained);
        let object = counter
            .replay
            .as_object_mut()
            .ok_or_else(PublicError::internal)?;
        object.remove("tools");
        if object.iter().all(|(key, value)| match key.as_str() {
            "provider_replay" => value.as_array().is_some_and(Vec::is_empty),
            "provider_thinking_config"
            | "reasoning_context"
            | "context_management"
            | "provider_output_config" => value.is_null(),
            _ => false,
        }) {
            counter.replay = Value::Null;
        }
        counter.count(tools, &[])
    }
}

impl Parts {
    fn json(&mut self, value: &Value) -> Result<(), PublicError> {
        let mut value = value.clone();
        self.scrub(&mut value);
        self.text(&json_text::dumps(&value, false)?);
        Ok(())
    }

    fn part(&mut self, part: &Value) -> Result<(), PublicError> {
        let decoded =
            || -> Result<i64, PublicError> {
                let data = part["data"].as_str().ok_or_else(PublicError::internal)?;
                Ok((data.len() * 3 / 4) as i64
                    - (data.len() - data.trim_end_matches('=').len()) as i64)
            };
        self.fixed += match part["kind"].as_str() {
            Some("text") => 0,
            Some("image") => 2048,
            Some("document") => {
                if let Some(name) = part["name"].as_str() {
                    self.text(name);
                }
                if part["data"].is_null() {
                    16384
                } else {
                    (decoded()?.div_euclid(32)).max(2048)
                }
            }
            Some("video") => {
                if part["data"].is_null() {
                    32768
                } else {
                    (decoded()?.div_euclid(128)).max(4096)
                }
            }
            Some("audio") => (decoded()?.div_euclid(256)).max(1024),
            _ => return Err(PublicError::internal()),
        };
        Ok(())
    }
}

pub(in crate::accounting) fn message_replay(
    message: &Value,
    index: usize,
) -> Result<Value, PublicError> {
    let mut replay = json!({"message_index": index});
    if !message["provider_item_id"].is_null() {
        for key in [
            "provider_item_id",
            "provider_output_index",
            "provider_status",
            "provider_phase",
        ] {
            replay[key] = message[key].clone();
        }
    }
    for key in ["provider_native_item", "provider_anthropic_block"] {
        if !message[key].is_null() {
            replay[key] = message[key].clone();
        }
    }
    if !items(&message["provider_reasoning"]).is_empty() {
        let blocks = items(&message["provider_reasoning"])
            .iter()
            .map(|block| {
                let mut block = block.clone();
                let encrypted = block["kind"] == "encrypted_reasoning";
                let object = block.as_object_mut().ok_or_else(PublicError::internal)?;
                object.remove("carrier_size_bytes");
                if encrypted {
                    if object.get("id").is_none_or(Value::is_null) {
                        object.remove("id");
                    }
                    // Excluded output metadata is appended after normal fields by Python.
                    let index = object.shift_remove("output_index").unwrap_or(Value::Null);
                    let status = object.shift_remove("status").unwrap_or(Value::Null);
                    object.insert("output_index".into(), index);
                    object.insert("status".into(), status);
                }
                Ok(block)
            })
            .collect::<Result<Vec<_>, PublicError>>()?;
        replay["provider_reasoning"] = json!(blocks);
    }
    let mut calls = Vec::new();
    for (index, call) in items(&message["tool_calls"]).iter().enumerate() {
        if [
            "raw_arguments",
            "provider_item_id",
            "provider_output_index",
            "provider_status",
            "provider_namespace",
            "provider_caller",
            "provider_thought_signature",
        ]
        .iter()
        .all(|k| call[k].is_null())
        {
            continue;
        }
        let mut entry = json!({"tool_call_index": index, "call_id": call["call_id"], "name": call["name"],
            "raw_arguments": call["raw_arguments"], "provider_item_id": call["provider_item_id"],
            "provider_output_index": call["provider_output_index"], "provider_status": call["provider_status"]});
        for key in [
            "provider_namespace",
            "provider_caller",
            "provider_thought_signature",
        ] {
            if !call[key].is_null() {
                entry[key] = call[key].clone();
            }
        }
        calls.push(entry);
    }
    if !calls.is_empty() {
        replay["tool_calls"] = json!(calls);
    }
    if message["tool_is_error"] == true {
        replay["tool_is_error"] = json!(true);
    }
    if !items(&message["tool_references"]).is_empty() {
        replay["tool_references"] = message["tool_references"].clone();
    }
    for key in [
        "provider_tool_name",
        "provider_tool_namespace",
        "provider_tool_caller",
    ] {
        if !message[key].is_null() {
            replay[key] = message[key].clone();
        }
    }
    Ok(replay)
}
