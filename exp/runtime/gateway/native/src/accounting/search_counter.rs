//! Extend the admission counter with loaded tools and complete search conversations.

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::errors::PublicError;

use super::{json_text, search::Tool, search_round::Frame, tokenizer};

#[path = "search_message_counter.rs"]
mod message_counter;
pub(super) use message_counter::message_replay;

#[derive(Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Parts {
    chunks: Vec<String>,
    fixed: i64,
    opaque_bytes: i64,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Counter {
    messages: Parts,
    suffix: Parts,
    tools: HashMap<String, Parts>,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    replay: Value,
    message_count: usize,
    message_framing: u64,
    tools_present: u64,
    opaque_bytes_per_token: u64,
    headroom_percent: u64,
    reset_letters: String,
}

impl Parts {
    fn append(&mut self, part: &Self) {
        self.chunks.extend(part.chunks.iter().cloned());
        self.fixed += part.fixed;
        self.opaque_bytes += part.opaque_bytes;
    }
    fn text(&mut self, text: &str) {
        if !text.is_empty() {
            self.chunks.push(text.to_owned());
        }
    }
    fn scrub(&mut self, value: &mut Value) {
        match value {
            Value::String(text)
                if text.len() >= 256
                    && text.bytes().all(|b| {
                        b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'_' | b'=' | b'-')
                    }) =>
            {
                self.opaque_bytes += (text.len() * 3 / 4) as i64
                    - (text.len() - text.trim_end_matches('=').len()) as i64;
                text.clear();
            }
            Value::Array(items) => items.iter_mut().for_each(|v| self.scrub(v)),
            Value::Object(items) => items.values_mut().for_each(|v| self.scrub(v)),
            _ => (),
        }
    }
}

impl Counter {
    pub fn count(&self, tools: &[Tool], frames: &[Frame]) -> Result<(u64, u64), PublicError> {
        if self.opaque_bytes_per_token == 0
            || self.headroom_percent > 1000
            || self.message_framing > 1000
            || self.tools_present > 1_000_000
        {
            return Err(PublicError::internal());
        }
        let mut parts = self.messages.clone();
        let mut message_index = self.message_count;
        let mut replay = if self.replay.is_null() {
            json!({"provider_replay": [],
            "provider_thinking_config": null, "reasoning_context": null,
            "context_management": null, "provider_output_config": null})
        } else {
            self.replay.clone()
        };
        for frame in frames {
            parts.fixed += (self.message_framing * (1 + frame.results.len() as u64)) as i64;
            for call in &frame.calls {
                parts.text(&call.call_id);
                parts.text(&call.name);
            }
            for (call, result) in frame.calls.iter().zip(&frame.results) {
                parts.text(result);
                parts.text(&call.call_id);
            }
            if !frame.calls.is_empty() {
                replay["provider_replay"].as_array_mut().ok_or_else(PublicError::internal)?.push(json!({
                    "message_index": message_index, "tool_calls": frame.calls.iter().enumerate().map(|(i, call)| json!({
                        "tool_call_index": i, "call_id": call.call_id, "name": call.name,
                        "raw_arguments": call.raw_arguments, "provider_item_id": null,
                        "provider_output_index": null, "provider_status": null,
                    })).collect::<Vec<_>>() }));
            }
            message_index += 1 + frame.results.len();
        }
        if !tools.is_empty() {
            parts.fixed += self.tools_present as i64;
        }
        for tool in tools {
            parts.append(
                self.tools
                    .get(&tool.key)
                    .ok_or_else(PublicError::internal)?,
            );
        }
        parts.append(&self.suffix);
        let carriers: Vec<_> = tools
            .iter()
            .enumerate()
            .filter_map(|(i, tool)| {
                let mut entry = json!({"tool_index": i, "name": tool.name});
                for key in ["eager_input_streaming", "allowed_callers", "input_examples"] {
                    if let Some(value) = tool.extra.get(key).filter(|v| !v.is_null()) {
                        entry[key] = value.clone();
                    }
                }
                (entry.as_object().unwrap().len() > 2).then_some(entry)
            })
            .collect();
        let object = replay.as_object_mut().ok_or_else(PublicError::internal)?;
        let original = std::mem::take(object);
        let mut carriers = (!carriers.is_empty()).then(|| json!(carriers));
        // The replay envelope places tool carriers before these optional tails.
        for (key, value) in original {
            if key == "tools" {
                continue;
            }
            if [
                "provider_beta_tokens",
                "provider_server_tools",
                "provider_native_tools",
            ]
            .contains(&key.as_str())
            {
                if let Some(tools) = carriers.take() {
                    object.insert("tools".into(), tools);
                }
            }
            object.insert(key, value);
        }
        if let Some(tools) = carriers {
            object.insert("tools".into(), tools);
        }
        if !self.replay.is_null() || !frames.is_empty() || object.contains_key("tools") {
            parts.scrub(&mut replay);
            parts.chunks.push(json_text::dumps(&replay, false)?);
        }
        let counted = tokenizer::bound(&parts.chunks.join("\n"), &self.reset_letters)? as i128
            + i128::from(parts.fixed)
            + i128::from(parts.opaque_bytes).div_euclid(i128::from(self.opaque_bytes_per_token));
        let counted = u64::try_from(counted).map_err(|_| PublicError::internal())?;
        let reserved =
            (u128::from(counted) * u128::from(100 + self.headroom_percent)).div_ceil(100);
        Ok((
            counted,
            u64::try_from(reserved).map_err(|_| PublicError::internal())?,
        ))
    }
}
