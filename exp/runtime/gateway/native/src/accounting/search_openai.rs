//! Chat message encoding and active reasoning-history validation.

use super::{
    search_encode::{self as encode, Options},
    search_history as history, search_media as media,
    search_subject::{folded, items},
};
use crate::errors::PublicError;
use serde_json::{json, Value};
use std::collections::HashSet;

pub(super) fn chat(
    payload: &mut Value,
    request: &Value,
    mut messages: Vec<Value>,
    o: &Options,
) -> Result<(), PublicError> {
    let start = messages
        .iter()
        .rposition(|m| m["role"] == "user")
        .map_or(0, |i| i + 1);
    let mut active = false;
    for (i, m) in messages.iter_mut().enumerate() {
        let blocks: Vec<_> = items(&m["provider_reasoning"])
            .iter()
            .filter(|b| b["kind"] == "reasoning_content")
            .collect();
        if i >= start && !blocks.is_empty() {
            active = true;
            if m["role"] != "assistant"
                || items(&m["tool_calls"]).is_empty()
                || blocks.len() != 1
                || o.reasoning_route.as_deref() != blocks[0]["route_sha256"].as_str()
                || o.reasoning_route.is_none()
            {
                return Err(PublicError::internal());
            }
        }
        if i < start {
            m["provider_reasoning"] = json!(items(&m["provider_reasoning"])
                .iter()
                .filter(|b| b["kind"] != "reasoning_content")
                .collect::<Vec<_>>());
        }
    }
    if active {
        complete(&messages[start..])?;
    }
    encode::remove(payload, "reasoning_history");
    if active && o.fireworks {
        payload["reasoning_history"] = json!("interleaved");
    }
    messages = history::progress(messages);
    if o.deepseek || o.model_id.to_lowercase().contains("deepseek") {
        messages = history::instructions(messages, false, true);
    }
    messages = history::attachments(messages, true);
    if o.leading_only {
        messages = history::instructions(messages, true, false);
    }
    let breakpoints = o.cache_options && request["prompt_cache_breakpoints"] == true;
    let mut wires = messages
        .iter()
        .map(|m| message(m, o, o.cache_control && !breakpoints))
        .collect::<Result<Vec<_>, _>>()?;
    if breakpoints {
        for (m, wire) in messages.iter().zip(&mut wires) {
            encode::breakpoints(m, wire, false);
        }
    }
    if request["json_object_output"] == true {
        if let Some(first) = wires.first_mut().filter(|m| m["role"] == "system") {
            if let Some(text) = first["content"].as_str() {
                first["content"] = json!(format!("{text}\n\n{}", o.json_instruction));
            } else if let Some(parts) = first["content"].as_array_mut() {
                parts.push(json!({"type": "text", "text": o.json_instruction}));
            } else {
                wires.insert(0, json!({"role": "system", "content": o.json_instruction}));
            }
        } else {
            wires.insert(0, json!({"role": "system", "content": o.json_instruction}));
        }
    }
    payload["messages"] = json!(wires);
    Ok(())
}

fn complete(messages: &[Value]) -> Result<(), PublicError> {
    let mut pending = HashSet::new();
    let mut calls = HashSet::new();
    let mut completed = HashSet::new();
    for m in messages {
        if m["role"] == "assistant" {
            if !pending.is_empty() {
                return Err(PublicError::internal());
            }
            let signed = items(&m["provider_reasoning"])
                .iter()
                .any(|b| b["kind"] == "reasoning_content");
            for c in items(&m["tool_calls"]) {
                let id = c["call_id"].as_str().ok_or_else(PublicError::internal)?;
                if !calls.insert(id) {
                    return Err(PublicError::internal());
                }
                if signed {
                    pending.insert(id);
                }
            }
        } else if m["role"] == "tool" && !calls.is_empty() {
            let id = m["tool_call_id"]
                .as_str()
                .ok_or_else(PublicError::internal)?;
            if !calls.contains(id) || !completed.insert(id) {
                return Err(PublicError::internal());
            }
            pending.remove(id);
        }
    }
    if pending.is_empty() {
        Ok(())
    } else {
        Err(PublicError::internal())
    }
}

fn message(m: &Value, o: &Options, cache: bool) -> Result<Value, PublicError> {
    if m["role"] == "tool" {
        let mut wire = json!({"role": "tool", "content": folded(m), "tool_call_id": m["tool_call_id"].as_str().unwrap_or("")});
        media::optional(&mut wire, "name", &m["provider_tool_name"]);
        if cache {
            media::optional(&mut wire, "cache_control", &m["cache_control"]);
        }
        return Ok(wire);
    }
    let parts = items(&m["content_parts"]);
    let content = if parts.is_empty() {
        json!(m["content"].as_str().unwrap_or(""))
    } else {
        json!(parts
            .iter()
            .enumerate()
            .map(|(i, p)| media::part(p, "chat", i + 1))
            .collect::<Result<Vec<_>, _>>()?)
    };
    let mut wire = json!({"role": if m["role"] == "developer" { json!("system") } else { m["role"].clone() }, "content": content});
    if cache {
        let marked = history::marked(items(&m["provider_text_blocks"]), false);
        let mut marked = marked.iter();
        if let Some(wires) = wire["content"].as_array_mut() {
            for (part, value) in parts.iter().zip(wires) {
                let marker = match part["kind"].as_str() {
                    Some("text") => marked
                        .next()
                        .and_then(|b| b.get("cache_control"))
                        .unwrap_or(&part["cache_control"]),
                    Some("image" | "document") => &part["cache_control"],
                    _ => &Value::Null,
                };
                if marker.is_object() {
                    value["cache_control"] = marker.clone();
                }
            }
        } else if !items(&m["provider_text_blocks"]).is_empty() {
            wire["content"] = json!(history::marked(items(&m["provider_text_blocks"]), false));
        }
    }
    if !items(&m["tool_calls"]).is_empty() {
        wire["tool_calls"] = json!(items(&m["tool_calls"]).iter().map(|c| {
            let mut value = json!({"id": c["call_id"], "type": "function", "function": {"name": c["name"], "arguments": encode::arguments(c)?}});
            if cache { media::optional(&mut value, "cache_control", &c["cache_control"]); }
            Ok(value)
        }).collect::<Result<Vec<_>,PublicError>>()?);
    }
    let blocks: Vec<_> = items(&m["provider_reasoning"])
        .iter()
        .filter(|b| !matches!(b["kind"].as_str(), Some("thinking" | "redacted_thinking")))
        .collect();
    if !blocks.is_empty() {
        if blocks.len() != 1 {
            return Err(PublicError::internal());
        }
        let b = blocks[0];
        if b["kind"] == "exposed_reasoning_content" {
            if o.exposed || o.deepseek {
                wire["reasoning_content"] = b["content"].clone();
            }
        } else if b["kind"] == "reasoning_content"
            && o.reasoning_route.is_some()
            && o.reasoning_route.as_deref() == b["route_sha256"].as_str()
        {
            wire["reasoning_content"] = b["content"].clone();
        } else {
            return Err(PublicError::internal());
        }
    } else if o.deepseek && m["role"] == "assistant" {
        wire["reasoning_content"] = json!("");
    }
    Ok(wire)
}
