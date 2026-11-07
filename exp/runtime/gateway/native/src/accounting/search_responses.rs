//! Responses conversation replay, retaining provider item order and attribution.

use super::{
    search_encode::{self as encode, Options},
    search_history::instruction,
    search_media as media,
    search_subject::{folded, items},
};
use crate::errors::PublicError;
use serde_json::{json, Value};

pub(super) fn rebuild(
    payload: &mut Value,
    request: &Value,
    messages: Vec<Value>,
    o: &Options,
) -> Result<(), PublicError> {
    let marked = o.cache_options && request["prompt_cache_breakpoints"] == true;
    let mut instructions = vec![];
    let mut roles = vec![];
    let mut out = vec![];
    for m in &messages {
        if !m["provider_native_item"].is_null() {
            if let Some(item) = native(&m["provider_native_item"]) {
                out.push(item);
            }
        } else if instruction(m) {
            let text = m["content"].as_str().ok_or_else(PublicError::internal)?;
            if !out.is_empty() || (marked && encode::has_breakpoints(m)) {
                let mut wire = json!({"role": m["role"], "content": text});
                if marked {
                    encode::breakpoints(m, &mut wire, true);
                }
                out.push(wire);
            } else {
                instructions.push(text);
                roles.push(m["role"].clone());
            }
        } else {
            let mut wires = message(m)?;
            if marked && matches!(m["role"].as_str(), Some("user" | "tool")) {
                for wire in &mut wires {
                    encode::breakpoints(m, wire, true);
                }
            }
            out.extend(wires);
        }
    }
    if out.is_empty() && !instructions.is_empty() {
        out.extend(
            roles
                .iter()
                .zip(&instructions)
                .map(|(role, text)| json!({"role": role, "content": text})),
        );
        instructions.clear();
    }
    if request["json_object_output"] == true {
        out.insert(0, json!({"role": "system", "content": o.json_instruction}));
    }
    encode::remove(payload, "instructions");
    if !instructions.is_empty() {
        payload["instructions"] = json!(instructions.join("\n\n"));
    }
    payload["input"] = json!(out);
    Ok(())
}

fn issued(id: &str, kind: &str) -> bool {
    let prefix = match kind {
        "message" => "msg",
        "function_call" => "fc",
        "reasoning" => "rs",
        _ => return true,
    };
    id.split_once('_').is_some_and(|(head, rest)| {
        head == prefix
            && !rest.is_empty()
            && rest.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
    })
}

fn native(item: &Value) -> Option<Value> {
    let mut v = item.clone();
    if v["type"] == "message" {
        if let Some(parts) = v["content"].as_array_mut() {
            for p in parts {
                if p["type"] == "output_text" {
                    encode::remove(p, "logprobs");
                }
            }
        }
    }
    if v["id"].as_str().is_some_and(|id| {
        id.starts_with("item_") || !issued(id, v["type"].as_str().unwrap_or("message"))
    }) {
        if v["type"] == "reasoning" && v.get("encrypted_content").is_none() {
            return None;
        }
        encode::remove(&mut v, "id");
    }
    if (v["type"].is_null() || v["type"] == "message")
        && matches!(v["role"].as_str(), Some("user" | "system" | "developer"))
        || v["type"] == "reasoning"
    {
        encode::remove(&mut v, "status");
    }
    Some(v)
}

fn message(m: &Value) -> Result<Vec<Value>, PublicError> {
    if m["role"] == "tool" || m["role"] == "user" {
        let parts = items(&m["content_parts"]);
        let tool = m["role"] == "tool";
        let content = if parts.is_empty() {
            if tool {
                json!(folded(m))
            } else {
                json!(m["content"].as_str().unwrap_or(""))
            }
        } else {
            let mut parts = parts
                .iter()
                .enumerate()
                .map(|(i, p)| media::part(p, "responses", i + 1))
                .collect::<Result<Vec<_>, _>>()?;
            if tool && m["tool_is_error"] == true {
                parts.insert(0, json!({"type": "input_text", "text": "[tool error] "}));
            }
            json!(parts)
        };
        let mut wire = if tool {
            json!({"type": "function_call_output", "call_id": m["tool_call_id"].as_str().unwrap_or(""), "output": content})
        } else {
            json!({"role": "user", "content": content})
        };
        if tool {
            for (to, from) in [
                ("name", "provider_tool_name"),
                ("namespace", "provider_tool_namespace"),
                ("caller", "provider_tool_caller"),
            ] {
                media::optional(&mut wire, to, &m[from]);
            }
        }
        return Ok(vec![wire]);
    }
    if m["role"] != "assistant" {
        return Err(PublicError::internal());
    }
    let mut out = vec![];
    let mut indexed = std::collections::BTreeMap::new();
    let mut push = |index: &Value, item: Value| -> Result<(), PublicError> {
        if index.is_null() {
            out.push(item);
        } else if indexed
            .insert(index.as_u64().ok_or_else(PublicError::internal)?, item)
            .is_some()
        {
            return Err(PublicError::internal());
        }
        Ok(())
    };
    for block in items(&m["provider_reasoning"]) {
        match block["kind"].as_str() {
            Some("exposed_reasoning_content" | "thinking" | "redacted_thinking") => continue,
            Some("encrypted_reasoning") => push(
                &block["output_index"],
                json!({"type": "reasoning", "summary": [], "encrypted_content": block["encrypted_content"]}),
            )?,
            _ => return Err(PublicError::internal()),
        }
    }
    if !m["content"].is_null() || !m["provider_item_id"].is_null() {
        let mut item = json!({"role": "assistant", "content": m["content"]});
        if !m["provider_output_index"].is_null() {
            let id = m["provider_item_id"]
                .as_str()
                .ok_or_else(PublicError::internal)?;
            item["content"] = json!(m["content"].as_str().unwrap_or(""));
            if issued(id, "message") {
                item = json!({"type": "message", "id": id, "role": "assistant", "status": m["provider_status"].as_str().filter(|s| !s.is_empty()).unwrap_or("completed"), "content": if m["content"].is_null() { vec![] } else { vec![json!({"type": "output_text", "text": m["content"], "annotations": []})] }});
                media::optional(&mut item, "phase", &m["provider_phase"]);
            }
        }
        push(&m["provider_output_index"], item)?;
    }
    for call in items(&m["tool_calls"]) {
        let mut item = json!({"type": "function_call", "call_id": call["call_id"], "name": call["name"], "arguments": encode::arguments(call)?});
        for (to, from) in [
            ("namespace", "provider_namespace"),
            ("caller", "provider_caller"),
            ("status", "provider_status"),
        ] {
            media::optional(&mut item, to, &call[from]);
        }
        if call["provider_item_id"]
            .as_str()
            .is_some_and(|id| issued(id, "function_call"))
        {
            item["id"] = call["provider_item_id"].clone();
        }
        push(&call["provider_output_index"], item)?;
    }
    Ok(indexed.into_values().chain(out).collect())
}
