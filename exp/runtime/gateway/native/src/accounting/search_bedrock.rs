//! Converse conversation encoding with cache points and canonical tool results.

use super::{
    search_encode::{self as encode, Options},
    search_history as history, search_media as media,
    search_subject::{folded, items, present},
};
use crate::errors::PublicError;
use serde_json::{json, Value};

pub(super) fn rebuild(
    payload: &mut Value,
    request: &Value,
    messages: Vec<Value>,
    o: &Options,
) -> Result<(), PublicError> {
    let mut system = vec![];
    let mut out = vec![];
    for m in &messages {
        if history::instruction(m) {
            if m["content"].is_null() {
                return Err(PublicError::internal());
            }
            system.extend(text(m));
            continue;
        }
        if m["role"] == "tool" {
            let mut result = vec![];
            if items(&m["content_parts"]).is_empty() {
                result.push(json!({"text":folded(m)}));
            } else {
                for (i, p) in items(&m["content_parts"]).iter().enumerate() {
                    if p["kind"] == "image" {
                        result.push(media::part(p, "bedrock", i + 1)?);
                    } else if p["kind"] == "text" && present(&p["text"]) {
                        result.push(json!({"text":p["text"]}));
                    }
                }
                if m["tool_is_error"] == true {
                    result.insert(0, json!({"text":"[tool error] "}));
                }
                if result.is_empty() {
                    result.push(json!({"text":m["content"].as_str().unwrap_or("")}));
                }
            }
            let mut blocks = vec![
                json!({"toolResult":{"toolUseId":m["tool_call_id"].as_str().unwrap_or(""),"content":result}}),
            ];
            point(&mut blocks, &m["cache_control"]);
            encode::push(&mut out, "user", blocks, "content");
        } else {
            let blocks = blocks(m)?;
            encode::push(
                &mut out,
                if m["role"] == "assistant" {
                    "assistant"
                } else {
                    "user"
                },
                blocks,
                "content",
            );
        }
    }
    if request["json_object_output"] == true {
        system.push(json!({"text":o.json_instruction}));
    }
    encode::remove(payload, "system");
    if !system.is_empty() {
        payload["system"] = json!(system);
    }
    if !request["provider_cache_control"].is_null() {
        if let Some(blocks) = out.last_mut().and_then(|m| m["content"].as_array_mut()) {
            if blocks.last().is_some_and(|b| b.get("cachePoint").is_none()) {
                point(blocks, &request["provider_cache_control"]);
            }
        }
    }
    payload["messages"] = json!(out);
    Ok(())
}

fn point(blocks: &mut Vec<Value>, marker: &Value) {
    if marker.is_null() {
        return;
    }
    let mut value = json!({"type":"default"});
    media::optional(&mut value, "ttl", &marker["ttl"]);
    blocks.push(json!({"cachePoint":value}));
}

fn text(m: &Value) -> Vec<Value> {
    if items(&m["provider_text_blocks"]).is_empty() {
        return vec![json!({"text":m["content"].as_str().unwrap_or("")})];
    }
    let mut out = vec![];
    for b in history::marked(items(&m["provider_text_blocks"]), false) {
        if present(&b["text"]) {
            out.push(json!({"text":b["text"]}));
        }
        if b["cache_control"].is_object() && !out.is_empty() {
            point(&mut out, &b["cache_control"]);
        }
    }
    out
}

fn blocks(m: &Value) -> Result<Vec<Value>, PublicError> {
    if m["role"] == "user" && m["content"].is_null() {
        return Err(PublicError::internal());
    }
    let mut out = vec![];
    if !items(&m["content_parts"]).is_empty() {
        let marked = history::marked(items(&m["provider_text_blocks"]), false);
        let mut marked = marked.iter();
        let mut ordinal = 0;
        for p in items(&m["content_parts"]) {
            let kind = p["kind"].as_str().unwrap_or("");
            if kind == "document" {
                ordinal += 1;
            }
            if kind == "text" {
                if !present(&p["text"]) {
                    continue;
                }
                out.push(json!({"text":p["text"]}));
                let marker = marked
                    .next()
                    .map(|b| &b["cache_control"])
                    .filter(|v| v.is_object())
                    .unwrap_or(&p["cache_control"]);
                point(&mut out, marker);
            } else {
                out.push(media::part(p, "bedrock", ordinal)?);
                if matches!(kind, "image" | "document") {
                    point(&mut out, &p["cache_control"]);
                }
            }
        }
        return Ok(out);
    }
    if present(&m["content"]) {
        out.extend(text(m));
    }
    for c in items(&m["tool_calls"]) {
        out.push(
            json!({"toolUse":{"toolUseId":c["call_id"],"name":c["name"],"input":c["arguments"]}}),
        );
        point(&mut out, &c["cache_control"]);
    }
    if out.is_empty() {
        Err(PublicError::internal())
    } else {
        Ok(out)
    }
}
