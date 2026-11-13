//! Messages conversation encoding, including ordered thinking and displaced markers.

use super::{
    search_encode::{self as encode, Options},
    search_history as history, search_media as media,
    search_subject::{items, present},
};
use crate::errors::PublicError;
use serde_json::{json, Value};

pub(super) fn rebuild(
    payload: &mut Value,
    request: &Value,
    messages: &[Value],
    o: &Options,
) -> Result<(), PublicError> {
    let mut system: Vec<(String, Vec<Value>)> = vec![];
    let mut out: Vec<Value> = vec![];
    let mut displaced = None;
    for m in messages {
        if history::instruction(m) {
            let text = m["content"].as_str().ok_or_else(PublicError::internal)?;
            let blocks = items(&m["provider_text_blocks"]);
            if out.is_empty() {
                system.push((text.into(), blocks.to_vec()));
            } else {
                encode::push(
                    &mut out,
                    "user",
                    if blocks.is_empty() {
                        vec![json!({"type": "text", "text": text})]
                    } else {
                        blocks.to_vec()
                    },
                    "content",
                );
            }
            continue;
        }
        let (role, mut blocks) = blocks(m)?;
        if blocks.is_empty() {
            if let Some(marker) = items(&m["provider_text_blocks"]).iter().find_map(|b| {
                b["cache_control"]
                    .as_object()
                    .map(|_| b["cache_control"].clone())
            }) {
                if let Some(last) = out
                    .iter_mut()
                    .rev()
                    .find(|m| !items(&m["content"]).is_empty())
                    .and_then(|m| m["content"].as_array_mut().and_then(|a| a.last_mut()))
                    .filter(|b| markable(b))
                {
                    if last.get("cache_control").is_none() {
                        last["cache_control"] = marker;
                    }
                } else {
                    displaced = Some(marker);
                }
            }
        } else if let Some(marker) = displaced.take() {
            if let Some(first) = blocks.iter_mut().find(|b| markable(b)) {
                if first.get("cache_control").is_none() {
                    first["cache_control"] = marker;
                }
            }
        }
        encode::push(&mut out, role, blocks, "content");
    }
    if request["json_object_output"] == true {
        system.push((o.json_instruction.clone(), vec![]));
    }
    encode::remove(payload, "system");
    if system.iter().any(|(_, blocks)| !blocks.is_empty()) {
        let mut blocks = vec![];
        for (text, run) in system {
            let separated = !blocks.is_empty();
            if run.is_empty() {
                blocks.push(json!({"type": "text", "text": format!("{}{text}", if separated { "\n\n" } else { "" })}));
            } else {
                let adjacent: String = run.iter().map(block_text).collect();
                for (i, mut b) in run.into_iter().enumerate() {
                    if if i == 0 { separated } else { adjacent != text } {
                        b["text"] = json!(format!("\n\n{}", block_text(&b)));
                    }
                    blocks.push(b);
                }
            }
        }
        let blocks = history::marked(&blocks, false);
        if blocks.iter().any(|b| !block_text(b).trim().is_empty()) {
            payload["system"] = json!(blocks);
        }
    } else {
        let text = system
            .iter()
            .map(|(text, _)| text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        if !text.trim().is_empty() {
            payload["system"] = json!(text);
        }
    }
    payload["messages"] = json!(out);
    Ok(())
}

fn block_text(b: &Value) -> String {
    b.get("text")
        .map(super::input_messages::python_text)
        .unwrap_or_default()
}

fn markable(b: &Value) -> bool {
    !matches!(b["type"].as_str(), Some("thinking" | "redacted_thinking"))
}

fn multimodal(m: &Value) -> Result<Vec<Value>, PublicError> {
    let marked = history::marked(items(&m["provider_text_blocks"]), false);
    let mut marked = marked.iter();
    let mut out = vec![];
    for (i, p) in items(&m["content_parts"]).iter().enumerate() {
        if p["kind"] == "text" {
            if !present(&p["text"]) {
                continue;
            }
            out.push(
                marked
                    .next()
                    .cloned()
                    .unwrap_or_else(|| json!({"type": "text", "text": p["text"]})),
            );
        } else {
            out.push(media::part(p, "anthropic", i + 1)?);
        }
    }
    Ok(out)
}

fn blocks(m: &Value) -> Result<(&'static str, Vec<Value>), PublicError> {
    if m["role"] == "tool" {
        let mut result = json!({"type": "tool_result", "tool_use_id": m["tool_call_id"].as_str().unwrap_or(""), "content": m["content"].as_str().unwrap_or("")});
        let mut parts = vec![];
        for (i, p) in items(&m["content_parts"]).iter().enumerate() {
            if p["kind"] == "text" {
                if present(&p["text"]) {
                    let mut b = json!({"type": "text", "text": p["text"]});
                    media::optional(&mut b, "cache_control", &p["cache_control"]);
                    parts.push(b);
                }
            } else {
                parts.push(media::part(p, "anthropic", i + 1)?);
            }
        }
        if !parts.is_empty() {
            result["content"] = json!(parts);
        }
        if !items(&m["tool_references"]).is_empty() {
            let mut parts = if result["content"].is_array() {
                items(&result["content"]).to_vec()
            } else if present(&result["content"]) {
                vec![json!({"type": "text", "text": result["content"]})]
            } else {
                vec![]
            };
            parts.extend(
                items(&m["tool_references"])
                    .iter()
                    .map(|name| json!({"type": "tool_reference", "tool_name": name})),
            );
            result["content"] = json!(parts);
        }
        if m["tool_is_error"] == true {
            result["is_error"] = json!(true);
        }
        media::optional(&mut result, "cache_control", &m["cache_control"]);
        return Ok(("user", vec![result]));
    }
    if m["role"] == "user" {
        if !items(&m["content_parts"]).is_empty() {
            return Ok(("user", multimodal(m)?));
        }
        let marked = history::marked(items(&m["provider_text_blocks"]), false);
        return Ok((
            "user",
            if marked.is_empty() {
                vec![json!({"type": "text", "text": m["content"].as_str().unwrap_or("")})]
            } else {
                marked
            },
        ));
    }
    if m["role"] != "assistant" {
        return Err(PublicError::internal());
    }
    if !m["provider_anthropic_block"].is_null() {
        return Ok(("assistant", vec![m["provider_anthropic_block"].clone()]));
    }
    let retained = items(&m["provider_anthropic_blocks"])
        .iter()
        .filter(|b| !markable(b))
        .count();
    let thinking = items(&m["provider_reasoning"])
        .iter()
        .filter(|b| matches!(b["kind"].as_str(), Some("thinking" | "redacted_thinking")))
        .count();
    if retained > 0 && retained == thinking {
        return Ok((
            "assistant",
            history::marked(items(&m["provider_anthropic_blocks"]), true),
        ));
    }
    let mut out = vec![];
    for b in items(&m["provider_reasoning"]) {
        match b["kind"].as_str() {
            Some("exposed_reasoning_content") => (),
            Some("thinking") => {
                let mut v = json!({"type": "thinking", "thinking": b["text"]});
                media::optional(&mut v, "signature", &b["signature"]);
                out.push(v);
            }
            Some("redacted_thinking") => {
                out.push(json!({"type": "redacted_thinking", "data": b["data"]}))
            }
            _ => return Err(PublicError::internal()),
        }
    }
    if !items(&m["provider_text_blocks"]).is_empty() {
        out.extend(history::marked(items(&m["provider_text_blocks"]), false));
    } else if present(&m["content"]) {
        out.push(json!({"type": "text", "text": m["content"]}));
    }
    for c in items(&m["tool_calls"]) {
        let mut v = json!({"type": "tool_use", "id": c["call_id"], "name": c["name"], "input": c["arguments"]});
        media::optional(&mut v, "cache_control", &c["cache_control"]);
        out.push(v);
    }
    if !out.is_empty()
        && out
            .iter()
            .all(|b| b["type"] == "text" && block_text(b).trim().is_empty())
    {
        out.clear();
    }
    Ok(("assistant", out))
}
