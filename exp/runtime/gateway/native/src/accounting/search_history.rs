//! Provider history transforms, preserving authenticated and cache-marked fields.

use super::search_subject::{items, present};
use serde_json::{json, Value};

pub(super) fn instruction(m: &Value) -> bool {
    matches!(m["role"].as_str(), Some("system" | "developer"))
}
fn plain(m: &Value) -> bool {
    !m["content"].is_null()
        && items(&m["content_parts"]).is_empty()
        && [
            "provider_native_item",
            "provider_anthropic_block",
            "provider_anthropic_blocks",
        ]
        .iter()
        .all(|k| m[k].is_null())
}

fn carriers(run: &[Value]) -> Vec<Value> {
    if !run.iter().any(|m| present(&m["provider_text_blocks"])) {
        return vec![];
    }
    let mut blocks = vec![];
    for (i, m) in run.iter().enumerate() {
        if i > 0 {
            blocks.push(json!({"type": "text", "text": "\n\n"}));
        }
        if present(&m["provider_text_blocks"]) {
            blocks.extend(items(&m["provider_text_blocks"]).to_vec());
        } else {
            blocks.push(json!({"type": "text", "text": m["content"].as_str().unwrap_or("")}));
        }
    }
    blocks
}

fn fold_run(head: &mut Vec<Value>, run: &[Value]) {
    if let Some(last) = head.last_mut().filter(|m| m["role"] == "user" && plain(m)) {
        let all: Vec<_> = std::iter::once(last.clone())
            .chain(run.iter().cloned())
            .collect();
        last["content"] = json!(all
            .iter()
            .map(|m| m["content"].as_str().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n\n"));
        last["provider_text_blocks"] = json!(carriers(&all));
    } else {
        head.extend(run.iter().map(|m| {
            let mut m = m.clone();
            m["role"] = json!("user");
            m
        }));
    }
}

pub(super) fn instructions(
    messages: Vec<Value>,
    merge_leading: bool,
    trailing_only: bool,
) -> Vec<Value> {
    let is_instruction = |m: &Value| instruction(m) && plain(m);
    if trailing_only {
        let start = messages
            .iter()
            .rposition(|m| !is_instruction(m))
            .map_or(0, |i| i + 1);
        if start == 0 || start == messages.len() {
            return messages;
        }
        let mut head = messages[..start].to_vec();
        fold_run(&mut head, &messages[start..]);
        return head;
    }
    let leading = messages.iter().take_while(|m| is_instruction(m)).count();
    let mut out = messages[..leading].to_vec();
    if merge_leading && leading > 1 {
        let mut first = out[0].clone();
        first["content"] = json!(out
            .iter()
            .map(|m| m["content"].as_str().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n\n"));
        first["provider_text_blocks"] = json!(carriers(&out));
        out = vec![first];
    }
    let mut i = leading;
    while i < messages.len() {
        if is_instruction(&messages[i]) {
            let end = (i..messages.len())
                .find(|j| !is_instruction(&messages[*j]))
                .unwrap_or(messages.len());
            fold_run(&mut out, &messages[i..end]);
            i = end;
        } else {
            out.push(messages[i].clone());
            i += 1;
        }
    }
    out
}

fn plain_assistant(m: &Value) -> bool {
    items(&m["content_parts"]).is_empty()
        && items(&m["provider_text_blocks"]).is_empty()
        && [
            "cache_control",
            "provider_native_item",
            "provider_anthropic_block",
            "provider_anthropic_blocks",
        ]
        .iter()
        .all(|k| m[k].is_null())
}

pub(super) fn progress(messages: Vec<Value>) -> Vec<Value> {
    let mut out = vec![];
    let mut i = 0;
    while i < messages.len() {
        let host = &messages[i];
        if host["role"] != "assistant" || items(&host["tool_calls"]).is_empty() {
            out.push(host.clone());
            i += 1;
            continue;
        }
        let mut end = i + 1;
        while end < messages.len()
            && messages[end]["role"] == "assistant"
            && items(&messages[end]["tool_calls"]).is_empty()
            && messages[end]["provider_native_item"].is_null()
        {
            end += 1;
        }
        if end == i + 1 || end == messages.len() || messages[end]["role"] != "tool" {
            out.push(host.clone());
            i += 1;
            continue;
        }
        let run = &messages[i..end];
        let reasoning: Vec<_> = run
            .iter()
            .flat_map(|m| items(&m["provider_reasoning"]).iter().cloned())
            .collect();
        if run.iter().all(plain_assistant)
            && reasoning
                .iter()
                .filter(|b| !matches!(b["kind"].as_str(), Some("thinking" | "redacted_thinking")))
                .count()
                <= 1
        {
            let mut merged = host.clone();
            let texts: Vec<_> = run
                .iter()
                .filter_map(|m| m["content"].as_str().filter(|s| !s.is_empty()))
                .collect();
            if !texts.is_empty() {
                merged["content"] = json!(texts.join("\n\n"));
            }
            merged["provider_reasoning"] = json!(reasoning);
            merged["capture_only_reasoning"] = json!(run
                .iter()
                .flat_map(|m| items(&m["capture_only_reasoning"]).iter().cloned())
                .collect::<Vec<_>>());
            out.push(merged);
        } else {
            out.extend_from_slice(&messages[i + 1..end]);
            out.push(host.clone());
        }
        i = end;
    }
    out
}

pub(super) fn attachments(messages: Vec<Value>, images: bool) -> Vec<Value> {
    let mut out = vec![];
    let mut pending = vec![];
    let mut counts = [0, 0];
    for mut message in messages {
        if message["role"] != "tool" {
            flush(&mut out, &mut pending);
            out.push(message);
            continue;
        }
        let parts = items(&message["content_parts"]);
        if !parts
            .iter()
            .any(|p| p["kind"] == "document" || (images && p["kind"] == "image"))
        {
            out.push(message);
            continue;
        }
        let mut texts = String::new();
        let mut kept = vec![];
        for part in parts {
            let kind = part["kind"].as_str().unwrap_or("");
            if kind == "document" || (images && kind == "image") {
                let index = usize::from(kind == "document");
                counts[index] += 1;
                let number = counts[index];
                pending.push((
                    number,
                    message["tool_call_id"].as_str().unwrap_or("").to_owned(),
                    part.clone(),
                ));
                let marker = format!("[{kind} {number}: attached in the next user message]");
                texts.push_str(&marker);
                kept.push(json!({"kind": "text", "text": marker, "cache_control": null}));
            } else if kind == "text" {
                let text = part["text"].as_str().unwrap_or("");
                texts.push_str(text);
                if !text.is_empty() {
                    kept.push(part.clone());
                }
            } else {
                kept.push(part.clone());
            }
        }
        if kept.iter().all(|p| p["kind"] == "text") {
            kept.clear();
        }
        message["content"] = json!(texts);
        message["content_parts"] = json!(kept);
        out.push(message);
    }
    flush(&mut out, &mut pending);
    out
}

fn flush(out: &mut Vec<Value>, pending: &mut Vec<(usize, String, Value)>) {
    if pending.is_empty() {
        return;
    }
    let images = pending.iter().any(|(_, _, p)| p["kind"] == "image");
    let docs = pending.iter().any(|(_, _, p)| p["kind"] == "document");
    let header = match (images, docs) {
        (true, false) => "Images returned by the tool results above. This model's wire carries only text inside tool messages, so each image is attached here, numbered where it appeared:",
        (false, true) => "Documents returned by the tool results above. This model's wire cannot carry a document inside a tool result, so each document is attached here, numbered where it appeared:",
        _ => "Images and documents returned by the tool results above. This model's wire carries only text inside tool messages, so each one is attached here, numbered where it appeared:",
    };
    let mut text = header.to_owned();
    let mut parts = vec![json!({"kind": "text", "text": header})];
    for (number, id, part) in pending.drain(..) {
        let label = if part["kind"] == "image" {
            "Image"
        } else {
            "Document"
        };
        let line = format!("\n{label} {number} (tool_call_id {id}):");
        text.push_str(&line);
        parts.push(json!({"kind": "text", "text": line}));
        parts.push(part);
    }
    out.push(json!({"role": "user", "content": text, "content_parts": parts}));
}

pub(super) fn marked(blocks: &[Value], ordered: bool) -> Vec<Value> {
    let mut out: Vec<Value> = vec![];
    let mut displaced: Option<Value> = None;
    let markable =
        |b: &Value| !matches!(b["type"].as_str(), Some("thinking" | "redacted_thinking"));
    for block in blocks {
        if (!ordered || block["type"] == "text") && !present(&block["text"]) {
            if let Some(marker) = block
                .get("cache_control")
                .filter(|v| !ordered || !v.is_null())
            {
                if let Some(last) = out.iter_mut().rev().find(|b| !ordered || markable(b)) {
                    if last.get("cache_control").is_none() {
                        last["cache_control"] = marker.clone();
                    }
                } else {
                    displaced = Some(marker.clone());
                }
            }
            continue;
        }
        let mut kept = block.clone();
        if (!ordered || markable(&kept)) && kept.get("cache_control").is_none() {
            if let Some(marker) = displaced.take() {
                kept["cache_control"] = marker;
            }
        }
        if !ordered {
            displaced = None;
        }
        out.push(kept);
    }
    out
}
