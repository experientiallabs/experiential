//! Canonical message invariants that JSON Schema cannot express.

use std::collections::HashSet;
use std::sync::OnceLock;

use serde_json::Value;

use super::search_subject::{items, present};

fn has(value: &Value, key: &str) -> bool {
    !value[key].is_null()
}

pub(super) fn coherent(m: &Value) -> Result<(), ()> {
    for call in items(&m["tool_calls"]) {
        if has(call, "arguments_treatment") {
            if !has(call, "raw_arguments") {
                return Err(());
            }
        } else if let Some(raw) = call["raw_arguments"].as_str() {
            let decoded: Value = crate::lossless_json::from_str(raw).map_err(|_| ())?;
            if !decoded.is_object() || !super::equality::equal(&decoded, &call["arguments"]) {
                return Err(());
            }
        }
        if (has(call, "provider_item_id") || has(call, "provider_status"))
            && !has(call, "provider_output_index")
        {
            return Err(());
        }
    }
    for part in items(&m["content_parts"]) {
        media(part)?;
    }
    if has(m, "provider_native_item") || has(m, "provider_anthropic_block") {
        if has(m, "content")
            || present(&m["tool_calls"])
            || present(&m["provider_reasoning"])
            || has(m, "provider_item_id")
            || has(m, "tool_call_id")
            || m["tool_is_error"] == true
        {
            return Err(());
        }
        return Ok(());
    }
    if !has(m, "content")
        && !present(&m["tool_calls"])
        && !present(&m["provider_reasoning"])
        && !has(m, "provider_item_id")
    {
        return Err(());
    }
    let assistant = m["role"] == "assistant";
    let tool = m["role"] == "tool";
    if !assistant
        && (["tool_calls", "provider_reasoning", "capture_only_reasoning"]
            .iter()
            .any(|k| present(&m[k]))
            || [
                "provider_item_id",
                "provider_output_index",
                "provider_status",
                "provider_phase",
            ]
            .iter()
            .any(|k| has(m, k)))
    {
        return Err(());
    }
    if has(m, "provider_item_id") != has(m, "provider_output_index")
        || ((has(m, "provider_status") || has(m, "provider_phase")) && !has(m, "provider_item_id"))
    {
        return Err(());
    }
    let mut ids = HashSet::new();
    if items(&m["tool_calls"])
        .iter()
        .any(|c| !ids.insert(c["call_id"].as_str()))
    {
        return Err(());
    }
    if tool != has(m, "tool_call_id") {
        return Err(());
    }
    if !tool
        && ([
            "provider_tool_name",
            "provider_tool_namespace",
            "provider_tool_caller",
            "cache_control",
        ]
        .iter()
        .any(|k| has(m, k))
            || m["tool_is_error"] == true
            || present(&m["tool_references"]))
    {
        return Err(());
    }
    if present(&m["provider_text_blocks"]) {
        if tool {
            return Err(());
        }
        let text: Vec<_> = items(&m["provider_text_blocks"])
            .iter()
            .map(|b| match &b["text"] {
                Value::String(s) => s.clone(),
                Value::Null if b.get("text").is_none() => String::new(),
                // Text blocks are opaque JSON, but their Python str() flattening
                // still participates in the message invariant.
                v => python_text(v),
            })
            .collect();
        let content = m["content"].as_str().unwrap_or("");
        if content != text.join("") && content != text.join("\n\n") {
            return Err(());
        }
    }
    let parts = items(&m["content_parts"]);
    if !parts.is_empty() {
        if !matches!(m["role"].as_str(), Some("user" | "tool" | "assistant"))
            || parts.iter().all(|p| p["kind"] == "text")
            || (tool
                && parts
                    .iter()
                    .any(|p| !matches!(p["kind"].as_str(), Some("text" | "image" | "document"))))
            || (assistant
                && parts
                    .iter()
                    .any(|p| !matches!(p["kind"].as_str(), Some("text" | "image"))))
        {
            return Err(());
        }
        let text: String = parts
            .iter()
            .filter(|p| p["kind"] == "text")
            .filter_map(|p| p["text"].as_str())
            .collect();
        if m["content"].as_str().unwrap_or("") != text {
            return Err(());
        }
    }
    Ok(())
}

pub(super) fn python_text(value: &Value) -> String {
    if let Some(text) = value.as_str() {
        return text.to_owned();
    }
    python_repr(value)
}

fn python_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(value) => {
            static NONPRINTING: std::sync::LazyLock<regex::Regex> =
                std::sync::LazyLock::new(|| {
                    regex::Regex::new(r"[\p{C}\p{Zl}\p{Zp}]|[\p{Zs}&&[^ ]]").unwrap()
                });
            let quote = if value.contains('\'') && !value.contains('"') {
                '"'
            } else {
                '\''
            };
            let mut text = String::from(quote);
            for c in value.chars() {
                match c {
                    '\n' => text.push_str("\\n"),
                    '\r' => text.push_str("\\r"),
                    '\t' => text.push_str("\\t"),
                    '\\' => text.push_str("\\\\"),
                    c if c == quote => {
                        text.push('\\');
                        text.push(c);
                    }
                    c if NONPRINTING.is_match(c.encode_utf8(&mut [0; 4])) => {
                        let n = c as u32;
                        text.push_str(&if n <= 255 {
                            format!("\\x{n:02x}")
                        } else if n <= 65535 {
                            format!("\\u{n:04x}")
                        } else {
                            format!("\\U{n:08x}")
                        });
                    }
                    c => text.push(c),
                }
            }
            text.push(quote);
            text
        }
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(python_repr).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(items) => format!(
            "{{{}}}",
            items
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    python_repr(&Value::String(key.clone())),
                    python_repr(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Number(number) => {
            super::canonical::number(number, false).unwrap_or_else(|_| number.to_string())
        }
    }
}

fn base64(text: &str) -> bool {
    let n = text
        .bytes()
        .take_while(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/'))
        .count();
    if !text.as_bytes()[n..].iter().all(|b| *b == b'=') {
        return false;
    }
    let padding = text.len() - n;
    match n % 4 {
        0 => padding == 0,
        2 => padding == 2,
        3 => padding == 1,
        _ => false,
    }
}

fn media(part: &Value) -> Result<(), ()> {
    let kind = part["kind"].as_str().ok_or(())?;
    if kind == "text" {
        return Ok(());
    }
    if kind != "audio"
        && ["data", "url", "handle"]
            .iter()
            .filter(|k| has(part, k))
            .count()
            != 1
    {
        return Err(());
    }
    if let Some(data) = part["data"].as_str() {
        if !has(part, "media_type") || !base64(data) {
            return Err(());
        }
    } else if let Some(url) = part["url"].as_str() {
        if !url.starts_with("https://") && !url.starts_with("http://") {
            return Err(());
        }
    }
    if has(part, "handle") {
        let handle = &part["handle"];
        let provider = handle["provider"].as_str().ok_or(())?;
        if matches!(kind, "image" | "video")
            && matches!(provider, "vertex" | "bedrock")
            && !has(part, "media_type")
        {
            return Err(());
        }
        static PATTERNS: OnceLock<Vec<regex::Regex>> = OnceLock::new();
        let patterns = PATTERNS.get_or_init(|| [
            r"\Afile-[A-Za-z0-9_-]{1,256}(?:\n)?\z", r"\Afile_[A-Za-z0-9_-]{1,256}(?:\n)?\z",
            r"\Ahttps://generativelanguage\.googleapis\.com/v1beta/files/[a-z0-9-]{1,128}(?:\n)?\z",
            r"(?s)\Ags://[a-z0-9][a-z0-9._-]{1,254}/.+\z", r"(?s)\As3://[a-z0-9][a-z0-9._-]{1,254}/.+\z",
            r"\A[0-9]{12}(?:\n)?\z",
        ].iter().map(|p| regex::Regex::new(p).unwrap()).collect());
        let index = ["openai", "anthropic", "gemini", "vertex", "bedrock"]
            .iter()
            .position(|p| *p == provider)
            .ok_or(())?;
        if !patterns[index].is_match(handle["reference"].as_str().ok_or(())?) {
            return Err(());
        }
        if has(handle, "bucket_owner")
            && (provider != "bedrock"
                || !patterns[5].is_match(handle["bucket_owner"].as_str().ok_or(())?))
        {
            return Err(());
        }
    }
    Ok(())
}
