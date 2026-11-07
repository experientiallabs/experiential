//! Admission-frozen capability fences and message-dependent provider shaping.

use super::{
    search_encode,
    search_subject::{items, present},
};
use crate::errors::PublicError;
use serde_json::{json, Value};

pub(super) fn prepare(
    messages: &mut [Value],
    dialect: &str,
    request: &Value,
    shaping: &Value,
) -> Result<(), PublicError> {
    if shaping.is_null() {
        return Ok(());
    }
    requirements(messages, dialect, request, &shaping["requirements"])?;
    for m in messages {
        if shaping["opaque_arguments"] != true {
            if let Some(calls) = m["tool_calls"].as_array_mut() {
                for c in calls {
                    if !c["arguments_treatment"].is_null() {
                        c["raw_arguments"] = Value::Null;
                        c["arguments_treatment"] = Value::Null;
                    }
                }
            }
        }
        if shaping["downgrade_original"] == true {
            if let Some(parts) = m["content_parts"].as_array_mut() {
                for p in parts {
                    if p["kind"] == "image" && p["detail"] == "original" {
                        p["detail"] = json!("high");
                    }
                }
            }
        }
        if dialect != "anthropic_messages" && !items(&m["tool_references"]).is_empty() {
            let text = items(&m["tool_references"]).iter().map(|name| {
                name.as_str().map(|name| format!("Tool loaded: {name}. Its full definition is now in your tools; you can call it.")).ok_or_else(PublicError::internal)
            }).collect::<Result<Vec<_>,_>>()?.join("\n");
            let content = m["content"].as_str().unwrap_or("");
            let appended = format!("{}{text}", if content.is_empty() { "" } else { "\n" });
            m["content"] = json!(format!("{content}{appended}"));
            m["tool_references"] = json!([]);
            if let Some(parts) = m["content_parts"].as_array_mut().filter(|a| !a.is_empty()) {
                parts.push(json!({"kind":"text","text":appended}));
            }
        }
    }
    Ok(())
}

fn requirements(
    messages: &[Value],
    dialect: &str,
    request: &Value,
    authority: &Value,
) -> Result<(), PublicError> {
    if authority.is_null() {
        return Ok(());
    }
    let supported = |capability: &str| {
        authority["capabilities"][format!("supports_{capability}")] == true
            || items(&authority["emulated"])
                .iter()
                .any(|v| v == capability)
    };
    if authority["model_tools"] != true
        && (present(&request["tools"]) || present(&request["provider_native_tools"]))
    {
        return Err(PublicError::internal());
    }
    if request["stream"] == true
        && present(&request["tools"])
        && !supported("streaming_tool_arguments")
    {
        return Err(PublicError::internal());
    }
    for m in messages {
        if m["role"] == "developer" && !supported("developer_messages") {
            return Err(PublicError::internal());
        }
        for p in items(&m["content_parts"]) {
            let kind = match p["kind"].as_str() {
                Some("text") => continue,
                Some("document") => "pdf",
                Some("image") => "image",
                Some("video") => "video",
                Some("audio") => "audio",
                _ => return Err(PublicError::internal()),
            };
            if !supported(&format!("{kind}_input"))
                || (!p["url"].is_null() && !supported(&format!("{kind}_url_input")))
            {
                return Err(PublicError::internal());
            }
            if m["role"] == "assistant"
                && kind == "image"
                && !matches!(dialect, "openai_compatible" | "gemini_generate_content")
            {
                return Err(PublicError::internal());
            }
            if !p["handle"].is_null()
                && (!supported("media_handle_input")
                    || p["handle"]["provider"] != authority["provider"])
            {
                return Err(PublicError::internal());
            }
        }
    }
    Ok(())
}

pub(super) fn finish(
    payload: &mut Value,
    dialect: &str,
    request: &Value,
    shaping: &Value,
) -> Result<(), PublicError> {
    if shaping.is_null() {
        return Ok(());
    }
    if shaping["vertex_anthropic"] == true
        && dialect == "anthropic_messages"
        && items(&payload["messages"])
            .iter()
            .flat_map(|m| items(&m["content"]))
            .any(|b| b["type"] == "file" || b["source"]["type"] == "file")
    {
        return Err(PublicError::internal());
    }
    if shaping["thinking"].is_object() {
        let mut open = false;
        for m in items(&request["messages"]).iter().rev() {
            if m["role"] == "user" {
                break;
            }
            if m["role"] == "assistant" && !items(&m["tool_calls"]).is_empty() {
                open = true;
                break;
            }
        }
        let variant = &shaping["thinking"][if open { "open" } else { "closed" }];
        for key in ["thinking", "output_config"] {
            if variant[key].is_null() {
                search_encode::remove(payload, key);
            } else {
                payload[key] = variant[key].clone();
            }
        }
    }
    Ok(())
}
