//! The hosted capture copy has always been bounded independently of serving.

use std::sync::Arc;

use serde_json::{json, Value};

use super::budget::{json_bytes, string_bytes};
use super::record::Request;

const COLUMN_BUDGET: usize = 4_110_000;
const HEAD_BYTES: usize = 4096;
const MARKER_ALLOWANCE: usize = 64;

fn string_leaves(value: &Value, path: &str, leaves: &mut Vec<(usize, String)>) {
    match value {
        Value::String(text) if text.len() > HEAD_BYTES => {
            leaves.push((text.len(), path.to_owned()));
        }
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                string_leaves(item, &format!("{path}/{index}"), leaves);
            }
        }
        Value::Object(items) => {
            for (key, item) in items {
                let key = key.replace('~', "~0").replace('/', "~1");
                string_leaves(item, &format!("{path}/{key}"), leaves);
            }
        }
        _ => {}
    }
}

/// Return both changed-string count and exact final size, reusing the caller's size walk.
fn trim_strings(value: &mut Value, maximum: usize, mut bytes: usize) -> (usize, usize) {
    let mut count = 0;
    if bytes <= maximum {
        return (0, bytes);
    }
    let mut leaves = Vec::new();
    string_leaves(value, "", &mut leaves);
    leaves.sort_by_key(|leaf| std::cmp::Reverse(leaf.0));
    for (_, path) in leaves {
        if bytes <= maximum {
            break;
        }
        let Some(Value::String(text)) = value.pointer_mut(&path) else {
            continue;
        };
        let original_bytes = string_bytes(text);
        // Match hosted capture's minimum-loss policy: keep as much of the
        // largest string as fits, reserving room for its explicit marker.
        // Equal-sized strings retain their original traversal order.
        let mut end = text
            .len()
            .saturating_sub(bytes - maximum)
            .saturating_sub(MARKER_ALLOWANCE)
            .max(HEAD_BYTES);
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        let removed = text.len() - end;
        text.truncate(end);
        text.push_str(&format!(" [truncated for capture: {removed} bytes]"));
        text.shrink_to_fit();
        // Only this string changed. Preserve exact escaped sizing without
        // rescanning every other message after each individual truncation.
        // A barely over-prefix string can grow when its marker is added.
        bytes = bytes - original_bytes + string_bytes(text);
        count += 1;
    }
    (count, bytes)
}

/// Alter only the freshly decoded capture tree, never the request being served.
pub(super) fn bound(request: &mut Request, maximum: usize) -> usize {
    let original_bytes = request.json_bytes();
    // Every component is smaller than the whole request. Ordinary requests need
    // one size walk, not separate column walks and repeated envelope walks.
    if original_bytes <= maximum.min(COLUMN_BUDGET) {
        return original_bytes;
    }
    if !request.context.get("request").is_some_and(Value::is_object) {
        return original_bytes;
    }
    let context = Arc::make_mut(&mut request.context);
    let Some(effective) = context.get_mut("request") else {
        return original_bytes;
    };
    let mut limits = serde_json::Map::new();
    if let Some(messages) = effective.get_mut("messages") {
        let original = json_bytes(messages);
        if original > COLUMN_BUDGET {
            let (strings, remaining) = trim_strings(messages, COLUMN_BUDGET, original);
            let mut dropped = 0;
            if remaining > COLUMN_BUDGET {
                dropped = messages.as_array().map_or(0, Vec::len);
                *messages = json!([{"role":"system", "truncated":true,
                    "content":format!("[{dropped} messages ({original} bytes) not captured: over the 4 MiB cap even with every string truncated]")}]);
            }
            limits.insert("messages_truncated".into(), Value::Bool(true));
            limits.insert("messages_bytes".into(), original.into());
            limits.insert("messages_truncated_strings".into(), strings.into());
            limits.insert("messages_dropped".into(), dropped.into());
        }
    }
    if let Some(tools) = effective.get_mut("tools") {
        if json_bytes(tools) > COLUMN_BUDGET {
            if let Some(tools) = tools.as_array_mut() {
                for tool in tools {
                    if let Some(parameters) = tool.get_mut("parameters") {
                        *parameters = json!({"truncated":true,"bytes":json_bytes(parameters)});
                    }
                }
            }
            limits.insert("tools_schemas_truncated".into(), Value::Bool(true));
        }
    }
    if !limits.is_empty() {
        // A lossless sidecar must not quietly restore an unbounded original.
        if context
            .as_object_mut()
            .unwrap()
            .remove("source_json")
            .is_some()
        {
            limits.insert("source_json_omitted".into(), Value::Bool(true));
        }
        context["capture_limits"] = Value::Object(limits);
    }
    let size = request.json_bytes();
    if size <= maximum {
        return size;
    }
    {
        let context = Arc::make_mut(&mut request.context);
        context.as_object_mut().unwrap().remove("source_json");
        let original = json_bytes(context);
        let (trimmed, _) = trim_strings(context, maximum.saturating_sub(4096), original);
        context["context_truncated_strings"] = trimmed.into();
    }
    let size = request.json_bytes();
    if size <= maximum {
        return size;
    }
    {
        // The previous hosted writer omitted an oversized envelope rather than
        // discard its prompt or reject serving. Preserve the same last resort
        // for structural tool schemas with no large strings to shorten.
        let context = Arc::make_mut(&mut request.context);
        let messages = context["request"]["messages"].take();
        let mut limits = context.get("capture_limits").cloned().unwrap_or(json!({}));
        limits["request_envelope_dropped"] = Value::Bool(true);
        *context = json!({"schema_version":1,"request":{"messages":messages},
            "capture_limits":limits});
    }
    let size = request.json_bytes();
    if size <= maximum {
        return size;
    }
    {
        // A configured request cap can be smaller than the column cap. Account
        // for the retained authority/envelope and truncation markers as well.
        let original = json_bytes(&request.context["request"]["messages"]);
        let overhead = size.saturating_sub(original);
        let message_budget = maximum.saturating_sub(overhead + 256);
        let context = Arc::make_mut(&mut request.context);
        let messages = &mut context["request"]["messages"];
        let (strings, remaining) = trim_strings(messages, message_budget, original);
        let mut dropped = 0;
        if remaining > message_budget {
            dropped = messages.as_array().map_or(0, Vec::len);
            *messages = json!([{"role":"system", "truncated":true,
                "content":format!("[{dropped} messages ({original} bytes) not captured: over capture limit]")}]);
        }
        let limits = &mut context["capture_limits"];
        limits["messages_truncated"] = Value::Bool(true);
        limits["messages_bytes"] = original.into();
        limits["messages_truncated_strings"] = strings.into();
        limits["messages_dropped"] = dropped.into();
    }
    request.json_bytes()
}

#[cfg(test)]
#[path = "bounds_test.rs"]
mod tests;
