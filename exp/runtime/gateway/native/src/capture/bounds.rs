//! The hosted capture copy has always been bounded independently of serving.

use std::sync::Arc;

use serde_json::{json, Value};

use super::budget::json_bytes;
use super::record::Request;

const COLUMN_BUDGET: usize = 4_110_000;
const HEAD_BYTES: usize = 4096;

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

fn trim_strings(value: &mut Value, maximum: usize) -> usize {
    let mut count = 0;
    if json_bytes(value) <= maximum {
        return 0;
    }
    let mut leaves = Vec::new();
    string_leaves(value, "", &mut leaves);
    leaves.sort_unstable_by_key(|leaf| std::cmp::Reverse(leaf.0));
    for (_, path) in leaves {
        if json_bytes(value) <= maximum {
            break;
        }
        let Some(Value::String(text)) = value.pointer_mut(&path) else {
            continue;
        };
        let mut end = HEAD_BYTES;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        let removed = text.len() - end;
        text.truncate(end);
        text.push_str(&format!(" [truncated for capture: {removed} bytes]"));
        text.shrink_to_fit();
        count += 1;
    }
    count
}

/// Alter only the freshly decoded capture tree, never the request being served.
pub(super) fn bound(request: &mut Request, maximum: usize) {
    if !request.context.get("request").is_some_and(Value::is_object) {
        return;
    }
    let context = Arc::make_mut(&mut request.context);
    let Some(effective) = context.get_mut("request") else {
        return;
    };
    let mut limits = serde_json::Map::new();
    if let Some(messages) = effective.get_mut("messages") {
        let original = json_bytes(messages);
        if original > COLUMN_BUDGET {
            let strings = trim_strings(messages, COLUMN_BUDGET);
            let mut dropped = 0;
            if json_bytes(messages) > COLUMN_BUDGET {
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
    if request.json_bytes() > maximum {
        let context = Arc::make_mut(&mut request.context);
        context.as_object_mut().unwrap().remove("source_json");
        let trimmed = trim_strings(context, maximum.saturating_sub(4096));
        context["context_truncated_strings"] = trimmed.into();
    }
}
