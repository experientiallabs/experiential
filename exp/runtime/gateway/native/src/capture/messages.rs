//! Public Messages assembly shared by complete gateway and partial passive evidence.

use super::projection::append;
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Default)]
struct Block {
    value: Value,
    arguments: Option<String>,
    closed: bool,
}

pub(super) fn assemble(frames: &[Value], complete_only: bool) -> Option<Value> {
    let mut message = json!({"usage": {}});
    let mut started = false;
    let mut stopped = false;
    let mut blocks: BTreeMap<u64, Block> = BTreeMap::new();
    for frame in frames {
        if complete_only && stopped {
            return None;
        }
        match frame.get("type")?.as_str()? {
            "ping" => {}
            "message_start" => {
                let start = frame.get("message")?.as_object()?;
                if complete_only
                    && (started
                        || start.get("type")?.as_str()? != "message"
                        || !start.get("content")?.as_array()?.is_empty())
                {
                    return None;
                }
                if complete_only {
                    message = Value::Object(start.clone());
                } else {
                    message.as_object_mut()?.extend(start.clone());
                }
                started = true;
            }
            "content_block_start" => {
                let index = frame.get("index")?.as_u64()?;
                if complete_only
                    && (!started || blocks.contains_key(&index) || index != blocks.len() as u64)
                {
                    return None;
                }
                blocks.insert(
                    index,
                    Block {
                        value: Value::Object(frame.get("content_block")?.as_object()?.clone()),
                        ..Block::default()
                    },
                );
            }
            "content_block_delta" => {
                let index = frame.get("index")?.as_u64()?;
                if complete_only && !blocks.contains_key(&index) {
                    return None;
                }
                let block = blocks.entry(index).or_insert_with(|| Block {
                    value: json!({}),
                    ..Block::default()
                });
                if complete_only && block.closed {
                    return None;
                }
                let delta = frame.get("delta")?;
                let fields: &[&str] = if complete_only {
                    match delta.get("type")?.as_str()? {
                        "text_delta" => &["text"],
                        "thinking_delta" => &["thinking"],
                        "signature_delta" => &["signature"],
                        "input_json_delta" => &["partial_json"],
                        "citations_delta" => &["citation"],
                        _ => return None,
                    }
                } else {
                    &["text", "thinking", "signature", "partial_json", "citation"]
                };
                for key in fields {
                    let Some(value) = delta.get(*key) else {
                        if complete_only {
                            return None;
                        }
                        continue;
                    };
                    match *key {
                        "partial_json" => block
                            .arguments
                            .get_or_insert_with(String::new)
                            .push_str(value.as_str()?),
                        "citation" => {
                            append(block.value.as_object_mut()?, "citations", &json!([value]))?
                        }
                        _ => {
                            value.as_str()?;
                            append(block.value.as_object_mut()?, key, value)?;
                        }
                    }
                }
            }
            "content_block_stop" => {
                let block = blocks.get_mut(&frame.get("index")?.as_u64()?)?;
                if complete_only && block.closed {
                    return None;
                }
                block.closed = true;
            }
            "message_delta" => {
                if complete_only && (!started || !frame["delta"].is_object()) {
                    return None;
                }
                if let Some(delta) = frame.get("delta").and_then(Value::as_object) {
                    message.as_object_mut()?.extend(delta.clone());
                }
                if let Some(usage) = frame.get("usage") {
                    message["usage"]
                        .as_object_mut()?
                        .extend(usage.as_object()?.clone());
                }
            }
            "message_stop" => stopped = true,
            "error" if !complete_only => message["error"] = frame["error"].clone(),
            _ if !complete_only => {}
            _ => return None,
        }
    }
    if complete_only
        && (!stopped
            || !message["stop_reason"].is_string()
            || blocks.values().any(|block| !block.closed))
    {
        return None;
    }
    for block in blocks.values_mut() {
        if let Some(arguments) = &block.arguments {
            match serde_json::from_str::<Value>(arguments) {
                Ok(value) if !complete_only || value.is_object() => block.value["input"] = value,
                _ if complete_only && arguments.is_empty() => {}
                _ if complete_only => return None,
                _ => block.value["capture_partial_input"] = json!(arguments),
            }
        }
    }
    message["content"] = Value::Array(blocks.into_values().map(|block| block.value).collect());
    Some(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn preserves_thinking_signatures_tool_inputs_and_requires_complete_lifecycle() {
        let frames = vec![
            json!({"type":"message_start","message":{"type":"message","id":"msg","role":"assistant","content":[],"usage":{"input_tokens":2},"stop_reason":null}}),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" exactly\0雪\n"}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed"}}),
            json!({"type":"content_block_stop","index":0}),
            json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call","name":"lookup","input":{}}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"x\":"}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"1}"}}),
            json!({"type":"content_block_stop","index":1}),
            json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}),
            json!({"type":"message_stop"}),
        ];
        let result = assemble(&frames, true).unwrap();
        assert_eq!(result["content"][0]["thinking"], " exactly\0雪\n");
        assert_eq!(result["content"][0]["signature"], "signed");
        assert_eq!(result["content"][1]["input"], json!({"x":1}));
        assert!(assemble(&frames[..frames.len() - 1], true).is_none());
        assert_eq!(assemble(&frames[..frames.len() - 1], false), Some(result));
    }

    #[test]
    fn empty_argument_delta_keeps_the_declared_zero_argument_input() {
        let frames = vec![
            json!({"type":"message_start","message":{"type":"message","id":"msg","role":"assistant","content":[],"usage":{},"stop_reason":null}}),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call","name":"lookup","input":{}}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}),
            json!({"type":"content_block_stop","index":0}),
            json!({"type":"message_delta","delta":{"stop_reason":"tool_use"}}),
            json!({"type":"message_stop"}),
        ];
        let message = assemble(&frames, true).unwrap();
        assert_eq!(message["content"][0]["input"], json!({}));
    }
}
