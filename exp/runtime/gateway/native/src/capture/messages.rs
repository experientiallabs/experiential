//! Reassemble observed Messages content while tracking complete block lifecycles.
use super::projection::append;
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub(super) fn assemble(frames: &[Value], input_sources: bool) -> (Value, bool) {
    let mut message = json!({"usage": {}});
    let mut blocks = BTreeMap::new();
    let mut arguments: BTreeMap<u64, String> = BTreeMap::new();
    let mut closed = std::collections::BTreeSet::new();
    let (mut started, mut stopped, mut valid) = (false, false, true);
    for frame in frames {
        valid &= !stopped;
        let step = (|| -> Option<()> {
            match frame.get("type")?.as_str()? {
                "ping" => {}
                "message_start" => {
                    valid &= !started;
                    let start = frame.get("message")?.as_object()?;
                    valid &= start.get("type").is_some_and(|v| v == "message")
                        && start
                            .get("content")
                            .and_then(Value::as_array)
                            .is_some_and(Vec::is_empty);
                    message.as_object_mut()?.extend(start.clone());
                    started = true;
                }
                "content_block_start" => {
                    let index = frame.get("index")?.as_u64()?;
                    valid &=
                        started && index == blocks.len() as u64 && !blocks.contains_key(&index);
                    blocks.insert(
                        index,
                        Value::Object(frame.get("content_block")?.as_object()?.clone()),
                    );
                }
                "content_block_delta" => {
                    let index = frame.get("index")?.as_u64()?;
                    valid &= blocks.contains_key(&index) && !closed.contains(&index);
                    let block = blocks
                        .entry(index)
                        .or_insert_with(|| json!({}))
                        .as_object_mut()?;
                    let delta = frame.get("delta")?;
                    valid &= match delta["type"].as_str() {
                        Some("text_delta") => delta["text"].is_string(),
                        Some("thinking_delta") => delta["thinking"].is_string(),
                        Some("signature_delta") => delta["signature"].is_string(),
                        Some("input_json_delta") => delta["partial_json"].is_string(),
                        Some("citations_delta") => delta.get("citation").is_some(),
                        _ => false,
                    };
                    for key in ["text", "thinking", "signature"] {
                        if let Some(value) = delta.get(key) {
                            append(block, key, value)?;
                        }
                    }
                    if let Some(partial) = delta.get("partial_json") {
                        arguments
                            .entry(index)
                            .or_default()
                            .push_str(partial.as_str()?);
                    }
                    if let Some(citation) = delta.get("citation") {
                        append(block, "citations", &json!([citation]))?;
                    }
                }
                "content_block_stop" => {
                    let index = frame.get("index")?.as_u64()?;
                    valid &= blocks.contains_key(&index) && closed.insert(index);
                }
                "message_delta" => {
                    valid &= started && frame["delta"].is_object();
                    if let Some(delta) = frame.get("delta") {
                        message.as_object_mut()?.extend(delta.as_object()?.clone());
                    }
                    if let Some(usage) = frame.get("usage") {
                        message["usage"]
                            .as_object_mut()?
                            .extend(usage.as_object()?.clone());
                    }
                }
                "message_stop" => stopped = true,
                "error" => {
                    message["error"] = frame
                        .get("error")
                        .cloned()
                        .unwrap_or(json!("provider stream error"));
                    valid = false;
                }
                _ => return None,
            }
            Some(())
        })();
        valid &= step.is_some();
    }
    for (index, argument) in arguments {
        let block = blocks.get_mut(&index).unwrap();
        if !argument.is_empty() {
            match serde_json::from_str::<Value>(&argument) {
                Ok(value) => {
                    valid &= value.is_object();
                    // Desktop decodes exact inputs before redaction; gateway retains raw frames separately.
                    if input_sources && super::response::contains_wide_number(&value) {
                        block["capture_input_source_json"] = Value::String(argument);
                    }
                    block["input"] = value;
                }
                Err(_) => {
                    valid = false;
                    block["capture_partial_input"] = Value::String(argument);
                }
            }
        }
    }
    valid &= started && stopped && blocks.keys().all(|index| closed.contains(index));
    message["content"] = Value::Array(blocks.into_values().collect());
    (message, valid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn preserves_thinking_signatures_tool_inputs_and_requires_complete_lifecycle() {
        let mut frames = vec![
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
        let (result, complete) = assemble(&frames, false);
        assert!(complete);
        assert_eq!(result["content"][0]["thinking"], " exactly\0雪\n");
        assert_eq!(result["content"][0]["signature"], "signed");
        assert_eq!(result["content"][1]["input"], json!({"x":1}));
        assert!(!assemble(&frames[..frames.len() - 1], false).1);
        frames[7]["delta"]["partial_json"] = json!("18446744073709551616}");
        for input_sources in [false, true] {
            let (body, complete) = assemble(&frames, input_sources);
            assert!(complete);
            assert_eq!(
                body["content"][1]
                    .get("capture_input_source_json")
                    .is_some(),
                input_sources
            );
        }
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
        let message = assemble(&frames, false).0;
        assert_eq!(message["content"][0]["input"], json!({}));
    }
}
