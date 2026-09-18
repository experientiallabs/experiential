//! Completed public protocol projection for the local evidence destination.
use super::record::{Protocol, Record, Response};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

pub(super) fn completed_response(record: &Record) -> Option<Value> {
    let protocol = &record.request.protocol;
    if !matches!(protocol, Protocol::ChatCompletions | Protocol::Responses) {
        return None;
    }
    match record.response.as_ref()? {
        Response::Json {
            status: 200..=299,
            body,
        } => {
            if body.get("error").is_some_and(|value| !value.is_null()) {
                return None;
            }
            if matches!(protocol, Protocol::ChatCompletions) {
                let choices = body.get("choices")?.as_array()?;
                if choices.is_empty()
                    || choices.iter().any(|value| value["finish_reason"].is_null())
                {
                    return None;
                }
            } else if !matches!(
                body.get("status").and_then(Value::as_str),
                Some("completed" | "incomplete")
            ) {
                return None;
            }
            Some(body.clone())
        }
        Response::Sse {
            status: 200..=299,
            frames,
            truncated: false,
            client_disconnected: false,
        } => {
            if frames.iter().any(|value| {
                value.get("error").is_some_and(|value| !value.is_null())
                    || matches!(
                        value.get("type").and_then(Value::as_str),
                        Some("response.failed" | "error")
                    )
            }) {
                return None;
            }
            if matches!(protocol, Protocol::Responses) {
                frames
                    .iter()
                    .rev()
                    .find(|value| {
                        matches!(
                            value.get("type").and_then(Value::as_str),
                            Some("response.completed" | "response.incomplete")
                        )
                    })?
                    .get("response")
                    .filter(|value| value.is_object())
                    .cloned()
            } else {
                if frames.last().and_then(Value::as_str) != Some("[DONE]") {
                    return None;
                }
                assemble_chat(frames[..frames.len() - 1].to_vec())
            }
        }
        _ => None,
    }
}
fn assemble_chat(chunks: Vec<Value>) -> Option<Value> {
    let first = chunks.first()?;
    let mut result = json!({"id": first.get("id")?, "object": "chat.completion",
        "model": first.get("model")?, "created": first.get("created").unwrap_or(&Value::Null)});
    let mut choices: BTreeMap<u64, Value> = BTreeMap::new();
    for chunk in &chunks {
        if let Some(usage) = chunk.get("usage").filter(|value| !value.is_null()) {
            result["usage"] = usage.clone();
        }
        for choice in chunk.get("choices")?.as_array()? {
            let index = choice.get("index")?.as_u64()?;
            let target = choices.entry(index).or_insert_with(
                || json!({"index": index, "message": {"role": "assistant"}, "finish_reason": null}),
            );
            let message = target.get_mut("message")?.as_object_mut()?;
            if let Some(delta) = choice.get("delta").and_then(Value::as_object) {
                for (key, value) in delta {
                    if key == "tool_calls" {
                        merge_tools(message, value)?;
                    } else if key == "role" {
                        message.insert(key.clone(), value.clone());
                    } else {
                        append(message, key, value)?;
                    }
                }
            }
            if let Some(reason) = choice.get("finish_reason").filter(|value| !value.is_null()) {
                target["finish_reason"] = reason.clone();
            }
        }
    }
    if choices.is_empty()
        || choices
            .values()
            .any(|choice| choice["finish_reason"].is_null())
    {
        return None;
    }
    result["choices"] = Value::Array(choices.into_values().collect());
    Some(result)
}

fn append(object: &mut Map<String, Value>, key: &str, addition: &Value) -> Option<()> {
    if addition.is_null() {
        return Some(());
    }
    match object.get_mut(key) {
        None => {
            object.insert(key.to_owned(), addition.clone());
        }
        Some(Value::String(text)) => text.push_str(addition.as_str()?),
        Some(Value::Array(values)) => values.extend(addition.as_array()?.iter().cloned()),
        Some(value) if value == addition => {}
        _ => return None,
    }
    Some(())
}

fn merge_tools(message: &mut Map<String, Value>, addition: &Value) -> Option<()> {
    let tools = message
        .entry("tool_calls")
        .or_insert_with(|| json!([]))
        .as_array_mut()?;
    for delta in addition.as_array()? {
        let index = delta.get("index")?.as_u64()? as usize;
        // Provider indexes cannot allocate an unbounded sparse vector.
        if index > tools.len() {
            return None;
        }
        if index == tools.len() {
            tools.push(json!({"type": "function", "function": {}}));
        }
        let target = tools[index].as_object_mut()?;
        for (key, value) in delta.as_object()? {
            match key.as_str() {
                "index" => {}
                "function" => {
                    let function = target.get_mut("function")?.as_object_mut()?;
                    for (key, value) in value.as_object()? {
                        append(function, key, value)?;
                    }
                }
                "type" | "id" => {
                    target.insert(key.clone(), value.clone());
                }
                _ => {
                    append(target, key, value)?;
                }
            }
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(response: Value) -> Record {
        serde_json::from_value(json!({
            "schema_version": 1, "request": {
                "request_id": "request", "scope": {"organization_id":"org",
                    "identity_id":"identity", "application_id":"app"},
                "protocol":"responses", "model_id":"model",
                "context":{"schema_version":1,"request":{}}
            }, "response":response, "deployment_id":null,"captured_at":1.0
        }))
        .unwrap()
    }

    #[test]
    fn response_null_error_is_not_a_failure_and_explicit_error_is_rejected() {
        let mut value = record(json!({"kind":"json","status":200,
            "body":{"id":"response","status":"completed","error":null,"output":[]}}));
        assert!(completed_response(&value).is_some());
        let Some(Response::Json { body, .. }) = &mut value.response else {
            panic!()
        };
        body["error"] = json!({"message":"failed"});
        assert!(completed_response(&value).is_none());
    }

    #[test]
    fn terminal_frames_do_not_override_capture_loss_or_disconnect() {
        for (truncated, disconnected) in [(false, false), (true, false), (false, true)] {
            let value = record(json!({"kind":"sse","status":200,"frames":[
                {"type":"response.completed","response":{"id":"response","status":"completed"}}
            ],"truncated":truncated,"client_disconnected":disconnected}));
            assert_eq!(
                completed_response(&value).is_some(),
                !truncated && !disconnected
            );
        }
    }

    #[test]
    fn completed_chat_reassembles_tool_arguments_without_inventing_evidence() {
        let first = json!({"id":"completion","model":"model","choices":[{"index":0,
            "delta":{"tool_calls":[{"index":0,"id":"call","type":"function",
                "function":{"name":"lookup","arguments":"{"}}]},"finish_reason":null}]});
        let last = json!({"id":"completion","model":"model","choices":[{"index":0,
            "delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]},
            "finish_reason":"tool_calls"}]});
        assert!(assemble_chat(vec![first.clone()]).is_none());
        let result = assemble_chat(vec![first, last]).unwrap();
        assert_eq!(
            result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{}"
        );
    }
}
