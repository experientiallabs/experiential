//! Completed public protocol projection for the local evidence destination.
use super::record::{Protocol, Record, Response};
use serde_json::{json, Map, Value};
use std::borrow::Cow;
use std::collections::BTreeMap;

pub(super) fn completed_response(record: &Record) -> Option<Cow<'_, Value>> {
    let protocol = &record.request.protocol;
    match record.response.as_ref()? {
        Response::Json {
            status: 200..=299,
            body,
            source_json,
        } => {
            let body = source_json
                .as_ref()
                .and_then(|source| serde_json::from_str::<Value>(source).ok())
                .map(Cow::Owned)
                .unwrap_or(Cow::Borrowed(body));
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
            } else if matches!(protocol, Protocol::Messages) {
                if body.get("type").and_then(Value::as_str) != Some("message")
                    || !body.get("content").is_some_and(Value::is_array)
                    || !body.get("stop_reason").is_some_and(Value::is_string)
                {
                    return None;
                }
            } else if !matches!(
                body.get("status").and_then(Value::as_str),
                Some("completed" | "incomplete")
            ) {
                return None;
            }
            Some(body)
        }
        Response::Sse {
            status: 200..=299,
            frames,
            truncated: false,
            client_disconnected: false,
            source_json,
        } => {
            let frames = source_json
                .as_ref()
                .and_then(|source| serde_json::from_str::<Vec<Value>>(source).ok())
                .map(Cow::Owned)
                .unwrap_or(Cow::Borrowed(frames.as_slice()));
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
                match frames {
                    Cow::Borrowed(frames) => terminal_response(frames).map(Cow::Borrowed),
                    Cow::Owned(frames) => terminal_response(&frames).cloned().map(Cow::Owned),
                }
            } else if matches!(protocol, Protocol::Messages) {
                super::messages::assemble(&frames, true).map(Cow::Owned)
            } else {
                if frames.last().and_then(Value::as_str) != Some("[DONE]") {
                    return None;
                }
                assemble_chat(&frames[..frames.len() - 1], true).map(Cow::Owned)
            }
        }
        _ => None,
    }
}

fn terminal_response(frames: &[Value]) -> Option<&Value> {
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
}
pub(super) fn assemble_chat(chunks: &[Value], complete_only: bool) -> Option<Value> {
    let first = chunks.first().unwrap_or(&Value::Null);
    if complete_only && (first.get("id").is_none() || first.get("model").is_none()) {
        return None;
    }
    let mut result = json!({"id": first["id"], "model": first["model"],
        "object": "chat.completion", "created": first["created"]});
    let mut choices: BTreeMap<u64, Value> = BTreeMap::new();
    let mut tools: BTreeMap<u64, BTreeMap<u64, Value>> = BTreeMap::new();
    for chunk in chunks {
        for key in ["id", "model", "created", "usage", "error"] {
            if let Some(value) = chunk.get(key).filter(|value| {
                !value.is_null() && (!complete_only || key == "usage" || key == "error")
            }) {
                result[key] = value.clone();
            }
        }
        let raw_choices = chunk.get("choices").and_then(Value::as_array);
        if complete_only && raw_choices.is_none() {
            return None;
        }
        for choice in raw_choices.into_iter().flatten() {
            let Some(index) = choice.get("index").and_then(Value::as_u64) else {
                if complete_only {
                    return None;
                }
                continue;
            };
            let target = choices.entry(index).or_insert_with(
                || json!({"index": index, "message": {"role": "assistant"}, "finish_reason": null}),
            );
            let message = target.get_mut("message")?.as_object_mut()?;
            if let Some(delta) = choice.get("delta").and_then(Value::as_object) {
                for (key, value) in delta {
                    if key == "tool_calls" {
                        merge_tools(tools.entry(index).or_default(), value)?;
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
    for (index, calls) in tools {
        choices.get_mut(&index)?["message"]["tool_calls"] =
            Value::Array(calls.into_values().collect());
    }
    if complete_only
        && (choices.is_empty()
            || choices
                .values()
                .any(|choice| choice["finish_reason"].is_null()))
    {
        return None;
    }
    result["choices"] = Value::Array(choices.into_values().collect());
    Some(result)
}

pub(super) fn append(object: &mut Map<String, Value>, key: &str, addition: &Value) -> Option<()> {
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

fn merge_tools(tools: &mut BTreeMap<u64, Value>, addition: &Value) -> Option<()> {
    for delta in addition.as_array()? {
        let index = delta.get("index")?.as_u64()?;
        // Output indexes may include omitted reasoning/text items. Allocate only
        // observed calls, never a vector sized by an untrusted provider index.
        let target = tools
            .entry(index)
            .or_insert_with(|| json!({"type": "function", "function": {}}))
            .as_object_mut()?;
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
            }, "response":response, "deployment_id":null,"captured_at":1.0,
            "metrics":null,"gemini_thought_parts":[],"gemini_thought_parts_source_json":null
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
            "finish_reason":"tool_calls"}], "usage":{"prompt_tokens":5,"completion_tokens":3}});
        assert!(assemble_chat(std::slice::from_ref(&first), true).is_none());
        assert!(assemble_chat(std::slice::from_ref(&first), false).is_some());
        let result = assemble_chat(&[first, last], true).unwrap();
        assert_eq!(
            result["usage"],
            json!({"prompt_tokens":5,"completion_tokens":3})
        );
        assert_eq!(
            result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{}"
        );
    }

    #[test]
    fn tool_indexes_are_sparse_labels_not_allocation_sizes() {
        for index in [1_u64, 8, u64::MAX] {
            let first = json!({"id":"completion","model":"model","choices":[{"index":0,
                "delta":{"tool_calls":[{"index":index,"id":"call","type":"function",
                    "function":{"name":"lookup","arguments":"{ \"id\" :"}}]},"finish_reason":null}]});
            let last = json!({"id":"completion","model":"model","choices":[{"index":0,
                "delta":{"tool_calls":[{"index":index,"function":{"arguments":" \"雪\" }"}}]},
                "finish_reason":"tool_calls"}]});
            let result = assemble_chat(&[first, last], true).unwrap();
            assert_eq!(
                result["choices"][0]["message"]["tool_calls"],
                json!([
                    {"id":"call","type":"function","function":{"name":"lookup","arguments":"{ \"id\" : \"雪\" }"}}
                ])
            );
        }
    }

    #[test]
    fn interleaved_sparse_tools_are_grouped_by_choice_then_tool_index() {
        let chunks = vec![
            json!({"id":"completion","model":"model","choices":[{"index":0,"delta":{"tool_calls":[
                {"index":9,"id":"second","function":{"name":"b","arguments":"["}},
                {"index":3,"id":"first","function":{"name":"a","arguments":"{"}}
            ]}}]}),
            json!({"choices":[{"index":1,"delta":{"tool_calls":[
                {"index":3,"id":"other-choice","function":{"name":"c","arguments":"{}"}}
            ]},"finish_reason":"tool_calls"}]}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[
                {"index":3,"function":{"arguments":"}"}},
                {"index":9,"function":{"arguments":"]"}}
            ]},"finish_reason":"tool_calls"}]}),
        ];
        let result = assemble_chat(&chunks, true).unwrap();
        let tools = &result["choices"][0]["message"]["tool_calls"];
        assert_eq!(tools[0]["id"], "first");
        assert_eq!(tools[0]["function"]["arguments"], "{}");
        assert_eq!(tools[1]["id"], "second");
        assert_eq!(tools[1]["function"]["arguments"], "[]");
        assert_eq!(
            result["choices"][1]["message"]["tool_calls"][0]["id"],
            "other-choice"
        );
    }
}
