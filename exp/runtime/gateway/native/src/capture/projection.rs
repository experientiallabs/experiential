//! Provider response evidence shared by passive capture and gateway destinations.
use super::record::{Protocol, Record, Response};
use serde_json::{json, Map, Value};
use std::borrow::Cow;
use std::collections::BTreeMap;

/// Observed output and completion evidence, independent of transport or destination.
pub(super) struct CapturedResponse<'a> {
    pub body: Cow<'a, Value>,
    pub completed: bool,
    pub projectable: bool,
}

impl<'a> CapturedResponse<'a> {
    /// Decode a bounded wire copy; callers own decompression and storage policy.
    pub fn decode(protocol: Protocol, bytes: &[u8], sse: bool) -> (String, bool, Option<String>) {
        if !sse {
            let source = std::str::from_utf8(bytes).unwrap_or("");
            return match serde_json::from_str::<Value>(source)
                .ok()
                .filter(Value::is_object)
            {
                Some(body) => (
                    source.to_owned(),
                    Self::json(protocol, Cow::Owned(body)).completed,
                    None,
                ),
                None => (
                    if bytes.is_empty() {
                        "{}"
                    } else {
                        "{\"capture_unparsed_response\":true}"
                    }
                    .to_owned(),
                    false,
                    None,
                ),
            };
        }
        let (frames, sources) = super::response::data_frames_with_sources(bytes);
        let captured = CapturedResponse::sse(protocol, &frames, true);
        let events_json = (matches!(protocol, Protocol::Responses)
            && (captured.body["capture_incomplete"] == true || !sources.is_empty()))
        .then(|| super::response::source_frames(&frames, &sources));
        (captured.body.to_string(), captured.completed, events_json)
    }

    pub fn json(protocol: Protocol, body: Cow<'a, Value>) -> Self {
        let projectable = terminal(protocol, &body) && !has_error(&body);
        let completed = match protocol {
            Protocol::Responses => body["status"] == "completed",
            Protocol::Messages => body["stop_reason"].as_str().is_some_and(|v| !v.is_empty()),
            Protocol::ChatCompletions => terminal(protocol, &body),
        };
        Self {
            body,
            completed,
            projectable,
        }
    }

    pub fn sse(protocol: Protocol, frames: &'a [Value], input_sources: bool) -> Self {
        let event = frames.iter().rev().find(|v| {
            matches!(
                v["type"].as_str(),
                Some("response.completed" | "response.incomplete" | "response.failed")
            )
        });
        let completed = match protocol {
            Protocol::Responses => event.is_some_and(|v| v["type"] == "response.completed"),
            Protocol::Messages => frames.iter().any(|v| v["type"] == "message_stop"),
            Protocol::ChatCompletions => frames.iter().any(|v| v.as_str() == Some("[DONE]")),
        };
        let mut valid = true;
        let body = match protocol {
            Protocol::Responses => event
                .and_then(|v| v.get("response"))
                .filter(|v| v.is_object())
                .map(Cow::Borrowed),
            Protocol::Messages => {
                let (body, lifecycle_complete) = super::messages::assemble(frames, input_sources);
                valid = lifecycle_complete;
                Some(Cow::Owned(body))
            }
            Protocol::ChatCompletions => {
                let (body, assembled) = assemble_chat(frames);
                valid = assembled
                    && frames.last().and_then(Value::as_str) == Some("[DONE]")
                    && frames
                        .first()
                        .is_some_and(|v| v.get("id").is_some() && v.get("model").is_some());
                Some(Cow::Owned(body))
            }
        };
        let mut body = body.unwrap_or_else(|| Cow::Owned(json!({"capture_incomplete": true})));
        if let Some(error) = frames.iter().find(|v| has_error(v)) {
            body.to_mut()["error"] = error
                .get("error")
                .filter(|v| !v.is_null())
                .cloned()
                .unwrap_or(json!("provider stream error"));
        }
        let projectable = valid
            && terminal(protocol, &body)
            && !frames.iter().any(has_error)
            && !has_error(&body);
        Self {
            body,
            completed,
            projectable,
        }
    }

    pub fn completed_record(record: &'a Record) -> Option<Cow<'a, Value>> {
        let protocol = record.request.protocol;
        let captured = match record.response.as_ref()? {
            Response::Json {
                status: 200..=299,
                body,
                source_json,
            } => Self::json(
                protocol,
                source_json
                    .as_ref()
                    .and_then(|s| serde_json::from_str(s).ok())
                    .map(Cow::Owned)
                    .unwrap_or(Cow::Borrowed(body)),
            ),
            Response::Sse {
                status: 200..=299,
                frames,
                truncated: false,
                client_disconnected: false,
                source_json,
            } => {
                if let Some(frames) = source_json
                    .as_ref()
                    .and_then(|s| serde_json::from_str::<Vec<Value>>(s).ok())
                {
                    let captured = CapturedResponse::sse(protocol, &frames, false);
                    return captured
                        .projectable
                        .then(|| Cow::Owned(captured.body.into_owned()));
                }
                Self::sse(protocol, frames, false)
            }
            _ => return None,
        };
        captured.projectable.then_some(captured.body)
    }
}

fn has_error(value: &Value) -> bool {
    value.get("error").is_some_and(|v| !v.is_null())
        || matches!(
            value.get("type").and_then(Value::as_str),
            Some("error" | "response.failed")
        )
}

fn terminal(protocol: Protocol, body: &Value) -> bool {
    match protocol {
        Protocol::Responses => matches!(body["status"].as_str(), Some("completed" | "incomplete")),
        Protocol::Messages => {
            body["type"] == "message"
                && body["content"].is_array()
                && body["stop_reason"].is_string()
        }
        Protocol::ChatCompletions => body["choices"].as_array().is_some_and(|choices| {
            !choices.is_empty()
                && choices
                    .iter()
                    .all(|v| v["finish_reason"].as_str().is_some_and(|v| !v.is_empty()))
        }),
    }
}

fn assemble_chat(chunks: &[Value]) -> (Value, bool) {
    let first = chunks.first().unwrap_or(&Value::Null);
    let mut result = json!({"id": first.get("id"), "object": "chat.completion",
        "model": first.get("model"), "created": first.get("created").unwrap_or(&Value::Null)});
    let mut choices: BTreeMap<u64, Value> = BTreeMap::new();
    let mut tools: BTreeMap<u64, BTreeMap<u64, Value>> = BTreeMap::new();
    let mut valid = true;
    for chunk in chunks {
        valid &= chunk.as_str() == Some("[DONE]") || chunk["choices"].is_array();
        for key in ["id", "model", "error"] {
            if let Some(value) = chunk.get(key) {
                result[key] = value.clone();
            }
        }
        if let Some(usage) = chunk.get("usage").filter(|value| !value.is_null()) {
            result["usage"] = usage.clone();
        }
        for choice in chunk
            .get("choices")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let assembled = (|| -> Option<()> {
                let index = choice.get("index")?.as_u64()?;
                let target = choices.entry(index).or_insert_with(|| {
                    json!({"index": index, "message": {"role": "assistant"}, "finish_reason": null})
                });
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
                Some(())
            })();
            valid &= assembled.is_some();
        }
    }
    for (index, calls) in tools {
        choices.get_mut(&index).unwrap()["message"]["tool_calls"] =
            Value::Array(calls.into_values().collect());
    }
    result["choices"] = Value::Array(choices.into_values().collect());
    (result, valid)
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
        assert!(CapturedResponse::completed_record(&value).is_some());
        let Some(Response::Json { body, .. }) = &mut value.response else {
            panic!()
        };
        body["error"] = json!({"message":"failed"});
        assert!(!CapturedResponse::completed_record(&value).is_some());
    }

    #[test]
    fn terminal_frames_do_not_override_capture_loss_or_disconnect() {
        for (truncated, disconnected) in [(false, false), (true, false), (false, true)] {
            let value = record(json!({"kind":"sse","status":200,"frames":[
                {"type":"response.completed","response":{"id":"response","status":"completed"}}
            ],"truncated":truncated,"client_disconnected":disconnected}));
            assert_eq!(
                CapturedResponse::completed_record(&value).is_some(),
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
        assert!(!terminal(
            Protocol::ChatCompletions,
            &assemble_chat(std::slice::from_ref(&first)).0
        ));
        let (result, valid) = assemble_chat(&[first, last, json!({"choices":[{}]})]);
        assert!(!valid);
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
            let result = assemble_chat(&[first, last]).0;
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
        let result = assemble_chat(&chunks).0;
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
