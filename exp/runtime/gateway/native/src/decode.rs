//! Chat Completions decoding into the canonical gateway request shape.
//!
//! Mirrors `exp.runtime.openai_protocol.requests.decode_chat`: a closed
//! top-level field manifest, a strict wire profile, and one canonical
//! `GatewayRequest`-shaped JSON object that the Python control plane
//! re-validates during admission (so persisted digests stay Python-owned).

use serde_json::{json, Map, Value};

use crate::errors::PublicError;

/// Top-level Chat fields the gateway accepts, from `CHAT_MANIFEST` (supported,
/// conditionally supported, and metadata-only dispositions).
const CHAT_ALLOWED_FIELDS: &[&str] = &[
    "model",
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "temperature",
    "top_p",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "metadata",
];

/// One decoded request: the public alias plus its canonical request JSON.
#[derive(Debug, Clone)]
pub struct DecodedChat {
    pub alias: String,
    /// Canonical `GatewayRequest`-shaped JSON object.
    pub canonical: Value,
    pub stream: bool,
    pub include_usage: bool,
    pub client_request_id: Option<String>,
}

/// Validate the two optional caller-operation headers, mirroring
/// `_caller_operation`.
pub fn caller_operation(
    idempotency_key: Option<&str>,
    client_request_id: Option<&str>,
) -> Result<Option<String>, PublicError> {
    for (name, value) in [
        ("Idempotency-Key", idempotency_key),
        ("X-Client-Request-Id", client_request_id),
    ] {
        if let Some(value) = value {
            if value.is_empty() || value.len() > 512 || value.chars().any(|c| (c as u32) < 32) {
                return Err(PublicError::invalid_field_message(
                    name,
                    &format!("{name} must be a non-empty display-safe value."),
                ));
            }
        }
    }
    if let (Some(idempotency), Some(client)) = (idempotency_key, client_request_id) {
        if idempotency != client {
            return Err(PublicError::new(
                400,
                "idempotency_conflict",
                "Idempotency-Key and X-Client-Request-Id must identify the same operation.",
                "invalid_request_error",
            )
            .with_param("Idempotency-Key"));
        }
    }
    Ok(idempotency_key.or(client_request_id).map(str::to_string))
}

/// Decode one Chat Completions body without silently dropping fields.
pub fn decode_chat(
    payload: &Map<String, Value>,
    idempotency_key: Option<&str>,
    client_request_id: Option<&str>,
) -> Result<DecodedChat, PublicError> {
    for field in payload.keys() {
        if !CHAT_ALLOWED_FIELDS.contains(&field.as_str()) {
            return Err(PublicError::unsupported_field(field));
        }
    }
    let operation = caller_operation(idempotency_key, client_request_id)?;

    let alias = required_string(payload, "model", 256)?;
    let stream = optional_bool(payload, "stream")?.unwrap_or(false);

    let max_tokens = optional_positive_int(payload, "max_tokens")?;
    let max_completion_tokens = optional_positive_int(payload, "max_completion_tokens")?;
    if max_tokens.is_some() && max_completion_tokens.is_some() {
        return Err(PublicError::invalid_field("body"));
    }
    let maximum_output_tokens = max_completion_tokens.or(max_tokens);

    let include_usage = match payload.get("stream_options") {
        None | Some(Value::Null) => false,
        Some(Value::Object(options)) => {
            for key in options.keys() {
                if key != "include_usage" {
                    return Err(PublicError::invalid_field(&format!("stream_options.{key}")));
                }
            }
            let include = match options.get("include_usage") {
                None => false,
                Some(Value::Bool(flag)) => *flag,
                Some(_) => {
                    return Err(PublicError::invalid_field("stream_options.include_usage"))
                }
            };
            if !stream {
                return Err(PublicError::invalid_field("body"));
            }
            include
        }
        Some(_) => return Err(PublicError::invalid_field("stream_options")),
    };

    let stop = decode_stop(payload)?;
    let temperature = optional_bounded_number(payload, "temperature", 0.0, 2.0)?;
    let top_p = optional_bounded_number(payload, "top_p", 0.0, 1.0)?;
    let messages = decode_messages(payload)?;
    let tools = decode_tools(payload)?;
    let tool_choice = decode_tool_choice(payload, &tools)?;
    let parallel_tool_calls = optional_bool(payload, "parallel_tool_calls")?;
    let structured_text = decode_response_format(payload)?;
    let metadata = match payload.get("metadata") {
        None | Some(Value::Null) => Value::Object(Map::new()),
        Some(Value::Object(map)) => Value::Object(map.clone()),
        Some(_) => return Err(PublicError::invalid_field("metadata")),
    };

    if include_usage && !stream {
        return Err(PublicError::invalid_field("body"));
    }

    let canonical = json!({
        "surface": "chat_completions",
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
        "structured_text": structured_text,
        "maximum_output_tokens": maximum_output_tokens,
        "stop": stop,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
        "include_usage": include_usage,
        "previous_response_id": Value::Null,
        "metadata": metadata,
        "idempotency_key": if idempotency_key.is_some() { operation.clone() } else { None },
        "client_request_id": if client_request_id.is_some() { operation.clone() } else { None },
    });

    Ok(DecodedChat {
        alias,
        canonical,
        stream,
        include_usage,
        client_request_id: if client_request_id.is_some() {
            operation
        } else {
            None
        },
    })
}

fn decode_stop(payload: &Map<String, Value>) -> Result<Vec<String>, PublicError> {
    let stop: Vec<String> = match payload.get("stop") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(single)) => vec![single.clone()],
        Some(Value::Array(items)) => {
            let mut collected = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::String(text) => collected.push(text.clone()),
                    _ => return Err(PublicError::invalid_field("stop")),
                }
            }
            collected
        }
        Some(_) => return Err(PublicError::invalid_field("stop")),
    };
    if stop.iter().any(String::is_empty) {
        return Err(PublicError::invalid_field("stop"));
    }
    let mut seen = std::collections::HashSet::new();
    if !stop.iter().all(|item| seen.insert(item.clone())) {
        return Err(PublicError::invalid_field("stop"));
    }
    Ok(stop)
}

fn decode_messages(payload: &Map<String, Value>) -> Result<Vec<Value>, PublicError> {
    let items = match payload.get("messages") {
        Some(Value::Array(items)) if !items.is_empty() => items,
        _ => return Err(PublicError::invalid_field("messages")),
    };
    let mut messages = Vec::with_capacity(items.len());
    for (message_index, item) in items.iter().enumerate() {
        let object = item
            .as_object()
            .ok_or_else(|| PublicError::invalid_field(&format!("messages.{message_index}")))?;
        messages.push(decode_message(object, message_index)?);
    }
    Ok(messages)
}

const MESSAGE_FIELDS: &[&str] = &[
    "role",
    "content",
    "tool_calls",
    "tool_call_id",
    "refusal",
    "annotations",
    "audio",
    "function_call",
];

fn decode_message(object: &Map<String, Value>, index: usize) -> Result<Value, PublicError> {
    for key in object.keys() {
        if !MESSAGE_FIELDS.contains(&key.as_str()) {
            return Err(PublicError::invalid_field(&format!("messages.{index}.{key}")));
        }
    }
    let role = match object.get("role").and_then(Value::as_str) {
        Some(role @ ("system" | "developer" | "user" | "assistant" | "tool")) => role,
        _ => return Err(PublicError::invalid_field(&format!("messages.{index}.role"))),
    };
    // SDK-echoed assistant fields are accepted only in their empty forms.
    if !matches!(object.get("refusal"), None | Some(Value::Null)) {
        return Err(PublicError::invalid_field(&format!("messages.{index}.refusal")));
    }
    match object.get("annotations") {
        None | Some(Value::Null) => {}
        Some(Value::Array(items)) if items.is_empty() => {}
        Some(_) => {
            return Err(PublicError::invalid_field(&format!(
                "messages.{index}.annotations"
            )))
        }
    }
    for empty_only in ["audio", "function_call"] {
        if !matches!(object.get(empty_only), None | Some(Value::Null)) {
            return Err(PublicError::invalid_field(&format!(
                "messages.{index}.{empty_only}"
            )));
        }
    }
    let content = decode_content(object.get("content"), index)?;
    let tool_call_id = match object.get("tool_call_id") {
        None | Some(Value::Null) => None,
        Some(Value::String(id)) if !id.is_empty() && id.len() <= 256 => Some(id.clone()),
        Some(_) => {
            return Err(PublicError::invalid_field(&format!(
                "messages.{index}.tool_call_id"
            )))
        }
    };
    let tool_calls = decode_history_tool_calls(object.get("tool_calls"), index)?;

    if role == "assistant" && content.is_null() && tool_calls.is_empty() {
        return Err(PublicError::invalid_field(&format!("messages.{index}")));
    }
    if role != "assistant" && !tool_calls.is_empty() {
        return Err(PublicError::invalid_field(&format!(
            "messages.{index}.tool_calls"
        )));
    }
    if role == "tool" && tool_call_id.is_none() {
        return Err(PublicError::invalid_field(&format!(
            "messages.{index}.tool_call_id"
        )));
    }
    if role != "tool" && tool_call_id.is_some() {
        return Err(PublicError::invalid_field(&format!(
            "messages.{index}.tool_call_id"
        )));
    }
    // The canonical GatewayMessage requires content or assistant tool calls.
    if content.is_null() && tool_calls.is_empty() {
        return Err(PublicError::invalid_field(&format!("messages.{index}")));
    }
    Ok(json!({
        "role": role,
        "content": content,
        "tool_call_id": tool_call_id,
        "tool_calls": tool_calls,
    }))
}

fn decode_content(value: Option<&Value>, index: usize) -> Result<Value, PublicError> {
    match value {
        None | Some(Value::Null) => Ok(Value::Null),
        Some(Value::String(text)) => Ok(Value::String(text.clone())),
        Some(Value::Array(parts)) => {
            let mut joined = String::new();
            for (part_index, part) in parts.iter().enumerate() {
                let object = part.as_object().ok_or_else(|| {
                    PublicError::invalid_field(&format!("messages.{index}.content.{part_index}"))
                })?;
                let part_type = object.get("type").and_then(Value::as_str);
                if !matches!(part_type, Some("text" | "input_text" | "output_text")) {
                    return Err(PublicError::invalid_field(&format!(
                        "messages.{index}.content.{part_index}.type"
                    )));
                }
                for key in object.keys() {
                    if key != "type" && key != "text" {
                        return Err(PublicError::invalid_field(&format!(
                            "messages.{index}.content.{part_index}.{key}"
                        )));
                    }
                }
                let text = object.get("text").and_then(Value::as_str).ok_or_else(|| {
                    PublicError::invalid_field(&format!(
                        "messages.{index}.content.{part_index}.text"
                    ))
                })?;
                joined.push_str(text);
            }
            Ok(Value::String(joined))
        }
        Some(_) => Err(PublicError::invalid_field(&format!(
            "messages.{index}.content"
        ))),
    }
}

fn decode_history_tool_calls(
    value: Option<&Value>,
    message_index: usize,
) -> Result<Vec<Value>, PublicError> {
    let items = match value {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Array(items)) => items,
        Some(_) => {
            return Err(PublicError::invalid_field(&format!(
                "messages.{message_index}.tool_calls"
            )))
        }
    };
    let mut calls = Vec::with_capacity(items.len());
    for (call_index, item) in items.iter().enumerate() {
        let prefix = format!("messages.{message_index}.tool_calls.{call_index}");
        let object = item
            .as_object()
            .ok_or_else(|| PublicError::invalid_field(&prefix))?;
        for key in object.keys() {
            if !["id", "type", "function"].contains(&key.as_str()) {
                return Err(PublicError::invalid_field(&format!("{prefix}.{key}")));
            }
        }
        match object.get("type") {
            None => {}
            Some(Value::String(kind)) if kind == "function" => {}
            Some(_) => return Err(PublicError::invalid_field(&format!("{prefix}.type"))),
        }
        let call_id = match object.get("id").and_then(Value::as_str) {
            Some(id) if !id.is_empty() && id.len() <= 256 => id.to_string(),
            _ => return Err(PublicError::invalid_field(&format!("{prefix}.id"))),
        };
        let function = object
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| PublicError::invalid_field(&format!("{prefix}.function")))?;
        for key in function.keys() {
            if !["name", "arguments"].contains(&key.as_str()) {
                return Err(PublicError::invalid_field(&format!("{prefix}.function.{key}")));
            }
        }
        let name = match function.get("name").and_then(Value::as_str) {
            Some(name) if !name.is_empty() && name.len() <= 256 => name.to_string(),
            _ => return Err(PublicError::invalid_field(&format!("{prefix}.function.name"))),
        };
        let raw_arguments = match function.get("arguments").and_then(Value::as_str) {
            Some(raw) if raw.len() <= 4_000_000 => raw,
            _ => {
                return Err(PublicError::invalid_field(&format!(
                    "{prefix}.function.arguments"
                )))
            }
        };
        let param = format!("{prefix}.function.arguments");
        let arguments = match serde_json::from_str::<Value>(raw_arguments) {
            Ok(Value::Object(parsed)) => Value::Object(parsed),
            _ => {
                return Err(PublicError::invalid_field_message(
                    &param,
                    &format!("'{param}' must encode one JSON object."),
                ))
            }
        };
        // raw_arguments preserves provider-order replay text; the pydantic
        // ToolCall model validates it against the parsed object and excludes
        // it from serialization, so persisted digests are unaffected.
        calls.push(json!({
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
            "raw_arguments": raw_arguments,
        }));
    }
    Ok(calls)
}

fn decode_tools(payload: &Map<String, Value>) -> Result<Vec<Value>, PublicError> {
    let items = match payload.get("tools") {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Array(items)) => items,
        Some(_) => return Err(PublicError::invalid_field("tools")),
    };
    let mut tools = Vec::with_capacity(items.len());
    let mut names = std::collections::HashSet::new();
    for (index, item) in items.iter().enumerate() {
        let prefix = format!("tools.{index}");
        let object = item
            .as_object()
            .ok_or_else(|| PublicError::invalid_field(&prefix))?;
        if object.get("type").and_then(Value::as_str) != Some("function") {
            return Err(PublicError::invalid_field(&format!("{prefix}.type")));
        }
        for key in object.keys() {
            if !["type", "function"].contains(&key.as_str()) {
                return Err(PublicError::invalid_field(&format!("{prefix}.{key}")));
            }
        }
        let function = object
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| PublicError::invalid_field(&format!("{prefix}.function")))?;
        for key in function.keys() {
            if !["name", "description", "parameters", "strict"].contains(&key.as_str()) {
                return Err(PublicError::invalid_field(&format!("{prefix}.function.{key}")));
            }
        }
        let name = match function.get("name").and_then(Value::as_str) {
            Some(name) if !name.is_empty() && name.len() <= 256 => name.to_string(),
            _ => return Err(PublicError::invalid_field(&format!("{prefix}.function.name"))),
        };
        if !names.insert(name.clone()) {
            return Err(PublicError::invalid_field("tools"));
        }
        let description = match function.get("description") {
            None | Some(Value::Null) => Value::Null,
            Some(Value::String(text)) if text.len() <= 8_192 => Value::String(text.clone()),
            Some(_) => {
                return Err(PublicError::invalid_field(&format!(
                    "{prefix}.function.description"
                )))
            }
        };
        let parameters = match function.get("parameters") {
            None => Value::Object(Map::new()),
            Some(Value::Object(map)) => Value::Object(map.clone()),
            Some(_) => {
                return Err(PublicError::invalid_field(&format!(
                    "{prefix}.function.parameters"
                )))
            }
        };
        let strict = match function.get("strict") {
            None | Some(Value::Null) => false,
            Some(Value::Bool(flag)) => *flag,
            Some(_) => {
                return Err(PublicError::invalid_field(&format!(
                    "{prefix}.function.strict"
                )))
            }
        };
        tools.push(json!({
            "name": name,
            "description": description,
            "parameters": parameters,
            "strict": strict,
        }));
    }
    Ok(tools)
}

fn decode_tool_choice(
    payload: &Map<String, Value>,
    tools: &[Value],
) -> Result<Value, PublicError> {
    let choice = match payload.get("tool_choice") {
        None | Some(Value::Null) => return Ok(Value::Null),
        Some(value) => value,
    };
    let normalized = match choice {
        Value::String(mode) if ["auto", "none", "required"].contains(&mode.as_str()) => {
            Value::String(mode.clone())
        }
        Value::Object(object) => {
            let function = object.get("function").and_then(Value::as_object);
            let name = function.and_then(|f| f.get("name")).and_then(Value::as_str);
            match (object.get("type").and_then(Value::as_str), name) {
                (Some("function"), Some(name)) => json!({ "name": name }),
                _ => return Err(PublicError::invalid_field("tool_choice")),
            }
        }
        _ => return Err(PublicError::invalid_field("tool_choice")),
    };
    let tool_names: Vec<&str> = tools
        .iter()
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
        .collect();
    if let Some(named) = normalized.get("name").and_then(Value::as_str) {
        if !tool_names.contains(&named) {
            return Err(PublicError::invalid_field("tool_choice"));
        }
    }
    if normalized.as_str() == Some("required") && tools.is_empty() {
        return Err(PublicError::invalid_field("tool_choice"));
    }
    Ok(normalized)
}

fn decode_response_format(payload: &Map<String, Value>) -> Result<Value, PublicError> {
    let object = match payload.get("response_format") {
        None | Some(Value::Null) => return Ok(Value::Null),
        Some(Value::Object(object)) => object,
        Some(_) => return Err(PublicError::invalid_field("response_format")),
    };
    for key in object.keys() {
        if !["type", "json_schema"].contains(&key.as_str()) {
            return Err(PublicError::invalid_field(&format!("response_format.{key}")));
        }
    }
    match object.get("type").and_then(Value::as_str) {
        Some("text") => {
            if object.get("json_schema").is_some_and(|v| !v.is_null()) {
                return Err(PublicError::invalid_field("response_format"));
            }
            Ok(Value::Null)
        }
        Some("json_schema") => {
            let schema = object
                .get("json_schema")
                .and_then(Value::as_object)
                .ok_or_else(|| PublicError::invalid_field("response_format.json_schema"))?;
            for key in schema.keys() {
                if !["name", "description", "schema", "strict"].contains(&key.as_str()) {
                    return Err(PublicError::invalid_field(&format!(
                        "response_format.json_schema.{key}"
                    )));
                }
            }
            let name = match schema.get("name").and_then(Value::as_str) {
                Some(name) if !name.is_empty() && name.len() <= 256 => name.to_string(),
                _ => {
                    return Err(PublicError::invalid_field("response_format.json_schema.name"))
                }
            };
            let description = match schema.get("description") {
                None | Some(Value::Null) => Value::Null,
                Some(Value::String(text)) if text.len() <= 8_192 => Value::String(text.clone()),
                Some(_) => {
                    return Err(PublicError::invalid_field(
                        "response_format.json_schema.description",
                    ))
                }
            };
            let json_schema = match schema.get("schema") {
                Some(Value::Object(map)) => Value::Object(map.clone()),
                _ => {
                    return Err(PublicError::invalid_field(
                        "response_format.json_schema.schema",
                    ))
                }
            };
            let strict = match schema.get("strict") {
                None | Some(Value::Null) => true,
                Some(Value::Bool(flag)) => *flag,
                Some(_) => {
                    return Err(PublicError::invalid_field(
                        "response_format.json_schema.strict",
                    ))
                }
            };
            Ok(json!({
                "name": name,
                "description": description,
                "json_schema": json_schema,
                "strict": strict,
            }))
        }
        _ => Err(PublicError::invalid_field("response_format.type")),
    }
}

fn required_string(
    payload: &Map<String, Value>,
    field: &str,
    max_length: usize,
) -> Result<String, PublicError> {
    match payload.get(field).and_then(Value::as_str) {
        Some(value) if !value.is_empty() && value.len() <= max_length => Ok(value.to_string()),
        _ => Err(PublicError::invalid_field(field)),
    }
}

fn optional_bool(payload: &Map<String, Value>, field: &str) -> Result<Option<bool>, PublicError> {
    match payload.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(flag)) => Ok(Some(*flag)),
        Some(_) => Err(PublicError::invalid_field(field)),
    }
}

fn optional_positive_int(
    payload: &Map<String, Value>,
    field: &str,
) -> Result<Option<u64>, PublicError> {
    match payload.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => match number.as_u64() {
            Some(value) if value > 0 => Ok(Some(value)),
            _ => Err(PublicError::invalid_field(field)),
        },
        Some(_) => Err(PublicError::invalid_field(field)),
    }
}

fn optional_bounded_number(
    payload: &Map<String, Value>,
    field: &str,
    minimum: f64,
    maximum: f64,
) -> Result<Option<Value>, PublicError> {
    match payload.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => {
            let as_float = number.as_f64().ok_or_else(|| PublicError::invalid_field(field))?;
            if !(minimum..=maximum).contains(&as_float) {
                return Err(PublicError::invalid_field(field));
            }
            Ok(Some(Value::Number(number.clone())))
        }
        Some(_) => Err(PublicError::invalid_field(field)),
    }
}
