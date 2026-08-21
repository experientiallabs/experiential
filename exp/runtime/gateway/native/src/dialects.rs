//! Provider wire dialects: payload builders mirroring
//! `exp.runtime.models.providers.streaming_requests` and SSE normalizers
//! mirroring the event mappers in `exp.runtime.models.providers.streaming`.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::errors::{Failure, FailureClass};
use crate::events::{
    openai_compatible_usage, openai_usage, require_string, require_u64, Event, ToolAccumulator,
    Usage,
};
use crate::sse::SseEvent;

/// The three upstream dialects this PoC speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    OpenAiResponses,
    AnthropicMessages,
    OpenAiCompatible,
}

impl Dialect {
    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "openai_responses" => Some(Dialect::OpenAiResponses),
            "anthropic_messages" => Some(Dialect::AnthropicMessages),
            "openai_compatible" => Some(Dialect::OpenAiCompatible),
            _ => None,
        }
    }
}

/// Wire-building hints resolved by the Python control plane at admission.
#[derive(Debug, Clone)]
pub struct WireHints {
    pub model_id: String,
    pub supports_temperature: bool,
    pub reasoning_effort: Option<String>,
    pub token_limit_key: String,
}

fn capability_failure(capability: &str) -> Failure {
    Failure::new(
        FailureClass::UnsupportedCapability,
        &format!("the resolved deployment does not support {capability}"),
    )
}

fn malformed(message: &str) -> Failure {
    Failure::new(FailureClass::MalformedResponse, message)
}

fn canonical_messages(request: &Value) -> Vec<&Map<String, Value>> {
    request
        .get("messages")
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_object).collect())
        .unwrap_or_default()
}

fn message_text(message: &Map<String, Value>) -> Option<&str> {
    message.get("content").and_then(Value::as_str)
}

fn message_tool_calls(message: &Map<String, Value>) -> Vec<&Map<String, Value>> {
    message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| calls.iter().filter_map(Value::as_object).collect())
        .unwrap_or_default()
}

fn raw_arguments_text(call: &Map<String, Value>) -> String {
    if let Some(raw) = call.get("raw_arguments").and_then(Value::as_str) {
        return raw.to_string();
    }
    // Fallback mirrors ToolCall.arguments_json() with default flags: Python's
    // json.dumps default separators include spaces.
    python_default_json(call.get("arguments").unwrap_or(&Value::Null))
}

/// Python `json.dumps` default formatting: ", " and ": " separators, no sort.
fn python_default_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(key, item)| {
                    format!(
                        "{}: {}",
                        serde_json::to_string(key).unwrap_or_default(),
                        python_default_json(item)
                    )
                })
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(python_default_json).collect();
            format!("[{}]", parts.join(", "))
        }
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

fn add_openai_tools(payload: &mut Map<String, Value>, request: &Value, responses: bool) {
    if let Some(tools) = request.get("tools").and_then(Value::as_array) {
        if !tools.is_empty() {
            let converted: Vec<Value> = tools
                .iter()
                .filter_map(Value::as_object)
                .map(|tool| {
                    let inner = json!({
                        "name": tool.get("name"),
                        "description": tool.get("description"),
                        "parameters": tool.get("parameters"),
                        "strict": tool.get("strict"),
                    });
                    if responses {
                        let mut with_type = Map::new();
                        with_type.insert("type".to_string(), json!("function"));
                        if let Value::Object(fields) = inner {
                            for (key, value) in fields {
                                with_type.insert(key, value);
                            }
                        }
                        Value::Object(with_type)
                    } else {
                        json!({"type": "function", "function": inner})
                    }
                })
                .collect();
            payload.insert("tools".to_string(), Value::Array(converted));
        }
    }
    match request.get("tool_choice") {
        None | Some(Value::Null) => {}
        Some(Value::String(mode)) => {
            payload.insert("tool_choice".to_string(), json!(mode));
        }
        Some(Value::Object(named)) => {
            let name = named.get("name").cloned().unwrap_or(Value::Null);
            let choice = if responses {
                json!({"type": "function", "name": name})
            } else {
                json!({"type": "function", "function": {"name": name}})
            };
            payload.insert("tool_choice".to_string(), choice);
        }
        Some(_) => {}
    }
}

/// Build the native streaming Responses payload
/// (`openai_responses_stream_payload`).
pub fn openai_responses_payload(request: &Value, hints: &WireHints) -> Result<Value, Failure> {
    let stop = request.get("stop").and_then(Value::as_array);
    if stop.is_some_and(|items| !items.is_empty()) {
        return Err(capability_failure("stop_sequences"));
    }
    let mut instructions: Vec<String> = Vec::new();
    let mut items: Vec<Value> = Vec::new();
    for message in canonical_messages(request) {
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");
        if role == "system" || role == "developer" {
            match message_text(message) {
                Some(text) => instructions.push(text.to_string()),
                None => return Err(malformed("instruction messages require text")),
            }
            continue;
        }
        if role == "tool" {
            items.push(json!({
                "type": "function_call_output",
                "call_id": message.get("tool_call_id").and_then(Value::as_str).unwrap_or(""),
                "output": message_text(message).unwrap_or(""),
            }));
        } else if role == "user" {
            items.push(json!({"role": "user", "content": message_text(message).unwrap_or("")}));
        } else if role == "assistant" {
            if let Some(text) = message_text(message) {
                items.push(json!({"role": "assistant", "content": text}));
            }
            for call in message_tool_calls(message) {
                items.push(json!({
                    "type": "function_call",
                    "call_id": call.get("call_id"),
                    "name": call.get("name"),
                    "arguments": raw_arguments_text(call),
                }));
            }
        } else {
            return Err(malformed("unsupported Responses message role"));
        }
    }
    let mut payload = Map::new();
    payload.insert("model".to_string(), json!(hints.model_id));
    payload.insert("input".to_string(), Value::Array(items));
    payload.insert("store".to_string(), json!(false));
    payload.insert("stream".to_string(), json!(true));
    if !instructions.is_empty() {
        payload.insert("instructions".to_string(), json!(instructions.join("\n\n")));
    }
    add_openai_tools(&mut payload, request, true);
    if let Some(parallel) = request.get("parallel_tool_calls").and_then(Value::as_bool) {
        payload.insert("parallel_tool_calls".to_string(), json!(parallel));
    }
    if let Some(structured) = request.get("structured_text").and_then(Value::as_object) {
        let mut format_payload = Map::new();
        format_payload.insert("type".to_string(), json!("json_schema"));
        format_payload.insert("name".to_string(), structured.get("name").cloned().unwrap_or(Value::Null));
        format_payload.insert(
            "schema".to_string(),
            structured.get("json_schema").cloned().unwrap_or(Value::Null),
        );
        format_payload.insert(
            "strict".to_string(),
            structured.get("strict").cloned().unwrap_or(json!(true)),
        );
        if let Some(description) = structured.get("description") {
            if !description.is_null() {
                format_payload.insert("description".to_string(), description.clone());
            }
        }
        payload.insert("text".to_string(), json!({"format": Value::Object(format_payload)}));
    }
    if let Some(maximum) = request.get("maximum_output_tokens").and_then(Value::as_u64) {
        payload.insert("max_output_tokens".to_string(), json!(maximum));
    }
    if let Some(temperature) = request.get("temperature") {
        if !temperature.is_null() && hints.supports_temperature {
            payload.insert("temperature".to_string(), temperature.clone());
        }
    }
    if let Some(top_p) = request.get("top_p") {
        if !top_p.is_null() {
            if !hints.supports_temperature {
                return Err(capability_failure("top_p"));
            }
            payload.insert("top_p".to_string(), top_p.clone());
        }
    }
    if let Some(effort) = &hints.reasoning_effort {
        payload.insert("reasoning".to_string(), json!({"effort": effort}));
    }
    Ok(Value::Object(payload))
}

/// Build the native streaming Messages payload
/// (`anthropic_messages_stream_payload`).
pub fn anthropic_messages_payload(request: &Value, hints: &WireHints) -> Result<Value, Failure> {
    if request
        .get("structured_text")
        .is_some_and(|value| !value.is_null())
    {
        return Err(capability_failure("structured_text"));
    }
    let mut system_parts: Vec<String> = Vec::new();
    let mut messages: Vec<Value> = Vec::new();
    for message in canonical_messages(request) {
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");
        if role == "system" || role == "developer" {
            match message_text(message) {
                Some(text) => system_parts.push(text.to_string()),
                None => return Err(malformed("instruction messages require text")),
            }
            continue;
        }
        let (wire_role, blocks): (&str, Vec<Value>) = if role == "tool" {
            (
                "user",
                vec![json!({
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id").and_then(Value::as_str).unwrap_or(""),
                    "content": message_text(message).unwrap_or(""),
                })],
            )
        } else if role == "user" {
            (
                "user",
                vec![json!({"type": "text", "text": message_text(message).unwrap_or("")})],
            )
        } else if role == "assistant" {
            let mut blocks: Vec<Value> = Vec::new();
            if let Some(text) = message_text(message) {
                blocks.push(json!({"type": "text", "text": text}));
            }
            for call in message_tool_calls(message) {
                blocks.push(json!({
                    "type": "tool_use",
                    "id": call.get("call_id"),
                    "name": call.get("name"),
                    "input": call.get("arguments"),
                }));
            }
            ("assistant", blocks)
        } else {
            return Err(malformed("unsupported Anthropic message role"));
        };
        let same_role = messages
            .last()
            .and_then(|last| last.get("role"))
            .and_then(Value::as_str)
            == Some(wire_role);
        if same_role {
            let last = messages.last_mut().expect("just checked non-empty");
            match last.get_mut("content").and_then(Value::as_array_mut) {
                Some(existing) => existing.extend(blocks),
                None => return Err(malformed("Anthropic message content is malformed")),
            }
        } else {
            messages.push(json!({"role": wire_role, "content": blocks}));
        }
    }
    let mut payload = Map::new();
    payload.insert("model".to_string(), json!(hints.model_id));
    payload.insert("messages".to_string(), Value::Array(messages));
    payload.insert(
        "max_tokens".to_string(),
        json!(request
            .get("maximum_output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(4096)),
    );
    payload.insert("stream".to_string(), json!(true));
    if !system_parts.is_empty() {
        payload.insert("system".to_string(), json!(system_parts.join("\n\n")));
    }
    if let Some(tools) = request.get("tools").and_then(Value::as_array) {
        if !tools.is_empty() {
            let converted: Vec<Value> = tools
                .iter()
                .filter_map(Value::as_object)
                .map(|tool| {
                    json!({
                        "name": tool.get("name"),
                        "description": tool.get("description"),
                        "input_schema": tool.get("parameters"),
                    })
                })
                .collect();
            payload.insert("tools".to_string(), Value::Array(converted));
        }
    }
    match request.get("tool_choice") {
        None | Some(Value::Null) => {}
        Some(Value::String(mode)) => {
            let mapped = match mode.as_str() {
                "required" => "any",
                other => other,
            };
            payload.insert("tool_choice".to_string(), json!({"type": mapped}));
        }
        Some(Value::Object(named)) => {
            payload.insert(
                "tool_choice".to_string(),
                json!({"type": "tool", "name": named.get("name")}),
            );
        }
        Some(_) => {}
    }
    if let Some(temperature) = request.get("temperature") {
        if !temperature.is_null() {
            payload.insert("temperature".to_string(), temperature.clone());
        }
    }
    if let Some(top_p) = request.get("top_p") {
        if !top_p.is_null() {
            payload.insert("top_p".to_string(), top_p.clone());
        }
    }
    if let Some(stop) = request.get("stop").and_then(Value::as_array) {
        if !stop.is_empty() {
            payload.insert("stop_sequences".to_string(), Value::Array(stop.clone()));
        }
    }
    Ok(Value::Object(payload))
}

/// Build the streaming Chat Completions payload
/// (`openai_compatible_stream_payload`).
pub fn openai_compatible_payload(request: &Value, hints: &WireHints) -> Result<Value, Failure> {
    let mut messages: Vec<Value> = Vec::new();
    for message in canonical_messages(request) {
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");
        if role == "tool" {
            messages.push(json!({
                "role": "tool",
                "content": message_text(message).unwrap_or(""),
                "tool_call_id": message.get("tool_call_id").and_then(Value::as_str).unwrap_or(""),
            }));
            continue;
        }
        let mut wire = Map::new();
        wire.insert("role".to_string(), json!(role));
        wire.insert(
            "content".to_string(),
            json!(message_text(message).unwrap_or("")),
        );
        let calls = message_tool_calls(message);
        if !calls.is_empty() {
            let converted: Vec<Value> = calls
                .iter()
                .map(|call| {
                    json!({
                        "id": call.get("call_id"),
                        "type": "function",
                        "function": {
                            "name": call.get("name"),
                            "arguments": raw_arguments_text(call),
                        },
                    })
                })
                .collect();
            wire.insert("tool_calls".to_string(), Value::Array(converted));
        }
        messages.push(Value::Object(wire));
    }
    let mut payload = Map::new();
    payload.insert("model".to_string(), json!(hints.model_id));
    payload.insert("messages".to_string(), Value::Array(messages));
    payload.insert("stream".to_string(), json!(true));
    payload.insert("stream_options".to_string(), json!({"include_usage": true}));
    add_openai_tools(&mut payload, request, false);
    if let Some(parallel) = request.get("parallel_tool_calls").and_then(Value::as_bool) {
        payload.insert("parallel_tool_calls".to_string(), json!(parallel));
    }
    if let Some(structured) = request.get("structured_text").and_then(Value::as_object) {
        let mut schema = Map::new();
        schema.insert("name".to_string(), structured.get("name").cloned().unwrap_or(Value::Null));
        schema.insert(
            "schema".to_string(),
            structured.get("json_schema").cloned().unwrap_or(Value::Null),
        );
        schema.insert(
            "strict".to_string(),
            structured.get("strict").cloned().unwrap_or(json!(true)),
        );
        if let Some(description) = structured.get("description") {
            if !description.is_null() {
                schema.insert("description".to_string(), description.clone());
            }
        }
        payload.insert(
            "response_format".to_string(),
            json!({"type": "json_schema", "json_schema": Value::Object(schema)}),
        );
    }
    if let Some(maximum) = request.get("maximum_output_tokens").and_then(Value::as_u64) {
        payload.insert(hints.token_limit_key.clone(), json!(maximum));
    }
    if let Some(temperature) = request.get("temperature") {
        if !temperature.is_null() {
            payload.insert("temperature".to_string(), temperature.clone());
        }
    }
    if let Some(top_p) = request.get("top_p") {
        if !top_p.is_null() {
            payload.insert("top_p".to_string(), top_p.clone());
        }
    }
    if let Some(stop) = request.get("stop").and_then(Value::as_array) {
        if !stop.is_empty() {
            payload.insert("stop".to_string(), Value::Array(stop.clone()));
        }
    }
    Ok(Value::Object(payload))
}

/// Build the upstream payload for one dialect.
pub fn build_payload(
    dialect: Dialect,
    request: &Value,
    hints: &WireHints,
) -> Result<Value, Failure> {
    match dialect {
        Dialect::OpenAiResponses => openai_responses_payload(request, hints),
        Dialect::AnthropicMessages => anthropic_messages_payload(request, hints),
        Dialect::OpenAiCompatible => openai_compatible_payload(request, hints),
    }
}

fn refusal_failure() -> Failure {
    Failure::new(FailureClass::Refusal, "provider refused the request")
}

fn provider_stream_failed() -> Failure {
    Failure::new(FailureClass::ProviderInternal, "provider stream failed")
}

fn parse_object(data: &str) -> Result<Map<String, Value>, Failure> {
    match serde_json::from_str::<Value>(data) {
        Ok(Value::Object(object)) => Ok(object),
        Ok(_) => Err(malformed("provider stream event must be a JSON object")),
        Err(_) => Err(malformed("provider stream event is not valid JSON")),
    }
}

fn optional_text(object: &Map<String, Value>, key: &str, label: &str) -> Result<String, Failure> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(text)) => Ok(text.clone()),
        Some(_) => Err(malformed(&format!("{label} must be text"))),
    }
}

fn finish_open_tools(tools: &mut BTreeMap<u32, ToolAccumulator>) -> Result<Vec<Event>, Failure> {
    let mut events = Vec::new();
    for (index, tool) in tools.iter_mut() {
        if !tool.completed {
            tool.completed = true;
            let call = tool.complete().map_err(|message| malformed(&message))?;
            events.push(Event::ToolCallCompleted { index: *index, call });
        }
    }
    Ok(events)
}

/// Incremental normalizer of one upstream SSE stream into gateway events.
pub struct Normalizer {
    dialect: Dialect,
    tools: BTreeMap<u32, ToolAccumulator>,
    refusal_seen: bool,
    terminal: bool,
    // Anthropic accumulation.
    input_tokens: u64,
    output_tokens: u64,
    cache_read: u64,
    cache_write: u64,
    stop_reason: Option<String>,
    // OpenAI-compatible accumulation.
    usage: Option<Usage>,
    finish_reason: Option<String>,
}

impl Normalizer {
    pub fn new(dialect: Dialect) -> Self {
        Self {
            dialect,
            tools: BTreeMap::new(),
            refusal_seen: false,
            terminal: false,
            input_tokens: 0,
            output_tokens: 0,
            cache_read: 0,
            cache_write: 0,
            stop_reason: None,
            usage: None,
            finish_reason: None,
        }
    }

    pub fn saw_terminal(&self) -> bool {
        self.terminal
    }

    /// Feed one decoded SSE frame; a terminal event ends the stream.
    pub fn feed(&mut self, frame: &SseEvent) -> Result<Vec<Event>, Failure> {
        if self.terminal {
            return Ok(Vec::new());
        }
        let events = match self.dialect {
            Dialect::OpenAiResponses => self.feed_openai_responses(frame),
            Dialect::AnthropicMessages => self.feed_anthropic(frame),
            Dialect::OpenAiCompatible => self.feed_openai_compatible(frame),
        }?;
        if events.iter().any(Event::is_terminal) {
            self.terminal = true;
        }
        Ok(events)
    }

    /// The upstream byte stream closed; mirror the ended-without-terminal path.
    pub fn stream_ended(&self) -> Result<(), Failure> {
        if self.terminal {
            return Ok(());
        }
        Err(Failure::new(
            FailureClass::MalformedResponse,
            "provider stream ended without a terminal event",
        ))
    }

    fn feed_openai_responses(&mut self, frame: &SseEvent) -> Result<Vec<Event>, Failure> {
        if frame.data == "[DONE]" {
            return Err(malformed(
                "OpenAI Responses stream ended before a terminal event",
            ));
        }
        let payload = parse_object(&frame.data)?;
        let event_type = payload
            .get("type")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| frame.event.clone())
            .unwrap_or_default();
        let mut events = Vec::new();
        match event_type.as_str() {
            "response.output_text.delta" => {
                let delta = optional_text(&payload, "delta", "OpenAI text delta")?;
                if !delta.is_empty() {
                    events.push(Event::TextDelta(delta));
                }
            }
            "response.refusal.delta" => {
                let delta = optional_text(&payload, "delta", "OpenAI refusal delta")?;
                self.refusal_seen = true;
                events.push(Event::RefusalDelta(delta));
            }
            "response.output_item.added" => {
                let item = payload
                    .get("item")
                    .and_then(Value::as_object)
                    .ok_or_else(|| malformed("OpenAI output item must be an object"))?;
                let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
                if item_type == "function_call" {
                    let index = require_u64(&payload, "output_index", "OpenAI output_index")
                        .map_err(|message| malformed(&message))? as u32;
                    if self.tools.contains_key(&index) {
                        return Err(malformed("OpenAI stream repeated a tool-call start"));
                    }
                    let call_id = item
                        .get("call_id")
                        .or_else(|| item.get("id"))
                        .and_then(Value::as_str)
                        .ok_or_else(|| malformed("OpenAI function call ID must be text"))?
                        .to_string();
                    let name = require_string(item, "name", "OpenAI function call name")
                        .map_err(|message| malformed(&message))?;
                    let mut tool = ToolAccumulator::new(index, call_id.clone(), name.clone());
                    events.push(Event::ToolCallStarted { index, call_id, name });
                    if let Some(Value::String(initial)) = item.get("arguments") {
                        if !initial.is_empty() {
                            tool.raw_arguments.push_str(initial);
                            events.push(Event::ToolArgumentsDelta {
                                index,
                                delta: initial.clone(),
                            });
                        }
                    }
                    self.tools.insert(index, tool);
                } else if item_type != "message" && item_type != "reasoning" {
                    return Err(malformed("OpenAI stream emitted an unsupported output item"));
                }
            }
            "response.function_call_arguments.delta" => {
                let index = require_u64(&payload, "output_index", "OpenAI output_index")
                    .map_err(|message| malformed(&message))? as u32;
                let delta = optional_text(&payload, "delta", "OpenAI argument delta")?;
                let tool = self
                    .tools
                    .get_mut(&index)
                    .ok_or_else(|| malformed("provider emitted arguments before a tool start"))?;
                tool.raw_arguments.push_str(&delta);
                events.push(Event::ToolArgumentsDelta { index, delta });
            }
            "response.function_call_arguments.done" | "response.output_item.done" => {
                let index = require_u64(&payload, "output_index", "OpenAI output_index")
                    .map_err(|message| malformed(&message))? as u32;
                if let Some(tool) = self.tools.get_mut(&index) {
                    if !tool.completed {
                        let mut final_arguments = payload
                            .get("arguments")
                            .and_then(Value::as_str)
                            .map(str::to_string);
                        if event_type == "response.output_item.done" {
                            if let Some(item) = payload.get("item").and_then(Value::as_object) {
                                if let Some(from_item) =
                                    item.get("arguments").and_then(Value::as_str)
                                {
                                    final_arguments = Some(from_item.to_string());
                                }
                            }
                        }
                        if let Some(final_arguments) = final_arguments {
                            if !tool.raw_arguments.is_empty()
                                && tool.raw_arguments != final_arguments
                            {
                                return Err(malformed(
                                    "OpenAI tool argument fragments changed at done",
                                ));
                            }
                            if tool.raw_arguments.is_empty() && !final_arguments.is_empty() {
                                tool.raw_arguments = final_arguments.clone();
                                events.push(Event::ToolArgumentsDelta {
                                    index,
                                    delta: final_arguments,
                                });
                            }
                        }
                        tool.completed = true;
                        let call = tool.complete().map_err(|message| malformed(&message))?;
                        events.push(Event::ToolCallCompleted { index, call });
                    }
                }
            }
            "response.completed" | "response.incomplete" => {
                let response = payload
                    .get("response")
                    .and_then(Value::as_object)
                    .ok_or_else(|| malformed("OpenAI terminal response must be an object"))?;
                events.extend(finish_open_tools(&mut self.tools)?);
                if let Some(usage) = openai_usage(response.get("usage")) {
                    events.push(Event::Usage(usage));
                }
                let is_incomplete = event_type == "response.incomplete"
                    || response.get("status").and_then(Value::as_str) == Some("incomplete");
                if !is_incomplete {
                    if self.refusal_seen {
                        events.push(Event::Failed(refusal_failure()));
                    } else {
                        events.push(Event::Completed);
                    }
                } else {
                    let details = response
                        .get("incomplete_details")
                        .and_then(Value::as_object)
                        .ok_or_else(|| malformed("OpenAI incomplete details must be an object"))?;
                    let reason = details
                        .get("reason")
                        .and_then(Value::as_str)
                        .ok_or_else(|| malformed("OpenAI incomplete reason must be text"))?;
                    if reason == "max_output_tokens" {
                        events.push(Event::Incomplete);
                    } else if reason == "content_filter" || reason == "safety" {
                        events.push(Event::Failed(refusal_failure()));
                    } else {
                        events.push(Event::Failed(Failure::new(
                            FailureClass::ProviderInternal,
                            "provider ended the stream incompletely",
                        )));
                    }
                }
            }
            "response.failed" => {
                events.push(Event::Failed(provider_stream_failed()));
            }
            _ => {}
        }
        Ok(events)
    }

    fn feed_anthropic(&mut self, frame: &SseEvent) -> Result<Vec<Event>, Failure> {
        let payload = parse_object(&frame.data)?;
        let event_type = payload
            .get("type")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| frame.event.clone())
            .unwrap_or_default();
        let mut events = Vec::new();
        match event_type.as_str() {
            "message_start" => {
                let message = payload
                    .get("message")
                    .and_then(Value::as_object)
                    .ok_or_else(|| malformed("Anthropic message_start.message must be an object"))?;
                let usage = message
                    .get("usage")
                    .and_then(Value::as_object)
                    .ok_or_else(|| malformed("Anthropic message_start.usage must be an object"))?;
                self.input_tokens = require_u64(usage, "input_tokens", "Anthropic input_tokens")
                    .map_err(|message| malformed(&message))?;
                self.cache_read = require_u64(
                    usage,
                    "cache_read_input_tokens",
                    "Anthropic cache_read_input_tokens",
                )
                .map_err(|message| malformed(&message))?;
                self.cache_write = require_u64(
                    usage,
                    "cache_creation_input_tokens",
                    "Anthropic cache_creation_input_tokens",
                )
                .map_err(|message| malformed(&message))?;
            }
            "content_block_start" => {
                let index = require_u64(&payload, "index", "Anthropic content index")
                    .map_err(|message| malformed(&message))? as u32;
                let block = payload
                    .get("content_block")
                    .and_then(Value::as_object)
                    .ok_or_else(|| malformed("Anthropic content block must be an object"))?;
                match block.get("type").and_then(Value::as_str) {
                    Some("tool_use") => {
                        let call_id = require_string(block, "id", "Anthropic tool ID")
                            .map_err(|message| malformed(&message))?;
                        let name = require_string(block, "name", "Anthropic tool name")
                            .map_err(|message| malformed(&message))?;
                        if self.tools.contains_key(&index) {
                            return Err(malformed("Anthropic stream repeated a tool-call start"));
                        }
                        self.tools
                            .insert(index, ToolAccumulator::new(index, call_id.clone(), name.clone()));
                        events.push(Event::ToolCallStarted { index, call_id, name });
                    }
                    Some("text") => {
                        let text = optional_text(block, "text", "Anthropic initial text")?;
                        if !text.is_empty() {
                            events.push(Event::TextDelta(text));
                        }
                    }
                    Some("refusal") => {
                        self.refusal_seen = true;
                        events.push(Event::RefusalDelta(optional_text(
                            block,
                            "refusal",
                            "Anthropic refusal",
                        )?));
                    }
                    // Blocks with no gateway-visible output (extended thinking)
                    // are skipped rather than rejected.
                    _ => {}
                }
            }
            "content_block_delta" => {
                let index = require_u64(&payload, "index", "Anthropic content index")
                    .map_err(|message| malformed(&message))? as u32;
                let delta = payload
                    .get("delta")
                    .and_then(Value::as_object)
                    .ok_or_else(|| malformed("Anthropic content delta must be an object"))?;
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        let text = optional_text(delta, "text", "Anthropic text delta")?;
                        if !text.is_empty() {
                            events.push(Event::TextDelta(text));
                        }
                    }
                    Some("input_json_delta") => {
                        let fragment =
                            optional_text(delta, "partial_json", "Anthropic argument delta")?;
                        let tool = self.tools.get_mut(&index).ok_or_else(|| {
                            malformed("provider emitted arguments before a tool start")
                        })?;
                        tool.raw_arguments.push_str(&fragment);
                        events.push(Event::ToolArgumentsDelta {
                            index,
                            delta: fragment,
                        });
                    }
                    Some("refusal_delta") => {
                        self.refusal_seen = true;
                        events.push(Event::RefusalDelta(optional_text(
                            delta,
                            "refusal",
                            "Anthropic refusal delta",
                        )?));
                    }
                    // thinking_delta and signature_delta are skipped.
                    _ => {}
                }
            }
            "content_block_stop" => {
                let index = require_u64(&payload, "index", "Anthropic content index")
                    .map_err(|message| malformed(&message))? as u32;
                if let Some(tool) = self.tools.get_mut(&index) {
                    if !tool.completed {
                        tool.completed = true;
                        let call = tool.complete().map_err(|message| malformed(&message))?;
                        events.push(Event::ToolCallCompleted { index, call });
                    }
                }
            }
            "message_delta" => {
                let delta = payload
                    .get("delta")
                    .and_then(Value::as_object)
                    .ok_or_else(|| malformed("Anthropic message delta must be an object"))?;
                if let Some(Value::String(reason)) = delta.get("stop_reason") {
                    self.stop_reason = Some(reason.clone());
                }
                let usage = payload
                    .get("usage")
                    .and_then(Value::as_object)
                    .ok_or_else(|| malformed("Anthropic message_delta.usage must be an object"))?;
                self.output_tokens = require_u64(usage, "output_tokens", "Anthropic output_tokens")
                    .map_err(|message| malformed(&message))?;
                if self.stop_reason.as_deref() == Some("refusal") && !self.refusal_seen {
                    self.refusal_seen = true;
                    events.push(Event::RefusalDelta(String::new()));
                }
            }
            "message_stop" => {
                events.extend(finish_open_tools(&mut self.tools)?);
                events.push(Event::Usage(Usage {
                    input_tokens: Some(self.input_tokens + self.cache_read + self.cache_write),
                    output_tokens: Some(self.output_tokens),
                    cached_input_tokens: Some(self.cache_read),
                    reasoning_tokens: None,
                }));
                if self.refusal_seen || self.stop_reason.as_deref() == Some("refusal") {
                    events.push(Event::Failed(refusal_failure()));
                } else if self.stop_reason.as_deref() == Some("max_tokens") {
                    events.push(Event::Incomplete);
                } else {
                    events.push(Event::Completed);
                }
            }
            "error" => {
                events.push(Event::Failed(provider_stream_failed()));
            }
            "ping" => {}
            _ => {
                return Err(malformed("Anthropic stream emitted an unsupported event"));
            }
        }
        Ok(events)
    }

    fn feed_openai_compatible(&mut self, frame: &SseEvent) -> Result<Vec<Event>, Failure> {
        if frame.data == "[DONE]" {
            let mut events = finish_open_tools(&mut self.tools)?;
            if let Some(usage) = self.usage.take() {
                events.push(Event::Usage(usage));
            }
            let finish = self.finish_reason.as_deref();
            if self.refusal_seen || matches!(finish, Some("content_filter" | "safety")) {
                events.push(Event::Failed(refusal_failure()));
            } else if finish == Some("length") {
                events.push(Event::Incomplete);
            } else {
                events.push(Event::Completed);
            }
            return Ok(events);
        }
        let payload = parse_object(&frame.data)?;
        if payload.get("error").is_some_and(|value| !value.is_null()) {
            return Ok(vec![Event::Failed(provider_stream_failed())]);
        }
        let mut events = Vec::new();
        if let Some(raw_usage) = payload.get("usage") {
            if !raw_usage.is_null() {
                self.usage = openai_compatible_usage(raw_usage);
            }
        }
        let choices = payload
            .get("choices")
            .and_then(Value::as_array)
            .ok_or_else(|| malformed("OpenAI-compatible choices must be an array"))?;
        if choices.is_empty() {
            return Ok(events);
        }
        if choices.len() != 1 {
            return Err(malformed("OpenAI-compatible stream must contain one choice"));
        }
        let choice = choices[0]
            .as_object()
            .ok_or_else(|| malformed("OpenAI-compatible choice must be an object"))?;
        let delta = choice
            .get("delta")
            .and_then(Value::as_object)
            .ok_or_else(|| malformed("OpenAI-compatible delta must be an object"))?;
        if let Some(Value::String(content)) = delta.get("content") {
            if !content.is_empty() {
                events.push(Event::TextDelta(content.clone()));
            }
        }
        if let Some(Value::String(refusal)) = delta.get("refusal") {
            self.refusal_seen = true;
            events.push(Event::RefusalDelta(refusal.clone()));
        }
        if let Some(raw_tools) = delta.get("tool_calls") {
            if !raw_tools.is_null() {
                let items = raw_tools
                    .as_array()
                    .ok_or_else(|| malformed("OpenAI-compatible tool_calls must be an array"))?;
                for value in items {
                    let item = value
                        .as_object()
                        .ok_or_else(|| malformed("OpenAI-compatible tool call must be an object"))?;
                    let index = require_u64(item, "index", "OpenAI-compatible tool index")
                        .map_err(|message| malformed(&message))? as u32;
                    let function = item
                        .get("function")
                        .and_then(Value::as_object)
                        .ok_or_else(|| {
                            malformed("OpenAI-compatible tool function must be an object")
                        })?;
                    if let Some(tool) = self.tools.get(&index) {
                        if let Some(Value::String(repeated_id)) = item.get("id") {
                            if repeated_id != &tool.call_id {
                                return Err(malformed(
                                    "OpenAI-compatible stream changed a tool-call ID",
                                ));
                            }
                        }
                        if let Some(Value::String(repeated_name)) = function.get("name") {
                            if repeated_name != &tool.name {
                                return Err(malformed(
                                    "OpenAI-compatible stream changed a tool-call name",
                                ));
                            }
                        }
                    } else {
                        let call_id = require_string(item, "id", "OpenAI-compatible tool ID")
                            .map_err(|message| malformed(&message))?;
                        let name = require_string(function, "name", "OpenAI-compatible tool name")
                            .map_err(|message| malformed(&message))?;
                        self.tools
                            .insert(index, ToolAccumulator::new(index, call_id.clone(), name.clone()));
                        events.push(Event::ToolCallStarted { index, call_id, name });
                    }
                    if let Some(fragment) = function.get("arguments") {
                        if !fragment.is_null() {
                            let raw_fragment = match fragment {
                                Value::String(text) => text.clone(),
                                _ => {
                                    return Err(malformed(
                                        "OpenAI-compatible argument delta must be text",
                                    ))
                                }
                            };
                            let tool = self.tools.get_mut(&index).expect("tool just ensured");
                            tool.raw_arguments.push_str(&raw_fragment);
                            events.push(Event::ToolArgumentsDelta {
                                index,
                                delta: raw_fragment,
                            });
                        }
                    }
                }
            }
        }
        if let Some(Value::String(finish)) = choice.get("finish_reason") {
            self.finish_reason = Some(finish.clone());
            if matches!(finish.as_str(), "content_filter" | "safety") && !self.refusal_seen {
                self.refusal_seen = true;
                events.push(Event::RefusalDelta(String::new()));
            }
        }
        Ok(events)
    }
}
