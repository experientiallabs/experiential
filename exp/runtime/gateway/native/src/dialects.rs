//! Provider wire dialects: SSE normalizers mirroring the event mappers in
//! `exp.runtime.models.providers.streaming`. Upstream payloads are built by
//! the python control plane with the shared `streaming_requests` builders and
//! arrive fully formed in the admission response.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::errors::{Failure, FailureClass};
use crate::events::{
    count_or_zero, openai_compatible_usage, openai_usage, require_string, require_u64, Event,
    ToolAccumulator, Usage,
};
use crate::eventstream::EventStreamDecoder;
use crate::sse::{SseDecoder, SseEvent};

/// The upstream dialects the native engine speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    OpenAiResponses,
    AnthropicMessages,
    OpenAiCompatible,
    GeminiGenerateContent,
    BedrockConverseStream,
}

impl Dialect {
    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "openai_responses" => Some(Dialect::OpenAiResponses),
            "anthropic_messages" => Some(Dialect::AnthropicMessages),
            "openai_compatible" => Some(Dialect::OpenAiCompatible),
            "gemini_generate_content" => Some(Dialect::GeminiGenerateContent),
            "bedrock_converse_stream" => Some(Dialect::BedrockConverseStream),
            _ => None,
        }
    }
}

/// Dialect-selected incremental frame decoder over provider response bytes.
/// SSE dialects reuse the shared SSE decoder; Bedrock decodes the AWS binary
/// event-stream framing into the same frame shape.
pub enum FrameDecoder {
    Sse(SseDecoder),
    EventStream(EventStreamDecoder),
}

impl FrameDecoder {
    pub fn new(dialect: Dialect) -> Self {
        match dialect {
            Dialect::BedrockConverseStream => FrameDecoder::EventStream(EventStreamDecoder::new()),
            _ => FrameDecoder::Sse(SseDecoder::new()),
        }
    }

    /// Feed one network chunk, returning every complete frame it closes.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, String> {
        match self {
            FrameDecoder::Sse(decoder) => decoder.feed(chunk),
            FrameDecoder::EventStream(decoder) => decoder.feed(chunk),
        }
    }

    /// Close the stream, recovering or rejecting trailing partial frames.
    pub fn finish(&mut self) -> Result<Option<SseEvent>, String> {
        match self {
            FrameDecoder::Sse(decoder) => decoder.finish(),
            FrameDecoder::EventStream(decoder) => decoder.finish(),
        }
    }
}

/// Aggregate per-request ceiling on retained provider output, mirroring the
/// Python engine's 64 MiB bounded-aggregation limit.
pub const MAXIMUM_RETAINED_OUTPUT_BYTES: usize = 64 * 1024 * 1024;

/// The sanitized message that marks an aggregate output overflow; the HTTP
/// layer maps it to the shared `provider_output_too_large` public error.
pub const OUTPUT_OVERFLOW_MESSAGE: &str = "provider output exceeded the gateway response limit";

fn malformed(message: &str) -> Failure {
    Failure::new(FailureClass::MalformedResponse, message)
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
            events.push(Event::ToolCallCompleted {
                index: *index,
                call,
            });
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
    accumulated_tool_bytes: usize,
    // Anthropic accumulation.
    input_tokens: u64,
    output_tokens: u64,
    cache_read: u64,
    cache_write: u64,
    stop_reason: Option<String>,
    // OpenAI-compatible and Gemini accumulation.
    usage: Option<Usage>,
    finish_reason: Option<String>,
    // Gemini accumulation: whole function calls arrive in one part, so the
    // provider supplies no tool index; assignment order mirrors the python
    // mapper's local counter.
    gemini_tool_index: u32,
}

impl Normalizer {
    pub fn new(dialect: Dialect) -> Self {
        Self {
            dialect,
            tools: BTreeMap::new(),
            refusal_seen: false,
            terminal: false,
            accumulated_tool_bytes: 0,
            input_tokens: 0,
            output_tokens: 0,
            cache_read: 0,
            cache_write: 0,
            stop_reason: None,
            usage: None,
            finish_reason: None,
            gemini_tool_index: 0,
        }
    }

    pub fn saw_terminal(&self) -> bool {
        self.terminal
    }

    /// Reserve retained-output budget for accumulated tool-argument text.
    fn reserve_tool_bytes(&mut self, additional: usize) -> Result<(), Failure> {
        self.accumulated_tool_bytes = self.accumulated_tool_bytes.saturating_add(additional);
        if self.accumulated_tool_bytes > MAXIMUM_RETAINED_OUTPUT_BYTES {
            return Err(Failure::new(
                FailureClass::ProviderInternal,
                OUTPUT_OVERFLOW_MESSAGE,
            ));
        }
        Ok(())
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
            Dialect::GeminiGenerateContent => self.feed_gemini(frame),
            Dialect::BedrockConverseStream => self.feed_bedrock(frame),
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
                        .map_err(|message| malformed(&message))?
                        as u32;
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
                    let mut tool = ToolAccumulator::new(call_id.clone(), name.clone());
                    events.push(Event::ToolCallStarted {
                        index,
                        call_id,
                        name,
                    });
                    if let Some(Value::String(initial)) = item.get("arguments") {
                        if !initial.is_empty() {
                            self.reserve_tool_bytes(initial.len())?;
                            tool.raw_arguments.push_str(initial);
                            events.push(Event::ToolArgumentsDelta {
                                index,
                                delta: initial.clone(),
                            });
                        }
                    }
                    self.tools.insert(index, tool);
                } else if item_type != "message" && item_type != "reasoning" {
                    return Err(malformed(
                        "OpenAI stream emitted an unsupported output item",
                    ));
                }
            }
            "response.function_call_arguments.delta" => {
                let index = require_u64(&payload, "output_index", "OpenAI output_index")
                    .map_err(|message| malformed(&message))? as u32;
                let delta = optional_text(&payload, "delta", "OpenAI argument delta")?;
                self.reserve_tool_bytes(delta.len())?;
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
                let pending = self.tools.get(&index).is_some_and(|tool| !tool.completed);
                if pending {
                    let mut final_arguments = payload
                        .get("arguments")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    if event_type == "response.output_item.done" {
                        if let Some(item) = payload.get("item").and_then(Value::as_object) {
                            if let Some(from_item) = item.get("arguments").and_then(Value::as_str) {
                                final_arguments = Some(from_item.to_string());
                            }
                        }
                    }
                    if let Some(final_arguments) = final_arguments {
                        let streamed = &self.tools[&index].raw_arguments;
                        if !streamed.is_empty() && *streamed != final_arguments {
                            return Err(malformed(
                                "OpenAI tool argument fragments changed at done",
                            ));
                        }
                        if streamed.is_empty() && !final_arguments.is_empty() {
                            self.reserve_tool_bytes(final_arguments.len())?;
                            let tool = self.tools.get_mut(&index).expect("tool just checked");
                            tool.raw_arguments = final_arguments.clone();
                            events.push(Event::ToolArgumentsDelta {
                                index,
                                delta: final_arguments,
                            });
                        }
                    }
                    let tool = self.tools.get_mut(&index).expect("tool just checked");
                    tool.completed = true;
                    let call = tool.complete().map_err(|message| malformed(&message))?;
                    events.push(Event::ToolCallCompleted { index, call });
                }
            }
            "response.completed" | "response.incomplete" => {
                let response = payload
                    .get("response")
                    .and_then(Value::as_object)
                    .ok_or_else(|| malformed("OpenAI terminal response must be an object"))?;
                events.extend(finish_open_tools(&mut self.tools)?);
                if let Some(usage) =
                    openai_usage(response.get("usage")).map_err(|message| malformed(&message))?
                {
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
                    .ok_or_else(|| {
                        malformed("Anthropic message_start.message must be an object")
                    })?;
                let usage = message
                    .get("usage")
                    .and_then(Value::as_object)
                    .ok_or_else(|| malformed("Anthropic message_start.usage must be an object"))?;
                // Absent usage fields count as zero (require_integer parity);
                // present malformed values fail the stream.
                self.input_tokens = count_or_zero(usage, "input_tokens", "Anthropic input_tokens")
                    .map_err(|message| malformed(&message))?;
                self.cache_read = count_or_zero(
                    usage,
                    "cache_read_input_tokens",
                    "Anthropic cache_read_input_tokens",
                )
                .map_err(|message| malformed(&message))?;
                self.cache_write = count_or_zero(
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
                            .insert(index, ToolAccumulator::new(call_id.clone(), name.clone()));
                        events.push(Event::ToolCallStarted {
                            index,
                            call_id,
                            name,
                        });
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
                        self.reserve_tool_bytes(fragment.len())?;
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
                self.output_tokens =
                    count_or_zero(usage, "output_tokens", "Anthropic output_tokens")
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
                self.usage = Some(
                    openai_compatible_usage(raw_usage).map_err(|message| malformed(&message))?,
                );
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
            return Err(malformed(
                "OpenAI-compatible stream must contain one choice",
            ));
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
                    let item = value.as_object().ok_or_else(|| {
                        malformed("OpenAI-compatible tool call must be an object")
                    })?;
                    let index = require_u64(item, "index", "OpenAI-compatible tool index")
                        .map_err(|message| malformed(&message))?
                        as u32;
                    let function =
                        item.get("function")
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
                            .insert(index, ToolAccumulator::new(call_id.clone(), name.clone()));
                        events.push(Event::ToolCallStarted {
                            index,
                            call_id,
                            name,
                        });
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
                            self.reserve_tool_bytes(raw_fragment.len())?;
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

    /// Normalize one Gemini `streamGenerateContent` SSE frame, mirroring the
    /// `_gemini_events` mapper: reasoning parts are skipped, whole function
    /// calls expand to start/arguments/completed, and the terminal candidate
    /// flushes the latest usage before its finish reason maps to the shared
    /// completion, incomplete, refusal, or provider-internal outcome.
    fn feed_gemini(&mut self, frame: &SseEvent) -> Result<Vec<Event>, Failure> {
        let payload = parse_object(&frame.data)?;
        if let Some(raw_usage) = payload.get("usageMetadata") {
            if !raw_usage.is_null() {
                self.usage = Some(gemini_usage(raw_usage).map_err(|message| malformed(&message))?);
            }
        }
        let candidates = payload
            .get("candidates")
            .and_then(Value::as_array)
            .ok_or_else(|| malformed("Gemini candidates must be an array"))?;
        let mut events = Vec::new();
        if candidates.is_empty() {
            return Ok(events);
        }
        if candidates.len() != 1 {
            return Err(malformed("Gemini stream must contain one candidate"));
        }
        let candidate = candidates[0]
            .as_object()
            .ok_or_else(|| malformed("Gemini candidate must be an object"))?;
        match candidate.get("content") {
            None | Some(Value::Null) => {}
            Some(content) => {
                let parts = content
                    .as_object()
                    .ok_or_else(|| malformed("Gemini candidate content must be an object"))?
                    .get("parts")
                    .and_then(Value::as_array)
                    .ok_or_else(|| malformed("Gemini candidate parts must be an array"))?;
                for raw_part in parts {
                    let part = raw_part
                        .as_object()
                        .ok_or_else(|| malformed("Gemini candidate part must be an object"))?;
                    // Reasoning parts (thought text and thought signatures)
                    // are not gateway-visible output.
                    if part.get("thought") == Some(&Value::Bool(true)) {
                        continue;
                    }
                    if let Some(call) = part.get("functionCall") {
                        if !call.is_null() {
                            events.extend(self.gemini_tool_events(call)?);
                            continue;
                        }
                    }
                    match part.get("text") {
                        Some(Value::String(text)) => {
                            if !text.is_empty() {
                                events.push(Event::TextDelta(text.clone()));
                            }
                        }
                        // A part with neither visible text nor a function call
                        // (for example a bare thought signature) carries no
                        // gateway-visible output.
                        None | Some(Value::Null) => {}
                        Some(_) => return Err(malformed("Gemini text part must be text")),
                    }
                }
            }
        }
        let finish_reason = match candidate.get("finishReason") {
            None | Some(Value::Null) => return Ok(events),
            Some(Value::String(reason)) => reason.clone(),
            Some(_) => return Err(malformed("Gemini finishReason must be text")),
        };
        if let Some(usage) = self.usage.take() {
            events.push(Event::Usage(usage));
        }
        match finish_reason.as_str() {
            "STOP" | "FINISH_REASON_UNSPECIFIED" => events.push(Event::Completed),
            "MAX_TOKENS" => events.push(Event::Incomplete),
            // The python mapper's refusal signal table: safety, copyright,
            // and sensitive-information stops are content-free refusals.
            "SAFETY" | "PROHIBITED_CONTENT" | "BLOCKLIST" | "RECITATION" | "SPII" => {
                events.push(Event::Failed(refusal_failure()));
            }
            _ => {
                events.push(Event::Failed(Failure::new(
                    FailureClass::ProviderInternal,
                    "provider ended the stream unexpectedly",
                )));
            }
        }
        Ok(events)
    }

    /// Expand one complete Gemini function call into the canonical tool-call
    /// lifecycle, assigning the deterministic local call-ID fallback and the
    /// canonical compact JSON argument text the python mapper produces.
    fn gemini_tool_events(&mut self, value: &Value) -> Result<Vec<Event>, Failure> {
        let call = value
            .as_object()
            .ok_or_else(|| malformed("Gemini functionCall must be an object"))?;
        let name = require_string(call, "name", "Gemini functionCall name")
            .map_err(|message| malformed(&message))?;
        let index = self.gemini_tool_index;
        self.gemini_tool_index += 1;
        let call_id = match call.get("id") {
            Some(Value::String(id)) if !id.is_empty() => id.clone(),
            _ => format!("gemini-call-{index}"),
        };
        let arguments = match call.get("args") {
            None => Value::Object(Map::new()),
            Some(Value::Object(map)) => Value::Object(map.clone()),
            Some(_) => return Err(malformed("Gemini functionCall args must be an object")),
        };
        let raw_arguments = serde_json::to_string(&arguments)
            .map_err(|_| malformed("Gemini functionCall args must be an object"))?;
        self.reserve_tool_bytes(raw_arguments.len())?;
        let mut tool = ToolAccumulator::new(call_id.clone(), name.clone());
        tool.raw_arguments = raw_arguments.clone();
        let completed = tool.complete().map_err(|message| malformed(&message))?;
        Ok(vec![
            Event::ToolCallStarted {
                index,
                call_id,
                name,
            },
            Event::ToolArgumentsDelta {
                index,
                delta: raw_arguments,
            },
            Event::ToolCallCompleted {
                index,
                call: completed,
            },
        ])
    }

    /// Normalize one Bedrock ConverseStream frame, mirroring the python
    /// `BedrockProviderStream._decode` mapper: tool calls stream as indexed
    /// content blocks, `messageStop` retains the stop reason, and the trailing
    /// `metadata` frame flushes usage and maps the retained reason to the
    /// shared terminal outcome. Service exceptions arrive as their own frames
    /// and map to the python mapper's failure classes.
    fn feed_bedrock(&mut self, frame: &SseEvent) -> Result<Vec<Event>, Failure> {
        match frame.event.as_deref().unwrap_or("") {
            "messageStart" => Ok(Vec::new()),
            "contentBlockStart" => self.bedrock_content_start(frame),
            "contentBlockDelta" => self.bedrock_content_delta(frame),
            "contentBlockStop" => self.bedrock_content_stop(frame),
            "messageStop" => {
                let payload = parse_object(&frame.data)?;
                self.stop_reason = Some(
                    require_string(&payload, "stopReason", "Bedrock stopReason")
                        .map_err(|message| malformed(&message))?,
                );
                Ok(Vec::new())
            }
            "metadata" => self.bedrock_metadata(frame),
            "throttlingException" => Ok(vec![Event::Failed(Failure::new(
                FailureClass::Throttled,
                "provider throttled the request",
            ))]),
            "modelTimeoutException" => Ok(vec![Event::Failed(Failure::new(
                FailureClass::Timeout,
                "provider request timed out",
            ))]),
            "internalServerException"
            | "modelStreamErrorException"
            | "serviceUnavailableException" => Ok(vec![Event::Failed(Failure::new(
                FailureClass::ProviderInternal,
                "provider stream failed",
            ))]),
            "validationException" => Ok(vec![Event::Failed(Failure::new(
                FailureClass::InvalidRequest,
                "provider rejected the request",
            ))]),
            _ => Err(malformed("Bedrock stream emitted an unsupported event")),
        }
    }

    /// Start one Bedrock tool call, or accept an empty text-block start.
    fn bedrock_content_start(&mut self, frame: &SseEvent) -> Result<Vec<Event>, Failure> {
        let payload = parse_object(&frame.data)?;
        let index = require_u64(&payload, "contentBlockIndex", "Bedrock contentBlockIndex")
            .map_err(|message| malformed(&message))? as u32;
        let start = match payload.get("start") {
            None => Map::new(),
            Some(Value::Object(map)) => map.clone(),
            Some(_) => {
                return Err(malformed(
                    "Bedrock contentBlockStart.start must be an object",
                ))
            }
        };
        let raw_tool = match start.get("toolUse") {
            None | Some(Value::Null) => {
                if start.is_empty() {
                    return Ok(Vec::new());
                }
                return Err(malformed("Bedrock content block start is unsupported"));
            }
            Some(value) => value,
        };
        let tool = raw_tool
            .as_object()
            .ok_or_else(|| malformed("Bedrock toolUse start must be an object"))?;
        if self.tools.contains_key(&index) {
            return Err(malformed("Bedrock stream repeated a tool-call start"));
        }
        let call_id = require_string(tool, "toolUseId", "Bedrock toolUseId")
            .map_err(|message| malformed(&message))?;
        let name = require_string(tool, "name", "Bedrock tool name")
            .map_err(|message| malformed(&message))?;
        self.tools
            .insert(index, ToolAccumulator::new(call_id.clone(), name.clone()));
        Ok(vec![Event::ToolCallStarted {
            index,
            call_id,
            name,
        }])
    }

    /// Normalize one Bedrock text or raw tool-input fragment.
    fn bedrock_content_delta(&mut self, frame: &SseEvent) -> Result<Vec<Event>, Failure> {
        let payload = parse_object(&frame.data)?;
        let index = require_u64(&payload, "contentBlockIndex", "Bedrock contentBlockIndex")
            .map_err(|message| malformed(&message))? as u32;
        let delta = payload
            .get("delta")
            .and_then(Value::as_object)
            .ok_or_else(|| malformed("Bedrock contentBlockDelta.delta must be an object"))?;
        if let Some(Value::String(text)) = delta.get("text") {
            if text.is_empty() {
                return Ok(Vec::new());
            }
            return Ok(vec![Event::TextDelta(text.clone())]);
        }
        let raw_tool = match delta.get("toolUse") {
            None | Some(Value::Null) => {
                return Err(malformed("Bedrock content block delta is unsupported"))
            }
            Some(value) => value,
        };
        if !self.tools.contains_key(&index) {
            return Err(malformed("Bedrock emitted arguments before a tool start"));
        }
        let tool_delta = raw_tool
            .as_object()
            .ok_or_else(|| malformed("Bedrock toolUse delta must be an object"))?;
        let fragment = match tool_delta.get("input") {
            Some(Value::String(fragment)) => fragment.clone(),
            _ => return Err(malformed("Bedrock tool input delta must be text")),
        };
        if fragment.is_empty() {
            return Ok(Vec::new());
        }
        self.reserve_tool_bytes(fragment.len())?;
        let tool = self.tools.get_mut(&index).expect("tool just checked");
        tool.raw_arguments.push_str(&fragment);
        Ok(vec![Event::ToolArgumentsDelta {
            index,
            delta: fragment,
        }])
    }

    /// Complete one open Bedrock tool call at its content-block stop.
    fn bedrock_content_stop(&mut self, frame: &SseEvent) -> Result<Vec<Event>, Failure> {
        let payload = parse_object(&frame.data)?;
        let index = require_u64(&payload, "contentBlockIndex", "Bedrock contentBlockIndex")
            .map_err(|message| malformed(&message))? as u32;
        let Some(mut tool) = self.tools.remove(&index) else {
            return Ok(Vec::new());
        };
        tool.completed = true;
        let call = tool.complete().map_err(|message| malformed(&message))?;
        Ok(vec![Event::ToolCallCompleted { index, call }])
    }

    /// Flush Bedrock usage and, once the stop reason is retained, terminate.
    fn bedrock_metadata(&mut self, frame: &SseEvent) -> Result<Vec<Event>, Failure> {
        let payload = parse_object(&frame.data)?;
        let usage = payload
            .get("usage")
            .and_then(Value::as_object)
            .ok_or_else(|| malformed("Bedrock metadata.usage must be an object"))?;
        let fresh = count_or_zero(usage, "inputTokens", "Bedrock inputTokens")
            .map_err(|message| malformed(&message))?;
        let cache_read = count_or_zero(
            usage,
            "cacheReadInputTokens",
            "Bedrock cacheReadInputTokens",
        )
        .map_err(|message| malformed(&message))?;
        let cache_write = count_or_zero(
            usage,
            "cacheWriteInputTokens",
            "Bedrock cacheWriteInputTokens",
        )
        .map_err(|message| malformed(&message))?;
        let output_tokens = count_or_zero(usage, "outputTokens", "Bedrock outputTokens")
            .map_err(|message| malformed(&message))?;
        let mut events = vec![Event::Usage(Usage {
            input_tokens: Some(fresh + cache_read + cache_write),
            output_tokens: Some(output_tokens),
            cached_input_tokens: Some(cache_read),
            reasoning_tokens: None,
        })];
        if let Some(reason) = self.stop_reason.take() {
            events.push(self.bedrock_terminal(&reason));
        }
        Ok(events)
    }

    /// Map the retained Bedrock stop reason to one terminal gateway event.
    fn bedrock_terminal(&mut self, reason: &str) -> Event {
        if !self.tools.is_empty() {
            self.tools.clear();
            return Event::Failed(Failure::new(
                FailureClass::MalformedResponse,
                "provider stream ended with an incomplete tool call",
            ));
        }
        match reason {
            "end_turn" | "stop_sequence" | "tool_use" => Event::Completed,
            "max_tokens" | "model_context_window_exceeded" => Event::Incomplete,
            "content_filtered" | "guardrail_intervened" => Event::Failed(refusal_failure()),
            _ => Event::Failed(Failure::new(
                FailureClass::ProviderInternal,
                "provider ended the stream unexpectedly",
            )),
        }
    }
}

/// Parse Gemini `usageMetadata`, mirroring the python `_usage` normalizer:
/// cached tokens are an input subset, absent counts are zero (`require_integer`
/// parity), and `thoughtsTokenCount` stays unknown when omitted.
fn gemini_usage(value: &Value) -> Result<Usage, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "Gemini usageMetadata must be an object".to_string())?;
    let reasoning_tokens = match object.get("thoughtsTokenCount") {
        None | Some(Value::Null) => None,
        Some(_) => Some(count_or_zero(
            object,
            "thoughtsTokenCount",
            "Gemini thoughtsTokenCount",
        )?),
    };
    Ok(Usage {
        input_tokens: Some(count_or_zero(
            object,
            "promptTokenCount",
            "Gemini promptTokenCount",
        )?),
        output_tokens: Some(count_or_zero(
            object,
            "candidatesTokenCount",
            "Gemini candidatesTokenCount",
        )?),
        cached_input_tokens: Some(count_or_zero(
            object,
            "cachedContentTokenCount",
            "Gemini cachedContentTokenCount",
        )?),
        reasoning_tokens,
    })
}

#[cfg(test)]
mod gemini_tests {
    use super::*;
    use crate::events::simplified_event;
    use crate::sse::SseDecoder;
    use serde_json::json;

    /// Drain one raw SSE byte stream through the decoder and normalizer the
    /// way the server's collection loop does, returning simplified events and
    /// the failure that ended the stream, when one did.
    fn run_stream(dialect: Dialect, chunks: &[&[u8]]) -> (Vec<Value>, Option<Failure>) {
        let mut normalizer = Normalizer::new(dialect);
        let mut decoder = SseDecoder::new();
        let mut simplified = Vec::new();
        for chunk in chunks {
            let frames = match decoder.feed(chunk) {
                Ok(frames) => frames,
                Err(message) => return (simplified, Some(malformed(&message))),
            };
            for frame in frames {
                match normalizer.feed(&frame) {
                    Ok(events) => {
                        simplified.extend(events.iter().map(simplified_event));
                    }
                    Err(failure) => return (simplified, Some(failure)),
                }
                if normalizer.saw_terminal() {
                    return (simplified, None);
                }
            }
        }
        match decoder.finish() {
            Ok(Some(frame)) => match normalizer.feed(&frame) {
                Ok(events) => simplified.extend(events.iter().map(simplified_event)),
                Err(failure) => return (simplified, Some(failure)),
            },
            Ok(None) => {}
            Err(message) => return (simplified, Some(malformed(&message))),
        }
        if normalizer.saw_terminal() {
            return (simplified, None);
        }
        match normalizer.stream_ended() {
            Ok(()) => (simplified, None),
            Err(failure) => (simplified, Some(failure)),
        }
    }

    fn sse(payload: &Value) -> Vec<u8> {
        format!("data: {payload}\n\n").into_bytes()
    }

    #[test]
    fn gemini_golden_stream_normalizes_text_tools_usage_and_completion() {
        // Golden fixture: raw provider bytes in, exact canonical events out.
        // `native_dialect_parity_test.py` holds the python-mapper comparison.
        let chunks = [
            sse(&json!({"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]})),
            sse(&json!({"candidates": [{"content": {"parts": [
                {"thought": true, "text": "hidden reasoning"},
                {"text": "lo"},
            ]}}]})),
            sse(&json!({"candidates": [{"content": {"parts": [{
                "functionCall": {
                    "id": "call-1",
                    "name": "lookup",
                    "args": {"city": "Zürich", "count": 2},
                }
            }]}}]})),
            sse(&json!({
                "candidates": [{"finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "candidatesTokenCount": 5,
                    "cachedContentTokenCount": 2,
                    "thoughtsTokenCount": 3,
                },
            })),
        ];
        let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(failure.is_none());
        let raw_arguments = "{\"city\":\"Zürich\",\"count\":2}";
        assert_eq!(
            events,
            vec![
                json!({"kind": "text_delta", "text": "Hel"}),
                json!({"kind": "text_delta", "text": "lo"}),
                json!({"kind": "tool_call_started", "index": 0, "call_id": "call-1", "name": "lookup"}),
                json!({"kind": "tool_arguments_delta", "index": 0, "text": raw_arguments}),
                json!({
                    "kind": "tool_call_completed",
                    "index": 0,
                    "call_id": "call-1",
                    "name": "lookup",
                    "raw_arguments": raw_arguments,
                }),
                json!({
                    "kind": "usage",
                    "input_tokens": 11,
                    "output_tokens": 5,
                    "cached_input_tokens": 2,
                    "reasoning_tokens": 3,
                }),
                json!({"kind": "completed"}),
            ]
        );
    }

    #[test]
    fn gemini_missing_call_id_uses_the_deterministic_local_fallback() {
        let chunks = [
            sse(&json!({"candidates": [{"content": {"parts": [
                {"functionCall": {"name": "first", "args": {}}},
                {"functionCall": {"name": "second"}},
            ]}}]})),
            sse(&json!({
                "candidates": [{"finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            })),
        ];
        let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(failure.is_none());
        assert_eq!(
            events[0],
            json!({"kind": "tool_call_started", "index": 0, "call_id": "gemini-call-0", "name": "first"})
        );
        assert_eq!(events[1]["text"], "{}");
        assert_eq!(
            events[3],
            json!({"kind": "tool_call_started", "index": 1, "call_id": "gemini-call-1", "name": "second"})
        );
        // Absent usage counts are zero (require_integer parity), and an
        // omitted thoughtsTokenCount stays unknown.
        assert_eq!(
            events[6],
            json!({
                "kind": "usage",
                "input_tokens": 1,
                "output_tokens": 1,
                "cached_input_tokens": 0,
                "reasoning_tokens": null,
            })
        );
        assert_eq!(events[7], json!({"kind": "completed"}));
    }

    #[test]
    fn gemini_safety_finish_maps_to_a_content_free_refusal() {
        for reason in [
            "SAFETY",
            "PROHIBITED_CONTENT",
            "BLOCKLIST",
            "RECITATION",
            "SPII",
        ] {
            let chunks = [sse(&json!({"candidates": [{"finishReason": reason}]}))];
            let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
            let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
            assert!(failure.is_none());
            assert_eq!(
                events,
                vec![json!({
                    "kind": "failed",
                    "failure_class": "refusal",
                    "safe_message": "provider refused the request",
                })]
            );
        }
    }

    #[test]
    fn gemini_max_tokens_is_incomplete_and_unknown_reasons_fail() {
        let incomplete = [sse(
            &json!({"candidates": [{"finishReason": "MAX_TOKENS"}]}),
        )];
        let refs: Vec<&[u8]> = incomplete.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(failure.is_none());
        assert_eq!(events, vec![json!({"kind": "incomplete"})]);

        let unknown = [sse(
            &json!({"candidates": [{"finishReason": "MALFUNCTION"}]}),
        )];
        let refs: Vec<&[u8]> = unknown.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert!(failure.is_none());
        assert_eq!(
            events,
            vec![json!({
                "kind": "failed",
                "failure_class": "provider_internal",
                "safe_message": "provider ended the stream unexpectedly",
            })]
        );
    }

    #[test]
    fn gemini_malformed_frames_fail_the_stream() {
        // A non-text text part fails, exactly like the python mapper.
        let bad_text = [sse(
            &json!({"candidates": [{"content": {"parts": [{"text": 5}]}}]}),
        )];
        let refs: Vec<&[u8]> = bad_text.iter().map(Vec::as_slice).collect();
        let (_, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert_eq!(
            failure.expect("must fail").failure_class,
            FailureClass::MalformedResponse
        );

        // Null function-call args fail (python: args must decode to a dict).
        let bad_args = [sse(&json!({"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "x", "args": null}}
        ]}}]}))];
        let refs: Vec<&[u8]> = bad_args.iter().map(Vec::as_slice).collect();
        let (_, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert_eq!(
            failure.expect("must fail").failure_class,
            FailureClass::MalformedResponse
        );

        // Two candidates in one frame fail.
        let two = [sse(&json!({"candidates": [{}, {}]}))];
        let refs: Vec<&[u8]> = two.iter().map(Vec::as_slice).collect();
        let (_, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert_eq!(
            failure.expect("must fail").failure_class,
            FailureClass::MalformedResponse
        );

        // A stream that closes without a terminal candidate fails.
        let unterminated = [sse(
            &json!({"candidates": [{"content": {"parts": [{"text": "a"}]}}]}),
        )];
        let refs: Vec<&[u8]> = unterminated.iter().map(Vec::as_slice).collect();
        let (events, failure) = run_stream(Dialect::GeminiGenerateContent, &refs);
        assert_eq!(events, vec![json!({"kind": "text_delta", "text": "a"})]);
        assert_eq!(
            failure.expect("must fail").failure_class,
            FailureClass::MalformedResponse
        );
    }

    #[test]
    fn gemini_frames_after_the_terminal_candidate_are_ignored() {
        let mut normalizer = Normalizer::new(Dialect::GeminiGenerateContent);
        let terminal = crate::sse::SseEvent {
            event: None,
            data: json!({"candidates": [{"finishReason": "STOP"}]}).to_string(),
        };
        let trailing = crate::sse::SseEvent {
            event: None,
            data: json!({"candidates": [{"content": {"parts": [{"text": "late"}]}}]}).to_string(),
        };
        let events = normalizer.feed(&terminal).expect("terminal frame");
        assert!(events.iter().any(Event::is_terminal));
        assert!(normalizer.saw_terminal());
        assert!(normalizer.feed(&trailing).expect("ignored").is_empty());
    }
}

#[cfg(test)]
mod bedrock_tests {
    use super::*;
    use crate::events::simplified_event;
    use crate::eventstream::encode_message;
    use serde_json::json;

    /// Drain one raw event-stream byte stream through the frame decoder and
    /// normalizer the way the server's collection loop does.
    fn run_stream(chunks: &[Vec<u8>]) -> (Vec<Value>, Option<Failure>) {
        let mut normalizer = Normalizer::new(Dialect::BedrockConverseStream);
        let mut decoder = FrameDecoder::new(Dialect::BedrockConverseStream);
        let mut simplified = Vec::new();
        for chunk in chunks {
            let frames = match decoder.feed(chunk) {
                Ok(frames) => frames,
                Err(message) => return (simplified, Some(malformed(&message))),
            };
            for frame in frames {
                match normalizer.feed(&frame) {
                    Ok(events) => simplified.extend(events.iter().map(simplified_event)),
                    Err(failure) => return (simplified, Some(failure)),
                }
                if normalizer.saw_terminal() {
                    return (simplified, None);
                }
            }
        }
        match decoder.finish() {
            Ok(Some(frame)) => match normalizer.feed(&frame) {
                Ok(events) => simplified.extend(events.iter().map(simplified_event)),
                Err(failure) => return (simplified, Some(failure)),
            },
            Ok(None) => {}
            Err(message) => return (simplified, Some(malformed(&message))),
        }
        if normalizer.saw_terminal() {
            return (simplified, None);
        }
        match normalizer.stream_ended() {
            Ok(()) => (simplified, None),
            Err(failure) => (simplified, Some(failure)),
        }
    }

    fn event(name: &str, payload: &Value) -> Vec<u8> {
        encode_message(
            &[(":message-type", "event"), (":event-type", name)],
            payload.to_string().as_bytes(),
        )
    }

    fn exception(name: &str) -> Vec<u8> {
        encode_message(
            &[(":message-type", "exception"), (":exception-type", name)],
            br#"{"message":"redacted"}"#,
        )
    }

    #[test]
    fn bedrock_golden_stream_normalizes_text_tools_usage_and_completion() {
        // Golden fixture: raw provider bytes in, exact canonical events out.
        // `native_dialect_parity_test.py` holds the python-mapper comparison.
        let chunks = vec![
            event("messageStart", &json!({"role": "assistant"})),
            event(
                "contentBlockDelta",
                &json!({"contentBlockIndex": 0, "delta": {"text": "Hel"}}),
            ),
            event(
                "contentBlockDelta",
                &json!({"contentBlockIndex": 0, "delta": {"text": "lo"}}),
            ),
            event("contentBlockStop", &json!({"contentBlockIndex": 0})),
            event(
                "contentBlockStart",
                &json!({
                    "contentBlockIndex": 1,
                    "start": {"toolUse": {"toolUseId": "call-1", "name": "lookup"}},
                }),
            ),
            event(
                "contentBlockDelta",
                &json!({
                    "contentBlockIndex": 1,
                    "delta": {"toolUse": {"input": "{\"city\":"}},
                }),
            ),
            event(
                "contentBlockDelta",
                &json!({
                    "contentBlockIndex": 1,
                    "delta": {"toolUse": {"input": "\"Zürich\"}"}},
                }),
            ),
            event("contentBlockStop", &json!({"contentBlockIndex": 1})),
            event("messageStop", &json!({"stopReason": "tool_use"})),
            event(
                "metadata",
                &json!({
                    "usage": {
                        "inputTokens": 9,
                        "outputTokens": 4,
                        "cacheReadInputTokens": 2,
                        "cacheWriteInputTokens": 1,
                    },
                    "metrics": {"latencyMs": 12},
                }),
            ),
        ];
        let (events, failure) = run_stream(&chunks);
        assert!(failure.is_none());
        assert_eq!(
            events,
            vec![
                json!({"kind": "text_delta", "text": "Hel"}),
                json!({"kind": "text_delta", "text": "lo"}),
                json!({"kind": "tool_call_started", "index": 1, "call_id": "call-1", "name": "lookup"}),
                json!({"kind": "tool_arguments_delta", "index": 1, "text": "{\"city\":"}),
                json!({"kind": "tool_arguments_delta", "index": 1, "text": "\"Zürich\"}"}),
                json!({
                    "kind": "tool_call_completed",
                    "index": 1,
                    "call_id": "call-1",
                    "name": "lookup",
                    "raw_arguments": "{\"city\":\"Zürich\"}",
                }),
                json!({
                    "kind": "usage",
                    "input_tokens": 12,
                    "output_tokens": 4,
                    "cached_input_tokens": 2,
                    "reasoning_tokens": null,
                }),
                json!({"kind": "completed"}),
            ]
        );
    }

    #[test]
    fn bedrock_stop_reasons_map_to_the_python_terminal_table() {
        for (reason, expected) in [
            ("end_turn", json!({"kind": "completed"})),
            ("stop_sequence", json!({"kind": "completed"})),
            ("max_tokens", json!({"kind": "incomplete"})),
            (
                "model_context_window_exceeded",
                json!({"kind": "incomplete"}),
            ),
            (
                "guardrail_intervened",
                json!({
                    "kind": "failed",
                    "failure_class": "refusal",
                    "safe_message": "provider refused the request",
                }),
            ),
            (
                "content_filtered",
                json!({
                    "kind": "failed",
                    "failure_class": "refusal",
                    "safe_message": "provider refused the request",
                }),
            ),
            (
                "surprise",
                json!({
                    "kind": "failed",
                    "failure_class": "provider_internal",
                    "safe_message": "provider ended the stream unexpectedly",
                }),
            ),
        ] {
            let chunks = vec![
                event("messageStop", &json!({"stopReason": reason})),
                event(
                    "metadata",
                    &json!({"usage": {"inputTokens": 1, "outputTokens": 1}}),
                ),
            ];
            let (events, failure) = run_stream(&chunks);
            assert!(failure.is_none());
            assert_eq!(events.len(), 2, "reason {reason}");
            assert_eq!(events[1], expected, "reason {reason}");
        }
    }

    #[test]
    fn bedrock_exception_frames_map_to_python_failure_classes() {
        for (name, class) in [
            ("throttlingException", "throttled"),
            ("modelTimeoutException", "timeout"),
            ("internalServerException", "provider_internal"),
            ("modelStreamErrorException", "provider_internal"),
            ("serviceUnavailableException", "provider_internal"),
            ("validationException", "invalid_request"),
        ] {
            let chunks = vec![exception(name)];
            let (events, failure) = run_stream(&chunks);
            assert!(failure.is_none(), "exception {name}");
            assert_eq!(events.len(), 1, "exception {name}");
            assert_eq!(events[0]["kind"], "failed", "exception {name}");
            assert_eq!(events[0]["failure_class"], class, "exception {name}");
        }
    }

    #[test]
    fn bedrock_incomplete_tool_calls_fail_at_the_terminal() {
        let chunks = vec![
            event(
                "contentBlockStart",
                &json!({
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": "call-1", "name": "lookup"}},
                }),
            ),
            event("messageStop", &json!({"stopReason": "end_turn"})),
            event(
                "metadata",
                &json!({"usage": {"inputTokens": 1, "outputTokens": 1}}),
            ),
        ];
        let (events, failure) = run_stream(&chunks);
        assert!(failure.is_none());
        assert_eq!(events[0]["kind"], "tool_call_started");
        assert_eq!(events[1]["kind"], "usage");
        assert_eq!(events[2]["kind"], "failed");
        assert_eq!(events[2]["failure_class"], "malformed_response");
    }

    #[test]
    fn bedrock_malformed_frames_fail_the_stream() {
        // Arguments before a tool start fail.
        let orphan = vec![event(
            "contentBlockDelta",
            &json!({"contentBlockIndex": 3, "delta": {"toolUse": {"input": "{}"}}}),
        )];
        let (_, failure) = run_stream(&orphan);
        assert_eq!(
            failure.expect("must fail").failure_class,
            FailureClass::MalformedResponse
        );

        // An unsupported event name fails.
        let unknown = vec![event("mysteryEvent", &json!({}))];
        let (_, failure) = run_stream(&unknown);
        assert_eq!(
            failure.expect("must fail").failure_class,
            FailureClass::MalformedResponse
        );

        // A non-text tool input delta fails.
        let bad_input = vec![
            event(
                "contentBlockStart",
                &json!({
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": "c", "name": "n"}},
                }),
            ),
            event(
                "contentBlockDelta",
                &json!({"contentBlockIndex": 0, "delta": {"toolUse": {"input": 4}}}),
            ),
        ];
        let (_, failure) = run_stream(&bad_input);
        assert_eq!(
            failure.expect("must fail").failure_class,
            FailureClass::MalformedResponse
        );

        // A stream that closes after messageStop but before metadata fails:
        // usage never arrived, so the terminal cannot be trusted.
        let unterminated = vec![event("messageStop", &json!({"stopReason": "end_turn"}))];
        let (events, failure) = run_stream(&unterminated);
        assert!(events.is_empty());
        assert_eq!(
            failure.expect("must fail").failure_class,
            FailureClass::MalformedResponse
        );
    }
}
