//! Public Responses encoding, the Rust mirror of `ResponsesSseEncoder` and
//! the Responses branch of `completed_body`.

use std::collections::{BTreeMap, HashMap};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::encode::{compact_json, stable_public_id};
use crate::errors::{Failure, PublicError};
use crate::events::{CompletedToolCall, Event, ProviderOutputItemKind, Usage};

mod provider;

fn invalid_provider_stream(message: &str) -> PublicError {
    PublicError::new(502, "invalid_provider_stream", message, "api_error")
}

fn default_true() -> bool {
    true
}

fn default_tool_choice() -> Value {
    Value::String("auto".to_string())
}

fn default_tools() -> Value {
    Value::Array(Vec::new())
}

fn default_reasoning() -> Value {
    json!({"effort": Value::Null, "summary": Value::Null})
}

/// Request-reflecting envelope fields built by the control plane from the
/// canonical execution request, embedded verbatim in every response object.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesEnvelope {
    #[serde(default)]
    pub metadata: Value,
    #[serde(default = "default_true")]
    pub parallel_tool_calls: bool,
    #[serde(default)]
    pub temperature: Value,
    #[serde(default)]
    pub top_p: Value,
    #[serde(default = "default_reasoning")]
    pub reasoning: Value,
    #[serde(default)]
    pub ignored_parameters: Vec<String>,
    #[serde(default = "default_tool_choice")]
    pub tool_choice: Value,
    #[serde(default = "default_tools")]
    pub tools: Value,
    #[serde(default)]
    pub max_output_tokens: Value,
    #[serde(default)]
    pub previous_response_id: Value,
    #[serde(default)]
    pub include_encrypted_reasoning: bool,
}

impl Default for ResponsesEnvelope {
    fn default() -> Self {
        Self {
            metadata: Value::Null,
            parallel_tool_calls: true,
            temperature: Value::Null,
            top_p: Value::Null,
            reasoning: default_reasoning(),
            ignored_parameters: Vec::new(),
            tool_choice: default_tool_choice(),
            tools: default_tools(),
            max_output_tokens: Value::Null,
            previous_response_id: Value::Null,
            include_encrypted_reasoning: false,
        }
    }
}

/// One accumulated Responses function call with stable item and output indices.
struct ToolState {
    item_id: String,
    output_index: usize,
    call_id: String,
    name: String,
    arguments: String,
    done: bool,
}

impl ToolState {
    /// The current official Responses function-call item.
    fn item(&self, completed: bool) -> Value {
        json!({
            "id": self.item_id,
            "type": "function_call",
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
            "status": if completed { "completed" } else { "in_progress" },
        })
    }
}

/// One accumulated reasoning item with provider-indexed summary parts and
/// an optional opaque encrypted payload the caller replays verbatim.
struct ReasoningState {
    item_id: String,
    output_index: usize,
    parts: BTreeMap<u32, String>,
    encrypted_content: Option<String>,
}

impl ReasoningState {
    fn item(&self, completed: bool, include_encrypted_content: bool) -> Value {
        let summary: Vec<Value> = if completed {
            self.parts
                .values()
                .map(|text| json!({"type": "summary_text", "text": text}))
                .collect()
        } else {
            Vec::new()
        };
        let mut item = json!({
            "id": self.item_id,
            "type": "reasoning",
            "summary": summary,
            "status": if completed { "completed" } else { "in_progress" },
        });
        if include_encrypted_content {
            if let Some(encrypted) = &self.encrypted_content {
                item.as_object_mut()
                    .expect("reasoning item is an object")
                    .insert("encrypted_content".to_string(), json!(encrypted));
            }
        }
        item
    }
}

/// Provider-owned output item reserved before its content-bearing event.
struct ProviderOutputStart {
    item_id: String,
    kind: ProviderOutputItemKind,
    output_index: usize,
}

#[derive(Clone, Copy)]
enum OutputSlot {
    Message,
    Tool(u32),
    Reasoning(u32),
}

/// Incremental Responses lifecycle encoder with one monotonic terminal event,
/// emitting byte-identical frames to the Python `ResponsesSseEncoder`.
pub struct ResponsesSseEncoder {
    response_id: String,
    message_id: String,
    model: String,
    created_at: f64,
    envelope: ResponsesEnvelope,
    started: bool,
    terminal: bool,
    sequence: u64,
    output_order: Vec<OutputSlot>,
    tools: HashMap<u32, ToolState>,
    reasoning: HashMap<u32, ReasoningState>,
    provider_output_starts: HashMap<u32, ProviderOutputStart>,
    message_output_index: Option<usize>,
    text: String,
    refusal: String,
    text_started: bool,
    refusal_started: bool,
    usage: Option<Usage>,
}

impl ResponsesSseEncoder {
    pub fn new(
        request_id: &str,
        model: &str,
        created_at: f64,
        envelope: ResponsesEnvelope,
    ) -> Self {
        Self {
            response_id: stable_public_id("resp", request_id),
            message_id: stable_public_id("msg", request_id),
            model: model.to_string(),
            created_at,
            envelope,
            started: false,
            terminal: false,
            sequence: 0,
            output_order: Vec::new(),
            tools: HashMap::new(),
            reasoning: HashMap::new(),
            provider_output_starts: HashMap::new(),
            message_output_index: None,
            text: String::new(),
            refusal: String::new(),
            text_started: false,
            refusal_started: false,
            usage: None,
        }
    }

    /// Emit required created and in-progress lifecycle events once.
    pub fn start(&mut self) -> Result<Vec<String>, PublicError> {
        if self.started {
            return Err(invalid_provider_stream(
                "Responses stream was started more than once.",
            ));
        }
        self.started = true;
        let created = self.event(
            "response.created",
            json!({"response": self.response("in_progress", None)}),
        );
        let in_progress = self.event(
            "response.in_progress",
            json!({"response": self.response("in_progress", None)}),
        );
        Ok(vec![created, in_progress])
    }

    /// Encode one ordered normalized event into Responses lifecycle frames.
    pub fn feed(&mut self, event: &Event) -> Result<Vec<String>, PublicError> {
        if !self.started {
            return Err(invalid_provider_stream(
                "Responses stream must be started before provider events.",
            ));
        }
        if self.terminal {
            return Err(invalid_provider_stream(
                "Responses stream received an event after its terminal.",
            ));
        }
        match event {
            Event::TextDelta(delta) => self.content_delta(true, delta),
            Event::RefusalDelta(delta) => self.content_delta(false, delta),
            Event::ProviderOutputItemStarted {
                output_index,
                item_id,
                kind,
            } => self.provider_output_item_started(*output_index, item_id, *kind),
            Event::ReasoningSummaryDelta {
                output_index,
                summary_index,
                item_id,
                delta,
            } => self.reasoning_summary_delta(*output_index, *summary_index, item_id, delta),
            // Lossy projection: Anthropic thinking text streams as a summary
            // part so callers receive what they pay for. Signatures and
            // redacted payloads are dropped deliberately, since this surface
            // cannot round-trip them.
            Event::ThinkingDelta { index, delta } => {
                let item_id =
                    stable_public_id("rs", &format!("{}:thinking:{index}", self.response_id));
                self.reasoning_summary_delta(*index, 0, &item_id, delta)
            }
            Event::ThinkingSignature { .. } | Event::RedactedThinking { .. } => Ok(Vec::new()),
            Event::EncryptedReasoning {
                output_index,
                item_id,
                encrypted_content,
            } => self.encrypted_reasoning(*output_index, item_id, encrypted_content),
            Event::ToolCallStarted {
                index,
                call_id,
                name,
            } => self.tool_started(*index, call_id, name),
            Event::ToolArgumentsDelta { index, delta } => self.tool_arguments(*index, delta),
            Event::ToolCallCompleted { index, call } => self.tool_completed(*index, call),
            Event::Usage(usage) => {
                if usage.has_token_counts() {
                    self.usage = Some(usage.clone());
                }
                Ok(Vec::new())
            }
            Event::Completed => Ok(self.finish("completed", None)),
            Event::Incomplete => Ok(self.finish("incomplete", None)),
            Event::Failed(failure) => Ok(self.finish("failed", Some(failure))),
        }
    }

    pub fn saw_terminal(&self) -> bool {
        self.terminal
    }

    /// Start one output message/content part as needed and emit its delta.
    fn content_delta(&mut self, is_text: bool, delta: &str) -> Result<Vec<String>, PublicError> {
        let mut frames: Vec<String> = Vec::new();
        let output_index = self.ensure_message(&mut frames);
        let content_index = 0;
        if is_text {
            if self.refusal_started {
                return Err(invalid_provider_stream(
                    "Responses output cannot mix text and refusal deltas.",
                ));
            }
            if !self.text_started {
                self.text_started = true;
                frames.push(self.event(
                    "response.content_part.added",
                    json!({
                        "item_id": self.message_id,
                        "output_index": output_index,
                        "content_index": content_index,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    }),
                ));
            }
            self.text.push_str(delta);
            frames.push(self.event(
                "response.output_text.delta",
                json!({
                    "item_id": self.message_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "delta": delta,
                    "logprobs": [],
                }),
            ));
        } else {
            if self.text_started {
                return Err(invalid_provider_stream(
                    "Responses output cannot mix text and refusal deltas.",
                ));
            }
            if !self.refusal_started {
                self.refusal_started = true;
                frames.push(self.event(
                    "response.content_part.added",
                    json!({
                        "item_id": self.message_id,
                        "output_index": output_index,
                        "content_index": content_index,
                        "part": {"type": "refusal", "refusal": ""},
                    }),
                ));
            }
            self.refusal.push_str(delta);
            frames.push(self.event(
                "response.refusal.delta",
                json!({
                    "item_id": self.message_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "delta": delta,
                }),
            ));
        }
        Ok(frames)
    }

    /// Create one stable assistant output item before its first content part.
    fn ensure_message(&mut self, frames: &mut Vec<String>) -> usize {
        if let Some(index) = self.message_output_index {
            return index;
        }
        let index = self.output_order.len();
        self.message_output_index = Some(index);
        self.output_order.push(OutputSlot::Message);
        frames.push(self.event(
            "response.output_item.added",
            json!({
                "output_index": index,
                "item": self.message_item(false),
            }),
        ));
        index
    }

    /// Start one reasoning item/summary part as needed and emit its text delta.
    fn reasoning_summary_delta(
        &mut self,
        provider_output_index: u32,
        summary_index: u32,
        item_id: &str,
        delta: &str,
    ) -> Result<Vec<String>, PublicError> {
        let mut frames = Vec::new();
        self.ensure_reasoning(provider_output_index, item_id, &mut frames)?;
        let (item_id, output_index, new_part) = {
            let state = self
                .reasoning
                .get_mut(&provider_output_index)
                .expect("reasoning state just ensured");
            let new_part = !state.parts.contains_key(&summary_index);
            state
                .parts
                .entry(summary_index)
                .or_default()
                .push_str(delta);
            (state.item_id.clone(), state.output_index, new_part)
        };
        if new_part {
            frames.push(self.event(
                "response.reasoning_summary_part.added",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": summary_index,
                    "part": {"type": "summary_text", "text": ""},
                }),
            ));
        }
        frames.push(self.event(
            "response.reasoning_summary_text.delta",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "summary_index": summary_index,
                "delta": delta,
            }),
        ));
        Ok(frames)
    }

    /// Emit one stable function-call output item start.
    fn tool_started(
        &mut self,
        index: u32,
        call_id: &str,
        name: &str,
    ) -> Result<Vec<String>, PublicError> {
        if self.tools.contains_key(&index) {
            return Err(invalid_provider_stream(
                "A Responses tool-call index was started twice.",
            ));
        }
        let reserved = self.provider_output_starts.get(&index);
        if reserved.is_some_and(|start| start.kind != ProviderOutputItemKind::FunctionCall) {
            return Err(invalid_provider_stream(
                "Responses tool call reused a non-tool provider output item.",
            ));
        }
        let (item_id, output_index, already_reserved) = match reserved {
            Some(start) => (start.item_id.clone(), start.output_index, true),
            None => (
                stable_public_id("fc", &format!("{}:{}", self.response_id, call_id)),
                self.output_order.len(),
                false,
            ),
        };
        let state = ToolState {
            item_id,
            output_index,
            call_id: call_id.to_string(),
            name: name.to_string(),
            arguments: String::new(),
            done: false,
        };
        let frame = self.event(
            "response.output_item.added",
            json!({
                "output_index": state.output_index,
                "item": state.item(false),
            }),
        );
        self.tools.insert(index, state);
        if !already_reserved {
            self.output_order.push(OutputSlot::Tool(index));
        }
        Ok(vec![frame])
    }

    /// Append and emit one raw provider-order function argument fragment.
    fn tool_arguments(&mut self, index: u32, delta: &str) -> Result<Vec<String>, PublicError> {
        let state = self.open_tool(index)?;
        state.arguments.push_str(delta);
        let (item_id, output_index) = (state.item_id.clone(), state.output_index);
        Ok(vec![self.event(
            "response.function_call_arguments.delta",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "delta": delta,
            }),
        )])
    }

    /// Verify accumulated raw arguments and emit argument/item completion.
    fn tool_completed(
        &mut self,
        index: u32,
        call: &CompletedToolCall,
    ) -> Result<Vec<String>, PublicError> {
        let provider_owned_identity = self.provider_output_starts.contains_key(&index);
        let state = self.open_tool(index)?;
        if state.call_id != call.call_id
            || state.name != call.name
            || state.arguments != call.raw_arguments
            || (provider_owned_identity
                && call.provider_item_id.as_deref() != Some(state.item_id.as_str()))
        {
            return Err(invalid_provider_stream(
                "Responses tool completion changed streamed identity or bytes.",
            ));
        }
        state.done = true;
        Ok(self.close_tool(index))
    }

    /// Resolve one already-started, still-open tool index.
    fn open_tool(&mut self, index: u32) -> Result<&mut ToolState, PublicError> {
        let state = self.tools.get_mut(&index).ok_or_else(|| {
            invalid_provider_stream("Responses tool event arrived before tool-call start.")
        })?;
        if state.done {
            return Err(invalid_provider_stream(
                "Responses tool event arrived after item completion.",
            ));
        }
        Ok(state)
    }

    /// Emit one function arguments-done and output-item-done pair.
    fn close_tool(&mut self, index: u32) -> Vec<String> {
        let (item_id, output_index, arguments, item) = {
            let state = match self.tools.get_mut(&index) {
                Some(state) => state,
                None => return Vec::new(),
            };
            state.done = true;
            (
                state.item_id.clone(),
                state.output_index,
                state.arguments.clone(),
                state.item(true),
            )
        };
        vec![
            self.event(
                "response.function_call_arguments.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "arguments": arguments,
                }),
            ),
            self.event(
                "response.output_item.done",
                json!({
                    "output_index": output_index,
                    "item": item,
                }),
            ),
        ]
    }

    /// Close open items and emit exactly one Responses terminal lifecycle event.
    fn finish(&mut self, status: &str, failure: Option<&Failure>) -> Vec<String> {
        let mut frames = Vec::new();
        for slot in self.output_order.clone() {
            match slot {
                OutputSlot::Message => frames.extend(self.close_message()),
                OutputSlot::Tool(index) if !self.tools[&index].done => {
                    frames.extend(self.close_tool(index));
                }
                OutputSlot::Reasoning(index) => frames.extend(self.close_reasoning(index)),
                OutputSlot::Tool(_) => {}
            }
        }
        self.terminal = true;
        let event_name = format!("response.{status}");
        let frame = self.event(
            &event_name,
            json!({"response": self.response(status, failure)}),
        );
        frames.push(frame);
        frames
    }

    /// Complete every summary part and its containing reasoning item.
    fn close_reasoning(&mut self, provider_output_index: u32) -> Vec<String> {
        let (item_id, output_index, parts, item) = {
            let state = match self.reasoning.get(&provider_output_index) {
                Some(state) => state,
                None => return Vec::new(),
            };
            (
                state.item_id.clone(),
                state.output_index,
                state.parts.clone(),
                state.item(true, self.envelope.include_encrypted_reasoning),
            )
        };
        let mut frames = Vec::new();
        for (summary_index, text) in parts {
            frames.push(self.event(
                "response.reasoning_summary_text.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": summary_index,
                    "text": text,
                }),
            ));
            frames.push(self.event(
                "response.reasoning_summary_part.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": summary_index,
                    "part": {"type": "summary_text", "text": text},
                }),
            ));
        }
        frames.push(self.event(
            "response.output_item.done",
            json!({"output_index": output_index, "item": item}),
        ));
        frames
    }

    /// Emit content and output completion for the one assistant message.
    fn close_message(&mut self) -> Vec<String> {
        let output_index = match self.message_output_index {
            Some(index) => index,
            None => return Vec::new(),
        };
        let mut frames: Vec<String> = Vec::new();
        let mut content_index = 0;
        if self.text_started {
            let text = self.text.clone();
            frames.push(self.event(
                "response.output_text.done",
                json!({
                    "item_id": self.message_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "text": text,
                    "logprobs": [],
                }),
            ));
            let part = json!({"type": "output_text", "text": self.text, "annotations": []});
            frames.push(self.event(
                "response.content_part.done",
                json!({
                    "item_id": self.message_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": part,
                }),
            ));
            content_index += 1;
        }
        if self.refusal_started {
            let refusal = self.refusal.clone();
            frames.push(self.event(
                "response.refusal.done",
                json!({
                    "item_id": self.message_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "refusal": refusal,
                }),
            ));
            let part = json!({"type": "refusal", "refusal": self.refusal});
            frames.push(self.event(
                "response.content_part.done",
                json!({
                    "item_id": self.message_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": part,
                }),
            ));
        }
        let item = self.message_item(true);
        frames.push(self.event(
            "response.output_item.done",
            json!({
                "output_index": output_index,
                "item": item,
            }),
        ));
        frames
    }

    /// The current official Responses assistant-message item.
    fn message_item(&self, completed: bool) -> Value {
        let mut content: Vec<Value> = Vec::new();
        if completed && self.text_started {
            content.push(json!({"type": "output_text", "text": self.text, "annotations": []}));
        }
        if completed && self.refusal_started {
            content.push(json!({"type": "refusal", "refusal": self.refusal}));
        }
        json!({
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "status": if completed { "completed" } else { "in_progress" },
            "content": content,
        })
    }

    /// Build one SDK-readable Responses envelope for the current lifecycle state.
    fn response(&self, status: &str, failure: Option<&Failure>) -> Value {
        let completed = status != "in_progress";
        let output: Vec<Value> =
            self.output_order
                .iter()
                .map(|slot| match slot {
                    OutputSlot::Message => self.message_item(completed),
                    OutputSlot::Tool(index) => self.tools[index].item(completed),
                    OutputSlot::Reasoning(index) => self.reasoning[index]
                        .item(completed, self.envelope.include_encrypted_reasoning),
                })
                .collect();
        let error = if status == "failed" {
            json!({
                "code": "server_error",
                "message": failure
                    .map(|failure| failure.safe_message.clone())
                    .unwrap_or_else(|| "Gateway stream failed.".to_string()),
            })
        } else {
            Value::Null
        };
        let mut response = json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "completed_at": if status == "completed" { json!(self.created_at) } else { Value::Null },
            "status": status,
            "error": error,
            "incomplete_details": if status == "incomplete" {
                json!({"reason": "max_output_tokens"})
            } else {
                Value::Null
            },
            "instructions": Value::Null,
            "metadata": self.envelope.metadata,
            "model": self.model,
            "output": output,
            "parallel_tool_calls": self.envelope.parallel_tool_calls,
            "temperature": self.envelope.temperature,
            "top_p": self.envelope.top_p,
            "reasoning": self.envelope.reasoning,
            "tool_choice": self.envelope.tool_choice,
            "tools": self.envelope.tools,
            "max_output_tokens": self.envelope.max_output_tokens,
            "previous_response_id": self.envelope.previous_response_id,
            "usage": if completed { responses_usage(self.usage.as_ref()) } else { Value::Null },
        });
        if !self.envelope.ignored_parameters.is_empty() {
            response
                .as_object_mut()
                .expect("response envelope is an object")
                .insert(
                    "x-experiential-ignored-parameters".to_string(),
                    json!(self.envelope.ignored_parameters),
                );
        }
        response
    }

    /// Assign one monotonic sequence number and frame a named SSE event.
    fn event(&mut self, event_type: &str, fields: Value) -> String {
        let mut payload = serde_json::Map::new();
        payload.insert("type".to_string(), Value::String(event_type.to_string()));
        payload.insert("sequence_number".to_string(), json!(self.sequence));
        if let Value::Object(entries) = fields {
            for (key, value) in entries {
                payload.insert(key, value);
            }
        }
        self.sequence += 1;
        let encoded = compact_json(&Value::Object(payload));
        format!("event: {event_type}\ndata: {encoded}\n\n")
    }
}

/// Usage shape from `exp.runtime.openai_protocol.streaming._responses_usage`.
fn responses_usage(usage: Option<&Usage>) -> Value {
    let usage = match usage {
        Some(usage) if usage.has_token_counts() => usage,
        _ => return Value::Null,
    };
    let input = usage.input_tokens.unwrap_or(0);
    let output = usage.output_tokens.unwrap_or(0);
    json!({
        "input_tokens": input,
        "input_tokens_details": {"cached_tokens": usage.cached_input_tokens.unwrap_or(0)},
        "output_tokens": output,
        "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens.unwrap_or(0)},
        "total_tokens": input + output,
    })
}

/// The aggregated non-streaming Responses outcome from one event stream.
pub struct AggregatedResponses {
    pub body: Value,
    pub failure: Option<Failure>,
    pub usage: Option<Usage>,
    pub incomplete: bool,
    pub tool_names: Vec<String>,
}

/// Build one non-streaming public Responses result from ordered events,
/// mirroring the Responses branch of `completed_body`.
pub fn completed_responses_body(
    request_id: &str,
    model: &str,
    created_at: f64,
    envelope: ResponsesEnvelope,
    events: &[Event],
) -> Result<AggregatedResponses, PublicError> {
    let terminal = events.iter().rev().find(|event| event.is_terminal());
    let terminal = match terminal {
        Some(event) => event,
        None => {
            return Err(PublicError::new(
                502,
                "all_routes_failed",
                "Provider stream ended without a terminal result.",
                "api_error",
            ))
        }
    };
    let mut usage: Option<Usage> = None;
    for event in events.iter().rev() {
        if let Event::Usage(candidate) = event {
            if candidate.has_token_counts() {
                usage = Some(candidate.clone());
                break;
            }
        }
    }
    let mut tool_names: Vec<String> = Vec::new();
    for event in events {
        if let Event::ToolCallCompleted { call, .. } = event {
            if !tool_names.contains(&call.name) {
                tool_names.push(call.name.clone());
            }
        }
    }
    if let Event::Failed(failure) = terminal {
        return Ok(AggregatedResponses {
            body: Value::Null,
            failure: Some(failure.clone()),
            usage,
            incomplete: false,
            tool_names,
        });
    }
    let mut encoder = ResponsesSseEncoder::new(request_id, model, created_at, envelope);
    encoder.start()?;
    let mut terminal_frames: Vec<String> = Vec::new();
    for event in events {
        let produced = encoder.feed(event)?;
        if event.is_terminal() {
            terminal_frames = produced;
        }
        if encoder.saw_terminal() {
            break;
        }
    }
    let last = terminal_frames.last().ok_or_else(|| {
        PublicError::new(
            502,
            "all_routes_failed",
            "Responses encoding produced no terminal result.",
            "api_error",
        )
    })?;
    let data = last
        .split_once("data: ")
        .map(|(_, tail)| tail)
        .unwrap_or_default();
    let payload: Value =
        serde_json::from_str(data.trim_end()).map_err(|_| PublicError::internal())?;
    let body = payload
        .get("response")
        .cloned()
        .ok_or_else(PublicError::internal)?;
    Ok(AggregatedResponses {
        body,
        failure: None,
        usage,
        incomplete: matches!(terminal, Event::Incomplete),
        tool_names,
    })
}

#[cfg(test)]
mod tests;
