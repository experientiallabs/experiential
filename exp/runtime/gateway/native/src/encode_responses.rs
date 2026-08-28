//! Public Responses encoding, the Rust mirror of `ResponsesSseEncoder` and
//! the Responses branch of `completed_body`.

use std::collections::{BTreeMap, HashMap};

use serde_json::{json, Value};

use crate::encode::{compact_json, stable_public_id};
use crate::errors::{Failure, PublicError};
use crate::events::{
    CompletedToolCall, Event, ProviderAssistantMessagePhase, ProviderOutputItemKind,
    ProviderOutputItemStatus, Usage,
};

mod aggregate;
mod envelope;
mod provider;

pub use aggregate::completed_responses_body;
pub use envelope::ResponsesEnvelope;

fn invalid_provider_stream(message: &str) -> PublicError {
    PublicError::new(502, "invalid_provider_stream", message, "api_error")
}

/// One accumulated Responses function call with stable item and output indices.
struct ToolState {
    item_id: Option<String>,
    output_index: usize,
    call_id: String,
    name: String,
    arguments: String,
    status: Option<ProviderOutputItemStatus>,
    done: bool,
}

impl ToolState {
    /// The current official Responses function-call item.
    fn item(&self, fallback_status: ProviderOutputItemStatus) -> Value {
        if let Some(item_id) = &self.item_id {
            json!({
                "id": item_id,
                "type": "function_call",
                "call_id": self.call_id,
                "name": self.name,
                "arguments": self.arguments,
                "status": self.status.unwrap_or(fallback_status).as_str(),
            })
        } else {
            json!({
                "type": "function_call",
                "call_id": self.call_id,
                "name": self.name,
                "arguments": self.arguments,
                "status": self.status.unwrap_or(fallback_status).as_str(),
            })
        }
    }
}

/// One accumulated reasoning item with provider-indexed summary parts and
/// an optional opaque encrypted payload the caller replays verbatim.
struct ReasoningState {
    item_id: String,
    output_index: usize,
    parts: BTreeMap<u32, String>,
    encrypted_content: Option<String>,
    status: Option<ProviderOutputItemStatus>,
    done: bool,
}

impl ReasoningState {
    fn item(
        &self,
        include_content: bool,
        fallback_status: ProviderOutputItemStatus,
        include_encrypted_content: bool,
    ) -> Value {
        let summary: Vec<Value> = if include_content {
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
            "status": self.status.unwrap_or(fallback_status).as_str(),
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
    item_id: Option<String>,
    kind: ProviderOutputItemKind,
    output_index: usize,
    status: Option<ProviderOutputItemStatus>,
    phase: Option<ProviderAssistantMessagePhase>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum MessageKey {
    Synthetic,
    Provider(u32),
}

/// One independently addressable assistant message output item.
struct MessageState {
    item_id: String,
    output_index: usize,
    status: Option<ProviderOutputItemStatus>,
    phase: Option<ProviderAssistantMessagePhase>,
    text: String,
    refusal: String,
    text_started: bool,
    refusal_started: bool,
    done: bool,
}

impl MessageState {
    fn item(&self, include_content: bool, fallback_status: ProviderOutputItemStatus) -> Value {
        let mut content = Vec::new();
        if include_content && self.text_started {
            content.push(json!({
                "type": "output_text",
                "text": self.text,
                "annotations": [],
            }));
        }
        if include_content && self.refusal_started {
            content.push(json!({"type": "refusal", "refusal": self.refusal}));
        }
        let mut item = json!({
            "id": self.item_id,
            "type": "message",
            "role": "assistant",
            "status": self.status.unwrap_or(fallback_status).as_str(),
            "content": content,
        });
        if let Some(phase) = self.phase {
            item["phase"] = json!(phase.as_str());
        }
        item
    }
}

#[derive(Clone, Copy)]
enum OutputSlot {
    Message(MessageKey),
    Tool(u32),
    Reasoning(u32),
}

/// Incremental Responses lifecycle encoder with one monotonic terminal event,
/// emitting byte-identical frames to the Python `ResponsesSseEncoder`.
pub struct ResponsesSseEncoder {
    response_id: String,
    synthetic_message_id: String,
    model: String,
    created_at: f64,
    envelope: ResponsesEnvelope,
    started: bool,
    terminal: bool,
    sequence: u64,
    output_order: Vec<OutputSlot>,
    tools: HashMap<u32, ToolState>,
    reasoning: HashMap<u32, ReasoningState>,
    messages: HashMap<MessageKey, MessageState>,
    provider_output_starts: HashMap<u32, ProviderOutputStart>,
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
            synthetic_message_id: stable_public_id("msg", request_id),
            model: model.to_string(),
            created_at,
            envelope,
            started: false,
            terminal: false,
            sequence: 0,
            output_order: Vec::new(),
            tools: HashMap::new(),
            reasoning: HashMap::new(),
            messages: HashMap::new(),
            provider_output_starts: HashMap::new(),
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
            Event::TextDelta(delta) => self.content_delta(MessageKey::Synthetic, None, true, delta),
            Event::RefusalDelta(delta) => {
                self.content_delta(MessageKey::Synthetic, None, false, delta)
            }
            Event::ProviderTextDelta {
                output_index,
                item_id,
                delta,
            } => self.content_delta(
                MessageKey::Provider(*output_index),
                Some(item_id),
                true,
                delta,
            ),
            Event::ProviderRefusalDelta {
                output_index,
                item_id,
                delta,
            } => self.content_delta(
                MessageKey::Provider(*output_index),
                Some(item_id),
                false,
                delta,
            ),
            Event::ProviderOutputItemStarted {
                output_index,
                item_id,
                kind,
                status,
                phase,
            } => self.provider_output_item_started(
                *output_index,
                item_id.as_deref(),
                *kind,
                *status,
                *phase,
            ),
            Event::ProviderOutputItemCompleted {
                output_index,
                item_id,
                kind,
                status,
                phase,
            } => self.provider_output_item_completed(
                *output_index,
                item_id.as_deref(),
                *kind,
                *status,
                *phase,
            ),
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
    fn content_delta(
        &mut self,
        key: MessageKey,
        provider_item_id: Option<&str>,
        is_text: bool,
        delta: &str,
    ) -> Result<Vec<String>, PublicError> {
        let mut frames: Vec<String> = Vec::new();
        self.ensure_message(key, provider_item_id, &mut frames)?;
        let content_index = 0;
        let (item_id, output_index, start_part) = {
            let state = self.messages.get_mut(&key).expect("message just ensured");
            if state.done {
                return Err(invalid_provider_stream(
                    "Responses content arrived after message completion.",
                ));
            }
            if is_text && state.refusal_started {
                return Err(invalid_provider_stream(
                    "Responses output cannot mix text and refusal deltas.",
                ));
            }
            if !is_text && state.text_started {
                return Err(invalid_provider_stream(
                    "Responses output cannot mix text and refusal deltas.",
                ));
            }
            let start_part = if is_text {
                let start = !state.text_started;
                state.text_started = true;
                state.text.push_str(delta);
                start
            } else {
                let start = !state.refusal_started;
                state.refusal_started = true;
                state.refusal.push_str(delta);
                start
            };
            (state.item_id.clone(), state.output_index, start_part)
        };
        if is_text {
            if start_part {
                frames.push(self.event(
                    "response.content_part.added",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": content_index,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    }),
                ));
            }
            frames.push(self.event(
                "response.output_text.delta",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "delta": delta,
                    "logprobs": [],
                }),
            ));
        } else {
            if start_part {
                frames.push(self.event(
                    "response.content_part.added",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": content_index,
                        "part": {"type": "refusal", "refusal": ""},
                    }),
                ));
            }
            frames.push(self.event(
                "response.refusal.delta",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "delta": delta,
                }),
            ));
        }
        Ok(frames)
    }

    /// Create one stable assistant output item before its first content part.
    fn ensure_message(
        &mut self,
        key: MessageKey,
        provider_item_id: Option<&str>,
        frames: &mut Vec<String>,
    ) -> Result<(), PublicError> {
        if let Some(state) = self.messages.get(&key) {
            if provider_item_id.is_none_or(|item_id| state.item_id == item_id) {
                return Ok(());
            }
            return Err(invalid_provider_stream(
                "Responses message item changed provider identity.",
            ));
        }
        let index = self.output_order.len();
        let item_id = match key {
            MessageKey::Synthetic => self.synthetic_message_id.clone(),
            MessageKey::Provider(_) => provider_item_id
                .ok_or_else(|| {
                    invalid_provider_stream("Responses provider message omitted its item ID.")
                })?
                .to_string(),
        };
        let state = MessageState {
            item_id,
            output_index: index,
            status: None,
            phase: None,
            text: String::new(),
            refusal: String::new(),
            text_started: false,
            refusal_started: false,
            done: false,
        };
        let item = state.item(false, ProviderOutputItemStatus::InProgress);
        self.messages.insert(key, state);
        self.output_order.push(OutputSlot::Message(key));
        frames.push(self.event(
            "response.output_item.added",
            json!({
                "output_index": index,
                "item": item,
            }),
        ));
        Ok(())
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
        let (item_id, output_index, status, already_reserved) = match reserved {
            Some(start) => (
                start.item_id.clone(),
                start.output_index,
                start.status,
                true,
            ),
            None => (
                Some(stable_public_id(
                    "fc",
                    &format!("{}:{}", self.response_id, call_id),
                )),
                self.output_order.len(),
                None,
                false,
            ),
        };
        let state = ToolState {
            item_id,
            output_index,
            call_id: call_id.to_string(),
            name: name.to_string(),
            arguments: String::new(),
            status,
            done: false,
        };
        let frame = self.event(
            "response.output_item.added",
            json!({
                "output_index": state.output_index,
                "item": state.item(ProviderOutputItemStatus::InProgress),
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
        let Some(item_id) = item_id else {
            // Official OpenAI 3.x function-argument stream events require an
            // item ID. ID-less calls remain valid and surface their complete
            // arguments on response.output_item.done instead.
            return Ok(Vec::new());
        };
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
            || (provider_owned_identity && call.provider_item_id != state.item_id)
        {
            return Err(invalid_provider_stream(
                "Responses tool completion changed streamed identity or bytes.",
            ));
        }
        if let Some(status) = call.provider_status {
            if state.status.is_some_and(|existing| existing != status) {
                return Err(invalid_provider_stream(
                    "Responses tool completion changed provider status.",
                ));
            }
            state.status = Some(status);
        }
        state.done = true;
        Ok(self.close_tool(index, ProviderOutputItemStatus::Completed))
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
    fn close_tool(&mut self, index: u32, fallback_status: ProviderOutputItemStatus) -> Vec<String> {
        let (item_id, output_index, arguments, item) = {
            let state = match self.tools.get_mut(&index) {
                Some(state) => state,
                None => return Vec::new(),
            };
            state.done = true;
            if matches!(
                state.status,
                None | Some(ProviderOutputItemStatus::InProgress)
            ) {
                state.status = Some(fallback_status);
            }
            (
                state.item_id.clone(),
                state.output_index,
                state.arguments.clone(),
                state.item(fallback_status),
            )
        };
        let mut frames = Vec::new();
        if let Some(item_id) = item_id {
            frames.push(self.event(
                "response.function_call_arguments.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "arguments": arguments,
                }),
            ));
        }
        frames.push(self.event(
            "response.output_item.done",
            json!({
                "output_index": output_index,
                "item": item,
            }),
        ));
        frames
    }

    /// Close open items and emit exactly one Responses terminal lifecycle event.
    fn finish(&mut self, status: &str, failure: Option<&Failure>) -> Vec<String> {
        let mut frames = Vec::new();
        let fallback_status = if status == "completed" {
            ProviderOutputItemStatus::Completed
        } else {
            ProviderOutputItemStatus::Incomplete
        };
        for slot in self.output_order.clone() {
            match slot {
                OutputSlot::Message(key) if !self.messages[&key].done => {
                    let item_status = if key == MessageKey::Synthetic {
                        ProviderOutputItemStatus::Completed
                    } else {
                        fallback_status
                    };
                    frames.extend(self.close_message(key, item_status));
                }
                OutputSlot::Tool(index) if !self.tools[&index].done => {
                    let item_status = if self.provider_output_starts.contains_key(&index) {
                        fallback_status
                    } else {
                        ProviderOutputItemStatus::Completed
                    };
                    frames.extend(self.close_tool(index, item_status));
                }
                OutputSlot::Reasoning(index) if !self.reasoning[&index].done => {
                    let item_status = if self.provider_output_starts.contains_key(&index) {
                        fallback_status
                    } else {
                        ProviderOutputItemStatus::Completed
                    };
                    frames.extend(self.close_reasoning(index, item_status));
                }
                OutputSlot::Message(_) | OutputSlot::Tool(_) | OutputSlot::Reasoning(_) => {}
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
    fn close_reasoning(
        &mut self,
        provider_output_index: u32,
        fallback_status: ProviderOutputItemStatus,
    ) -> Vec<String> {
        let (item_id, output_index, parts, item) = {
            let state = match self.reasoning.get_mut(&provider_output_index) {
                Some(state) => state,
                None => return Vec::new(),
            };
            if state.done {
                return Vec::new();
            }
            state.done = true;
            if matches!(
                state.status,
                None | Some(ProviderOutputItemStatus::InProgress)
            ) {
                state.status = Some(fallback_status);
            }
            (
                state.item_id.clone(),
                state.output_index,
                state.parts.clone(),
                state.item(
                    true,
                    fallback_status,
                    self.envelope.include_encrypted_reasoning,
                ),
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

    /// Emit content and output completion for one assistant message.
    fn close_message(
        &mut self,
        key: MessageKey,
        fallback_status: ProviderOutputItemStatus,
    ) -> Vec<String> {
        let (item_id, output_index, text, refusal, text_started, refusal_started, item) = {
            let state = match self.messages.get_mut(&key) {
                Some(state) => state,
                None => return Vec::new(),
            };
            if state.done {
                return Vec::new();
            }
            state.done = true;
            if matches!(
                state.status,
                None | Some(ProviderOutputItemStatus::InProgress)
            ) {
                state.status = Some(fallback_status);
            }
            (
                state.item_id.clone(),
                state.output_index,
                state.text.clone(),
                state.refusal.clone(),
                state.text_started,
                state.refusal_started,
                state.item(true, fallback_status),
            )
        };
        let mut frames: Vec<String> = Vec::new();
        let mut content_index = 0;
        if text_started {
            frames.push(self.event(
                "response.output_text.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "text": text,
                    "logprobs": [],
                }),
            ));
            let part = json!({"type": "output_text", "text": text, "annotations": []});
            frames.push(self.event(
                "response.content_part.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": part,
                }),
            ));
            content_index += 1;
        }
        if refusal_started {
            frames.push(self.event(
                "response.refusal.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "refusal": refusal,
                }),
            ));
            let part = json!({"type": "refusal", "refusal": refusal});
            frames.push(self.event(
                "response.content_part.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": part,
                }),
            ));
        }
        frames.push(self.event(
            "response.output_item.done",
            json!({
                "output_index": output_index,
                "item": item,
            }),
        ));
        frames
    }

    /// Build one SDK-readable Responses envelope for the current lifecycle state.
    fn response(&self, status: &str, failure: Option<&Failure>) -> Value {
        let include_content = status != "in_progress";
        let fallback_status = match status {
            "in_progress" => ProviderOutputItemStatus::InProgress,
            "completed" => ProviderOutputItemStatus::Completed,
            _ => ProviderOutputItemStatus::Incomplete,
        };
        let output: Vec<Value> = self
            .output_order
            .iter()
            .map(|slot| match slot {
                OutputSlot::Message(key) => {
                    self.messages[key].item(include_content, fallback_status)
                }
                OutputSlot::Tool(index) => self.tools[index].item(fallback_status),
                OutputSlot::Reasoning(index) => self.reasoning[index].item(
                    include_content,
                    fallback_status,
                    self.envelope.include_encrypted_reasoning,
                ),
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
            "usage": if include_content {
                aggregate::responses_usage(self.usage.as_ref())
            } else {
                Value::Null
            },
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

#[cfg(test)]
mod tests;
