//! Public Anthropic Messages encoding, the Rust mirror of
//! `exp.runtime.anthropic_protocol.encoding` (`MessagesSseEncoder` and
//! `completed_messages_body`) and of the Anthropic error envelope in
//! `exp.runtime.anthropic_protocol.errors`.

use std::collections::{HashMap, HashSet};

use serde_json::{json, Map, Value};

use crate::dialects::MAXIMUM_RETAINED_OUTPUT_BYTES;
use crate::encode::{compact_json, stable_public_id};
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{Event, Usage};

const REFUSAL_MESSAGE: &str = "provider refused the request";

/// The sanitized failure for provider refusals on this surface, mirroring
/// `refusal_failure` in the python encoder.
pub fn refusal_failure() -> Failure {
    Failure::new(FailureClass::Refusal, REFUSAL_MESSAGE)
}

/// Render one sanitized public error as the Anthropic error envelope,
/// mirroring `anthropic_error_body`: status decides the Anthropic type
/// first, then the OpenAI envelope type, and a present `param` pointer is
/// folded into the message text.
pub fn anthropic_error_body(error: &PublicError) -> Value {
    let error_type = match error.status_code {
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 => "overloaded_error",
        _ if error.error_type == "invalid_request_error" => "invalid_request_error",
        _ => "api_error",
    };
    let message = match &error.param {
        Some(param) if !param.is_empty() => format!("{} (param: {param})", error.message),
        _ => error.message.clone(),
    };
    json!({
        "type": "error",
        "error": {"type": error_type, "message": message},
    })
}

fn invalid_provider_stream(message: &str) -> PublicError {
    PublicError::new(502, "invalid_provider_stream", message, "api_error")
}

/// Map the terminal outcome to the Anthropic stop reason.
fn stop_reason(incomplete: bool, saw_tool_use: bool) -> &'static str {
    if incomplete {
        "max_tokens"
    } else if saw_tool_use {
        "tool_use"
    } else {
        "end_turn"
    }
}

/// The Anthropic usage shape from `messages_usage`: cached reads come back
/// out of the normalized input total, and unknown usage reports zero counts
/// because the Anthropic shape requires both fields.
fn messages_usage(usage: Option<&Usage>) -> Value {
    let usage = match usage {
        Some(usage) if usage.has_token_counts() => usage,
        _ => return json!({"input_tokens": 0, "output_tokens": 0}),
    };
    let cached = usage.cached_input_tokens.unwrap_or(0);
    let mut body = Map::new();
    body.insert(
        "input_tokens".to_string(),
        json!(usage.input_tokens.unwrap_or(0).saturating_sub(cached)),
    );
    body.insert(
        "output_tokens".to_string(),
        json!(usage.output_tokens.unwrap_or(0)),
    );
    if cached > 0 {
        body.insert("cache_read_input_tokens".to_string(), json!(cached));
    }
    Value::Object(body)
}

/// Frame one named, compact, UTF-8-preserving Anthropic SSE event.
fn event_frame(name: &str, payload: &Value) -> String {
    format!("event: {name}\ndata: {}\n\n", compact_json(payload))
}

/// Frame one terminal Anthropic `error` SSE event for a sanitized failure.
fn error_frame(failure: &Failure) -> String {
    event_frame("error", &anthropic_error_body(&failure.public_error()))
}

/// The block families one Messages stream schedules, with their provider
/// grouping index where content arrives incrementally.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Text,
    Tool(u32),
    Thinking(u32),
    Redacted,
}

/// One scheduled content block, buffered until it can stream in order.
///
/// Anthropic SSE streams content blocks strictly sequentially and a closed
/// index cannot reopen, while the canonical stream may legally interleave
/// parallel tool calls and trailing text. Blocks are therefore scheduled in
/// start order: the earliest block streams live, later blocks accumulate
/// their content in `pending` until every earlier block has closed.
struct PendingBlock {
    kind: BlockKind,
    pending: String,
    /// Thinking only: the opaque signature flushes as one `signature_delta`
    /// immediately before the block closes.
    pending_signature: String,
    /// Redacted only: the whole opaque payload travels in the start frame.
    redacted_data: Option<String>,
    anthropic_index: Option<u32>,
}

impl PendingBlock {
    fn new(kind: BlockKind) -> Self {
        Self {
            kind,
            pending: String::new(),
            pending_signature: String::new(),
            redacted_data: None,
            anthropic_index: None,
        }
    }
}

/// Stateful Anthropic Messages SSE encoder with one open block and one
/// terminal, emitting byte-identical frames to the python encoder.
///
/// Blocks are scheduled in start order (see [`PendingBlock`]): the earliest
/// block streams live while content for later blocks buffers within the
/// gateway's bounded retained-output budget, so interleaved parallel tool
/// calls and deferred completions encode as a valid strictly sequential
/// Anthropic lifecycle with the same block order as the non-streaming
/// aggregation.
pub struct MessagesSseEncoder {
    message_id: String,
    model: String,
    started: bool,
    terminal: bool,
    draining: bool,
    next_block_index: u32,
    blocks: Vec<PendingBlock>,
    open_position: Option<usize>,
    next_unopened: usize,
    buffered_bytes: usize,
    tool_identities: HashMap<u32, (String, String)>,
    tool_arguments: HashMap<u32, String>,
    tool_completed: HashSet<u32>,
    saw_tool_use: bool,
    refusal_seen: bool,
    usage: Option<Usage>,
}

impl MessagesSseEncoder {
    pub fn new(request_id: &str, model: &str) -> Self {
        Self {
            message_id: stable_public_id("msg", request_id),
            model: model.to_string(),
            started: false,
            terminal: false,
            draining: false,
            next_block_index: 0,
            blocks: Vec::new(),
            open_position: None,
            next_unopened: 0,
            buffered_bytes: 0,
            tool_identities: HashMap::new(),
            tool_arguments: HashMap::new(),
            tool_completed: HashSet::new(),
            saw_tool_use: false,
            refusal_seen: false,
            usage: None,
        }
    }

    /// Emit the `message_start` and `ping` lifecycle events once.
    pub fn start(&mut self) -> Result<Vec<String>, PublicError> {
        if self.started {
            return Err(invalid_provider_stream(
                "Messages stream was started more than once.",
            ));
        }
        self.started = true;
        let message = json!({
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [],
            "stop_reason": Value::Null,
            "stop_sequence": Value::Null,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        });
        Ok(vec![
            event_frame(
                "message_start",
                &json!({"type": "message_start", "message": message}),
            ),
            event_frame("ping", &json!({"type": "ping"})),
        ])
    }

    pub fn saw_terminal(&self) -> bool {
        self.terminal
    }

    /// Encode one ordered normalized provider event into zero or more frames.
    pub fn feed(&mut self, event: &Event) -> Result<Vec<String>, PublicError> {
        if !self.started {
            return Err(invalid_provider_stream(
                "Messages stream must be started before provider events.",
            ));
        }
        if self.terminal {
            return Err(invalid_provider_stream(
                "Messages stream received an event after its terminal.",
            ));
        }
        match event {
            Event::TextDelta(text) => self.text_delta(text),
            Event::RefusalDelta(_) => {
                // There is no Anthropic refusal block; the refusal is
                // reported as one sanitized terminal error instead.
                self.refusal_seen = true;
                Ok(Vec::new())
            }
            // OpenAI-only reasoning shapes have no Messages representation.
            Event::ProviderOutputItemStarted { .. }
            | Event::ReasoningSummaryDelta { .. }
            | Event::EncryptedReasoning { .. } => Ok(Vec::new()),
            Event::ThinkingDelta { index, delta } => self.thinking_delta(*index, delta),
            Event::ThinkingSignature { index, signature } => {
                self.thinking_signature(*index, signature)
            }
            Event::RedactedThinking { data, .. } => self.redacted_thinking(data),
            Event::ToolCallStarted {
                index,
                call_id,
                name,
            } => self.tool_started(*index, call_id, name),
            Event::ToolArgumentsDelta { index, delta } => self.tool_arguments_delta(*index, delta),
            Event::ToolCallCompleted { index, call } => {
                // Some upstream dialects (OpenAI-compatible streams) emit
                // every tool completion only at their terminal sentinel, and
                // parallel tool calls may interleave, so completion verifies
                // against the accumulated state and the scheduler closes the
                // block once it is the open one.
                let identity = self.tool_identities.get(index);
                if identity.is_none() || self.tool_completed.contains(index) {
                    return Err(invalid_provider_stream(
                        "Messages tool completion omitted its started tool call.",
                    ));
                }
                let streamed = self.tool_arguments.get(index).cloned().unwrap_or_default();
                if identity != Some(&(call.call_id.clone(), call.name.clone()))
                    || streamed != call.raw_arguments
                {
                    return Err(invalid_provider_stream(
                        "Messages tool completion changed streamed identity or bytes.",
                    ));
                }
                self.tool_completed.insert(*index);
                Ok(self.advance())
            }
            Event::Usage(usage) => {
                if usage.has_token_counts() {
                    self.usage = Some(usage.clone());
                }
                Ok(Vec::new())
            }
            Event::Completed | Event::Incomplete => {
                self.terminal = true;
                if self.refusal_seen {
                    return Ok(vec![error_frame(&refusal_failure())]);
                }
                self.draining = true;
                let mut frames = self.advance();
                frames.push(event_frame(
                    "message_delta",
                    &json!({
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": stop_reason(
                                matches!(event, Event::Incomplete),
                                self.saw_tool_use,
                            ),
                            "stop_sequence": Value::Null,
                        },
                        "usage": messages_usage(self.usage.as_ref()),
                    }),
                ));
                frames.push(event_frame(
                    "message_stop",
                    &json!({"type": "message_stop"}),
                ));
                Ok(frames)
            }
            Event::Failed(failure) => {
                self.terminal = true;
                Ok(vec![error_frame(failure)])
            }
        }
    }

    /// Schedule one text delta on the last text block, buffering as needed.
    fn text_delta(&mut self, delta: &str) -> Result<Vec<String>, PublicError> {
        let needs_new_block = match self.blocks.last() {
            Some(block) => block.kind != BlockKind::Text,
            None => true,
        };
        if needs_new_block {
            self.blocks.push(PendingBlock::new(BlockKind::Text));
        }
        let position = self.blocks.len() - 1;
        self.buffer(position, delta)?;
        let mut frames = self.advance();
        self.flush_open(&mut frames);
        Ok(frames)
    }

    /// Schedule one thinking text delta on its provider-indexed block.
    fn thinking_delta(&mut self, index: u32, delta: &str) -> Result<Vec<String>, PublicError> {
        let position = self.thinking_position(index);
        self.buffer(position, delta)?;
        let mut frames = self.advance();
        self.flush_open(&mut frames);
        Ok(frames)
    }

    /// Retain one opaque signature fragment; it flushes at block close.
    fn thinking_signature(
        &mut self,
        index: u32,
        signature: &str,
    ) -> Result<Vec<String>, PublicError> {
        let position = self.thinking_position(index);
        self.buffered_bytes = self.buffered_bytes.saturating_add(signature.len());
        if self.buffered_bytes > MAXIMUM_RETAINED_OUTPUT_BYTES {
            return Err(invalid_provider_stream(
                "Messages stream buffered blocks exceeded the gateway response limit.",
            ));
        }
        self.blocks[position].pending_signature.push_str(signature);
        Ok(self.advance())
    }

    /// Schedule one complete redacted-thinking block at its arrival position.
    fn redacted_thinking(&mut self, data: &str) -> Result<Vec<String>, PublicError> {
        self.buffered_bytes = self.buffered_bytes.saturating_add(data.len());
        if self.buffered_bytes > MAXIMUM_RETAINED_OUTPUT_BYTES {
            return Err(invalid_provider_stream(
                "Messages stream buffered blocks exceeded the gateway response limit.",
            ));
        }
        let mut block = PendingBlock::new(BlockKind::Redacted);
        block.redacted_data = Some(data.to_string());
        self.blocks.push(block);
        Ok(self.advance())
    }

    /// Find or schedule the thinking block for one provider index.
    fn thinking_position(&mut self, index: u32) -> usize {
        if let Some(position) = self
            .blocks
            .iter()
            .position(|block| block.kind == BlockKind::Thinking(index))
        {
            return position;
        }
        self.blocks
            .push(PendingBlock::new(BlockKind::Thinking(index)));
        self.blocks.len() - 1
    }

    /// Schedule one tool_use block at its start position.
    fn tool_started(
        &mut self,
        tool_index: u32,
        call_id: &str,
        name: &str,
    ) -> Result<Vec<String>, PublicError> {
        if self.tool_identities.contains_key(&tool_index) {
            return Err(invalid_provider_stream(
                "A Messages tool-call index was started twice.",
            ));
        }
        self.tool_identities
            .insert(tool_index, (call_id.to_string(), name.to_string()));
        self.tool_arguments.insert(tool_index, String::new());
        self.saw_tool_use = true;
        self.blocks
            .push(PendingBlock::new(BlockKind::Tool(tool_index)));
        Ok(self.advance())
    }

    /// Schedule one raw argument fragment, buffering behind earlier blocks.
    fn tool_arguments_delta(
        &mut self,
        tool_index: u32,
        delta: &str,
    ) -> Result<Vec<String>, PublicError> {
        if !self.tool_identities.contains_key(&tool_index) {
            return Err(invalid_provider_stream(
                "Messages tool arguments arrived before tool-call start.",
            ));
        }
        if self.tool_completed.contains(&tool_index) {
            return Err(invalid_provider_stream(
                "Messages tool arguments arrived after completion.",
            ));
        }
        self.tool_arguments
            .get_mut(&tool_index)
            .expect("started tool has accumulated arguments")
            .push_str(delta);
        let position = self
            .blocks
            .iter()
            .position(|block| block.kind == BlockKind::Tool(tool_index))
            .expect("started tool has a scheduled block");
        self.buffer(position, delta)?;
        let mut frames = self.advance();
        self.flush_open(&mut frames);
        Ok(frames)
    }

    /// Retain one content fragment for its block within the bounded budget.
    fn buffer(&mut self, position: usize, delta: &str) -> Result<(), PublicError> {
        self.buffered_bytes = self.buffered_bytes.saturating_add(delta.len());
        if self.buffered_bytes > MAXIMUM_RETAINED_OUTPUT_BYTES {
            return Err(invalid_provider_stream(
                "Messages stream buffered blocks exceeded the gateway response limit.",
            ));
        }
        self.blocks[position].pending.push_str(delta);
        Ok(())
    }

    /// Close and open blocks in start order as far as the stream allows.
    ///
    /// A text block closes once a later block exists (or at drain); a tool
    /// block closes only after its verified completion. Opening a block
    /// assigns the next sequential Anthropic index and flushes its buffered
    /// content as one delta.
    fn advance(&mut self) -> Vec<String> {
        let mut frames = Vec::new();
        loop {
            if let Some(position) = self.open_position {
                let block = &self.blocks[position];
                let last = position == self.blocks.len() - 1;
                let closable = self.draining
                    || match block.kind {
                        BlockKind::Text | BlockKind::Thinking(_) | BlockKind::Redacted => !last,
                        BlockKind::Tool(tool_index) => self.tool_completed.contains(&tool_index),
                    };
                if !closable {
                    return frames;
                }
                self.flush_signature(position, &mut frames);
                let block = &self.blocks[position];
                frames.push(event_frame(
                    "content_block_stop",
                    &json!({"type": "content_block_stop", "index": block.anthropic_index}),
                ));
                self.open_position = None;
                continue;
            }
            if self.next_unopened >= self.blocks.len() {
                return frames;
            }
            let position = self.next_unopened;
            self.open_position = Some(position);
            self.next_unopened += 1;
            let anthropic_index = self.next_block_index;
            self.next_block_index += 1;
            let block = &mut self.blocks[position];
            block.anthropic_index = Some(anthropic_index);
            let content_block = match block.kind {
                BlockKind::Text => json!({"type": "text", "text": ""}),
                // The SDK thinking block type requires both fields, so the
                // start frame carries their empty forms like the provider.
                BlockKind::Thinking(_) => {
                    json!({"type": "thinking", "thinking": "", "signature": ""})
                }
                BlockKind::Redacted => {
                    let data = block.redacted_data.take().unwrap_or_default();
                    self.buffered_bytes = self.buffered_bytes.saturating_sub(data.len());
                    json!({"type": "redacted_thinking", "data": data})
                }
                BlockKind::Tool(tool_index) => {
                    let identity = self
                        .tool_identities
                        .get(&tool_index)
                        .expect("scheduled tool has an identity");
                    json!({
                        "type": "tool_use",
                        "id": identity.0,
                        "name": identity.1,
                        "input": {},
                    })
                }
            };
            frames.push(event_frame(
                "content_block_start",
                &json!({
                    "type": "content_block_start",
                    "index": anthropic_index,
                    "content_block": content_block,
                }),
            ));
            self.flush_open(&mut frames);
        }
    }

    /// Emit one closing `signature_delta` for a thinking block, if retained.
    fn flush_signature(&mut self, position: usize, frames: &mut Vec<String>) {
        let block = &mut self.blocks[position];
        if !matches!(block.kind, BlockKind::Thinking(_)) || block.pending_signature.is_empty() {
            return;
        }
        frames.push(event_frame(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": block.anthropic_index,
                "delta": {"type": "signature_delta", "signature": block.pending_signature},
            }),
        ));
        self.buffered_bytes = self
            .buffered_bytes
            .saturating_sub(block.pending_signature.len());
        block.pending_signature.clear();
    }

    /// Emit the open block's buffered content as one delta, if any.
    fn flush_open(&mut self, frames: &mut Vec<String>) {
        let Some(position) = self.open_position else {
            return;
        };
        let block = &mut self.blocks[position];
        if block.pending.is_empty() {
            return;
        }
        let delta = match block.kind {
            BlockKind::Text => json!({"type": "text_delta", "text": block.pending}),
            BlockKind::Thinking(_) => json!({"type": "thinking_delta", "thinking": block.pending}),
            // Redacted blocks carry their payload in the start frame and
            // never buffer deltas.
            BlockKind::Redacted => return,
            BlockKind::Tool(_) => {
                json!({"type": "input_json_delta", "partial_json": block.pending})
            }
        };
        frames.push(event_frame(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": block.anthropic_index,
                "delta": delta,
            }),
        ));
        self.buffered_bytes = self.buffered_bytes.saturating_sub(block.pending.len());
        block.pending.clear();
    }
}

/// The terminal outcome aggregated from one Messages event stream.
pub struct AggregatedMessage {
    pub body: Value,
    pub failure: Option<Failure>,
    pub usage: Option<Usage>,
    pub incomplete: bool,
    pub tool_names: Vec<String>,
}

/// Build one non-streaming Anthropic message from ordered events, mirroring
/// the python `completed_messages_body`. Provider refusal content has no
/// Anthropic message shape, so it aggregates as a sanitized failure.
pub fn completed_messages_body(
    request_id: &str,
    model: &str,
    events: &[Event],
) -> Result<AggregatedMessage, PublicError> {
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
        return Ok(AggregatedMessage {
            body: Value::Null,
            failure: Some(failure.clone()),
            usage,
            incomplete: false,
            tool_names,
        });
    }
    let incomplete = matches!(terminal, Event::Incomplete);
    if events
        .iter()
        .any(|event| matches!(event, Event::RefusalDelta(_)))
    {
        return Ok(AggregatedMessage {
            body: Value::Null,
            failure: Some(refusal_failure()),
            usage,
            incomplete,
            tool_names,
        });
    }
    // Blocks preserve provider order, merging adjacent text deltas, so the
    // non-streaming content sequence equals the streaming block sequence.
    // Tool blocks anchor at their start position: some dialects (OpenAI-
    // compatible streams) emit every tool completion only at their terminal
    // sentinel, after later text.
    let mut slots: Vec<Option<Value>> = Vec::new();
    let mut tool_positions: HashMap<u32, usize> = HashMap::new();
    let mut thinking_positions: HashMap<u32, usize> = HashMap::new();
    let mut saw_tool_use = false;
    // Resolve one thinking slot per provider index, creating the block with
    // the SDK-required empty fields on first use.
    fn thinking_slot<'a>(
        slots: &'a mut Vec<Option<Value>>,
        positions: &mut HashMap<u32, usize>,
        index: u32,
    ) -> &'a mut Value {
        let position = *positions.entry(index).or_insert_with(|| {
            slots.push(Some(
                json!({"type": "thinking", "thinking": "", "signature": ""}),
            ));
            slots.len() - 1
        });
        slots[position].as_mut().expect("thinking slot is filled")
    }
    for event in events {
        match event {
            Event::TextDelta(delta) if !delta.is_empty() => {
                let appended = match slots.last_mut() {
                    Some(Some(block)) if block["type"] == json!("text") => {
                        if let Some(Value::String(text)) = block.get_mut("text") {
                            text.push_str(delta);
                            true
                        } else {
                            false
                        }
                    }
                    _ => false,
                };
                if !appended {
                    slots.push(Some(json!({"type": "text", "text": delta})));
                }
            }
            Event::ThinkingDelta { index, delta } if !delta.is_empty() => {
                let block = thinking_slot(&mut slots, &mut thinking_positions, *index);
                if let Some(Value::String(text)) = block.get_mut("thinking") {
                    text.push_str(delta);
                }
            }
            Event::ThinkingSignature { index, signature } => {
                let block = thinking_slot(&mut slots, &mut thinking_positions, *index);
                if let Some(Value::String(text)) = block.get_mut("signature") {
                    text.push_str(signature);
                }
            }
            Event::RedactedThinking { data, .. } => {
                slots.push(Some(json!({"type": "redacted_thinking", "data": data})));
            }
            Event::ToolCallStarted { index, .. } => {
                tool_positions.insert(*index, slots.len());
                slots.push(None);
            }
            Event::ToolCallCompleted { index, call } => {
                if let Some(position) = tool_positions.get(index) {
                    saw_tool_use = true;
                    // The raw argument text was validated as one JSON object
                    // by the normalizer; preserve_order keeps its key order,
                    // matching the python engine's parsed-object
                    // serialization.
                    let input: Value = serde_json::from_str(&call.raw_arguments)
                        .map_err(|_| PublicError::internal())?;
                    slots[*position] = Some(json!({
                        "type": "tool_use",
                        "id": call.call_id,
                        "name": call.name,
                        "input": input,
                    }));
                }
            }
            _ => {}
        }
    }
    let content: Vec<Value> = slots.into_iter().flatten().collect();
    let body = json!({
        "id": stable_public_id("msg", request_id),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason(incomplete, saw_tool_use),
        "stop_sequence": Value::Null,
        "usage": messages_usage(usage.as_ref()),
    });
    Ok(AggregatedMessage {
        body,
        failure: None,
        usage,
        incomplete,
        tool_names,
    })
}

#[cfg(test)]
mod tests;
