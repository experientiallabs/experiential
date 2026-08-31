//! Public Chat Completions encoding, the Rust mirror of `ChatSseEncoder` and
//! the chat branch of `completed_body`.

use std::collections::{hash_map::Entry, HashMap};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::errors::{Failure, PublicError};
use crate::events::{Event, Usage};

/// Derive one replay-stable public object ID, mirroring `stable_public_id`.
pub fn stable_public_id(prefix: &str, request_id: &str) -> String {
    let digest = Sha256::digest(request_id.as_bytes());
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    format!("{prefix}_{}", &hex[..32])
}

fn invalid_provider_stream(message: &str) -> PublicError {
    PublicError::new(502, "invalid_provider_stream", message, "api_error")
}

const MAXIMUM_REASONING_CONTENT_BYTES: usize = 8 * 1024 * 1024;

/// Hidden reasoning plus exact completed tool identities awaiting AEAD sealing.
pub struct ReasoningCarrierCandidate {
    pub route_sha256: String,
    pub content: String,
    pub assistant_content: Option<String>,
    pub tool_calls: Vec<ReasoningCarrierToolCall>,
}

/// One exact provider-issued tool call bound into the issuing-turn digest.
pub struct ReasoningCarrierToolCall {
    pub call_id: String,
    pub name: String,
    pub raw_arguments: String,
}

#[derive(Default)]
struct ReasoningCarrierState {
    route_sha256: Option<String>,
    content: String,
    assistant_content: String,
    tool_order: Vec<u32>,
    tool_ids: HashMap<u32, (String, String)>,
    completed: HashMap<u32, crate::events::CompletedToolCall>,
}

impl ReasoningCarrierState {
    fn observe(&mut self, event: &Event) -> Result<(), PublicError> {
        match event {
            Event::TextDelta(delta) => self.assistant_content.push_str(delta),
            Event::ReasoningContentDelta {
                route_sha256,
                delta,
            } => {
                if self
                    .route_sha256
                    .as_ref()
                    .is_some_and(|current| current != route_sha256)
                {
                    return Err(invalid_provider_stream(
                        "Chat reasoning content changed provider route.",
                    ));
                }
                if self.content.len().saturating_add(delta.len()) > MAXIMUM_REASONING_CONTENT_BYTES
                {
                    return Err(invalid_provider_stream(
                        "Chat reasoning content exceeded the gateway carrier bound.",
                    ));
                }
                self.route_sha256 = Some(route_sha256.clone());
                self.content.push_str(delta);
            }
            Event::ToolCallStarted {
                index,
                call_id,
                name,
            } => {
                if self.tool_ids.contains_key(index)
                    || self
                        .tool_ids
                        .values()
                        .any(|(existing, _name)| existing == call_id)
                {
                    return Err(invalid_provider_stream(
                        "A Chat tool-call ID was started twice.",
                    ));
                }
                self.tool_ids
                    .insert(*index, (call_id.clone(), name.clone()));
                self.tool_order.push(*index);
            }
            Event::ToolCallCompleted { index, call } => {
                match self.tool_ids.get(index) {
                    Some((call_id, name)) if call_id == &call.call_id && name == &call.name => {}
                    _ => {
                        return Err(invalid_provider_stream(
                            "Chat tool completion changed or duplicated its identity.",
                        ))
                    }
                }
                match self.completed.entry(*index) {
                    Entry::Vacant(entry) => {
                        entry.insert(call.clone());
                    }
                    Entry::Occupied(_) => {
                        return Err(invalid_provider_stream(
                            "Chat tool completion changed or duplicated its identity.",
                        ))
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn candidate(&self) -> Result<Option<ReasoningCarrierCandidate>, PublicError> {
        if self.content.is_empty() || self.tool_ids.is_empty() {
            return Ok(None);
        }
        if self.completed.len() != self.tool_ids.len() {
            return Err(invalid_provider_stream(
                "Chat reasoning content requires complete unique tool calls.",
            ));
        }
        let route_sha256 = self.route_sha256.clone().ok_or_else(|| {
            invalid_provider_stream("Chat reasoning content omitted provider route identity.")
        })?;
        Ok(Some(ReasoningCarrierCandidate {
            route_sha256,
            content: self.content.clone(),
            assistant_content: (!self.assistant_content.is_empty())
                .then(|| self.assistant_content.clone()),
            tool_calls: self
                .tool_order
                .iter()
                .copied()
                .map(|index| {
                    let call = &self.completed[&index];
                    ReasoningCarrierToolCall {
                        call_id: call.call_id.clone(),
                        name: call.name.clone(),
                        raw_arguments: call.raw_arguments.clone(),
                    }
                })
                .collect(),
        }))
    }
}

/// Validate retained events and build a carrier candidate when needed.
pub fn reasoning_carrier_candidate(
    events: &[Event],
) -> Result<Option<ReasoningCarrierCandidate>, PublicError> {
    let mut state = ReasoningCarrierState::default();
    for event in events {
        state.observe(event)?;
    }
    state.candidate()
}

/// Stateful Chat Completions SSE encoder with stable tool indices and one
/// terminal, emitting byte-identical frames to the Python encoder.
pub struct ChatSseEncoder {
    completion_id: String,
    model: String,
    created_at: i64,
    include_usage: bool,
    ignored_parameters: Vec<String>,
    started: bool,
    terminal: bool,
    tool_indices: HashMap<u32, (String, String)>,
    tool_arguments: HashMap<u32, String>,
    usage: Option<Usage>,
    reasoning: ReasoningCarrierState,
    reasoning_content_carrier: Option<String>,
}

impl ChatSseEncoder {
    /// Build an encoder that discloses controls omitted by route shaping.
    pub fn new_with_ignored(
        request_id: &str,
        model: &str,
        created_at: i64,
        include_usage: bool,
        ignored_parameters: Vec<String>,
    ) -> Self {
        Self {
            completion_id: stable_public_id("chatcmpl", request_id),
            model: model.to_string(),
            created_at,
            include_usage,
            ignored_parameters,
            started: false,
            terminal: false,
            tool_indices: HashMap::new(),
            tool_arguments: HashMap::new(),
            usage: None,
            reasoning: ReasoningCarrierState::default(),
            reasoning_content_carrier: None,
        }
    }

    /// Attach an authenticated carrier before the terminal is encoded.
    pub fn set_reasoning_content_carrier(&mut self, carrier: String) {
        self.reasoning_content_carrier = Some(carrier);
    }

    /// Return the validated candidate accumulated by a live stream.
    pub fn reasoning_carrier_candidate(
        &self,
    ) -> Result<Option<ReasoningCarrierCandidate>, PublicError> {
        self.reasoning.candidate()
    }

    /// Emit the single initial assistant-role chunk.
    pub fn start(&mut self) -> Result<Vec<String>, PublicError> {
        if self.started {
            return Err(invalid_provider_stream(
                "Chat stream was started more than once.",
            ));
        }
        self.started = true;
        Ok(vec![self.chunk(json!({"role": "assistant"}), None)])
    }

    /// Encode one ordered normalized provider event into zero or more frames.
    pub fn feed(&mut self, event: &Event) -> Result<Vec<String>, PublicError> {
        if !self.started {
            return Err(invalid_provider_stream(
                "Chat stream must be started before provider events.",
            ));
        }
        if self.terminal {
            return Err(invalid_provider_stream(
                "Chat stream received an event after its terminal.",
            ));
        }
        self.reasoning.observe(event)?;
        match event {
            Event::TextDelta(text) => Ok(vec![self.chunk(json!({"content": text}), None)]),
            Event::RefusalDelta(text) => Ok(vec![self.chunk(json!({"refusal": text}), None)]),
            Event::ProviderTextDelta { delta, .. } => {
                Ok(vec![self.chunk(json!({"content": delta}), None)])
            }
            Event::ProviderRefusalDelta { delta, .. } => {
                Ok(vec![self.chunk(json!({"refusal": delta}), None)])
            }
            Event::ReasoningContentDelta { .. } => Ok(Vec::new()),
            // The Chat wire has no reasoning representation, so provider
            // reasoning follows the summary path and is deliberately dropped.
            Event::ProviderOutputItemStarted { .. }
            | Event::ProviderOutputItemCompleted { .. }
            | Event::ReasoningSummaryDelta { .. }
            | Event::ThinkingDelta { .. }
            | Event::ThinkingSignature { .. }
            | Event::RedactedThinking { .. }
            | Event::EncryptedReasoning { .. } => Ok(Vec::new()),
            // Server tools enter only through a Messages request onto an
            // Anthropic-only route, so a Chat stream can never legally carry
            // one; an arriving block is dropped like the reasoning carriers.
            Event::ServerToolBlock { .. } => Ok(Vec::new()),
            Event::ToolCallStarted {
                index,
                call_id,
                name,
            } => {
                if self.tool_indices.contains_key(index) {
                    return Err(invalid_provider_stream(
                        "A Chat tool-call index was started twice.",
                    ));
                }
                self.tool_indices
                    .insert(*index, (call_id.clone(), name.clone()));
                self.tool_arguments.insert(*index, String::new());
                Ok(vec![self.chunk(
                    json!({
                        "tool_calls": [
                            {
                                "index": index,
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
                            }
                        ]
                    }),
                    None,
                )])
            }
            Event::ToolArgumentsDelta { index, delta } => {
                let accumulated = self.tool_arguments.get_mut(index).ok_or_else(|| {
                    invalid_provider_stream("Chat tool arguments arrived before tool-call start.")
                })?;
                accumulated.push_str(delta);
                Ok(vec![self.chunk(
                    json!({
                        "tool_calls": [
                            {
                                "index": index,
                                "function": {"arguments": delta},
                            }
                        ]
                    }),
                    None,
                )])
            }
            Event::ToolCallCompleted { index, call } => {
                if call.custom {
                    // Custom tools only enter through a Responses request,
                    // which never encodes on the Chat surface.
                    return Err(invalid_provider_stream(
                        "Chat cannot represent a custom tool call.",
                    ));
                }
                let identity = self.tool_indices.get(index).ok_or_else(|| {
                    invalid_provider_stream("Chat tool completion omitted its started tool call.")
                })?;
                let streamed = self.tool_arguments.get(index).cloned().unwrap_or_default();
                if identity != &(call.call_id.clone(), call.name.clone())
                    || streamed != call.raw_arguments
                {
                    return Err(invalid_provider_stream(
                        "Chat tool completion changed streamed identity or bytes.",
                    ));
                }
                Ok(Vec::new())
            }
            Event::Usage(usage) => {
                if usage.has_token_counts() {
                    self.usage = Some(usage.clone());
                }
                Ok(Vec::new())
            }
            // A paused server-tool turn cannot legally reach the Chat
            // surface (server tools enter only through Messages); the
            // defensive mapping is ordinary completion.
            Event::Completed | Event::Incomplete | Event::Paused => {
                self.terminal = true;
                let finish_reason = if matches!(event, Event::Incomplete) {
                    "length"
                } else if !self.tool_indices.is_empty() {
                    "tool_calls"
                } else {
                    "stop"
                };
                let mut frames = Vec::new();
                if matches!(event, Event::Completed) && self.reasoning.candidate()?.is_some() {
                    let carrier = self.reasoning_content_carrier.as_ref().ok_or_else(|| {
                        invalid_provider_stream(
                            "Chat reasoning content was not sealed by the gateway authority.",
                        )
                    })?;
                    frames.push(self.chunk(json!({"reasoning_content": carrier}), None));
                }
                frames.push(self.chunk(json!({}), Some(finish_reason)));
                if self.include_usage {
                    if let Some(usage) = &self.usage {
                        frames.push(self.usage_chunk(usage));
                    }
                }
                frames.push("data: [DONE]\n\n".to_string());
                Ok(frames)
            }
            Event::Failed(failure) => {
                self.terminal = true;
                Ok(vec![
                    chat_data(&failure.public_error().json_body()),
                    "data: [DONE]\n\n".to_string(),
                ])
            }
        }
    }

    pub fn saw_terminal(&self) -> bool {
        self.terminal
    }

    fn chunk(&self, delta: Value, finish_reason: Option<&str>) -> String {
        let mut payload = json!({
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created_at,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                    "logprobs": Value::Null,
                }
            ],
        });
        if !self.ignored_parameters.is_empty() {
            payload
                .as_object_mut()
                .expect("chat chunk is an object")
                .insert(
                    "x-experiential-ignored-parameters".to_string(),
                    json!(self.ignored_parameters),
                );
        }
        chat_data(&payload)
    }

    fn usage_chunk(&self, usage: &Usage) -> String {
        let payload = json!({
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created_at,
            "model": self.model,
            "choices": [],
            "usage": streaming_chat_usage(usage),
        });
        chat_data(&payload)
    }
}

/// Frame one compact UTF-8-preserving Chat SSE data event.
pub fn chat_data(payload: &Value) -> String {
    format!("data: {}\n\n", compact_json(payload))
}

/// Compact JSON with no insignificant whitespace, matching Python's
/// `json.dumps(..., separators=(",", ":"), ensure_ascii=False)`.
pub fn compact_json(payload: &Value) -> String {
    serde_json::to_string(payload).unwrap_or_else(|_| "null".to_string())
}

/// Streaming usage shape from `exp.runtime.openai_protocol.streaming._chat_usage`.
fn streaming_chat_usage(usage: &Usage) -> Value {
    let input = usage.input_tokens.unwrap_or(0);
    let output = usage.output_tokens.unwrap_or(0);
    json!({
        "prompt_tokens": input,
        "completion_tokens": output,
        "total_tokens": input + output,
        "prompt_tokens_details": {"cached_tokens": usage.cached_input_tokens.unwrap_or(0)},
        "completion_tokens_details": {"reasoning_tokens": usage.reasoning_tokens.unwrap_or(0)},
    })
}

/// Non-streaming usage shape from `exp.runtime.openai_protocol.response.chat_usage`.
fn completed_chat_usage(usage: Option<&Usage>) -> Value {
    let usage = match usage {
        Some(usage) if usage.has_token_counts() => usage,
        _ => return Value::Null,
    };
    let input = usage.input_tokens.unwrap_or(0);
    let output = usage.output_tokens.unwrap_or(0);
    let details = match usage.cached_input_tokens {
        Some(cached) => json!({"cached_tokens": cached}),
        None => Value::Null,
    };
    let output_details = match usage.reasoning_tokens {
        Some(reasoning) => json!({"reasoning_tokens": reasoning}),
        None => Value::Null,
    };
    json!({
        "prompt_tokens": input,
        "completion_tokens": output,
        "total_tokens": input + output,
        "prompt_tokens_details": details,
        "completion_tokens_details": output_details,
    })
}

/// The terminal outcome aggregated from one event stream.
pub struct AggregatedCompletion {
    pub body: Value,
    pub failure: Option<Failure>,
    pub usage: Option<Usage>,
    pub incomplete: bool,
    pub tool_names: Vec<String>,
}

/// Build one non-streaming Chat result with ignored-control disclosure.
pub fn completed_chat_body_with_ignored(
    request_id: &str,
    model: &str,
    created_at: i64,
    events: &[Event],
    ignored_parameters: &[String],
) -> Result<AggregatedCompletion, PublicError> {
    completed_chat_body_with_carrier(
        request_id,
        model,
        created_at,
        events,
        ignored_parameters,
        None,
    )
}

/// Build one non-streaming Chat result with an authenticated reasoning carrier.
pub fn completed_chat_body_with_carrier(
    request_id: &str,
    model: &str,
    created_at: i64,
    events: &[Event],
    ignored_parameters: &[String],
    reasoning_content_carrier: Option<&str>,
) -> Result<AggregatedCompletion, PublicError> {
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
        return Ok(AggregatedCompletion {
            body: Value::Null,
            failure: Some(failure.clone()),
            usage,
            incomplete: false,
            tool_names,
        });
    }
    let text: String = events
        .iter()
        .filter_map(|event| match event {
            Event::TextDelta(delta) | Event::ProviderTextDelta { delta, .. } => {
                Some(delta.as_str())
            }
            _ => None,
        })
        .collect();
    let refusal: String = events
        .iter()
        .filter_map(|event| match event {
            Event::RefusalDelta(delta) | Event::ProviderRefusalDelta { delta, .. } => {
                Some(delta.as_str())
            }
            _ => None,
        })
        .collect();
    let tool_calls: Vec<Value> = events
        .iter()
        .filter_map(|event| match event {
            Event::ToolCallCompleted { call, .. } => Some(json!({
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.raw_arguments},
            })),
            _ => None,
        })
        .collect();
    let reasoning = reasoning_carrier_candidate(events)?;
    let incomplete = matches!(terminal, Event::Incomplete);
    let finish_reason = if incomplete {
        "length"
    } else if !tool_calls.is_empty() {
        "tool_calls"
    } else {
        "stop"
    };
    let has_tool_calls = !tool_calls.is_empty();
    let mut message = json!({
        "role": "assistant",
        "content": if text.is_empty() { Value::Null } else { Value::String(text) },
        "refusal": if refusal.is_empty() { Value::Null } else { Value::String(refusal) },
        "tool_calls": if tool_calls.is_empty() { Value::Null } else { Value::Array(tool_calls) },
    });
    if matches!(terminal, Event::Completed) && has_tool_calls && reasoning.is_some() {
        let carrier = reasoning_content_carrier.ok_or_else(|| {
            invalid_provider_stream(
                "Chat reasoning content was not sealed by the gateway authority.",
            )
        })?;
        message
            .as_object_mut()
            .expect("chat message is an object")
            .insert(
                "reasoning_content".to_string(),
                Value::String(carrier.to_string()),
            );
    }
    let mut body = json!({
        "id": stable_public_id("chatcmpl", request_id),
        "object": "chat.completion",
        "created": created_at,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
                "logprobs": Value::Null,
            }
        ],
        "usage": completed_chat_usage(usage.as_ref()),
    });
    if !ignored_parameters.is_empty() {
        body.as_object_mut()
            .expect("chat completion is an object")
            .insert(
                "x-experiential-ignored-parameters".to_string(),
                json!(ignored_parameters),
            );
    }
    Ok(AggregatedCompletion {
        body,
        failure: None,
        usage,
        incomplete,
        tool_names,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fireworks_tool_events() -> Vec<Event> {
        vec![
            Event::ReasoningContentDelta {
                route_sha256: "a".repeat(64),
                delta: "hidden provider reasoning".to_string(),
            },
            Event::ToolCallStarted {
                index: 0,
                call_id: "call-one".to_string(),
                name: "lookup".to_string(),
            },
            Event::ToolArgumentsDelta {
                index: 0,
                delta: "{}".to_string(),
            },
            Event::ToolCallCompleted {
                index: 0,
                call: crate::events::CompletedToolCall {
                    call_id: "call-one".to_string(),
                    name: "lookup".to_string(),
                    raw_arguments: "{}".to_string(),
                    provider_item_id: None,
                    provider_status: None,
                    custom: false,
                },
            },
            Event::Completed,
        ]
    }

    #[test]
    fn fireworks_chat_reasoning_round_trips_only_as_sealed_carrier() {
        let events = fireworks_tool_events();
        let mut stream = ChatSseEncoder::new_with_ignored(
            "request-1",
            "coding",
            1_700_000_000,
            false,
            Vec::new(),
        );
        stream.set_reasoning_content_carrier("authenticated-carrier-v2".to_string());
        let mut frames = stream.start().expect("stream start must encode");
        for event in &events {
            frames.extend(stream.feed(event).expect("event must encode"));
        }
        let public = frames.join("");
        assert!(!public.contains("hidden provider reasoning"));
        assert!(public.contains("authenticated-carrier-v2"));

        let completed = completed_chat_body_with_carrier(
            "request-1",
            "coding",
            1_700_000_000,
            &events,
            &[],
            Some("authenticated-carrier-v2"),
        )
        .expect("completed body must preserve the carrier");
        assert_eq!(
            completed.body["choices"][0]["message"]["reasoning_content"],
            json!("authenticated-carrier-v2")
        );
        assert!(!completed
            .body
            .to_string()
            .contains("hidden provider reasoning"));
    }

    #[test]
    fn fireworks_chat_reasoning_fails_closed_without_carrier_or_unique_completion() {
        let events = fireworks_tool_events();
        assert!(completed_chat_body_with_ignored(
            "request-1",
            "coding",
            1_700_000_000,
            &events,
            &[],
        )
        .is_err());

        let mut duplicate = events[..events.len() - 1].to_vec();
        duplicate.push(events[3].clone());
        duplicate.push(Event::Completed);
        assert!(reasoning_carrier_candidate(&duplicate).is_err());
    }

    #[test]
    fn reasoning_carrier_preserves_provider_tool_start_order() {
        let events = vec![
            Event::ReasoningContentDelta {
                route_sha256: "a".repeat(64),
                delta: "hidden".to_string(),
            },
            Event::ToolCallStarted {
                index: 1,
                call_id: "call-one".to_string(),
                name: "first".to_string(),
            },
            Event::ToolCallStarted {
                index: 0,
                call_id: "call-zero".to_string(),
                name: "second".to_string(),
            },
            Event::ToolCallCompleted {
                index: 0,
                call: crate::events::CompletedToolCall {
                    call_id: "call-zero".to_string(),
                    name: "second".to_string(),
                    raw_arguments: "{\"order\":0}".to_string(),
                    provider_item_id: None,
                    provider_status: None,
                    custom: false,
                },
            },
            Event::ToolCallCompleted {
                index: 1,
                call: crate::events::CompletedToolCall {
                    call_id: "call-one".to_string(),
                    name: "first".to_string(),
                    raw_arguments: "{\"order\":1}".to_string(),
                    provider_item_id: None,
                    provider_status: None,
                    custom: false,
                },
            },
        ];

        let candidate = reasoning_carrier_candidate(&events)
            .expect("provider events must validate")
            .expect("reasoning plus tools must produce a carrier");

        assert_eq!(
            candidate
                .tool_calls
                .iter()
                .map(|call| call.call_id.as_str())
                .collect::<Vec<_>>(),
            vec!["call-one", "call-zero"]
        );
    }

    #[test]
    fn ignored_generation_controls_are_disclosed_by_both_chat_encoders() {
        let ignored = vec!["top_p".to_string(), "reasoning_effort".to_string()];
        let mut stream = ChatSseEncoder::new_with_ignored(
            "request-1",
            "coding",
            1_700_000_000,
            false,
            ignored.clone(),
        );
        let frames = stream.start().expect("stream start must encode");
        assert!(frames[0]
            .contains("\"x-experiential-ignored-parameters\":[\"top_p\",\"reasoning_effort\"]"));

        let completed = completed_chat_body_with_ignored(
            "request-1",
            "coding",
            1_700_000_000,
            &[Event::Completed],
            &ignored,
        )
        .expect("completed body must encode");
        assert_eq!(
            completed.body["x-experiential-ignored-parameters"],
            json!(["top_p", "reasoning_effort"])
        );
    }
}
