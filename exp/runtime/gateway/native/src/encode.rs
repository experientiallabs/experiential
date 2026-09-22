//! Public Chat Completions encoding, the Rust mirror of `ChatSseEncoder` and
//! the chat branch of `completed_body`.

use std::collections::{hash_map::Entry, HashMap};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::errors::{Failure, PublicError};
use crate::events::{ChoiceLogprobs, Event, Usage};
use crate::tool_search::annotate_tool_search_usage_details;
use crate::web_search::{annotate_usage_details, ChatWebSearch, WebSearchAdmission};

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

/// Shared by the Chat and Messages encoders: both surfaces retain the same
/// hidden-reasoning candidate and seal it under the same authority.
#[derive(Default)]
pub(crate) struct ReasoningCarrierState {
    route_sha256: Option<String>,
    content: String,
    assistant_content: String,
    tool_order: Vec<u32>,
    tool_ids: HashMap<u32, (String, String)>,
    completed: HashMap<u32, crate::events::CompletedToolCall>,
}

impl ReasoningCarrierState {
    pub(crate) fn observe(&mut self, event: &Event) -> Result<(), PublicError> {
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
                ..
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

    pub(crate) fn candidate(&self) -> Result<Option<ReasoningCarrierCandidate>, PublicError> {
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
    incomplete_tool_arguments: bool,
    tool_indices: HashMap<u32, (String, String)>,
    tool_arguments: HashMap<u32, String>,
    usage: Option<Usage>,
    reasoning: ReasoningCarrierState,
    reasoning_content_carrier: Option<String>,
    reasoning_output_exposed: bool,
    web_search: Option<ChatWebSearch>,
    /// Gateway-run tool-search rounds metered on the usage chunk; zero
    /// leaves every frame byte-identical.
    tool_search_requests: u32,
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
            incomplete_tool_arguments: false,
            tool_indices: HashMap::new(),
            tool_arguments: HashMap::new(),
            usage: None,
            reasoning: ReasoningCarrierState::default(),
            reasoning_content_carrier: None,
            reasoning_output_exposed: false,
            web_search: None,
            tool_search_requests: 0,
        }
    }

    /// Meter the gateway-run tool-search rounds on the usage chunk; the Chat
    /// surface renders no item for them.
    pub fn set_tool_search_requests(&mut self, requests: u32) {
        self.tool_search_requests = requests;
    }

    /// Attach an authenticated carrier before the terminal is encoded.
    pub fn set_reasoning_content_carrier(&mut self, carrier: String) {
        self.reasoning_content_carrier = Some(carrier);
    }

    /// Cite the gateway-executed web search at the terminal: `None` (no
    /// search) leaves every frame byte-identical.
    pub fn set_web_search(&mut self, web_search: Option<WebSearchAdmission>) {
        self.web_search = web_search.map(ChatWebSearch::new);
    }

    /// Expose the model's plaintext reasoning to the caller on output.
    ///
    /// Off by default so hidden-reasoning providers (OpenAI o-series, which
    /// carry no plaintext reasoning event at all) never leak. Turned on only for
    /// rungs the catalog marks `reasoning_output_exposed`, so a Tencent/DeepSeek
    /// client sees the `reasoning_content` deltas it is already billed for. The
    /// sealed round-trip carrier at the terminal is independent of this.
    pub fn set_reasoning_output_exposed(&mut self, exposed: bool) {
        self.reasoning_output_exposed = exposed;
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
        if let Some(web_search) = self.web_search.as_mut() {
            web_search.observe(event);
        }
        match event {
            Event::ProviderResponsesLogprobs { .. } => Err(invalid_provider_stream(
                "Responses probability events require the Responses encoder",
            )),
            Event::ChoiceLogprobsDelta(update) => Ok(vec![self.chunk_with_logprobs(
                json!({}),
                None,
                update.logprobs.as_ref(),
            )]),
            Event::Image(url) => Ok(vec![self.chunk(
                json!({"images": [crate::image_output::chat_image_value(url)]}),
                None,
            )]),
            Event::TextDelta(text) => Ok(vec![self.chunk(json!({"content": text}), None)]),
            Event::RefusalDelta(text) => Ok(vec![self.chunk(json!({"refusal": text}), None)]),
            Event::ProviderTextDelta { delta, .. } => {
                Ok(vec![self.chunk(json!({"content": delta}), None)])
            }
            Event::ProviderRefusalDelta { delta, .. } => {
                Ok(vec![self.chunk(json!({"refusal": delta}), None)])
            }
            Event::ReasoningContentDelta { delta, .. } => {
                // On an exposure-gated rung (Tencent/DeepSeek), the model's
                // plaintext reasoning is returned to the caller as
                // `choices[].delta.reasoning_content` — the tokens are already
                // billed. Elsewhere it stays dropped (the Chat wire has no
                // reasoning field by default); the sealed round-trip carrier at
                // the terminal is emitted independently regardless.
                if self.reasoning_output_exposed && !delta.is_empty() {
                    Ok(vec![self.chunk(json!({"reasoning_content": delta}), None)])
                } else {
                    Ok(Vec::new())
                }
            }
            // The Chat wire has no reasoning representation, so provider summary
            // and opaque reasoning follow the summary path and are dropped.
            Event::ProviderOutputItemStarted { .. }
            | Event::ProviderOutputItemCompleted { .. }
            | Event::ReasoningSummaryDelta { .. }
            | Event::GeminiThoughtPart(_)
            | Event::ThinkingDelta { .. }
            | Event::ThinkingSignature { .. }
            | Event::RedactedThinking { .. }
            | Event::EncryptedReasoning { .. } => Ok(Vec::new()),
            // Anthropic text-block boundaries and citation metadata have no
            // Chat representation; the text itself streams through TextDelta.
            Event::TextBlockStarted { .. } | Event::CitationDelta { .. } => Ok(Vec::new()),
            // Server tools enter only through a Messages request, which
            // never encodes on the Chat surface.
            Event::ServerToolUseStarted { .. }
            | Event::ServerToolArgumentsDelta { .. }
            | Event::ServerToolUseCompleted { .. }
            | Event::ServerToolResult { .. } => Err(invalid_provider_stream(
                "Chat cannot represent a provider server tool.",
            )),
            // Hosted tool items enter only through Responses-native tool
            // declarations, which never admit on the Chat surface.
            Event::HostedToolItemStarted { .. }
            | Event::HostedToolItemProgress { .. }
            | Event::HostedToolItemCompleted { .. } => Err(invalid_provider_stream(
                "Chat cannot represent a provider-hosted Responses tool item.",
            )),
            // OpenAI text annotations have no Chat representation; the text
            // itself streams through its delta events.
            Event::ProviderTextAnnotation { .. } => Ok(Vec::new()),
            Event::ToolCallStarted {
                index,
                call_id,
                name,
                ..
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
            Event::Completed
            | Event::Incomplete
            | Event::IncompleteToolArguments
            | Event::StoppedAtSequence(_)
            | Event::PausedTurn => {
                self.terminal = true;
                self.incomplete_tool_arguments = matches!(event, Event::IncompleteToolArguments);
                let finish_reason =
                    if matches!(event, Event::Incomplete | Event::IncompleteToolArguments) {
                        "length"
                    } else if !self.tool_indices.is_empty() {
                        "tool_calls"
                    } else {
                        "stop"
                    };
                let mut frames = Vec::new();
                if matches!(event, Event::Completed | Event::StoppedAtSequence(_))
                    && self.reasoning.candidate()?.is_some()
                    && !self.reasoning_output_exposed
                {
                    let carrier = self.reasoning_content_carrier.as_ref().ok_or_else(|| {
                        invalid_provider_stream(
                            "Chat reasoning content was not sealed by the gateway authority.",
                        )
                    })?;
                    frames.push(self.chunk(json!({"reasoning_content": carrier}), None));
                }
                // Search citations ride one annotations delta immediately
                // before the finish chunk, and only when a result URL appears.
                let annotations = self.web_search.as_ref().map(ChatWebSearch::annotations);
                if let Some(annotations) = annotations.filter(|found| !found.is_empty()) {
                    frames.push(self.chunk(json!({"annotations": annotations}), None));
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
        self.chunk_with_logprobs(delta, finish_reason, None)
    }

    fn chunk_with_logprobs(
        &self,
        delta: Value,
        finish_reason: Option<&str>,
        logprobs: Option<&ChoiceLogprobs>,
    ) -> String {
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
                    "logprobs": logprobs.map_or(Value::Null, |value| serde_json::to_value(value).unwrap_or(Value::Null)),
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
        if finish_reason.is_some() && self.incomplete_tool_arguments {
            payload["x-experiential-incomplete-reason"] = json!("tool_arguments_incomplete");
        }
        chat_data(&payload)
    }

    fn usage_chunk(&self, usage: &Usage) -> String {
        let mut usage = streaming_chat_usage(usage);
        annotate_usage_details(
            &mut usage,
            self.web_search.as_ref().map(ChatWebSearch::admission),
        );
        annotate_tool_search_usage_details(&mut usage, self.tool_search_requests);
        let payload = json!({
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created_at,
            "model": self.model,
            "choices": [],
            "usage": usage,
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
        "prompt_tokens_details": chat_input_details(usage, true),
        "completion_tokens_details": {"reasoning_tokens": usage.reasoning_tokens.unwrap_or(0)},
    })
}

/// Input subsets remain unknown when absent; cache writes are never invented.
fn chat_input_details(usage: &Usage, streaming: bool) -> Value {
    let mut details = serde_json::Map::new();
    if let Some(cached) = usage.cached_input_tokens.or(streaming.then_some(0)) {
        details.insert("cached_tokens".into(), json!(cached));
    }
    if let Some(written) = usage.cache_creation_input_tokens {
        details.insert("cache_write_tokens".into(), json!(written));
    }
    if details.is_empty() {
        Value::Null
    } else {
        Value::Object(details)
    }
}

/// Non-streaming usage shape from `exp.runtime.openai_protocol.response.chat_usage`.
fn completed_chat_usage(usage: Option<&Usage>) -> Value {
    let usage = match usage {
        Some(usage) if usage.has_token_counts() => usage,
        _ => return Value::Null,
    };
    let input = usage.input_tokens.unwrap_or(0);
    let output = usage.output_tokens.unwrap_or(0);
    let details = chat_input_details(usage, false);
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
    reasoning_output_exposed: bool,
) -> Result<AggregatedCompletion, PublicError> {
    completed_chat_body_with_carrier(
        request_id,
        model,
        created_at,
        events,
        ignored_parameters,
        None,
        reasoning_output_exposed,
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
    reasoning_output_exposed: bool,
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
    let logprobs = crate::logprobs::aggregate(events);
    let reasoning = reasoning_carrier_candidate(events)?;
    let incomplete = matches!(terminal, Event::Incomplete | Event::IncompleteToolArguments);
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
    });
    let images: Vec<Value> = events
        .iter()
        .filter_map(|event| match event {
            Event::Image(url) => Some(crate::image_output::chat_image_value(url)),
            _ => None,
        })
        .collect();
    if !images.is_empty() {
        message["images"] = Value::Array(images);
    }
    // OpenAI documents `tool_calls` as an optional array and omits it from a
    // message that made no calls; it never serializes `null` there. Strict
    // OpenAI-schema consumers (the OpenRouter SDK's response parser, for one)
    // reject `"tool_calls": null` while accepting an absent key, so the key
    // appears only when there is at least one call.
    if has_tool_calls {
        message
            .as_object_mut()
            .expect("chat message is an object")
            .insert("tool_calls".to_string(), Value::Array(tool_calls));
    }
    if matches!(terminal, Event::Completed | Event::StoppedAtSequence(_))
        && has_tool_calls
        && reasoning.is_some()
        && !reasoning_output_exposed
    {
        // A tool turn's reasoning round-trips as the sealed opaque carrier
        // (never raw plaintext — that would be a CoT-injection vector on the
        // way back in), so `reasoning_content` carries the carrier.
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
    } else if reasoning_output_exposed {
        // A non-tool reasoning turn on an exposure-gated rung
        // (Tencent/DeepSeek) has no carrier to round-trip, so the plaintext
        // reasoning is returned for display only. The tokens are already
        // billed; every other rung keeps reasoning stripped.
        let reasoning_text: String = events
            .iter()
            .filter_map(|event| match event {
                Event::ReasoningContentDelta { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        if !reasoning_text.is_empty() {
            message
                .as_object_mut()
                .expect("chat message is an object")
                .insert(
                    "reasoning_content".to_string(),
                    Value::String(reasoning_text),
                );
        }
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
                "logprobs": logprobs.as_ref().map_or(Value::Null, |value| serde_json::to_value(value).unwrap_or(Value::Null)),
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
    if matches!(terminal, Event::IncompleteToolArguments) {
        body["x-experiential-incomplete-reason"] = json!("tool_arguments_incomplete");
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
#[path = "encode_tests.rs"]
mod tests;
