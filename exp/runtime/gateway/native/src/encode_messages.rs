//! Public Anthropic Messages encoding, the Rust mirror of
//! `exp.runtime.anthropic_protocol.encoding` (`MessagesSseEncoder` and
//! `completed_messages_body`) and of the Anthropic error envelope in
//! `exp.runtime.anthropic_protocol.errors`.

use std::collections::{HashMap, HashSet};

use serde_json::{json, Map, Value};

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

/// Stateful Anthropic Messages SSE encoder with one open block and one
/// terminal, emitting byte-identical frames to the python encoder.
pub struct MessagesSseEncoder {
    message_id: String,
    model: String,
    started: bool,
    terminal: bool,
    next_block_index: u32,
    open_block: Option<u32>,
    open_tool_for: Option<u32>,
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
            next_block_index: 0,
            open_block: None,
            open_tool_for: None,
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
            Event::TextDelta(text) => Ok(self.text_delta(text)),
            Event::RefusalDelta(_) => {
                // There is no Anthropic refusal block; the refusal is
                // reported as one sanitized terminal error instead.
                self.refusal_seen = true;
                Ok(Vec::new())
            }
            Event::ToolCallStarted {
                index,
                call_id,
                name,
            } => self.tool_started(*index, call_id, name),
            Event::ToolArgumentsDelta { index, delta } => self.tool_arguments_delta(*index, delta),
            Event::ToolCallCompleted { index, call } => {
                // Some upstream dialects (OpenAI-compatible streams) emit
                // every tool completion only at their terminal sentinel,
                // after later text already closed the tool_use block, so
                // completion verifies against the accumulated state rather
                // than requiring the block to still be open.
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
                if self.open_tool_for == Some(*index) {
                    Ok(self.close_block())
                } else {
                    Ok(Vec::new())
                }
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
                let mut frames = self.close_block();
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

    /// Open the text block as needed and emit one text delta.
    fn text_delta(&mut self, delta: &str) -> Vec<String> {
        let mut frames = Vec::new();
        if self.open_tool_for.is_some() || self.open_block.is_none() {
            frames.extend(self.close_block());
            let index = self.next_block_index;
            self.next_block_index += 1;
            self.open_block = Some(index);
            frames.push(event_frame(
                "content_block_start",
                &json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                }),
            ));
        }
        frames.push(event_frame(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": self.open_block,
                "delta": {"type": "text_delta", "text": delta},
            }),
        ));
        frames
    }

    /// Close the open block and start one tool_use block.
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
        let mut frames = self.close_block();
        let index = self.next_block_index;
        self.next_block_index += 1;
        self.open_block = Some(index);
        self.open_tool_for = Some(tool_index);
        frames.push(event_frame(
            "content_block_start",
            &json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": {},
                },
            }),
        ));
        Ok(frames)
    }

    /// Emit one raw provider-order argument fragment for the open tool.
    fn tool_arguments_delta(
        &mut self,
        tool_index: u32,
        delta: &str,
    ) -> Result<Vec<String>, PublicError> {
        if self.open_tool_for != Some(tool_index) || self.open_block.is_none() {
            return Err(invalid_provider_stream(
                "Messages tool arguments arrived before tool-call start.",
            ));
        }
        let accumulated = self
            .tool_arguments
            .get_mut(&tool_index)
            .expect("open tool has accumulated arguments");
        accumulated.push_str(delta);
        Ok(vec![event_frame(
            "content_block_delta",
            &json!({
                "type": "content_block_delta",
                "index": self.open_block,
                "delta": {"type": "input_json_delta", "partial_json": delta},
            }),
        )])
    }

    /// Emit `content_block_stop` for the currently open block, if any.
    fn close_block(&mut self) -> Vec<String> {
        let Some(index) = self.open_block.take() else {
            return Vec::new();
        };
        self.open_tool_for = None;
        vec![event_frame(
            "content_block_stop",
            &json!({"type": "content_block_stop", "index": index}),
        )]
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
    let mut saw_tool_use = false;
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
mod tests {
    use super::*;
    use crate::events::CompletedToolCall;

    #[test]
    fn usage_reports_cached_reads_out_of_the_input_total() {
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(4),
            cached_input_tokens: Some(3),
            reasoning_tokens: None,
        };
        assert_eq!(
            messages_usage(Some(&usage)),
            json!({"input_tokens": 7, "output_tokens": 4, "cache_read_input_tokens": 3})
        );
        assert_eq!(
            messages_usage(None),
            json!({"input_tokens": 0, "output_tokens": 0})
        );
    }

    #[test]
    fn error_body_folds_param_and_maps_status_first() {
        let mut error = PublicError::new(
            400,
            "invalid_parameter",
            "Invalid value.",
            "invalid_request_error",
        );
        error.param = Some("top_k".to_string());
        assert_eq!(
            anthropic_error_body(&error),
            json!({
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "Invalid value. (param: top_k)",
                },
            })
        );
        let throttled = PublicError::new(429, "unavailable_route", "Throttled.", "api_error");
        assert_eq!(
            anthropic_error_body(&throttled)["error"]["type"],
            json!("rate_limit_error")
        );
    }

    #[test]
    fn completed_body_orders_text_before_tool_use_blocks() {
        let events = vec![
            Event::TextDelta("hi".to_string()),
            Event::ToolCallStarted {
                index: 0,
                call_id: "call-1".to_string(),
                name: "search".to_string(),
            },
            Event::ToolArgumentsDelta {
                index: 0,
                delta: "{\"b\":1,\"a\":2}".to_string(),
            },
            Event::ToolCallCompleted {
                index: 0,
                call: CompletedToolCall {
                    call_id: "call-1".to_string(),
                    name: "search".to_string(),
                    raw_arguments: "{\"b\":1,\"a\":2}".to_string(),
                },
            },
            Event::Completed,
        ];
        let aggregated =
            completed_messages_body("request-abc", "coding", &events).expect("aggregates");
        assert!(aggregated.failure.is_none());
        assert_eq!(aggregated.body["stop_reason"], json!("tool_use"));
        assert_eq!(aggregated.body["content"][0]["type"], json!("text"));
        // preserve_order keeps the provider's key order in the parsed input.
        assert_eq!(
            compact_json(&aggregated.body["content"][1]["input"]),
            "{\"b\":1,\"a\":2}"
        );
        assert_eq!(aggregated.tool_names, vec!["search".to_string()]);
    }

    #[test]
    fn completed_body_preserves_interleaved_block_order() {
        let events = vec![
            Event::ToolCallStarted {
                index: 0,
                call_id: "call-1".to_string(),
                name: "search".to_string(),
            },
            Event::ToolCallCompleted {
                index: 0,
                call: CompletedToolCall {
                    call_id: "call-1".to_string(),
                    name: "search".to_string(),
                    raw_arguments: "{}".to_string(),
                },
            },
            Event::TextDelta("after ".to_string()),
            Event::TextDelta("the tool".to_string()),
            Event::Completed,
        ];
        let aggregated =
            completed_messages_body("request-abc", "coding", &events).expect("aggregates");
        assert_eq!(aggregated.body["content"][0]["type"], json!("tool_use"));
        assert_eq!(
            aggregated.body["content"][1],
            json!({"type": "text", "text": "after the tool"})
        );
    }

    #[test]
    fn deferred_tool_completion_keeps_the_started_block_position() {
        // OpenAI-compatible streams complete every tool only at [DONE], so
        // text may arrive between the tool's arguments and its completion.
        let events = vec![
            Event::ToolCallStarted {
                index: 0,
                call_id: "call-1".to_string(),
                name: "search".to_string(),
            },
            Event::ToolArgumentsDelta {
                index: 0,
                delta: "{}".to_string(),
            },
            Event::TextDelta("after".to_string()),
            Event::ToolCallCompleted {
                index: 0,
                call: CompletedToolCall {
                    call_id: "call-1".to_string(),
                    name: "search".to_string(),
                    raw_arguments: "{}".to_string(),
                },
            },
            Event::Completed,
        ];
        let aggregated =
            completed_messages_body("request-abc", "coding", &events).expect("aggregates");
        assert_eq!(aggregated.body["content"][0]["type"], json!("tool_use"));
        assert_eq!(
            aggregated.body["content"][1],
            json!({"type": "text", "text": "after"})
        );
        let mut encoder = MessagesSseEncoder::new("request-abc", "coding");
        let mut frames = encoder.start().expect("starts");
        for event in &events {
            frames.extend(
                encoder
                    .feed(event)
                    .expect("streams the deferred completion"),
            );
        }
        assert!(frames.last().expect("terminal").contains("message_stop"));
    }

    #[test]
    fn refusal_content_aggregates_as_a_sanitized_failure() {
        let events = vec![Event::RefusalDelta("no".to_string()), Event::Completed];
        let aggregated =
            completed_messages_body("request-abc", "coding", &events).expect("aggregates");
        let failure = aggregated.failure.expect("refusal failure");
        assert_eq!(failure.failure_class, FailureClass::Refusal);
    }
}
