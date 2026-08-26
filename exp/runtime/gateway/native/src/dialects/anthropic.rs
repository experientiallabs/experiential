//! Anthropic Messages frame mapping, mirroring the python event mapper:
//! usage legs accumulate across the message lifecycle and fold at
//! `message_stop`, refusal blocks and refusal stop reasons mark the stream,
//! and extended-thinking blocks are skipped rather than rejected.

use serde_json::Value;

use super::{
    finish_open_tools, malformed, optional_text, parse_object, provider_stream_failed,
    refusal_failure, Normalizer,
};
use crate::errors::Failure;
use crate::events::{
    count_or_zero, require_string, require_u64, Event, ToolAccumulator, Usage, MAXIMUM_LEDGER_COUNT,
};

impl Normalizer {
    pub(super) fn feed_anthropic(
        &mut self,
        frame: &crate::sse::SseEvent,
    ) -> Result<Vec<Event>, Failure> {
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
                        self.reserve_tool_entry(index)?;
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
                // Individually persistable legs whose folded total is not
                // are a provider contract violation, exactly like the
                // Bedrock cache-leg fold.
                let input_tokens = self
                    .input_tokens
                    .checked_add(self.cache_read)
                    .and_then(|total| total.checked_add(self.cache_write))
                    .filter(|total| *total <= MAXIMUM_LEDGER_COUNT)
                    .ok_or_else(|| {
                        malformed("Anthropic input token total overflows a persistable count")
                    })?;
                events.push(Event::Usage(Usage {
                    input_tokens: Some(input_tokens),
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
}
