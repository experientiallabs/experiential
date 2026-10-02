//! Anthropic Messages frame mapping: usage legs accumulate across the
//! message lifecycle and fold at `message_stop`, refusal blocks and refusal
//! stop reasons mark the stream, and extended-thinking blocks normalize to
//! dedicated thinking events so callers receive the reasoning they pay for.

use serde_json::Value;

mod cache_write;

use super::{
    finish_open_tools, finish_open_tools_truncated, malformed, optional_text, parse_object,
    refusal_failure, Normalizer,
};
use crate::encode::compact_json;
use crate::errors::Failure;
use crate::events::{
    bounded_ledger_sum, count_if_present, count_or_zero, require_string, require_u64, Event,
    ToolAccumulator, Usage,
};

impl Normalizer {
    /// Fold the latest decoded Anthropic counters into the shared usage shape.
    fn anthropic_usage(&self) -> Result<Usage, Failure> {
        let input_tokens = self
            .input_tokens
            .map(|fresh| {
                bounded_ledger_sum(
                    &[fresh, self.cache_read, self.cache_write],
                    "Anthropic input",
                )
            })
            .transpose()
            .map_err(|message| malformed(&message))?;
        Ok(Usage {
            input_tokens,
            output_tokens: self.output_tokens,
            cached_input_tokens: Some(self.cache_read),
            // Keep the cache-less wire shape; thinking is billed inside output
            // and has no provider-reported subset to expose here.
            cache_creation_input_tokens: (self.cache_write > 0).then_some(self.cache_write),
            cache_creation_1h_input_tokens: self.cache_write_1h.filter(|_| self.cache_write > 0),
            reasoning_tokens: None,
        })
    }

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
                // Primary omissions are unknown. Optional cache legs use the
                // provider's zero-when-omitted convention.
                self.input_tokens = count_if_present(usage, "input_tokens", "Anthropic usage")
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
                self.cache_write_1h = cache_write::hour_subset(usage, self.cache_write)?;
                self.output_tokens = self.output_tokens.max(
                    count_if_present(usage, "output_tokens", "Anthropic usage")
                        .map_err(|message| malformed(&message))?,
                );
                // Surface the start-frame meters at once: the Messages encoder
                // mirrors them on its own `message_start` (Claude Code reads
                // the input legs there), and the settlement tracker holds them
                // as the best known count until the terminal report, which
                // supersedes them at `message_stop` (server-tool turns re-read
                // fetched results as input, so the start count undercounts).
                let usage = self.anthropic_usage()?;
                self.usage = Some(usage.clone());
                events.push(Event::Usage(usage));
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
                            custom: false,
                            index,
                            call_id,
                            name,
                            namespace: None,
                            caller: None,
                        });
                    }
                    Some("server_tool_use") => {
                        // Provider-executed server tool (web search): same
                        // start/argument lifecycle as a client tool, but on
                        // dedicated events so it never becomes client tool
                        // history or a tool_use stop reason.
                        let call_id = require_string(block, "id", "Anthropic server tool ID")
                            .map_err(|message| malformed(&message))?;
                        let name = require_string(block, "name", "Anthropic server tool name")
                            .map_err(|message| malformed(&message))?;
                        if self.tools.contains_key(&index) {
                            return Err(malformed("Anthropic stream repeated a tool-call start"));
                        }
                        self.reserve_tool_entry(index)?;
                        let mut tool = ToolAccumulator::new(call_id.clone(), name.clone());
                        tool.server = true;
                        self.tools.insert(index, tool);
                        events.push(Event::ServerToolUseStarted {
                            index,
                            call_id,
                            name,
                        });
                    }
                    Some(block_type) if block_type.ends_with("_tool_result") => {
                        // A server tool's result (`web_search_tool_result`,
                        // `tool_search_tool_result`, ...) arrives whole in the
                        // start frame and is carried verbatim so the caller
                        // (and its next-turn echo) sees exactly what the
                        // provider produced.
                        let serialized = compact_json(&Value::Object(block.clone()));
                        self.reserve_tool_bytes(serialized.len())?;
                        events.push(Event::ServerToolResult {
                            index,
                            block: serialized,
                        });
                    }
                    Some("text") => {
                        // The boundary event lets the Messages encoder mirror
                        // the provider's text-block structure, which is what
                        // citations attach to.
                        events.push(Event::TextBlockStarted { index });
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
                    Some("thinking") => {
                        let text = optional_text(block, "thinking", "Anthropic initial thinking")?;
                        if !text.is_empty() {
                            events.push(Event::ThinkingDelta { index, delta: text });
                        }
                    }
                    Some("redacted_thinking") => {
                        // Redacted thinking arrives whole in the start frame.
                        let data = optional_text(block, "data", "Anthropic redacted thinking")?;
                        events.push(Event::RedactedThinking { index, data });
                    }
                    // Unknown block kinds with no gateway-visible output are
                    // skipped rather than rejected.
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
                        events.push(if tool.server {
                            Event::ServerToolArgumentsDelta {
                                index,
                                delta: fragment,
                            }
                        } else {
                            Event::ToolArgumentsDelta {
                                index,
                                delta: fragment,
                            }
                        });
                    }
                    Some("citations_delta") => {
                        // One whole citation object attached to the open text
                        // block, carried verbatim (server-tool answers cite
                        // their web sources through these).
                        let citation = delta
                            .get("citation")
                            .and_then(Value::as_object)
                            .ok_or_else(|| {
                                malformed("Anthropic citations_delta.citation must be an object")
                            })?;
                        let serialized = compact_json(&Value::Object(citation.clone()));
                        self.reserve_tool_bytes(serialized.len())?;
                        events.push(Event::CitationDelta {
                            index,
                            citation: serialized,
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
                    Some("thinking_delta") => {
                        let text = optional_text(delta, "thinking", "Anthropic thinking delta")?;
                        if !text.is_empty() {
                            events.push(Event::ThinkingDelta { index, delta: text });
                        }
                    }
                    Some("signature_delta") => {
                        let signature =
                            optional_text(delta, "signature", "Anthropic signature delta")?;
                        if !signature.is_empty() {
                            events.push(Event::ThinkingSignature { index, signature });
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let index = require_u64(&payload, "index", "Anthropic content index")
                    .map_err(|message| malformed(&message))? as u32;
                if let Some(mut tool) = self.tools.remove(&index) {
                    self.anthropic_stopped_tools.insert(index);
                    if !tool.completed {
                        // The stop reason arrives in the following
                        // message_delta, so a fragment left open by the
                        // output budget cannot be told from garbage yet.
                        self.complete_tool_deferring_failure(index, &mut tool, &mut events);
                    }
                    self.tools.insert(index, tool);
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
                self.output_tokens = self.output_tokens.max(
                    count_if_present(usage, "output_tokens", "Anthropic usage")
                        .map_err(|message| malformed(&message))?,
                );
                // The terminal usage report supersedes message_start when its
                // input legs are present: server-tool turns re-read fetched
                // results as input, so the start-frame count undercounts the
                // billed total severely (verified live 2026-08-31).
                self.input_tokens = self.input_tokens.max(
                    count_if_present(usage, "input_tokens", "Anthropic message_delta")
                        .map_err(|message| malformed(&message))?,
                );
                for (key, slot) in [
                    ("cache_read_input_tokens", &mut self.cache_read),
                    ("cache_creation_input_tokens", &mut self.cache_write),
                ] {
                    if let Some(value) = count_if_present(usage, key, "Anthropic message_delta")
                        .map_err(|message| malformed(&message))?
                    {
                        *slot = value;
                    }
                }
                if usage.contains_key("cache_creation_input_tokens")
                    || usage.contains_key("cache_creation")
                {
                    self.cache_write_1h = cache_write::hour_subset(usage, self.cache_write)?;
                }
                // The relay may be cancelled before message_stop arrives.
                // Retain these decoded meters without changing event timing.
                self.usage = Some(self.anthropic_usage()?);
                if self.stop_reason.as_deref() == Some("refusal") && !self.refusal_seen {
                    self.refusal_seen = true;
                    events.push(Event::RefusalDelta(String::new()));
                }
            }
            "message_stop" => {
                if !self.refusal_seen
                    && !matches!(
                        self.stop_reason.as_deref(),
                        Some(
                            "end_turn"
                                | "stop_sequence"
                                | "tool_use"
                                | "pause_turn"
                                | "max_tokens"
                                | "model_context_window_exceeded"
                                | "refusal"
                        )
                    )
                {
                    return Ok(vec![
                        Event::Usage(self.anthropic_usage()?),
                        Event::Failed(malformed(
                            "Anthropic stream ended without a recognized stop reason",
                        )),
                    ]);
                }
                let truncated = matches!(
                    self.stop_reason.as_deref(),
                    Some("max_tokens" | "model_context_window_exceeded")
                );
                self.resolve_deferred_tool_failure(truncated)?;
                if self.refusal_seen
                    || !matches!(
                        self.stop_reason.as_deref(),
                        Some("end_turn" | "stop_sequence" | "tool_use" | "pause_turn")
                    )
                {
                    // No successful final reason authorized a zero-argument
                    // call. Preserve the start, without inventing its input.
                    for tool in self.tools.values_mut() {
                        if !tool.custom && tool.raw_arguments.is_empty() {
                            tool.completed = true;
                        }
                    }
                }
                events.extend(if truncated {
                    finish_open_tools_truncated(&mut self.tools)?
                } else {
                    finish_open_tools(&mut self.tools)?
                });
                events.push(Event::Usage(self.anthropic_usage()?));
                if self.refusal_seen || self.stop_reason.as_deref() == Some("refusal") {
                    events.push(Event::Failed(refusal_failure()));
                } else if truncated {
                    events.push(Event::Incomplete);
                } else if self.stop_reason.as_deref() == Some("pause_turn") {
                    // A paused server-tool turn must keep its stop reason:
                    // the caller resumes it by resending the conversation,
                    // and an end_turn rewrite would end the task instead.
                    events.push(Event::PausedTurn);
                } else if self.dropped_cut_call {
                    // A call cut mid-fragment under a non-truncating stop
                    // reason was dropped at its block stop.
                    events.push(Event::Incomplete);
                } else {
                    events.push(Event::Completed);
                }
            }
            "error" => {
                // The provider names its failure mechanism only inside this
                // frame; the bounded detail rides the failure into the ledger.
                let (code, message) = match payload.get("error").and_then(Value::as_object) {
                    Some(error) => (
                        error.get("type").and_then(Value::as_str),
                        error.get("message").and_then(Value::as_str),
                    ),
                    None => (None, None),
                };
                events.push(Event::Failed(self.provider_stream_failure(
                    "anthropic_messages",
                    code,
                    message,
                    None,
                )));
            }
            "ping" => {}
            _ => {
                return Err(malformed(&format!(
                    "Anthropic stream emitted an unsupported event (type {})",
                    super::bounded_wire_token(&event_type),
                )));
            }
        }
        Ok(events)
    }
}

#[cfg(test)]
mod tests;
