//! Anthropic Messages frame mapping: usage legs accumulate across the
//! message lifecycle and fold at `message_stop`, refusal blocks and refusal
//! stop reasons mark the stream, and extended-thinking blocks normalize to
//! dedicated thinking events so callers receive the reasoning they pay for.

use serde_json::Value;

use super::{
    complete_streamed_tool, finish_open_tools, malformed, optional_text, parse_object,
    provider_stream_failed, refusal_failure, Normalizer,
};
use crate::encode::compact_json;
use crate::errors::Failure;
use crate::events::{
    count_if_present, count_or_zero, require_string, require_u64, Event, ToolAccumulator, Usage,
    MAXIMUM_LEDGER_COUNT,
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
                    Some("web_search_tool_result") => {
                        // The result arrives whole in the start frame and is
                        // carried verbatim so the caller (and its next-turn
                        // echo) sees exactly what the provider produced.
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
                if let Some(tool) = self.tools.get_mut(&index) {
                    if !tool.completed {
                        complete_streamed_tool(index, tool, &mut events)?;
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
                // The terminal usage report supersedes message_start when its
                // input legs are present: server-tool turns re-read fetched
                // results as input, so the start-frame count undercounts the
                // billed total severely (verified live 2026-08-31).
                for (key, slot) in [
                    ("input_tokens", &mut self.input_tokens),
                    ("cache_read_input_tokens", &mut self.cache_read),
                    ("cache_creation_input_tokens", &mut self.cache_write),
                ] {
                    if let Some(value) = count_if_present(usage, key, "Anthropic message_delta")
                        .map_err(|message| malformed(&message))?
                    {
                        *slot = value;
                    }
                }
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
                    // Present only when nonzero so cache-less streams keep
                    // their exact pre-field usage shape.
                    cache_creation_input_tokens: (self.cache_write > 0).then_some(self.cache_write),
                    // Anthropic reports thinking inside output_tokens and
                    // publishes no separate count, so the reasoning subset
                    // stays unknown instead of being invented.
                    reasoning_tokens: None,
                }));
                if self.refusal_seen || self.stop_reason.as_deref() == Some("refusal") {
                    events.push(Event::Failed(refusal_failure()));
                } else if self.stop_reason.as_deref() == Some("max_tokens") {
                    events.push(Event::Incomplete);
                } else if self.stop_reason.as_deref() == Some("pause_turn") {
                    // A paused server-tool turn must keep its stop reason:
                    // the caller resumes it by resending the conversation,
                    // and an end_turn rewrite would end the task instead.
                    events.push(Event::PausedTurn);
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

#[cfg(test)]
mod tests {
    use crate::dialects::{Dialect, Normalizer};
    use crate::events::Event;
    use crate::sse::SseEvent;

    fn frame(payload: serde_json::Value) -> SseEvent {
        SseEvent {
            event: None,
            data: payload.to_string(),
        }
    }

    #[test]
    fn thinking_blocks_normalize_to_dedicated_events() {
        let mut normalizer = Normalizer::new(Dialect::AnthropicMessages);
        let start = frame(serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        }));
        assert!(normalizer.feed(&start).expect("start").is_empty());

        let delta = frame(serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "step one"},
        }));
        let events = normalizer.feed(&delta).expect("thinking delta");
        assert!(matches!(
            events.as_slice(),
            [Event::ThinkingDelta { index: 0, delta }] if delta == "step one"
        ));

        let signature = frame(serde_json::json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig=="},
        }));
        let events = normalizer.feed(&signature).expect("signature delta");
        assert!(matches!(
            events.as_slice(),
            [Event::ThinkingSignature { index: 0, signature }] if signature == "sig=="
        ));

        let redacted = frame(serde_json::json!({
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "redacted_thinking", "data": "opaque=="},
        }));
        let events = normalizer.feed(&redacted).expect("redacted block");
        assert!(matches!(
            events.as_slice(),
            [Event::RedactedThinking { index: 1, data }] if data == "opaque=="
        ));
    }

    /// Frame shapes captured live from one web_search stream (2026-08-31):
    /// server tool use with a leading empty input fragment, the whole result
    /// in its start frame, a cited answer, terminal usage that supersedes the
    /// start-frame input count, and the pause_turn stop reason.
    #[test]
    fn server_tool_frames_normalize_to_dedicated_events() {
        let mut normalizer = Normalizer::new(Dialect::AnthropicMessages);
        let start_message = frame(serde_json::json!({
            "type": "message_start",
            "message": {"usage": {"input_tokens": 2230, "output_tokens": 25}},
        }));
        assert!(normalizer.feed(&start_message).expect("start").is_empty());

        let start = frame(serde_json::json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {},
            },
        }));
        let events = normalizer.feed(&start).expect("server tool start");
        assert!(matches!(
            events.as_slice(),
            [Event::ServerToolUseStarted { index: 0, call_id, name }]
                if call_id == "srvtoolu_1" && name == "web_search"
        ));

        // The provider legally leads with an empty input fragment.
        for partial in ["", "{\"query\": \"python\"}"] {
            let delta = frame(serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": partial},
            }));
            let events = normalizer.feed(&delta).expect("server input delta");
            assert!(matches!(
                events.as_slice(),
                [Event::ServerToolArgumentsDelta { index: 0, delta }] if delta == partial
            ));
        }
        let stop = frame(serde_json::json!({"type": "content_block_stop", "index": 0}));
        let events = normalizer.feed(&stop).expect("server tool stop");
        assert!(matches!(
            events.as_slice(),
            [Event::ServerToolUseCompleted { index: 0, call }]
                if call.raw_arguments == "{\"query\": \"python\"}" && !call.custom
        ));

        let result = frame(serde_json::json!({
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_1",
                "content": [{"type": "web_search_result", "url": "https://example.com"}],
                "caller": {"type": "direct"},
            },
        }));
        let events = normalizer.feed(&result).expect("server tool result");
        match events.as_slice() {
            [Event::ServerToolResult { index: 1, block }] => {
                assert!(block.contains("\"caller\":{\"type\":\"direct\"}"));
            }
            other => panic!("unexpected events: {other:?}"),
        }

        let text_start = frame(serde_json::json!({
            "type": "content_block_start",
            "index": 2,
            "content_block": {"citations": [], "type": "text", "text": ""},
        }));
        let events = normalizer.feed(&text_start).expect("text start");
        assert!(matches!(
            events.as_slice(),
            [Event::TextBlockStarted { index: 2 }]
        ));
        let citation = frame(serde_json::json!({
            "type": "content_block_delta",
            "index": 2,
            "delta": {
                "type": "citations_delta",
                "citation": {"type": "web_search_result_location", "cited_text": "3.14.7"},
            },
        }));
        let events = normalizer.feed(&citation).expect("citation delta");
        match events.as_slice() {
            [Event::CitationDelta { index: 2, citation }] => {
                assert!(citation.contains("\"cited_text\":\"3.14.7\""));
            }
            other => panic!("unexpected events: {other:?}"),
        }

        // The terminal usage report supersedes the start-frame input legs.
        let message_delta = frame(serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": "pause_turn", "stop_sequence": null},
            "usage": {
                "input_tokens": 12284,
                "cache_read_input_tokens": 4,
                "cache_creation_input_tokens": 6,
                "output_tokens": 103,
                "server_tool_use": {"web_search_requests": 1},
            },
        }));
        assert!(normalizer.feed(&message_delta).expect("delta").is_empty());
        let stop_message = frame(serde_json::json!({"type": "message_stop"}));
        let events = normalizer.feed(&stop_message).expect("message stop");
        match events.as_slice() {
            [Event::Usage(usage), Event::PausedTurn] => {
                assert_eq!(usage.input_tokens, Some(12294));
                assert_eq!(usage.output_tokens, Some(103));
                assert_eq!(usage.cached_input_tokens, Some(4));
            }
            other => panic!("unexpected events: {other:?}"),
        }
    }
}
