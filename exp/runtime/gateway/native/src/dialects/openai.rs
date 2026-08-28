//! OpenAI frame mappings: the Responses SSE dialect and the Chat-Completions
//! compatible dialect, mirroring the python event mappers.

use serde_json::Value;

use super::{
    finish_open_tools, malformed, optional_text, parse_object, provider_stream_failed,
    refusal_failure, Normalizer,
};
use crate::errors::{Failure, FailureClass};
use crate::events::{
    openai_compatible_usage, openai_usage, require_string, require_u64, Event, ToolAccumulator,
};

impl Normalizer {
    pub(super) fn feed_openai_responses(
        &mut self,
        frame: &crate::sse::SseEvent,
    ) -> Result<Vec<Event>, Failure> {
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
            "response.reasoning_summary_text.delta" => {
                let output_index =
                    require_u64(&payload, "output_index", "OpenAI reasoning output_index")
                        .map_err(|message| malformed(&message))? as u32;
                let summary_index =
                    require_u64(&payload, "summary_index", "OpenAI reasoning summary_index")
                        .map_err(|message| malformed(&message))? as u32;
                let delta = optional_text(&payload, "delta", "OpenAI reasoning summary delta")?;
                if delta.is_empty() {
                    return Ok(events);
                }
                let key = (output_index, summary_index);
                self.reserve_summary_entry(key)?;
                self.reserve_summary_bytes(delta.len())?;
                self.reasoning_summaries
                    .entry(key)
                    .or_default()
                    .push_str(&delta);
                events.push(Event::ReasoningSummaryDelta {
                    output_index,
                    summary_index,
                    delta,
                });
            }
            "response.reasoning_summary_text.done" => {
                let output_index =
                    require_u64(&payload, "output_index", "OpenAI reasoning output_index")
                        .map_err(|message| malformed(&message))? as u32;
                let summary_index =
                    require_u64(&payload, "summary_index", "OpenAI reasoning summary_index")
                        .map_err(|message| malformed(&message))? as u32;
                let final_text = require_string(&payload, "text", "OpenAI reasoning summary text")
                    .map_err(|message| malformed(&message))?;
                let key = (output_index, summary_index);
                let streamed = self
                    .reasoning_summaries
                    .get(&key)
                    .map(String::as_str)
                    .unwrap_or("");
                if !streamed.is_empty() && streamed != final_text {
                    return Err(malformed(
                        "OpenAI reasoning summary fragments changed at done",
                    ));
                }
                if streamed.is_empty() && !final_text.is_empty() {
                    self.reserve_summary_entry(key)?;
                    self.reserve_summary_bytes(final_text.len())?;
                    self.reasoning_summaries.insert(key, final_text.clone());
                    events.push(Event::ReasoningSummaryDelta {
                        output_index,
                        summary_index,
                        delta: final_text,
                    });
                }
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
                    self.reserve_tool_entry(index)?;
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
                if event_type == "response.output_item.done" {
                    if let Some(item) = payload.get("item").and_then(Value::as_object) {
                        // Requested encrypted reasoning arrives whole on the
                        // completed reasoning item; pass the opaque payload
                        // through under the shared retained-output budget.
                        if item.get("type").and_then(Value::as_str) == Some("reasoning") {
                            if let Some(Value::String(encrypted)) = item.get("encrypted_content") {
                                if !encrypted.is_empty() {
                                    self.reserve_summary_bytes(encrypted.len())?;
                                    events.push(Event::EncryptedReasoning {
                                        output_index: index,
                                        encrypted_content: encrypted.clone(),
                                    });
                                }
                            }
                        }
                    }
                }
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
                        events.push(Event::Failed(
                            Failure::new(
                                FailureClass::ProviderInternal,
                                "provider ended the stream incompletely",
                            )
                            .with_retry(true, true),
                        ));
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

    pub(super) fn feed_openai_compatible(
        &mut self,
        frame: &crate::sse::SseEvent,
    ) -> Result<Vec<Event>, Failure> {
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
                        self.reserve_tool_entry(index)?;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialects::{Dialect, MAXIMUM_RETAINED_PROVIDER_ENTRIES, OUTPUT_OVERFLOW_MESSAGE};
    use crate::sse::SseEvent;

    fn reasoning_delta(output_index: u32, summary_index: u32, delta: &str) -> SseEvent {
        SseEvent {
            event: None,
            data: serde_json::json!({
                "type": "response.reasoning_summary_text.delta",
                "output_index": output_index,
                "summary_index": summary_index,
                "delta": delta,
            })
            .to_string(),
        }
    }

    #[test]
    fn responses_reasoning_summary_is_normalized_and_verified() {
        let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
        let delta = SseEvent {
            event: None,
            data: serde_json::json!({
                "type": "response.reasoning_summary_text.delta",
                "output_index": 2,
                "summary_index": 1,
                "delta": "checked",
            })
            .to_string(),
        };
        let events = normalizer
            .feed(&delta)
            .expect("summary delta must normalize");
        assert!(matches!(
            events.as_slice(),
            [Event::ReasoningSummaryDelta {
                output_index: 2,
                summary_index: 1,
                delta,
            }] if delta == "checked"
        ));

        let done = SseEvent {
            event: None,
            data: serde_json::json!({
                "type": "response.reasoning_summary_text.done",
                "output_index": 2,
                "summary_index": 1,
                "text": "checked",
            })
            .to_string(),
        };
        assert!(normalizer
            .feed(&done)
            .expect("matching summary completion must validate")
            .is_empty());
    }

    #[test]
    fn empty_reasoning_deltas_do_not_allocate_provider_state() {
        let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
        for output_index in 0..=MAXIMUM_RETAINED_PROVIDER_ENTRIES as u32 {
            assert!(normalizer
                .feed(&reasoning_delta(output_index, 0, ""))
                .expect("empty summary delta must be ignored")
                .is_empty());
        }
        assert!(normalizer.reasoning_summaries.is_empty());

        assert_eq!(
            normalizer
                .feed(&reasoning_delta(0, 0, "bounded"))
                .expect("non-empty summary still fits")
                .len(),
            1
        );
    }

    #[test]
    fn retained_provider_entries_are_bounded_across_tools_and_summaries() {
        let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
        for index in 0..MAXIMUM_RETAINED_PROVIDER_ENTRIES as u32 {
            normalizer
                .reserve_tool_entry(index)
                .expect("entry below ceiling must fit");
            normalizer.tools.insert(
                index,
                ToolAccumulator::new(format!("call-{index}"), "lookup".to_string()),
            );
        }

        let failure = normalizer
            .feed(&reasoning_delta(0, 0, "overflow"))
            .expect_err("entry above ceiling must fail");
        assert_eq!(failure.failure_class, FailureClass::ProviderInternal);
        assert_eq!(failure.safe_message, OUTPUT_OVERFLOW_MESSAGE);
    }

    #[test]
    fn completed_reasoning_items_pass_encrypted_content_through() {
        let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
        let done = SseEvent {
            event: None,
            data: serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "rs_provider",
                    "type": "reasoning",
                    "summary": [],
                    "encrypted_content": "blob==",
                    "status": "completed",
                },
            })
            .to_string(),
        };
        let events = normalizer.feed(&done).expect("reasoning item completes");
        assert!(matches!(
            events.as_slice(),
            [Event::EncryptedReasoning {
                output_index: 0,
                encrypted_content,
            }] if encrypted_content == "blob=="
        ));

        // A reasoning item without the requested include stays silent.
        let bare = SseEvent {
            event: None,
            data: serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {"id": "rs_2", "type": "reasoning", "summary": []},
            })
            .to_string(),
        };
        assert!(normalizer.feed(&bare).expect("bare item").is_empty());
    }
}
