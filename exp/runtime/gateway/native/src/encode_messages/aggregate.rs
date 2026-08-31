//! Non-streaming aggregation for the public Anthropic Messages surface,
//! split from `encode_messages` so the implementation stays within the
//! repository line budget.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::encode::stable_public_id;
use crate::errors::{Failure, PublicError};
use crate::events::{Event, Usage};

use super::{messages_usage, refusal_failure, stop_reason};

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
        if let Event::ToolCallCompleted { call, .. } | Event::ServerToolUseCompleted { call, .. } =
            event
        {
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
    if events.iter().any(|event| {
        matches!(
            event,
            Event::RefusalDelta(_) | Event::ProviderRefusalDelta { .. }
        )
    }) {
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
    let mut server_positions: HashMap<u32, usize> = HashMap::new();
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
            Event::TextBlockStarted { .. } => {
                // A provider block boundary starts a fresh text slot so
                // adjacent provider text blocks (and their citations) never
                // merge.
                slots.push(Some(json!({"type": "text", "text": ""})));
            }
            Event::CitationDelta { citation, .. } => {
                let position = slots.iter().rposition(
                    |slot| matches!(slot, Some(block) if block["type"] == json!("text")),
                );
                let position = match position {
                    Some(position) => position,
                    None => {
                        slots.push(Some(json!({"type": "text", "text": ""})));
                        slots.len() - 1
                    }
                };
                let parsed: Value =
                    serde_json::from_str(citation).map_err(|_| PublicError::internal())?;
                let block = slots[position].as_mut().expect("text slot is filled");
                match block.get_mut("citations") {
                    Some(Value::Array(citations)) => citations.push(parsed),
                    _ => {
                        block["citations"] = Value::Array(vec![parsed]);
                    }
                }
            }
            Event::TextDelta(delta) | Event::ProviderTextDelta { delta, .. }
                if !delta.is_empty() =>
            {
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
            Event::ServerToolUseStarted { index, .. } => {
                server_positions.insert(*index, slots.len());
                slots.push(None);
            }
            Event::ServerToolUseCompleted { index, call } => {
                // Provider-executed tool use anchors at its start position
                // and never contributes to the tool_use stop reason.
                if let Some(position) = server_positions.get(index) {
                    let input: Value = serde_json::from_str(&call.raw_arguments)
                        .map_err(|_| PublicError::internal())?;
                    slots[*position] = Some(json!({
                        "type": "server_tool_use",
                        "id": call.call_id,
                        "name": call.name,
                        "input": input,
                    }));
                }
            }
            Event::ServerToolResult { block, .. } => {
                let parsed: Value =
                    serde_json::from_str(block).map_err(|_| PublicError::internal())?;
                slots.push(Some(parsed));
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
        "stop_reason": stop_reason(terminal, saw_tool_use),
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
