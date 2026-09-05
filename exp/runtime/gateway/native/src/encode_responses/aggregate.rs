//! Non-streaming Responses aggregation over normalized events.

use super::*;

/// Usage shape from `exp.runtime.openai_protocol.streaming._responses_usage`.
pub(super) fn responses_usage(usage: Option<&Usage>) -> Value {
    let usage = match usage {
        Some(usage) if usage.has_token_counts() => usage,
        _ => return Value::Null,
    };
    let input = usage.input_tokens.unwrap_or(0);
    let output = usage.output_tokens.unwrap_or(0);
    json!({
        "input_tokens": input,
        // `cache_write_tokens` joined the official shape (openai-python 3.x
        // marks it required), so SDK-strict callers need it present.
        "input_tokens_details": {
            "cached_tokens": usage.cached_input_tokens.unwrap_or(0),
            "cache_write_tokens": usage.cache_creation_input_tokens.unwrap_or(0),
        },
        "output_tokens": output,
        "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens.unwrap_or(0)},
        "total_tokens": input + output,
    })
}

/// The aggregated non-streaming Responses outcome from one event stream.
pub struct AggregatedResponses {
    pub body: Value,
    pub failure: Option<Failure>,
    pub usage: Option<Usage>,
    pub incomplete: bool,
    pub tool_names: Vec<String>,
}

/// Build one non-streaming public Responses result from ordered events.
pub fn completed_responses_body(
    request_id: &str,
    model: &str,
    created_at: f64,
    envelope: ResponsesEnvelope,
    events: &[Event],
) -> Result<AggregatedResponses, PublicError> {
    completed_responses_body_with_carrier(request_id, model, created_at, envelope, events, None)
}

/// Build one non-streaming result with an authenticated Fireworks carrier.
pub fn completed_responses_body_with_carrier(
    request_id: &str,
    model: &str,
    created_at: f64,
    envelope: ResponsesEnvelope,
    events: &[Event],
    reasoning_content_carrier: Option<&str>,
) -> Result<AggregatedResponses, PublicError> {
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
    let mut tool_names = Vec::new();
    for event in events {
        match event {
            Event::ToolCallCompleted { call, .. } => {
                if !tool_names.contains(&call.name) {
                    tool_names.push(call.name.clone());
                }
            }
            // Hosted tool invocations are provider-executed but still
            // invoked tools; their item type names the activity for the
            // ledger, mirroring `track_event`.
            Event::HostedToolItemCompleted { item_type, .. } => {
                if !tool_names.contains(item_type) {
                    tool_names.push(item_type.clone());
                }
            }
            _ => {}
        }
    }
    if let Event::Failed(failure) = terminal {
        return Ok(AggregatedResponses {
            body: Value::Null,
            failure: Some(failure.clone()),
            usage,
            incomplete: false,
            tool_names,
        });
    }
    let mut encoder = ResponsesSseEncoder::new(request_id, model, created_at, envelope);
    if let Some(carrier) = reasoning_content_carrier {
        encoder.set_reasoning_content_carrier(carrier.to_string())?;
    }
    encoder.start()?;
    let mut terminal_frames = Vec::new();
    for event in events {
        let produced = encoder.feed(event)?;
        if event.is_terminal() {
            terminal_frames = produced;
        }
        if encoder.saw_terminal() {
            break;
        }
    }
    let last = terminal_frames.last().ok_or_else(|| {
        PublicError::new(
            502,
            "all_routes_failed",
            "Responses encoding produced no terminal result.",
            "api_error",
        )
    })?;
    let data = last
        .split_once("data: ")
        .map(|(_, tail)| tail)
        .unwrap_or_default();
    let payload: Value =
        serde_json::from_str(data.trim_end()).map_err(|_| PublicError::internal())?;
    let body = payload
        .get("response")
        .cloned()
        .ok_or_else(PublicError::internal)?;
    Ok(AggregatedResponses {
        body,
        failure: None,
        usage,
        incomplete: matches!(terminal, Event::Incomplete),
        tool_names,
    })
}
