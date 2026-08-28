//! Bounded Responses continuation retention, including sealed Fireworks state.

use serde_json::{json, Value};

use crate::dialects::MAXIMUM_RETAINED_OUTPUT_BYTES;
use crate::encode::compact_json;
use crate::errors::PublicError;
use crate::events::{CompletedToolCall, Event};
use crate::relay::event_retained_bytes;
use crate::server::AppState;

#[derive(Default)]
pub(crate) struct ResponsesRetention {
    pub(crate) text: String,
    pub(crate) refusal: bool,
    pub(crate) tool_calls: Vec<CompletedToolCall>,
    pub(crate) carrier_events: Vec<Event>,
    pub(crate) retained_bytes: usize,
    pub(crate) overflowed: bool,
}

impl ResponsesRetention {
    pub(crate) fn track(&mut self, event: &Event) {
        if self.overflowed {
            return;
        }
        self.retained_bytes = self
            .retained_bytes
            .saturating_add(event_retained_bytes(event));
        if self.retained_bytes > MAXIMUM_RETAINED_OUTPUT_BYTES {
            self.overflowed = true;
            self.text.clear();
            self.tool_calls.clear();
            self.carrier_events.clear();
            return;
        }
        if matches!(
            event,
            Event::TextDelta(_)
                | Event::ReasoningContentDelta { .. }
                | Event::ToolCallStarted { .. }
                | Event::ToolArgumentsDelta { .. }
                | Event::ToolCallCompleted { .. }
        ) {
            self.carrier_events.push(event.clone());
        }
        match event {
            Event::TextDelta(delta) => self.text.push_str(delta),
            Event::RefusalDelta(_) => self.refusal = true,
            Event::ToolCallCompleted { call, .. } => self.tool_calls.push(call.clone()),
            _ => {}
        }
    }
}

fn remember_argument(
    request_id: &str,
    retention: &ResponsesRetention,
    reasoning_content_carrier: Option<&str>,
) -> String {
    compact_json(&json!({
        "request_id": request_id,
        "text": retention.text,
        "refusal": retention.refusal,
        "reasoning_content_carrier": reasoning_content_carrier,
        "tool_calls": retention.tool_calls.iter().map(|call| json!({
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.raw_arguments,
        })).collect::<Vec<Value>>(),
    }))
}

pub(crate) async fn remember_continuation(
    state: &AppState,
    request_id: &str,
    retention: &ResponsesRetention,
    reasoning_content_carrier: Option<&str>,
) -> Result<(), PublicError> {
    if retention.overflowed
        || retention.refusal
        || (retention.text.is_empty() && retention.tool_calls.is_empty())
    {
        return Ok(());
    }
    state
        .bridge
        .call(
            "remember",
            remember_argument(request_id, retention, reasoning_content_carrier),
        )
        .await
        .map(|_| ())
}
