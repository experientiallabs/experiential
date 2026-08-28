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
    pub(crate) tool_calls: Vec<(u32, CompletedToolCall)>,
    pub(crate) encrypted_reasoning: Vec<(u32, String, String)>,
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
            self.encrypted_reasoning.clear();
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
            Event::ToolCallCompleted { index, call } => {
                self.tool_calls.push((*index, call.clone()));
            }
            Event::EncryptedReasoning {
                output_index,
                item_id,
                encrypted_content,
            } => self.encrypted_reasoning.push((
                *output_index,
                item_id.clone(),
                encrypted_content.clone(),
            )),
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
        "encrypted_reasoning": retention.encrypted_reasoning.iter().map(
            |(output_index, item_id, encrypted_content)| json!({
                "output_index": output_index,
                "item_id": item_id,
                "encrypted_content": encrypted_content,
            })
        ).collect::<Vec<Value>>(),
        "tool_calls": retention.tool_calls.iter().map(|(output_index, call)| json!({
            "output_index": output_index,
            "item_id": call.provider_item_id,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encrypted_reasoning_is_retained_for_the_private_bridge_payload() {
        let mut retention = ResponsesRetention::default();
        retention.track(&Event::EncryptedReasoning {
            output_index: 3,
            item_id: "rs-provider".to_string(),
            encrypted_content: "provider-opaque".to_string(),
        });

        let payload: Value =
            serde_json::from_str(&remember_argument("request-one", &retention, None))
                .expect("remember payload must be valid JSON");

        assert_eq!(
            payload["encrypted_reasoning"],
            json!([{
                "output_index": 3,
                "item_id": "rs-provider",
                "encrypted_content": "provider-opaque",
            }])
        );
    }
}
