//! Bounded native Responses continuation aggregation.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Value};

use crate::dialects::MAXIMUM_RETAINED_OUTPUT_BYTES;
use crate::encode::compact_json;
use crate::errors::PublicError;
use crate::events::{
    CompletedToolCall, Event, ProviderAssistantMessagePhase, ProviderOutputItemKind,
    ProviderOutputItemStatus,
};
use crate::relay::event_retained_bytes;
use crate::server::AppState;

#[derive(Default)]
struct RetainedMessage {
    item_id: String,
    text: String,
    status: Option<ProviderOutputItemStatus>,
    phase: Option<ProviderAssistantMessagePhase>,
    done: bool,
}

#[derive(Default)]
struct RetainedReasoning {
    item_id: String,
    encrypted_content: String,
    status: Option<ProviderOutputItemStatus>,
    done: bool,
}

/// One hosted tool output item retained as its verbatim compact JSON so a
/// `previous_response_id` continuation can replay it byte-for-byte.
struct RetainedHostedItem {
    item: String,
}

/// Aggregated assistant output tracked while relaying one Responses stream.
#[derive(Default)]
pub(crate) struct ResponsesRetention {
    text: String,
    messages: BTreeMap<u32, RetainedMessage>,
    refusal: bool,
    tool_calls: Vec<(u32, CompletedToolCall, bool)>,
    provider_tools: BTreeSet<u32>,
    completed_tools: BTreeSet<u32>,
    tool_statuses: BTreeMap<u32, ProviderOutputItemStatus>,
    reasoning: BTreeMap<u32, RetainedReasoning>,
    hosted: BTreeMap<u32, RetainedHostedItem>,
    pub(crate) carrier_events: Vec<Event>,
    retained_bytes: usize,
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
            self.messages.clear();
            self.tool_calls.clear();
            self.reasoning.clear();
            self.hosted.clear();
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
            Event::ProviderOutputItemStarted {
                output_index,
                item_id: Some(item_id),
                kind: ProviderOutputItemKind::Message,
                status,
                phase,
            } => {
                self.messages.insert(
                    *output_index,
                    RetainedMessage {
                        item_id: item_id.clone(),
                        status: *status,
                        phase: *phase,
                        ..RetainedMessage::default()
                    },
                );
            }
            Event::ProviderOutputItemStarted {
                output_index,
                kind: ProviderOutputItemKind::FunctionCall | ProviderOutputItemKind::CustomToolCall,
                status,
                ..
            } => {
                self.provider_tools.insert(*output_index);
                if let Some(status) = status {
                    self.tool_statuses.insert(*output_index, *status);
                }
            }
            Event::ProviderOutputItemStarted {
                output_index,
                item_id: Some(item_id),
                kind: ProviderOutputItemKind::Reasoning,
                status,
                ..
            } => {
                self.reasoning.insert(
                    *output_index,
                    RetainedReasoning {
                        item_id: item_id.clone(),
                        status: *status,
                        ..RetainedReasoning::default()
                    },
                );
            }
            Event::ProviderOutputItemCompleted {
                output_index,
                kind: ProviderOutputItemKind::Message,
                status,
                phase,
                ..
            } => {
                if let Some(message) = self.messages.get_mut(output_index) {
                    message.status = status.or(message.status);
                    message.phase = phase.or(message.phase);
                    message.done = true;
                }
            }
            Event::ProviderOutputItemCompleted {
                output_index,
                kind: ProviderOutputItemKind::FunctionCall,
                status,
                ..
            } => {
                self.completed_tools.insert(*output_index);
                if let Some(status) = status {
                    self.tool_statuses.insert(*output_index, *status);
                }
            }
            Event::ProviderOutputItemCompleted {
                output_index,
                kind: ProviderOutputItemKind::Reasoning,
                status,
                ..
            } => {
                if let Some(reasoning) = self.reasoning.get_mut(output_index) {
                    reasoning.status = status.or(reasoning.status);
                    reasoning.done = true;
                }
            }
            Event::TextDelta(delta) => self.text.push_str(delta),
            Event::ProviderTextDelta {
                output_index,
                delta,
                ..
            } => {
                if let Some(message) = self.messages.get_mut(output_index) {
                    message.text.push_str(delta);
                }
            }
            Event::RefusalDelta(_) | Event::ProviderRefusalDelta { .. } => self.refusal = true,
            Event::ToolCallCompleted { index, call } => {
                let provider_owned = self.provider_tools.contains(index);
                let mut call = call.clone();
                call.provider_status = call
                    .provider_status
                    .or_else(|| self.tool_statuses.get(index).copied());
                self.tool_calls.push((*index, call, provider_owned));
            }
            Event::EncryptedReasoning {
                output_index,
                item_id,
                encrypted_content,
            } => {
                let reasoning = self.reasoning.entry(*output_index).or_default();
                reasoning.item_id = item_id.clone();
                reasoning.encrypted_content = encrypted_content.clone();
            }
            // The final verbatim item (the `done` shape when it arrived, else
            // the last-seen one) is what a continuation replays.
            Event::HostedToolItemStarted {
                output_index, item, ..
            }
            | Event::HostedToolItemCompleted {
                output_index, item, ..
            } => {
                self.hosted
                    .insert(*output_index, RetainedHostedItem { item: item.clone() });
            }
            Event::Completed => self.finish_open_items(ProviderOutputItemStatus::Completed),
            Event::Incomplete | Event::Failed(_) => {
                self.finish_open_items(ProviderOutputItemStatus::Incomplete);
            }
            _ => {}
        }
    }

    fn finish_open_items(&mut self, status: ProviderOutputItemStatus) {
        for message in self.messages.values_mut().filter(|item| !item.done) {
            message.status = Some(status);
            message.done = true;
        }
        for reasoning in self.reasoning.values_mut().filter(|item| !item.done) {
            reasoning.status = Some(status);
            reasoning.done = true;
        }
        for output_index in self
            .provider_tools
            .iter()
            .filter(|index| !self.completed_tools.contains(index))
        {
            self.tool_statuses.insert(*output_index, status);
        }
        for (output_index, call, provider_owned) in &mut self.tool_calls {
            if *provider_owned
                && matches!(
                    call.provider_status,
                    None | Some(ProviderOutputItemStatus::InProgress)
                )
            {
                call.provider_status = self.tool_statuses.get(output_index).copied();
            }
        }
    }

    pub(crate) fn refusal(&self) -> bool {
        self.refusal
    }
}

/// Build the retention payload consumed by the control plane's `remember`.
pub(crate) fn remember_argument(
    request_id: &str,
    retention: &ResponsesRetention,
    reasoning_content_carrier: Option<&str>,
) -> String {
    compact_json(&json!({
        "request_id": request_id,
        "text": retention.text,
        "message_outputs": retention.messages.iter().map(|(output_index, message)| json!({
            "output_index": output_index,
            "item_id": message.item_id,
            "text": message.text,
            "status": message.status.map(ProviderOutputItemStatus::as_str),
            "phase": message.phase.map(ProviderAssistantMessagePhase::as_str),
        })).collect::<Vec<Value>>(),
        "refusal": retention.refusal,
        "reasoning_content_carrier": reasoning_content_carrier,
        "hosted_items": retention.hosted.iter().map(|(output_index, hosted)| json!({
            "output_index": output_index,
            "item": serde_json::from_str::<Value>(&hosted.item).unwrap_or(Value::Null),
        })).collect::<Vec<Value>>(),
        "encrypted_reasoning": retention.reasoning.iter()
            .filter(|(_, reasoning)| !reasoning.encrypted_content.is_empty())
            .map(|(output_index, reasoning)| json!({
                "output_index": output_index,
                "item_id": reasoning.item_id,
                "encrypted_content": reasoning.encrypted_content,
                "status": reasoning.status.map(ProviderOutputItemStatus::as_str),
            }))
            .collect::<Vec<Value>>(),
        "tool_calls": retention
            .tool_calls
            .iter()
            .map(|(output_index, call, provider_owned)| json!({
                "output_index": provider_owned.then_some(output_index),
                "item_id": call.provider_item_id,
                "call_id": call.call_id,
                "name": call.name,
                "namespace": call.namespace,
                "caller": call.caller,
                "arguments": call.raw_arguments,
                "status": call.provider_status.map(ProviderOutputItemStatus::as_str),
                "custom": call.custom,
            }))
            .collect::<Vec<Value>>(),
    }))
}

/// Retain one finished Responses continuation before the terminal frames
/// flush, mirroring the python service's ordering. Returns the public error
/// when bounded retention fails closed.
///
/// An output-less turn (thinking spent the whole output budget, so the
/// response is `incomplete` with no items) is retained too: the caller holds
/// its response id, and `previous_response_id` naming it must resolve to the
/// conversation so far rather than `previous_response_not_found`.
pub(crate) async fn remember_continuation(
    state: &AppState,
    request_id: &str,
    retention: &ResponsesRetention,
    reasoning_content_carrier: Option<&str>,
) -> Result<(), PublicError> {
    if retention.overflowed || retention.refusal() {
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
    fn retention_preserves_multi_message_status_phase_and_idless_call() {
        let events = [
            Event::ProviderOutputItemStarted {
                output_index: 0,
                item_id: Some("rs-0".to_string()),
                kind: ProviderOutputItemKind::Reasoning,
                status: Some(ProviderOutputItemStatus::InProgress),
                phase: None,
            },
            Event::EncryptedReasoning {
                output_index: 0,
                item_id: "rs-0".to_string(),
                encrypted_content: "opaque".to_string(),
            },
            Event::ProviderOutputItemCompleted {
                output_index: 0,
                item_id: Some("rs-0".to_string()),
                kind: ProviderOutputItemKind::Reasoning,
                status: Some(ProviderOutputItemStatus::Incomplete),
                phase: None,
            },
            Event::ProviderOutputItemStarted {
                output_index: 1,
                item_id: Some("msg-commentary".to_string()),
                kind: ProviderOutputItemKind::Message,
                status: Some(ProviderOutputItemStatus::InProgress),
                phase: Some(ProviderAssistantMessagePhase::Commentary),
            },
            Event::ProviderTextDelta {
                output_index: 1,
                item_id: "msg-commentary".to_string(),
                delta: "Checking.".to_string(),
            },
            Event::ProviderOutputItemCompleted {
                output_index: 1,
                item_id: Some("msg-commentary".to_string()),
                kind: ProviderOutputItemKind::Message,
                status: Some(ProviderOutputItemStatus::Incomplete),
                phase: Some(ProviderAssistantMessagePhase::Commentary),
            },
            Event::ProviderOutputItemStarted {
                output_index: 2,
                item_id: None,
                kind: ProviderOutputItemKind::FunctionCall,
                status: Some(ProviderOutputItemStatus::InProgress),
                phase: None,
            },
            Event::ProviderOutputItemCompleted {
                output_index: 2,
                item_id: None,
                kind: ProviderOutputItemKind::FunctionCall,
                status: Some(ProviderOutputItemStatus::Incomplete),
                phase: None,
            },
            Event::ToolCallCompleted {
                index: 2,
                call: CompletedToolCall {
                    namespace: None,
                    caller: None,
                    call_id: "call-required".to_string(),
                    name: "lookup".to_string(),
                    provider_item_id: None,
                    provider_status: Some(ProviderOutputItemStatus::Incomplete),
                    raw_arguments: "{}".to_string(),
                    custom: false,
                },
            },
            Event::ProviderOutputItemStarted {
                output_index: 3,
                item_id: Some("msg-final".to_string()),
                kind: ProviderOutputItemKind::Message,
                status: Some(ProviderOutputItemStatus::InProgress),
                phase: Some(ProviderAssistantMessagePhase::FinalAnswer),
            },
            Event::ProviderTextDelta {
                output_index: 3,
                item_id: "msg-final".to_string(),
                delta: "Done.".to_string(),
            },
            Event::ProviderOutputItemCompleted {
                output_index: 3,
                item_id: Some("msg-final".to_string()),
                kind: ProviderOutputItemKind::Message,
                status: Some(ProviderOutputItemStatus::Completed),
                phase: Some(ProviderAssistantMessagePhase::FinalAnswer),
            },
            Event::Incomplete,
        ];
        let mut retention = ResponsesRetention::default();
        for event in &events {
            retention.track(event);
        }
        let payload: Value =
            serde_json::from_str(&remember_argument("request-1", &retention, None))
                .expect("retention payload is JSON");
        assert_eq!(payload["encrypted_reasoning"][0]["status"], "incomplete");
        assert_eq!(payload["message_outputs"][0]["status"], "incomplete");
        assert_eq!(payload["message_outputs"][0]["phase"], "commentary");
        assert_eq!(payload["message_outputs"][1]["phase"], "final_answer");
        assert_eq!(payload["tool_calls"][0]["output_index"], 2);
        assert!(payload["tool_calls"][0]["item_id"].is_null());
        assert_eq!(payload["tool_calls"][0]["status"], "incomplete");
    }

    #[test]
    fn retention_carries_a_tool_call_namespace_to_the_remember_payload() {
        let events = [
            Event::ToolCallCompleted {
                index: 0,
                call: CompletedToolCall {
                    namespace: Some("collaboration".to_string()),
                    caller: None,
                    call_id: "call-ns".to_string(),
                    name: "spawn_agent".to_string(),
                    provider_item_id: None,
                    provider_status: None,
                    raw_arguments: "{}".to_string(),
                    custom: false,
                },
            },
            Event::Completed,
        ];
        let mut retention = ResponsesRetention::default();
        for event in &events {
            retention.track(event);
        }
        let payload: Value =
            serde_json::from_str(&remember_argument("request-ns", &retention, None))
                .expect("retention payload is JSON");
        assert_eq!(payload["tool_calls"][0]["namespace"], "collaboration");
        assert_eq!(payload["tool_calls"][0]["name"], "spawn_agent");
    }

    #[test]
    fn hosted_items_retain_their_final_verbatim_json_at_their_index() {
        let events = [
            Event::HostedToolItemStarted {
                output_index: 0,
                item_id: "ws_1".to_string(),
                item_type: "web_search_call".to_string(),
                item: "{\"id\":\"ws_1\",\"type\":\"web_search_call\",\"status\":\"in_progress\"}"
                    .to_string(),
            },
            Event::HostedToolItemCompleted {
                output_index: 0,
                item_id: "ws_1".to_string(),
                item_type: "web_search_call".to_string(),
                item: "{\"id\":\"ws_1\",\"type\":\"web_search_call\",\"status\":\"completed\",\
                       \"action\":{\"type\":\"search\",\"query\":\"pi\"}}"
                    .to_string(),
            },
            Event::ProviderOutputItemStarted {
                output_index: 1,
                item_id: Some("msg_1".to_string()),
                kind: ProviderOutputItemKind::Message,
                status: None,
                phase: None,
            },
            Event::ProviderTextDelta {
                output_index: 1,
                item_id: "msg_1".to_string(),
                delta: "3.14159".to_string(),
            },
            Event::Completed,
        ];
        let mut retention = ResponsesRetention::default();
        for event in &events {
            retention.track(event);
        }
        let payload: Value =
            serde_json::from_str(&remember_argument("request-ws", &retention, None))
                .expect("retention payload is JSON");
        assert_eq!(payload["hosted_items"][0]["output_index"], 0);
        // The done-frame item (not the added one) is what a continuation replays.
        assert_eq!(payload["hosted_items"][0]["item"]["action"]["query"], "pi");
        assert_eq!(payload["message_outputs"][0]["item_id"], "msg_1");
    }

    #[test]
    fn retention_carries_a_tool_call_caller_to_the_remember_payload() {
        let events = [
            Event::ToolCallCompleted {
                index: 0,
                call: CompletedToolCall {
                    namespace: None,
                    caller: Some(serde_json::json!({"type": "program", "id": "prog_1"})),
                    call_id: "call-caller".to_string(),
                    name: "lookup".to_string(),
                    provider_item_id: None,
                    provider_status: None,
                    raw_arguments: "{}".to_string(),
                    custom: false,
                },
            },
            Event::Completed,
        ];
        let mut retention = ResponsesRetention::default();
        for event in &events {
            retention.track(event);
        }
        let payload: Value =
            serde_json::from_str(&remember_argument("request-caller", &retention, None))
                .expect("retention payload is JSON");
        assert_eq!(payload["tool_calls"][0]["caller"]["id"], "prog_1");
        assert_eq!(payload["tool_calls"][0]["caller"]["type"], "program");
    }

    #[test]
    fn empty_provider_message_still_retains_lifecycle_identity() {
        let events = [
            Event::ProviderOutputItemStarted {
                output_index: 0,
                item_id: Some("msg-empty".to_string()),
                kind: ProviderOutputItemKind::Message,
                status: Some(ProviderOutputItemStatus::InProgress),
                phase: Some(ProviderAssistantMessagePhase::Commentary),
            },
            Event::ProviderOutputItemCompleted {
                output_index: 0,
                item_id: Some("msg-empty".to_string()),
                kind: ProviderOutputItemKind::Message,
                status: Some(ProviderOutputItemStatus::Incomplete),
                phase: Some(ProviderAssistantMessagePhase::Commentary),
            },
            Event::Incomplete,
        ];
        let mut retention = ResponsesRetention::default();
        for event in &events {
            retention.track(event);
        }

        let payload: Value =
            serde_json::from_str(&remember_argument("request-1", &retention, None))
                .expect("retention payload is JSON");
        assert_eq!(payload["message_outputs"][0]["item_id"], "msg-empty");
        assert_eq!(payload["message_outputs"][0]["text"], "");
        assert_eq!(payload["message_outputs"][0]["status"], "incomplete");
        assert_eq!(payload["message_outputs"][0]["phase"], "commentary");
    }
}
