//! Host-owned incremental inspection, separate from customer redaction policies.

use std::collections::BTreeMap;
use std::time::Instant;

use serde_json::{json, Value};

use crate::bridge::Bridge;
use crate::errors::{Failure, FailureClass};
use crate::events::Event;

const SEGMENT_BYTES: usize = 256;
const MAX_PENDING_BYTES: usize = 1_048_576;
// Tool completion events repeat the deltas' argument bytes. Bound this actual
// event payload separately without halving the inspectable-content limit.
const MAX_RETAINED_BYTES: usize = 2 * MAX_PENDING_BYTES;
const MAX_PENDING_EVENTS: usize = 1024;

/// A runtime error never implies that the content violated a policy.
fn unavailable() -> Failure {
    Failure::new(
        FailureClass::Unavailable,
        "Content inspection is unavailable. Retry later.",
    )
}

fn unsupported() -> Failure {
    Failure::new(
        FailureClass::UnsupportedCapability,
        "Content inspection does not support this output. Use a supported text request.",
    )
}

/// Rust retains exact events; the host receives only ordered inspectable text.
pub(crate) struct RuntimeInspector {
    request_id: String,
    pending: Vec<Event>,
    fragments: Vec<Value>,
    bytes: usize,
    retained_bytes: usize,
    tools: BTreeMap<(bool, u32), usize>,
    deadline: Instant,
}

impl RuntimeInspector {
    pub(crate) fn new(request_id: &str, deadline: Instant) -> Self {
        Self {
            request_id: request_id.to_owned(),
            pending: Vec::new(),
            fragments: Vec::new(),
            bytes: 0,
            retained_bytes: 0,
            tools: BTreeMap::new(),
            deadline,
        }
    }

    fn fragment(&mut self, kind: &str, channel: String, text: &str, name: Option<&str>) {
        self.bytes = self
            .bytes
            .saturating_add(text.len() + channel.len() + name.map_or(0, str::len));
        self.fragments
            .push(json!({"kind": kind, "channel": channel, "text": text, "name": name}));
    }

    /// Hold every tool frame until complete arguments have been inspected together.
    fn project(&mut self, event: &Event) -> Result<(), Failure> {
        match event {
            Event::TextDelta(text) => self.fragment("text", "text".into(), text, None),
            Event::RefusalDelta(text) => self.fragment("refusal", "refusal".into(), text, None),
            Event::ProviderTextDelta {
                output_index,
                item_id,
                delta,
            } => self.fragment(
                "text",
                format!("text:{output_index}:{item_id}"),
                delta,
                None,
            ),
            Event::ProviderRefusalDelta {
                output_index,
                item_id,
                delta,
            } => self.fragment(
                "refusal",
                format!("refusal:{output_index}:{item_id}"),
                delta,
                None,
            ),
            Event::ReasoningTextDelta(text) => {
                self.fragment("reasoning", "reasoning".into(), text, None)
            }
            Event::ReasoningContentDelta { delta, .. } => {
                self.fragment("reasoning", "reasoning".into(), delta, None)
            }
            Event::ThinkingDelta { index, delta } => {
                self.fragment("reasoning", format!("thinking:{index}"), delta, None)
            }
            Event::ReasoningSummaryDelta {
                output_index,
                item_id,
                summary_index,
                delta,
            } => self.fragment(
                "reasoning",
                format!("summary:{output_index}:{item_id}:{summary_index}"),
                delta,
                None,
            ),
            Event::ToolCallStarted { index, .. } => {
                if self.tools.insert((false, *index), 0).is_some() {
                    return Err(unavailable());
                }
            }
            Event::ServerToolUseStarted { index, .. } => {
                if self.tools.insert((true, *index), 0).is_some() {
                    return Err(unavailable());
                }
            }
            Event::ToolArgumentsDelta { index, delta }
            | Event::ServerToolArgumentsDelta { index, delta } => {
                let server = matches!(event, Event::ServerToolArgumentsDelta { .. });
                let Some(argument_bytes) = self.tools.get_mut(&(server, *index)) else {
                    return Err(unavailable());
                };
                *argument_bytes = argument_bytes.saturating_add(delta.len());
                self.bytes = self.bytes.saturating_add(delta.len());
            }
            Event::ToolCallCompleted { index, call }
            | Event::ServerToolUseCompleted { index, call } => {
                let server = matches!(event, Event::ServerToolUseCompleted { .. });
                let Some(argument_bytes) = self.tools.remove(&(server, *index)) else {
                    return Err(unavailable());
                };
                self.bytes = self.bytes.saturating_sub(argument_bytes);
                self.fragment(
                    "tool",
                    format!("tool:{server}:{index}:{}", call.call_id),
                    &call.raw_arguments,
                    Some(&call.name),
                );
            }
            Event::ServerToolResult { index, block } => {
                self.fragment("retrieved", format!("result:{index}"), block, None)
            }
            Event::HostedToolItemStarted {
                output_index, item, ..
            }
            | Event::HostedToolItemCompleted {
                output_index, item, ..
            } => self.fragment("retrieved", format!("hosted:{output_index}"), item, None),
            Event::HostedToolItemProgress {
                output_index,
                payload,
                ..
            } => self.fragment("retrieved", format!("hosted:{output_index}"), payload, None),
            Event::ProviderTextAnnotation {
                output_index,
                annotation,
                ..
            } => self.fragment(
                "retrieved",
                format!("annotation:{output_index}"),
                annotation,
                None,
            ),
            Event::CitationDelta { index, citation } => {
                self.fragment("retrieved", format!("citation:{index}"), citation, None)
            }
            Event::Image(_)
            | Event::ChoiceLogprobsDelta(_)
            | Event::ProviderResponsesLogprobs { .. } => return Err(unsupported()),
            Event::Failed(failure) => {
                self.fragment("text", "error".into(), &failure.safe_message, None)
            }
            // Opaque carriers and structural metadata contain no inspectable text.
            Event::GeminiThoughtPart(_)
            | Event::ThinkingSignature { .. }
            | Event::RedactedThinking { .. }
            | Event::EncryptedReasoning { .. }
            | Event::ProviderOutputItemStarted { .. }
            | Event::ProviderOutputItemCompleted { .. }
            | Event::TextBlockStarted { .. }
            | Event::Usage(_)
            | Event::Completed
            | Event::Incomplete
            | Event::StoppedAtSequence(_)
            | Event::PausedTurn => {}
        }
        Ok(())
    }

    /// Inspect before releasing this segment, under the original request deadline.
    pub(crate) async fn admit(
        &mut self,
        bridge: &Bridge,
        event: Event,
    ) -> Result<Vec<Event>, Failure> {
        self.retained_bytes = self
            .retained_bytes
            .saturating_add(crate::relay::event_retained_bytes(&event));
        self.project(&event)?;
        let final_segment = event.is_terminal();
        let tool_finished = matches!(
            event,
            Event::ToolCallCompleted { .. } | Event::ServerToolUseCompleted { .. }
        );
        self.pending.push(event);
        if self.bytes > MAX_PENDING_BYTES
            || self.retained_bytes > MAX_RETAINED_BYTES
            || self.pending.len() > MAX_PENDING_EVENTS
        {
            return Err(unsupported());
        }
        if final_segment && !self.tools.is_empty() {
            return Err(unavailable());
        }
        if !self.tools.is_empty()
            || (!final_segment && !tool_finished && self.bytes < SEGMENT_BYTES)
        {
            return Ok(Vec::new());
        }
        self.flush(bridge, final_segment).await
    }

    /// Inspect a complete gateway-generated prefix before encoders synthesize it.
    pub(crate) async fn flush(
        &mut self,
        bridge: &Bridge,
        final_segment: bool,
    ) -> Result<Vec<Event>, Failure> {
        if !self.tools.is_empty() {
            return Err(unavailable());
        }
        let argument = crate::encode::compact_json(&json!({
            "request_id": self.request_id, "fragments": self.fragments, "final": final_segment,
        }));
        let remaining = self.deadline.saturating_duration_since(Instant::now());
        let payload =
            tokio::time::timeout(remaining, bridge.call("inspect_runtime_output", argument))
                .await
                .map_err(|_| unavailable())?
                .map_err(|_| unavailable())?;
        let decision: super::OutputDecision =
            serde_json::from_str(&payload).map_err(|_| unavailable())?;
        if decision.action != "allow" {
            return Err(decision.failure.unwrap_or_else(unavailable));
        }
        self.fragments.clear();
        self.bytes = 0;
        self.retained_bytes = 0;
        Ok(std::mem::take(&mut self.pending))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_tool_deltas_fail_before_they_can_be_released() {
        let mut inspector = RuntimeInspector::new("request", Instant::now());
        let result = inspector.project(&Event::ToolArgumentsDelta {
            index: 7,
            delta: "uninspected".into(),
        });
        assert!(matches!(
            result,
            Err(Failure {
                failure_class: FailureClass::Unavailable,
                ..
            })
        ));
        assert!(inspector.fragments.is_empty());
    }

    #[test]
    fn tool_start_keeps_later_text_pending_until_the_tool_finishes() {
        let mut inspector = RuntimeInspector::new("request", Instant::now());
        inspector
            .project(&Event::ServerToolUseStarted {
                index: 1,
                call_id: "call".into(),
                name: "lookup".into(),
            })
            .unwrap();
        inspector
            .project(&Event::ServerToolArgumentsDelta {
                index: 1,
                delta: "{\"q\":\"split".into(),
            })
            .unwrap();
        inspector
            .project(&Event::TextDelta("later text".into()))
            .unwrap();
        assert!(inspector.tools.contains_key(&(true, 1)));
        assert_eq!(inspector.fragments.len(), 1);
        assert_eq!(inspector.fragments[0]["text"], "later text");
    }

    #[test]
    fn complete_tool_arguments_above_half_the_content_limit_are_counted_once() {
        let arguments = "x".repeat(600_000);
        let mut inspector = RuntimeInspector::new("request", Instant::now());
        let events = [
            Event::ServerToolUseStarted {
                index: 1,
                call_id: "call".into(),
                name: "lookup".into(),
            },
            Event::ServerToolArgumentsDelta {
                index: 1,
                delta: arguments.clone(),
            },
            Event::ServerToolUseCompleted {
                index: 1,
                call: crate::events::CompletedToolCall {
                    call_id: "call".into(),
                    name: "lookup".into(),
                    namespace: None,
                    caller: None,
                    provider_item_id: None,
                    provider_status: None,
                    raw_arguments: arguments,
                    custom: false,
                },
            },
        ];
        for event in &events {
            inspector.project(event).unwrap();
        }
        assert!(inspector.tools.is_empty());
        assert!(inspector.bytes > 600_000 && inspector.bytes < MAX_PENDING_BYTES);
        let retained: usize = events.iter().map(crate::relay::event_retained_bytes).sum();
        assert!(retained > MAX_PENDING_BYTES && retained < MAX_RETAINED_BYTES);
        assert_eq!(
            inspector.fragments[0]["text"].as_str().unwrap().len(),
            600_000
        );
    }

    #[test]
    fn output_channels_preserve_refusal_and_retrieval_provenance() {
        let mut inspector = RuntimeInspector::new("request", Instant::now());
        inspector
            .project(&Event::RefusalDelta("I cannot assist".into()))
            .unwrap();
        inspector
            .project(&Event::ServerToolResult {
                index: 2,
                block: "quoted material".into(),
            })
            .unwrap();
        assert_eq!(inspector.fragments[0]["kind"], "refusal");
        assert_eq!(inspector.fragments[1]["kind"], "retrieved");
    }

    #[test]
    fn images_are_unsupported_not_content_violations() {
        let mut inspector = RuntimeInspector::new("request", Instant::now());
        assert!(matches!(
            inspector.project(&Event::Image("image".into())),
            Err(Failure {
                failure_class: FailureClass::UnsupportedCapability,
                ..
            })
        ));
    }
}

/// Compose mandatory inspection after optional redaction and before encoding.
pub(crate) async fn inspect_events(
    inspector: Option<&mut RuntimeInspector>,
    bridge: &Bridge,
    events: Vec<Event>,
) -> Result<Vec<Event>, Failure> {
    let Some(inspector) = inspector else {
        return Ok(events);
    };
    let mut released = Vec::new();
    for event in events {
        released.extend(inspector.admit(bridge, event).await?);
    }
    Ok(released)
}
