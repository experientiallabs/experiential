//! JSON-typed output-chain boundary for identity-scoped guardrails.
//!
//! Rust owns buffering and delivery. Python owns policy lookup and classifier
//! adapters. This module never logs request text, completions, or replacements.

use serde::Deserialize;
use serde_json::{json, Value};

use crate::bridge::Bridge;
use crate::errors::{Failure, FailureClass};
use crate::events::Event;

/// Decision returned by one Python `enforce_output` callback.
#[derive(Debug, Deserialize)]
struct OutputDecision {
    action: String,
    #[serde(default)]
    replacement_text: Option<String>,
    #[serde(default)]
    failure: Option<Failure>,
}

/// Build one fail-closed guardrail failure without request content.
fn closed_failure() -> Failure {
    Failure::new(
        FailureClass::Guardrail,
        "A gateway guardrail could not complete this request.",
    )
}

/// Prefer the classifier-supplied failure when it is already sanitized.
fn decision_failure(decision: &OutputDecision) -> Failure {
    decision.failure.clone().unwrap_or_else(|| {
        let message = if decision.action == "block" {
            "The request was blocked by a gateway guardrail."
        } else {
            "A gateway guardrail could not complete this request."
        };
        Failure::new(FailureClass::Guardrail, message)
    })
}

/// Project collected events into the JSON payload Python inspects once.
pub fn output_argument(request_id: &str, events: &[Event]) -> String {
    let mut text = String::new();
    let mut refusal = false;
    let mut tool_calls: Vec<Value> = Vec::new();
    for event in events {
        match event {
            Event::TextDelta(delta) => text.push_str(delta),
            Event::RefusalDelta(_) => refusal = true,
            Event::ToolCallCompleted { call, .. } => {
                tool_calls.push(json!({
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": call.raw_arguments,
                }));
            }
            _ => {}
        }
    }
    crate::encode::compact_json(&json!({
        "request_id": request_id,
        "text": text,
        "refusal": refusal,
        "tool_calls": tool_calls,
    }))
}

/// Replace text deltas with one rewritten delta. Tool-call events stay intact.
pub fn apply_text_replacement(events: &[Event], replacement: &str) -> Vec<Event> {
    let mut rewritten = Vec::with_capacity(events.len());
    let mut inserted = false;
    for event in events {
        match event {
            Event::TextDelta(_) => {
                if inserted {
                    continue;
                }
                rewritten.push(Event::TextDelta(replacement.to_string()));
                inserted = true;
            }
            Event::Completed | Event::Incomplete | Event::Failed(_) if !inserted => {
                rewritten.push(Event::TextDelta(replacement.to_string()));
                inserted = true;
                rewritten.push(event.clone());
            }
            other => rewritten.push(other.clone()),
        }
    }
    rewritten
}

/// Invoke the Python output chain once and return the validated events.
pub async fn enforce_collected_output(
    bridge: &Bridge,
    request_id: &str,
    events: Vec<Event>,
) -> Result<Vec<Event>, Failure> {
    let argument = output_argument(request_id, &events);
    let payload = bridge
        .call("enforce_output", argument)
        .await
        .map_err(|_| closed_failure())?;
    let decision: OutputDecision = serde_json::from_str(&payload).map_err(|_| closed_failure())?;
    match decision.action.as_str() {
        "allow" => Ok(events),
        "modify" => {
            if events
                .iter()
                .any(|event| matches!(event, Event::ToolCallCompleted { .. }))
            {
                return Err(Failure::new(
                    FailureClass::Guardrail,
                    "The request was blocked by a gateway guardrail.",
                ));
            }
            let replacement = decision
                .replacement_text
                .as_deref()
                .ok_or_else(closed_failure)?;
            Ok(apply_text_replacement(&events, replacement))
        }
        "block" | "error" => Err(decision_failure(&decision)),
        _ => Err(closed_failure()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::CompletedToolCall;

    #[test]
    fn output_argument_is_content_shaped_and_request_keyed() {
        let events = vec![
            Event::TextDelta("hello".to_string()),
            Event::ToolCallCompleted {
                index: 0,
                call: CompletedToolCall {
                    call_id: "call-1".to_string(),
                    name: "lookup".to_string(),
                    raw_arguments: "{\"q\":\"x\"}".to_string(),
                },
            },
            Event::Completed,
        ];
        let payload: Value = serde_json::from_str(&output_argument("req-1", &events)).unwrap();
        assert_eq!(payload["request_id"], "req-1");
        assert_eq!(payload["text"], "hello");
        assert_eq!(payload["refusal"], false);
        assert_eq!(payload["tool_calls"][0]["name"], "lookup");
        assert_eq!(payload["tool_calls"][0]["arguments"], "{\"q\":\"x\"}");
    }

    #[test]
    fn text_replacement_collapses_deltas_and_leaves_tool_calls() {
        let events = vec![
            Event::TextDelta("hel".to_string()),
            Event::TextDelta("lo".to_string()),
            Event::ToolCallCompleted {
                index: 0,
                call: CompletedToolCall {
                    call_id: "call-1".to_string(),
                    name: "lookup".to_string(),
                    raw_arguments: "{}".to_string(),
                },
            },
            Event::Completed,
        ];
        let rewritten = apply_text_replacement(&events, "safe");
        let texts: Vec<&str> = rewritten
            .iter()
            .filter_map(|event| match event {
                Event::TextDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, ["safe"]);
        assert!(rewritten
            .iter()
            .any(|event| matches!(event, Event::ToolCallCompleted { .. })));
    }

    #[test]
    fn missing_text_inserts_replacement_before_terminal() {
        let events = vec![Event::Completed];
        let rewritten = apply_text_replacement(&events, "safe");
        assert!(matches!(rewritten[0], Event::TextDelta(ref text) if text == "safe"));
        assert!(matches!(rewritten[1], Event::Completed));
    }
}
