//! The terminal fields of a public Anthropic message: stop reason, stop
//! sequence, and the provider's relayed safeguard verdicts, shared by the
//! streaming `message_delta` and the non-streaming message body.

use serde_json::{json, Value};

use crate::events::Event;

/// Map the terminal outcome to the Anthropic stop reason. Server tool use is
/// provider-executed and deliberately never yields `tool_use`; a paused
/// server-tool turn must keep `pause_turn` so the caller resumes it.
pub(crate) fn stop_reason(terminal: &Event, saw_tool_use: bool) -> &'static str {
    match terminal {
        Event::Incomplete => "max_tokens",
        Event::PausedTurn => "pause_turn",
        // A gateway-emulated stop cut the visible text: the caller's sequence
        // ended the turn, exactly as Anthropic reports a native match.
        Event::StoppedAtSequence(_) => "stop_sequence",
        _ if saw_tool_use => "tool_use",
        _ => "end_turn",
    }
}

/// The matched stop sequence for the `stop_sequence` field, or null.
pub(crate) fn stop_sequence_value(terminal: &Event) -> Value {
    match terminal {
        Event::StoppedAtSequence(sequence) => Value::String(sequence.clone()),
        _ => Value::Null,
    }
}

/// The streaming `message_delta.delta` object for one terminal outcome.
///
/// `safeguard_results` is the upstream value, relayed unchanged at the same
/// position Anthropic sends it; the key is present exactly when the provider
/// sent it, so a rung that never reviewed the request never claims it did.
pub(crate) fn message_delta_body(
    terminal: &Event,
    saw_tool_use: bool,
    safeguard_results: Option<&Value>,
) -> Value {
    let mut delta = json!({
        "stop_reason": stop_reason(terminal, saw_tool_use),
        "stop_sequence": stop_sequence_value(terminal),
    });
    if let Some(results) = safeguard_results {
        delta["safeguard_results"] = results.clone();
    }
    delta
}

/// The last relayed safeguard verdicts in an ordered event list, if any.
pub(crate) fn last_safeguard_results(events: &[Event]) -> Option<&Value> {
    events.iter().rev().find_map(|event| match event {
        Event::SafeguardResults(results) => Some(results),
        _ => None,
    })
}
