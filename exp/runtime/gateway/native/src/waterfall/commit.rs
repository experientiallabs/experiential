//! The commit predicate: the first event that makes an attempt the answer.
//!
//! Outward semantic output commits the attempt: subsequent failure belongs
//! to this response and can never replay or splice another deployment.
//! Private reasoning is withheld until outward output or a successful
//! terminal selects its carrier. It can advance generation timing without
//! committing; refusal deltas withheld by policy do neither. Structural
//! output may commit before genuine generation and retains the first-token
//! allowance rather than starting the shorter generation-idle window.

use crate::events::{Event, Usage};

/// One coalesced, route-bound private reasoning carrier. The normalizer
/// charges every fragment against MAXIMUM_RETAINED_OUTPUT_BYTES before it
/// creates this event; coalescing also bounds per-fragment allocation overhead.
/// A losing attempt drops this buffer without sealing or exposing it.
#[derive(Default)]
pub(super) struct PrivateReasoning(Option<Event>);

impl PrivateReasoning {
    pub(super) fn withhold(&mut self, event: &Event, exposed: bool) -> bool {
        if let Event::ReasoningTextDelta(delta) = event {
            // Display-only reasoning waits for the turn's first real output:
            // a stall or an answerless stop before it still fails over, and
            // the coalesced text leads the committed prefix.
            if delta.is_empty() {
                return false;
            }
            match &mut self.0 {
                Some(Event::ReasoningTextDelta(held)) => held.push_str(delta),
                None => self.0 = Some(event.clone()),
                _ => return false,
            }
            return true;
        }
        let Event::ReasoningContentDelta {
            route_sha256,
            delta,
        } = event
        else {
            return false;
        };
        if exposed || delta.is_empty() {
            return false;
        }
        match &mut self.0 {
            Some(Event::ReasoningContentDelta {
                delta: held,
                route_sha256: route,
            }) if route == route_sha256 => held.push_str(delta),
            None => self.0 = Some(event.clone()),
            _ => return false, // Let the encoder reject an impossible route change.
        }
        true
    }

    /// Successful private-only terminals must still run the ordinary encoder
    /// and continuation seal path, not silently lose their carrier or usage.
    /// Display-only reasoning has no carrier: a turn that produced nothing
    /// else is an empty completion and keeps that handling.
    pub(super) fn completes(&self, event: &Event) -> bool {
        matches!(self.0, Some(Event::ReasoningContentDelta { .. }))
            && event.is_terminal()
            && !matches!(event, Event::Failed(_))
    }

    pub(super) fn prefix(
        &mut self,
        withheld: &mut Vec<Event>,
        event: Event,
        usage: Option<&Usage>,
    ) -> Vec<Event> {
        let mut prefix: Vec<_> = self.0.take().into_iter().collect();
        prefix.append(withheld);
        if event.is_terminal() {
            if let Some(usage) = usage {
                prefix.push(Event::Usage(usage.clone()));
            }
        }
        prefix.push(event);
        prefix
    }
}

/// Whether `event` carries model output that commits the attempt.
pub(crate) fn is_semantic(event: &Event) -> bool {
    if matches!(event, Event::ReasoningContentDelta { delta, .. } if delta.is_empty()) {
        return false;
    }
    matches!(
        event,
        Event::Image(_)
            | Event::TextDelta(_)
            | Event::RefusalDelta(_)
            | Event::ChoiceLogprobsDelta(_)
            | Event::ProviderTextDelta { .. }
            | Event::ProviderRefusalDelta { .. }
            | Event::ProviderOutputItemStarted { .. }
            | Event::ProviderOutputItemCompleted { .. }
            | Event::ReasoningSummaryDelta { .. }
            | Event::ThinkingDelta { .. }
            | Event::ThinkingSignature { .. }
            | Event::RedactedThinking { .. }
            | Event::EncryptedReasoning { .. }
            | Event::ReasoningContentDelta { .. }
            | Event::ToolCallStarted { .. }
            | Event::ToolArgumentsDelta { .. }
            | Event::ToolCallCompleted { .. }
            | Event::TextBlockStarted { .. }
            | Event::CitationDelta { .. }
            | Event::ServerToolUseStarted { .. }
            | Event::ServerToolArgumentsDelta { .. }
            | Event::ServerToolUseCompleted { .. }
            | Event::ServerToolResult { .. }
            | Event::HostedToolItemStarted { .. }
            | Event::HostedToolItemProgress { .. }
            | Event::HostedToolItemCompleted { .. }
            | Event::ProviderTextAnnotation { .. }
    ) || matches!(event, Event::ProviderResponsesLogprobs { records, .. }
        if records.as_array().is_some_and(|items| !items.is_empty()))
}

/// Bytes a turn's held tool calls may retain (each event charged its full
/// owned size) before they stream live.
pub(super) const MAXIMUM_HELD_TOOL_CALL_BYTES: usize = 262_144;
/// Events a turn's held tool calls may retain before they stream live.
pub(super) const MAXIMUM_HELD_TOOL_CALL_EVENTS: usize = 8_192;

/// The bytes `event` owns, or `None` once they exceed `room`: the larger of
/// the retention estimate and the event's debug rendering, counted by a
/// writer that never allocates and stops at `room`.
fn owned_bytes_within(event: &Event, room: usize) -> Option<usize> {
    struct Counter {
        bytes: usize,
        room: usize,
    }
    impl std::fmt::Write for Counter {
        fn write_str(&mut self, text: &str) -> std::fmt::Result {
            self.bytes += text.len();
            if self.bytes > self.room {
                return Err(std::fmt::Error);
            }
            Ok(())
        }
    }
    let estimate = crate::relay::event_retained_bytes(event);
    if estimate > room {
        return None;
    }
    let mut counter = Counter { bytes: 0, room };
    std::fmt::Write::write_fmt(&mut counter, format_args!("{event:?}")).ok()?;
    Some(estimate.max(counter.bytes))
}

/// What [`HeldToolCalls::consider`] decided for one event.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Hold {
    /// The event joined the hold; the attempt stays uncommitted.
    Held,
    /// Commit now: the held events lead the prefix, then this event.
    Release,
    /// A failure ended the held turn: the held events were dropped (the
    /// caller never saw them) and the failure takes the ordinary path.
    Discard,
    /// Not the hold's business: the ordinary commit rules decide.
    Pass,
}

/// A turn whose first outward output is a client tool call is held
/// uncommitted until it ends, so a call whose arguments never become one
/// valid JSON object (sampling, not the request) can still take the ladder:
/// the caller has seen nothing of it. Once the hold begins every later event
/// joins it in order, exactly as it would have streamed after a commit, so a
/// release replays the same sequence the caller would otherwise have
/// received. The hold is bounded: past [`MAXIMUM_HELD_TOOL_CALL_BYTES`] (a
/// large file write) the turn commits and streams live as before, and a tool
/// the provider itself executes commits at once (its effects must never be
/// repeated by a retry).
#[derive(Default)]
pub(super) struct HeldToolCalls {
    events: Vec<Event>,
    bytes: usize,
}

impl HeldToolCalls {
    /// Decide `event`. A hold starts only on a client function call's start
    /// while no refusal is withheld (`no_refusal_withheld`): withheld
    /// metadata (an empty logprobs delta) is fine, since it leads the held
    /// events in the prefix exactly as it arrived, while a withheld refusal
    /// keeps its own ladder rules. A translated freeform wrapper (Codex
    /// `custom` on a function wire) is not held: its inverter withholds the
    /// wrapper's bytes itself, so no size bound would ever release the hold
    /// and the caller would get nothing until the whole input was generated.
    pub(super) fn consider(&mut self, event: &Event, no_refusal_withheld: bool) -> Hold {
        if self.events.is_empty() {
            // Freeform (custom) input is opaque text that never fails to
            // parse, so it gains nothing from a hold and streams as before.
            // A Responses rung opens the function call's item scaffold
            // just before the call itself; the scaffold carries no call data,
            // so it starts the hold too.
            let opens_function_call = matches!(
                event,
                Event::ToolCallStarted { custom: false, .. }
                    | Event::ProviderOutputItemStarted {
                        kind: crate::events::ProviderOutputItemKind::FunctionCall,
                        ..
                    }
            );
            if no_refusal_withheld && opens_function_call {
                return self.push(event);
            }
            return Hold::Pass;
        }
        match event {
            Event::Failed(_) => {
                self.events.clear();
                self.bytes = 0;
                Hold::Discard
            }
            // Pre-commit usage stays tracked, never relayed: a release on the
            // terminal leads it with the tracked report, as any commit does.
            Event::Usage(_) => Hold::Held,
            event if event.is_terminal() => Hold::Release,
            event if super::buffered::provider_executed(event) => Hold::Release,
            event => self.push(event),
        }
    }

    fn push(&mut self, event: &Event) -> Hold {
        // The retention estimator leaves out identifiers, opaque `caller`
        // metadata and item ids the held clone owns; its debug rendering
        // carries every owned byte (escaped, so never fewer). It is measured
        // without allocating and stops as soon as it passes the room left.
        let room = MAXIMUM_HELD_TOOL_CALL_BYTES.saturating_sub(self.bytes);
        let fits = self.events.len() < MAXIMUM_HELD_TOOL_CALL_EVENTS;
        let Some(bytes) = owned_bytes_within(event, room).filter(|_| fits) else {
            return if self.events.is_empty() {
                Hold::Pass
            } else {
                Hold::Release
            };
        };
        self.bytes += bytes;
        self.events.push(event.clone());
        Hold::Held
    }

    /// The held events, in arrival order, leaving the hold empty.
    pub(super) fn take(&mut self) -> Vec<Event> {
        self.bytes = 0;
        std::mem::take(&mut self.events)
    }
}

#[cfg(test)]
mod held_tool_call_tests {
    use super::*;

    fn started() -> Event {
        Event::ToolCallStarted {
            custom: false,
            index: 0,
            call_id: "call_1".to_string(),
            name: "write".to_string(),
            namespace: None,
            caller: None,
            thought_signature: None,
        }
    }

    fn delta(bytes: usize) -> Event {
        Event::ToolArgumentsDelta {
            index: 0,
            delta: "x".repeat(bytes),
        }
    }

    #[test]
    fn only_a_client_tool_call_opening_with_nothing_withheld_starts_a_hold() {
        let mut held = HeldToolCalls::default();
        assert_eq!(
            held.consider(&Event::TextDelta("hi".into()), true),
            Hold::Pass
        );
        assert_eq!(held.consider(&started(), false), Hold::Pass);
        let server = Event::ServerToolUseStarted {
            index: 0,
            call_id: "srv_1".to_string(),
            name: "web_search".to_string(),
        };
        assert_eq!(held.consider(&server, true), Hold::Pass);
        let custom = Event::ToolCallStarted {
            custom: true,
            index: 0,
            call_id: "call_1".to_string(),
            name: "apply_patch".to_string(),
            namespace: None,
            caller: None,
            thought_signature: None,
        };
        assert_eq!(held.consider(&custom, true), Hold::Pass);
        assert_eq!(held.consider(&started(), true), Hold::Held);
    }

    #[test]
    fn a_held_turn_keeps_every_later_event_in_order_until_its_terminal() {
        let mut held = HeldToolCalls::default();
        assert_eq!(held.consider(&started(), true), Hold::Held);
        assert_eq!(held.consider(&delta(8), true), Hold::Held);
        assert_eq!(
            held.consider(&Event::TextDelta("after".into()), true),
            Hold::Held
        );
        assert_eq!(
            held.consider(&Event::Usage(Usage::default()), true),
            Hold::Held
        );
        assert_eq!(held.consider(&Event::Completed, true), Hold::Release);
        let events = held.take();
        assert_eq!(events.len(), 3, "usage stays tracked, never held");
        assert!(matches!(events[0], Event::ToolCallStarted { .. }));
        assert!(matches!(events[2], Event::TextDelta(_)));
    }

    #[test]
    fn a_responses_function_call_scaffold_starts_the_hold() {
        let mut held = HeldToolCalls::default();
        let scaffold = |kind| Event::ProviderOutputItemStarted {
            output_index: 0,
            item_id: Some("fc_1".to_string()),
            kind,
            status: None,
            phase: None,
        };
        assert_eq!(
            held.consider(
                &scaffold(crate::events::ProviderOutputItemKind::Message),
                true
            ),
            Hold::Pass
        );
        assert_eq!(
            held.consider(
                &scaffold(crate::events::ProviderOutputItemKind::FunctionCall),
                true
            ),
            Hold::Held
        );
        assert_eq!(held.consider(&started(), true), Hold::Held);
    }

    #[test]
    fn a_call_opening_is_charged_its_identifiers() {
        let mut held = HeldToolCalls::default();
        let long = Event::ToolCallStarted {
            custom: false,
            index: 0,
            call_id: "c".repeat(MAXIMUM_HELD_TOOL_CALL_BYTES),
            name: "write".to_string(),
            namespace: None,
            caller: None,
            thought_signature: None,
        };
        assert_eq!(held.consider(&long, true), Hold::Pass);
        held.consider(&started(), true);
        assert_eq!(held.consider(&long, true), Hold::Release);
    }

    #[test]
    fn a_call_opening_is_charged_its_caller_metadata() {
        let mut held = HeldToolCalls::default();
        held.consider(&started(), true);
        let heavy = Event::ToolCallStarted {
            custom: false,
            index: 1,
            call_id: "call_2".to_string(),
            name: "write".to_string(),
            namespace: None,
            caller: Some(serde_json::json!({"trace": "t".repeat(MAXIMUM_HELD_TOOL_CALL_BYTES)})),
            thought_signature: None,
        };
        assert_eq!(held.consider(&heavy, true), Hold::Release);
    }

    #[test]
    fn a_failure_discards_the_held_turn() {
        let mut held = HeldToolCalls::default();
        held.consider(&started(), true);
        held.consider(&delta(8), true);
        let failed = Event::Failed(crate::errors::Failure::new(
            crate::errors::FailureClass::MalformedResponse,
            "broken",
        ));
        assert_eq!(held.consider(&failed, true), Hold::Discard);
        assert!(held.take().is_empty());
    }

    #[test]
    fn a_large_or_provider_executed_turn_streams_live() {
        let mut held = HeldToolCalls::default();
        held.consider(&started(), true);
        assert_eq!(
            held.consider(&delta(MAXIMUM_HELD_TOOL_CALL_BYTES), true),
            Hold::Release
        );
        let mut held = HeldToolCalls::default();
        held.consider(&started(), true);
        let server = Event::ServerToolUseStarted {
            index: 1,
            call_id: "srv_1".to_string(),
            name: "web_search".to_string(),
        };
        assert_eq!(held.consider(&server, true), Hold::Release);
    }
}
