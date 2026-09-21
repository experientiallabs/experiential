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
    pub(super) fn completes(&self, event: &Event) -> bool {
        self.0.is_some() && event.is_terminal() && !matches!(event, Event::Failed(_))
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
        Event::TextDelta(_)
            | Event::RefusalDelta(_)
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
    )
}
