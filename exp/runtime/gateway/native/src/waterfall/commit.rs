//! The commit predicate: the first event that makes an attempt the answer.
//!
//! An attempt is COMMITTED on its first semantic event -- the first sign of
//! the model's own output (content, reasoning, a tool call, an output item)
//! rather than stream scaffolding (role-only chunks, pings, keepalive
//! comments, usage). Before commit a failure fails over to the next rung;
//! after it the attempt is the response and a failure is the caller's. The
//! relay's fail-fast first-token bound is armed until exactly this event
//! (`relay.rs`), so the two agree on the moment a stall stops being
//! failover-eligible.

use crate::events::Event;

/// Whether `event` carries model output that commits the attempt.
pub(crate) fn is_semantic(event: &Event) -> bool {
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
