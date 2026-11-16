//! Draining a committed attempt whole, for surfaces that answer only once
//! the terminal arrived.

use std::time::{Duration, Instant};

use super::{
    ended_without_terminal, event_retained_bytes, track_event, MAXIMUM_RETAINED_OUTPUT_BYTES,
};
use crate::dialects::OUTPUT_OVERFLOW_MESSAGE;
use crate::errors::{Failure, FailureClass};
use crate::events::Event;
use crate::waterfall::CommittedAttempt;

/// Drain one non-streaming attempt, bounding output like the Python aggregation.
pub async fn collect_committed(
    committed: &mut CommittedAttempt,
    deadline: Instant,
    phase_timeout: Duration,
    request_started: Instant,
) -> Result<Vec<Event>, Failure> {
    collect_committed_observed(committed, deadline, phase_timeout, request_started, |_| {}).await
}

/// [`collect_committed`], showing `observe` every event drained, the ones a
/// failure discards included.
pub(crate) async fn collect_committed_observed(
    committed: &mut CommittedAttempt,
    deadline: Instant,
    phase_timeout: Duration,
    request_started: Instant,
    mut observe: impl FnMut(&Event),
) -> Result<Vec<Event>, Failure> {
    let mut events: Vec<Event> = Vec::new();
    let mut retained_bytes = 0usize;
    let mut lease = crate::retained::RetainedLease::default();
    let mut retain = |events: &mut Vec<Event>, event: Event| -> Result<(), Failure> {
        let bytes = event_retained_bytes(&event);
        retained_bytes = retained_bytes.saturating_add(bytes);
        if retained_bytes > MAXIMUM_RETAINED_OUTPUT_BYTES {
            return Err(Failure::new(
                FailureClass::ProviderInternal,
                OUTPUT_OVERFLOW_MESSAGE,
            ));
        }
        if !lease.reserve(bytes) {
            return Err(Failure::new(
                FailureClass::Unavailable,
                crate::retained::WORKER_RETENTION_EXHAUSTED_MESSAGE,
            ));
        }
        events.push(event);
        Ok(())
    };
    if let Some(failure) = committed.collection_failure.take() {
        return Err(failure);
    }
    for event in committed.prefix.drain(..) {
        observe(&event);
        retain(&mut events, event)?;
    }
    if events.last().is_some_and(Event::is_terminal) {
        return Ok(events);
    }
    loop {
        let next = committed
            .relay
            .next_event(deadline, phase_timeout, request_started)
            .await;
        if matches!(next, Err(_) | Ok(None) | Ok(Some(Event::Failed(_)))) {
            committed.usage = committed.relay.usage_before_failure(committed.usage.take());
        }
        match next? {
            Some(event) => {
                observe(&event);
                track_event(&event, &mut committed.usage, &mut committed.tool_names);
                let terminal = event.is_terminal();
                retain(&mut events, event)?;
                if terminal {
                    return Ok(events);
                }
            }
            None => return Err(ended_without_terminal()),
        }
    }
}
