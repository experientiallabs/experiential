//! Capture authorized provider evidence before public protocol projection.

use std::sync::Arc;

use super::collector::Collector;
use crate::admission::Admission;
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::Event;
use crate::settlement::AttemptGuard;
use crate::waterfall::Won;

/// Capture a hosted prompt before any committed output becomes client-visible.
/// Losing lanes never checkpoint: a later BYOK winner must not inherit a
/// host-funded lane's capture. Default collectors require durable acknowledgement;
/// explicitly asynchronous hosts wait for queue admission. SQL remains the final
/// live-consent authority.
pub(crate) async fn checkpoint_winner(
    collector: Option<&Arc<Collector>>,
    admission: &Admission,
    guard: &mut AttemptGuard,
    won: Won,
) -> Won {
    let (Some(collector), Won::Committed(attempt)) = (collector, &won) else {
        return won;
    };
    if admission.route.get(attempt.depth).is_some_and(|wire| {
        wire.billing_customer_managed || collector.checkpoint(&admission.request_id)
    }) {
        return won;
    }
    // Dispatch has already reserved a physical attempt. Request-only abandon
    // would disarm its drop backstop without closing that reservation.
    let usage = attempt.usage.clone();
    let tool_names = attempt.tool_names.clone();
    drop(won);
    guard
        .settle(
            "failed",
            usage.as_ref(),
            &tool_names,
            Some(&Failure::new(
                FailureClass::Internal,
                "capture checkpoint failed",
            )),
            true,
        )
        .await;
    Won::Failed(PublicError::internal())
}

pub(crate) struct Observer {
    collector: Arc<Collector>,
    request_id: String,
    reasoning_exposed: bool,
}

impl Observer {
    pub(crate) fn observe(&self, event: &Event) {
        match event {
            Event::GeminiThoughtPart(part) => {
                self.collector
                    .gemini_thought_part(&self.request_id, part.clone());
            }
            Event::ReasoningContentDelta { delta, .. } if self.reasoning_exposed => {
                self.collector.reasoning(&self.request_id, delta);
            }
            Event::ToolCallCompleted { call, .. } => {
                self.collector.tool_call(&self.request_id, call)
            }
            _ => {}
        }
    }
}

/// Only the selected attempt contributes. Capture policy still gates persistence;
/// private provider reasoning never becomes plaintext merely because capture is on.
pub(crate) fn observe_winner(
    collector: Option<Arc<Collector>>,
    admission: &Admission,
    guard: &crate::settlement::AttemptGuard,
    won: &mut Won,
) {
    let Some(collector) = collector else { return };
    let depth = match won {
        Won::Committed(attempt) => attempt.depth,
        Won::Settled(attempt) => attempt.depth,
        Won::Failed(_) => return,
    };
    let observer = Observer {
        collector,
        request_id: admission.request_id.clone(),
        reasoning_exposed: admission.reasoning_exposed_at(depth),
    };
    observer
        .collector
        .observe_attempt(&admission.request_id, guard.capture_observation());
    match won {
        Won::Committed(attempt) => {
            for event in &attempt.prefix {
                observer.observe(event);
            }
            attempt.relay.set_capture_reasoning(observer);
        }
        Won::Settled(attempt) => {
            for event in &attempt.events {
                observer.observe(event);
            }
        }
        Won::Failed(_) => {}
    }
}
