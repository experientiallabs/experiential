//! Capture authorized provider evidence before public protocol projection.

use std::sync::Arc;
use std::time::Instant;

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
    mut won: Won,
    deadline: Instant,
) -> Won {
    let (Some(collector), Won::Committed(attempt)) = (collector, &mut won) else {
        return won;
    };
    if admission
        .route
        .get(attempt.depth)
        .is_some_and(|wire| wire.billing_customer_managed)
    {
        return won;
    }
    let receipt = collector.checkpoint_receipt(&admission.request_id);
    let mut unfinished = UnexposedCheckpoint {
        collector,
        request_id: &admission.request_id,
        armed: true,
    };
    let failure = match receipt {
        Ok(None) => {
            unfinished.armed = false;
            return won;
        }
        Ok(Some(receipt)) => match tokio::time::timeout_at(deadline.into(), receipt).await {
            Ok(Ok(true)) => {
                unfinished.armed = false;
                return won;
            }
            Err(_) => Failure::new(
                FailureClass::Timeout,
                "capture checkpoint deadline exceeded",
            ),
            _ => Failure::new(FailureClass::Internal, "capture checkpoint failed"),
        },
        Err(()) => Failure::new(FailureClass::Internal, "capture checkpoint failed"),
    };
    // Close the physical transport before any durable settlement callback can wait.
    attempt.relay.close_transport();
    let usage = attempt.usage.clone();
    let tool_names = attempt.tool_names.clone();
    drop(won);
    drop(unfinished);
    if failure.failure_class == FailureClass::Timeout {
        guard.settle_cancelled(usage.as_ref(), &tool_names).await;
    } else {
        guard
            .settle("failed", usage.as_ref(), &tool_names, Some(&failure), true)
            .await;
    }
    Won::Failed(if failure.failure_class == FailureClass::Timeout {
        crate::relay::collection_public_error(&failure)
    } else {
        PublicError::internal()
    })
}

struct UnexposedCheckpoint<'a> {
    collector: &'a Collector,
    request_id: &'a str,
    armed: bool,
}

impl Drop for UnexposedCheckpoint<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.collector.finish_unexposed(self.request_id);
        }
    }
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
