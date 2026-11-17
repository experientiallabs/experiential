//! Buffered surfaces: drain a committed attempt inside the waterfall.
//!
//! A non-streaming request (or a stream an output guardrail buffers) sends
//! the caller nothing until the attempt's terminal. Commitment there is the
//! engine's bookkeeping, not a byte on the wire, so a connection the provider
//! breaks mid-body is still a failure the ladder may take: the attempt is
//! settled failed with whatever usage it reported, and the next reservation restarts the answer on a sibling rung
//! (or redials the same one), with a new attempt row per dial exactly like a
//! pre-commit failure. A streaming surface never comes through here: once a
//! delta reached the caller, the stream ends with the protocol's own error
//! event instead of a faked continuation. An attempt whose provider already
//! ran a tool (a hosted MCP call, a server tool) is never restarted either:
//! that work may have had effects, so a second dial could repeat them.

use std::time::Duration;

use super::{AttemptEnd, CommittedAttempt, DeploymentWire, WaterfallContext};
use crate::errors::{Failure, FailureClass};
use crate::events::Event;
use crate::relay::collect_committed_observed;
use crate::settlement::AttemptGuard;

/// Whether a failure met while draining a buffered attempt may restart it on
/// the ladder: an upstream connection, stall or provider-side fault the data
/// plane classified as redialable or failover-eligible, or a provider
/// content-filter refusal when `content_filter_failover` (the waterfall's
/// verdict that another provider's rung may still take one). Caller-owned,
/// model-content and gateway-internal failures (an output overflow, an
/// exhausted retention budget) keep the surface's own answer.
pub(super) fn restartable(failure: &Failure, content_filter_failover: bool) -> bool {
    let broken = matches!(
        failure.failure_class,
        FailureClass::Transport
            | FailureClass::Timeout
            | FailureClass::ProviderInternal
            | FailureClass::MalformedResponse
    ) && (failure.failover_eligible || failure.retryable_same_deployment);
    // The provider's own content filter cut an answer the caller has not
    // received, and another provider's rung may still take it.
    broken || (failure.provider_content_filter && content_filter_failover)
}

/// Whether `event` reports a tool the provider itself executes, whose
/// effects a restart could repeat.
pub(super) fn provider_executed(event: &Event) -> bool {
    matches!(
        event,
        Event::ServerToolUseStarted { .. }
            | Event::ServerToolArgumentsDelta { .. }
            | Event::ServerToolUseCompleted { .. }
            | Event::ServerToolResult { .. }
            | Event::HostedToolItemStarted { .. }
            | Event::HostedToolItemProgress { .. }
            | Event::HostedToolItemCompleted { .. }
    )
}

/// Drain `committed` for a buffered surface (`content_filter_failover` as
/// for [`restartable`]). `None` keeps the attempt as the
/// winner: either its whole answer now sits in `prefix` (terminal last), or
/// a failure the ladder must not restart is parked in `collection_failure`
/// for the surface to answer exactly as before. `Some` is the ladder end the
/// waterfall continues with.
pub(super) async fn drain(
    ctx: &WaterfallContext<'_>,
    guard: &mut AttemptGuard,
    committed: &mut CommittedAttempt,
    wire: &DeploymentWire,
    content_filter_failover: bool,
) -> Option<AttemptEnd> {
    let phase_timeout = Duration::from_secs_f64(wire.timeout_seconds.max(0.001));
    let mut provider_ran_tool = false;
    let collected = collect_committed_observed(
        committed,
        ctx.deadline,
        phase_timeout,
        guard.started,
        |event| provider_ran_tool |= provider_executed(event),
    )
    .await;
    guard.record_first_token(committed.relay.first_token_at());
    let failure = match collected {
        Ok(mut events) => match events.last() {
            // A provider-declared mid-body failure the caller has not seen.
            Some(Event::Failed(failure))
                if restartable(failure, content_filter_failover) && !provider_ran_tool =>
            {
                let failure = failure.clone();
                events.clear();
                failure
            }
            _ => {
                committed.prefix = events;
                return None;
            }
        },
        Err(failure) if restartable(&failure, content_filter_failover) && !provider_ran_tool => {
            failure
        }
        Err(failure) => {
            committed.collection_failure = Some(failure);
            return None;
        }
    };
    // The failed attempt keeps the upstream label the relay already parsed.
    guard.record_upstream_provider(committed.relay.upstream_provider());
    committed.relay.close_transport();
    // A filter cut restarts under the alias's own refusal failover too.
    let refusal_eligible =
        failure.failure_class == FailureClass::Refusal && ctx.policy.refusal_failover;
    Some(AttemptEnd::Ladder {
        failure,
        refusal_eligible,
        exhaustion_flush: Vec::new(),
        usage: committed.usage.take(),
        tool_names: std::mem::take(&mut committed.tool_names),
        opened: true,
        encrypted_reasoning_stripped: committed.encrypted_reasoning_stripped,
    })
}
