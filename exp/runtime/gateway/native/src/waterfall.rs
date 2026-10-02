//! Native execution of the certified deployment waterfall.
//!
//! The control plane's `admit` returns the full ordered route (one wire
//! configuration per certified deployment) plus the frozen retry policy
//! facts; this module loops physical dispatches under the request deadline,
//! mirroring the python executor's semantics: each dispatch is durably
//! reserved through the `start_attempt` bridge callback immediately before
//! network work, same-deployment redials happen only for retryable failure
//! classes and only before commitment, a pre-commit throttle on a rung the
//! pool's `throttle_redial` schedule marks worth waiting for is re-dialed on
//! the same deployment after a bounded backoff (see `throttle_backoff`),
//! failover advances to the next certified deployment for failover-eligible
//! failures before commitment, and the first outward semantic event
//! permanently freezes the serving deployment. When the alias revision
//! enables refusal failover, refusal deltas are withheld in a bounded
//! in-memory buffer so a refusal-only terminal can advance to the next
//! deployment without exposing the refused route; mixed output or buffer
//! overflow commits and flushes. A turn whose only output was a call to the
//! gateway's tool-search tool (withheld inside the relay, see `tool_search`)
//! does not commit either: the control plane runs the search and rebuilds
//! the rung's dispatch, and the same depth is dialed again under a bounded
//! round budget (`search_round`). Candidate
//! selection policy (health circuits, budgets, attempt counting) stays in
//! python: the loop only states its position and the classified failure, and
//! the control plane answers with a reservation, a later depth, or
//! exhaustion.

#[cfg(test)]
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::encode::compact_json;
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{Event, Usage};
use crate::metrics::METRICS;
use crate::relay::{collection_public_error, remaining, UpstreamRelay};
use crate::settlement::AttemptGuard;
use crate::throttle_backoff::{track_retry_after, with_largest_retry_after};
use crate::tool_search::{ToolSearchRound, WithheldSearchCall};

/// Byte bound for withheld refusal deltas, matching the python executor's
/// `_MAX_WITHHELD_REFUSAL_BYTES`.
pub const MAXIMUM_WITHHELD_REFUSAL_BYTES: usize = 65_536;

/// Event-count bound for withheld refusal deltas, matching the python
/// executor's `_MAX_WITHHELD_REFUSAL_EVENTS`.
pub const MAXIMUM_WITHHELD_REFUSAL_EVENTS: usize = 256;

/// The winning outcome of one waterfall run.
pub enum Won {
    /// A deployment committed: its outward prefix is decided and the live
    /// relay continues the same physical attempt.
    Committed(Box<CommittedAttempt>),
    /// The attempt reached a terminal before commitment and is already
    /// durably settled; `events` are the decided outward events.
    Settled(SettledAttempt),
    /// The ladder is exhausted (or accounting failed); the request is
    /// finalized and this public error answers the caller.
    Failed(PublicError),
}

/// One committed physical attempt with its live upstream relay.
pub struct CommittedAttempt {
    pub depth: usize,
    pub prefix: Vec<Event>,
    pub relay: UpstreamRelay,
    pub usage: Option<Usage>,
    pub tool_names: Vec<String>,
    /// Whether refusal deltas already reached (or will reach) the caller;
    /// a later typed refusal terminal then completes instead of failing.
    pub visible_refusal: bool,
    /// Whether this attempt served the re-dial that stripped the replayed
    /// encrypted reasoning items the rung refused (disclosed to the caller
    /// through `crate::replay_repair::REPLAY_REPAIR_HEADER`).
    pub encrypted_reasoning_stripped: bool,
    /// The gateway-run tool-search rounds that led to this attempt, in
    /// order; empty on every request the gateway searched nothing for.
    pub tool_search_rounds: Vec<ToolSearchRound>,
    /// The model called the gateway's search tool in the same turn as other
    /// semantic output: the rung committed on that output and the call was
    /// dropped (disclosed as `tool_search->dropped(after_output)`).
    pub tool_search_dropped_after_output: bool,
}

/// One attempt whose terminal was reached and settled before commitment.
pub struct SettledAttempt {
    pub depth: usize,
    pub events: Vec<Event>,
    /// See [`CommittedAttempt::encrypted_reasoning_stripped`].
    pub encrypted_reasoning_stripped: bool,
    /// See [`Served::empty_completion`]: the ladder exhausted on empty turns
    /// and these events are the typed empty answer.
    pub empty_completion: bool,
    /// See [`CommittedAttempt::tool_search_rounds`].
    pub tool_search_rounds: Vec<ToolSearchRound>,
}

/// The facts of the attempt that served, as the response surfaces name them.
#[derive(Debug, Clone, Copy)]
pub struct Served {
    pub depth: usize,
    pub encrypted_reasoning_stripped: bool,
    /// The answer is an empty turn the ladder could not improve on (every
    /// rung, or the committed rung, closed with nothing): the caller holds a
    /// typed 200 under `x-gateway-warning: empty_completion`, the ledger the
    /// typed `empty_completion` failure.
    pub empty_completion: bool,
}

impl CommittedAttempt {
    /// The serving facts of this committed attempt.
    pub fn served(&self) -> Served {
        Served {
            depth: self.depth,
            encrypted_reasoning_stripped: self.encrypted_reasoning_stripped,
            empty_completion: false,
        }
    }
}

impl SettledAttempt {
    /// The serving facts of this settled attempt.
    pub fn served(&self) -> Served {
        Served {
            depth: self.depth,
            encrypted_reasoning_stripped: self.encrypted_reasoning_stripped,
            empty_completion: self.empty_completion,
        }
    }
}

/// The control plane's answer to one `start_attempt` callback.
#[derive(Debug, Deserialize)]
pub(crate) struct StartResponse {
    #[serde(default)]
    pub(crate) attempt_id: Option<String>,
    #[serde(default)]
    pub(crate) route_depth: Option<usize>,
    #[serde(default)]
    pub(crate) exhausted: bool,
    #[serde(default)]
    pub(crate) failure: Option<Failure>,
    /// A durable zero-attempt certificate from the trusted accounting callback.
    #[serde(default)]
    pub(crate) known_unbilled: bool,
}

/// One pre-commit attempt outcome, private to the waterfall loop.
enum AttemptEnd {
    Committed(Box<CommittedAttempt>),
    Settled(SettledAttempt),
    /// The attempt failed before commitment; try the ladder.
    Ladder {
        failure: Failure,
        refusal_eligible: bool,
        /// Withheld refusal deltas plus the failing terminal, flushed
        /// outward only when the ladder is exhausted with a non-refusal
        /// failure.
        exhaustion_flush: Vec<Event>,
        usage: Option<Usage>,
        tool_names: Vec<String>,
        opened: bool,
        /// Whether the failing dial was the stripped re-dial, so an
        /// exhaustion flush of its withheld output still discloses it.
        encrypted_reasoning_stripped: bool,
    },
    /// The dial's only output was one or more calls to the gateway's
    /// tool-search tool, withheld inside the relay, and it ended
    /// successfully: nothing committed, and the waterfall runs the search
    /// and dials the same depth again.
    ToolSearchRound {
        calls: Vec<WithheldSearchCall>,
        usage: Option<Usage>,
        tool_names: Vec<String>,
    },
    /// Refused replay input was repaired; another physical reservation is required.
    Repair {
        failure: Failure,
        usage: Option<Usage>,
        tool_names: Vec<String>,
    },
    /// Accounting failed mid-attempt; the request is answered internal.
    Accounting,
    /// The attempt settled, but retaining its output-less continuation
    /// failed; the public retention error answers the caller.
    Retention(PublicError),
}

/// Run one certified waterfall to its committed or terminal attempt.
///
/// Every started attempt settles exactly once through `guard`; on return the
/// request is either finalized (`Settled`/`Failed`) or owned by the single
/// committed attempt the caller must settle.
pub async fn acquire_attempt(ctx: &WaterfallContext<'_>, guard: &mut AttemptGuard) -> Won {
    if !ctx.policy.valid() {
        guard
            .abandon(&Failure::new(
                FailureClass::Internal,
                "gateway retry policy contract failed",
            ))
            .await;
        return Won::Failed(PublicError::internal());
    }
    let mut total_attempts: u32 = 0;
    let mut counts: Vec<u32> = vec![0; ctx.route.len()];
    let mut physical_counts: Vec<u32> = vec![0; ctx.route.len()];
    let mut reasoning_repair = false;
    // Post-backoff redials made per depth: the schedule's per-rung cap
    // counts only these, never a retryable-class redial of the same rung.
    let mut throttle_redials: Vec<u32> = vec![0; ctx.route.len()];
    // Per depth, the replayed payload with the encrypted reasoning items the
    // rung refused stripped out: every later dial of that rung in this
    // request (a throttle redial, a same-rung retry) sends it directly
    // instead of earning the refusal again. Another rung may still decrypt
    // the original, so it starts from the payload as replayed.
    let mut repaired: Vec<Option<Value>> = vec![None; ctx.route.len()];
    let mut current_depth: Option<usize> = None;
    let mut last_failure: Option<Failure> = None;
    // The longest wait any throttled rung stated, so an exhausted ladder
    // tells the caller the whole story rather than the last rung's.
    let mut largest_retry_after: Option<u32> = None;
    // Whether this reservation re-dials the rung that just throttled after
    // the schedule's backoff was waited out; consumed by one `start_attempt`.
    let mut throttle_backoff = false;
    // Per depth, the wire a tool-search round rebuilt for the rung (the
    // conversation extended with the search call and its result, the
    // matched tools loaded); every later dial of that depth sends it.
    let mut search_wires: Vec<Option<DeploymentWire>> = vec![None; ctx.route.len()];
    // The rounds completed so far, handed to the winning attempt to render.
    let mut tool_search_rounds: Vec<ToolSearchRound> = Vec::new();
    let mut rounds_done: u32 = 0;
    // Whether this reservation re-dials the same depth after a tool-search
    // round; consumed by one `start_attempt`.
    let mut tool_search_round = false;
    loop {
        let search_redial = std::mem::take(&mut tool_search_round);
        let repair_redial = std::mem::take(&mut reasoning_repair);
        let argument = compact_json(&json!({
            "request_id": ctx.request_id,
            "raw_key": ctx.raw_key,
            "attempt_ordinal": total_attempts,
            "current_depth": current_depth,
            // A post-backoff redial of the throttled rung: the control plane
            // claims the same depth through its own throttle window and
            // discloses the attempt as `throttle_backoff`.
            "throttle_backoff": std::mem::take(&mut throttle_backoff),
            // A same-depth re-dial after a gateway tool-search round: the
            // control plane reserves the same rung and discloses the attempt
            // as `tool_search_round`.
            "tool_search_round": search_redial,
            "reasoning_repair": repair_redial,
            "failure": last_failure.as_ref().map(|failure| json!({
                "failure_class": failure.failure_class.as_str(),
                "safe_message": failure.safe_message,
                "retryable_same_deployment": failure.retryable_same_deployment,
                "failover_eligible": failure.failover_eligible,
                // The control plane echoes the exhausting failure back, and its
                // answer wins over this one, so client-error attribution has to
                // survive the round trip to reach the caller.
                "rejected_parameter": failure.rejected_parameter,
                "provider_detail": failure.provider_detail,
                // Ownership survives the round trip too: the echoed exhaustion
                // must still render as the customer's 400.
                "customer_owned": failure.customer_owned,
                // The refusal category survives the round trip so an exhausted
                // refusal ladder still names its reason to the caller.
                "refusal_reason": failure.refusal_reason.map(|reason| reason.as_str()),
                // A throttle's stated wait survives too, so an exhausted
                // throttle ladder advertises it as `Retry-After`.
                "retry_after_seconds": failure.retry_after_seconds,
            })),
        }));
        let started_text = match ctx.bridge.call("start_attempt", argument).await {
            Ok(text) => text,
            Err(error) => {
                // The control plane finalized the request (budget quota, a
                // pre-dispatch reservation failure, or an expired deadline)
                // before raising; the public error is authoritative.
                guard.disarm_finalized("failed");
                return Won::Failed(error);
            }
        };
        let started: StartResponse = match serde_json::from_str(&started_text) {
            Ok(started) => started,
            Err(_) => {
                guard
                    .abandon(&Failure::new(
                        FailureClass::Internal,
                        "gateway attempt wire contract failed",
                    ))
                    .await;
                return Won::Failed(PublicError::internal());
            }
        };
        if started.exhausted {
            // The control plane already finalized the request with this
            // failure; answer the caller with its public form.
            guard.disarm_finalized("failed");
            let failure = started.failure.or(last_failure).unwrap_or_else(|| {
                Failure::new(
                    FailureClass::ProviderInternal,
                    "all exact-model deployments are unavailable",
                )
            });
            let failure = with_largest_retry_after(failure, largest_retry_after);
            let mut error = collection_public_error(&failure.boundary());
            error.known_unbilled = started.known_unbilled;
            return Won::Failed(error);
        }
        let (Some(attempt_id), Some(depth)) = (started.attempt_id, started.route_depth) else {
            guard
                .abandon(&Failure::new(
                    FailureClass::Internal,
                    "gateway attempt wire contract failed",
                ))
                .await;
            return Won::Failed(PublicError::internal());
        };
        let Some(wire) = ctx.route.get(depth) else {
            guard.rebind(attempt_id);
            let failure = Failure::new(
                FailureClass::Internal,
                "gateway attempt wire contract failed",
            );
            guard
                .settle("failed", None, &[], Some(&failure), true)
                .await;
            return Won::Failed(PublicError::internal());
        };
        if !fallback_rules::dial_admitted(wire, current_depth, depth, last_failure.as_ref()) {
            // The control plane reserved a failover-only rung outside its
            // rules (a first dial, or a successor to a failure its set does
            // not name); the two halves of the contract disagree, so the
            // request fails closed rather than dialing the rung.
            guard.rebind(attempt_id);
            let failure = Failure::new(
                FailureClass::Internal,
                "gateway attempt wire contract failed: failover-only rung reserved outside its rules",
            );
            guard
                .settle("failed", None, &[], Some(&failure), true)
                .await;
            return Won::Failed(PublicError::internal());
        }
        if current_depth == Some(depth) && !search_redial {
            METRICS.record_open_retry();
        }
        guard.rebind(attempt_id);
        if !ctx.policy.permits(total_attempts, physical_counts[depth]) {
            let failure = Failure::new(
                FailureClass::Internal,
                "gateway reserved beyond the request attempt budget",
            );
            guard
                .settle("failed", None, &[], Some(&failure), true)
                .await;
            return Won::Failed(PublicError::internal());
        }
        total_attempts += 1;
        physical_counts[depth] += 1;
        if !search_redial && !repair_redial {
            counts[depth] += 1;
        }
        // A rung a tool-search round rebuilt dials its rebuilt wire; the
        // route's own entry still answers every policy question above.
        let dial_wire = search_wires[depth].as_ref().unwrap_or(wire);
        let end = run_attempt(
            ctx,
            guard,
            dial_wire,
            depth,
            &mut repaired[depth],
            repair_redial,
        )
        .await;
        match end {
            AttemptEnd::Committed(mut committed) => {
                // The committed stream has parsed at least its first
                // semantic chunk, so an aggregator's upstream label (if any)
                // is known here; settle it with whatever outcome follows.
                guard.record_upstream_provider(committed.relay.upstream_provider());
                committed.tool_search_rounds = std::mem::take(&mut tool_search_rounds);
                return Won::Committed(committed);
            }
            AttemptEnd::Settled(mut settled) => {
                settled.tool_search_rounds = std::mem::take(&mut tool_search_rounds);
                return Won::Settled(settled);
            }
            AttemptEnd::ToolSearchRound {
                calls,
                usage,
                tool_names,
            } => {
                // The search-call turn was the rung's own answer, not a
                // retry of it: the round budget bounds it, not the
                // same-deployment cap.
                let max_rounds = ctx.tool_search.map_or(0, |search| search.max_rounds);
                if rounds_done >= max_rounds
                    || !ctx.policy.permits(total_attempts, physical_counts[depth])
                    || remaining(ctx.deadline).is_zero()
                {
                    // The model called the search tool past the budget (or
                    // after the control plane withdrew the tool): the turn
                    // has no answer in it, and the request fails closed with
                    // the gateway's own error rather than a half answer.
                    let failure = if !ctx.policy.permits(total_attempts, physical_counts[depth]) {
                        Failure::new(FailureClass::InvalidRequest, "The request exhausted gateway.retry attempt limits before producing an answer. Increase the limits and resend.")
                    } else {
                        search_round::budget_exhausted()
                    };
                    guard
                        .settle("failed", usage.as_ref(), &tool_names, Some(&failure), true)
                        .await;
                    return Won::Failed(if failure.failure_class == FailureClass::Internal {
                        PublicError::internal()
                    } else {
                        collection_public_error(&failure.boundary())
                    });
                }
                rounds_done += 1;
                let reply = match search_round::negotiate(
                    ctx,
                    guard,
                    depth,
                    rounds_done,
                    &calls,
                    usage.as_ref(),
                    &tool_names,
                    &mut tool_search_rounds,
                )
                .await
                {
                    Ok(reply) => reply,
                    Err(won) => return won,
                };
                if reply.exhausted {
                    rounds_done = max_rounds;
                }
                // The rebuilt wire carries the extended conversation; a
                // payload stripped from the OLD conversation on an earlier
                // dial of this rung must not be dialed over it.
                repaired[depth] = None;
                search_wires[depth] = Some(reply.wire);
                current_depth = Some(depth);
                last_failure = None;
                tool_search_round = true;
                continue;
            }
            AttemptEnd::Repair {
                failure,
                usage,
                tool_names,
            } => {
                let possible = ctx.policy.permits(total_attempts, physical_counts[depth])
                    && !remaining(ctx.deadline).is_zero();
                if !guard
                    .settle(
                        "failed",
                        usage.as_ref(),
                        &tool_names,
                        Some(&failure),
                        !possible,
                    )
                    .await
                {
                    return Won::Failed(PublicError::internal());
                }
                if !possible {
                    return Won::Failed(collection_public_error(&failure.boundary()));
                }
                current_depth = Some(depth);
                last_failure = None;
                reasoning_repair = true;
                continue;
            }
            AttemptEnd::Accounting => return Won::Failed(PublicError::internal()),
            AttemptEnd::Retention(error) => return Won::Failed(error),
            AttemptEnd::Ladder {
                failure,
                refusal_eligible,
                exhaustion_flush,
                usage,
                tool_names,
                opened,
                encrypted_reasoning_stripped,
            } => {
                if opened {
                    guard.mark_opened();
                }
                track_retry_after(&mut largest_retry_after, &failure);
                let boundary = failure.clone().boundary();
                // A pre-commit throttle on a rung worth waiting for is
                // re-dialed after the schedule's backoff; otherwise the
                // existing redial and failover rules decide.
                let fits = ctx.policy.permits(total_attempts, physical_counts[depth]);
                let backoff = if fits {
                    throttle_backoff_delay(
                        ctx,
                        wire,
                        &failure,
                        throttle_redials[depth],
                        total_attempts,
                    )
                } else {
                    None
                };
                let mut next_policy = ctx.policy;
                let retry_wait = if failure.retryable_same_deployment
                    && fits
                    && counts[depth] < ctx.policy.maximum_same_deployment_attempts
                {
                    ctx.policy.backoff.map(|backoff| {
                        backoff.delay(
                            counts[depth].saturating_sub(1),
                            failure.retry_after_seconds,
                            remaining(ctx.deadline).saturating_sub(first_byte_allowance(
                                wire,
                                ctx.time_to_first_byte,
                                ctx.time_to_first_byte_slope_seconds_per_million_input_tokens,
                                ctx.approximate_input_tokens,
                            )),
                            ctx.request_id,
                        )
                    })
                } else {
                    None
                };
                let mut failure = failure;
                if retry_wait == Some(None) {
                    failure.retryable_same_deployment = false;
                }
                if !fits {
                    next_policy.maximum_same_deployment_attempts = counts[depth];
                }
                let possible = backoff.is_some()
                    || successor_possible(
                        next_policy,
                        ctx.route,
                        ctx.deadline,
                        total_attempts,
                        counts[depth],
                        depth,
                        &failure,
                        refusal_eligible,
                    );
                if !guard
                    .settle(
                        "failed",
                        usage.as_ref(),
                        &tool_names,
                        Some(&boundary),
                        !possible,
                    )
                    .await
                {
                    return Won::Failed(PublicError::internal());
                }
                if let Some(delay) = backoff {
                    // The failed attempt is settled; the wait is the only
                    // thing between it and the redial's own reservation.
                    tokio::time::sleep(delay).await;
                    throttle_redials[depth] += 1;
                    throttle_backoff = true;
                } else if possible {
                    if let Some(Some(delay)) = retry_wait {
                        tokio::time::sleep(delay).await;
                    }
                }
                if possible {
                    current_depth = Some(depth);
                    last_failure = Some(failure);
                    continue;
                }
                if !exhaustion_flush.is_empty() {
                    // Exhausted with withheld refusals and a non-refusal
                    // failure: flush the bounded refusal output and the
                    // failing terminal outward, exactly once.
                    return Won::Settled(SettledAttempt {
                        depth,
                        events: exhaustion_flush,
                        encrypted_reasoning_stripped,
                        empty_completion: false,
                        tool_search_rounds: Vec::new(),
                    });
                }
                if failure.failure_class == FailureClass::EmptyCompletion {
                    // Every rung the ladder could reach closed this
                    // conversation with nothing. That is the model's answer,
                    // not an outage: the last attempt is settled `failed` /
                    // `empty_completion` above ($0, the ledger's record), and
                    // the caller receives the empty turn as a typed 200
                    // (`x-gateway-warning: empty_completion`) that no SDK
                    // auto-retries -- a 502 here made one Claude Code session
                    // re-send a 44k-token prompt every minute (2026-09-15).
                    let mut events = Vec::with_capacity(2);
                    if let Some(tracked) = usage {
                        events.push(Event::Usage(tracked));
                    }
                    events.push(Event::Completed);
                    return Won::Settled(SettledAttempt {
                        depth,
                        events,
                        encrypted_reasoning_stripped,
                        empty_completion: true,
                        tool_search_rounds: std::mem::take(&mut tool_search_rounds),
                    });
                }
                let boundary = with_largest_retry_after(boundary, largest_retry_after);
                return Won::Failed(collection_public_error(&boundary));
            }
        }
    }
}

mod attempt;
use attempt::run_attempt;

mod commit;
pub(crate) use commit::is_semantic;
mod fallback_rules;
mod wire;
pub(crate) use wire::{first_byte_allowance, first_token_allowance, open_phase_bound};
pub use wire::{DeploymentWire, RoutePolicy, WaterfallContext};

mod empty;
use empty::settle_output_less;
pub(crate) use empty::{
    billed_empty_completion, empty_completion_failure, unreported_empty_completion,
};

mod dispatch;
use dispatch::{customer_owned, dispatch_headers};

mod successor;
pub(crate) use successor::successor_possible;
use successor::throttle_backoff_delay;

mod search_round;

#[cfg(test)]
mod ladder_tests;
#[cfg(test)]
mod progress_tests;
#[cfg(test)]
mod repair_ladder_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tool_search_ladder_tests;
