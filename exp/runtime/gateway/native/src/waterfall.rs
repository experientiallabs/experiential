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

use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::dialects::Dialect;
use crate::encode::compact_json;
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{Event, Usage};
use crate::metrics::METRICS;
use crate::rate_limit_headers::harvest_rate_limit_headers;
use crate::relay::{
    collection_public_error, ended_without_terminal, remaining, track_event, UpstreamRelay,
};
use crate::replay_repair::AttemptRepair;
use crate::settlement::AttemptGuard;
use crate::throttle_backoff::{track_retry_after, with_largest_retry_after};
use crate::tool_search::{withheld_overflow_failure, ToolSearchRound, WithheldSearchCall};
use crate::upstream::open_stream;

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
        /// failure (the python executor's `withheld_non_refusal_failure`).
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
    let mut total_attempts: u32 = 0;
    let mut counts: Vec<u32> = vec![0; ctx.route.len()];
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
            return Won::Failed(collection_public_error(&failure.boundary()));
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
        total_attempts += 1;
        counts[depth] += 1;
        // A rung a tool-search round rebuilt dials its rebuilt wire; the
        // route's own entry still answers every policy question above.
        let dial_wire = search_wires[depth].as_ref().unwrap_or(wire);
        let end = run_attempt(ctx, guard, dial_wire, depth, &mut repaired[depth]).await;
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
                counts[depth] = counts[depth].saturating_sub(1);
                let max_rounds = ctx.tool_search.map_or(0, |search| search.max_rounds);
                if rounds_done >= max_rounds
                    || total_attempts >= ctx.policy.maximum_total_attempts
                    || remaining(ctx.deadline).is_zero()
                {
                    // The model called the search tool past the budget (or
                    // after the control plane withdrew the tool): the turn
                    // has no answer in it, and the request fails closed with
                    // the gateway's own error rather than a half answer.
                    let failure = search_round::budget_exhausted();
                    guard
                        .settle("failed", usage.as_ref(), &tool_names, Some(&failure), true)
                        .await;
                    return Won::Failed(PublicError::internal());
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
                let backoff = throttle_backoff_delay(
                    ctx,
                    wire,
                    &failure,
                    throttle_redials[depth],
                    total_attempts,
                );
                let possible = backoff.is_some()
                    || successor_possible(
                        ctx.policy,
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

async fn run_attempt(
    ctx: &WaterfallContext<'_>,
    guard: &mut AttemptGuard,
    wire: &DeploymentWire,
    depth: usize,
    repaired: &mut Option<Value>,
) -> AttemptEnd {
    let Some(dialect) = Dialect::from_str(&wire.dialect) else {
        // Admission validated every dialect; reaching here is wire drift.
        return AttemptEnd::Ladder {
            failure: Failure::new(
                FailureClass::Internal,
                "gateway engine does not support the resolved provider dialect",
            ),
            refusal_eligible: false,
            exhaustion_flush: Vec::new(),
            usage: None,
            tool_names: Vec::new(),
            opened: false,
            encrypted_reasoning_stripped: false,
        };
    };
    // Body-signing dialects sign immediately before every physical attempt
    // so neither queue time nor a spent prior attempt can age the signature
    // (a same-deployment redial or a failover advance both call `run_attempt`
    // again, so each always gets a fresh signature); signing failures are
    // neither same-deployment-retryable nor failover-eligible, matching the
    // python executor's hard stop on an authentication failure.
    let headers = match dispatch_headers(ctx.bridge, ctx.request_id, wire).await {
        Ok(headers) => headers,
        Err(_) => {
            return AttemptEnd::Ladder {
                failure: Failure::new(
                    FailureClass::ProviderAuthentication,
                    "provider dispatch signing failed",
                ),
                refusal_eligible: false,
                exhaustion_flush: Vec::new(),
                usage: None,
                tool_names: Vec::new(),
                opened: false,
                encrypted_reasoning_stripped: false,
            };
        }
    };
    // The connection's raw timeout paces each BODY chunk read, exactly like
    // the python streaming path. The open (request/response-header) phase is
    // bounded by the fail-fast time-to-first-byte window alone (fresh per
    // attempt) and the request deadline: a dead lane that never answers is
    // abandoned in seconds, and a deployment whose authored first-byte
    // allowance exceeds the per-chunk timeout (a 120 s header hold on a
    // reasoning lane that thinks before its first header) is honored rather
    // than silently cut at the per-chunk 60 s. Before 0.3.74 the open bound
    // also took the per-chunk timeout, so every authored allowance above it
    // was a no-op: 251 of 251 header timeouts on lanes carrying 90 s and
    // 120 s allowances cut at exactly 60 s (production, 2026-09-16).
    let phase_timeout = Duration::from_secs_f64(wire.timeout_seconds.max(0.001));
    let first_byte_allowance_for = || {
        first_byte_allowance(
            wire,
            ctx.time_to_first_byte,
            ctx.time_to_first_byte_slope_seconds_per_million_input_tokens,
            ctx.approximate_input_tokens,
        )
    };
    let mut first_byte_deadline = Instant::now() + first_byte_allowance_for();
    // The first-token bound the relay enforces once the headers are in: its
    // own base (thinking models on a chat wire stream nothing for a minute
    // and more), the same input slope, absolute from the same dial.
    let first_token_allowance_for = || {
        first_token_allowance(
            wire,
            ctx.time_to_first_token,
            ctx.time_to_first_byte_slope_seconds_per_million_input_tokens,
            ctx.approximate_input_tokens,
        )
    };
    let mut first_token_deadline = Instant::now() + first_token_allowance_for();
    // What is already known repairs the first dial: the payload this request
    // stripped on an earlier dial of the rung, else the payloads this worker
    // remembers the caller's provider refusing.
    let mut repair = AttemptRepair::begin(wire, ctx.caller_scope, repaired, ctx.request_id);
    // At most one re-dial per attempt: when the rung refuses a replayed
    // reasoning item's encrypted payload (sealed by another organization or
    // tenant, or never issued by this provider), whether before the stream (a
    // 4xx) or inside it (OpenRouter's Responses relay answers 200 and then
    // fails the stream on its first frame), the same rung is dialed again,
    // same reservation, with those items stripped, before the caller's 400 may
    // surface. A payload with nothing to strip, or a refusal of the stripped
    // payload itself, surfaces.
    let mut redialed = false;
    // Usage a refused in-stream dial reported (a `response.failed` frame can
    // carry the tokens the provider billed for the legs it processed) rides
    // into the re-dial's relay and joins its first usage report, so the
    // reservation settles every token this attempt was charged for.
    let mut carried_usage: Option<Usage> = None;
    'dial: loop {
        let observation = guard.begin_dial_observation();
        if carried_usage.is_some() {
            // A new dispatch makes the earlier dial only a subtotal until
            // this dial supplies its own meter, including before headers.
            observation.record_dial_total(Usage::default());
        }
        let open_bound = open_phase_bound(remaining(ctx.deadline), remaining(first_byte_deadline));
        guard.mark_dispatched();
        let response = match open_stream(
            ctx.http,
            &wire.url,
            &headers,
            &wire.idempotency_key,
            repair.payload(),
            repair.raw_body(),
            open_bound,
            dialect,
        )
        .await
        {
            Ok(response) => response,
            Err(failure) => {
                if !redialed && repair.repair_after(&failure) {
                    // The stripped dial is a physical attempt of its own and
                    // gets a fresh first-byte window, exactly like a same-rung
                    // redial; the refused open must not eat into it.
                    redialed = true;
                    first_byte_deadline = Instant::now() + first_byte_allowance_for();
                    first_token_deadline = Instant::now() + first_token_allowance_for();
                    continue 'dial;
                }
                return AttemptEnd::Ladder {
                    failure: customer_owned(failure, wire),
                    refusal_eligible: false,
                    exhaustion_flush: Vec::new(),
                    usage: None,
                    tool_names: Vec::new(),
                    opened: false,
                    encrypted_reasoning_stripped: false,
                };
            }
        };
        repair.dial_opened();
        let encrypted_reasoning_stripped = repair.stripped();
        guard.mark_opened();
        // The opened response's allowlisted rate-limit headers settle with this
        // attempt whatever its terminal outcome; a failed OPEN instead carries
        // them on its failure (attached in `open_stream`).
        guard.record_rate_limit_headers(harvest_rate_limit_headers(response.headers()));
        let mut relay = match wire
            .fireworks_reasoning_route_sha256
            .clone()
            .or_else(|| wire.hunyuan_reasoning_route_sha256.clone())
        {
            Some(route_sha256) => UpstreamRelay::new_with_reasoning_content_route(
                response,
                dialect,
                first_token_deadline,
                Some(route_sha256),
            ),
            None => UpstreamRelay::new(response, dialect, first_token_deadline),
        };
        relay.set_observation(observation);
        relay.set_carried_usage(carried_usage.take());
        relay.set_stop_sequences(wire.stop_sequences.iter().cloned());
        relay.set_serialize_tool_calls(wire.serialize_tool_calls);
        relay.set_native_tool_translation(wire.native_tool_translation.clone());
        relay.set_tool_search_tool_name(ctx.tool_search.map(|search| search.tool_name.clone()));
        if !wire.model_id.is_empty() {
            relay.set_request_words([wire.model_id.clone()]);
        }
        if wire.billing_customer_managed {
            // Applied to every failure the relay yields, before or after commit,
            // so a committed stream's late credential error is the customer's too.
            relay.set_customer_managed_provider(Some(wire.provider.clone()));
        }
        // Refusal deltas are withheld when the alias revision opted into
        // refusal failover, or when a failover-only rung downstream accepts
        // an unnamed refusal (`fallback_rules`); the ladder decision below
        // still distinguishes the two, so the policy alone never advances a
        // refusal onto an unrestricted rung it did not opt into.
        let refusal_failover = ctx.policy.refusal_failover
            || fallback_rules::refusal_deltas_withheld_for(ctx.route, depth);
        // Per dial: tracked facts belong to the dial that produced them; a
        // refused dial's billed usage travels through the relay above.
        let mut usage: Option<Usage> = None;
        let mut tool_names: Vec<String> = Vec::new();
        let mut withheld: Vec<Event> = Vec::new();
        let mut withheld_bytes = 0usize;
        let mut private_reasoning = commit::PrivateReasoning::default();
        loop {
            let event = match relay
                .next_event(ctx.deadline, phase_timeout, guard.started)
                .await
            {
                Ok(Some(event)) => event,
                Ok(None) => {
                    return AttemptEnd::Ladder {
                        failure: ended_without_terminal(),
                        refusal_eligible: false,
                        exhaustion_flush: Vec::new(),
                        usage: relay.usage_before_failure(usage),
                        tool_names,
                        opened: true,
                        encrypted_reasoning_stripped,
                    }
                }
                Err(failure) => {
                    return AttemptEnd::Ladder {
                        failure,
                        refusal_eligible: false,
                        exhaustion_flush: Vec::new(),
                        usage: relay.usage_before_failure(usage),
                        tool_names,
                        opened: true,
                        encrypted_reasoning_stripped,
                    }
                }
            };
            track_event(&event, &mut usage, &mut tool_names);
            guard.record_first_token(relay.first_token_at());
            if private_reasoning.withhold(&event, wire.reasoning_output_exposed) {
                relay.private_progress();
                continue;
            }
            let refusal_text = match &event {
                Event::RefusalDelta(text) | Event::ProviderRefusalDelta { delta: text, .. } => {
                    Some(text)
                }
                _ => None,
            };
            if let Some(text) = refusal_text {
                if refusal_failover {
                    let event_bytes = text.len();
                    if withheld_bytes + event_bytes > MAXIMUM_WITHHELD_REFUSAL_BYTES
                        || withheld.len() + 1 > MAXIMUM_WITHHELD_REFUSAL_EVENTS
                    {
                        // Buffer overflow commits and flushes.
                        let prefix = private_reasoning.prefix(&mut withheld, event, usage.as_ref());
                        let tool_search_dropped_after_output = relay.withheld_search_call_seen();
                        relay.commit();
                        return AttemptEnd::Committed(Box::new(CommittedAttempt {
                            depth,
                            prefix,
                            relay,
                            usage,
                            tool_names,
                            visible_refusal: true,
                            encrypted_reasoning_stripped,
                            tool_search_rounds: Vec::new(),
                            tool_search_dropped_after_output,
                        }));
                    }
                    withheld_bytes += event_bytes;
                    withheld.push(event);
                    continue;
                }
            }
            if is_semantic(&event)
                || (private_reasoning.completes(&event) && !relay.withheld_search_call_seen())
            {
                // Outward output freezes this deployment. A private-only
                // successful terminal retains its existing encoding and seal
                // contract; a private-only failure remains failover-safe.
                let visible_refusal = !withheld.is_empty()
                    || matches!(
                        event,
                        Event::RefusalDelta(_) | Event::ProviderRefusalDelta { .. }
                    );
                let prefix = private_reasoning.prefix(&mut withheld, event, usage.as_ref());
                // A search call withheld in the same turn is dropped: the
                // rung is frozen on this output, and the caller is told.
                let tool_search_dropped_after_output = relay.withheld_search_call_seen();
                relay.commit();
                return AttemptEnd::Committed(Box::new(CommittedAttempt {
                    depth,
                    prefix,
                    relay,
                    usage,
                    tool_names,
                    visible_refusal,
                    encrypted_reasoning_stripped,
                    tool_search_rounds: Vec::new(),
                    tool_search_dropped_after_output,
                }));
            }
            if !event.is_terminal() {
                // Pre-commit non-semantic events are dropped from the outward
                // stream (usage stays tracked), matching the python executor.
                continue;
            }
            match &event {
                Event::Failed(failure) => {
                    usage = relay.usage_before_failure(usage);
                    if !redialed && withheld.is_empty() && repair.repair_after(failure) {
                        // The rung opened the stream and refused the replayed
                        // encrypted reasoning on its first frame: the same repair
                        // as a pre-stream 4xx, nothing outward was committed.
                        redialed = true;
                        carried_usage = Some(usage.take().unwrap_or_default());
                        first_byte_deadline = Instant::now() + first_byte_allowance_for();
                        first_token_deadline = Instant::now() + first_token_allowance_for();
                        continue 'dial;
                    }
                    let typed_refusal = failure.failure_class == FailureClass::Refusal;
                    let exhaustion_flush = if !withheld.is_empty() && !typed_refusal {
                        let mut flush = std::mem::take(&mut withheld);
                        flush.push(event.clone());
                        flush
                    } else {
                        withheld.clear();
                        Vec::new()
                    };
                    return AttemptEnd::Ladder {
                        failure: failure.clone(),
                        refusal_eligible: typed_refusal && ctx.policy.refusal_failover,
                        exhaustion_flush,
                        usage,
                        tool_names,
                        opened: true,
                        encrypted_reasoning_stripped,
                    };
                }
                _ => {
                    if !withheld.is_empty() {
                        // A refusal-only stream that terminated successfully is
                        // a provider refusal: withhold the output and advance,
                        // matching the python executor's converted terminal.
                        withheld.clear();
                        return AttemptEnd::Ladder {
                            failure: Failure::new(
                                FailureClass::Refusal,
                                "provider refused the request",
                            ),
                            refusal_eligible: ctx.policy.refusal_failover,
                            exhaustion_flush: Vec::new(),
                            usage,
                            tool_names,
                            opened: true,
                            encrypted_reasoning_stripped,
                        };
                    }
                    if matches!(event, Event::Completed | Event::StoppedAtSequence(_))
                        && relay.withheld_search_overflowed()
                    {
                        // The model flooded the gateway's search tool past the
                        // per-dial bound: the gateway's own limit, so the dial
                        // fails closed instead of running an oversized round.
                        return AttemptEnd::Ladder {
                            failure: withheld_overflow_failure(),
                            refusal_eligible: false,
                            exhaustion_flush: Vec::new(),
                            usage,
                            tool_names,
                            opened: true,
                            encrypted_reasoning_stripped,
                        };
                    }
                    if matches!(event, Event::Completed | Event::StoppedAtSequence(_))
                        && relay.withheld_search_call_count() > 0
                    {
                        // The turn's only output was the gateway's search
                        // tool: nothing reached the caller and nothing
                        // committed, so the waterfall runs the search and
                        // dials this depth again with the extended context.
                        return AttemptEnd::ToolSearchRound {
                            calls: relay.take_withheld_search_calls(),
                            usage,
                            tool_names,
                        };
                    }
                    if billed_empty_completion(&event, usage.as_ref()) {
                        // A `stop` that billed output tokens yet carried no
                        // semantic event is the provider's fault, not an answer
                        // (a reasoning-only turn on a rung whose reasoning the
                        // gateway strips): it takes the ladder like any other
                        // pre-commit failure instead of settling an empty success.
                        return AttemptEnd::Ladder {
                            failure: empty_completion_failure(wire),
                            refusal_eligible: false,
                            exhaustion_flush: Vec::new(),
                            usage,
                            tool_names,
                            opened: true,
                            encrypted_reasoning_stripped,
                        };
                    }
                    if unreported_empty_completion(&event, usage.as_ref()) {
                        if ctx.output_token_cap.is_some() {
                            // A `stop` with no output and no usage report on a
                            // capped request: the only benign reading is a
                            // budget the provider's hidden reasoning exhausted
                            // before the first visible token, mislabelled as a
                            // plain stop (Meta muse-spark under a small
                            // `max_tokens`, 2026-09-15). Answer `Incomplete` so
                            // the caller sees `length` and raises the cap,
                            // instead of an empty completed answer.
                            return settle_output_less(
                                ctx,
                                guard,
                                Event::Incomplete,
                                usage,
                                tool_names,
                                depth,
                                encrypted_reasoning_stripped,
                            )
                            .await;
                        }
                        // Uncapped, nothing sent, nothing accounted: the provider
                        // delivered nothing at all. Nothing was committed outward,
                        // so the ladder is safe, exactly like the billed twin.
                        return AttemptEnd::Ladder {
                            failure: empty_completion_failure(wire),
                            refusal_eligible: false,
                            exhaustion_flush: Vec::new(),
                            usage,
                            tool_names,
                            opened: true,
                            encrypted_reasoning_stripped,
                        };
                    }
                    // A successful terminal with no semantic output and nothing
                    // billed for it (a budget exhausted before the first delta,
                    // a zero-token stop): retain the output-less continuation
                    // while the attempt is still in flight, settle, then answer
                    // with the tracked usage ahead of the terminal so the
                    // encoders keep the client-visible token accounting.
                    return settle_output_less(
                        ctx,
                        guard,
                        event,
                        usage,
                        tool_names,
                        depth,
                        encrypted_reasoning_stripped,
                    )
                    .await;
                }
            }
        }
    }
}

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
