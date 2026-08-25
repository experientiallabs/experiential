//! Native execution of the certified deployment waterfall.
//!
//! The control plane's `admit` returns the full ordered route (one wire
//! configuration per certified deployment) plus the frozen retry policy
//! facts; this module loops physical dispatches under the request deadline,
//! mirroring the python executor's semantics: each dispatch is durably
//! reserved through the `start_attempt` bridge callback immediately before
//! network work, same-deployment redials happen only for retryable failure
//! classes and only before commitment, failover advances to the next
//! certified deployment for failover-eligible failures before commitment,
//! and the first outward semantic event permanently freezes the serving
//! deployment. When the alias revision enables refusal failover, refusal
//! deltas are withheld in a bounded in-memory buffer so a refusal-only
//! terminal can advance to the next deployment without exposing the refused
//! route; mixed output or buffer overflow commits and flushes. Candidate
//! selection policy (health circuits, budgets, attempt counting) stays in
//! python: the loop only states its position and the classified failure, and
//! the control plane answers with a reservation, a later depth, or
//! exhaustion.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures_util::stream::BoxStream;
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::bridge::Bridge;
use crate::dialects::{
    Dialect, Normalizer, MAXIMUM_RETAINED_OUTPUT_BYTES, OUTPUT_OVERFLOW_MESSAGE,
};
use crate::encode::compact_json;
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{Event, Usage};
use crate::metrics::METRICS;
use crate::sse::SseDecoder;
use crate::upstream::open_stream;

/// Byte bound for withheld refusal deltas, matching the python executor's
/// `_MAX_WITHHELD_REFUSAL_BYTES`.
pub const MAXIMUM_WITHHELD_REFUSAL_BYTES: usize = 65_536;

/// Event-count bound for withheld refusal deltas, matching the python
/// executor's `_MAX_WITHHELD_REFUSAL_EVENTS`.
pub const MAXIMUM_WITHHELD_REFUSAL_EVENTS: usize = 256;

/// One deployment's wire configuration inside the admitted ordered route.
/// Payloads are built python-side per deployment, since model identities and
/// dialects may differ across one certified pool.
#[derive(Debug, Clone, Deserialize)]
pub struct DeploymentWire {
    pub provider: String,
    pub deployment_id: String,
    pub dialect: String,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub timeout_seconds: f64,
    pub upstream_payload: Value,
    pub idempotency_key: String,
}

/// The frozen retry-policy facts returned by admission.
#[derive(Debug, Clone, Copy)]
pub struct RoutePolicy {
    pub maximum_total_attempts: u32,
    pub maximum_same_deployment_attempts: u32,
    pub refusal_failover: bool,
}

/// Everything one waterfall run needs besides its request guard.
pub struct WaterfallContext<'a> {
    pub bridge: &'a Arc<Bridge>,
    pub http: &'a reqwest::Client,
    pub request_id: &'a str,
    /// The presented virtual key, forwarded so hosted budget-error policy
    /// can shape a rejected reservation for the caller.
    pub raw_key: &'a str,
    pub route: &'a [DeploymentWire],
    pub policy: RoutePolicy,
    pub deadline: Instant,
}

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
}

/// One attempt whose terminal was reached and settled before commitment.
pub struct SettledAttempt {
    pub depth: usize,
    pub events: Vec<Event>,
}

/// Map one collection failure to its public error, honoring the shared
/// aggregate-output overflow contract.
pub fn collection_public_error(failure: &Failure) -> PublicError {
    if failure.safe_message == OUTPUT_OVERFLOW_MESSAGE {
        return PublicError::provider_output_too_large();
    }
    failure.public_error()
}

/// Approximate retained size of one aggregated event, in bytes. Completed
/// tool calls charge their full argument text, matching the python engine's
/// bounded aggregation, which also charges the completed call after its
/// streamed deltas.
pub fn event_retained_bytes(event: &Event) -> usize {
    match event {
        Event::TextDelta(text) | Event::RefusalDelta(text) => text.len(),
        Event::ToolArgumentsDelta { delta, .. } => delta.len(),
        Event::ToolCallCompleted { call, .. } => call.raw_arguments.len().max(64),
        _ => 64,
    }
}

/// Classify one mid-stream chunk timeout the way the python transport does:
/// a stalled provider read is a retryable transport failure unless the
/// request's own deadline is exhausted.
pub fn stream_timeout_failure(deadline: Instant) -> Failure {
    if remaining(deadline).is_zero() {
        Failure::new(FailureClass::Timeout, "gateway execution deadline exceeded")
    } else {
        Failure::new(
            FailureClass::Transport,
            "provider transport failed; retry the request",
        )
        .with_retry(true, true)
    }
}

/// The synthesized failure for a provider stream that closed without a
/// terminal event, matching the python executor's classification.
fn ended_without_terminal() -> Failure {
    Failure::new(
        FailureClass::MalformedResponse,
        "provider stream ended without a terminal event",
    )
    .with_retry(true, true)
}

pub fn remaining(deadline: Instant) -> Duration {
    deadline.saturating_duration_since(Instant::now())
}

fn is_semantic(event: &Event) -> bool {
    matches!(
        event,
        Event::TextDelta(_)
            | Event::RefusalDelta(_)
            | Event::ToolCallStarted { .. }
            | Event::ToolArgumentsDelta { .. }
            | Event::ToolCallCompleted { .. }
    )
}

/// Record the latest complete usage observation and invoked tool names.
pub fn track_event(event: &Event, usage: &mut Option<Usage>, tool_names: &mut Vec<String>) {
    match event {
        Event::Usage(candidate) if candidate.has_token_counts() => {
            *usage = Some(candidate.clone());
        }
        Event::ToolCallCompleted { call, .. } if !tool_names.contains(&call.name) => {
            tool_names.push(call.name.clone());
        }
        _ => {}
    }
}

/// One upstream SSE response being decoded and normalized incrementally.
pub struct UpstreamRelay {
    stream: BoxStream<'static, reqwest::Result<Bytes>>,
    decoder: SseDecoder,
    normalizer: Normalizer,
    pending: VecDeque<Event>,
    eof: bool,
    first_byte_recorded: bool,
}

impl UpstreamRelay {
    pub fn new(response: reqwest::Response, dialect: Dialect) -> Self {
        Self {
            stream: response.bytes_stream().boxed(),
            decoder: SseDecoder::new(),
            normalizer: Normalizer::new(dialect),
            pending: VecDeque::new(),
            eof: false,
            first_byte_recorded: false,
        }
    }

    /// Yield the next normalized event. `Ok(None)` means the upstream closed
    /// without a terminal event (the caller synthesizes that failure); a
    /// stream whose terminal was already yielded returns `Ok(None)` too, but
    /// callers stop at the terminal before observing it.
    pub async fn next_event(
        &mut self,
        deadline: Instant,
        phase_timeout: Duration,
        request_started: Instant,
    ) -> Result<Option<Event>, Failure> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Ok(Some(event));
            }
            if self.eof {
                return Ok(None);
            }
            let bound = remaining(deadline).min(phase_timeout);
            let chunk = match tokio::time::timeout(bound, self.stream.next()).await {
                Ok(Some(Ok(chunk))) => chunk,
                Ok(Some(Err(_))) => {
                    return Err(Failure::new(
                        FailureClass::Transport,
                        "provider transport failed; retry the request",
                    )
                    .with_retry(true, true))
                }
                Ok(None) => {
                    self.eof = true;
                    // Recover a final unterminated SSE frame at EOF, exactly
                    // like the python decoder, so a provider that omits the
                    // closing blank line still settles by its terminal event.
                    let tail = self.decoder.finish().map_err(|message| {
                        Failure::new(FailureClass::MalformedResponse, &message)
                            .with_retry(false, true)
                    })?;
                    if let Some(frame) = tail {
                        let events = self.normalizer.feed(&frame)?;
                        self.pending.extend(events);
                    }
                    continue;
                }
                Err(_) => return Err(stream_timeout_failure(deadline)),
            };
            if !self.first_byte_recorded {
                METRICS
                    .time_to_first_byte_ms
                    .record(request_started.elapsed());
                self.first_byte_recorded = true;
            }
            let frames = self.decoder.feed(&chunk).map_err(|message| {
                Failure::new(FailureClass::MalformedResponse, &message).with_retry(false, true)
            })?;
            for frame in frames {
                let events = self.normalizer.feed(&frame)?;
                self.pending.extend(events);
            }
        }
    }
}

/// Build the settle callback argument shared by explicit settlement and the
/// drop backstop.
#[allow(clippy::too_many_arguments)]
fn settle_argument(
    request_id: &str,
    attempt_id: &str,
    outcome: &str,
    usage: Option<&Usage>,
    tool_names: &[String],
    failure: Option<&Failure>,
    finalize: bool,
    opened: bool,
) -> String {
    compact_json(&json!({
        "request_id": request_id,
        "attempt_id": attempt_id,
        "outcome": outcome,
        "usage": usage.map(|usage| json!({
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
        })),
        "tool_names": tool_names,
        "failure": failure.map(|failure| json!({
            "failure_class": failure.failure_class.as_str(),
            "safe_message": failure.safe_message,
        })),
        "finalize": finalize,
        "opened": opened,
    }))
}

/// Deliver one control-plane write with bounded backoff; the control plane
/// keeps the in-flight entry on a failed terminal write, so retries can
/// still land. A persistent failure stays latched as accounting-unhealthy
/// control-plane side and is reconciled at the next startup.
async fn deliver(bridge: &Bridge, method: &'static str, argument: String) -> bool {
    for backoff_ms in [0u64, 100, 500, 2_000] {
        if backoff_ms > 0 {
            tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
            METRICS.record_settlement_retry();
        }
        if bridge.call(method, argument.clone()).await.is_ok() {
            return true;
        }
    }
    // The control plane keeps the in-flight entry; its sweep keeps retrying
    // and latches readiness if the loss is durable. Leave an operator signal
    // as one structured, content-free stderr line beside the counter.
    METRICS.record_settlement_give_up();
    let parsed: Value = serde_json::from_str(&argument).unwrap_or(Value::Null);
    let line = json!({
        "event": "settlement_give_up",
        "method": method,
        "request_id": parsed.get("request_id").cloned().unwrap_or(Value::Null),
        "attempt_id": parsed.get("attempt_id").cloned().unwrap_or(Value::Null),
        "outcome": parsed.get("outcome").cloned().unwrap_or(Value::Null),
    });
    eprintln!("exp-gateway-native: {line}");
    false
}

/// Exactly-once settlement owner for one admitted request and its physical
/// attempts.
///
/// Every admitted request settles through this guard. Each reserved attempt
/// is bound with `rebind`; a non-finalizing settlement closes that attempt
/// and leaves the request open for the next dispatch. If the owning future
/// is dropped before the terminal settlement lands (client disconnect
/// cancels the handler, a panic unwinds the stream task), `Drop` spawns the
/// closing write so the ledger rows and their budget reservations are always
/// closed: the decided settlement verbatim when delivery was cut short, a
/// cancellation of the active attempt otherwise, or an `abandon` of the
/// accepted request when no attempt is active.
pub struct AttemptGuard {
    pub bridge: Arc<Bridge>,
    request_id: String,
    attempt_id: Option<String>,
    pending: Arc<AtomicUsize>,
    armed: bool,
    outcome_recorded: bool,
    /// Whether the active attempt's provider dispatch opened successfully;
    /// carried into settlement for deployment-health recording.
    opened: bool,
    /// The exact settlement whose delivery is in flight. The drop backstop
    /// re-delivers this decided settlement instead of a cancellation, so a
    /// task cancelled mid-write can neither downgrade the ledger outcome nor
    /// diverge from the recorded metric.
    decided_settlement: Option<String>,
    pub started: Instant,
}

/// Holds one unit of the shutdown drain counter for a detached stream task,
/// so graceful shutdown waits (bounded by the graceful timeout) for the
/// task's terminal settlement instead of dropping the runtime under it.
pub struct SettlementTask(Arc<AtomicUsize>);

impl SettlementTask {
    fn hold(counter: Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::SeqCst);
        Self(counter)
    }
}

impl Drop for SettlementTask {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl AttemptGuard {
    /// Count the owning detached task against graceful-shutdown draining.
    pub fn hold_task(&self) -> SettlementTask {
        SettlementTask::hold(self.pending.clone())
    }

    pub fn new(
        bridge: Arc<Bridge>,
        pending: Arc<AtomicUsize>,
        request_id: String,
        started: Instant,
    ) -> Self {
        METRICS.record_served();
        METRICS.enter_request();
        Self {
            bridge,
            request_id,
            attempt_id: None,
            pending,
            armed: true,
            outcome_recorded: false,
            opened: false,
            decided_settlement: None,
            started,
        }
    }

    /// Bind one freshly reserved attempt as the active settlement target.
    pub fn rebind(&mut self, attempt_id: String) {
        self.attempt_id = Some(attempt_id);
        self.opened = false;
        self.decided_settlement = None;
    }

    /// Record that the active attempt's provider dispatch opened.
    pub fn mark_opened(&mut self) {
        self.opened = true;
    }

    /// Record this request's terminal outcome and duration exactly once, at
    /// the moment the outcome is decided. Recording happens before delivery
    /// is awaited, so a task cancelled mid-write cannot re-report a decided
    /// outcome as a cancellation.
    fn record_terminal(&mut self, outcome: &str, cancelled: bool) {
        if self.outcome_recorded {
            return;
        }
        self.outcome_recorded = true;
        METRICS.record_outcome(outcome, cancelled);
        METRICS.request_duration_ms.record(self.started.elapsed());
        METRICS.exit_request();
    }

    /// Durably settle the active attempt. A finalizing settlement also
    /// terminalizes the request and disarms the drop backstop; a
    /// non-finalizing one closes only the attempt so the waterfall can
    /// dispatch its successor. Returns whether the write reached the ledger.
    pub async fn settle(
        &mut self,
        outcome: &str,
        usage: Option<&Usage>,
        tool_names: &[String],
        failure: Option<&Failure>,
        finalize: bool,
    ) -> bool {
        let Some(attempt_id) = self.attempt_id.clone() else {
            // No active attempt: nothing durable to close here. The abandon
            // path owns request-only terminalization.
            return true;
        };
        let argument = settle_argument(
            &self.request_id,
            &attempt_id,
            outcome,
            usage,
            tool_names,
            failure,
            finalize,
            self.opened,
        );
        if finalize {
            let cancelled = failure.map(|failure| failure.failure_class == FailureClass::Cancelled)
                == Some(true);
            self.record_terminal(outcome, cancelled);
        }
        self.decided_settlement = Some(argument.clone());
        let delivered = deliver(&self.bridge, "settle", argument).await;
        if finalize {
            // The control plane retained a failed terminal write verbatim,
            // so its sweep (not the drop backstop) owns the retry.
            self.decided_settlement = None;
            self.armed = false;
        } else if delivered {
            self.decided_settlement = None;
            self.attempt_id = None;
            self.opened = false;
        }
        // A failed non-finalizing delivery keeps the decided settlement
        // armed: the drop backstop re-delivers the ORIGINAL outcome verbatim
        // instead of downgrading the pending provider failure to a
        // cancellation, and the caller treats the failure as fatal so no
        // successor is ever dispatched over an unsettled attempt.
        delivered
    }

    /// Settle the active attempt as cancelled and finalize the request.
    pub async fn settle_cancelled(&mut self, usage: Option<&Usage>, tool_names: &[String]) -> bool {
        if self.attempt_id.is_some() {
            self.settle(
                "failed",
                usage,
                tool_names,
                Some(&Failure::new(
                    FailureClass::Cancelled,
                    "gateway request was cancelled",
                )),
                true,
            )
            .await
        } else {
            self.abandon(&Failure::new(
                FailureClass::Cancelled,
                "gateway request was cancelled",
            ))
            .await
        }
    }

    /// Terminalize an accepted request that has no active attempt.
    pub async fn abandon(&mut self, failure: &Failure) -> bool {
        self.record_terminal("failed", failure.failure_class == FailureClass::Cancelled);
        self.armed = false;
        deliver(
            &self.bridge,
            "abandon",
            compact_json(&json!({
                "request_id": self.request_id,
                "failure": {
                    "failure_class": failure.failure_class.as_str(),
                    "safe_message": failure.safe_message,
                },
            })),
        )
        .await
    }

    /// Disarm the guard after the control plane itself finalized the request
    /// (an exhausted ladder or a terminal budget rejection).
    pub fn disarm_finalized(&mut self, outcome: &str) {
        self.record_terminal(outcome, false);
        self.armed = false;
        self.attempt_id = None;
        self.decided_settlement = None;
    }
}

impl Drop for AttemptGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // A settlement already decided (its delivery was cut short by the
        // cancellation) is re-delivered verbatim, matching the control
        // plane's own never-downgrade sweep semantics; an attempt with no
        // decided outcome settles as cancelled, and an accepted request with
        // no active attempt is abandoned.
        let (method, argument): (&'static str, String) = match self.decided_settlement.take() {
            Some(argument) => ("settle", argument),
            None => match self.attempt_id.take() {
                Some(attempt_id) => {
                    self.record_terminal("failed", true);
                    (
                        "settle",
                        settle_argument(
                            &self.request_id,
                            &attempt_id,
                            "failed",
                            None,
                            &[],
                            Some(&Failure::new(
                                FailureClass::Cancelled,
                                "gateway request was cancelled",
                            )),
                            true,
                            self.opened,
                        ),
                    )
                }
                None => {
                    self.record_terminal("failed", true);
                    (
                        "abandon",
                        compact_json(&json!({
                            "request_id": self.request_id,
                            "failure": {
                                "failure_class": "cancelled",
                                "safe_message": "gateway request was cancelled",
                            },
                        })),
                    )
                }
            },
        };
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            // Runtime teardown; startup reconciliation closes the row.
            return;
        };
        let bridge = self.bridge.clone();
        let pending = self.pending.clone();
        pending.fetch_add(1, Ordering::SeqCst);
        handle.spawn(async move {
            deliver(&bridge, method, argument).await;
            pending.fetch_sub(1, Ordering::SeqCst);
        });
    }
}

/// The control plane's answer to one `start_attempt` callback.
#[derive(Debug, Deserialize)]
struct StartResponse {
    #[serde(default)]
    attempt_id: Option<String>,
    #[serde(default)]
    route_depth: Option<usize>,
    #[serde(default)]
    exhausted: bool,
    #[serde(default)]
    failure: Option<Failure>,
}

/// Whether the classified failure leaves any successor dispatch possible
/// under the rust-side facts (caps, flags, remaining route, deadline). The
/// control plane re-checks with health and budget state and may still answer
/// with exhaustion.
#[allow(clippy::too_many_arguments)]
fn successor_possible(
    policy: RoutePolicy,
    route_length: usize,
    deadline: Instant,
    total_attempts: u32,
    same_deployment_attempts: u32,
    depth: usize,
    failure: &Failure,
    refusal_eligible: bool,
) -> bool {
    if total_attempts >= policy.maximum_total_attempts || remaining(deadline).is_zero() {
        return false;
    }
    let same = failure.retryable_same_deployment
        && same_deployment_attempts < policy.maximum_same_deployment_attempts;
    let failover = (failure.failover_eligible || refusal_eligible) && depth + 1 < route_length;
    same || failover
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
    },
    /// Accounting failed mid-attempt; the request is answered internal.
    Accounting,
}

/// Run one certified waterfall to its committed or terminal attempt.
///
/// Every started attempt settles exactly once through `guard`; on return the
/// request is either finalized (`Settled`/`Failed`) or owned by the single
/// committed attempt the caller must settle.
pub async fn acquire_attempt(ctx: &WaterfallContext<'_>, guard: &mut AttemptGuard) -> Won {
    let mut total_attempts: u32 = 0;
    let mut counts: Vec<u32> = vec![0; ctx.route.len()];
    let mut current_depth: Option<usize> = None;
    let mut last_failure: Option<Failure> = None;
    loop {
        let argument = compact_json(&json!({
            "request_id": ctx.request_id,
            "raw_key": ctx.raw_key,
            "attempt_ordinal": total_attempts,
            "current_depth": current_depth,
            "failure": last_failure.as_ref().map(|failure| json!({
                "failure_class": failure.failure_class.as_str(),
                "safe_message": failure.safe_message,
                "retryable_same_deployment": failure.retryable_same_deployment,
                "failover_eligible": failure.failover_eligible,
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
        if current_depth == Some(depth) {
            METRICS.record_open_retry();
        }
        guard.rebind(attempt_id);
        total_attempts += 1;
        counts[depth] += 1;
        let end = run_attempt(ctx, guard, wire, depth).await;
        match end {
            AttemptEnd::Committed(committed) => return Won::Committed(committed),
            AttemptEnd::Settled(settled) => return Won::Settled(settled),
            AttemptEnd::Accounting => return Won::Failed(PublicError::internal()),
            AttemptEnd::Ladder {
                failure,
                refusal_eligible,
                exhaustion_flush,
                usage,
                tool_names,
                opened,
            } => {
                if opened {
                    guard.mark_opened();
                }
                let boundary = failure.clone().boundary();
                let possible = successor_possible(
                    ctx.policy,
                    ctx.route.len(),
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
                    });
                }
                return Won::Failed(collection_public_error(&boundary));
            }
        }
    }
}

/// Open and read one physical attempt up to commitment or its terminal.
async fn run_attempt(
    ctx: &WaterfallContext<'_>,
    guard: &mut AttemptGuard,
    wire: &DeploymentWire,
    depth: usize,
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
        };
    };
    // The connection's raw timeout bounds each transport phase (open, then
    // every chunk read), exactly like the python streaming path.
    let phase_timeout = Duration::from_secs_f64(wire.timeout_seconds.max(0.001));
    let open_bound = remaining(ctx.deadline).min(phase_timeout);
    let response = match open_stream(
        ctx.http,
        &wire.url,
        &wire.headers,
        &wire.idempotency_key,
        &wire.upstream_payload,
        open_bound,
    )
    .await
    {
        Ok(response) => response,
        Err(failure) => {
            return AttemptEnd::Ladder {
                failure,
                refusal_eligible: false,
                exhaustion_flush: Vec::new(),
                usage: None,
                tool_names: Vec::new(),
                opened: false,
            }
        }
    };
    guard.mark_opened();
    let mut relay = UpstreamRelay::new(response, dialect);
    let mut usage: Option<Usage> = None;
    let mut tool_names: Vec<String> = Vec::new();
    let mut withheld: Vec<Event> = Vec::new();
    let mut withheld_bytes = 0usize;
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
                    usage,
                    tool_names,
                    opened: true,
                }
            }
            Err(failure) => {
                return AttemptEnd::Ladder {
                    failure,
                    refusal_eligible: false,
                    exhaustion_flush: Vec::new(),
                    usage,
                    tool_names,
                    opened: true,
                }
            }
        };
        track_event(&event, &mut usage, &mut tool_names);
        if let Event::RefusalDelta(text) = &event {
            if ctx.policy.refusal_failover {
                let event_bytes = text.len();
                if withheld_bytes + event_bytes > MAXIMUM_WITHHELD_REFUSAL_BYTES
                    || withheld.len() + 1 > MAXIMUM_WITHHELD_REFUSAL_EVENTS
                {
                    // Buffer overflow commits and flushes.
                    let mut prefix = std::mem::take(&mut withheld);
                    prefix.push(event);
                    return AttemptEnd::Committed(Box::new(CommittedAttempt {
                        depth,
                        prefix,
                        relay,
                        usage,
                        tool_names,
                        visible_refusal: true,
                    }));
                }
                withheld_bytes += event_bytes;
                withheld.push(event);
                continue;
            }
        }
        if is_semantic(&event) {
            // First outward semantic output freezes this deployment; any
            // withheld refusals flush ahead of it.
            let visible_refusal = !withheld.is_empty() || matches!(event, Event::RefusalDelta(_));
            let mut prefix = std::mem::take(&mut withheld);
            prefix.push(event);
            return AttemptEnd::Committed(Box::new(CommittedAttempt {
                depth,
                prefix,
                relay,
                usage,
                tool_names,
                visible_refusal,
            }));
        }
        if !event.is_terminal() {
            // Pre-commit non-semantic events are dropped from the outward
            // stream (usage stays tracked), matching the python executor.
            continue;
        }
        match &event {
            Event::Failed(failure) => {
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
                    };
                }
                // A successful terminal with no semantic output: settle and
                // answer with the terminal alone.
                let outcome = if matches!(event, Event::Incomplete) {
                    "incomplete"
                } else {
                    "completed"
                };
                if !guard
                    .settle(outcome, usage.as_ref(), &tool_names, None, true)
                    .await
                {
                    return AttemptEnd::Accounting;
                }
                return AttemptEnd::Settled(SettledAttempt {
                    depth,
                    events: vec![event],
                });
            }
        }
    }
}

/// Drain one committed attempt to completion for non-streaming responses,
/// bounding total retained output like the python service's aggregation.
pub async fn collect_committed(
    committed: &mut CommittedAttempt,
    deadline: Instant,
    phase_timeout: Duration,
    request_started: Instant,
) -> Result<Vec<Event>, Failure> {
    let mut events: Vec<Event> = Vec::new();
    let mut retained_bytes = 0usize;
    let mut retain = |events: &mut Vec<Event>, event: Event| -> Result<(), Failure> {
        retained_bytes = retained_bytes.saturating_add(event_retained_bytes(&event));
        if retained_bytes > MAXIMUM_RETAINED_OUTPUT_BYTES {
            return Err(Failure::new(
                FailureClass::ProviderInternal,
                OUTPUT_OVERFLOW_MESSAGE,
            ));
        }
        events.push(event);
        Ok(())
    };
    for event in committed.prefix.drain(..) {
        retain(&mut events, event)?;
    }
    if events.last().is_some_and(Event::is_terminal) {
        return Ok(events);
    }
    loop {
        match committed
            .relay
            .next_event(deadline, phase_timeout, request_started)
            .await?
        {
            Some(event) => {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(refusal_failover: bool) -> RoutePolicy {
        RoutePolicy {
            maximum_total_attempts: 8,
            maximum_same_deployment_attempts: 2,
            refusal_failover,
        }
    }

    fn far_deadline() -> Instant {
        Instant::now() + Duration::from_secs(60)
    }

    #[test]
    fn successor_requires_capacity_and_an_eligible_class() {
        let retryable = Failure::new(FailureClass::ProviderInternal, "boom").with_retry(true, true);
        // Same-deployment retry within the per-deployment cap.
        assert!(successor_possible(
            policy(false),
            1,
            far_deadline(),
            1,
            1,
            0,
            &retryable,
            false,
        ));
        // The per-deployment cap forbids a redial but failover still runs.
        assert!(successor_possible(
            policy(false),
            2,
            far_deadline(),
            2,
            2,
            0,
            &retryable,
            false,
        ));
        // A single-deployment route with the redial cap reached is exhausted.
        assert!(!successor_possible(
            policy(false),
            1,
            far_deadline(),
            2,
            2,
            0,
            &retryable,
            false,
        ));
        // The hard total cap ends the ladder regardless of class.
        assert!(!successor_possible(
            policy(false),
            4,
            far_deadline(),
            8,
            1,
            0,
            &retryable,
            false,
        ));
        // An expired deadline ends the ladder.
        assert!(!successor_possible(
            policy(false),
            4,
            Instant::now(),
            1,
            1,
            0,
            &retryable,
            false,
        ));
    }

    #[test]
    fn ineligible_classes_never_advance_without_refusal_opt_in() {
        let invalid = Failure::new(FailureClass::InvalidRequest, "bad request");
        assert!(!successor_possible(
            policy(false),
            4,
            far_deadline(),
            1,
            1,
            0,
            &invalid,
            false,
        ));
        let refusal = Failure::new(FailureClass::Refusal, "provider refused the request");
        assert!(!successor_possible(
            policy(false),
            4,
            far_deadline(),
            1,
            1,
            0,
            &refusal,
            false,
        ));
        // The refusal advances only when the alias revision opted in.
        assert!(successor_possible(
            policy(true),
            4,
            far_deadline(),
            1,
            1,
            0,
            &refusal,
            true,
        ));
        // Refusal failover cannot pass the last deployment.
        assert!(!successor_possible(
            policy(true),
            1,
            far_deadline(),
            1,
            1,
            0,
            &refusal,
            true,
        ));
    }

    #[test]
    fn failover_only_classes_skip_the_redial_and_advance() {
        let throttled = Failure::new(FailureClass::Throttled, "throttled").with_retry(false, true);
        assert!(successor_possible(
            policy(false),
            2,
            far_deadline(),
            1,
            1,
            0,
            &throttled,
            false,
        ));
        assert!(!successor_possible(
            policy(false),
            1,
            far_deadline(),
            1,
            1,
            0,
            &throttled,
            false,
        ));
    }
}
