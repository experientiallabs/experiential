//! Exactly-once settlement of admitted requests and their physical attempts.
//!
//! Every durable accounting write flows through this module: the bounded
//! retry delivery to the control plane, the `AttemptGuard` that owns one
//! request's terminal outcome (including the drop backstop for cancelled
//! handlers and unwound stream tasks), and the shutdown-drain hold that keeps
//! graceful stop waiting for detached stream settlements.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::bridge::Bridge;
use crate::encode::compact_json;
use crate::errors::{Failure, FailureClass};
use crate::events::{Event, Usage};
use crate::metrics::METRICS;
use crate::replay::OwnerLease;
use crate::waterfall::CommittedAttempt;

#[path = "settlement_observation.rs"]
mod observation;
pub(crate) use observation::Observation;

/// Settle one guarded attempt as failed and release its owner lease.
pub(crate) async fn settle_guarded_failure(
    guard: &mut AttemptGuard,
    committed: &mut CommittedAttempt,
    lease: &mut Option<OwnerLease>,
    failure: &Failure,
) {
    let usage = committed.usage.clone();
    guard
        .settle(
            "failed",
            usage.as_ref(),
            &committed.tool_names,
            Some(failure),
            true,
        )
        .await;
    if let Some(mut owner) = lease.take() {
        owner.abandon().await;
    }
}

/// Format one wall-clock instant as an RFC 3339 / ISO 8601 UTC string with
/// millisecond precision, e.g. `2026-08-30T12:34:56.789+00:00`.
///
/// The control plane parses this with `datetime.fromisoformat`, so the output
/// stays inside that grammar (explicit `+00:00` offset, millisecond fraction).
/// The crate carries no datetime dependency, so the civil date is derived from
/// the Unix epoch with Howard Hinnant's `civil_from_days` algorithm; a time
/// before the epoch (never expected for a served token) clamps to the epoch.
fn system_time_to_rfc3339(at: SystemTime) -> String {
    let since_epoch = at.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = since_epoch.as_secs() as i64;
    let millis = since_epoch.subsec_millis();
    let days = secs.div_euclid(86_400);
    let seconds_of_day = secs.rem_euclid(86_400);
    let (hour, minute, second) = (
        seconds_of_day / 3_600,
        (seconds_of_day % 3_600) / 60,
        seconds_of_day % 60,
    );
    // civil_from_days: shift the epoch to a 0000-03-01 era so leap handling is
    // branch-free, then recover the Gregorian year/month/day.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097; // [0, 146_096]
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let mp = (5 * day_of_year + 2) / 153; // [0, 11], months shifted so March = 0
    let day = day_of_year - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if month <= 2 { year + 1 } else { year };
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}+00:00")
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
    first_token_at: Option<SystemTime>,
    rate_limit_headers: Option<&serde_json::Map<String, Value>>,
    upstream_provider: Option<&str>,
    web_search_requests: u32,
    tool_search_requests: u32,
) -> String {
    let mut argument = json!({
        "request_id": request_id,
        "attempt_id": attempt_id,
        "outcome": outcome,
        "usage": usage.map(|usage| json!({
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens,
            "cache_creation_1h_input_tokens": usage.cache_creation_1h_input_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
        })),
        "tool_names": tool_names,
        "failure": failure.map(|failure| json!({
            "failure_class": failure.failure_class.as_str(),
            "safe_message": failure.safe_message,
            // The provider's own sanitized rejection sentence (client-error
            // class only); accounting persists it on the failed-attempt row so
            // an operator sees WHY the provider refused the call.
            "provider_detail": failure.provider_detail,
            // The bounded refusal category (refusal class only), so the
            // control plane can count refusals by reason without parsing the
            // free-form provider_detail.
            "refusal_reason": failure.refusal_reason.map(|reason| reason.as_str()),
            // The customer's own BYOK credential/account failure: the ledger
            // files it as the caller's invalid request (see
            // native_accounting.ledger_failure).
            "customer_owned": failure.customer_owned,
            // The provider's own stated wait (a throttled open's integer
            // Retry-After), so the control plane sizes the deployment's
            // throttle window from it instead of the fixed default.
            "retry_after_seconds": failure.retry_after_seconds,
        })),
        "finalize": finalize,
        "opened": opened,
        // Explicit HTTP-rejection provenance for decisions only. Unknown
        // outcomes and cancellation never authorize zero-cost accounting.
        "decision_provider_rejected": failure.is_some_and(|failure| failure.decision_provider_rejected),
        "first_token_at": first_token_at.map(system_time_to_rfc3339),
        // Allowlisted rate-limit headers of the attempt's provider response
        // (successes and failures alike, absent when none were present); the
        // control plane normalizes and persists them per attempt.
        "rate_limit_headers": rate_limit_headers,
        // The upstream an aggregator rung named as serving the attempt
        // (OpenRouter's per-chunk `provider` under its metadata opt-in),
        // absent when the stream named none; the control plane persists it
        // per attempt so a zero-data-retention dispatch shows WHICH
        // retention-free upstream answered.
        "upstream_provider": upstream_provider,
    });
    // The gateway's own pre-dispatch web searches, a request-level cost the
    // control plane bills once: only the settlement that closes the request
    // carries the count, so a failed rung's non-finalizing settlement can
    // never bill it a second time. Absent at zero, so unsearched requests
    // settle with exactly the bytes they always did.
    if finalize && web_search_requests > 0 {
        argument["web_search_requests"] = json!(web_search_requests);
    }
    // The gateway's own tool-search rounds bill the same way: once, on the
    // settlement that closes the request, absent at zero.
    if finalize && tool_search_requests > 0 {
        argument["tool_search_requests"] = json!(tool_search_requests);
    }
    compact_json(&argument)
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
    /// Dispatch began, even if response headers have not arrived yet.
    dispatched: bool,
    /// The exact settlement whose delivery is in flight. The drop backstop
    /// re-delivers this decided settlement instead of a cancellation, so a
    /// task cancelled mid-write can neither downgrade the ledger outcome nor
    /// diverge from the recorded metric.
    decided_settlement: Option<String>,
    pub started: Instant,
    /// Wall-clock time the winning attempt streamed its first output token,
    /// reported in the finalizing settlement so the control plane can derive
    /// time-to-first-token. `None` until an attempt observes a first token.
    first_token_at: Option<SystemTime>,
    /// Allowlisted rate-limit headers of the active attempt's OPENED provider
    /// response, recorded once per attempt at open and settled alongside the
    /// outcome. An attempt that failed at open instead carries them on its
    /// `Failure`, which settlement hoists into the same payload field.
    rate_limit_headers: Option<serde_json::Map<String, Value>>,
    /// The upstream an aggregator named as serving the active attempt's
    /// committed stream, recorded at commit and settled alongside the outcome.
    upstream_provider: Option<String>,
    /// How many web searches the control plane executed for this request
    /// before dispatch (from the admission); a request-level fact that
    /// survives `rebind` and rides only the finalizing settlement.
    web_search_requests: u32,
    /// How many gateway-run tool-search rounds this request completed so
    /// far (recorded by the waterfall as each round lands); request-level
    /// like the web-search count, so it survives `rebind` and rides only the
    /// finalizing settlement.
    tool_search_requests: u32,
    observation: Observation,
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
            dispatched: false,
            decided_settlement: None,
            started,
            first_token_at: None,
            rate_limit_headers: None,
            upstream_provider: None,
            web_search_requests: 0,
            tool_search_requests: 0,
            observation: Observation::default(),
        }
    }

    pub(crate) fn begin_dial_observation(&mut self) -> Observation {
        self.observation = self.observation.next_dial();
        self.observation.clone()
    }

    /// Share the selected attempt's meter without another JSON boundary or token count.
    pub(crate) fn capture_observation(&self) -> Observation {
        self.observation.clone()
    }

    /// Cancellation never certifies a partial meter as the provider's final bill.
    fn log_cancelled_meter(&self, observed: &observation::Observed) {
        let line = json!({
            "event": "stream_meter_at_cancel", "request_id": self.request_id,
            "attempt_id": self.attempt_id,
            "provider_terminal_observed": observed.terminal.is_some(),
            "usage_final": observed.terminal.is_some()
                && observed.usage.as_ref().is_some_and(Usage::has_token_counts),
            "input_tokens": observed.usage.as_ref().and_then(|usage| usage.input_tokens),
            "output_tokens": observed.usage.as_ref().and_then(|usage| usage.output_tokens),
        });
        eprintln!("exp-gateway-native: {line}");
    }

    /// Record the admission's count of gateway-executed web searches, so the
    /// finalizing settlement bills them.
    pub fn record_web_search_requests(&mut self, requests: u32) {
        self.web_search_requests = requests;
    }

    /// Record how many gateway-run tool-search rounds the request has
    /// completed, so the finalizing settlement bills them.
    pub fn record_tool_search_requests(&mut self, requests: u32) {
        self.tool_search_requests = requests;
    }

    /// Bind one freshly reserved attempt as the active settlement target.
    pub fn rebind(&mut self, attempt_id: String) {
        self.attempt_id = Some(attempt_id);
        self.observation = Observation::default();
        self.opened = false;
        self.dispatched = false;
        self.decided_settlement = None;
        // Each physical attempt observes its own first token; a prior failed
        // attempt's timing never carries into its successor. The same holds
        // for its provider response's rate-limit headers and the upstream
        // its stream named.
        self.first_token_at = None;
        self.rate_limit_headers = None;
        self.upstream_provider = None;
    }

    /// Record the wall-clock time the active attempt streamed its first output
    /// token, read from its relay just before settlement. Only the first
    /// observation for the attempt is kept.
    pub fn record_first_token(&mut self, at: Option<SystemTime>) {
        if self.first_token_at.is_none() {
            self.first_token_at = at;
        }
    }

    /// Mark physical dispatch before awaiting headers, so an early drop keeps its hold.
    pub(crate) fn mark_dispatched(&mut self) {
        self.dispatched = true;
    }

    /// Record that the active attempt's provider dispatch opened.
    pub fn mark_opened(&mut self) {
        self.dispatched = true;
        self.opened = true;
    }

    /// Record the allowlisted rate-limit headers of the active attempt's
    /// opened provider response, for settlement.
    pub fn record_rate_limit_headers(&mut self, headers: Option<serde_json::Map<String, Value>>) {
        self.rate_limit_headers = headers;
    }

    /// Record the upstream the active attempt's committed stream named as
    /// serving it (an aggregator's per-chunk provider label), for settlement.
    pub fn record_upstream_provider(&mut self, provider: Option<String>) {
        self.upstream_provider = provider;
    }

    /// Preserve independently validated decision usage before answer validation.
    pub fn record_decision_usage(&mut self, usage: Usage) {
        self.observation.record(&Event::Usage(usage));
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
        if !self.armed {
            return true;
        }
        let Some(attempt_id) = self.attempt_id.clone() else {
            // No active attempt: nothing durable to close here. The abandon
            // path owns request-only terminalization.
            return true;
        };
        let observed = self.observation.snapshot();
        let usage = observed.usage.as_ref().or(usage);
        self.record_first_token(observed.first_token_at);
        // An opened attempt's headers live on the guard; an attempt that
        // failed at open carries them on its failure instead.
        let rate_limit_headers = self
            .rate_limit_headers
            .as_ref()
            .or_else(|| failure.and_then(|failure| failure.rate_limit_headers.as_deref()));
        let argument = settle_argument(
            &self.request_id,
            &attempt_id,
            outcome,
            usage,
            tool_names,
            failure,
            finalize,
            self.opened,
            self.first_token_at,
            rate_limit_headers,
            self.upstream_provider.as_deref(),
            self.web_search_requests,
            self.tool_search_requests,
        );
        let argument = disconnect_provenance(
            argument,
            self.dispatched,
            self.dispatched
                && observed.terminal.is_none()
                && failure.is_some_and(|failure| failure.failure_class == FailureClass::Cancelled),
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
        if !self.armed {
            return true;
        }
        if self.attempt_id.is_some() {
            let observed = self.observation.snapshot();
            self.record_first_token(observed.first_token_at);
            self.log_cancelled_meter(&observed);
            let (outcome, failure) = cancellation_outcome(observed.terminal.as_ref());
            self.settle(
                outcome,
                observed.usage.as_ref().or(usage),
                if observed.tool_names.is_empty() {
                    tool_names
                } else {
                    &observed.tool_names
                },
                failure.as_ref(),
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

/// Stamp trusted dispatch evidence without changing the base settlement vocabulary.
fn disconnect_provenance(argument: String, dispatched: bool, incomplete: bool) -> String {
    let mut payload: Value =
        serde_json::from_str(&argument).expect("settlement JSON was encoded locally");
    payload["dispatched"] = json!(dispatched);
    payload["usage_incomplete_due_to_disconnect"] = json!(incomplete);
    compact_json(&payload)
}

/// An already observed provider terminal wins over subscriber cancellation.
fn cancellation_outcome(terminal: Option<&Event>) -> (&'static str, Option<Failure>) {
    match terminal {
        Some(Event::Failed(failure)) => ("failed", Some(failure.clone().boundary())),
        Some(Event::Incomplete | Event::IncompleteToolArguments) => ("incomplete", None),
        Some(_) => ("completed", None),
        None => (
            "failed",
            Some(Failure::new(
                FailureClass::Cancelled,
                "gateway request was cancelled",
            )),
        ),
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
        let decided = self.decided_settlement.is_some();
        let (method, argument): (&'static str, String) = match self.decided_settlement.take() {
            Some(argument) => ("settle", argument),
            None => match self.attempt_id.take() {
                Some(attempt_id) => {
                    let observed = self.observation.snapshot();
                    self.log_cancelled_meter(&observed);
                    let (outcome, failure) = cancellation_outcome(observed.terminal.as_ref());
                    self.record_terminal(outcome, observed.terminal.is_none());
                    crate::respond::log_stream_exit(&self.request_id, "handler_cancelled");
                    (
                        "settle",
                        settle_argument(
                            &self.request_id,
                            &attempt_id,
                            outcome,
                            observed.usage.as_ref(),
                            &observed.tool_names,
                            failure.as_ref(),
                            true,
                            self.opened,
                            self.first_token_at.or(observed.first_token_at),
                            self.rate_limit_headers.as_ref(),
                            self.upstream_provider.as_deref(),
                            self.web_search_requests,
                            self.tool_search_requests,
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
        let argument = if method == "settle" && !decided {
            // A decided retry already carries its exact provenance. New drop
            // settlements identify only genuinely dispatched cancellation.
            let observed = self.observation.snapshot();
            disconnect_provenance(
                argument,
                self.dispatched,
                self.dispatched && observed.terminal.is_none(),
            )
        } else {
            argument
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

#[cfg(test)]
#[path = "settlement_tests.rs"]
mod tests;
