//! Exactly-once settlement of admitted requests and their physical attempts.
//!
//! Every durable accounting write flows through this module: the bounded
//! retry delivery to the control plane, the `AttemptGuard` that owns one
//! request's terminal outcome (including the drop backstop for cancelled
//! handlers and unwound stream tasks), and the shutdown-drain hold that keeps
//! graceful stop waiting for detached stream settlements.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::bridge::Bridge;
use crate::encode::compact_json;
use crate::errors::{Failure, FailureClass};
use crate::events::Usage;
use crate::metrics::METRICS;

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
