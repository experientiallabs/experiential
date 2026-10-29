//! Withhold the first outward event while the shared engine checks the input.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::json;
use tokio::sync::OnceCell;

use crate::bridge::Bridge;
use crate::encode::compact_json;
use crate::errors::{Failure, FailureClass, PublicError};
use crate::relay::{collection_public_error, remaining};
use crate::settlement::AttemptGuard;
use crate::waterfall::{acquire_attempt, WaterfallContext, Won};

/// One cached decision shared by response release and every attempt settlement.
pub(crate) struct InputGate {
    request_id: String,
    deadline: Instant,
    decision: OnceCell<Result<InputDecision, InputFailure>>,
}

#[derive(Clone, Copy)]
pub(crate) enum InputDecision {
    Allowed,
    AdmissionClosed,
}

impl InputDecision {
    pub(crate) fn require_allow(self) -> Result<(), InputFailure> {
        match self {
            Self::Allowed => Ok(()),
            Self::AdmissionClosed => Err(unavailable()),
        }
    }
}

#[derive(Deserialize)]
struct Decision {
    action: String,
    failure: Option<InputFailure>,
}

/// Closed vocabulary of public errors authorized by the request-owned input session.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum InputFailureCode {
    CaptureUnavailable,
}

/// Only the typed public code is consumed; other session facts remain control-plane owned.
#[derive(Debug, Clone, Default, Deserialize)]
struct InputFailureDetails {
    code: Option<InputFailureCode>,
}

/// One cached session failure with its public code and unchanged settlement facts.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct InputFailure {
    #[serde(flatten)]
    pub(crate) failure: Failure,
    #[serde(default)]
    safe_details: InputFailureDetails,
}

impl InputFailure {
    /// Preserve capture admission's typed 503 without changing ordinary classifier failures.
    fn public_error(self) -> PublicError {
        let failure = self.failure.boundary();
        match self.safe_details.code {
            Some(InputFailureCode::CaptureUnavailable)
                if failure.failure_class == FailureClass::Unavailable =>
            {
                PublicError::new(
                    503,
                    "capture_unavailable",
                    &failure.safe_message,
                    "invalid_request_error",
                )
            }
            _ => collection_public_error(&failure),
        }
    }
}

fn unavailable() -> InputFailure {
    InputFailure {
        failure: Failure::new(
            FailureClass::Unavailable,
            "Content inspection is unavailable. Retry later.",
        ),
        safe_details: InputFailureDetails::default(),
    }
}

impl InputGate {
    pub(crate) async fn wait(&self, bridge: &Bridge) -> Result<InputDecision, InputFailure> {
        self.decision
            .get_or_init(|| async {
                // Each callback is a nonblocking poll. No classifier can occupy
                // all bridge workers and prevent provider dispatch/settlement.
                let argument = compact_json(&json!({"request_id": self.request_id}));
                let mut delay_ms = 1;
                loop {
                    let response = tokio::time::timeout(
                        remaining(self.deadline),
                        bridge.call("guardrail_input_status", argument.clone()),
                    )
                    .await
                    .map_err(|_| unavailable())?
                    .map_err(|_| unavailable())?;
                    let decision: Decision =
                        serde_json::from_str(&response).map_err(|_| unavailable())?;
                    match decision.action.as_str() {
                        "allow" => return Ok(InputDecision::Allowed),
                        "closed" => return Ok(InputDecision::AdmissionClosed),
                        "error" => return Err(decision.failure.unwrap_or_else(unavailable)),
                        "pending" if Instant::now() < self.deadline => {
                            tokio::time::sleep(
                                Duration::from_millis(delay_ms).min(remaining(self.deadline)),
                            )
                            .await;
                            delay_ms = (delay_ms * 2).min(10);
                        }
                        _ => return Err(unavailable()),
                    }
                }
            })
            .await
            .clone()
    }
}

fn release(won: Won, decision: InputDecision) -> Result<Won, InputFailure> {
    // A terminal admission failure owns its public error. A closed admission
    // never authorizes releasing a provider completion or any prefix.
    if matches!(won, Won::Failed(_)) {
        return Ok(won);
    }
    decision.require_allow().map(|()| won)
}

/// Generate and inspect concurrently, withholding all protocol/replay/capture output.
pub(crate) async fn acquire_guarded_attempt(
    ctx: &WaterfallContext<'_>,
    guard: &mut AttemptGuard,
    enabled: bool,
) -> Won {
    if !enabled {
        return acquire_attempt(ctx, guard).await;
    }
    let gate = Arc::new(InputGate {
        request_id: ctx.request_id.to_owned(),
        deadline: ctx.deadline,
        decision: OnceCell::new(),
    });
    guard.input_gate = Some(gate.clone());
    let outcome = {
        let acquire = acquire_attempt(ctx, guard);
        tokio::pin!(acquire);
        tokio::select! {
            biased;
            decision = gate.wait(ctx.bridge) => match decision {
                Ok(decision) => release(acquire.await, decision),
                Err(failure) => Err(failure),
            },
            won = &mut acquire => {
                if matches!(won, Won::Failed(_)) {
                    Ok(won)
                } else {
                    gate.wait(ctx.bridge).await.and_then(|decision| release(won, decision))
                }
            },
        }
    };
    match outcome {
        Ok(won) => won,
        Err(failure) => {
            // Dropping acquire/the committed relay closes upstream immediately.
            // The guard retains witnessed usage for our cost, while the host
            // applies the input verdict to customer settlement.
            guard.fail_before_release(&failure.failure).await;
            Won::Failed(failure.public_error())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::prelude::*;

    #[tokio::test]
    async fn denial_cancelling_provider_settlement_keeps_health_facts() {
        Python::initialize();
        for drop_handler in [false, true] {
            let plane = Python::attach(|py| {
                pyo3::types::PyModule::from_code(py, c"import json\nclass Plane:\n def __init__(self): self.writes = []; self.polls = 0\n def guardrail_input_status(self, argument):\n  self.polls += 1\n  if self.polls == 1: return '{\"action\":\"pending\"}'\n  return json.dumps({'action':'error','failure':{'failure_class':'guardrail','safe_message':'Input was rejected.'}})\n def settle(self, argument):\n  self.writes.append(json.loads(argument))\n  return '{}'\n def close_thread_resources(self, argument): return '{}'\n", c"provider_input_plane.py", c"provider_input_plane")
                    .unwrap().getattr("Plane").unwrap().call0().unwrap().unbind()
            });
            let bridge =
                Arc::new(Bridge::new(Python::attach(|py| plane.clone_ref(py)), 1).unwrap());
            let gate = Arc::new(InputGate {
                request_id: "request".into(),
                deadline: Instant::now() + Duration::from_secs(2),
                decision: OnceCell::new(),
            });
            let mut guard = AttemptGuard::new(
                bridge.clone(),
                Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                "request".into(),
                Instant::now(),
            );
            guard.rebind("attempt".into());
            guard.mark_dispatched();
            guard.input_gate = Some(gate.clone());
            let headers = json!({
                "retry-after": "120",
                "x-ratelimit-remaining-requests": "0",
                "x-ratelimit-reset-requests": "120s"
            })
            .as_object()
            .unwrap()
            .clone();
            let provider = crate::upstream::transport_failure(Some(429))
                .with_rate_limit_facts(Some(headers.clone()), Some(120));
            // A failed open is retained before the waterfall can await the shared gate.
            guard
                .capture_observation()
                .record(&crate::events::Event::Failed(provider.clone()));
            let mut settling_polled = false;
            let denied = {
                let settling = async {
                    settling_polled = true;
                    guard
                        .settle("failed", None, &[], Some(&provider), false)
                        .await
                };
                tokio::pin!(settling);
                tokio::select! {
                    biased;
                    decision = gate.wait(&bridge) => decision.err().expect("input denial"),
                    _ = &mut settling => panic!("the outer input gate must win"),
                }
            };
            assert!(
                settling_polled,
                "cancel an actual in-flight settlement future"
            );
            if drop_handler {
                drop(guard);
            } else {
                guard.fail_before_release(&denied.failure).await;
            }
            tokio::time::timeout(Duration::from_secs(2), async {
                while Python::attach(|py| plane.bind(py).getattr("writes").unwrap().len().unwrap())
                    == 0
                {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            })
            .await
            .expect("terminal settlement delivered");
            let writes: String = Python::attach(|py| {
                py.import("json")
                    .unwrap()
                    .call_method1("dumps", (plane.bind(py).getattr("writes").unwrap(),))
                    .unwrap()
                    .extract()
                    .unwrap()
            });
            let writes: Vec<serde_json::Value> = serde_json::from_str(&writes).unwrap();
            assert_eq!(writes.len(), 1);
            assert_eq!(writes[0]["failure"]["failure_class"], "throttled");
            assert_eq!(writes[0]["failure"]["retry_after_seconds"], 120);
            assert_eq!(writes[0]["rate_limit_headers"], json!(headers));
            assert_eq!(writes[0]["finalize"], true);
            assert_eq!(writes[0]["usage_incomplete_due_to_disconnect"], false);
        }
    }

    #[test]
    fn capture_error_keeps_its_code_and_underlying_settlement_failure() {
        let decision: Decision = serde_json::from_value(json!({
            "action": "error",
            "failure": {
                "failure_class": "unavailable",
                "safe_message": "Traffic capture is unavailable or at capacity.",
                "safe_details": {"code": "capture_unavailable", "input_guardrail_denied": true},
                "retryable_same_deployment": false,
                "failover_eligible": false
            }
        }))
        .expect("typed session failure");
        let failure = decision.failure.expect("capture failure");
        assert_eq!(failure.failure.failure_class, FailureClass::Unavailable);
        assert!(!failure.failure.retryable_same_deployment);
        assert!(!failure.failure.failover_eligible);
        let error = failure.public_error();
        assert_eq!(error.status_code, 503);
        assert_eq!(error.code, "capture_unavailable");
        assert_eq!(error.error_type, "invalid_request_error");
        assert_eq!(error.retry_after_seconds, None);
    }

    #[test]
    fn ordinary_classifier_failure_keeps_gateway_unavailable() {
        let decision: Decision = serde_json::from_value(json!({
            "action": "error",
            "failure": {
                "failure_class": "unavailable",
                "safe_message": "Content inspection is unavailable.",
                "safe_details": {"input_guardrail_denied": true}
            }
        }))
        .expect("ordinary session failure");
        let error = decision.failure.expect("classifier failure").public_error();
        assert_eq!(error.status_code, 503);
        assert_eq!(error.code, "gateway_unavailable");
        assert_eq!(error.error_type, "api_error");
    }

    #[test]
    fn unknown_public_code_rejects_the_input_decision() {
        let decision = serde_json::from_value::<Decision>(json!({
            "action": "error",
            "failure": {
                "failure_class": "unavailable",
                "safe_message": "untrusted diagnostic",
                "safe_details": {"code": "unsupported-code"}
            }
        }));
        assert!(decision.is_err(), "unknown public code must fail closed");
        assert_eq!(unavailable().public_error().code, "gateway_unavailable");
    }
}
