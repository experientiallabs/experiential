//! Withhold the first outward event while the shared engine checks the input.

use std::sync::Arc;
#[cfg(test)]
use std::time::Duration;
#[cfg(test)]
use std::time::Instant;

use serde::Deserialize;
#[cfg(test)]
use serde_json::json;
use tokio::sync::OnceCell;

use crate::bridge::Bridge;
use crate::errors::{Failure, FailureClass, PublicError};
use crate::relay::collection_public_error;
use crate::settlement::AttemptGuard;
use crate::waterfall::{acquire_attempt, WaterfallContext, Won};

/// One cached decision shared by response release and every attempt settlement.
pub(crate) struct InputGate {
    request_id: String,
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
    pub(crate) fn public_error(self) -> PublicError {
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
                let value = bridge
                    .input_resources(&self.request_id)
                    .ok_or_else(unavailable)?
                    .wait()
                    .await;
                let decision: Decision =
                    crate::lossless_json::from_value(value).map_err(|_| unavailable())?;
                match decision.action.as_str() {
                    "allow" => Ok(InputDecision::Allowed),
                    "closed" => Ok(InputDecision::AdmissionClosed),
                    "error" => Err(decision.failure.unwrap_or_else(unavailable)),
                    _ => Err(unavailable()),
                }
            })
            .await
            .clone()
    }

    pub(crate) async fn response_ready(
        &self,
        bridge: &Bridge,
    ) -> Result<InputDecision, InputFailure> {
        let resources = bridge
            .input_resources(&self.request_id)
            .ok_or_else(unavailable)?;
        resources.begin_response();
        self.wait(bridge).await
    }

    pub(crate) fn new(ctx: &WaterfallContext<'_>) -> Arc<Self> {
        Arc::new(Self {
            request_id: ctx.request_id.to_owned(),
            decision: OnceCell::new(),
        })
    }

    /// Race one physical generation against its own input decision.
    pub(crate) async fn inspect_attempt<T>(
        &self,
        bridge: &Bridge,
        attempt: impl std::future::Future<Output = T>,
    ) -> Result<T, InputFailure> {
        tokio::pin!(attempt);
        tokio::select! {
            biased;
            decision = self.wait(bridge) => {
                decision?.require_allow()?;
                Ok(attempt.await)
            },
            end = &mut attempt => Ok(end),
        }
    }
}

/// Enable request-owned input decisions for each physical generation.
pub(crate) async fn acquire_guarded_attempt(
    ctx: &WaterfallContext<'_>,
    guard: &mut AttemptGuard,
    enabled: bool,
) -> Won {
    if enabled {
        guard.input_gate = Some(InputGate::new(ctx));
    }
    acquire_attempt(ctx, guard).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::prelude::*;

    struct NativePlane {
        input: Arc<crate::accounting::InputResources>,
        writes: tokio::sync::mpsc::UnboundedSender<serde_json::Value>,
    }
    impl crate::native_callbacks::CallbackTarget for NativePlane {
        fn handles(&self, method: &str) -> bool {
            method == "settle"
        }
        fn call(&self, method: &str, argument: &str) -> Result<String, PublicError> {
            assert_eq!(method, "settle");
            self.writes
                .send(crate::lossless_json::from_str(argument).unwrap())
                .unwrap();
            Ok("{}".into())
        }
        fn input_resources(&self, _: &str) -> Option<Arc<crate::accounting::InputResources>> {
            Some(self.input.clone())
        }
    }

    fn setup(
        grace: u64,
    ) -> (
        Arc<Bridge>,
        tokio::sync::mpsc::UnboundedReceiver<serde_json::Value>,
        std::sync::mpsc::Sender<()>,
        std::thread::JoinHandle<()>,
    ) {
        use std::io::{Read, Write};
        let server = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/inspect", server.local_addr().unwrap());
        let (release, released) = std::sync::mpsc::channel();
        let (received, accepted) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            let (mut stream, _) = server.accept().unwrap();
            let mut request = [0u8; 4096];
            let _ = stream.read(&mut request).unwrap();
            received.send(()).unwrap();
            released.recv_timeout(Duration::from_secs(5)).unwrap();
            let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 16\r\nConnection: close\r\n\r\n{\"flagged\":true}");
        });
        let input = Arc::new(crate::accounting::InputResources::test_input(
            json!([{
                "policy": {"mode": "enforce", "protected": true, "input_fail_open": true, "input_max_wait_ms": grace},
                "check": {"check_id": "moderation", "action": "block", "timeout_ms": 4000},
                "adapter": {"kind": "http_json", "url": url, "max_response_bytes": 1024, "payload": {},
                    "transport": {"public_only": false, "headers": {}}}
            }]),
            5.0,
        ));
        accepted.recv_timeout(Duration::from_secs(3)).unwrap();
        let (writes, receiver) = tokio::sync::mpsc::unbounded_channel();
        Python::initialize();
        let plane = Python::attach(|py| {
            pyo3::types::PyModule::from_code(py, c"class Plane:\n def __getattr__(self, name):\n  raise AssertionError('post-admission Python callback: ' + name)\n def close_thread_resources(self, argument): return '{}'\n", c"unused_plane.py", c"unused_plane").unwrap()
                .getattr("Plane").unwrap().call0().unwrap().unbind()
        });
        let target =
            crate::native_callbacks::NativeTarget::new(Arc::new(NativePlane { input, writes }));
        (
            Arc::new(Bridge::new(plane, 1).unwrap().with_native(Some(target))),
            receiver,
            release,
            thread,
        )
    }

    #[tokio::test]
    async fn denial_cancelling_provider_settlement_keeps_health_facts() {
        for drop_handler in [false, true] {
            let (bridge, mut writes, release, server) = setup(100);
            let gate = Arc::new(InputGate {
                request_id: "request".into(),
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
            let headers = json!({"retry-after": "120", "x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "120s"}).as_object().unwrap().clone();
            let provider = crate::upstream::transport_failure(Some(429))
                .with_rate_limit_facts(Some(headers.clone()), Some(120));
            guard
                .capture_observation()
                .record(&crate::events::Event::Failed(provider.clone()));
            let started = Arc::new(tokio::sync::Notify::new());
            let notified = started.clone();
            tokio::spawn(async move {
                notified.notified().await;
                release.send(()).unwrap();
            });
            let denied = {
                let settling = async {
                    started.notify_one();
                    guard
                        .settle("failed", None, &[], Some(&provider), true)
                        .await
                };
                tokio::pin!(settling);
                tokio::select! {
                    biased;
                    decision = gate.wait(&bridge) => decision.err().expect("input denial"),
                    _ = &mut settling => panic!("the outer input gate must win"),
                }
            };
            assert_eq!(denied.failure.failure_class, FailureClass::Guardrail);
            if drop_handler {
                drop(guard);
            } else {
                guard.fail_before_release(&denied.failure).await;
            }
            let write = tokio::time::timeout(Duration::from_secs(2), writes.recv())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(write["failure"]["failure_class"], "throttled");
            assert_eq!(write["failure"]["retry_after_seconds"], 120);
            assert_eq!(write["rate_limit_headers"], json!(headers));
            assert_eq!(write["finalize"], true);
            assert_eq!(write["usage_incomplete_due_to_disconnect"], false);
            assert!(writes.try_recv().is_err());
            server.join().unwrap();
        }
    }

    #[tokio::test]
    async fn ready_response_releases_after_configured_grace_without_polling() {
        for grace in [0, 20] {
            let (bridge, _, release, server) = setup(grace);
            let gate = InputGate {
                request_id: "request".into(),
                decision: OnceCell::new(),
            };
            let started = Instant::now();
            let result =
                tokio::time::timeout(Duration::from_millis(500), gate.response_ready(&bridge))
                    .await
                    .unwrap();
            result.unwrap().require_allow().unwrap();
            assert!(started.elapsed() >= Duration::from_millis(grace));
            let resources = bridge.input_resources("request").unwrap();
            assert_eq!(
                resources.status()["moderation"]["reason"],
                "response_ready_timeout"
            );
            release.send(()).unwrap();
            server.join().unwrap();
            gate.wait(&bridge).await.unwrap().require_allow().unwrap();
        }
    }

    #[test]
    fn capture_error_keeps_its_code_and_underlying_settlement_failure() {
        let decision: Decision = crate::lossless_json::from_value(json!({
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
        let decision: Decision = crate::lossless_json::from_value(json!({
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
        let decision = crate::lossless_json::from_value::<Decision>(json!({
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
