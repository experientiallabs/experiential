//! Request-owned optional inspection shared by delivery and durable settlement.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::Notify;

use super::input::{InputDecision, InputFailure};
use crate::capture::collector::Collector;
use crate::capture::python::CaptureCollector;
use crate::errors::FailureClass;

#[derive(Clone)]
enum Inspection {
    Pending,
    Allowed,
    Expired,
    Denied(InputFailure),
}

enum Release {
    Pending,
    Registering,
    Ready(Result<InputDecision, InputFailure>),
    Closed,
}

struct DeferredCapture {
    collector: Arc<Collector>,
    bytes: Vec<u8>,
}

/// The sole optional verdict; neither interpreter scheduling nor a late result renews it.
pub(crate) struct InputLatch {
    pub(crate) request_id: String,
    wait: Duration,
    started: OnceLock<Instant>,
    inspection: Mutex<Inspection>,
    release: Mutex<Release>,
    capture: Mutex<Option<DeferredCapture>>,
    changed: Notify,
}

impl InputLatch {
    pub(crate) fn begin(&self) {
        self.started.get_or_init(Instant::now);
        self.changed.notify_waiters();
    }

    fn elapsed(&self) -> Option<Duration> {
        self.started.get().map(Instant::elapsed)
    }

    fn inspection(&self) -> Inspection {
        let mut state = self.inspection.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(*state, Inspection::Pending)
            && self.elapsed().is_some_and(|elapsed| elapsed >= self.wait)
        {
            *state = Inspection::Expired;
        }
        state.clone()
    }

    fn publish(&self, verdict: Inspection) -> bool {
        let mut state = self.inspection.lock().unwrap_or_else(|e| e.into_inner());
        if !matches!(*state, Inspection::Pending) {
            return false;
        }
        *state = if self.elapsed().is_some_and(|elapsed| elapsed >= self.wait) {
            Inspection::Expired
        } else {
            verdict
        };
        self.changed.notify_waiters();
        true
    }

    fn close(&self) {
        let mut release = self.release.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(*release, Release::Pending | Release::Registering) {
            *release = Release::Closed;
            self.changed.notify_waiters();
        }
        drop(release);
        self.capture
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
    }

    fn decision_json(&self) -> String {
        let release = self.release.lock().unwrap_or_else(|e| e.into_inner());
        match &*release {
            Release::Pending | Release::Registering => json!({"action":"pending"}),
            Release::Closed | Release::Ready(Ok(InputDecision::AdmissionClosed)) => {
                json!({"action":"closed"})
            }
            Release::Ready(Ok(InputDecision::Allowed)) => json!({"action":"allow"}),
            Release::Ready(Err(failure)) => json!({"action":"error", "failure":failure}),
        }
        .to_string()
    }

    /// Optional inspection expires without a Python callback. Required capture does not.
    pub(crate) async fn wait(&self, deadline: Instant) -> Result<InputDecision, InputFailure> {
        let inspected = loop {
            let changed = self.changed.notified();
            if matches!(
                *self.release.lock().unwrap_or_else(|e| e.into_inner()),
                Release::Closed
            ) {
                return Ok(InputDecision::AdmissionClosed);
            }
            let state = self.inspection();
            if !matches!(state, Inspection::Pending) {
                break state;
            }
            let end = self
                .started
                .get()
                .map_or(deadline, |started| deadline.min(*started + self.wait));
            if Instant::now() >= end {
                self.publish(Inspection::Expired);
                continue;
            }
            tokio::select! {
                _ = changed => {},
                _ = tokio::time::sleep_until(end.into()) => {},
            }
        };
        {
            let mut release = self.release.lock().unwrap_or_else(|e| e.into_inner());
            match &*release {
                Release::Ready(decision) => return decision.clone(),
                Release::Closed => return Ok(InputDecision::AdmissionClosed),
                Release::Pending => *release = Release::Registering,
                Release::Registering => return Err(super::input::unavailable()),
            }
        }
        // Transfer the envelope once; the collector owns its bounded/spilled
        // representation after registration, so the latch retains no copy.
        let capture = self
            .capture
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        let mut began = None;
        let decision = match inspected {
            Inspection::Denied(failure) => Err(failure),
            Inspection::Allowed | Inspection::Expired => {
                if let Some(capture) = capture {
                    let registered =
                        crate::capture::python::begin_bytes(&capture.collector, &capture.bytes);
                    drop(capture.bytes);
                    if registered {
                        began = Some(capture.collector);
                        Ok(InputDecision::Allowed)
                    } else {
                        Err(super::input::capture_unavailable())
                    }
                } else {
                    Ok(InputDecision::Allowed)
                }
            }
            Inspection::Pending => unreachable!(),
        };
        let mut release = self.release.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(*release, Release::Closed) {
            drop(release);
            if let Some(collector) = began {
                collector.settle(&self.request_id, false, false);
            }
            return Ok(InputDecision::AdmissionClosed);
        }
        *release = Release::Ready(decision.clone());
        decision
    }
}

/// An in-process capability owned by one admitted request, never supplied by HTTP callers.
#[pyclass(name = "OptionalInputDecision")]
pub struct PyOptionalInputDecision {
    pub(crate) inner: Arc<InputLatch>,
}

#[pymethods]
impl PyOptionalInputDecision {
    #[new]
    #[pyo3(signature = (request_id, wait_ms, collector=None, capture_json=None))]
    fn new(
        request_id: String,
        wait_ms: u64,
        collector: Option<PyRef<'_, CaptureCollector>>,
        capture_json: Option<&[u8]>,
    ) -> PyResult<Self> {
        if request_id.is_empty() || wait_ms > 5000 || collector.is_some() != capture_json.is_some()
        {
            return Err(PyValueError::new_err(
                "invalid optional input release contract",
            ));
        }
        let capture = match (collector, capture_json) {
            (Some(collector), Some(bytes)) => {
                #[derive(Deserialize)]
                struct CaptureOwner {
                    request_id: String,
                }
                let owner: CaptureOwner = serde_json::from_slice(bytes)
                    .map_err(|_| PyValueError::new_err("invalid deferred capture envelope"))?;
                if owner.request_id != request_id {
                    return Err(PyValueError::new_err(
                        "deferred capture owner does not match",
                    ));
                }
                Some(DeferredCapture {
                    collector: collector.inner.clone(),
                    bytes: bytes.to_vec(),
                })
            }
            _ => None,
        };
        Ok(Self {
            inner: Arc::new(InputLatch {
                request_id,
                wait: Duration::from_millis(wait_ms),
                started: OnceLock::new(),
                inspection: Mutex::new(Inspection::Pending),
                release: Mutex::new(Release::Pending),
                capture: Mutex::new(capture),
                changed: Notify::new(),
            }),
        })
    }

    /// Publish only a completed optional decision; a deadline or prior result wins forever.
    fn publish(&self, decision_json: &str) -> PyResult<bool> {
        let value: serde_json::Value = serde_json::from_str(decision_json)
            .map_err(|_| PyValueError::new_err("invalid optional input decision"))?;
        let decision = match value.get("action").and_then(|v| v.as_str()) {
            Some("allow") => Inspection::Allowed,
            Some("error") => {
                let failure: InputFailure =
                    serde_json::from_value(value.get("failure").cloned().unwrap_or_default())
                        .map_err(|_| PyValueError::new_err("invalid optional input failure"))?;
                if failure.failure.failure_class == FailureClass::Guardrail {
                    Inspection::Denied(failure)
                } else {
                    Inspection::Allowed
                }
            }
            _ => return Err(PyValueError::new_err("invalid optional input action")),
        };
        Ok(self.inner.publish(decision))
    }

    fn elapsed_seconds(&self) -> Option<f64> {
        self.inner.elapsed().map(|elapsed| elapsed.as_secs_f64())
    }

    fn expired(&self) -> bool {
        matches!(self.inner.inspection(), Inspection::Expired)
    }

    fn decision_json(&self) -> String {
        self.inner.decision_json()
    }

    fn close(&self) {
        self.inner.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn latch() -> InputLatch {
        InputLatch {
            request_id: "owned-request".into(),
            wait: Duration::from_millis(250),
            started: OnceLock::new(),
            inspection: Mutex::new(Inspection::Pending),
            release: Mutex::new(Release::Pending),
            capture: Mutex::new(None),
            changed: Notify::new(),
        }
    }

    fn captured_latch() -> InputLatch {
        use crate::capture::delivery::Sink;
        use crate::capture::record::Record;

        struct DiscardSink;
        impl Sink for DiscardSink {
            type Prepared = ();
            fn preparation_bytes(_: usize) -> usize {
                0
            }
            fn prepare(&self, _: &Record, _: usize) -> Result<(), ()> {
                Ok(())
            }
            fn write(&mut self, _: &()) -> Result<(), ()> {
                Ok(())
            }
        }
        let config = serde_json::from_value(json!({
            "delivery": {"maximum_records":8,"maximum_bytes":65536,"maximum_record_bytes":8192},
            "maximum_pending_records":4,"maximum_pending_bytes":65536,
            "maximum_request_bytes":4096,"maximum_response_bytes":4096,
            "ttl_seconds":30,"settlement_required":true
        }))
        .unwrap();
        let input = latch();
        *input.capture.lock().unwrap() = Some(DeferredCapture {
            collector: Arc::new(Collector::new(config, DiscardSink).unwrap()),
            bytes: serde_json::to_vec(&json!({
                "request_id":"owned-request",
                "scope":{"organization_id":"org","identity_id":"identity","application_id":"app"},
                "protocol":"chat_completions","model_id":"model",
                "context":{"schema_version":1,"request":{"messages":[{"role":"user","content":"task"}]}}
            })).unwrap(),
        });
        input
    }

    #[tokio::test]
    async fn deferred_envelope_is_released_after_registration_denial_or_close() {
        for action in ["allow", "deny", "close"] {
            let input = captured_latch();
            let collector = input
                .capture
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .collector
                .clone();
            if action == "close" {
                input.close();
            } else {
                input.publish(if action == "allow" {
                    Inspection::Allowed
                } else {
                    denial()
                });
                let decision = input.wait(Instant::now()).await;
                assert_eq!(decision.is_ok(), action == "allow");
            }
            assert!(input.capture.lock().unwrap().is_none());
            collector.settle("owned-request", false, false);
            assert!(collector.close_until(Instant::now() + Duration::from_secs(1)));
        }
    }

    fn denial() -> Inspection {
        let mut failure = super::super::input::unavailable();
        failure.failure = crate::errors::Failure::new(FailureClass::Guardrail, "Input rejected.");
        Inspection::Denied(failure)
    }

    #[tokio::test]
    async fn optional_wait_expires_without_interpreter_progress() {
        let input = latch();
        let deadline = Instant::now() + Duration::from_secs(3);
        // Provider execution consumes none of the optional response allowance.
        assert!(
            tokio::time::timeout(Duration::from_millis(30), input.wait(deadline))
                .await
                .is_err()
        );
        assert!(matches!(input.inspection(), Inspection::Pending));
        input.begin();
        let started = Instant::now();
        let decision = tokio::time::timeout(Duration::from_millis(400), input.wait(deadline))
            .await
            .expect("native timer must not depend on Python progress")
            .unwrap();
        assert!(matches!(decision, InputDecision::Allowed));
        assert!(started.elapsed() >= Duration::from_millis(245));
        assert!(matches!(input.inspection(), Inspection::Expired));
        assert_eq!(input.decision_json(), "{\"action\":\"allow\"}");
    }

    #[tokio::test]
    async fn elapsed_budget_cannot_be_renewed_by_a_late_verdict_or_wait() {
        let input = latch();
        input
            .started
            .set(Instant::now() - Duration::from_millis(251))
            .unwrap();
        input.publish(denial());
        assert!(matches!(input.inspection(), Inspection::Expired));
        let result = input
            .wait(Instant::now() + Duration::from_secs(1))
            .await
            .unwrap();
        assert!(matches!(result, InputDecision::Allowed));
        assert!(!input.publish(denial()));
        input.begin();
        assert!(input.elapsed().unwrap() >= Duration::from_millis(251));
        assert!(matches!(
            input.wait(Instant::now()).await.unwrap(),
            InputDecision::Allowed
        ));
    }

    #[tokio::test]
    async fn timely_denial_remains_denied_after_expiry() {
        let input = latch();
        assert!(input.publish(denial()));
        input
            .started
            .set(Instant::now() - Duration::from_secs(1))
            .unwrap();
        let failure = input
            .wait(Instant::now())
            .await
            .err()
            .expect("timely denial must win");
        assert_eq!(failure.failure.failure_class, FailureClass::Guardrail);
        assert!(!input.publish(Inspection::Allowed));
    }

    #[tokio::test]
    async fn zero_wait_preserves_published_denial_but_expires_pending_inspection() {
        for published_before_boundary in [true, false] {
            let mut input = latch();
            input.wait = Duration::ZERO;
            if published_before_boundary {
                assert!(input.publish(denial()));
            }
            input.begin();
            if !published_before_boundary {
                assert!(matches!(input.inspection(), Inspection::Expired));
                assert!(!input.publish(denial()));
            }
            let decision = input.wait(Instant::now()).await;
            if published_before_boundary {
                assert_eq!(
                    decision
                        .err()
                        .expect("published denial must survive")
                        .failure
                        .failure_class,
                    FailureClass::Guardrail
                );
            } else {
                assert!(matches!(decision.unwrap(), InputDecision::Allowed));
            }
        }
    }

    #[tokio::test]
    async fn cancellation_never_reopens_a_pending_release() {
        let input = latch();
        input.close();
        input.publish(Inspection::Allowed);
        assert!(matches!(
            input.wait(Instant::now()).await.unwrap(),
            InputDecision::AdmissionClosed
        ));
        assert_eq!(input.decision_json(), "{\"action\":\"closed\"}");
    }

    #[tokio::test]
    async fn completed_release_keeps_its_settlement_disposition_after_close() {
        let input = latch();
        input.publish(Inspection::Allowed);
        assert!(matches!(
            input.wait(Instant::now()).await.unwrap(),
            InputDecision::Allowed
        ));
        input.close();
        assert_eq!(input.decision_json(), "{\"action\":\"allow\"}");
        assert!(!input.publish(denial()));
    }
}
