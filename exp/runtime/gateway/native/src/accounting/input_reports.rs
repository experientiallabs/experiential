//! Content-free moderation records delivered away from response and settlement tasks.

use crate::native_callbacks::NativeTarget;
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc::{sync_channel, SyncSender},
    Arc, Condvar, Mutex, OnceLock,
};
use std::time::Instant;

/// One accounting instance owns only its observations and queued storage writes.
#[derive(Default)]
pub(super) struct Reports {
    pending: Mutex<usize>,
    changed: Condvar,
    dropped: AtomicU64,
    failed: AtomicU64,
    telemetry: Mutex<super::moderation_telemetry::Telemetry>,
}

impl Reports {
    pub fn drain_telemetry(&self) -> Value {
        self.telemetry
            .lock()
            .map(|mut state| state.drain())
            .unwrap_or_else(|_| json!({"aggregates":[],"dropped":1}))
    }
    /// Shutdown never waits for another worker's requests or enters a response task.
    pub fn drain(&self, deadline: Instant) -> bool {
        let Ok(pending) = self.pending.lock() else {
            return false;
        };
        self.changed
            .wait_timeout_while(
                pending,
                deadline.saturating_duration_since(Instant::now()),
                |n| *n != 0,
            )
            .is_ok_and(|(pending, _)| *pending == 0)
    }

    pub fn counters(&self) -> Value {
        json!({"dropped": self.dropped.load(Ordering::Relaxed),
            "failed": self.failed.load(Ordering::Relaxed),
            "telemetry":self.telemetry.lock().map(|state| state.metrics()).unwrap_or(Value::Null),
            "pending": self.pending.lock().map(|n| *n).unwrap_or(0)})
    }
}

struct Pending {
    reports: Arc<Reports>,
    completed: AtomicBool,
}
impl Pending {
    fn new(reports: Arc<Reports>) -> Option<Self> {
        *reports.pending.lock().ok()? += 1;
        Some(Self {
            reports,
            completed: AtomicBool::new(false),
        })
    }
    fn complete(&self) {
        if !self.completed.swap(true, Ordering::AcqRel) {
            if let Ok(mut pending) = self.reports.pending.lock() {
                *pending = pending.saturating_sub(1);
                self.reports.changed.notify_all();
            }
        }
    }
}
impl Drop for Pending {
    fn drop(&mut self) {
        self.complete();
    }
}

struct Job {
    target: Arc<NativeTarget>,
    argument: String,
    pending: Pending,
}

fn sender() -> Option<&'static SyncSender<Job>> {
    static QUEUE: OnceLock<Option<SyncSender<Job>>> = OnceLock::new();
    QUEUE
        .get_or_init(|| {
            let (sender, receiver) = sync_channel::<Job>(256);
            let receiver = Arc::new(Mutex::new(receiver));
            for index in 0..2 {
                let receiver = receiver.clone();
                std::thread::Builder::new()
                    .name(format!("gateway-moderation-record-{index}"))
                    .spawn(move || loop {
                        let job = match receiver.lock() {
                            Ok(receiver) => receiver.recv(),
                            Err(_) => break,
                        };
                        let Ok(job) = job else { break };
                        if job
                            .target
                            .call("record_input_moderation", &job.argument)
                            .is_err()
                        {
                            job.pending.reports.failed.fetch_add(1, Ordering::Relaxed);
                        }
                        // The job's lease is released after the synchronous native write returns.
                    })
                    .ok()?;
            }
            Some(sender)
        })
        .as_ref()
}

#[derive(Clone)]
pub(super) struct Reporter {
    inspection: Arc<Pending>,
    target: Arc<NativeTarget>,
    request_id: String,
    inspection_id: String,
}

#[derive(Clone)]
pub(super) struct Factory {
    pub target: Arc<NativeTarget>,
    pub request_id: String,
    pub reports: Arc<Reports>,
}

impl Factory {
    /// Preserve the existing decision rollup for native output and rewriting checks.
    pub fn decision(&self, check: &super::inspection::Check, result: &Value, started: Instant) {
        let Some(reporter) = self.reporter() else {
            return;
        };
        let failed_open = result["failed_open"] == true;
        let action = if failed_open {
            "allow"
        } else {
            result["failure"]["safe_details"]["action"]
                .as_str()
                .or_else(|| result["action"].as_str())
                .unwrap_or("error")
        };
        let unavailable = result["failure"]["failure_class"] == "unavailable";
        let unsupported = result["failure"]["failure_class"] == "unsupported_capability";
        let checkless = result["failure"].is_object()
            && result["failure"]["safe_details"]["check_id"].is_null();
        reporter.submit(&json!({"moderation": {
            "status": if failed_open {"failed_open"} else if unavailable {"unavailable"} else if action == "block" || action == "error" {"denied"} else {"allowed"},
            "observations": [{"policy_id":check.policy["policy_id"], "revision":check.policy["revision"],
                "check_id":if checkless {Value::Null} else {check.check["check_id"].clone()},
                "adapter_id":if checkless {Value::Null} else {check.check["adapter_id"].clone()},
                "capability":if checkless {Value::Null} else {check.check["capability"].clone()},
                "stage":if checkless {Value::Null} else {check.check["stage"].clone()},
                "mode":"enforce", "action":action,
                "would_action":check.check["action"],
                "classifier_kind":result["classifier_kind"], "scores":result["scores"],
                "cached":result["cached"], "subject_sha256":result["subject_sha256"],
                "outcome":if unavailable {"unavailable"} else if unsupported {"unsupported"} else if failed_open {"uncertain"} else if action == "allow" {"allow"} else {"flagged"},
                "latency_ms":started.elapsed().as_secs_f64() * 1000.0}]
        }}));
    }

    pub fn reporter(&self) -> Option<Reporter> {
        Reporter::new(
            self.target.clone(),
            self.request_id.clone(),
            self.reports.clone(),
        )
    }
}

impl Reporter {
    pub fn new(
        target: Arc<NativeTarget>,
        request_id: String,
        reports: Arc<Reports>,
    ) -> Option<Self> {
        if !target.handles("record_input_moderation") {
            return None;
        }
        Some(Self {
            inspection: Arc::new(Pending::new(reports)?),
            target,
            request_id,
            inspection_id: uuid::Uuid::new_v4().to_string(),
        })
    }
    pub fn submit(&self, decision: &Value) {
        let reports = &self.inspection.reports;
        if let Ok(mut telemetry) = reports.telemetry.lock() {
            telemetry.record(&decision["moderation"]);
        }
        let argument = json!({"request_id": self.request_id, "inspection_id": self.inspection_id,
            "moderation": decision["moderation"]})
        .to_string();
        let accepted = if argument.len() > 262_144 {
            false
        } else if let Some(pending) = Pending::new(reports.clone()) {
            sender().is_some_and(|sender| {
                sender
                    .try_send(Job {
                        target: self.target.clone(),
                        argument,
                        pending,
                    })
                    .is_ok()
            })
        } else {
            false
        };
        if !accepted {
            reports.dropped.fetch_add(1, Ordering::Relaxed);
        }
        self.inspection.complete();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn observations_are_owned_until_submission_or_cancellation_and_never_cross_workers() {
        let first = Arc::new(Reports::default());
        let other = Arc::new(Reports::default());
        let inspection = Arc::new(Pending::new(first.clone()).unwrap());
        let retained = inspection.clone();
        assert!(!first.drain(Instant::now()));
        assert!(other.drain(Instant::now()));
        drop(inspection);
        assert!(!first.drain(Instant::now()));
        let write = Pending::new(first.clone()).unwrap();
        retained.complete();
        retained.complete();
        assert_eq!(first.counters()["pending"], 1);
        assert!(!first.drain(Instant::now()));
        drop(write);
        assert!(first.drain(Instant::now() + Duration::from_millis(5)));
        drop(retained);
        assert_eq!(first.counters()["pending"], 0);
    }
}
