//! Read-only input inspection that overlaps dispatch without entering Python.

use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{json, Value};
use tokio::task::AbortHandle;

use super::inspection::{self, Check};

struct State {
    decision: Value,
    task: Option<AbortHandle>,
    reporter: Option<super::input_reports::Reporter>,
    reported: bool,
    report_ready: bool,
}

pub(super) struct InputGate {
    deadline: Instant,
    state: Mutex<State>,
    changed: Condvar,
    checks: Vec<Value>,
    progress: Arc<Mutex<Vec<Value>>>,
    started: Instant,
}

pub(super) fn denied(mut value: Value) -> Value {
    value["action"] = json!("error");
    value["failure"]["safe_details"]["input_guardrail_denied"] = json!(true);
    value
}

pub(super) fn unavailable() -> Value {
    denied(
        json!({"failure": {"failure_class": "unavailable", "safe_message": "Content inspection is unavailable. Retry later.", "safe_details": {}, "retryable_same_deployment": false, "failover_eligible": false}}),
    )
}

impl InputGate {
    pub fn start(checks: Vec<Check>, remaining: f64) -> Result<Arc<Self>, ()> {
        if !remaining.is_finite()
            || !(0.0..=86_400.0).contains(&remaining)
            || checks.len() > 256
            || checks.iter().any(|c| {
                !matches!(
                    c.check["action"].as_str(),
                    Some("allow" | "block" | "error")
                ) || !matches!(
                    c.adapter["kind"].as_str(),
                    Some(
                        "allow"
                            | "unavailable"
                            | "coverage_error"
                            | "keyword"
                            | "regex"
                            | "http_json"
                            | "decision_model"
                    )
                )
            })
        {
            return Err(());
        }
        let observe = !checks.is_empty() && checks.iter().all(|c| c.policy["mode"] == "observe");
        let records = checks
            .iter()
            .map(|c| {
                json!({
                    "policy_id": c.policy["policy_id"], "revision": c.policy["revision"],
                    "check_id": c.check["check_id"], "adapter_id": c.check["adapter_id"],
                    "capability": c.check["capability"], "stage": c.check["stage"],
                    "mode": if observe {"observe"} else {"enforce"},
                    "action": if observe {"allow"} else {"error"}
                })
            })
            .collect();
        let gate = Arc::new(Self {
            deadline: Instant::now() + Duration::from_secs_f64(remaining),
            state: Mutex::new(State {
                decision: json!({"action": "pending"}),
                task: None,
                reporter: None,
                reported: false,
                report_ready: false,
            }),
            changed: Condvar::new(),
            checks: records,
            progress: Arc::default(),
            started: Instant::now(),
        });
        let weak = Arc::downgrade(&gate);
        // Observations outlive the financial owner but remain bounded and deadline-owned.
        let observation = observe.then(|| gate.clone());
        static JOBS: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
        static BYTES: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
        let permits = if observe {
            let bytes = checks
                .iter()
                .fold(0usize, |n, c| n.saturating_add(c.adapter.to_string().len()));
            let job = JOBS
                .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(128)))
                .clone()
                .try_acquire_owned();
            let memory = BYTES
                .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(64 * 1024 * 1024)))
                .clone()
                .try_acquire_many_owned(u32::try_from(bytes.max(1)).unwrap_or(u32::MAX));
            match (job, memory) {
                (Ok(job), Ok(memory)) => Some((job, memory)),
                _ => {
                    let mut state = gate.state.lock().map_err(|_| ())?;
                    state.decision = gate.incomplete("overload");
                    state.report_ready = true;
                    drop(state);
                    return Ok(gate);
                }
            }
        } else {
            None
        };
        let deadline = gate.deadline;
        let progress = gate.progress.clone();
        let handle = super::inspection_http::runtime()?.spawn(async move {
            let _observation = observation;
            let _permits = permits;
            let result = run(checks, deadline, progress).await;
            if let Some(gate) = weak.upgrade() {
                if let Ok(mut state) = gate.state.lock() {
                    if state.decision["action"] == "pending" {
                        state.decision = if Instant::now() >= deadline {
                            gate.incomplete("timeout")
                        } else {
                            result
                        };
                    }
                    state.report_ready = true;
                    state.task = None;
                    gate.changed.notify_all();
                }
                gate.publish();
            }
        });
        let mut state = gate.state.lock().map_err(|_| ())?;
        if state.decision["action"] == "pending" {
            state.task = Some(handle.abort_handle());
        }
        drop(state);
        Ok(gate)
    }

    pub fn decision(&self) -> Value {
        let Ok(mut state) = self.state.lock() else {
            return unavailable();
        };
        if state.decision["action"] == "pending" && Instant::now() >= self.deadline {
            state.decision = self.incomplete("timeout");
            state.report_ready = true;
            if let Some(task) = state.task.take() {
                task.abort();
            }
            self.changed.notify_all();
        }
        let decision = state.decision.clone();
        drop(state);
        self.publish();
        decision
    }

    pub fn bind_reporter(&self, reporter: Option<super::input_reports::Reporter>) {
        if let Ok(mut state) = self.state.lock() {
            state.reporter = reporter;
        }
        self.publish();
    }

    fn publish(&self) {
        let report = self.state.lock().ok().and_then(|mut state| {
            if state.reported
                || !state.report_ready
                || state.decision["action"] == "pending"
                || state.decision["moderation"].is_null()
            {
                return None;
            }
            let reporter = state.reporter.clone()?;
            state.reported = true;
            Some((reporter, state.decision.clone()))
        });
        if let Some((reporter, decision)) = report {
            reporter.submit(&decision);
        }
    }

    fn incomplete(&self, reason: &str) -> Value {
        let progress = self.progress.lock().map(|v| v.clone()).unwrap_or_default();
        let observations = self
            .checks
            .iter()
            .enumerate()
            .map(|(index, check)| {
                if let Some(record) = progress.get(index) {
                    return record.clone();
                }
                let mut record = check.clone();
                record["outcome"] = json!(reason);
                record["latency_ms"] = json!(self.started.elapsed().as_secs_f64() * 1000.0);
                record
            })
            .collect();
        interrupted(observations, reason)
    }

    pub fn wait(&self) -> Value {
        let Ok(mut state) = self.state.lock() else {
            return unavailable();
        };
        while state.decision["action"] == "pending" {
            let remaining = self.deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let Ok((next, _)) = self.changed.wait_timeout(state, remaining) else {
                return unavailable();
            };
            state = next;
        }
        drop(state);
        self.decision()
    }

    pub fn cancel(&self, failure: Option<Value>) {
        if let Ok(mut state) = self.state.lock() {
            let unfinished = state.decision["action"] == "pending";
            if unfinished {
                state.decision = self.incomplete("cancelled");
            }
            if let Some(failure) = failure {
                state.decision["failure"] = denied(json!({"failure": failure}))["failure"].clone();
                state.decision["action"] = json!("error");
            }
            if unfinished {
                state.report_ready = true;
                if let Some(task) = state.task.take() {
                    task.abort();
                }
            }
            self.changed.notify_all();
        }
        self.publish();
    }
}
impl Drop for InputGate {
    fn drop(&mut self) {
        if let Ok(state) = self.state.get_mut() {
            if let Some(task) = state.task.take() {
                task.abort();
            }
        }
    }
}

fn interrupted(observations: Vec<Value>, reason: &str) -> Value {
    let fail_open = observations
        .iter()
        .all(|record| record["mode"] == "observe" || record["action"] == "allow");
    let completed_denial = observations.iter().find(|record| {
        record["mode"] != "observe" && record["outcome"] == "flagged" && record["action"] != "allow"
    });
    let (mut decision, status) = if let Some(record) = completed_denial {
        (
            denied(inspection::failed(
                record["action"].as_str().unwrap_or("block"),
                record["check_id"].as_str(),
            )),
            "denied",
        )
    } else if fail_open {
        (json!({"action": "allow"}), "failed_open")
    } else {
        (unavailable(), "unavailable")
    };
    decision["moderation"] =
        json!({"status": status, "reason": reason, "observations": observations});
    decision
}

async fn run(checks: Vec<Check>, deadline: Instant, progress: Arc<Mutex<Vec<Value>>>) -> Value {
    if checks.is_empty() {
        return json!({"action": "allow"});
    }
    let mut records = Vec::with_capacity(checks.len());
    let mut missed = false;
    let mut observe_only = true;
    for c in checks {
        let observe = c.policy["mode"] == "observe";
        observe_only &= observe;
        let started = Instant::now();
        let budget = Duration::from_millis(c.check["timeout_ms"].as_u64().unwrap_or(0))
            .min(deadline.saturating_duration_since(started));
        let result = tokio::time::timeout(budget, inspect(&c, budget)).await;
        let incomplete = if result.is_err() {
            "timeout"
        } else {
            "unavailable"
        };
        let result = result.unwrap_or(Err(()));
        let outcome = match &result {
            Ok(v) if v["coverage_error"] == true => "unsupported",
            Ok(v) if v["flagged"] == true => "flagged",
            Ok(_) => "allow",
            Err(_) => incomplete,
        };
        let verdict = result.as_ref().ok();
        let action = if outcome == "flagged" {
            c.check["action"].as_str().unwrap_or("block")
        } else if !observe && (outcome == "unsupported" || result.is_err()) {
            "error"
        } else {
            "allow"
        };
        records.push(
            json!({"policy_id": c.policy["policy_id"], "revision": c.policy["revision"],
            "check_id": c.check["check_id"], "adapter_id": c.check["adapter_id"],
            "capability": c.check["capability"], "stage": c.check["stage"], "action": action,
            "mode": if observe {"observe"} else {"enforce"},
            "would_action": c.check["action"], "classifier_kind": c.adapter["kind"],
            "outcome": outcome, "latency_ms": started.elapsed().as_secs_f64() * 1000.0,
            "scores": verdict.map(|v| &v["scores"]), "cached": verdict.map(|v| &v["cached"]),
            "subject_sha256": verdict.map(|v| &v["subject_sha256"])}),
        );
        if let Ok(mut progress) = progress.lock() {
            *progress = records.clone();
        }
        let failure = match result {
            Ok(v) if v["coverage_error"] == true => {
                missed = true;
                Some(inspection::decision(
                    "error",
                    if c.adapter["oversized"] == true { None } else { c.check["check_id"].as_str() },
                    "unsupported_capability",
                    "Content inspection does not support this complete request. Use a supported context or modality.",
                ))
            }
            Ok(v) if v["flagged"] == true && c.check["action"] != "allow" => {
                if !observe {
                    let mut decision = denied(inspection::failed(
                        c.check["action"].as_str().unwrap_or("block"),
                        c.check["check_id"].as_str(),
                    ));
                    decision["moderation"] = json!({"status": "denied", "observations": records});
                    return decision;
                }
                None
            }
            Err(_) => {
                missed = true;
                inspection::uncertain(&c)
            }
            _ => None,
        };
        if !observe {
            if let Some(failure) = failure {
                let mut decision = denied(failure);
                decision["moderation"] = json!({"status": "unavailable", "observations": records});
                return decision;
            }
        }
    }
    let reason = records
        .iter()
        .find_map(|record| match record["outcome"].as_str() {
            Some("timeout" | "unavailable" | "unsupported" | "response_ready_timeout") => {
                Some(record["outcome"].clone())
            }
            _ => None,
        });
    let mut decision = json!({"action": "allow", "moderation": {"status": if missed {"failed_open"} else if observe_only {"observed"} else {"allowed"}, "observations": records}});
    if let Some(reason) = reason {
        decision["moderation"]["reason"] = reason;
    }
    decision
}

pub(super) async fn inspect(c: &Check, budget: Duration) -> Result<Value, ()> {
    match c.adapter["kind"].as_str() {
        Some("allow") => Ok(json!({"flagged": false})),
        Some("coverage_error") => Ok(json!({"coverage_error": true})),
        Some("keyword" | "regex") => {
            // Waiting inspections retain no blocking worker. A timed-out job
            // holds its permit until its CPU work ends, bounding orphaned work.
            static CPU: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
            let c = c.clone();
            let deadline = Instant::now() + budget;
            tokio::time::timeout(budget, async {
                let permit = CPU
                    .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(32)))
                    .clone()
                    .acquire_owned()
                    .await
                    .map_err(|_| ())?;
                tokio::task::spawn_blocking(move || {
                    let _permit = permit;
                    if Instant::now() >= deadline {
                        return Err(());
                    }
                    text_check(&c)
                })
                .await
                .map_err(|_| ())?
            })
            .await
            .map_err(|_| ())?
        }
        Some("http_json") => {
            let mut adapter = c.adapter.clone();
            adapter["fail_fast"] = json!(c.policy["mode"] == "observe");
            let result = super::inspection_http::inspect_async(
                &adapter,
                c.adapter["payload"].clone(),
                budget,
            )
            .await?;
            if c.adapter["input_template"] == true {
                super::input_rewrite::verdict(result, c.check["action"].as_str().ok_or(())?)
            } else {
                inspection::validate_verdict(result, c.check["action"].as_str().ok_or(())?)
            }
        }
        Some("decision_model") => {
            let mut adapter = c.adapter.clone();
            adapter["fail_fast"] = json!(c.policy["mode"] == "observe");
            super::input_decisions::inspect(&adapter, budget).await
        }
        _ => Err(()),
    }
}

fn text_check(c: &Check) -> Result<Value, ()> {
    if c.adapter["redaction_messages"].is_array() {
        return super::input_rewrite::regex(&c.adapter);
    }
    let subjects = c.adapter["subjects"]
        .as_array()
        .ok_or(())?
        .iter()
        .map(|v| v.as_str().ok_or(()))
        .collect::<Result<Vec<_>, _>>()?;
    let flagged = if c.adapter["kind"] == "keyword" {
        let text = inspection::fold(&subjects.join("\n"));
        c.adapter["needles"]
            .as_array()
            .ok_or(())?
            .iter()
            .any(|n| n.as_str().is_some_and(|n| text.contains(n)))
    } else {
        let detector = inspection::cached_detector(&c.adapter["specification"]).ok_or(())?;
        let mut flagged = false;
        for text in subjects {
            flagged |= detector.matches(text).map_err(|_| ())?;
        }
        flagged
    };
    Ok(json!({"flagged": flagged}))
}

#[pyclass(module = "exp_gateway_native")]
pub(crate) struct NativeInputInspection {
    pub(super) inner: Arc<InputGate>,
}

#[pymethods]
impl NativeInputInspection {
    #[new]
    fn new(py: Python<'_>, checks_json: &str, remaining_seconds: f64) -> PyResult<Self> {
        py.detach(|| {
            let checks = crate::lossless_json::from_str(checks_json).map_err(|_| ())?;
            InputGate::start(checks, remaining_seconds).map(|inner| Self { inner })
        })
        .map_err(|_| PyValueError::new_err("Invalid native input inspection contract"))
    }
    fn decision_json(&self) -> String {
        self.inner.decision().to_string()
    }
    fn wait_json(&self, py: Python<'_>) -> String {
        py.detach(|| self.inner.wait().to_string())
    }
    #[pyo3(signature = (failure_json=None))]
    fn cancel(&self, failure_json: Option<&str>) -> PyResult<()> {
        let failure = failure_json
            .map(crate::lossless_json::from_str)
            .transpose()
            .map_err(|_| PyValueError::new_err("Invalid inspection failure"))?;
        self.inner.cancel(failure);
        Ok(())
    }
}
