//! Read-only input inspection that overlaps dispatch without entering Python.

use std::future::Future;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{json, Value};
use tokio::task::AbortHandle;

use super::inspection::{self, Check};

struct State {
    decision: Value,
    strict_decision: Value,
    task: Option<AbortHandle>,
    reporter: Option<super::input_reports::Reporter>,
    reported: bool,
    report_ready: bool,
}

pub(super) struct InputGate {
    deadline: Instant,
    state: Mutex<State>,
    changed: Condvar,
    notified: tokio::sync::Notify,
    observe: bool,
    #[cfg(test)]
    grace: Duration,
    checks: Vec<(Value, bool)>,
    progress: Arc<Mutex<Vec<Value>>>,
    observed_progress: Arc<Mutex<Vec<Value>>>,
    strict_progress: Arc<Mutex<Vec<Value>>>,
    strict_count: usize,
    enforce_count: usize,
    response_ready: tokio::sync::watch::Sender<Option<Instant>>,
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
    pub fn start(mut checks: Vec<Check>, remaining: f64) -> Result<Arc<Self>, ()> {
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
        // Strict, fail-open, and observe policies progress independently.
        let group = |c: &Check| {
            if c.policy["mode"] == "observe" {
                2
            } else if c.policy["input_fail_open"] == true {
                1
            } else {
                0
            }
        };
        checks.sort_by_key(group);
        let strict_count = checks.iter().take_while(|c| group(c) == 0).count();
        let enforce_count = checks.iter().take_while(|c| group(c) != 2).count();
        let observe = !checks.is_empty() && checks.iter().all(|c| c.policy["mode"] == "observe");
        #[cfg(test)]
        let grace = Duration::from_millis(
            checks
                .iter()
                .filter(|c| c.policy["mode"] != "observe")
                .filter_map(|c| c.policy["input_max_wait_ms"].as_u64())
                .max()
                .unwrap_or(0)
                .min(5000),
        );
        let gate = Arc::new(Self {
            deadline: Instant::now() + Duration::from_secs_f64(remaining),
            state: Mutex::new(State {
                decision: json!({"action": "pending"}),
                strict_decision: json!({"action": if strict_count == 0 {"allow"} else {"pending"}}),
                task: None,
                reporter: None,
                reported: false,
                report_ready: false,
            }),
            changed: Condvar::new(),
            notified: tokio::sync::Notify::new(),
            observe,
            #[cfg(test)]
            grace,
            progress: Arc::default(),
            observed_progress: Arc::default(),
            strict_progress: Arc::default(),
            strict_count,
            enforce_count,
            response_ready: tokio::sync::watch::channel(None).0,
            started: Instant::now(),
            checks: checks
                .iter()
                .map(|c| {
                    (
                        json!({"policy_id": c.policy["policy_id"],
                "revision": c.policy["revision"], "check_id": c.check["check_id"],
                "adapter_id": c.check["adapter_id"], "capability": c.check["capability"],
                "stage": c.check["stage"], "action": "allow",
                "mode": if c.policy["mode"] == "observe" {"observe"} else {"enforce"}}),
                        c.policy["mode"] == "observe"
                            || c.policy["input_fail_open"] == true
                            || c.policy["protected"] == false,
                    )
                })
                .collect(),
        });
        let weak = Arc::downgrade(&gate);
        let observation = (enforce_count < checks.len()).then(|| gate.clone());
        let deadline = gate.deadline;
        static JOBS: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
        static BYTES: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
        let weight = checks.iter().fold(0usize, |size, c| {
            size.saturating_add(c.adapter.to_string().len())
        });
        let units = (weight / 1024 + 1).min(u32::MAX as usize) as u32;
        let job = JOBS
            .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(256)))
            .clone()
            .try_acquire_owned();
        let bytes = BYTES
            .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(64 * 1024)))
            .clone()
            .try_acquire_many_owned(units);
        let (Ok(job), Ok(bytes), Ok(runtime)) = (job, bytes, super::inspection_http::runtime())
        else {
            let mut state = gate.state.lock().map_err(|_| ())?;
            state.decision = gate.incomplete("capacity_or_unavailable");
            state.report_ready = true;
            drop(state);
            return Ok(gate);
        };
        let progress = gate.progress.clone();
        let observed_progress = gate.observed_progress.clone();
        let ready = gate.response_ready.subscribe();
        let observed_checks = checks.split_off(enforce_count);
        let permissive_checks = checks.split_off(strict_count);
        let strict_progress = gate.strict_progress.clone();
        let handle = runtime.spawn(async move {
            let _capacity = (job, bytes);
            let serving = async {
                let strict = async {
                    let result = run(checks, deadline, strict_progress, ready.clone()).await;
                    if let Some(gate) = weak.upgrade() {
                        if let Ok(mut state) = gate.state.lock() {
                            state.strict_decision = result.clone();
                            gate.changed.notify_all();
                            gate.notified.notify_waiters();
                        }
                    }
                    result
                };
                let permissive = run(permissive_checks, deadline, progress, ready.clone());
                tokio::pin!(strict, permissive);
                let (first, strict_first) = tokio::select! {
                    value = &mut strict => (value, true),
                    value = &mut permissive => (value, false),
                };
                let decision = if first["action"] == "error" {
                    first
                } else {
                    let second = if strict_first {
                        permissive.await
                    } else {
                        strict.await
                    };
                    combine(first, second)
                };
                if enforce_count > 0 {
                    if let Some(gate) = weak.upgrade() {
                        if let Ok(mut state) = gate.state.lock() {
                            if state.decision["action"] == "pending" {
                                state.decision = decision.clone();
                                if state.decision["action"] == "error" {
                                    let reason = if Instant::now() >= deadline {
                                        "timeout"
                                    } else {
                                        "cancelled"
                                    };
                                    let mut records = gate.incomplete(reason)["moderation"]
                                        ["observations"]
                                        .as_array()
                                        .cloned()
                                        .unwrap_or_default();
                                    records.truncate(enforce_count);
                                    state.decision["moderation"]["observations"] = json!(records);
                                }
                                // A serial caller can drop its last gate as soon as
                                // this decision wakes it. With no observers left,
                                // make the report publishable in the same transition.
                                state.report_ready = enforce_count == gate.checks.len();
                                gate.changed.notify_all();
                                gate.notified.notify_waiters();
                            }
                        }
                        gate.publish();
                    }
                }
                decision
            };
            let (enforced, observed) = tokio::join!(
                serving,
                run(observed_checks, deadline, observed_progress, ready.clone())
            );
            if let Some(gate) = observation.or_else(|| weak.upgrade()) {
                if let Ok(mut state) = gate.state.lock() {
                    if state.decision["action"] == "pending" {
                        state.decision = if Instant::now() >= deadline {
                            gate.incomplete("timeout")
                        } else if enforce_count == 0 {
                            observed.clone()
                        } else {
                            enforced
                        };
                    }
                    if enforce_count > 0 {
                        if let Some(records) = observed["moderation"]["observations"].as_array() {
                            if let Some(existing) =
                                state.decision["moderation"]["observations"].as_array_mut()
                            {
                                existing.extend(records.iter().cloned());
                            }
                        }
                    }
                    state.report_ready = true;
                    state.task = None;
                    gate.changed.notify_all();
                    gate.notified.notify_waiters();
                }
                gate.publish();
            }
        });
        let mut state = gate.state.lock().map_err(|_| ())?;
        if !state.report_ready {
            state.task = Some(handle.abort_handle());
        }
        drop(state);
        Ok(gate)
    }

    pub fn decision(&self) -> Value {
        let Ok(mut state) = self.state.lock() else {
            return self.incomplete("unavailable");
        };
        if state.decision["action"] == "pending" && Instant::now() >= self.deadline {
            state.decision = self.incomplete("timeout");
            state.report_ready = true;
            if let Some(task) = state.task.take() {
                task.abort();
            }
            self.changed.notify_all();
            self.notified.notify_waiters();
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
        let progress = self.progress.lock().map(|p| p.clone()).unwrap_or_default();
        let strict_progress = self
            .strict_progress
            .lock()
            .map(|p| p.clone())
            .unwrap_or_default();
        let observed_progress = self
            .observed_progress
            .lock()
            .map(|p| p.clone())
            .unwrap_or_default();
        let observations: Vec<_> = self
            .checks
            .iter()
            .enumerate()
            .map(|(index, (check, fail_open))| {
                let completed = if index < self.strict_count {
                    strict_progress.get(index)
                } else if index < self.enforce_count {
                    progress.get(index - self.strict_count)
                } else {
                    observed_progress.get(index - self.enforce_count)
                };
                if let Some(record) = completed {
                    return record.clone();
                }
                let mut record = check.clone();
                record["outcome"] = json!(reason);
                record["action"] = json!(if *fail_open { "allow" } else { "error" });
                record["latency_ms"] = json!(self.started.elapsed().as_secs_f64() * 1000.0);
                record
            })
            .collect();
        interrupted(observations, reason)
    }

    pub fn observe(&self) -> bool {
        self.observe
    }

    #[cfg(test)]
    pub fn grace(&self) -> Option<Duration> {
        (self.strict_count == 0).then_some(self.grace)
    }

    /// Wake on completion or cancellation, without occupying a bridge worker.
    pub async fn wait_async(&self) -> Value {
        loop {
            let notified = self.notified.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let decision = self.decision();
            if decision["action"] != "pending" {
                return decision;
            }
            let _ = tokio::time::timeout_at(self.deadline.into(), notified).await;
        }
    }

    /// Search side effects wait only for independently running strict policies.
    pub async fn wait_strict(&self) -> Value {
        loop {
            let notified = self.notified.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let overall = self.decision();
            if overall["action"] == "error" {
                return overall;
            }
            let strict = self
                .state
                .lock()
                .map(|s| s.strict_decision.clone())
                .unwrap_or_else(|_| unavailable());
            if strict["action"] != "pending" {
                return strict;
            }
            if overall["action"] != "pending" {
                return overall;
            }
            let _ = tokio::time::timeout_at(self.deadline.into(), notified).await;
        }
    }

    /// Publish the outward boundary once; each policy owns its own grace.
    pub fn begin_response(&self, at: Instant) {
        self.response_ready.send_if_modified(|value| {
            if value.is_some() {
                false
            } else {
                *value = Some(at);
                true
            }
        });
    }

    /// Commit the release decision before any outward provider byte is sent.
    #[cfg(test)]
    pub fn release_incomplete(&self, reason: &str) {
        if let Ok(mut state) = self.state.lock() {
            if state.decision["action"] == "pending" && self.grace().is_some() {
                state.decision = self.incomplete(reason);
                state.report_ready = true;
                if let Some(task) = state.task.take() {
                    task.abort();
                }
                self.changed.notify_all();
                self.notified.notify_waiters();
            }
        }
        self.publish();
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
            let unfinished = state.decision["action"] == "pending" || failure.is_some();
            if let Some(failure) = failure {
                state.decision = denied(json!({"action": "error", "failure": failure}));
            } else if state.decision["action"] == "pending" {
                state.decision = self.incomplete("cancelled");
            }
            if unfinished {
                state.report_ready = true;
                if let Some(task) = state.task.take() {
                    task.abort();
                }
            }
            self.changed.notify_all();
            self.notified.notify_waiters();
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

fn combine(first: Value, second: Value) -> Value {
    let mut records = first["moderation"]["observations"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    records.extend(
        second["moderation"]["observations"]
            .as_array()
            .into_iter()
            .flatten()
            .cloned(),
    );
    let mut decision = if second["action"] == "error"
        || first["moderation"].is_null()
        || second["moderation"]["status"] == "failed_open"
    {
        second
    } else {
        first
    };
    if !records.is_empty() {
        decision["moderation"]["observations"] = json!(records);
    }
    decision
}

async fn run(
    checks: Vec<Check>,
    deadline: Instant,
    progress: Arc<Mutex<Vec<Value>>>,
    mut ready: tokio::sync::watch::Receiver<Option<Instant>>,
) -> Value {
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
        let response_expired =
            response_deadline(&c, *ready.borrow()).is_some_and(|at| at <= started);
        let (result, incomplete) = if !budget.is_zero() && response_expired {
            (Err(()), "response_ready_timeout")
        } else {
            inspect_until_ready(&c, budget, &mut ready, inspect(&c, budget)).await
        };
        let outcome = match &result {
            Ok(v) if v["coverage_error"] == true => "unsupported",
            Ok(v) if v["flagged"] == true => "flagged",
            Ok(_) => "allow",
            Err(_) => incomplete,
        };
        let verdict = result.as_ref().ok();
        let action = if outcome == "flagged" {
            c.check["action"].as_str().unwrap_or("block")
        } else if !observe
            && c.policy["input_fail_open"] != true
            && (outcome == "unsupported" || result.is_err())
        {
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
        if !observe && c.policy["input_fail_open"] != true {
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

fn response_deadline(check: &Check, ready: Option<Instant>) -> Option<Instant> {
    if check.policy["mode"] == "observe" || check.policy["input_fail_open"] != true {
        return None;
    }
    ready.map(|at| {
        at + Duration::from_millis(
            check.policy["input_max_wait_ms"]
                .as_u64()
                .unwrap_or(0)
                .min(5000),
        )
    })
}

/// Watch a one-shot response boundary while retaining the same in-flight classifier.
async fn inspect_until_ready(
    check: &Check,
    budget: Duration,
    ready: &mut tokio::sync::watch::Receiver<Option<Instant>>,
    inspection: impl Future<Output = Result<Value, ()>>,
) -> (Result<Value, ()>, &'static str) {
    if budget.is_zero() {
        return (Err(()), "timeout");
    }
    let deadline = Instant::now() + budget;
    tokio::pin!(inspection);
    loop {
        let release = response_deadline(check, *ready.borrow_and_update());
        let until = release.map_or(deadline, |at| at.min(deadline));
        let reason = if release.is_some_and(|at| at <= deadline) {
            "response_ready_timeout"
        } else {
            "timeout"
        };
        tokio::select! {
            biased;
            // A completed verdict wins the boundary without extending its wait.
            result = &mut inspection => {
                let incomplete = if Instant::now() >= until { reason } else { "unavailable" };
                return (result, incomplete);
            },
            changed = ready.changed() => {
                if changed.is_err() { return (Err(()), "cancelled"); }
            },
            _ = async {}, if Instant::now() >= until => return (Err(()), reason),
            _ = tokio::time::sleep_until(until.into()) => return (Err(()), reason),
        }
    }
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
            adapter["fail_fast"] =
                json!(c.policy["input_fail_open"] == true || c.policy["mode"] == "observe");
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
            adapter["fail_fast"] =
                json!(c.policy["input_fail_open"] == true || c.policy["mode"] == "observe");
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

#[cfg(test)]
mod interruption_tests {
    use super::*;

    #[test]
    fn completed_denials_survive_cancellation_and_deadline_recovery() {
        for reason in ["cancelled", "timeout"] {
            for action in ["block", "error"] {
                let records = vec![
                    json!({"mode":"enforce", "outcome":"flagged", "action":action, "check_id":"denied-check"}),
                    json!({"mode":"enforce", "outcome":reason, "action":"error"}),
                ];
                let decision = interrupted(records.clone(), reason);
                assert_eq!(decision["failure"]["failure_class"], "guardrail");
                assert_eq!(decision["failure"]["safe_details"]["action"], action);
                assert_eq!(
                    decision["failure"]["safe_details"]["check_id"],
                    "denied-check"
                );
                assert_eq!(decision["moderation"]["status"], "denied");
                assert_eq!(decision["moderation"]["observations"], json!(records));
            }
        }
    }
}

#[cfg(test)]
mod readiness_tests {
    use super::*;
    use futures_util::FutureExt;

    #[tokio::test]
    async fn unstarted_inspections_do_not_use_an_expired_response_budget() {
        for grace_ms in [0, 20] {
            let check = Check {
                policy: json!({"input_fail_open": true, "input_max_wait_ms": grace_ms}),
                check: json!({"action": "block", "timeout_ms": 1000}),
                adapter: json!({"kind": "allow"}),
            };
            let (_sender, ready) = tokio::sync::watch::channel(Some(
                Instant::now() - Duration::from_millis(grace_ms + 1),
            ));
            let decision = run(
                vec![check],
                Instant::now() + Duration::from_secs(1),
                Arc::default(),
                ready,
            )
            .await;
            assert_eq!(decision["action"], "allow");
            assert_eq!(decision["moderation"]["status"], "failed_open");
            assert_eq!(
                decision["moderation"]["observations"][0]["outcome"],
                "response_ready_timeout"
            );
        }
    }

    #[tokio::test]
    async fn ready_verdict_wins_expired_grace_without_waiting_for_pending_work() {
        for grace_ms in [0, 20] {
            let check = Check {
                policy: json!({"input_fail_open": true, "input_max_wait_ms": grace_ms}),
                check: json!({}),
                adapter: json!({}),
            };
            let (_sender, mut ready) = tokio::sync::watch::channel(Some(
                Instant::now() - Duration::from_millis(grace_ms + 1),
            ));
            for flagged in [false, true] {
                let (verdict, _) = inspect_until_ready(
                    &check,
                    Duration::from_secs(1),
                    &mut ready,
                    std::future::ready(Ok(json!({"flagged": flagged}))),
                )
                .await;
                assert_eq!(verdict.unwrap()["flagged"], flagged);
            }
            let (verdict, reason) = inspect_until_ready(
                &check,
                Duration::from_secs(1),
                &mut ready,
                std::future::pending(),
            )
            .now_or_never()
            .expect("expired grace must finish on the first poll of a pending classifier");
            assert!(verdict.is_err());
            assert_eq!(reason, "response_ready_timeout");
        }
    }
}
