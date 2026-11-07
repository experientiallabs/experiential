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
}

pub(super) struct InputGate {
    deadline: Instant,
    state: Mutex<State>,
    changed: Condvar,
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
            || checks.iter().any(|c| {
                !matches!(
                    c.check["action"].as_str(),
                    Some("allow" | "block" | "error")
                ) || !matches!(
                    c.adapter["kind"].as_str(),
                    Some(
                        "allow"
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
        let gate = Arc::new(Self {
            deadline: Instant::now() + Duration::from_secs_f64(remaining),
            state: Mutex::new(State {
                decision: json!({"action": "pending"}),
                task: None,
            }),
            changed: Condvar::new(),
        });
        let weak = Arc::downgrade(&gate);
        let deadline = gate.deadline;
        let handle = super::inspection_http::runtime()?.spawn(async move {
            let result = run(checks, deadline).await;
            if let Some(gate) = weak.upgrade() {
                if let Ok(mut state) = gate.state.lock() {
                    if state.decision["action"] == "pending" {
                        state.decision = if Instant::now() >= deadline {
                            unavailable()
                        } else {
                            result
                        };
                    }
                    state.task = None;
                    gate.changed.notify_all();
                }
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
            state.decision = unavailable();
            if let Some(task) = state.task.take() {
                task.abort();
            }
            self.changed.notify_all();
        }
        state.decision.clone()
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
            if let Some(failure) = failure {
                state.decision = denied(json!({"action": "error", "failure": failure}));
            } else if matches!(state.decision["action"].as_str(), Some("allow" | "pending")) {
                state.decision = unavailable();
            }
            if let Some(task) = state.task.take() {
                task.abort();
            }
            self.changed.notify_all();
        }
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

async fn run(checks: Vec<Check>, deadline: Instant) -> Value {
    for c in checks {
        let started = Instant::now();
        let budget = Duration::from_millis(c.check["timeout_ms"].as_u64().unwrap_or(0))
            .min(deadline.saturating_duration_since(started));
        let result = if budget.is_zero() {
            Err(())
        } else {
            inspect(&c, budget).await
        };
        let result = if started.elapsed() > budget {
            Err(())
        } else {
            result
        };
        match result {
            Ok(v) if v["coverage_error"] == true => {
                let id = if c.adapter["oversized"] == true {
                    None
                } else {
                    c.check["check_id"].as_str()
                };
                return denied(inspection::decision("error", id, "unsupported_capability",
                    "Content inspection does not support this complete request. Use a supported context or modality."));
            }
            Ok(v) if v["flagged"] == true && c.check["action"] != "allow" => {
                return denied(inspection::failed(
                    c.check["action"].as_str().unwrap_or("error"),
                    c.check["check_id"].as_str(),
                ));
            }
            Err(_) => {
                if let Some(value) = inspection::uncertain(&c) {
                    return denied(value);
                }
            }
            _ => (),
        }
    }
    json!({"action": "allow"})
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
            let result = super::inspection_http::inspect_async(
                &c.adapter,
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
        Some("decision_model") => super::input_decisions::inspect(&c.adapter, budget).await,
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
