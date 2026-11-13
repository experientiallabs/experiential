//! Native request retention, fair recovery and admission fences.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::{Duration, Instant};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{json, Value};

use crate::errors::PublicError;
use crate::native_callbacks::{self, CallbackTarget, NativeCallbacks, NativeTarget};

use super::dispatch::Dispatch;
use super::health::Policy;
use super::ownership::Owner;
use super::route_config::Registration;
use super::shared::Shared;

pub(super) struct Entry {
    pub owner: Arc<Owner>,
    resources: Arc<super::resources::Resources>,
    registry: Weak<Registry>,
}

impl Drop for Entry {
    fn drop(&mut self) {
        if let Some(registry) = self.registry.upgrade() {
            if let Ok(mut r) = registry.requests.try_lock() {
                if r.live
                    .get(&self.owner.request_id)
                    .is_some_and(|entry| std::ptr::eq(entry.as_ptr(), self))
                {
                    r.live.remove(&self.owner.request_id);
                }
            }
        }
    }
}

#[derive(Default)]
struct Requests {
    active: HashMap<String, Arc<Entry>>,
    live: HashMap<String, Weak<Entry>>,
    organizations: HashMap<String, HashSet<String>>,
    rotation: VecDeque<String>,
    recovering: HashSet<String>,
    closing: bool,
    last_pruned: Option<Instant>,
}

pub(super) struct Registry {
    pub shared: Arc<Shared>,
    reports: Arc<super::input_reports::Reports>,
    requests: Mutex<Requests>,
    changed: Condvar,
}

impl Registry {
    fn lookup(&self, id: &str) -> Option<Arc<Entry>> {
        self.requests.lock().ok()?.live.get(id)?.upgrade()
    }

    fn register(&self, entry: Arc<Entry>) -> Result<(), PublicError> {
        let mut r = self.requests.lock().map_err(|_| PublicError::internal())?;
        let id = &entry.owner.request_id;
        if r.closing || r.live.get(id).is_some_and(|old| old.strong_count() > 0) {
            return Err(PublicError::internal());
        }
        r.live.insert(id.clone(), Arc::downgrade(&entry));
        r.organizations
            .entry(entry.owner.organization_id.clone())
            .or_default()
            .insert(id.clone());
        r.rotation.push_back(id.clone());
        r.active.insert(id.clone(), entry);
        self.changed.notify_all();
        Ok(())
    }

    /// Retires an owner that owes no write, and returns one that owes a write
    /// again (a late meter that failed after retirement) to fenced recovery.
    fn reconcile(&self, entry: &Arc<Entry>) {
        let retired = entry.owner.retired();
        let Ok(mut r) = self.requests.lock() else {
            return;
        };
        let id = &entry.owner.request_id;
        let org = &entry.owner.organization_id;
        if retired {
            if r.active.remove(id).is_none() {
                return;
            }
            if let Some(ids) = r.organizations.get_mut(org) {
                ids.remove(id);
                if ids.is_empty() {
                    r.organizations.remove(org);
                }
            }
        } else if !r.active.contains_key(id) {
            r.organizations
                .entry(org.clone())
                .or_default()
                .insert(id.clone());
            r.rotation.push_back(id.clone());
            r.active.insert(id.clone(), entry.clone());
        } else {
            return;
        }
        self.changed.notify_all();
    }

    fn clear(&self, org: Option<&str>) -> bool {
        let Ok(r) = self.requests.lock() else {
            return false;
        };
        if r.closing {
            return false;
        }
        match org {
            Some(org) => r.organizations.get(org).is_none_or(|ids| {
                ids.iter()
                    .all(|id| r.active.get(id).is_none_or(|entry| !entry.owner.fenced()))
            }),
            None => r.active.values().all(|entry| !entry.owner.fenced()),
        }
    }

    fn claim_recovery(&self, visited: &HashSet<String>) -> Option<(Arc<Entry>, bool)> {
        let mut r = self.requests.lock().ok()?;
        let now = Instant::now();
        if r.last_pruned
            .is_none_or(|at| now.duration_since(at) >= Duration::from_secs(60))
        {
            // Admission does no scan. A contended final drop leaves only a weak
            // index entry, which recovery prunes at most once per minute.
            r.live.retain(|_, entry| entry.strong_count() > 0);
            r.last_pruned = Some(now);
        }
        for _ in 0..r.rotation.len() {
            let id = r.rotation.pop_front()?;
            let Some(entry) = r.active.get(&id).cloned() else {
                continue;
            };
            r.rotation.push_back(id.clone());
            if !visited.contains(&id)
                && !r.recovering.contains(&id)
                && (r.closing || entry.owner.needs_recovery(now))
            {
                r.recovering.insert(id);
                return Some((entry, r.closing));
            }
        }
        None
    }

    fn recover_one(&self, visited: &HashSet<String>) -> Option<String> {
        let (entry, closing) = self.claim_recovery(visited)?;
        let retained = entry.owner.fenced();
        if closing {
            let _ = entry
                .owner
                .call("abandon", json!({"request_id": entry.owner.request_id}));
        } else {
            let _ = entry.owner.recover(Instant::now());
        }
        self.reconcile(&entry);
        if entry.owner.retired() {
            super::counters::increment(if retained {
                &self.shared.counters.replayed
            } else {
                &self.shared.counters.abandoned
            });
        }
        if let Ok(mut r) = self.requests.lock() {
            r.recovering.remove(&entry.owner.request_id);
        }
        Some(entry.owner.request_id.clone())
    }

    fn begin_close(&self) {
        if let Ok(mut r) = self.requests.lock() {
            r.closing = true;
            self.changed.notify_all();
        }
    }

    fn drained(&self) -> bool {
        self.requests.lock().is_ok_and(|r| r.active.is_empty())
    }

    fn worker(self: Arc<Self>) {
        loop {
            if self
                .requests
                .lock()
                .is_ok_and(|r| r.closing && r.active.is_empty())
            {
                return;
            }
            self.recover_one(&HashSet::new());
            // A bounded pool services different owners even beside a slow write.
            // Failed owners rotate to the tail; none can monopolize a worker.
            let Ok(guard) = self.requests.lock() else {
                return;
            };
            let _ = self.changed.wait_timeout(guard, Duration::from_millis(250));
        }
    }
}

impl CallbackTarget for Registry {
    fn handles(&self, method: &str) -> bool {
        native_callbacks::post_admission(method)
    }
    fn call(&self, method: &str, argument: &str) -> Result<String, PublicError> {
        let input: Value =
            crate::lossless_json::from_str(argument).map_err(|_| PublicError::internal())?;
        let id = input["request_id"]
            .as_str()
            .ok_or_else(PublicError::internal)?;
        let Some(entry) = self.lookup(id) else {
            return match method {
                "settle" => Ok("{}".into()),
                "abandon" => Ok("{\"terminalized\":true}".into()),
                _ => Err(PublicError::internal()),
            };
        };
        let result = entry.owner.call(method, input);
        self.reconcile(&entry);
        result.map(|result| result.to_string())
    }
    fn requires_request_owner(&self) -> bool {
        true
    }
    fn retain_request(&self, id: &str) -> Option<Arc<dyn Send + Sync>> {
        self.lookup(id).map(|entry| entry as Arc<dyn Send + Sync>)
    }
    fn input_resources(&self, id: &str) -> Option<Arc<super::resources::Resources>> {
        self.lookup(id).map(|entry| entry.resources.clone())
    }
}

/// Admission hands off one validated route and one retained native transaction owner.
#[pyclass]
pub struct NativeAccounting {
    inner: Arc<Registry>,
    background_recovery: bool,
}

#[pymethods]
impl NativeAccounting {
    #[new]
    #[pyo3(signature = (health_policy_json="{}", default_bound=None, activity_window_seconds=10.0, sticky_maximum=65536, background_recovery=true, recovery=None))]
    fn new(
        health_policy_json: &str,
        default_bound: Option<u64>,
        activity_window_seconds: f64,
        sticky_maximum: usize,
        background_recovery: bool,
        recovery: Option<&super::NativeSessionRecovery>,
    ) -> PyResult<Self> {
        let policy: Policy = crate::lossless_json::from_str(health_policy_json)
            .map_err(|_| PyValueError::new_err("Invalid health policy"))?;
        super::estimate::warm().map_err(py_error)?;
        let mut shared = Shared::new(
            policy,
            default_bound,
            activity_window_seconds,
            sticky_maximum,
        )
        .map_err(py_error)?;
        if let Some(recovery) = recovery {
            shared.recovery = recovery.inner.clone();
        }
        let shared = Arc::new(shared);
        let inner = Arc::new(Registry {
            shared,
            reports: Arc::default(),
            requests: Mutex::new(Requests::default()),
            changed: Condvar::new(),
        });
        if background_recovery {
            for index in 0..4 {
                let registry = inner.clone();
                if std::thread::Builder::new()
                    .name(format!("gateway-accounting-{index}"))
                    .spawn(move || registry.worker())
                    .is_err()
                {
                    inner.begin_close();
                    return Err(PyValueError::new_err(
                        "Could not start native accounting recovery",
                    ));
                }
            }
        }
        Ok(Self {
            inner,
            background_recovery,
        })
    }

    fn callbacks(&self) -> NativeCallbacks {
        NativeCallbacks {
            inner: NativeTarget::new(self.inner.clone()),
        }
    }

    /// A request rejected before route registration still has one retained native writer.
    fn finish_admission(
        &self,
        py: Python<'_>,
        organization_id: &str,
        argument: &str,
        ledger: &NativeCallbacks,
    ) -> PyResult<bool> {
        let target = ledger.inner.clone();
        py.detach(|| {
            let input: Value =
                crate::lossless_json::from_str(argument).map_err(|_| PublicError::internal())?;
            let id = input["request_id"]
                .as_str()
                .filter(|id| !id.is_empty() && id.len() <= 512)
                .ok_or_else(PublicError::internal)?;
            if organization_id.is_empty() || organization_id.len() > 512 {
                return Err(PublicError::internal());
            }
            if self.inner.lookup(id).is_none() {
                let entry = Arc::new(Entry {
                    resources: Arc::new(super::resources::Resources::default()),
                    owner: Owner::new(
                        id.into(),
                        organization_id.into(),
                        Instant::now(),
                        Arc::new(super::admission::Admission { target }),
                    ),
                    registry: Arc::downgrade(&self.inner),
                });
                // A duplicate caller may have registered this same request meanwhile.
                if let Err(error) = self.inner.register(entry) {
                    if self.inner.lookup(id).is_none() {
                        return Err(error);
                    }
                }
            }
            let entry = self.inner.lookup(id).ok_or_else(PublicError::internal)?;
            if entry.owner.organization_id != organization_id {
                return Err(PublicError::internal());
            }
            let result = entry.owner.call("abandon", input);
            self.inner.reconcile(&entry);
            result.map(|result| result["terminalized"] == true)
        })
        .map_err(py_error)
    }

    /// Native administrative access uses the same owner as the serving bridge.
    fn call(&self, py: Python<'_>, method: &str, argument: &str) -> PyResult<String> {
        py.detach(|| self.inner.call(method, argument))
            .map_err(py_error)
    }

    #[pyo3(signature = (configuration_json, ledger, *, input=None, observation=None, capture=None, pending_capture=None))]
    #[allow(clippy::too_many_arguments)] // Distinct retained handles cross the initial PyO3 handoff.
    fn register(
        &self,
        py: Python<'_>,
        configuration_json: &str,
        ledger: &NativeCallbacks,
        input: Option<&super::input::NativeInputInspection>,
        observation: Option<&super::input::NativeInputInspection>,
        capture: Option<&crate::capture::python::CaptureCollector>,
        pending_capture: Option<&[u8]>,
    ) -> PyResult<()> {
        let host = ledger.inner.clone();
        let input = input.map(|v| v.inner.clone());
        let observation = observation.map(|v| v.inner.clone());
        let capture = capture.map(|v| v.inner.clone());
        let pending_capture = pending_capture.map(<[u8]>::to_vec);
        py.detach(|| {
            let started = Instant::now();
            let mut config: Registration = crate::lossless_json::from_str(configuration_json)
                .map_err(|_| PublicError::internal())?;
            config.validate()?;
            let deadline = started + Duration::from_secs_f64(config.remaining_seconds);
            config.remaining_seconds = deadline
                .saturating_duration_since(Instant::now())
                .as_secs_f64();
            let id = config.request_id.clone();
            let org = config.organization_id.clone();
            for input in input.iter().chain(observation.iter()) {
                input.bind_reporter(super::input_reports::Reporter::new(
                    host.clone(),
                    id.clone(),
                    self.inner.reports.clone(),
                ));
            }
            let mut resources =
                super::resources::Resources::new(input, capture, pending_capture, &id)?;
            resources.reporting = Some(super::input_reports::Factory {
                target: host.clone(),
                request_id: id.clone(),
                reports: self.inner.reports.clone(),
            });
            let dispatch =
                Dispatch::with_resources(config, self.inner.shared.clone(), host, resources)?;
            self.inner.register(Arc::new(Entry {
                resources: dispatch.resources.clone(),
                owner: Owner::new(id, org, deadline, dispatch),
                registry: Arc::downgrade(&self.inner),
            }))
        })
        .map_err(py_error)
    }

    #[pyo3(signature = (organization_id=None))]
    fn is_clear(&self, organization_id: Option<&str>) -> bool {
        self.inner.clear(organization_id)
    }

    /// Report financial recovery independently of whether admission is closed.
    fn has_failed_writes(&self) -> bool {
        self.inner.requests.lock().map_or(true, |requests| {
            requests.active.values().any(|entry| entry.owner.fenced())
        })
    }

    fn drained(&self) -> bool {
        self.inner.drained()
    }

    fn owns_request(&self, request_id: &str) -> bool {
        self.inner.lookup(request_id).is_some()
    }

    /// A failed admission acknowledgement remains owned by native recovery.
    fn abort_admission(
        &self,
        py: Python<'_>,
        request_id: &str,
        web_search_requests: u64,
    ) -> PyResult<()> {
        py.detach(|| {
            self.inner.call(
                "abandon",
                &json!({"request_id": request_id, "web_search_requests": web_search_requests,
                "failure": {
                    "failure_class": "internal",
                    "safe_message": "gateway could not hand off admitted request accounting"
                }})
                .to_string(),
            )
        })
        .map(|_| ())
        .map_err(py_error)
    }

    fn fenced_organizations(&self) -> Vec<String> {
        let Ok(r) = self.inner.requests.lock() else {
            return Vec::new();
        };
        r.organizations
            .iter()
            .filter(|&(_, ids)| {
                ids.iter()
                    .any(|id| r.active.get(id).is_some_and(|entry| entry.owner.fenced()))
            })
            .map(|(org, _)| org.clone())
            .collect()
    }

    fn inflight(&self) -> usize {
        self.inner.requests.lock().map_or(0, |r| r.active.len())
    }

    fn counters_json(&self) -> String {
        self.inner.shared.counters.snapshot().to_string()
    }

    /// Completed serial inspection uses the same storage lifetime as parallel inspection.
    fn record_input(
        &self,
        inspection: &super::input::NativeInputInspection,
        ledger: &NativeCallbacks,
        request_id: String,
    ) {
        inspection
            .inner
            .bind_reporter(super::input_reports::Reporter::new(
                ledger.inner.clone(),
                request_id,
                self.inner.reports.clone(),
            ));
    }

    fn moderation_recording_json(&self) -> String {
        self.inner.reports.counters().to_string()
    }

    fn drain_moderation_telemetry_json(&self) -> String {
        self.inner.reports.drain_telemetry().to_string()
    }

    fn recover(&self, py: Python<'_>) -> usize {
        py.detach(|| {
            let mut visited = HashSet::new();
            while visited.len() < 16 {
                let Some(id) = self.inner.recover_one(&visited) else {
                    break;
                };
                visited.insert(id);
            }
            visited.len()
        })
    }

    #[pyo3(signature = (timeout_seconds=5.0))]
    fn close(&self, py: Python<'_>, timeout_seconds: f64) -> PyResult<bool> {
        if !timeout_seconds.is_finite() || !(0.0..=60.0).contains(&timeout_seconds) {
            return Err(PyValueError::new_err(
                "Accounting close timeout must be between zero and 60 seconds",
            ));
        }
        self.inner.begin_close();
        Ok(py.detach(|| {
            let deadline = Instant::now() + Duration::from_secs_f64(timeout_seconds);
            loop {
                if self.inner.drained() {
                    return self.inner.reports.drain(deadline);
                }
                if Instant::now() >= deadline {
                    return false;
                }
                if !self.background_recovery {
                    let mut visited = HashSet::new();
                    while visited.len() < 16 && Instant::now() < deadline {
                        let Some(id) = self.inner.recover_one(&visited) else {
                            break;
                        };
                        visited.insert(id);
                    }
                }
                let Ok(r) = self.inner.requests.lock() else {
                    return false;
                };
                if r.active.is_empty() {
                    drop(r);
                    return self.inner.reports.drain(deadline);
                }
                let wait = deadline
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(250));
                if self.inner.changed.wait_timeout(r, wait).is_err() {
                    return false;
                }
            }
        }))
    }

    fn state_json(&self, operation: &str, argument: &str) -> PyResult<String> {
        super::state_query::call(&self.inner.shared, operation, argument).map_err(py_error)
    }
}

impl Drop for NativeAccounting {
    fn drop(&mut self) {
        self.inner.begin_close();
    }
}

fn py_error(error: PublicError) -> PyErr {
    PyValueError::new_err(serde_json::to_string(&error).unwrap_or_default())
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
