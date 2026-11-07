//! Admission-only Python access; serving retains the Rust registry directly.

use super::{Key, Registry, Scope, Snapshot, Success};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use serde::Deserialize;
use serde_json::Value;
use std::sync::{Arc, Mutex};

#[pyclass(frozen)]
pub struct NativeSessionRecovery {
    pub(in crate::accounting) inner: Arc<Mutex<Registry>>,
}

fn decode<T: serde::de::DeserializeOwned + 'static>(data: &str) -> PyResult<T> {
    crate::lossless_json::from_str(data)
        .map_err(|_| PyValueError::new_err("Invalid session recovery authority"))
}
fn encoded(value: &impl serde::Serialize) -> PyResult<String> {
    serde_json::to_string(value)
        .map_err(|_| PyValueError::new_err("Invalid session recovery evidence"))
}
#[derive(Deserialize)]
struct Operation {
    key: Key,
    #[serde(default)]
    deployment: String,
    #[serde(default)]
    scope: Option<Scope>,
    #[serde(default)]
    cause: String,
    #[serde(default)]
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    retry: f64,
    #[serde(default)]
    meter: Option<Success>,
    #[serde(default)]
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    observed: Option<f64>,
    #[serde(default)]
    live: bool,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    now: f64,
}
#[derive(Deserialize)]
struct Choose {
    key: Key,
    candidates: Vec<(String, Scope)>,
    snapshot: Option<Snapshot>,
    #[serde(deserialize_with = "crate::lossless_json::deserialize")]
    now: f64,
}

#[pymethods]
impl NativeSessionRecovery {
    #[new]
    #[pyo3(signature=(maximum=65536, max_age=3600.0, observation_lifetime=60.0, cooldown=5.0))]
    fn new(
        maximum: usize,
        max_age: f64,
        observation_lifetime: f64,
        cooldown: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(Mutex::new(
                Registry::new(maximum, max_age, observation_lifetime, cooldown).ok_or_else(
                    || PyValueError::new_err("Recovery bounds must be finite and positive"),
                )?,
            )),
        })
    }
    fn state_json(&self, operation: &str, arguments: &str) -> PyResult<String> {
        let a: Operation = decode(arguments)?;
        let mut state = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Session recovery unavailable"))?;
        match operation {
            "depart" => {
                if !["transport", "credential", "throttle", "local_capacity"]
                    .contains(&a.cause.as_str())
                {
                    return Err(PyValueError::new_err("Invalid recovery cause"));
                }
                state.depart(
                    &a.key,
                    &a.deployment,
                    a.scope
                        .ok_or_else(|| PyValueError::new_err("Missing recovery scope"))?,
                    &a.cause,
                    a.retry,
                    a.observed.unwrap_or(a.now),
                );
            }
            "success" => state.success(
                &a.key,
                &a.deployment,
                a.scope
                    .ok_or_else(|| PyValueError::new_err("Missing recovery scope"))?,
                a.meter
                    .ok_or_else(|| PyValueError::new_err("Missing cache evidence"))?,
                a.observed.unwrap_or(a.now),
                a.now,
            ),
            "retained" => return encoded(&state.retained(&a.key, a.live, a.now)),
            "inspect" => return encoded(&state.inspect(&a.key)),
            _ => return Err(PyValueError::new_err("Unknown session recovery operation")),
        }
        encoded(&Value::Null)
    }
    #[pyo3(signature=(arguments, eligible, capacity=None))]
    fn choose_json(
        &self,
        arguments: &str,
        eligible: Bound<'_, PyAny>,
        capacity: Option<Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        let a: Choose = decode(arguments)?;
        let mut state = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Session recovery unavailable"))?;
        let result = state.choose(
            &a.key,
            &a.candidates,
            |id| eligible.call1((id,))?.extract::<bool>(),
            a.snapshot.as_ref(),
            capacity.map(|callback| move |id: &str| callback.call1((id,))?.extract::<bool>()),
            a.now,
        )?;
        encoded(&result)
    }
}
