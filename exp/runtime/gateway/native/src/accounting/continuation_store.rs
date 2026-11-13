//! Bounded process-local continuation shared by admission and native completion.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pyo3::{exceptions::PyValueError, prelude::*};
use serde_json::{json, Value};

use crate::errors::PublicError;

type Key = (String, String, String, String);

struct Entry {
    value: Value,
    bytes: usize,
    expires: Instant,
    access: u64,
}

#[derive(Default)]
struct Entries {
    values: HashMap<Key, Entry>,
    expires: BTreeMap<(Instant, Key), ()>,
    accesses: BTreeMap<u64, Key>,
    bytes: usize,
    sequence: u64,
}

impl Entries {
    fn remove(&mut self, key: &Key) {
        if let Some(entry) = self.values.remove(key) {
            self.expires.remove(&(entry.expires, key.clone()));
            self.accesses.remove(&entry.access);
            self.bytes -= entry.bytes;
        }
    }
    fn expire(&mut self, now: Instant) {
        while let Some(((expires, key), ())) = self.expires.first_key_value() {
            if *expires > now {
                break;
            }
            let key = key.clone();
            self.remove(&key);
        }
    }
    fn sequence(&mut self) -> Result<u64, PublicError> {
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(PublicError::internal)?;
        Ok(self.sequence)
    }
}

pub(super) struct Store {
    capacity: usize,
    byte_cap: usize,
    ttl: Duration,
    entries: Mutex<Entries>,
}

fn key(namespace: &Value, response: &str) -> Result<Key, PublicError> {
    let text = |name| {
        namespace[name]
            .as_str()
            .filter(|v| !v.is_empty() && v.len() <= 128)
            .map(str::to_owned)
            .ok_or_else(PublicError::internal)
    };
    if response.is_empty() || response.len() > 512 {
        return Err(PublicError::internal());
    }
    Ok((
        text("organization_id")?,
        text("identity_id")?,
        text("alias_revision_id")?,
        response.into(),
    ))
}

impl Store {
    pub fn remember(&self, value: Value) -> Result<(), PublicError> {
        let key = key(
            &value["namespace"],
            value["response_id"]
                .as_str()
                .ok_or_else(PublicError::internal)?,
        )?;
        if !value["state"].is_object() || !value["replay"].is_array() {
            return Err(PublicError::internal());
        }
        let bytes = serde_json::to_vec(&value)
            .map_err(|_| PublicError::internal())?
            .len();
        if bytes > self.byte_cap {
            return Err(PublicError::new(400, "continuation_unavailable",
                "The response is too large for bounded local continuation. Resend the full conversation history.", "invalid_request_error"));
        }
        let now = Instant::now();
        let mut entries = self.entries.lock().map_err(|_| PublicError::internal())?;
        entries.expire(now);
        entries.remove(&key);
        while entries.values.len() >= self.capacity || entries.bytes > self.byte_cap - bytes {
            let oldest = entries
                .accesses
                .first_key_value()
                .ok_or_else(PublicError::internal)?
                .1
                .clone();
            entries.remove(&oldest);
        }
        let access = entries.sequence()?;
        let expires = now + self.ttl;
        entries.bytes += bytes;
        entries.accesses.insert(access, key.clone());
        entries.expires.insert((expires, key.clone()), ());
        entries.values.insert(
            key,
            Entry {
                value,
                bytes,
                expires,
                access,
            },
        );
        Ok(())
    }

    fn resolve(&self, namespace: Value, response: &str) -> Result<Value, PublicError> {
        let key = key(&namespace, response)?;
        let mut entries = self.entries.lock().map_err(|_| PublicError::internal())?;
        entries.expire(Instant::now());
        let access = entries.sequence()?;
        let entry = entries.values.get_mut(&key).ok_or_else(|| PublicError::new(400,
            "previous_response_not_found", "previous_response_id is unavailable or expired in this namespace. Resend the full conversation history.", "invalid_request_error"))?;
        let old = std::mem::replace(&mut entry.access, access);
        let value = entry.value.clone();
        entries.accesses.remove(&old);
        entries.accesses.insert(access, key);
        Ok(value)
    }
}

#[pyclass(frozen)]
pub struct NativeContinuations {
    pub(super) inner: Arc<Store>,
}

#[pymethods]
impl NativeContinuations {
    #[new]
    #[pyo3(signature = (*, capacity=4096, byte_cap=67_108_864, ttl_seconds=86_400.0))]
    fn new(capacity: usize, byte_cap: usize, ttl_seconds: f64) -> PyResult<Self> {
        if capacity == 0
            || byte_cap == 0
            || !ttl_seconds.is_finite()
            || !(0.0..=31_536_000.0).contains(&ttl_seconds)
            || ttl_seconds == 0.0
        {
            return Err(PyValueError::new_err(
                "Continuation bounds must be positive and finite",
            ));
        }
        Ok(Self {
            inner: Arc::new(Store {
                capacity,
                byte_cap,
                ttl: Duration::from_secs_f64(ttl_seconds),
                entries: Mutex::default(),
            }),
        })
    }
    fn remember(&self, py: Python<'_>, argument: &str) -> PyResult<()> {
        py.detach(|| {
            self.inner.remember(
                crate::lossless_json::from_str(argument).map_err(|_| PublicError::internal())?,
            )
        })
        .map_err(py_error)
    }
    fn resolve(&self, py: Python<'_>, namespace: &str, response_id: &str) -> PyResult<String> {
        py.detach(|| {
            self.inner
                .resolve(
                    crate::lossless_json::from_str(namespace)
                        .map_err(|_| PublicError::internal())?,
                    response_id,
                )
                .map(|value| value.to_string())
        })
        .map_err(py_error)
    }
}

fn py_error(error: PublicError) -> PyErr {
    PyValueError::new_err(json!(error).to_string())
}
