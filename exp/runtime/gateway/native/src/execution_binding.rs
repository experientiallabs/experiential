//! Explicit host-selected callback targets and execution resources.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tokio::sync::Semaphore;

use crate::errors::PublicError;

/// Resources shared only by requests the host explicitly assigns to this handle.
pub(crate) struct ExecutionResources {
    pub(crate) target: Py<PyAny>,
    pub(crate) active: Arc<Semaphore>,
    pub(crate) serving: Arc<Semaphore>,
    pub(crate) settlement: Arc<Semaphore>,
    owners: AtomicUsize,
}

/// A callback target with explicit capacity, independent of caller identity.
#[pyclass]
pub struct ExecutionHandle {
    pub(crate) inner: Arc<ExecutionResources>,
}

#[pymethods]
impl ExecutionHandle {
    #[new]
    #[pyo3(signature = (control_plane, *, max_active_requests, callback_permits, settlement_permits))]
    pub(crate) fn new(
        control_plane: Py<PyAny>,
        max_active_requests: usize,
        callback_permits: usize,
        settlement_permits: usize,
    ) -> PyResult<Self> {
        if !(1..=65_536).contains(&max_active_requests)
            || !(1..=256).contains(&callback_permits)
            || !(1..=256).contains(&settlement_permits)
        {
            return Err(PyValueError::new_err(
                "execution limits must be positive: active at most 65536, callbacks at most 256",
            ));
        }
        Ok(Self {
            inner: Arc::new(ExecutionResources {
                target: control_plane,
                active: Arc::new(Semaphore::new(max_active_requests)),
                serving: Arc::new(Semaphore::new(callback_permits)),
                settlement: Arc::new(Semaphore::new(settlement_permits)),
                owners: AtomicUsize::new(0),
            }),
        })
    }

    /// True while a registration, request, callback or detached cleanup owns this handle.
    #[getter]
    pub(crate) fn in_use(&self) -> bool {
        self.inner.owners.load(Ordering::Acquire) != 0
    }
}

/// Shared by every part of one request, including work that outlives its front.
pub(crate) struct ExecutionLease {
    pub(crate) resources: Arc<ExecutionResources>,
}

impl ExecutionLease {
    fn new(resources: Arc<ExecutionResources>) -> Arc<Self> {
        resources.owners.fetch_add(1, Ordering::AcqRel);
        Arc::new(Self { resources })
    }
}

impl Drop for ExecutionLease {
    fn drop(&mut self) {
        self.resources.owners.fetch_sub(1, Ordering::AcqRel);
    }
}

/// One-use, in-process execution bindings installed before a host relays a request.
#[pyclass]
#[derive(Clone)]
pub struct ExecutionBindings {
    entries: Arc<Mutex<HashMap<String, Arc<ExecutionLease>>>>,
    capacity: usize,
}

#[pymethods]
impl ExecutionBindings {
    #[new]
    #[pyo3(signature = (capacity=4096))]
    pub(crate) fn new(capacity: usize) -> PyResult<Self> {
        if !(1..=65_536).contains(&capacity) {
            return Err(PyValueError::new_err(
                "execution binding capacity must be 1 through 65536",
            ));
        }
        Ok(Self {
            entries: Arc::new(Mutex::new(HashMap::new())),
            capacity,
        })
    }

    /// The host chooses the target and budget; no identity is derived here.
    pub(crate) fn register(&self, token: String, handle: &ExecutionHandle) -> PyResult<()> {
        if !(32..=512).contains(&token.len()) {
            return Err(PyValueError::new_err("invalid execution relay token"));
        }
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| PyRuntimeError::new_err("execution bindings unavailable"))?;
        if entries.contains_key(&token) {
            return Err(PyValueError::new_err(
                "execution relay token is already registered",
            ));
        }
        if entries.len() >= self.capacity {
            return Err(PyRuntimeError::new_err(
                "execution binding capacity exhausted",
            ));
        }
        entries.insert(token, ExecutionLease::new(handle.inner.clone()));
        Ok(())
    }

    /// Revoke an unclaimed token without invalidating work that already owns it.
    pub(crate) fn remove(&self, token: &str) -> PyResult<()> {
        self.entries
            .lock()
            .map_err(|_| PyRuntimeError::new_err("execution bindings unavailable"))?
            .remove(token);
        Ok(())
    }
}

impl ExecutionBindings {
    /// Transfer a registration's ownership to exactly one native request.
    pub(crate) fn claim(&self, token: &str) -> Result<Arc<ExecutionLease>, PublicError> {
        self.entries
            .lock()
            .map_err(|_| PublicError::internal())?
            .remove(token)
            .ok_or_else(PublicError::invalid_key)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn handle() -> ExecutionHandle {
        Python::initialize();
        ExecutionHandle::new(Python::attach(|py| py.None()), 128, 32, 8).unwrap()
    }

    pub(crate) fn lease(active: usize, serving: usize, cleanup: usize) -> Arc<ExecutionLease> {
        Python::initialize();
        let handle =
            ExecutionHandle::new(Python::attach(|py| py.None()), active, serving, cleanup).unwrap();
        ExecutionLease::new(handle.inner)
    }

    #[test]
    fn claim_is_once_and_front_removal_cannot_revoke_existing_owners() {
        let bindings = ExecutionBindings::new(2).unwrap();
        let handle = handle();
        let token = "x".repeat(32);
        bindings.register(token.clone(), &handle).unwrap();
        assert!(handle.in_use());
        let request = bindings.claim(&token).unwrap();
        let callback = request.clone();
        assert!(bindings.claim(&token).is_err());
        bindings.remove(&token).unwrap();
        drop(request);
        assert!(handle.in_use());
        drop(callback);
        assert!(!handle.in_use());
    }

    #[test]
    fn overload_or_duplicate_registration_never_rebinds_a_token() {
        let bindings = ExecutionBindings::new(1).unwrap();
        let first = handle();
        let other = handle();
        let token = "x".repeat(32);
        bindings.register(token.clone(), &first).unwrap();
        assert!(bindings.register(token.clone(), &other).is_err());
        assert!(bindings.register("y".repeat(32), &other).is_err());
        assert!(!other.in_use());
        bindings.remove(&token).unwrap();
        assert!(!first.in_use());
    }
}
