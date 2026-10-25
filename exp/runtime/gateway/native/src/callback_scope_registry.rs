//! In-process scheduling authority shared by a trusted host front and Rust.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Scheduling metadata for ephemeral relay credentials, never authentication.
#[pyclass]
pub struct CallbackScopes {
    pub(crate) inner: Arc<Mutex<HashMap<String, String>>>,
    capacity: usize,
}

#[pymethods]
impl CallbackScopes {
    #[new]
    #[pyo3(signature = (capacity=4096))]
    fn new(capacity: usize) -> PyResult<Self> {
        if capacity == 0 || capacity > 65_536 {
            return Err(PyValueError::new_err(
                "callback scope capacity must be 1 through 65536",
            ));
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            capacity,
        })
    }

    /// Only the trusted host registers a freshly generated relay token and owner.
    fn register(&self, token: String, organization: String) -> PyResult<()> {
        if token.len() < 32
            || token.len() > 512
            || organization.is_empty()
            || organization.len() > 128
        {
            return Err(PyValueError::new_err(
                "invalid callback scheduling identity",
            ));
        }
        let mut entries = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("callback scopes unavailable"))?;
        if let Some(existing) = entries.get(&token) {
            if existing != &organization {
                return Err(PyValueError::new_err(
                    "callback token ownership cannot change",
                ));
            }
            return Ok(());
        }
        if entries.len() >= self.capacity {
            return Err(PyRuntimeError::new_err("callback scope capacity exhausted"));
        }
        entries.insert(token, organization);
        Ok(())
    }

    /// Relay teardown removes only the token that relay registered.
    fn remove(&self, token: &str) -> PyResult<()> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("callback scopes unavailable"))?
            .remove(token);
        Ok(())
    }
}
