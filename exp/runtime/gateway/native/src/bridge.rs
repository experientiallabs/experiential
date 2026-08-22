//! Calls into the Python control plane (`NativeControlPlane`).
//!
//! Every call crosses the boundary as one JSON string in and one JSON string
//! out, executed on the blocking pool under a bounded permit count so GIL
//! contention stays fixed regardless of data-plane concurrency.

use std::sync::Arc;

use pyo3::prelude::*;
use tokio::sync::Semaphore;

use crate::errors::PublicError;

/// Bounded bridge to one Python `NativeControlPlane` instance.
pub struct Bridge {
    object: Py<PyAny>,
    permits: Arc<Semaphore>,
}

impl Bridge {
    pub fn new(object: Py<PyAny>, maximum_concurrent_calls: usize) -> Self {
        Self {
            object,
            permits: Arc::new(Semaphore::new(maximum_concurrent_calls.max(1))),
        }
    }

    /// Call one control-plane method with a JSON-string argument.
    pub async fn call(
        &self,
        method: &'static str,
        argument: String,
    ) -> Result<String, PublicError> {
        let _permit = self
            .permits
            .acquire()
            .await
            .map_err(|_| PublicError::internal())?;
        let object = Python::attach(|py| self.object.clone_ref(py));
        let outcome = tokio::task::spawn_blocking(move || {
            Python::attach(|py| -> Result<String, PublicError> {
                let bound = object.bind(py);
                match bound.call_method1(method, (argument,)) {
                    Ok(result) => result
                        .extract::<String>()
                        .map_err(|_| PublicError::internal()),
                    Err(error) => Err(public_error_from_pyerr(py, &error)),
                }
            })
        })
        .await;
        match outcome {
            Ok(result) => result,
            Err(_) => Err(PublicError::internal()),
        }
    }
}

/// Map one Python exception to a public error.
///
/// The control plane attaches a `public_error_json` attribute to every
/// sanitized boundary failure; anything without it is an internal error,
/// mirroring the Python engine's catch-all in `_exception_response`.
fn public_error_from_pyerr(py: Python<'_>, error: &PyErr) -> PublicError {
    let value = error.value(py);
    if let Ok(payload) = value.getattr("public_error_json") {
        if let Ok(text) = payload.extract::<String>() {
            if let Ok(public) = serde_json::from_str::<PublicError>(&text) {
                return public;
            }
        }
    }
    PublicError::internal()
}
