//! Thin Python configuration and persistence boundary for the shared Rust collector.

use std::sync::Arc;
use std::time::{Duration, Instant};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::collector::{Collector, Configuration};
use super::delivery::Sink;
use super::record::Request;

struct PythonSink(Py<PyAny>);

impl Sink for PythonSink {
    fn write(&mut self, record: &str) -> Result<(), ()> {
        // This is the dedicated delivery worker, never a serving or bridge thread.
        // Exception text can contain SQL parameters or content, so only count failure.
        Python::try_attach(|py| self.0.bind(py).call1((record,)).map(|_| ()).map_err(|_| ()))
            .unwrap_or(Err(()))
    }
}

/// One collector shared by the authority adapter, native server and storage destination.
#[pyclass]
pub struct CaptureCollector {
    pub(crate) inner: Arc<Collector>,
}

#[pymethods]
impl CaptureCollector {
    /// Use the same collector and delivery worker with a native local SQLite sink.
    #[staticmethod]
    fn sqlite(py: Python<'_>, config_json: &str, local_json: &str) -> PyResult<Option<Self>> {
        let config: Configuration = serde_json::from_str(config_json)
            .map_err(|_| PyValueError::new_err("invalid capture configuration"))?;
        config.validate().map_err(PyValueError::new_err)?;
        if config.settlement_required {
            return Err(PyValueError::new_err(
                "local capture cannot require hosted settlement",
            ));
        }
        let local: super::local::CaptureConfiguration = serde_json::from_str(local_json)
            .map_err(|_| PyValueError::new_err("invalid local capture configuration"))?;
        if config.delivery.maximum_records != local.queue_capacity {
            return Err(PyValueError::new_err("local delivery bounds must match"));
        }
        py.detach(|| {
            let Some(sink) = super::local::SqliteSink::open(local)? else {
                return Ok(None);
            };
            Collector::new(config, sink)
                .map(|collector| {
                    Some(Self {
                        inner: Arc::new(collector),
                    })
                })
                .map_err(str::to_owned)
        })
        .map_err(PyValueError::new_err)
    }

    /// Configure bounded native capture with an off-path synchronous destination.
    #[new]
    fn new(py: Python<'_>, config_json: &str, sink: Py<PyAny>) -> PyResult<Self> {
        if !sink.bind(py).is_callable() {
            return Err(PyValueError::new_err("capture sink must be callable"));
        }
        let config: Configuration = serde_json::from_str(config_json)
            .map_err(|_| PyValueError::new_err("invalid capture configuration"))?;
        py.detach(|| Collector::new(config, PythonSink(sink)))
            .map(|collector| Self {
                inner: Arc::new(collector),
            })
            .map_err(PyValueError::new_err)
    }

    /// Register effective input from authenticated admission, without writing content.
    fn begin(&self, py: Python<'_>, request_json: &str) -> bool {
        let collector = self.inner.clone();
        py.detach(|| {
            if request_json.len() > collector.config.maximum_request_bytes {
                return collector.skip();
            }
            let Ok(request) = serde_json::from_str::<Request>(request_json) else {
                return collector.skip();
            };
            collector.begin(request)
        })
    }

    /// Freeze resolved model provenance before response collection begins.
    fn select_model(&self, py: Python<'_>, request_id: &str, model_id: &str) {
        let collector = self.inner.clone();
        py.detach(|| collector.select_model(request_id, model_id));
    }

    /// Apply the host's final content eligibility, independently of inference accounting.
    fn settle(&self, py: Python<'_>, request_id: &str, keep_prompt: bool, keep_response: bool) {
        let collector = self.inner.clone();
        py.detach(|| collector.settle(request_id, keep_prompt, keep_response));
    }

    /// Close within the stated drain budget, releasing the GIL while a sink finishes.
    #[pyo3(signature = (timeout_seconds=10.0))]
    fn close(&self, py: Python<'_>, timeout_seconds: f64) -> PyResult<bool> {
        if !timeout_seconds.is_finite() || !(0.0..=3600.0).contains(&timeout_seconds) {
            return Err(PyValueError::new_err("invalid capture drain timeout"));
        }
        let collector = self.inner.clone();
        let deadline = Instant::now() + Duration::from_secs_f64(timeout_seconds);
        Ok(py.detach(|| collector.close_until(deadline)))
    }

    /// Content-free pending, byte, success, failure, overload and skip counters.
    fn counts(&self) -> (u64, u64, u64, u64, u64, u64) {
        let [pending, bytes, persisted, failed, dropped, skipped] = self.inner.counts();
        (pending, bytes, persisted, failed, dropped, skipped)
    }
}
