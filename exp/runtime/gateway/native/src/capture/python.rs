//! Thin Python configuration and persistence boundary for the shared Rust collector.

use std::sync::Arc;
use std::time::{Duration, Instant};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use serde::Deserialize;
use serde_json::{value::RawValue, Value};

use super::collector::{Collector, Configuration};
use super::delivery::Sink;
use super::record::{Record, Request};

/// Python view of the same response evidence used by the gateway's local sink.
#[pyclass(frozen, module = "exp_gateway_native")]
pub struct CaptureResponse {
    #[pyo3(get)]
    body_json: String,
    #[pyo3(get)]
    completed: bool,
    #[pyo3(get)]
    events_json: Option<String>,
}

#[pymethods]
impl CaptureResponse {
    #[new]
    fn new(py: Python<'_>, protocol: &str, body: &[u8], sse: bool) -> PyResult<Self> {
        use super::{projection::CapturedResponse, record::Protocol};
        let protocol = match protocol {
            "responses" => Protocol::Responses,
            "chat" => Protocol::ChatCompletions,
            "messages" => Protocol::Messages,
            _ => return Err(PyValueError::new_err("unknown capture protocol")),
        };
        let (body_json, completed, events_json) =
            py.detach(|| CapturedResponse::decode(protocol, body, sse));
        Ok(Self {
            body_json,
            completed,
            events_json,
        })
    }
}

struct PythonSink(Py<PyAny>);

#[derive(Deserialize)]
struct ContextSource<'a> {
    #[serde(borrow)]
    context: &'a RawValue,
}

const BATCH_RECORDS: usize = 64;
const BATCH_BYTES: usize = 2 * 1024 * 1024;

struct PythonBatchSink {
    callback: Py<PyAny>,
    completion_references: bool,
    bytes_output: bool,
}

struct EncodedRecord {
    value: Py<PyAny>,
    bytes: usize,
}

impl Sink for PythonBatchSink {
    type Prepared = EncodedRecord;

    fn preparation_bytes(maximum_record_bytes: usize) -> usize {
        // The last record may cross the soft batch target. Bound both strings
        // (up to four bytes per character) and the concurrent UTF-8 encoding.
        (maximum_record_bytes + BATCH_BYTES) * 5 + BATCH_RECORDS * 256
    }

    fn prepare(&self, record: &Record, maximum_bytes: usize) -> Result<Self::Prepared, ()> {
        let encoded = if self.completion_references {
            record.encode_update(maximum_bytes)
        } else {
            record.encode(maximum_bytes)
        }
        .ok_or(())?;
        let bytes = encoded.len();
        Python::try_attach(|py| {
            let value = if self.bytes_output {
                PyBytes::new(py, encoded.as_bytes()).into_any().unbind()
            } else {
                encoded.into_pyobject(py)?.into_any().unbind()
            };
            Ok::<_, PyErr>(EncodedRecord { value, bytes })
        })
        .ok_or(())?
        .map_err(|_| ())
    }

    fn write(&mut self, prepared: &Self::Prepared) -> Result<(), ()> {
        if self.write_batch(&[prepared]) == [true] {
            Ok(())
        } else {
            Err(())
        }
    }

    fn batch_records(&self) -> usize {
        BATCH_RECORDS
    }

    fn batch_bytes(&self) -> usize {
        BATCH_BYTES
    }

    fn batch_delay(&self) -> Duration {
        Duration::from_millis(10)
    }

    fn prepared_bytes(&self, prepared: &Self::Prepared) -> usize {
        prepared.bytes
    }

    fn write_batch(&mut self, prepared: &[&Self::Prepared]) -> Vec<bool> {
        Python::try_attach(|py| {
            // A tuple of references: payloads are never joined, parsed or encoded
            // again for batching. The callback returns per-record durable acks.
            let records = PyTuple::new(py, prepared.iter().map(|p| p.value.bind(py))).ok()?;
            self.callback
                .bind(py)
                .call1((records,))
                .ok()?
                .extract::<Vec<bool>>()
                .ok()
        })
        .flatten()
        .unwrap_or_default()
    }
}

impl Sink for PythonSink {
    type Prepared = Py<PyAny>;

    fn preparation_bytes(maximum_record_bytes: usize) -> usize {
        // CPython may use four bytes per character even in mostly-ASCII JSON
        // when one supplementary Unicode character occurs. Include the UTF-8
        // encoding that overlaps construction, plus object header/slack.
        maximum_record_bytes.saturating_mul(5).saturating_add(256)
    }

    fn prepare(&self, record: &Record, maximum_bytes: usize) -> Result<Self::Prepared, ()> {
        let encoded = record.encode(maximum_bytes).ok_or(())?;
        Python::try_attach(|py| {
            encoded
                .into_pyobject(py)
                .map(|value| value.into_any().unbind())
        })
        .ok_or(())?
        .map_err(|_| ())
    }

    fn write(&mut self, prepared: &Self::Prepared) -> Result<(), ()> {
        // This is the dedicated delivery worker, never a serving or bridge thread.
        // Exception text can contain SQL parameters or content, so only count failure.
        Python::try_attach(|py| {
            self.0
                .bind(py)
                .call1((prepared.bind(py),))
                .map(|_| ())
                .map_err(|_| ())
        })
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
    /// Deliver bounded groups with per-update commit acknowledgements. Default
    /// records remain complete schema 1. Opt-in schema-2 completion references
    /// require a destination that retries until the prompt checkpoint is durable.
    #[staticmethod]
    #[pyo3(signature = (config_json, sink, *, completion_references=false, bytes_output=false))]
    fn batched(
        py: Python<'_>,
        config_json: &str,
        sink: Py<PyAny>,
        completion_references: bool,
        bytes_output: bool,
    ) -> PyResult<Self> {
        if !sink.bind(py).is_callable() {
            return Err(PyValueError::new_err("capture batch sink must be callable"));
        }
        let config: Configuration = serde_json::from_str(config_json)
            .map_err(|_| PyValueError::new_err("invalid capture configuration"))?;
        py.detach(|| {
            Collector::new(
                config,
                PythonBatchSink {
                    callback: sink,
                    completion_references,
                    bytes_output,
                },
            )
        })
        .map(|collector| Self {
            inner: Arc::new(collector),
        })
        .map_err(PyValueError::new_err)
    }

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
        self.begin_bytes(py, request_json.as_bytes())
    }

    /// Borrow immutable UTF-8 bytes without widening and re-encoding a Python string.
    fn begin_bytes(&self, py: Python<'_>, request_json: &[u8]) -> bool {
        let collector = self.inner.clone();
        py.detach(|| {
            if !collector.config.truncate_request
                && request_json.len() > collector.config.maximum_request_bytes
            {
                return collector.skip();
            }
            let Ok(mut request) = serde_json::from_slice::<Request>(request_json) else {
                return collector.skip();
            };
            // The ordinary JSON projection uses finite-width numbers. Keep the
            // original context only for exceptional numeric values, using the
            // same lossless sidecar consumed by capture ingestion. Do not change
            // the gateway's global number representation to repair capture.
            if request
                .context
                .get("source_json")
                .is_none_or(Value::is_null)
                && super::response::contains_wide_number(&request.context)
            {
                let Ok(source) = serde_json::from_slice::<ContextSource>(request_json) else {
                    return collector.skip();
                };
                let Some(context) = Arc::make_mut(&mut request.context).as_object_mut() else {
                    return collector.skip();
                };
                context.insert(
                    "source_json".into(),
                    Value::String(source.context.get().to_owned()),
                );
            }
            collector.begin(request)
        })
    }

    /// Freeze resolved model provenance before response collection begins.
    fn select_model(&self, py: Python<'_>, request_id: &str, model_id: &str) {
        let collector = self.inner.clone();
        py.detach(|| collector.select_model(request_id, model_id));
    }

    /// Claim caller-facing metadata only for an admitted original response.
    fn claim_relay(&self, py: Python<'_>, request_id: &str) -> bool {
        py.detach(|| self.inner.claim_relay(request_id))
    }

    /// Transfer raw wire bytes once; the delivery worker alone parses their JSON.
    fn finish_relay(
        &self,
        py: Python<'_>,
        request_id: &str,
        metadata_json: &str,
        body: Vec<u8>,
    ) -> bool {
        py.detach(|| {
            if metadata_json.len() > 65536 || body.len() > 4 * 1024 * 1024 {
                return false;
            }
            let Ok(metadata) = serde_json::from_str(metadata_json) else {
                return false;
            };
            self.inner
                .finish_relay(request_id, super::relay::Relay { metadata, body })
        })
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

    /// Retention or WAL cleanup failures, separate from durable-write failures.
    fn maintenance_failures(&self) -> u64 {
        self.inner.maintenance_failures()
    }
}
