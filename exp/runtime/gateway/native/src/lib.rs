//! PyO3 entry points for the Rust gateway data plane.
//!
//! `serve` blocks the calling Python thread (with the GIL released) while the
//! tokio server owns the socket; the Python control plane is reached through
//! bounded callbacks. The fixture functions expose the Rust SSE encoder and
//! failure taxonomy for byte-level parity tests against the Python engine.

mod admission;
mod bridge;
mod dialects;
mod encode;
mod encode_messages;
mod encode_responses;
mod errors;
mod events;
mod eventstream;
mod guardrails;
mod memory;
mod metrics;
mod proxy;
mod relay;
mod replay;
mod respond;
mod route_chat;
mod route_messages;
mod route_responses;
mod server;
mod settlement;
mod sse;
mod upstream;
mod waterfall;

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::bridge::Bridge;
use crate::server::ServeConfig;

/// Serve the gateway data plane until shutdown (SIGINT or SIGTERM).
///
/// `control_plane` is a Python object exposing `authenticate`, `admit`,
/// `start_attempt`, `sign_dispatch`, `settle`, `abandon`, `remember`,
/// `enforce_output`, `models`, `model_detail`, `usage_json`, `usage_page`,
/// `metrics_json`, `metrics_text`, and `readiness`, each taking and
/// returning one JSON string. `config_json` carries host, port, and
/// concurrency bounds. `enforce_output` is called only when admission sets
/// `output_guardrail`.
#[pyfunction]
fn serve(py: Python<'_>, control_plane: Py<PyAny>, config_json: &str) -> PyResult<()> {
    let config: ServeConfig = serde_json::from_str(config_json)
        .map_err(|error| PyValueError::new_err(format!("invalid serve config: {error}")))?;
    let bridge = Arc::new(Bridge::new(control_plane, config.callback_permits));
    let outcome = py.detach(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|error| format!("tokio runtime construction failed: {error}"))?;
        runtime.block_on(server::run(bridge, config))
    });
    outcome.map_err(PyRuntimeError::new_err)
}

/// Encode one normalized event fixture through the Rust Chat SSE encoder for
/// byte parity tests. `events_json` is a list of simplified event objects.
#[pyfunction]
fn encode_chat_fixture(
    request_id: &str,
    model: &str,
    created_at: i64,
    include_usage: bool,
    events_json: &str,
) -> PyResult<Vec<String>> {
    let events = parse_fixture_events(events_json).map_err(PyValueError::new_err)?;
    let mut encoder = encode::ChatSseEncoder::new(request_id, model, created_at, include_usage);
    let mut frames = encoder
        .start()
        .map_err(|error| PyValueError::new_err(error_payload(&error)))?;
    for event in &events {
        frames.extend(
            encoder
                .feed(event)
                .map_err(|error| PyValueError::new_err(error_payload(&error)))?,
        );
    }
    Ok(frames)
}

/// Snapshot the process-global data-plane metrics registry as one JSON
/// object. Python hosts compose this content-free snapshot with the control
/// plane's own counters (see `NativeControlPlane.metrics_snapshot`).
#[pyfunction]
fn metrics_snapshot_json() -> String {
    metrics::METRICS.snapshot().to_string()
}

/// Encode one normalized event fixture through the Rust Responses SSE
/// encoder for byte parity tests. `envelope_json` carries the
/// request-reflecting envelope fields; `events_json` is a list of simplified
/// event objects.
#[pyfunction]
fn encode_responses_fixture(
    request_id: &str,
    model: &str,
    created_at: f64,
    envelope_json: &str,
    events_json: &str,
) -> PyResult<Vec<String>> {
    let envelope: encode_responses::ResponsesEnvelope = serde_json::from_str(envelope_json)
        .map_err(|error| PyValueError::new_err(format!("invalid envelope: {error}")))?;
    let events = parse_fixture_events(events_json).map_err(PyValueError::new_err)?;
    let mut encoder =
        encode_responses::ResponsesSseEncoder::new(request_id, model, created_at, envelope);
    let mut frames = encoder
        .start()
        .map_err(|error| PyValueError::new_err(error_payload(&error)))?;
    for event in &events {
        frames.extend(
            encoder
                .feed(event)
                .map_err(|error| PyValueError::new_err(error_payload(&error)))?,
        );
    }
    Ok(frames)
}

/// Build one non-streaming Responses body fixture through the Rust
/// aggregation for byte parity tests against the python `completed_body`.
#[pyfunction]
fn completed_responses_fixture(
    request_id: &str,
    model: &str,
    created_at: f64,
    envelope_json: &str,
    events_json: &str,
) -> PyResult<String> {
    let envelope: encode_responses::ResponsesEnvelope = serde_json::from_str(envelope_json)
        .map_err(|error| PyValueError::new_err(format!("invalid envelope: {error}")))?;
    let events = parse_fixture_events(events_json).map_err(PyValueError::new_err)?;
    let aggregated = encode_responses::completed_responses_body(
        request_id, model, created_at, envelope, &events,
    )
    .map_err(|error| PyValueError::new_err(error_payload(&error)))?;
    Ok(serde_json::to_string(&aggregated.body).unwrap_or_else(|_| "null".to_string()))
}

/// Encode one normalized event fixture through the Rust Anthropic Messages
/// SSE encoder for byte parity tests. `events_json` is a list of simplified
/// event objects.
#[pyfunction]
fn encode_messages_fixture(
    request_id: &str,
    model: &str,
    events_json: &str,
) -> PyResult<Vec<String>> {
    let events = parse_fixture_events(events_json).map_err(PyValueError::new_err)?;
    let mut encoder = encode_messages::MessagesSseEncoder::new(request_id, model);
    let mut frames = encoder
        .start()
        .map_err(|error| PyValueError::new_err(error_payload(&error)))?;
    for event in &events {
        frames.extend(
            encoder
                .feed(event)
                .map_err(|error| PyValueError::new_err(error_payload(&error)))?,
        );
    }
    Ok(frames)
}

/// Build one non-streaming Anthropic message body fixture through the Rust
/// aggregation for byte parity tests against `completed_messages_body`.
#[pyfunction]
fn completed_messages_fixture(
    request_id: &str,
    model: &str,
    events_json: &str,
) -> PyResult<String> {
    let events = parse_fixture_events(events_json).map_err(PyValueError::new_err)?;
    let aggregated = encode_messages::completed_messages_body(request_id, model, &events)
        .map_err(|error| PyValueError::new_err(error_payload(&error)))?;
    if let Some(failure) = aggregated.failure {
        return Err(PyValueError::new_err(error_payload(
            &failure.public_error(),
        )));
    }
    Ok(serde_json::to_string(&aggregated.body).unwrap_or_else(|_| "null".to_string()))
}

/// Render one OpenAI-shaped public error as the Anthropic error envelope for
/// translation parity tests against `anthropic_error_body`.
#[pyfunction]
fn anthropic_error_fixture(public_error_json: &str) -> PyResult<String> {
    let error: errors::PublicError = serde_json::from_str(public_error_json)
        .map_err(|error| PyValueError::new_err(format!("invalid public error: {error}")))?;
    Ok(
        serde_json::to_string(&encode_messages::anthropic_error_body(&error))
            .unwrap_or_else(|_| "null".to_string()),
    )
}

/// Normalize one raw provider stream fixture through the Rust frame decoder
/// and dialect normalizer for parity tests against the python event mappers.
///
/// `chunks_json` is a JSON array of latin-1 encoded chunk strings (one
/// character per raw byte, so binary framings round-trip losslessly). The
/// result is a JSON object with `events` (simplified canonical events in
/// order) and `failure` (the class and safe message that ended the stream, or
/// null when it ended on its own terminal event).
#[pyfunction]
fn normalize_stream_fixture(dialect: &str, chunks_json: &str) -> PyResult<String> {
    let dialect = dialects::Dialect::from_str(dialect)
        .ok_or_else(|| PyValueError::new_err(format!("unknown dialect: {dialect}")))?;
    let chunks: Vec<String> = serde_json::from_str(chunks_json)
        .map_err(|error| PyValueError::new_err(format!("invalid chunks: {error}")))?;
    let bytes: Vec<Vec<u8>> = chunks
        .iter()
        .map(|chunk| respond::latin1_bytes(chunk))
        .collect();
    let (simplified, failure) = dialects::drain_stream_fixture(dialect, &bytes);
    let body = serde_json::json!({
        "events": simplified,
        "failure": failure.map(|failure| serde_json::json!({
            "failure_class": failure.failure_class.as_str(),
            "safe_message": failure.safe_message,
        })),
    });
    Ok(body.to_string())
}

/// Map one failure class and safe message to the Rust public-error JSON for
/// taxonomy parity tests against `public_failure_error`.
#[pyfunction]
fn failure_public_error_fixture(failure_class: &str, safe_message: &str) -> PyResult<String> {
    let parsed: errors::FailureClass = serde_json::from_value(serde_json::Value::String(
        failure_class.to_string(),
    ))
    .map_err(|_| PyValueError::new_err(format!("unknown failure class: {failure_class}")))?;
    let failure = errors::Failure::new(parsed, safe_message);
    Ok(error_payload(&failure.public_error()))
}

fn parse_fixture_events(events_json: &str) -> Result<Vec<events::Event>, String> {
    let raw: Vec<serde_json::Value> =
        serde_json::from_str(events_json).map_err(|error| format!("invalid events: {error}"))?;
    let mut parsed = Vec::with_capacity(raw.len());
    for value in raw {
        let object = value.as_object().ok_or("event must be an object")?;
        let kind = object
            .get("kind")
            .and_then(serde_json::Value::as_str)
            .ok_or("event requires kind")?;
        let text = object
            .get("text")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_string();
        let index = object
            .get("index")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as u32;
        let event = match kind {
            "text_delta" => events::Event::TextDelta(text),
            "refusal_delta" => events::Event::RefusalDelta(text),
            "reasoning_summary_delta" => events::Event::ReasoningSummaryDelta {
                output_index: object
                    .get("output_index")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0) as u32,
                summary_index: object
                    .get("summary_index")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0) as u32,
                delta: text,
            },
            "tool_call_started" => events::Event::ToolCallStarted {
                index,
                call_id: object
                    .get("call_id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                name: object
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            },
            "tool_arguments_delta" => events::Event::ToolArgumentsDelta { index, delta: text },
            "tool_call_completed" => events::Event::ToolCallCompleted {
                index,
                call: events::CompletedToolCall {
                    call_id: object
                        .get("call_id")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    name: object
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    raw_arguments: object
                        .get("raw_arguments")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                },
            },
            "usage" => events::Event::Usage(events::Usage {
                input_tokens: object
                    .get("input_tokens")
                    .and_then(serde_json::Value::as_u64),
                output_tokens: object
                    .get("output_tokens")
                    .and_then(serde_json::Value::as_u64),
                cached_input_tokens: object
                    .get("cached_input_tokens")
                    .and_then(serde_json::Value::as_u64),
                reasoning_tokens: object
                    .get("reasoning_tokens")
                    .and_then(serde_json::Value::as_u64),
            }),
            "completed" => events::Event::Completed,
            "incomplete" => events::Event::Incomplete,
            "failed" => events::Event::Failed(errors::Failure::new(
                errors::FailureClass::ProviderInternal,
                if text.is_empty() {
                    "provider stream failed"
                } else {
                    &text
                },
            )),
            other => return Err(format!("unknown event kind: {other}")),
        };
        parsed.push(event);
    }
    Ok(parsed)
}

fn error_payload(error: &errors::PublicError) -> String {
    serde_json::to_string(error).unwrap_or_else(|_| "{}".to_string())
}

/// The exp_gateway_native extension module.
#[pymodule]
fn exp_gateway_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(serve, module)?)?;
    module.add_function(wrap_pyfunction!(metrics_snapshot_json, module)?)?;
    module.add_function(wrap_pyfunction!(encode_chat_fixture, module)?)?;
    module.add_function(wrap_pyfunction!(encode_messages_fixture, module)?)?;
    module.add_function(wrap_pyfunction!(encode_responses_fixture, module)?)?;
    module.add_function(wrap_pyfunction!(completed_messages_fixture, module)?)?;
    module.add_function(wrap_pyfunction!(completed_responses_fixture, module)?)?;
    module.add_function(wrap_pyfunction!(anthropic_error_fixture, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_stream_fixture, module)?)?;
    module.add_function(wrap_pyfunction!(failure_public_error_fixture, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
