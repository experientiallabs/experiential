//! PyO3 entry points for the Rust gateway data plane.
//!
//! `serve` blocks the calling Python thread (with the GIL released) while the
//! tokio server owns the socket; the Python control plane is reached through
//! bounded callbacks. The fixture functions expose the Rust SSE encoder and
//! failure taxonomy for byte-level parity tests against the Python engine.

mod bridge;
mod dialects;
mod encode;
mod errors;
mod events;
mod memory;
mod server;
mod sse;
mod upstream;

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::bridge::Bridge;
use crate::server::ServeConfig;

/// Serve the gateway data plane until shutdown (SIGINT or SIGTERM).
///
/// `control_plane` is a Python object exposing `authenticate`, `admit`,
/// `settle`, `models`, `model_detail`, `usage_json`, `usage_page`, and
/// `readiness`, each taking and returning one JSON string. `config_json`
/// carries host, port, and concurrency bounds.
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
    module.add_function(wrap_pyfunction!(encode_chat_fixture, module)?)?;
    module.add_function(wrap_pyfunction!(failure_public_error_fixture, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
