//! PyO3 entry points for the Rust gateway data plane.
//!
//! `serve` blocks the calling Python thread (with the GIL released) while the
//! tokio server owns the socket; the Python control plane is reached through
//! bounded callbacks. `decode_chat_canonical` exposes the Rust decoder for
//! byte-level parity tests against the Python decoder.

mod bridge;
mod decode;
mod dialects;
mod encode;
mod errors;
mod events;
mod server;
mod sse;
mod upstream;

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::bridge::Bridge;
use crate::server::ServeConfig;

/// Serve the gateway data plane until shutdown (SIGINT).
///
/// `control_plane` is a Python object exposing `authenticate`, `admit`,
/// `settle`, `models`, `model_detail`, `usage_json`, and `readiness`, each
/// taking and returning one JSON string. `config_json` carries host, port,
/// and concurrency bounds.
#[pyfunction]
fn serve(py: Python<'_>, control_plane: Py<PyAny>, config_json: &str) -> PyResult<()> {
    let config: ServeConfig = serde_json::from_str(config_json)
        .map_err(|error| PyValueError::new_err(format!("invalid serve config: {error}")))?;
    let bridge = Arc::new(Bridge::new(control_plane, config.callback_permits));
    let outcome = py.allow_threads(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|error| format!("tokio runtime construction failed: {error}"))?;
        runtime.block_on(server::run(bridge, config))
    });
    outcome.map_err(PyRuntimeError::new_err)
}

/// Decode one Chat Completions body with the Rust decoder for parity tests.
///
/// Returns a JSON envelope `{"alias": ..., "canonical": ...}` or raises
/// `ValueError` whose message is the public-error JSON payload.
#[pyfunction]
#[pyo3(signature = (body, idempotency_key=None, client_request_id=None))]
fn decode_chat_canonical(
    body: &str,
    idempotency_key: Option<&str>,
    client_request_id: Option<&str>,
) -> PyResult<String> {
    let payload: serde_json::Value = serde_json::from_str(body)
        .map_err(|_| PyValueError::new_err(error_payload(&errors::PublicError::invalid_json())))?;
    let object = payload
        .as_object()
        .ok_or_else(|| PyValueError::new_err(error_payload(&errors::PublicError::not_json_object())))?;
    match decode::decode_chat(object, idempotency_key, client_request_id) {
        Ok(decoded) => Ok(serde_json::json!({
            "alias": decoded.alias,
            "canonical": decoded.canonical,
        })
        .to_string()),
        Err(error) => Err(PyValueError::new_err(error_payload(&error))),
    }
}

/// Build one upstream payload with the Rust dialect builders for parity tests.
#[pyfunction]
#[pyo3(signature = (dialect, canonical_request, model_id, supports_temperature=true, reasoning_effort=None, token_limit_key="max_tokens"))]
fn build_upstream_payload(
    dialect: &str,
    canonical_request: &str,
    model_id: &str,
    supports_temperature: bool,
    reasoning_effort: Option<String>,
    token_limit_key: &str,
) -> PyResult<String> {
    let request: serde_json::Value = serde_json::from_str(canonical_request)
        .map_err(|error| PyValueError::new_err(format!("invalid canonical request: {error}")))?;
    let parsed = dialects::Dialect::from_str(dialect)
        .ok_or_else(|| PyValueError::new_err(format!("unknown dialect: {dialect}")))?;
    let hints = dialects::WireHints {
        model_id: model_id.to_string(),
        supports_temperature,
        reasoning_effort,
        token_limit_key: token_limit_key.to_string(),
    };
    match dialects::build_payload(parsed, &request, &hints) {
        Ok(payload) => Ok(payload.to_string()),
        Err(failure) => Err(PyValueError::new_err(
            serde_json::to_string(&failure).unwrap_or_default(),
        )),
    }
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
    let events = parse_fixture_events(events_json)
        .map_err(|message| PyValueError::new_err(message))?;
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
                input_tokens: object.get("input_tokens").and_then(serde_json::Value::as_u64),
                output_tokens: object.get("output_tokens").and_then(serde_json::Value::as_u64),
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
                if text.is_empty() { "provider stream failed" } else { &text },
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
    module.add_function(wrap_pyfunction!(decode_chat_canonical, module)?)?;
    module.add_function(wrap_pyfunction!(build_upstream_payload, module)?)?;
    module.add_function(wrap_pyfunction!(encode_chat_fixture, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
