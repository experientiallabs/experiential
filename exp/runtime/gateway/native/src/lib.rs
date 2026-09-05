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
mod param_attribution;
mod relay;
mod replay;
mod respond;
mod responses_retention;
mod route_batches;
mod route_chat;
mod route_embeddings;
mod route_images;
mod route_messages;
mod route_responses;
mod route_responses_ws;
mod server;
mod settlement;
mod sse;
mod upstream;
mod waterfall;

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::bridge::Bridge;
use crate::server::ServeConfig;

/// Embedder-owned stop signal for one `serve` call.
///
/// Created with `shutdown_handle()` and passed to `serve`, whose server then
/// stops gracefully on either a process signal or `request_shutdown`. This
/// lets a host that runs the data plane on a background thread stop it
/// programmatically, since threads cannot receive SIGINT.
#[pyclass]
pub struct ShutdownHandle {
    sender: Mutex<Option<tokio::sync::watch::Sender<bool>>>,
    receiver: Mutex<Option<tokio::sync::watch::Receiver<bool>>>,
}

#[pymethods]
impl ShutdownHandle {
    /// Request one graceful stop; later calls are no-ops.
    fn request_shutdown(&self) {
        if let Ok(guard) = self.sender.lock() {
            if let Some(sender) = guard.as_ref() {
                let _ = sender.send(true);
            }
        }
    }
}

impl ShutdownHandle {
    /// Take the receiver half exactly once for the serving runtime.
    fn take_receiver(&self) -> Option<tokio::sync::watch::Receiver<bool>> {
        self.receiver.lock().ok().and_then(|mut guard| guard.take())
    }
}

/// Create one stop handle to pass to `serve`.
#[pyfunction]
fn shutdown_handle() -> ShutdownHandle {
    let (sender, receiver) = tokio::sync::watch::channel(false);
    ShutdownHandle {
        sender: Mutex::new(Some(sender)),
        receiver: Mutex::new(Some(receiver)),
    }
}

/// Serve the gateway data plane until shutdown (SIGINT, SIGTERM, or an
/// optional embedder-owned `ShutdownHandle`).
///
/// `control_plane` is a Python object exposing `authenticate`, `admit`,
/// `start_attempt`, `sign_dispatch`, `settle`, `abandon`, `remember`,
/// `enforce_output`, `models`, `model_detail`, `usage_json`, `usage_page`,
/// `metrics_json`, `metrics_text`, `readiness`, and
/// `close_thread_resources`, each taking and returning one JSON string.
/// `config_json` carries host, port, and concurrency bounds.
/// `enforce_output` is called only when admission sets `output_guardrail`;
/// `close_thread_resources` is called once per bridge worker thread as it
/// exits so per-thread caches release with the pool.
#[pyfunction]
#[pyo3(signature = (control_plane, config_json, shutdown=None, on_listening=None))]
fn serve(
    py: Python<'_>,
    control_plane: Py<PyAny>,
    config_json: &str,
    shutdown: Option<&ShutdownHandle>,
    on_listening: Option<Py<PyAny>>,
) -> PyResult<()> {
    let config: ServeConfig = serde_json::from_str(config_json)
        .map_err(|error| PyValueError::new_err(format!("invalid serve config: {error}")))?;
    let bridge = Arc::new(
        Bridge::new(control_plane, config.callback_permits).map_err(PyRuntimeError::new_err)?,
    );
    let stop = shutdown.and_then(ShutdownHandle::take_receiver);
    let outcome = py.detach(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|error| format!("tokio runtime construction failed: {error}"))?;
        runtime.block_on(server::run(bridge, config, stop, on_listening))
    });
    outcome.map_err(PyRuntimeError::new_err)
}

/// Encode one normalized event fixture through the Rust Chat SSE encoder for
/// byte parity tests. `events_json` is a list of simplified event objects and
/// `ignored_parameters` names route-shaped controls disclosed on the final
/// chunk.
#[pyfunction]
#[pyo3(signature = (request_id, model, created_at, include_usage, events_json, ignored_parameters=Vec::new()))]
fn encode_chat_fixture(
    request_id: &str,
    model: &str,
    created_at: i64,
    include_usage: bool,
    events_json: &str,
    ignored_parameters: Vec<String>,
) -> PyResult<Vec<String>> {
    let events = parse_fixture_events(events_json).map_err(PyValueError::new_err)?;
    let mut encoder = encode::ChatSseEncoder::new_with_ignored(
        request_id,
        model,
        created_at,
        include_usage,
        ignored_parameters,
    );
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
        let output_index = object
            .get("output_index")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as u32;
        let item_id = object
            .get("item_id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        let status = match object.get("status") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(value)) => Some(
                events::ProviderOutputItemStatus::from_str(value)
                    .ok_or_else(|| format!("unknown provider output item status: {value}"))?,
            ),
            Some(_) => return Err("provider output item status must be text".to_string()),
        };
        let phase = match object.get("phase") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(value)) => Some(
                events::ProviderAssistantMessagePhase::from_str(value)
                    .ok_or_else(|| format!("unknown provider message phase: {value}"))?,
            ),
            Some(_) => return Err("provider message phase must be text".to_string()),
        };
        let provider_kind = || match object.get("item_type").and_then(serde_json::Value::as_str) {
            Some("reasoning") => Ok(events::ProviderOutputItemKind::Reasoning),
            Some("function_call") => Ok(events::ProviderOutputItemKind::FunctionCall),
            Some("message") => Ok(events::ProviderOutputItemKind::Message),
            Some(other) => Err(format!("unknown provider output item kind: {other}")),
            None => Err("provider output item requires item_type".to_string()),
        };
        let event = match kind {
            "text_delta" => events::Event::TextDelta(text),
            "refusal_delta" => events::Event::RefusalDelta(text),
            "provider_text_delta" => events::Event::ProviderTextDelta {
                output_index,
                item_id: item_id.ok_or("provider text delta requires item_id")?,
                delta: text,
            },
            "provider_refusal_delta" => events::Event::ProviderRefusalDelta {
                output_index,
                item_id: item_id.ok_or("provider refusal delta requires item_id")?,
                delta: text,
            },
            "provider_output_item_started" => events::Event::ProviderOutputItemStarted {
                output_index,
                item_id,
                kind: provider_kind()?,
                status,
                phase,
            },
            "provider_output_item_completed" => events::Event::ProviderOutputItemCompleted {
                output_index,
                item_id,
                kind: provider_kind()?,
                status,
                phase,
            },
            "reasoning_summary_delta" => events::Event::ReasoningSummaryDelta {
                output_index: object
                    .get("output_index")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0) as u32,
                summary_index: object
                    .get("summary_index")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0) as u32,
                item_id: object
                    .get("item_id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                delta: text,
            },
            "thinking_delta" => events::Event::ThinkingDelta { index, delta: text },
            "thinking_signature" => events::Event::ThinkingSignature {
                index,
                signature: object
                    .get("signature")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            },
            "redacted_thinking" => events::Event::RedactedThinking {
                index,
                data: object
                    .get("data")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            },
            "encrypted_reasoning" => events::Event::EncryptedReasoning {
                output_index: object
                    .get("output_index")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0) as u32,
                item_id: object
                    .get("item_id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                encrypted_content: object
                    .get("encrypted_content")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
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
                namespace: object
                    .get("namespace")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
                caller: object
                    .get("caller")
                    .filter(|value| value.is_object())
                    .cloned(),
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
                    namespace: object
                        .get("namespace")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string),
                    caller: object
                        .get("caller")
                        .filter(|value| value.is_object())
                        .cloned(),
                    provider_item_id: object
                        .get("item_id")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string),
                    provider_status: status,
                    raw_arguments: object
                        .get("raw_arguments")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    custom: false,
                },
            },
            "usage" => events::Event::Usage(events::Usage {
                input_tokens: object
                    .get("input_tokens")
                    .and_then(serde_json::Value::as_u64),
                cache_creation_input_tokens: object
                    .get("cache_creation_input_tokens")
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
            "server_tool_use_started" => events::Event::ServerToolUseStarted {
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
            "server_tool_arguments_delta" => {
                events::Event::ServerToolArgumentsDelta { index, delta: text }
            }
            "server_tool_use_completed" => events::Event::ServerToolUseCompleted {
                index,
                call: events::CompletedToolCall {
                    call_id: object
                        .get("call_id")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    namespace: None,
                    caller: None,
                    name: object
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    provider_item_id: None,
                    provider_status: None,
                    raw_arguments: object
                        .get("raw_arguments")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    custom: false,
                },
            },
            "server_tool_result" => events::Event::ServerToolResult {
                index,
                block: object
                    .get("block")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            },
            "hosted_tool_item_started" => events::Event::HostedToolItemStarted {
                output_index: object
                    .get("output_index")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0) as u32,
                item_id: object
                    .get("item_id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                item_type: object
                    .get("item_type")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                item: object
                    .get("item")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            },
            "hosted_tool_item_progress" => events::Event::HostedToolItemProgress {
                output_index: object
                    .get("output_index")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0) as u32,
                item_id: object
                    .get("item_id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                event_type: object
                    .get("event_type")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                payload: object
                    .get("payload")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            },
            "hosted_tool_item_completed" => events::Event::HostedToolItemCompleted {
                output_index: object
                    .get("output_index")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0) as u32,
                item_id: object
                    .get("item_id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                item_type: object
                    .get("item_type")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                item: object
                    .get("item")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            },
            "provider_text_annotation" => events::Event::ProviderTextAnnotation {
                output_index: object
                    .get("output_index")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0) as u32,
                item_id: object
                    .get("item_id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                annotation: object
                    .get("annotation")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            },
            "text_block_started" => events::Event::TextBlockStarted { index },
            "citation_delta" => events::Event::CitationDelta {
                index,
                citation: object
                    .get("citation")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            },
            "completed" => events::Event::Completed,
            "incomplete" => events::Event::Incomplete,
            "paused_turn" => events::Event::PausedTurn,
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
    module.add_class::<ShutdownHandle>()?;
    module.add_function(wrap_pyfunction!(shutdown_handle, module)?)?;
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
