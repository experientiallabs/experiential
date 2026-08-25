//! The native Anthropic Messages surface: the `/v1/messages` handler, the
//! explicit `count_tokens` refusal, the Anthropic error envelope, and the
//! Messages-shaped settled, aggregated, guarded, and live-streaming response
//! paths. The Anthropic protocol defines no idempotency header, so this
//! surface never joins the keyed replay stores, matching the python engine.

use std::sync::atomic::Ordering;
use std::time::Instant;

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::admission::{
    acquire_permit, apply_output_guardrail, commit_dependent, commit_independent, new_guard,
    Admission,
};
use crate::encode::compact_json;
use crate::encode_messages::{anthropic_error_body, completed_messages_body, MessagesSseEncoder};
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{Event, Usage};
use crate::metrics::{classify_escalation, METRICS};
use crate::relay::{collect_committed, collection_public_error, track_event};
use crate::respond::{
    bearer_key, complete_visible_refusal, escalation_error, json_response, read_body, send_bounded,
    settle_stream_end, sse_body_response,
};
use crate::server::AppState;
use crate::settlement::AttemptGuard;
use crate::waterfall::{acquire_attempt, CommittedAttempt, SettledAttempt, WaterfallContext, Won};

/// Anthropic-enveloped variant of `error_response` for the Messages surface,
/// mirroring `anthropic_error_response` in the python engine.
fn messages_error_response(error: &PublicError) -> Response {
    let mut builder = Response::builder()
        .status(
            StatusCode::from_u16(error.status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        )
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(wait) = error.retry_after_seconds {
        builder = builder.header(header::RETRY_AFTER, wait.to_string());
    }
    builder
        .body(Body::from(compact_json(&anthropic_error_body(error))))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

/// Anthropic callers present `x-api-key` (their SDK default) or a standard
/// Bearer header; both carry the same virtual key, mirroring the python
/// engine's `presented_api_key`.
fn messages_api_key(headers: &HeaderMap) -> Result<String, PublicError> {
    if let Some(value) = headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
    {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }
    bearer_key(headers).map_err(|_| {
        PublicError::new(
            401,
            "invalid_key",
            "A valid API key is required: send x-api-key or Authorization: Bearer.",
            "authentication_error",
        )
    })
}

/// Refuse token counting in the caller's own envelope. Anthropic clients
/// probe this endpoint; the gateway has no tokenizer authority to answer
/// truthfully, so it refuses explicitly in Anthropic shape and clients fall
/// back to their local estimate.
pub(crate) async fn messages_count_tokens() -> Response {
    messages_error_response(&PublicError::new(
        404,
        "route_not_served",
        "count_tokens is not served by this gateway.",
        "invalid_request_error",
    ))
}

pub(crate) async fn messages(
    State(state): State<AppState>,
    request: axum::extract::Request,
) -> Response {
    state.handled_requests.fetch_add(1, Ordering::Relaxed);
    let started = Instant::now();
    let deadline = started + state.request_timeout;
    let (parts, raw_body) = request.into_parts();
    let headers = parts.headers;
    let body = match read_body(raw_body).await {
        Ok(body) => body,
        Err(error) => return messages_error_response(&error),
    };

    let raw_key = match messages_api_key(&headers) {
        Ok(key) => key,
        Err(error) => return messages_error_response(&error),
    };
    let authenticate = compact_json(&json!({"raw_key": raw_key}));
    if let Err(error) = state.bridge.call("authenticate", authenticate).await {
        return messages_error_response(&error);
    }

    // The Anthropic protocol defines no idempotency header, so this surface
    // deliberately ignores `Idempotency-Key` and `X-Client-Request-Id` and
    // never joins the keyed replay stores, matching the python engine.
    let body_text = match String::from_utf8(body.to_vec()) {
        Ok(text) => text,
        Err(_) => return messages_error_response(&PublicError::invalid_json()),
    };

    let admit_argument = compact_json(&json!({
        "raw_key": raw_key,
        "body": body_text,
        "surface": "messages",
    }));
    let admission_text = match state.bridge.call("admit", admit_argument).await {
        Ok(text) => text,
        Err(error) => return messages_error_response(&error),
    };
    let admission_value: Value = match serde_json::from_str(&admission_text) {
        Ok(value) => value,
        Err(_) => return messages_error_response(&PublicError::internal()),
    };
    if let Some(reason) = admission_value.get("escalate") {
        METRICS.record_escalation(classify_escalation(reason.as_str().unwrap_or_default()));
        // No ledger row exists; startup validation guarantees native
        // servability, so an escalation disposition fails closed here.
        return messages_error_response(&escalation_error());
    }
    let admission: Admission = match serde_json::from_value(admission_value.clone()) {
        Ok(admission) => admission,
        Err(_) => {
            // The request is durably accepted; abandon it before failing so
            // wire-contract drift cannot leak an open request row.
            return messages_wire_drift_response(&state, &admission_value, started).await;
        }
    };
    let mut guard = new_guard(&state, admission.request_id.clone(), started);

    let permit = match acquire_permit(&state, &mut guard, deadline).await {
        Ok(permit) => permit,
        Err(response) => return *response,
    };

    // Run the certified waterfall to its committed or terminal attempt.
    let context = WaterfallContext {
        bridge: &state.bridge,
        http: &state.http,
        request_id: &admission.request_id,
        raw_key: &raw_key,
        route: &admission.route,
        policy: admission.policy(),
        deadline,
    };
    let won = acquire_attempt(&context, &mut guard).await;

    match won {
        Won::Failed(error) => messages_error_response(&error),
        Won::Settled(settled) => settled_messages_response(&admission, settled).await,
        Won::Committed(committed) => {
            let committed = *committed;
            if admission.output_guardrail {
                guarded_messages(admission, guard, committed, deadline, permit).await
            } else if admission.stream {
                stream_messages(admission, guard, committed, deadline, permit).await
            } else {
                completed_messages(admission, guard, committed, deadline, permit).await
            }
        }
    }
}

/// Abandon a durably accepted request whose admission reply failed to parse,
/// mirroring `wire_drift_response` for the Messages surface's enveloped
/// error shape.
async fn messages_wire_drift_response(
    state: &AppState,
    admission_value: &Value,
    started: Instant,
) -> Response {
    let request_id = admission_value
        .get("request_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if !request_id.is_empty() {
        let mut guard = new_guard(state, request_id, started);
        guard
            .abandon(&Failure::new(
                FailureClass::Internal,
                "gateway admission wire contract failed",
            ))
            .await;
    }
    messages_error_response(&PublicError::internal())
}

/// Answer one attempt that the waterfall already settled: a successful
/// terminal with no semantic output, or an exhausted ladder flushing its
/// bounded withheld refusal output ahead of the failing terminal.
async fn settled_messages_response(admission: &Admission, settled: SettledAttempt) -> Response {
    let mut events = settled.events;
    let refusal_completed = complete_visible_refusal(&mut events);
    if refusal_completed.is_none() {
        if let Some(Event::Failed(failure)) = events.last() {
            let error = collection_public_error(&failure.clone().boundary());
            if admission.stream {
                // The withheld refusal output and its failing terminal flush
                // outward as the stream's only frames.
                let body = match encode_messages_sse(admission, &events) {
                    Ok(body) => body,
                    Err(error) => return messages_error_response(&error),
                };
                let mut headers = commit_independent(admission, None);
                headers.extend(commit_dependent(admission, settled.depth));
                return sse_body_response(&headers, body);
            }
            return messages_error_response(&error);
        }
    }
    let mut headers = commit_independent(admission, None);
    headers.extend(commit_dependent(admission, settled.depth));
    if admission.stream {
        let body = match encode_messages_sse(admission, &events) {
            Ok(body) => body,
            Err(error) => return messages_error_response(&error),
        };
        return sse_body_response(&headers, body);
    }
    let aggregated = match completed_messages_body(&admission.request_id, &admission.alias, &events)
    {
        Ok(aggregated) => aggregated,
        Err(error) => return messages_error_response(&error),
    };
    if let Some(failure) = &aggregated.failure {
        return messages_error_response(&failure.clone().boundary().public_error());
    }
    json_response(StatusCode::OK, &aggregated.body, &headers)
}

/// Aggregate one committed non-streaming or guarded Messages attempt and
/// answer it, settling exactly once.
#[allow(clippy::too_many_arguments)]
async fn respond_from_messages_events(
    admission: Admission,
    mut guard: AttemptGuard,
    depth: usize,
    mut events: Vec<Event>,
    usage: Option<Usage>,
    tool_names: Vec<String>,
    stream_body: bool,
) -> Response {
    let refusal_completed = complete_visible_refusal(&mut events);
    let aggregated = match completed_messages_body(&admission.request_id, &admission.alias, &events)
    {
        Ok(aggregated) => aggregated,
        Err(error) => {
            guard
                .settle(
                    "failed",
                    usage.as_ref(),
                    &tool_names,
                    Some(
                        &Failure::new(
                            FailureClass::MalformedResponse,
                            "provider stream ended without a terminal event",
                        )
                        .boundary(),
                    ),
                    true,
                )
                .await;
            return messages_error_response(&error);
        }
    };
    if let Some(failure) = &aggregated.failure {
        let failure = failure.clone().boundary();
        let error = failure.public_error();
        guard
            .settle(
                "failed",
                aggregated.usage.as_ref().or(usage.as_ref()),
                &aggregated.tool_names,
                Some(&failure),
                true,
            )
            .await;
        return messages_error_response(&error);
    }
    let settled = if let Some(refusal) = &refusal_completed {
        // The caller saw the refusal output, so the public result completes;
        // the ledger still records the provider's typed refusal.
        guard
            .settle(
                "failed",
                aggregated.usage.as_ref().or(usage.as_ref()),
                &aggregated.tool_names,
                Some(refusal),
                true,
            )
            .await
    } else {
        let outcome = if aggregated.incomplete {
            "incomplete"
        } else {
            "completed"
        };
        guard
            .settle(
                outcome,
                aggregated.usage.as_ref().or(usage.as_ref()),
                &aggregated.tool_names,
                None,
                true,
            )
            .await
    };
    if !settled {
        // Success is only reported once the terminal accounting write landed.
        return messages_error_response(&PublicError::internal());
    }
    let mut headers = commit_independent(&admission, None);
    headers.extend(commit_dependent(&admission, depth));
    if stream_body {
        let body = match encode_messages_sse(&admission, &events) {
            Ok(body) => body,
            Err(error) => return messages_error_response(&error),
        };
        return sse_body_response(&headers, body);
    }
    json_response(StatusCode::OK, &aggregated.body, &headers)
}

fn encode_messages_sse(admission: &Admission, events: &[Event]) -> Result<Vec<u8>, PublicError> {
    let mut encoder = MessagesSseEncoder::new(&admission.request_id, &admission.alias);
    let mut body = Vec::new();
    for frame in encoder.start()? {
        body.extend_from_slice(frame.as_bytes());
    }
    for event in events {
        for frame in encoder.feed(event)? {
            body.extend_from_slice(frame.as_bytes());
        }
    }
    Ok(body)
}

async fn completed_messages(
    admission: Admission,
    mut guard: AttemptGuard,
    mut committed: CommittedAttempt,
    deadline: Instant,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    let _permit = permit;
    let phase_timeout = admission.phase_timeout(committed.depth);
    let events =
        match collect_committed(&mut committed, deadline, phase_timeout, guard.started).await {
            Ok(events) => events,
            Err(failure) => {
                let failure = failure.boundary();
                let error = collection_public_error(&failure);
                guard
                    .settle(
                        "failed",
                        committed.usage.as_ref(),
                        &committed.tool_names,
                        Some(&failure),
                        true,
                    )
                    .await;
                return messages_error_response(&error);
            }
        };
    respond_from_messages_events(
        admission,
        guard,
        committed.depth,
        events,
        committed.usage,
        committed.tool_names,
        false,
    )
    .await
}

async fn guarded_messages(
    admission: Admission,
    mut guard: AttemptGuard,
    mut committed: CommittedAttempt,
    deadline: Instant,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    let _permit = permit;
    let phase_timeout = admission.phase_timeout(committed.depth);
    let collected =
        match collect_committed(&mut committed, deadline, phase_timeout, guard.started).await {
            Ok(events) => events,
            Err(failure) => {
                let failure = failure.boundary();
                let error = collection_public_error(&failure);
                guard
                    .settle(
                        "failed",
                        committed.usage.as_ref(),
                        &committed.tool_names,
                        Some(&failure),
                        true,
                    )
                    .await;
                return messages_error_response(&error);
            }
        };
    let events = match apply_output_guardrail(&admission, &guard.bridge, collected).await {
        Ok(events) => events,
        Err(failure) => {
            guard
                .settle(
                    "failed",
                    committed.usage.as_ref(),
                    &committed.tool_names,
                    Some(&failure),
                    true,
                )
                .await;
            return messages_error_response(&failure.public_error());
        }
    };
    let stream_body = admission.stream;
    respond_from_messages_events(
        admission,
        guard,
        committed.depth,
        events,
        committed.usage,
        committed.tool_names,
        stream_body,
    )
    .await
}

async fn stream_messages(
    admission: Admission,
    guard: AttemptGuard,
    committed: CommittedAttempt,
    deadline: Instant,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> Response {
    let (sender, receiver) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    let header_pairs = {
        let mut headers = commit_independent(&admission, None);
        headers.extend(commit_dependent(&admission, committed.depth));
        headers
    };
    let request_id = admission.request_id.clone();
    let alias = admission.alias.clone();
    let phase_timeout = admission.phase_timeout(committed.depth);
    let task_hold = guard.hold_task();
    tokio::spawn(async move {
        let _task = task_hold;
        let _permit = permit;
        let mut guard = guard;
        let mut committed = committed;
        let mut encoder = MessagesSseEncoder::new(&request_id, &alias);
        let mut usage: Option<Usage> = committed.usage.take();
        let mut tool_names: Vec<String> = std::mem::take(&mut committed.tool_names);
        let mut visible_refusal = committed.visible_refusal;
        let mut terminal: Option<Event> = None;

        macro_rules! fail_stream {
            ($failure:expr) => {{
                let failure = $failure.boundary();
                emit_messages_failure(&sender, deadline, &mut encoder, &failure).await;
                guard
                    .settle("failed", usage.as_ref(), &tool_names, Some(&failure), true)
                    .await;
                return;
            }};
        }

        let start_frames = match encoder.start() {
            Ok(frames) => frames,
            Err(_) => {
                fail_stream!(Failure::new(
                    FailureClass::Internal,
                    "gateway could not encode the provider stream",
                ))
            }
        };
        for frame in start_frames {
            if !send_bounded(&sender, deadline, Bytes::from(frame)).await {
                guard.settle_cancelled(usage.as_ref(), &tool_names).await;
                return;
            }
        }

        let mut prefix: std::collections::VecDeque<Event> = committed.prefix.drain(..).collect();
        loop {
            let event = if let Some(event) = prefix.pop_front() {
                event
            } else {
                match committed
                    .relay
                    .next_event(deadline, phase_timeout, guard.started)
                    .await
                {
                    Ok(Some(event)) => event,
                    Ok(None) => {
                        fail_stream!(Failure::new(
                            FailureClass::MalformedResponse,
                            "provider stream ended without a terminal event",
                        ))
                    }
                    Err(failure) => fail_stream!(failure),
                }
            };
            track_event(&event, &mut usage, &mut tool_names);
            if matches!(event, Event::RefusalDelta(_)) {
                visible_refusal = true;
            }
            // A typed refusal after visible refusal output completes
            // publicly; the ledger still records the provider's refusal.
            let outward = match &event {
                Event::Failed(failure)
                    if failure.failure_class == FailureClass::Refusal && visible_refusal =>
                {
                    Event::Completed
                }
                other => other.clone(),
            };
            if event.is_terminal() {
                terminal = Some(event.clone());
                if !settle_stream_end(&mut guard, Some(&event), usage.as_ref(), &tool_names, false)
                    .await
                {
                    return;
                }
            }
            let encoded = match encoder.feed(&outward) {
                Ok(encoded) => encoded,
                Err(_) => {
                    if terminal.is_some() {
                        // The attempt already settled by its provider
                        // terminal; the stream simply ends short.
                        return;
                    }
                    fail_stream!(Failure::new(
                        FailureClass::Internal,
                        "gateway could not encode the provider stream",
                    ))
                }
            };
            for data in encoded {
                if !send_bounded(&sender, deadline, Bytes::from(data)).await {
                    settle_stream_end(
                        &mut guard,
                        terminal.as_ref(),
                        usage.as_ref(),
                        &tool_names,
                        true,
                    )
                    .await;
                    return;
                }
            }
            if terminal.is_some() {
                return;
            }
        }
    });

    let body = Body::from_stream(ReceiverStream::new(receiver));
    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream");
    for (name, value) in &header_pairs {
        if let (Ok(name), Ok(value)) = (
            header::HeaderName::try_from(name.as_str()),
            HeaderValue::try_from(value.as_str()),
        ) {
            builder = builder.header(name, value);
        }
    }
    builder
        .body(body)
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

/// Emit the Messages encoder's sanitized failure lifecycle (one Anthropic
/// `error` event) when the stream has not already reached a terminal.
async fn emit_messages_failure(
    sender: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    deadline: Instant,
    encoder: &mut MessagesSseEncoder,
    failure: &Failure,
) {
    if encoder.saw_terminal() {
        return;
    }
    let frames = encoder
        .feed(&Event::Failed(failure.clone()))
        .unwrap_or_default();
    for frame in frames {
        if !send_bounded(sender, deadline, Bytes::from(frame)).await {
            return;
        }
    }
}
