//! The native OpenAI Responses surface: the `/v1/responses` handler with its
//! keyed-replay protocol, bounded continuation retention through the control
//! plane's `remember`, and the Responses-shaped settled, aggregated, guarded,
//! and live-streaming response paths.

use std::sync::atomic::Ordering;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::admission::{
    acquire_permit, apply_output_guardrail, commit_dependent, commit_independent, new_guard,
    wire_drift_response, Admission,
};
use crate::dialects::MAXIMUM_RETAINED_OUTPUT_BYTES;
use crate::encode::compact_json;
use crate::encode_responses::{completed_responses_body, ResponsesSseEncoder};
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{CompletedToolCall, Event, Usage};
use crate::metrics::{classify_escalation, METRICS};
use crate::relay::{collect_committed, collection_public_error, event_retained_bytes, track_event};
use crate::replay::{CachedResponse, Claim, OwnerLease, ReplayKey};
use crate::respond::{
    bearer_key, cached_response, capture_frame, complete_visible_refusal, error_response,
    escalation_error, finish_stream_terminal, json_response, latin1_header, read_body,
    send_bounded, settle_stream_end, sse_body_response,
};
use crate::server::AppState;
use crate::settlement::AttemptGuard;
use crate::waterfall::{acquire_attempt, CommittedAttempt, SettledAttempt, WaterfallContext, Won};

pub(crate) async fn responses(
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
        Err(error) => return error_response(&error),
    };

    let raw_key = match bearer_key(&headers) {
        Ok(key) => key,
        Err(error) => return error_response(&error),
    };
    let authenticate = compact_json(&json!({"raw_key": raw_key}));
    if let Err(error) = state.bridge.call("authenticate", authenticate).await {
        return error_response(&error);
    }

    let body_text = match String::from_utf8(body.to_vec()) {
        Ok(text) => text,
        Err(_) => return error_response(&PublicError::invalid_json()),
    };

    // Replay-keyed Responses runs the python engine's exact idempotency
    // protocol natively, sharing the same bounded replay store and
    // tenant-scoped key derivation the chat surface uses (the surface is
    // part of the key, so chat and Responses operations never collide).
    let idempotency_key = latin1_header(&headers, "idempotency-key");
    let client_request_id = latin1_header(&headers, "x-client-request-id");
    let mut lease: Option<OwnerLease> = None;
    if idempotency_key.is_some() || client_request_id.is_some() {
        let scope_argument = compact_json(&json!({
            "raw_key": raw_key,
            "body": body_text,
            "surface": "responses",
            "idempotency_key": idempotency_key,
            "client_request_id": client_request_id,
        }));
        let scope_text = match state.bridge.call("claim_scope", scope_argument).await {
            Ok(text) => text,
            Err(error) => return error_response(&error),
        };
        let scope_value: Value = match serde_json::from_str(&scope_text) {
            Ok(value) => value,
            Err(_) => return error_response(&PublicError::internal()),
        };
        if let Some(reason) = scope_value.get("escalate") {
            METRICS.record_escalation(classify_escalation(reason.as_str().unwrap_or_default()));
            // No replay claim exists; startup validation guarantees native
            // servability, so an escalation disposition fails closed here.
            return error_response(&escalation_error());
        }
        let key: ReplayKey = match serde_json::from_value(scope_value) {
            Ok(key) => key,
            Err(_) => return error_response(&PublicError::internal()),
        };
        match state.replays.claim(key).await {
            Err(error) => return error_response(&error),
            Ok(Claim::Replay(cached)) => return cached_response(&cached),
            Ok(Claim::Join(joiner)) => {
                // Joining never touches the ledger or budget: only the owner
                // accounts for the single provider call.
                return match joiner.result().await {
                    Ok(cached) => cached_response(&cached),
                    Err(error) => error_response(&error),
                };
            }
            Ok(Claim::Owner(owner)) => lease = Some(owner),
        }
    }

    let admit_argument = compact_json(&json!({
        "raw_key": raw_key,
        "body": body_text,
        "surface": "responses",
        "idempotency_key": idempotency_key,
        "client_request_id": client_request_id,
    }));
    let admission_text = match state.bridge.call("admit", admit_argument).await {
        Ok(text) => text,
        Err(error) => {
            // A failed keyed admission abandons ownership so waiting
            // duplicates fail closed instead of hanging.
            if let Some(mut owner) = lease.take() {
                owner.abandon().await;
            }
            return error_response(&error);
        }
    };
    let admission_value: Value = match serde_json::from_str(&admission_text) {
        Ok(value) => value,
        Err(_) => return error_response(&PublicError::internal()),
    };
    if let Some(reason) = admission_value.get("escalate") {
        METRICS.record_escalation(classify_escalation(reason.as_str().unwrap_or_default()));
        // No ledger row exists; startup validation guarantees native
        // servability, so an escalation disposition fails closed here.
        if let Some(mut owner) = lease.take() {
            owner.abandon().await;
        }
        return error_response(&escalation_error());
    }
    let admission: Admission = match serde_json::from_value(admission_value.clone()) {
        Ok(admission) => admission,
        Err(_) => {
            if let Some(mut owner) = lease.take() {
                owner.abandon().await;
            }
            return wire_drift_response(&state, &admission_value, started).await;
        }
    };
    let mut guard = new_guard(&state, admission.request_id.clone(), started);
    // The replay key was authorized independently of admission; a revision
    // swap between the two fails closed exactly like the chat surface.
    if lease
        .as_ref()
        .is_some_and(|owner| owner.alias_revision_id() != admission.alias_revision_id)
    {
        if let Some(mut owner) = lease.take() {
            owner.abandon().await;
        }
        guard
            .abandon(&Failure::new(
                FailureClass::Internal,
                "the alias revision changed during keyed admission",
            ))
            .await;
        let mut error = PublicError::new(
            409,
            "idempotency_replay_unavailable",
            "The alias revision changed while the keyed request was admitted. Retry the request.",
            "api_error",
        );
        error.param = Some("Idempotency-Key".to_string());
        return error_response(&error);
    }

    let permit = match acquire_permit(&state, &mut guard, deadline).await {
        Ok(permit) => permit,
        Err(response) => {
            if let Some(mut owner) = lease.take() {
                owner.abandon().await;
            }
            return *response;
        }
    };

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

    // Responses envelopes carry a float wall clock, like the python encoder.
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs_f64())
        .unwrap_or(0.0);

    match won {
        Won::Failed(error) => {
            if let Some(mut owner) = lease.take() {
                owner.abandon().await;
            }
            error_response(&error)
        }
        Won::Settled(settled) => {
            settled_responses_response(&admission, settled, created_at, lease, client_request_id)
                .await
        }
        Won::Committed(committed) => {
            let committed = *committed;
            if admission.output_guardrail {
                guarded_responses(
                    state,
                    admission,
                    guard,
                    committed,
                    created_at,
                    deadline,
                    permit,
                    lease,
                    client_request_id,
                )
                .await
            } else if admission.stream {
                stream_responses(
                    state.clone(),
                    admission,
                    guard,
                    committed,
                    created_at,
                    deadline,
                    permit,
                    lease,
                    client_request_id,
                )
                .await
            } else {
                completed_responses(
                    &state,
                    admission,
                    guard,
                    committed,
                    created_at,
                    deadline,
                    permit,
                    lease,
                    client_request_id,
                )
                .await
            }
        }
    }
}

/// Aggregated assistant output tracked while relaying one Responses stream,
/// mirroring `assistant_message` inputs for continuation retention.
#[derive(Default)]
struct ResponsesRetention {
    text: String,
    refusal: bool,
    tool_calls: Vec<CompletedToolCall>,
    retained_bytes: usize,
    overflowed: bool,
}

impl ResponsesRetention {
    fn track(&mut self, event: &Event) {
        if self.overflowed {
            return;
        }
        self.retained_bytes = self
            .retained_bytes
            .saturating_add(event_retained_bytes(event));
        if self.retained_bytes > MAXIMUM_RETAINED_OUTPUT_BYTES {
            // Retention is bounded like the python service's aggregation:
            // the stream keeps flowing but nothing oversize is remembered.
            self.overflowed = true;
            self.text.clear();
            self.tool_calls.clear();
            return;
        }
        match event {
            Event::TextDelta(delta) => self.text.push_str(delta),
            Event::RefusalDelta(_) => self.refusal = true,
            Event::ToolCallCompleted { call, .. } => self.tool_calls.push(call.clone()),
            _ => {}
        }
    }
}

/// Build the retention payload consumed by the control plane's `remember`.
fn remember_argument(request_id: &str, retention: &ResponsesRetention) -> String {
    compact_json(&json!({
        "request_id": request_id,
        "text": retention.text,
        "refusal": retention.refusal,
        "tool_calls": retention
            .tool_calls
            .iter()
            .map(|call| json!({
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.raw_arguments,
            }))
            .collect::<Vec<Value>>(),
    }))
}

/// Retain one completed Responses continuation before the terminal frames
/// flush, mirroring the python service's ordering. Returns the public error
/// when bounded retention fails closed.
async fn remember_continuation(
    state: &AppState,
    request_id: &str,
    retention: &ResponsesRetention,
) -> Result<(), PublicError> {
    if retention.overflowed
        || retention.refusal
        || (retention.text.is_empty() && retention.tool_calls.is_empty())
    {
        return Ok(());
    }
    state
        .bridge
        .call("remember", remember_argument(request_id, retention))
        .await
        .map(|_| ())
}

/// Answer one Responses attempt that the waterfall already settled: a
/// successful terminal with no semantic output, or an exhausted ladder
/// flushing withheld refusal output ahead of the failing terminal.
async fn settled_responses_response(
    admission: &Admission,
    settled: SettledAttempt,
    created_at: f64,
    mut lease: Option<OwnerLease>,
    client_request_id: Option<String>,
) -> Response {
    let mut events = settled.events;
    let refusal_completed = complete_visible_refusal(&mut events);
    let failed = refusal_completed.is_none() && matches!(events.last(), Some(Event::Failed(_)));
    if failed && !admission.stream {
        if let Some(mut owner) = lease.take() {
            owner.abandon().await;
        }
        if let Some(Event::Failed(failure)) = events.last() {
            return error_response(&collection_public_error(&failure.clone().boundary()));
        }
    }
    let mut headers = commit_independent(admission, client_request_id.as_deref());
    headers.extend(commit_dependent(admission, settled.depth));
    if admission.stream {
        let body = match encode_responses_sse(admission, created_at, &events) {
            Ok(body) => body,
            Err(error) => return error_response(&error),
        };
        if failed {
            // A failed flush is not a replayable success.
            if let Some(mut owner) = lease.take() {
                owner.abandon().await;
            }
            return sse_body_response(&headers, body);
        }
        if let Some(mut owner) = lease.take() {
            let mut sorted = headers.clone();
            sorted.sort();
            let cached = CachedResponse {
                status_code: 200,
                media_type: "text/event-stream; charset=utf-8".to_string(),
                headers: sorted,
                body: body.clone(),
            };
            return match owner.complete(cached.clone()).await {
                Ok(()) => cached_response(&cached),
                Err(error) => error_response(&error),
            };
        }
        return sse_body_response(&headers, body);
    }
    let envelope = admission.envelope.clone().unwrap_or_default();
    let aggregated = match completed_responses_body(
        &admission.request_id,
        &admission.alias,
        created_at,
        envelope,
        &events,
    ) {
        Ok(aggregated) => aggregated,
        Err(error) => return error_response(&error),
    };
    if let Some(failure) = &aggregated.failure {
        if let Some(mut owner) = lease.take() {
            owner.abandon().await;
        }
        return error_response(&failure.clone().boundary().public_error());
    }
    if let Some(mut owner) = lease.take() {
        let mut sorted = headers.clone();
        sorted.sort();
        let cached = CachedResponse {
            status_code: 200,
            media_type: "application/json".to_string(),
            headers: sorted,
            body: compact_json(&aggregated.body).into_bytes(),
        };
        return match owner.complete(cached.clone()).await {
            Ok(()) => cached_response(&cached),
            Err(error) => error_response(&error),
        };
    }
    json_response(StatusCode::OK, &aggregated.body, &headers)
}

/// Aggregate one committed Responses attempt and answer it, retaining the
/// continuation and settling exactly once.
#[allow(clippy::too_many_arguments)]
async fn respond_from_responses_events(
    state: &AppState,
    admission: Admission,
    mut guard: AttemptGuard,
    depth: usize,
    mut events: Vec<Event>,
    usage: Option<Usage>,
    tool_names: Vec<String>,
    created_at: f64,
    mut lease: Option<OwnerLease>,
    client_request_id: Option<String>,
    stream_body: bool,
) -> Response {
    let refusal_completed = complete_visible_refusal(&mut events);
    let envelope = admission.envelope.clone().unwrap_or_default();
    let aggregated = match completed_responses_body(
        &admission.request_id,
        &admission.alias,
        created_at,
        envelope,
        &events,
    ) {
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
            if let Some(mut owner) = lease.take() {
                owner.abandon().await;
            }
            return error_response(&error);
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
        if let Some(mut owner) = lease.take() {
            owner.abandon().await;
        }
        return error_response(&error);
    }
    // Retention runs while the attempt row is still in flight so the control
    // plane can resolve its namespaced continuation context, and before the
    // body is answered so an oversize continuation fails closed like python.
    let retention = ResponsesRetention {
        text: aggregated.text.clone(),
        refusal: !aggregated.refusal.is_empty() || refusal_completed.is_some(),
        tool_calls: aggregated.tool_calls.clone(),
        ..ResponsesRetention::default()
    };
    let remembered = remember_continuation(state, &admission.request_id, &retention).await;
    let settled = if let Some(refusal) = &refusal_completed {
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
    if let Err(error) = remembered {
        // The provider outcome already settled above, exactly like the python
        // executor; only the HTTP result reports the retention failure.
        if let Some(mut owner) = lease.take() {
            owner.abandon().await;
        }
        return error_response(&error);
    }
    if !settled {
        // Success is only reported once the terminal accounting write landed.
        if let Some(mut owner) = lease.take() {
            owner.abandon().await;
        }
        return error_response(&PublicError::internal());
    }
    let mut headers = commit_independent(&admission, client_request_id.as_deref());
    headers.extend(commit_dependent(&admission, depth));
    if stream_body {
        let body = match encode_responses_sse(&admission, created_at, &events) {
            Ok(body) => body,
            Err(error) => return error_response(&error),
        };
        if let Some(mut owner) = lease.take() {
            let mut sorted = headers.clone();
            sorted.sort();
            let cached = CachedResponse {
                status_code: 200,
                media_type: "text/event-stream; charset=utf-8".to_string(),
                headers: sorted,
                body: body.clone(),
            };
            return match owner.complete(cached.clone()).await {
                Ok(()) => cached_response(&cached),
                Err(error) => error_response(&error),
            };
        }
        return sse_body_response(&headers, body);
    }
    if let Some(mut owner) = lease.take() {
        let mut sorted = headers.clone();
        sorted.sort();
        let cached = CachedResponse {
            status_code: 200,
            media_type: "application/json".to_string(),
            headers: sorted,
            body: compact_json(&aggregated.body).into_bytes(),
        };
        return match owner.complete(cached.clone()).await {
            Ok(()) => cached_response(&cached),
            Err(error) => error_response(&error),
        };
    }
    json_response(StatusCode::OK, &aggregated.body, &headers)
}

#[allow(clippy::too_many_arguments)]
async fn completed_responses(
    state: &AppState,
    admission: Admission,
    mut guard: AttemptGuard,
    mut committed: CommittedAttempt,
    created_at: f64,
    deadline: Instant,
    permit: tokio::sync::OwnedSemaphorePermit,
    mut lease: Option<OwnerLease>,
    client_request_id: Option<String>,
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
                if let Some(mut owner) = lease.take() {
                    owner.abandon().await;
                }
                return error_response(&error);
            }
        };
    respond_from_responses_events(
        state,
        admission,
        guard,
        committed.depth,
        events,
        committed.usage,
        committed.tool_names,
        created_at,
        lease,
        client_request_id,
        false,
    )
    .await
}

fn encode_responses_sse(
    admission: &Admission,
    created_at: f64,
    events: &[Event],
) -> Result<Vec<u8>, PublicError> {
    let envelope = admission.envelope.clone().unwrap_or_default();
    let mut encoder = ResponsesSseEncoder::new(
        &admission.request_id,
        &admission.alias,
        created_at,
        envelope,
    );
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

#[allow(clippy::too_many_arguments)]
async fn guarded_responses(
    state: AppState,
    admission: Admission,
    mut guard: AttemptGuard,
    mut committed: CommittedAttempt,
    created_at: f64,
    deadline: Instant,
    permit: tokio::sync::OwnedSemaphorePermit,
    mut lease: Option<OwnerLease>,
    client_request_id: Option<String>,
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
                if let Some(mut owner) = lease.take() {
                    owner.abandon().await;
                }
                return error_response(&error);
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
            if let Some(mut owner) = lease.take() {
                owner.abandon().await;
            }
            return error_response(&failure.public_error());
        }
    };
    let stream_body = admission.stream;
    respond_from_responses_events(
        &state,
        admission,
        guard,
        committed.depth,
        events,
        committed.usage,
        committed.tool_names,
        created_at,
        lease,
        client_request_id,
        stream_body,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn stream_responses(
    state: AppState,
    admission: Admission,
    guard: AttemptGuard,
    committed: CommittedAttempt,
    created_at: f64,
    deadline: Instant,
    permit: tokio::sync::OwnedSemaphorePermit,
    lease: Option<OwnerLease>,
    client_request_id: Option<String>,
) -> Response {
    let (sender, receiver) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    let mut header_pairs = commit_independent(&admission, client_request_id.as_deref());
    header_pairs.extend(commit_dependent(&admission, committed.depth));
    let request_id = admission.request_id.clone();
    let alias = admission.alias.clone();
    let envelope = admission.envelope.clone().unwrap_or_default();
    let phase_timeout = admission.phase_timeout(committed.depth);
    let cached_headers = {
        let mut sorted = header_pairs.clone();
        sorted.sort();
        sorted
    };
    let task_hold = guard.hold_task();
    tokio::spawn(async move {
        let _task = task_hold;
        let _permit = permit;
        let mut guard = guard;
        let mut committed = committed;
        let mut lease = lease;
        // Keyed streams capture every public frame so the owner can publish
        // the exact byte stream; terminal frames flow through the shared
        // publication tail, matching the chat surface.
        let mut capture: Vec<u8> = Vec::new();
        let mut replayable = lease.is_some();
        let mut encoder = ResponsesSseEncoder::new(&request_id, &alias, created_at, envelope);
        let mut usage: Option<Usage> = committed.usage.take();
        let mut tool_names: Vec<String> = std::mem::take(&mut committed.tool_names);
        let mut visible_refusal = committed.visible_refusal;
        let mut terminal: Option<Event> = None;
        let mut retention = ResponsesRetention::default();
        // Terminal frames are withheld until continuation retention lands,
        // mirroring the python stream body's ordering.
        let terminal_frames: Vec<String>;

        macro_rules! fail_stream {
            ($failure:expr) => {{
                let failure = $failure.boundary();
                emit_responses_failure(&sender, deadline, &mut encoder, &failure).await;
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
            let data = Bytes::from(frame);
            if lease.is_some() {
                replayable = capture_frame(&mut capture, &data, replayable);
            }
            if !send_bounded(&sender, deadline, data).await {
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
            retention.track(&event);
            if matches!(event, Event::RefusalDelta(_)) {
                visible_refusal = true;
            }
            let outward = match &event {
                Event::Failed(failure)
                    if failure.failure_class == FailureClass::Refusal && visible_refusal =>
                {
                    Event::Completed
                }
                other => other.clone(),
            };
            // The terminal is recorded before its frames flush, so a
            // disconnect during the final flush still settles by the
            // provider's outcome instead of as a cancellation.
            if event.is_terminal() {
                terminal = Some(event.clone());
            }
            let encoded = match encoder.feed(&outward) {
                Ok(encoded) => encoded,
                Err(_) => {
                    fail_stream!(Failure::new(
                        FailureClass::Internal,
                        "gateway could not encode the provider stream",
                    ))
                }
            };
            if terminal.is_some() {
                terminal_frames = encoded;
                break;
            }
            for data in encoded {
                let data = Bytes::from(data);
                if lease.is_some() {
                    replayable = capture_frame(&mut capture, &data, replayable);
                }
                if !send_bounded(&sender, deadline, data).await {
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
        }

        // Retention runs before the terminal frames flush; a bounded
        // retention failure truncates the stream before its terminal, the
        // same observable behavior as the python service.
        let retainable = !matches!(terminal, Some(Event::Failed(_)));
        if retainable {
            if let Err(_error) =
                remember_continuation(&state, &admission.request_id, &retention).await
            {
                settle_stream_end(
                    &mut guard,
                    terminal.as_ref(),
                    usage.as_ref(),
                    &tool_names,
                    false,
                )
                .await;
                return;
            }
        }
        // The durable settlement lands before any keyed publication so a
        // replayable success can never outlive a lost accounting write; a
        // failed terminal abandons ownership so duplicates fail closed.
        settle_stream_end(
            &mut guard,
            terminal.as_ref(),
            usage.as_ref(),
            &tool_names,
            false,
        )
        .await;
        if matches!(terminal, Some(Event::Failed(_))) {
            if let Some(mut owner) = lease.take() {
                owner.abandon().await;
            }
        }
        finish_stream_terminal(
            &sender,
            deadline,
            &mut lease,
            replayable,
            &mut capture,
            &cached_headers,
            terminal_frames.into_iter().map(Bytes::from).collect(),
        )
        .await;
    });

    let body = Body::from_stream(ReceiverStream::new(receiver));
    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8");
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

/// Emit the Responses encoder's sanitized failure lifecycle when the stream
/// has not already reached a terminal.
async fn emit_responses_failure(
    sender: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    deadline: Instant,
    encoder: &mut ResponsesSseEncoder,
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
