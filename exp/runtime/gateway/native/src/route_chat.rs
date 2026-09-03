//! The native OpenAI Chat Completions surface: the `/v1/chat/completions`
//! handler with its keyed-replay protocol, plus the chat-shaped settled,
//! aggregated, guarded, and live-streaming response paths.

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
use crate::encode::{
    chat_data, compact_json, completed_chat_body_with_carrier, completed_chat_body_with_ignored,
    reasoning_carrier_candidate, ChatSseEncoder, ReasoningCarrierCandidate,
};
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{Event, Usage};
use crate::metrics::{classify_escalation, METRICS};
use crate::relay::{collect_committed, collection_public_error, track_event};
use crate::replay::{CachedResponse, Claim, OwnerLease, ReplayKey};
use crate::respond::{
    bearer_key, cached_response, capture_frame, complete_visible_refusal, error_response,
    escalation_error, finish_stream_terminal, json_response, latin1_header, read_body,
    send_bounded, settle_stream_end, sse_body_response,
};
use crate::server::AppState;
use crate::settlement::AttemptGuard;
use crate::waterfall::{acquire_attempt, CommittedAttempt, SettledAttempt, WaterfallContext, Won};

pub(crate) async fn chat(
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

    // Replay-keyed chat runs the python engine's exact idempotency protocol
    // natively: the shared control plane computes the tenant-scoped replay
    // key (or escalates a request the native path cannot serve), then the
    // bounded replay store dedupes concurrent duplicates and replays the
    // owner's exact stored response. Headers are decoded latin-1 so any
    // HTTP-legal value matches the python engine's view byte for byte.
    // Only the standard Idempotency-Key opts into replay: callers reuse
    // x-client-request-id as a session correlation id across distinct
    // sequential requests, so it never keys an operation.
    let idempotency_key = latin1_header(&headers, "idempotency-key");
    let client_request_id = latin1_header(&headers, "x-client-request-id");
    let mut lease: Option<OwnerLease> = None;
    if idempotency_key.is_some() {
        let scope_argument = compact_json(&json!({
            "raw_key": raw_key,
            "body": body_text,
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
            // The request is durably accepted; abandon it before failing so
            // wire-contract drift cannot leak an open request row.
            if let Some(mut owner) = lease.take() {
                owner.abandon().await;
            }
            return wire_drift_response(&state, &admission_value, started).await;
        }
    };
    let mut guard = new_guard(&state, admission.request_id.clone(), started);
    // The replay key was authorized independently of admission. If an alias
    // activation landed between the two, the admitted work belongs to a newer
    // revision than the claimed replay scope, so the request fails closed:
    // executing without ownership would let a concurrent duplicate own the
    // new revision's key and run the same keyed operation a second time.
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

    // Run the certified waterfall to its committed or terminal attempt.
    let context = WaterfallContext {
        bridge: &state.bridge,
        http: &state.http,
        request_id: &admission.request_id,
        raw_key: &raw_key,
        route: &admission.route,
        policy: admission.policy(),
        deadline,
        time_to_first_byte: state.time_to_first_byte,
        time_to_first_byte_slope_seconds_per_million_input_tokens: state
            .time_to_first_byte_slope_seconds_per_million_input_tokens,
        // Bytes over four approximates input tokens; a timeout heuristic
        // only, never a billing quantity.
        approximate_input_tokens: (body_text.len() as f64) / 4.0,
        output_less_retention: None,
    };
    let won = acquire_attempt(&context, &mut guard).await;

    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or(0);

    match won {
        Won::Failed(error) => {
            if let Some(mut owner) = lease.take() {
                owner.abandon().await;
            }
            error_response(&error)
        }
        Won::Settled(settled) => {
            settled_chat_response(&admission, settled, created_at, lease, client_request_id).await
        }
        Won::Committed(committed) => {
            let committed = *committed;
            if admission.output_guardrail {
                guarded_chat_response(
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
                stream_response(
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
                completed_response(
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

/// Answer one attempt that the waterfall already settled: a successful
/// terminal with no semantic output, or an exhausted ladder flushing its
/// bounded withheld refusal output ahead of the failing terminal.
async fn settled_chat_response(
    admission: &Admission,
    settled: SettledAttempt,
    created_at: i64,
    mut lease: Option<OwnerLease>,
    client_request_id: Option<String>,
) -> Response {
    let mut events = settled.events;
    let refusal_completed = complete_visible_refusal(&mut events);
    if refusal_completed.is_none() {
        if let Some(Event::Failed(failure)) = events.last() {
            let error = collection_public_error(&failure.clone().boundary());
            if admission.stream {
                // The withheld refusal output and its failing terminal flush
                // outward as the stream's only frames.
                let body = match encode_chat_sse(admission, created_at, &events, None) {
                    Ok(body) => body,
                    Err(error) => return error_response(&error),
                };
                let mut headers = commit_independent(admission, client_request_id.as_deref());
                headers.extend(commit_dependent(admission, settled.depth));
                if let Some(mut owner) = lease.take() {
                    owner.abandon().await;
                }
                return sse_body_response(&headers, body);
            }
            if let Some(mut owner) = lease.take() {
                owner.abandon().await;
            }
            return error_response(&error);
        }
    }
    let mut headers = commit_independent(admission, client_request_id.as_deref());
    headers.extend(commit_dependent(admission, settled.depth));
    if admission.stream {
        let body = match encode_chat_sse(admission, created_at, &events, None) {
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
    let aggregated = match completed_chat_body_with_ignored(
        &admission.request_id,
        &admission.alias,
        created_at,
        &events,
        &admission.ignored_parameters,
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

fn encode_chat_sse(
    admission: &Admission,
    created_at: i64,
    events: &[Event],
    reasoning_content_carrier: Option<&str>,
) -> Result<Vec<u8>, PublicError> {
    let mut encoder = ChatSseEncoder::new_with_ignored(
        &admission.request_id,
        &admission.alias,
        created_at,
        admission.include_usage,
        admission.ignored_parameters.clone(),
    );
    if let Some(carrier) = reasoning_content_carrier {
        encoder.set_reasoning_content_carrier(carrier.to_string());
    }
    let mut body = Vec::new();
    for frame in encoder.start().map_err(|_| PublicError::internal())? {
        body.extend_from_slice(frame.as_bytes());
    }
    for event in events {
        for frame in encoder.feed(event).map_err(|_| PublicError::internal())? {
            body.extend_from_slice(frame.as_bytes());
        }
    }
    Ok(body)
}

/// Seal one completed provider turn when it contains Fireworks reasoning.
pub(crate) async fn seal_reasoning_events(
    bridge: &crate::bridge::Bridge,
    request_id: &str,
    route_depth: usize,
    events: &[Event],
) -> Result<Option<String>, Failure> {
    if !matches!(
        events.iter().rev().find(|event| event.is_terminal()),
        Some(Event::Completed)
    ) {
        return Ok(None);
    }
    let candidate = reasoning_carrier_candidate(events).map_err(|_| {
        Failure::new(
            FailureClass::MalformedResponse,
            "provider returned malformed reasoning continuation data",
        )
    })?;
    seal_reasoning_candidate(bridge, request_id, route_depth, candidate).await
}

/// Ask the Python authority to bind and encrypt one validated native turn.
pub(crate) async fn seal_reasoning_candidate(
    bridge: &crate::bridge::Bridge,
    request_id: &str,
    route_depth: usize,
    candidate: Option<ReasoningCarrierCandidate>,
) -> Result<Option<String>, Failure> {
    let Some(candidate) = candidate else {
        return Ok(None);
    };
    let argument = compact_json(&json!({
        "request_id": request_id,
        "route_depth": route_depth,
        "route_sha256": candidate.route_sha256,
        "content": candidate.content,
        "assistant_content": candidate.assistant_content,
        "tool_calls": candidate.tool_calls.into_iter().map(|call| json!({
            "call_id": call.call_id,
            "name": call.name,
            "raw_arguments": call.raw_arguments,
        })).collect::<Vec<_>>(),
    }));
    let response = bridge
        .call("seal_reasoning_content", argument)
        .await
        .map_err(|_| {
            Failure::new(
                FailureClass::MalformedResponse,
                "provider reasoning continuation could not be authenticated",
            )
        })?;
    let payload: Value = serde_json::from_str(&response).map_err(|_| {
        Failure::new(
            FailureClass::Internal,
            "gateway returned an invalid reasoning carrier response",
        )
    })?;
    let carrier = payload
        .get("carrier")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            Failure::new(
                FailureClass::Internal,
                "gateway omitted the authenticated reasoning carrier",
            )
        })?;
    Ok(Some(carrier.to_string()))
}

/// Aggregate one committed non-streaming or guarded chat attempt and answer
/// it, settling exactly once and publishing keyed results.
#[allow(clippy::too_many_arguments)]
async fn respond_from_chat_events(
    admission: Admission,
    mut guard: AttemptGuard,
    depth: usize,
    mut events: Vec<Event>,
    usage: Option<Usage>,
    tool_names: Vec<String>,
    created_at: i64,
    mut lease: Option<OwnerLease>,
    client_request_id: Option<String>,
    stream_body: bool,
) -> Response {
    let refusal_completed = complete_visible_refusal(&mut events);
    let carrier = if refusal_completed.is_some() {
        None
    } else {
        match seal_reasoning_events(&guard.bridge, &admission.request_id, depth, &events).await {
            Ok(carrier) => carrier,
            Err(failure) => {
                guard
                    .settle("failed", usage.as_ref(), &tool_names, Some(&failure), true)
                    .await;
                if let Some(mut owner) = lease.take() {
                    owner.abandon().await;
                }
                return error_response(&failure.public_error());
            }
        }
    };
    let aggregated = match completed_chat_body_with_carrier(
        &admission.request_id,
        &admission.alias,
        created_at,
        &events,
        &admission.ignored_parameters,
        carrier.as_deref(),
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
        if let Some(mut owner) = lease.take() {
            owner.abandon().await;
        }
        return error_response(&PublicError::internal());
    }
    let mut headers = commit_independent(&admission, client_request_id.as_deref());
    headers.extend(commit_dependent(&admission, depth));
    if stream_body {
        let body = match encode_chat_sse(&admission, created_at, &events, carrier.as_deref()) {
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
        // Publish the exact response body and headers, then answer from the
        // stored copy, matching the python engine's `_cached_response`.
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
async fn completed_response(
    admission: Admission,
    mut guard: AttemptGuard,
    mut committed: CommittedAttempt,
    created_at: i64,
    deadline: Instant,
    permit: tokio::sync::OwnedSemaphorePermit,
    mut lease: Option<OwnerLease>,
    client_request_id: Option<String>,
) -> Response {
    let _permit = permit;
    let phase_timeout = admission.phase_timeout(committed.depth);
    let collection =
        collect_committed(&mut committed, deadline, phase_timeout, guard.started).await;
    // Record TTFT before any settle so a mid-collection failure still keeps an observed first token.
    guard.record_first_token(committed.relay.first_token_at());
    let events = match collection {
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
    respond_from_chat_events(
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

#[allow(clippy::too_many_arguments)]
async fn guarded_chat_response(
    admission: Admission,
    mut guard: AttemptGuard,
    mut committed: CommittedAttempt,
    created_at: i64,
    deadline: Instant,
    permit: tokio::sync::OwnedSemaphorePermit,
    mut lease: Option<OwnerLease>,
    client_request_id: Option<String>,
) -> Response {
    let _permit = permit;
    let phase_timeout = admission.phase_timeout(committed.depth);
    let collection =
        collect_committed(&mut committed, deadline, phase_timeout, guard.started).await;
    // Record TTFT before any settle so a mid-collection failure still keeps an observed first token.
    guard.record_first_token(committed.relay.first_token_at());
    let collected = match collection {
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
    respond_from_chat_events(
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
async fn stream_response(
    admission: Admission,
    guard: AttemptGuard,
    committed: CommittedAttempt,
    created_at: i64,
    deadline: Instant,
    permit: tokio::sync::OwnedSemaphorePermit,
    lease: Option<OwnerLease>,
    client_request_id: Option<String>,
) -> Response {
    let (sender, receiver) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    let mut header_pairs = commit_independent(&admission, client_request_id.as_deref());
    header_pairs.extend(commit_dependent(&admission, committed.depth));
    let include_usage = admission.include_usage;
    let request_id = admission.request_id.clone();
    let alias = admission.alias.clone();
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
        let mut lease = lease;
        let mut committed = committed;
        let mut encoder = ChatSseEncoder::new_with_ignored(
            &request_id,
            &alias,
            created_at,
            include_usage,
            admission.ignored_parameters.clone(),
        );
        let mut usage: Option<Usage> = committed.usage.take();
        let mut tool_names: Vec<String> = std::mem::take(&mut committed.tool_names);
        let mut visible_refusal = committed.visible_refusal;
        let mut terminal: Option<Event> = None;
        // Keyed streams capture every public frame so the owner can publish
        // the exact byte stream; terminal frames are withheld until that
        // publication succeeds, matching the python engine's `_stream_body`.
        let mut capture: Vec<u8> = Vec::new();
        let mut replayable = lease.is_some();

        macro_rules! fail_stream {
            ($failure:expr) => {{
                let failure = $failure.boundary();
                let frames = failure_frames(&mut encoder, &failure);
                guard
                    .settle("failed", usage.as_ref(), &tool_names, Some(&failure), true)
                    .await;
                finish_stream_terminal(
                    &sender,
                    deadline,
                    &mut lease,
                    replayable,
                    &mut capture,
                    &cached_headers,
                    frames,
                )
                .await;
                return;
            }};
        }

        // Mirror any prefix-peeked first token before a start-frame send can cancel and drop it.
        guard.record_first_token(committed.relay.first_token_at());
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
            // Mirror the relay's first-token time onto the guard as tokens stream.
            guard.record_first_token(committed.relay.first_token_at());
            if matches!(
                event,
                Event::RefusalDelta(_) | Event::ProviderRefusalDelta { .. }
            ) {
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
                if matches!(event, Event::Completed) {
                    let candidate = match encoder.reasoning_carrier_candidate() {
                        Ok(candidate) => candidate,
                        Err(_) => {
                            fail_stream!(Failure::new(
                                FailureClass::MalformedResponse,
                                "provider returned malformed reasoning continuation data",
                            ))
                        }
                    };
                    match seal_reasoning_candidate(
                        &guard.bridge,
                        &request_id,
                        committed.depth,
                        candidate,
                    )
                    .await
                    {
                        Ok(Some(carrier)) => encoder.set_reasoning_content_carrier(carrier),
                        Ok(None) => {}
                        Err(failure) => fail_stream!(failure),
                    }
                }
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
            if terminal.is_some() {
                // Terminal frames flow through the shared publication tail so
                // keyed owners publish the exact byte stream first.
                finish_stream_terminal(
                    &sender,
                    deadline,
                    &mut lease,
                    replayable,
                    &mut capture,
                    &cached_headers,
                    encoded.into_iter().map(Bytes::from).collect(),
                )
                .await;
                return;
            }
            for data in encoded {
                let data = Bytes::from(data);
                if lease.is_some() {
                    replayable = capture_frame(&mut capture, &data, replayable);
                }
                if !send_bounded(&sender, deadline, data).await {
                    settle_stream_end(&mut guard, None, usage.as_ref(), &tool_names, true).await;
                    return;
                }
            }
        }
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

/// Build the encoder's sanitized failure frame and done sentinel when the
/// stream has not already reached a terminal.
fn failure_frames(encoder: &mut ChatSseEncoder, failure: &Failure) -> Vec<Bytes> {
    if encoder.saw_terminal() {
        return Vec::new();
    }
    encoder
        .feed(&Event::Failed(failure.clone()))
        .unwrap_or_else(|_| {
            vec![
                chat_data(&failure.public_error().json_body()),
                "data: [DONE]\n\n".to_string(),
            ]
        })
        .into_iter()
        .map(Bytes::from)
        .collect()
}
