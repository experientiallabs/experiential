//! The native OpenAI Chat Completions surface: the `/v1/chat/completions`
//! handler with its keyed-replay protocol, plus the chat-shaped settled,
//! aggregated, guarded, and live-streaming response paths.

use std::sync::atomic::Ordering;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Response;
use serde_json::{json, Value};

use crate::admission::{
    acquire_permit, apply_output_guardrail, new_guard, served_headers, wire_drift_response,
    Admission,
};
use crate::capture::reasoning::{checkpoint_winner, observe_winner};
use crate::encode::{
    compact_json, completed_chat_body_with_carrier, completed_chat_body_with_ignored,
    reasoning_carrier_candidate, ChatSseEncoder, ReasoningCarrierCandidate,
};
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{Event, Usage};
use crate::metrics::{classify_escalation, METRICS};
use crate::relay::{collect_committed, collection_public_error};
use crate::replay::{CachedResponse, Claim, OwnerLease, ReplayKey};
use crate::respond::{
    bearer_key, cached_response, client_ip, complete_visible_refusal, error_response,
    escalation_error, json_response, latin1_header, read_body, sse_body_response,
};
use crate::server::AppState;
use crate::settlement::{settle_guarded_failure, AttemptGuard};
use crate::tool_search::{
    adopt_outcome, annotate_chat_completion_for, configure_chat_encoder, disclose_after_collection,
};
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
        "client_ip": client_ip(&headers),
        "capture_session_id": crate::capture::session_id(&headers),
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
    let mut admission: Admission =
        match serde_json::from_value::<Admission>(admission_value.clone()) {
            Ok(admission) if admission.preserves_chat_probabilities() => admission,
            _ => {
                // The request is durably accepted; abandon it before failing so
                // wire-contract drift cannot leak an open request row.
                if let Some(mut owner) = lease.take() {
                    owner.abandon().await;
                }
                return wire_drift_response(&state, &admission_value, started).await;
            }
        };
    let mut guard = new_guard(&state, admission.request_id.clone(), started);
    guard.record_web_search_requests(admission.web_search_requests());
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
        caller_scope: admission.caller_scope.as_deref(),
        route: &admission.route,
        policy: admission.policy(),
        deadline,
        time_to_first_byte: state.time_to_first_byte,
        time_to_first_byte_slope_seconds_per_million_input_tokens: state
            .time_to_first_byte_slope_seconds_per_million_input_tokens,
        time_to_first_token: state.time_to_first_token,
        // Bytes over four approximates input tokens; a timeout heuristic
        // only, never a billing quantity.
        approximate_input_tokens: (body_text.len() as f64) / 4.0,
        chat_logprobs: true,
        output_less_retention: None,
        output_token_cap: admission.maximum_output_tokens,
        tool_search: admission.tool_search.as_ref(),
    };
    let mut won = acquire_attempt(&context, &mut guard).await;
    adopt_outcome(&mut admission, &mut won);
    won = checkpoint_winner(
        state.capture.as_ref(),
        &admission,
        &mut guard,
        won,
        deadline,
    )
    .await;
    observe_winner(state.capture.clone(), &admission, &guard, &mut won);

    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or(0);

    let capture = state.capture.clone();
    let capture_request_id = admission.request_id.clone();
    let response = match won {
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
            let incremental = admission.stream_incremental(committed.depth);
            if admission.buffers_output() && !incremental {
                guarded_chat_response(
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
                stream_response(
                    admission,
                    guard,
                    committed,
                    created_at,
                    deadline,
                    permit,
                    lease,
                    client_request_id,
                    incremental,
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
    };
    crate::capture::response::capture_response(capture, &capture_request_id, response)
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
    let served = settled.served();
    let mut events = settled.events;
    let refusal_completed = complete_visible_refusal(&mut events);
    if refusal_completed.is_none() {
        if let Some(Event::Failed(failure)) = events.last() {
            let error = collection_public_error(&failure.clone().boundary());
            if admission.stream {
                // The withheld refusal output and its failing terminal flush
                // outward as the stream's only frames. This settled path never
                // carries reasoning, so plaintext exposure is off throughout.
                let body = match encode_chat_sse(admission, created_at, &events, None, false) {
                    Ok(body) => body,
                    Err(error) => return error_response(&error),
                };
                let headers = served_headers(admission, client_request_id.as_deref(), served);
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
    let headers = served_headers(admission, client_request_id.as_deref(), served);
    if admission.stream {
        let body = match encode_chat_sse(admission, created_at, &events, None, false) {
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
    let mut aggregated = match completed_chat_body_with_ignored(
        &admission.request_id,
        &admission.alias,
        created_at,
        &events,
        &admission.ignored_parameters,
        false,
    ) {
        Ok(aggregated) => aggregated,
        Err(error) => return error_response(&error),
    };
    annotate_chat_completion_for(&mut aggregated.body, admission);
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
    reasoning_output_exposed: bool,
) -> Result<Vec<u8>, PublicError> {
    let mut encoder = ChatSseEncoder::new_with_ignored(
        &admission.request_id,
        &admission.alias,
        created_at,
        admission.include_usage,
        admission.ignored_parameters.clone(),
    );
    configure_chat_encoder(&mut encoder, admission);
    encoder.set_reasoning_output_exposed(reasoning_output_exposed);
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
    served: crate::waterfall::Served,
    mut events: Vec<Event>,
    usage: Option<Usage>,
    tool_names: Vec<String>,
    created_at: i64,
    mut lease: Option<OwnerLease>,
    client_request_id: Option<String>,
    stream_body: bool,
) -> Response {
    let depth = served.depth;
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
    let mut aggregated = match completed_chat_body_with_carrier(
        &admission.request_id,
        &admission.alias,
        created_at,
        &events,
        &admission.ignored_parameters,
        carrier.as_deref(),
        admission.reasoning_exposed_at(depth),
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
    annotate_chat_completion_for(&mut aggregated.body, &admission);
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
    let headers = served_headers(&admission, client_request_id.as_deref(), served);
    if stream_body {
        let body = match encode_chat_sse(
            &admission,
            created_at,
            &events,
            carrier.as_deref(),
            admission.reasoning_exposed_at(depth),
        ) {
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
    mut admission: Admission,
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
    disclose_after_collection(&mut admission, &committed);
    respond_from_chat_events(
        admission,
        guard,
        committed.served(),
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
    state: AppState,
    mut admission: Admission,
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
            settle_guarded_failure(&mut guard, &mut committed, &mut lease, &failure).await;
            return error_response(&error);
        }
    };
    disclose_after_collection(&mut admission, &committed);
    let events = match apply_output_guardrail(&state, &admission, collected, deadline).await {
        Ok(events) => events,
        Err(failure) => {
            settle_guarded_failure(&mut guard, &mut committed, &mut lease, &failure).await;
            return error_response(&failure.public_error());
        }
    };
    let stream_body = admission.stream;
    respond_from_chat_events(
        admission,
        guard,
        committed.served(),
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

#[path = "chat_stream.rs"]
mod chat_stream;
use chat_stream::stream_response;
