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
    acquire_permit, apply_output_guardrail, new_guard, served_headers, wire_drift_response,
    Admission,
};
use crate::capture::reasoning::{checkpoint_winner, observe_winner};
use crate::encode::{compact_json, reasoning_carrier_candidate};
use crate::encode_responses::ResponsesSseEncoder;
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{Event, Usage};
use crate::guardrails::{released_events, StreamRedactor};
use crate::metrics::{classify_escalation, METRICS};
use crate::relay::{collect_committed, collection_public_error, track_event};
use crate::replay::{CachedResponse, Claim, OwnerLease, ReplayKey};
use crate::respond::{
    bearer_key, cached_response, capture_frame, client_ip, complete_visible_refusal,
    emit_responses_failure, error_response, escalation_error, finish_stream_terminal,
    json_response, latin1_header, outward_event, read_body, settle_stream_end, sse_body_response,
    stream_delivery::Delivery,
};
use crate::responses_retention::{remember_argument, remember_continuation, ResponsesRetention};
use crate::route_chat::{seal_reasoning_candidate, seal_reasoning_events};
use crate::server::AppState;
use crate::settlement::AttemptGuard;
use crate::tool_search::{
    adopt_outcome, completed_responses_body_for, configure_responses_encoder,
    disclose_after_collection, encode_responses_sse,
};
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
    // Only the standard Idempotency-Key opts into replay: Codex reuses
    // x-client-request-id (its session id) across distinct sequential
    // requests, so that header is correlation and affinity identity, never
    // an operation key.
    let idempotency_key = latin1_header(&headers, "idempotency-key");
    let client_request_id = latin1_header(&headers, "x-client-request-id");
    let mut lease: Option<OwnerLease> = None;
    if idempotency_key.is_some() {
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
    let mut admission: Admission = match serde_json::from_value(admission_value.clone()) {
        Ok(admission) => admission,
        Err(_) => {
            if let Some(mut owner) = lease.take() {
                owner.abandon().await;
            }
            return wire_drift_response(&state, &admission_value, started).await;
        }
    };
    let mut guard = new_guard(&state, admission.request_id.clone(), started);
    guard.record_web_search_requests(admission.web_search_requests());
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
        chat_logprobs: false,
        // A turn that ends before any semantic output is still a response
        // the caller can continue from; the waterfall retains it in flight.
        output_less_retention: Some(remember_argument(
            &admission.request_id,
            &ResponsesRetention::default(),
            None,
        )),
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
            settled_responses_response(&admission, settled, created_at, lease, client_request_id)
                .await
        }
        Won::Committed(committed) => {
            let committed = *committed;
            let incremental = admission.stream_incremental(committed.depth);
            if admission.buffers_output() && !incremental {
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
                    incremental,
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
    };
    crate::capture::response::capture_response(capture, &capture_request_id, response)
}

/// Answer one Responses attempt that the waterfall already settled: a
/// successful terminal with no semantic output, or an exhausted ladder
/// flushing withheld refusal output ahead of the failing terminal.
async fn settled_responses_response(
    admission: &Admission,
    settled: SettledAttempt,
    created_at: i64,
    mut lease: Option<OwnerLease>,
    client_request_id: Option<String>,
) -> Response {
    let served = settled.served();
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
    let headers = served_headers(admission, client_request_id.as_deref(), served);
    if admission.stream {
        let body = match encode_responses_sse(admission, created_at, &events, None) {
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
    let aggregated = match completed_responses_body_for(admission, created_at, &events, None) {
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
    let reasoning_content_carrier =
        match seal_reasoning_events(&guard.bridge, &admission.request_id, depth, &events).await {
            Ok(carrier) => carrier,
            Err(failure) => {
                let failure = failure.boundary();
                guard
                    .settle("failed", usage.as_ref(), &tool_names, Some(&failure), true)
                    .await;
                if let Some(mut owner) = lease.take() {
                    owner.abandon().await;
                }
                return error_response(&failure.public_error());
            }
        };
    let aggregated = match completed_responses_body_for(
        &admission,
        created_at,
        &events,
        reasoning_content_carrier.as_deref(),
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
    let mut retention = ResponsesRetention::default();
    for event in &events {
        retention.track(event);
    }
    let remembered = remember_continuation(
        state,
        &admission.request_id,
        &retention,
        reasoning_content_carrier.as_deref(),
    )
    .await;
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
    let headers = served_headers(&admission, client_request_id.as_deref(), served);
    if stream_body {
        let body = match encode_responses_sse(
            &admission,
            created_at,
            &events,
            reasoning_content_carrier.as_deref(),
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
    respond_from_responses_events(
        state,
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
async fn guarded_responses(
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
    let events = match apply_output_guardrail(&state, &admission, collected, deadline).await {
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

#[allow(clippy::too_many_arguments)]
async fn stream_responses(
    state: AppState,
    admission: Admission,
    guard: AttemptGuard,
    committed: CommittedAttempt,
    created_at: i64,
    deadline: Instant,
    permit: tokio::sync::OwnedSemaphorePermit,
    lease: Option<OwnerLease>,
    client_request_id: Option<String>,
    incremental_guardrail: bool,
) -> Response {
    let (sender, receiver) = mpsc::channel::<Result<Bytes, std::io::Error>>(64);
    let header_pairs = served_headers(&admission, client_request_id.as_deref(), committed.served());
    let request_id = admission.request_id.clone();
    let alias = admission.alias.clone();
    let envelope = admission.envelope.clone().unwrap_or_default();
    let phase_timeout = admission.phase_timeout(committed.depth);
    let mut cached_headers = header_pairs.clone();
    cached_headers.sort();
    let task_hold = guard.hold_task();
    tokio::spawn(async move {
        let _task = task_hold;
        let _permit = permit;
        let mut guard = guard;
        let mut committed = committed;
        let mut lease = lease;
        let mut delivery = Delivery::new(sender.clone(), lease.is_some());
        // Keyed streams capture every public frame so the owner can publish
        // the exact byte stream; terminal frames flow through the shared
        // publication tail, matching the chat surface.
        let mut capture: Vec<u8> = Vec::new();
        let mut replayable = lease.is_some();
        let mut encoder = ResponsesSseEncoder::new(&request_id, &alias, created_at, envelope);
        configure_responses_encoder(&mut encoder, &admission);
        let mut usage: Option<Usage> = committed.usage.take();
        let mut tool_names: Vec<String> = std::mem::take(&mut committed.tool_names);
        let mut visible_refusal = committed.visible_refusal;
        let mut terminal: Option<Event> = None;
        let mut retention = ResponsesRetention::default();
        let mut reasoning_content_carrier: Option<String> = None;
        // Deterministic output redaction as bytes flow: only the trailing
        // window the detector cannot yet decide about is withheld.
        let mut redactor = incremental_guardrail.then(|| StreamRedactor::new(&request_id));
        // Terminal frames are withheld until continuation retention lands,
        // mirroring the python stream body's ordering.
        let terminal_frames: Vec<String>;

        macro_rules! fail_stream {
            ($failure:expr) => {{
                committed.relay.close_transport();
                let failure = $failure.boundary();
                emit_responses_failure(&sender, deadline, &mut encoder, &failure).await;
                guard
                    .settle("failed", usage.as_ref(), &tool_names, Some(&failure), true)
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
                delivery.retain_replay(replayable);
            }
            if !delivery.send(deadline, data).await {
                committed.relay.close_transport();
                guard.settle_cancelled(usage.as_ref(), &tool_names).await;
                return;
            }
        }

        let mut prefix: std::collections::VecDeque<Event> = committed.prefix.drain(..).collect();
        'stream: loop {
            if crate::relay::remaining(deadline).is_zero() {
                committed.relay.close_transport();
                guard.settle_cancelled(usage.as_ref(), &tool_names).await;
                return;
            }
            let event = if let Some(event) = prefix.pop_front() {
                event
            } else {
                match delivery
                    .next(
                        committed
                            .relay
                            .next_event(deadline, phase_timeout, guard.started),
                    )
                    .await
                {
                    None => {
                        committed.relay.close_transport();
                        guard.settle_cancelled(usage.as_ref(), &tool_names).await;
                        return;
                    }
                    Some(Ok(Some(event))) => event,
                    Some(Ok(None)) => {
                        usage = committed.relay.usage_before_failure(usage.take());
                        fail_stream!(Failure::new(
                            FailureClass::MalformedResponse,
                            "provider stream ended without a terminal event",
                        ))
                    }
                    Some(Err(failure)) => {
                        usage = committed.relay.usage_before_failure(usage.take());
                        fail_stream!(failure)
                    }
                }
            };
            track_event(&event, &mut usage, &mut tool_names);
            if matches!(event, Event::Failed(_)) {
                usage = committed.relay.usage_before_failure(usage.take());
            }
            if redactor.is_none() {
                // A guarded stream retains what the caller actually saw, so
                // a continuation replays the redacted text, never the raw
                // completion; retention then runs over the released events.
                retention.track(&event);
            }
            // Mirror the relay's first-token time onto the guard as tokens stream.
            guard.record_first_token(committed.relay.first_token_at());
            let outward = outward_event(&event, &mut visible_refusal);
            // A byte that reaches the caller has already been through the
            // detector, and a terminal flushes whatever is still buffered.
            let guarded = redactor.is_some();
            let outward_events = match released_events(
                redactor.as_mut(),
                &guard.bridge,
                outward,
                event.is_terminal(),
            )
            .await
            {
                Ok(events) => events,
                Err(failure) => fail_stream!(failure),
            };
            if guarded {
                for released in &outward_events {
                    retention.track(released);
                }
            }
            // The terminal is recorded before its frames flush, so a
            // disconnect during the final flush still settles by the
            // provider's outcome instead of as a cancellation.
            if event.is_terminal() {
                committed.relay.close_transport();
                terminal = Some(event.clone());
                if matches!(event, Event::Completed) {
                    let candidate = match reasoning_carrier_candidate(&retention.carrier_events) {
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
                        Ok(Some(carrier)) => {
                            if encoder
                                .set_reasoning_content_carrier(carrier.clone())
                                .is_err()
                            {
                                fail_stream!(Failure::new(
                                    FailureClass::MalformedResponse,
                                    "provider reasoning continuation could not be authenticated",
                                ))
                            }
                            reasoning_content_carrier = Some(carrier);
                        }
                        Ok(None) => {}
                        Err(failure) => fail_stream!(failure),
                    }
                }
            }
            for outward in outward_events {
                let encoded = match encoder.feed(&outward) {
                    Ok(encoded) => encoded,
                    Err(_) => {
                        fail_stream!(Failure::new(
                            FailureClass::Internal,
                            "gateway could not encode the provider stream",
                        ))
                    }
                };
                if outward.is_terminal() {
                    terminal_frames = encoded;
                    break 'stream;
                }
                for data in encoded {
                    let data = Bytes::from(data);
                    if lease.is_some() {
                        replayable = capture_frame(&mut capture, &data, replayable);
                        delivery.retain_replay(replayable);
                    }
                    if !delivery.send(deadline, data).await {
                        committed.relay.close_transport();
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
        }

        // Retention runs before the terminal frames flush; a bounded
        // retention failure truncates the stream before its terminal, the
        // same observable behavior as the python service.
        let retained = matches!(terminal, Some(Event::Failed(_)))
            || remember_continuation(
                &state,
                &admission.request_id,
                &retention,
                reasoning_content_carrier.as_deref(),
            )
            .await
            .is_ok();
        // Settle every terminal once, including a failed continuation write.
        let settled = settle_stream_end(
            &mut guard,
            terminal.as_ref(),
            usage.as_ref(),
            &tool_names,
            false,
        )
        .await;
        if !retained || !settled {
            return;
        }
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
