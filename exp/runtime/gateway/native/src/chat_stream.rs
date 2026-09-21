//! Live Chat SSE delivery and bounded keyed owner capture.

use std::time::Instant;

use axum::body::Body;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::Response;
use bytes::Bytes;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::seal_reasoning_candidate;
use crate::admission::{served_headers, Admission};
use crate::encode::ChatSseEncoder;
use crate::errors::{Failure, FailureClass};
use crate::events::{Event, Usage};
use crate::guardrails::{released_events, StreamRedactor};
use crate::relay::track_event;
use crate::replay::OwnerLease;
use crate::respond::{
    capture_frame, failure_frames, finish_stream_terminal, log_stream_exit, outward_event,
    settle_stream_end, stream_delivery::Delivery,
};
use crate::settlement::AttemptGuard;
use crate::tool_search::configure_chat_encoder;
use crate::waterfall::CommittedAttempt;

#[allow(clippy::too_many_arguments)]
pub(super) async fn stream_response(
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
        let mut delivery = Delivery::new(sender.clone(), lease.is_some());
        let mut encoder = ChatSseEncoder::new_with_ignored(
            &request_id,
            &alias,
            created_at,
            include_usage,
            admission.ignored_parameters.clone(),
        );
        configure_chat_encoder(&mut encoder, &admission);
        encoder.set_reasoning_output_exposed(admission.reasoning_exposed_at(committed.depth));
        let mut usage: Option<Usage> = committed.usage.take();
        let mut tool_names: Vec<String> = std::mem::take(&mut committed.tool_names);
        let mut visible_refusal = committed.visible_refusal;
        let mut terminal: Option<Event> = None;
        // Keyed streams capture every public frame so the owner can publish
        // the exact byte stream; terminal frames are withheld until that
        // publication succeeds, matching the python engine's `_stream_body`.
        let mut capture: Vec<u8> = Vec::new();
        let mut replayable = lease.is_some();
        // Deterministic output redaction as bytes flow: only the trailing
        // window the detector cannot yet decide about is withheld.
        let mut redactor = incremental_guardrail.then(|| StreamRedactor::new(&request_id));

        macro_rules! fail_stream {
            ($failure:expr) => {{
                committed.relay.close_transport();
                log_stream_exit(&request_id, "provider_or_encoding_failure");
                let failure = $failure.boundary();
                let frames = failure_frames(&mut encoder, &failure);
                if !guard
                    .settle("failed", usage.as_ref(), &tool_names, Some(&failure), true)
                    .await
                {
                    log_stream_exit(&request_id, "settlement_unavailable");
                    return;
                }
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
                delivery.retain_replay(replayable);
            }
            if !delivery.send(deadline, data).await {
                committed.relay.close_transport();
                log_stream_exit(&request_id, "subscriber_closed_or_delivery_deadline");
                guard.settle_cancelled(usage.as_ref(), &tool_names).await;
                return;
            }
        }

        let mut prefix: std::collections::VecDeque<Event> = committed.prefix.drain(..).collect();
        loop {
            if crate::relay::remaining(deadline).is_zero() {
                committed.relay.close_transport();
                log_stream_exit(&request_id, "delivery_deadline");
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
                        log_stream_exit(&request_id, "subscriber_closed");
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
            // Mirror the relay's first-token time onto the guard as tokens stream.
            guard.record_first_token(committed.relay.first_token_at());
            let outward = outward_event(&event, &mut visible_refusal);
            // A byte that reaches the caller has already been through the
            // detector, and a terminal flushes whatever is still buffered.
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
            if event.is_terminal() {
                committed.relay.close_transport();
                if matches!(event, Event::Completed | Event::StoppedAtSequence(_)) {
                    // The encoder requires the carrier on BOTH completing
                    // terminals; a stop sequence closing a reasoning tool turn
                    // used to end the stream short of its terminal frames.
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
                    log_stream_exit(&request_id, "settlement_unavailable");
                    return;
                }
            }
            for outward in outward_events {
                let encoded = match encoder.feed(&outward) {
                    Ok(encoded) => encoded,
                    Err(_) => {
                        if terminal.is_some() {
                            // The attempt already settled by its provider
                            // terminal; the stream simply ends short.
                            log_stream_exit(&request_id, "terminal_encoding_failed");
                            return;
                        }
                        fail_stream!(Failure::new(
                            FailureClass::Internal,
                            "gateway could not encode the provider stream",
                        ))
                    }
                };
                if outward.is_terminal() {
                    // Terminal frames flow through the shared publication tail
                    // so keyed owners publish the exact byte stream first.
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
                        delivery.retain_replay(replayable);
                    }
                    if !delivery.send(deadline, data).await {
                        committed.relay.close_transport();
                        log_stream_exit(&request_id, "subscriber_closed_or_delivery_deadline");
                        settle_stream_end(&mut guard, None, usage.as_ref(), &tool_names, true)
                            .await;
                        return;
                    }
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
