//! Incremental upstream relay: one provider response decoded and normalized
//! into gateway events, plus the shared collection helpers that bound and
//! classify what the relay yields. The waterfall commits a relay to one
//! deployment; the HTTP surfaces then drain it live or to completion.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures_util::stream::BoxStream;
use futures_util::StreamExt;

use crate::dialects::{
    Dialect, FrameDecoder, Normalizer, MAXIMUM_RETAINED_OUTPUT_BYTES, OUTPUT_OVERFLOW_MESSAGE,
};
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{Event, Usage};
use crate::metrics::METRICS;
use crate::waterfall::CommittedAttempt;

/// Map one collection failure to its public error, honoring the shared
/// aggregate-output overflow contract.
pub fn collection_public_error(failure: &Failure) -> PublicError {
    if failure.safe_message == OUTPUT_OVERFLOW_MESSAGE {
        return PublicError::provider_output_too_large();
    }
    failure.public_error()
}

/// Approximate retained size of one aggregated event, in bytes. Completed
/// tool calls charge their full argument text, matching the python engine's
/// bounded aggregation, which also charges the completed call after its
/// streamed deltas.
pub fn event_retained_bytes(event: &Event) -> usize {
    match event {
        Event::TextDelta(text) | Event::RefusalDelta(text) => text.len(),
        Event::ReasoningSummaryDelta { delta, .. } => delta.len(),
        Event::ToolArgumentsDelta { delta, .. } => delta.len(),
        Event::ToolCallCompleted { call, .. } => call.raw_arguments.len().max(64),
        _ => 64,
    }
}

/// Classify one mid-stream chunk timeout the way the python transport does:
/// a stalled provider read is a retryable transport failure unless the
/// request's own deadline is exhausted.
pub fn stream_timeout_failure(deadline: Instant) -> Failure {
    if remaining(deadline).is_zero() {
        Failure::new(FailureClass::Timeout, "gateway execution deadline exceeded")
    } else {
        Failure::new(
            FailureClass::Transport,
            "provider transport failed; retry the request",
        )
        .with_retry(true, true)
    }
}

/// Classify a provider that accepted the connection but did not stream its
/// first byte within the fail-fast time-to-first-byte bound. A stalled lead
/// deployment must not hold the request for its full per-chunk timeout, so
/// this is a transient, capacity-shaped failure that is failover-eligible.
///
/// It is deliberately *not* same-deployment retryable: a lane that accepted
/// the connection but never answered is the clearest dead-lane signal, and
/// redialing it would only stall again for another window. Skipping the redial
/// and advancing straight to the next certified deployment is what keeps a
/// fresh pod's cost on a dead lane near one fail-fast window instead of several.
pub fn first_byte_timeout_failure() -> Failure {
    Failure::new(
        FailureClass::Timeout,
        "provider did not send the first token in time; failing over to the next deployment",
    )
    .with_retry(false, true)
}

/// The synthesized failure for a provider stream that closed without a
/// terminal event, matching the python executor's classification.
pub(crate) fn ended_without_terminal() -> Failure {
    Failure::new(
        FailureClass::MalformedResponse,
        "provider stream ended without a terminal event",
    )
    .with_retry(true, true)
}

pub fn remaining(deadline: Instant) -> Duration {
    deadline.saturating_duration_since(Instant::now())
}

/// Record the latest complete usage observation and invoked tool names.
pub fn track_event(event: &Event, usage: &mut Option<Usage>, tool_names: &mut Vec<String>) {
    match event {
        Event::Usage(candidate) if candidate.has_token_counts() => {
            *usage = Some(candidate.clone());
        }
        Event::ToolCallCompleted { call, .. } if !tool_names.contains(&call.name) => {
            tool_names.push(call.name.clone());
        }
        _ => {}
    }
}

/// One upstream response being decoded and normalized incrementally, over
/// whichever wire framing the dialect uses (SSE, or the AWS binary
/// event-stream framing for Bedrock).
pub struct UpstreamRelay {
    stream: BoxStream<'static, reqwest::Result<Bytes>>,
    decoder: FrameDecoder,
    normalizer: Normalizer,
    pending: VecDeque<Event>,
    eof: bool,
    first_byte_recorded: bool,
    /// Fail-fast bound for the very first provider byte. Once the first byte
    /// arrives (`first_byte_recorded`), subsequent reads use the deployment's
    /// per-chunk timeout instead, so a slow reasoning model can stream for a
    /// long time after it has started answering.
    first_byte_deadline: Instant,
}

impl UpstreamRelay {
    pub fn new(
        response: reqwest::Response,
        dialect: Dialect,
        first_byte_deadline: Instant,
    ) -> Self {
        Self::from_stream(
            response.bytes_stream().boxed(),
            dialect,
            first_byte_deadline,
        )
    }

    fn from_stream(
        stream: BoxStream<'static, reqwest::Result<Bytes>>,
        dialect: Dialect,
        first_byte_deadline: Instant,
    ) -> Self {
        Self {
            stream,
            decoder: FrameDecoder::new(dialect),
            normalizer: Normalizer::new(dialect),
            pending: VecDeque::new(),
            eof: false,
            first_byte_recorded: false,
            first_byte_deadline,
        }
    }

    /// Yield the next normalized event. `Ok(None)` means the upstream closed
    /// without a terminal event (the caller synthesizes that failure); a
    /// stream whose terminal was already yielded returns `Ok(None)` too, but
    /// callers stop at the terminal before observing it.
    pub async fn next_event(
        &mut self,
        deadline: Instant,
        phase_timeout: Duration,
        request_started: Instant,
    ) -> Result<Option<Event>, Failure> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Ok(Some(event));
            }
            if self.eof {
                return Ok(None);
            }
            // Before the first byte the fail-fast time-to-first-byte bound
            // applies; after it, each chunk is paced by the deployment's own
            // per-chunk timeout so long-running generation is never capped.
            let waiting_for_first_byte = !self.first_byte_recorded;
            let bound = if waiting_for_first_byte {
                remaining(deadline).min(remaining(self.first_byte_deadline))
            } else {
                remaining(deadline).min(phase_timeout)
            };
            let chunk = match tokio::time::timeout(bound, self.stream.next()).await {
                Ok(Some(Ok(chunk))) => chunk,
                Ok(Some(Err(_))) => {
                    return Err(Failure::new(
                        FailureClass::Transport,
                        "provider transport failed; retry the request",
                    )
                    .with_retry(true, true))
                }
                Ok(None) => {
                    self.eof = true;
                    // Recover a final unterminated SSE frame at EOF, exactly
                    // like the python decoder, so a provider that omits the
                    // closing blank line still settles by its terminal event.
                    let tail = self.decoder.finish().map_err(|message| {
                        Failure::new(FailureClass::MalformedResponse, &message)
                            .with_retry(false, true)
                    })?;
                    if let Some(frame) = tail {
                        let events = self.normalizer.feed(&frame)?;
                        self.pending.extend(events);
                    }
                    continue;
                }
                Err(_) => {
                    // A first-byte stall while the request deadline still has
                    // budget is the fail-fast case: classify it as a
                    // failover-eligible transient so the next rung is tried at
                    // once. A later chunk stall, or an exhausted request
                    // deadline, keeps the existing transport/deadline mapping.
                    if waiting_for_first_byte && !remaining(deadline).is_zero() {
                        return Err(first_byte_timeout_failure());
                    }
                    return Err(stream_timeout_failure(deadline));
                }
            };
            if !self.first_byte_recorded {
                METRICS
                    .time_to_first_byte_ms
                    .record(request_started.elapsed());
                self.first_byte_recorded = true;
            }
            let frames = self.decoder.feed(&chunk).map_err(|message| {
                Failure::new(FailureClass::MalformedResponse, &message).with_retry(false, true)
            })?;
            for frame in frames {
                let events = self.normalizer.feed(&frame)?;
                self.pending.extend(events);
            }
        }
    }
}

/// Drain one committed attempt to completion for non-streaming responses,
/// bounding total retained output like the python service's aggregation.
pub async fn collect_committed(
    committed: &mut CommittedAttempt,
    deadline: Instant,
    phase_timeout: Duration,
    request_started: Instant,
) -> Result<Vec<Event>, Failure> {
    let mut events: Vec<Event> = Vec::new();
    let mut retained_bytes = 0usize;
    let mut retain = |events: &mut Vec<Event>, event: Event| -> Result<(), Failure> {
        retained_bytes = retained_bytes.saturating_add(event_retained_bytes(&event));
        if retained_bytes > MAXIMUM_RETAINED_OUTPUT_BYTES {
            return Err(Failure::new(
                FailureClass::ProviderInternal,
                OUTPUT_OVERFLOW_MESSAGE,
            ));
        }
        events.push(event);
        Ok(())
    };
    for event in committed.prefix.drain(..) {
        retain(&mut events, event)?;
    }
    if events.last().is_some_and(Event::is_terminal) {
        return Ok(events);
    }
    loop {
        match committed
            .relay
            .next_event(deadline, phase_timeout, request_started)
            .await?
        {
            Some(event) => {
                track_event(&event, &mut committed.usage, &mut committed.tool_names);
                let terminal = event.is_terminal();
                retain(&mut events, event)?;
                if terminal {
                    return Ok(events);
                }
            }
            None => return Err(ended_without_terminal()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;

    #[test]
    fn a_first_byte_stall_fails_over_without_redialing_the_dead_lane() {
        let failure = first_byte_timeout_failure();
        assert_eq!(failure.failure_class, FailureClass::Timeout);
        assert!(failure.failover_eligible);
        // A stalled lane is skipped, not redialed: redialing it would only
        // stall again for another window.
        assert!(!failure.retryable_same_deployment);
    }

    #[tokio::test]
    async fn a_stalled_first_byte_trips_the_ttft_bound_not_the_chunk_timeout() {
        // A provider that opened the stream but never sends a byte must fail
        // over in about the time-to-first-byte window, not the (far larger)
        // per-chunk deployment timeout.
        let never = stream::pending::<reqwest::Result<Bytes>>().boxed();
        let time_to_first_byte = Duration::from_millis(80);
        let mut relay = UpstreamRelay::from_stream(
            never,
            Dialect::OpenAiCompatible,
            Instant::now() + time_to_first_byte,
        );
        let request_deadline = Instant::now() + Duration::from_secs(120);
        let per_chunk_timeout = Duration::from_secs(35);

        let started = Instant::now();
        let outcome = relay
            .next_event(request_deadline, per_chunk_timeout, started)
            .await;
        let elapsed = started.elapsed();

        assert!(
            elapsed < Duration::from_secs(2),
            "expected a fail-fast time-to-first-byte trip, waited {elapsed:?}"
        );
        let failure = outcome.expect_err("a never-yielding stream must not succeed");
        assert_eq!(failure.failure_class, FailureClass::Timeout);
        assert!(
            failure.failover_eligible,
            "a first-byte stall must advance to the next deployment"
        );
    }
}
