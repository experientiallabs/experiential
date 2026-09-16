//! Incremental upstream relay: one provider response decoded and normalized
//! into gateway events, plus the shared collection helpers that bound and
//! classify what the relay yields. The waterfall commits a relay to one
//! deployment; the HTTP surfaces then drain it live or to completion.

use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};

use bytes::Bytes;
use futures_util::stream::BoxStream;
use futures_util::StreamExt;

use crate::dialects::{
    Dialect, FrameDecoder, Normalizer, MAXIMUM_RETAINED_OUTPUT_BYTES, OUTPUT_OVERFLOW_MESSAGE,
};
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{Event, Usage};
use crate::metrics::METRICS;
use crate::stop_sequences::StopSequenceGuard;
use crate::tool_serialization::ToolCallSerializer;
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
        Event::ProviderTextDelta { delta, .. } | Event::ProviderRefusalDelta { delta, .. } => {
            delta.len()
        }
        Event::ReasoningSummaryDelta { delta, .. } => delta.len(),
        Event::ThinkingDelta { delta, .. } => delta.len(),
        Event::ThinkingSignature { signature, .. } => signature.len(),
        Event::RedactedThinking { data, .. } => data.len(),
        Event::EncryptedReasoning {
            encrypted_content, ..
        } => encrypted_content.len(),
        Event::ReasoningContentDelta { delta, .. } => delta.len(),
        Event::ToolArgumentsDelta { delta, .. } | Event::ServerToolArgumentsDelta { delta, .. } => {
            delta.len()
        }
        Event::ToolCallCompleted { call, .. } | Event::ServerToolUseCompleted { call, .. } => {
            call.raw_arguments.len().max(64)
        }
        Event::ServerToolResult { block, .. } => block.len(),
        Event::CitationDelta { citation, .. } => citation.len(),
        Event::HostedToolItemStarted { item, .. } | Event::HostedToolItemCompleted { item, .. } => {
            item.len()
        }
        Event::HostedToolItemProgress { payload, .. } => payload.len(),
        Event::ProviderTextAnnotation { annotation, .. } => annotation.len(),
        Event::ChoiceLogprobsDelta(delta) => delta.retained_bytes(),
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
        "provider did not send the first token in time",
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
        Event::ToolCallCompleted { call, .. } | Event::ServerToolUseCompleted { call, .. }
            if !tool_names.contains(&call.name) =>
        {
            // Server tool invocations are provider-executed but still
            // invoked tools: their names join usage so operators can see
            // (and price) per-invocation server tool activity.
            tool_names.push(call.name.clone());
        }
        // Hosted Responses tool INVOCATIONS are provider-executed too; the
        // item type ("web_search_call", "mcp_call", ...) names the activity.
        // Results, approvals, and opaque conversation items never record a
        // call that did not occur.
        Event::HostedToolItemCompleted { item_type, .. }
            if crate::events::hosted_item_type_is_invocation(item_type)
                && !tool_names.contains(item_type) =>
        {
            tool_names.push(item_type.clone());
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
    /// Normalized events not yet passed through the stop-sequence guard.
    pending: VecDeque<Event>,
    /// Guarded events ready to yield.
    ready: VecDeque<Event>,
    /// Gateway-emulated stop sequences for this rung, when the provider wire
    /// carries none; `None` passes every event straight through.
    stop_guard: Option<StopSequenceGuard>,
    /// Gateway-emulated `parallel_tool_calls: false`: one tool call per turn.
    tool_serializer: Option<ToolCallSerializer>,
    /// The provider of a customer-managed (BYOK) rung: a credential or account
    /// failure the provider declares on this stream, before or after commit,
    /// is re-owned as the customer's. `None` on house rungs.
    customer_managed_provider: Option<String>,
    eof: bool,
    first_byte_recorded: bool,
    /// Fail-fast bound for the very first provider byte. Once the first byte
    /// arrives (`first_byte_recorded`), subsequent reads use the deployment's
    /// per-chunk timeout instead, so a slow reasoning model can stream for a
    /// long time after it has started answering.
    first_byte_deadline: Instant,
    /// Wall-clock time this relay yielded its first output token (a content,
    /// reasoning, or tool-call delta), or `None` before any token arrives.
    /// Distinct from `first_byte_recorded`: the first byte can be an SSE frame
    /// carrying only role/lifecycle scaffolding, so time-to-first-token is
    /// stamped on the first event that carries visible model output.
    first_token_at: Option<SystemTime>,
    /// Tokens an earlier, refused dial of the same attempt was billed for,
    /// folded into the first usage report this relay yields so the
    /// reservation settles both dials' tokens as one.
    carried_usage: Option<Usage>,
}

impl UpstreamRelay {
    pub fn new(
        response: reqwest::Response,
        dialect: Dialect,
        first_byte_deadline: Instant,
    ) -> Self {
        Self::new_with_reasoning_content_route(response, dialect, first_byte_deadline, None)
    }

    pub fn new_with_reasoning_content_route(
        response: reqwest::Response,
        dialect: Dialect,
        first_byte_deadline: Instant,
        reasoning_content_route_sha256: Option<String>,
    ) -> Self {
        Self::from_stream_with_reasoning_content_route(
            response.bytes_stream().boxed(),
            dialect,
            first_byte_deadline,
            reasoning_content_route_sha256,
        )
    }

    #[cfg(test)]
    fn from_stream(
        stream: BoxStream<'static, reqwest::Result<Bytes>>,
        dialect: Dialect,
        first_byte_deadline: Instant,
    ) -> Self {
        Self::from_stream_with_reasoning_content_route(stream, dialect, first_byte_deadline, None)
    }

    fn from_stream_with_reasoning_content_route(
        stream: BoxStream<'static, reqwest::Result<Bytes>>,
        dialect: Dialect,
        first_byte_deadline: Instant,
        reasoning_content_route_sha256: Option<String>,
    ) -> Self {
        Self {
            stream,
            decoder: FrameDecoder::new(dialect),
            normalizer: Normalizer::new_with_reasoning_content_route(
                dialect,
                reasoning_content_route_sha256,
            ),
            pending: VecDeque::new(),
            ready: VecDeque::new(),
            stop_guard: None,
            tool_serializer: None,
            customer_managed_provider: None,
            eof: false,
            first_byte_recorded: false,
            first_byte_deadline,
            first_token_at: None,
            carried_usage: None,
        }
    }

    /// The wall-clock time this relay yielded its first output token, or
    /// `None` if it has not produced one yet. Read at settlement to report
    /// the winning attempt's time-to-first-token.
    pub fn first_token_at(&self) -> Option<SystemTime> {
        self.first_token_at
    }

    /// Caller-known label words (the dispatched model id) exempt from the
    /// provider-identifier screen on stream-error detail.
    pub fn set_request_words<I, S>(&mut self, words: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.normalizer.set_request_words(words);
    }

    pub fn set_chat_logprobs(&mut self, enabled: bool) {
        self.normalizer.enable_chat_logprobs(enabled);
    }

    /// Name the customer-managed provider this relay dispatches on, so every
    /// provider-declared credential or quota failure it yields is the
    /// customer's (see `stream_errors::customer_credential_failure`).
    pub fn set_customer_managed_provider(&mut self, provider: Option<String>) {
        self.customer_managed_provider = provider;
    }

    /// Serialize this relay's tool calls to one per turn (the caller sent
    /// `parallel_tool_calls: false` to a wire without that control).
    pub fn set_serialize_tool_calls(&mut self, serialize: bool) {
        self.tool_serializer = serialize.then(ToolCallSerializer::new);
    }

    /// Carry the tokens a refused earlier dial of this attempt was billed
    /// for; they join the first usage report this relay yields, once.
    pub fn set_carried_usage(&mut self, carried: Option<Usage>) {
        self.carried_usage = carried;
    }

    /// Enforce the caller's stop sequences on this relay's visible text.
    /// Installed before the first event is yielded; an empty set is a no-op.
    pub fn set_stop_sequences<I, S>(&mut self, sequences: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.stop_guard = StopSequenceGuard::new(sequences);
    }

    /// Move one normalized event through the stop-sequence guard (if any)
    /// onto the ready queue.
    fn guard_next_pending(&mut self) -> bool {
        let Some(mut event) = self.pending.pop_front() else {
            return false;
        };
        if let (Some(provider), Event::Failed(failure)) =
            (self.customer_managed_provider.as_deref(), &event)
        {
            event = Event::Failed(crate::stream_errors::customer_credential_failure(
                failure.clone(),
                provider,
            ));
        }
        if let Some(serializer) = self.tool_serializer.as_mut() {
            let Some(kept) = serializer.filter(event) else {
                return true;
            };
            event = kept;
        }
        match self.stop_guard.as_mut() {
            Some(guard) => self.ready.extend(guard.filter(event)),
            None => self.ready.push_back(event),
        }
        true
    }

    /// Route an abnormal stream termination through the normalizer's recovery.
    /// The relay is done either way, so mark EOF; when recovery applies (a
    /// Gemini stream that emitted content) the synthesized terminal is buffered
    /// for the caller to drain, otherwise the (possibly reclassified) failure
    /// propagates. See `Normalizer::recover_abnormal_end`.
    fn recover_or_fail(&mut self, failure: Failure) -> Result<(), Failure> {
        self.eof = true;
        let events = self.normalizer.recover_abnormal_end(failure)?;
        self.pending.extend(events);
        Ok(())
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
            if let Some(mut event) = self.ready.pop_front() {
                // Every yielded event exits here, so this is the one place that
                // stamps time-to-first-token: the first event carrying visible
                // model output. Prefix events peeked during commit also passed
                // through here, so the winning attempt's first token is stamped
                // whether it is later replayed from a prefix or drained live.
                if self.first_token_at.is_none() && event.is_output_token() {
                    self.first_token_at = Some(SystemTime::now());
                }
                if let (Event::Usage(usage), Some(carried)) =
                    (&mut event, self.carried_usage.as_ref())
                {
                    if usage.has_token_counts() {
                        *usage = fold_usage(carried, usage.clone());
                        self.carried_usage = None;
                    }
                }
                return Ok(Some(event));
            }
            if self.guard_next_pending() {
                continue;
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
                Ok(Some(Err(error))) => {
                    // A transport break mid-stream: recover a Gemini partial as
                    // Incomplete, otherwise surface the retryable transport
                    // failure. Pre-content it stays a retryable transport error
                    // either way. The engine's account of the break (never
                    // provider text) rides to the ledger.
                    self.recover_or_fail(
                        Failure::new(
                            FailureClass::Transport,
                            "provider transport failed; retry the request",
                        )
                        .with_retry(true, true)
                        .with_provider_detail(Some(
                            crate::upstream::transport_error_detail("stream", &error),
                        )),
                    )?;
                    continue;
                }
                Ok(None) => {
                    self.eof = true;
                    // Recover a final unterminated SSE frame at EOF, exactly
                    // like the python decoder, so a provider that omits the
                    // closing blank line still settles by its terminal event.
                    // A malformed trailing frame, or a normalizer that rejects
                    // it, is an abnormal end: recover a Gemini partial as
                    // Incomplete instead of discarding the answer.
                    let tail = match self.decoder.finish() {
                        Ok(tail) => tail,
                        Err(message) => {
                            self.recover_or_fail(
                                Failure::new(FailureClass::MalformedResponse, &message)
                                    .with_retry(false, true),
                            )?;
                            continue;
                        }
                    };
                    if let Some(frame) = tail {
                        match self.normalizer.feed(&frame) {
                            Ok(events) => self.pending.extend(events),
                            Err(failure) => {
                                self.recover_or_fail(failure)?;
                                continue;
                            }
                        }
                    }
                    // A stream may end cleanly without a terminal frame: a
                    // Gemini stream after its last content frame (no
                    // finishReason), or an OpenAI-compatible stream whose
                    // finish_reason chunk arrived without a `[DONE]` sentinel
                    // (Azure Foundry's DeepSeek content-filter ending). The
                    // normalizer synthesizes the terminal the dialect already
                    // declared so a real answer or refusal is not thrown away
                    // as malformed; a stream that declared nothing stays
                    // terminal-less and the caller still synthesizes
                    // `ended_without_terminal`.
                    match self.normalizer.on_stream_end() {
                        Ok(events) => self.pending.extend(events),
                        Err(failure) => {
                            self.recover_or_fail(failure)?;
                        }
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
            // A malformed frame, or a normalizer that rejects one, is an
            // abnormal end: recover a Gemini partial as Incomplete, otherwise
            // surface the failure. Recovery buffers a terminal and marks EOF,
            // so stop draining this chunk and let the outer loop yield it.
            let frames = match self.decoder.feed(&chunk) {
                Ok(frames) => frames,
                Err(message) => {
                    self.recover_or_fail(
                        Failure::new(FailureClass::MalformedResponse, &message)
                            .with_retry(false, true),
                    )?;
                    continue;
                }
            };
            for frame in frames {
                match self.normalizer.feed(&frame) {
                    Ok(events) => self.pending.extend(events),
                    Err(failure) => {
                        self.recover_or_fail(failure)?;
                        break;
                    }
                }
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
    use crate::dialects::Dialect;
    use crate::events::Event;
    use futures_util::stream;

    #[tokio::test]
    async fn first_token_at_is_stamped_on_the_first_output_delta() {
        // A content delta then the OpenAI terminal sentinel.
        let frames = vec![
            Ok::<_, reqwest::Error>(Bytes::from(
                "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            )),
            Ok::<_, reqwest::Error>(Bytes::from("data: [DONE]\n\n")),
        ];
        let mut relay = UpstreamRelay::from_stream(
            stream::iter(frames).boxed(),
            Dialect::OpenAiCompatible,
            Instant::now() + Duration::from_secs(5),
        );
        assert!(
            relay.first_token_at().is_none(),
            "no first-token time before any event is yielded"
        );
        let deadline = Instant::now() + Duration::from_secs(30);
        let per_chunk = Duration::from_secs(5);
        let first = relay
            .next_event(deadline, per_chunk, Instant::now())
            .await
            .expect("the stream yields")
            .expect("an event is produced");
        assert!(
            matches!(&first, Event::TextDelta(text) if text == "hi"),
            "the first output event is the content delta"
        );
        assert!(
            relay.first_token_at().is_some(),
            "the first content delta stamps time-to-first-token"
        );
    }

    #[tokio::test]
    async fn stop_sequences_cut_the_relayed_text_and_keep_usage_and_settlement_exact() {
        // A Chat-compatible stream stands in for any dialect: "</block>" spans
        // two content deltas, more text follows it, then usage and the
        // provider's own terminal arrive. The guard cuts at the match, drops
        // the trailing text, still yields the usage, and replaces the terminal.
        let frames = vec![
            Ok::<_, reqwest::Error>(Bytes::from(
                "data: {\"choices\":[{\"delta\":{\"content\":\"allow</bl\"}}]}\n\n",
            )),
            Ok::<_, reqwest::Error>(Bytes::from(
                "data: {\"choices\":[{\"delta\":{\"content\":\"ock>ignored\"}}]}\n\n",
            )),
            Ok::<_, reqwest::Error>(Bytes::from(
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3}}\n\n",
            )),
            Ok::<_, reqwest::Error>(Bytes::from(
                "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            )),
            Ok::<_, reqwest::Error>(Bytes::from("data: [DONE]\n\n")),
        ];
        let mut relay = UpstreamRelay::from_stream(
            stream::iter(frames).boxed(),
            Dialect::OpenAiCompatible,
            Instant::now() + Duration::from_secs(5),
        );
        relay.set_stop_sequences(["</block>"]);
        let deadline = Instant::now() + Duration::from_secs(30);
        let per_chunk = Duration::from_secs(5);
        let mut events = Vec::new();
        loop {
            match relay.next_event(deadline, per_chunk, Instant::now()).await {
                Ok(Some(event)) => {
                    let terminal = event.is_terminal();
                    events.push(event);
                    if terminal {
                        break;
                    }
                }
                Ok(None) => panic!("the stream must end on a terminal"),
                Err(failure) => panic!("unexpected failure: {failure:?}"),
            }
        }
        let text: String = events
            .iter()
            .filter_map(|event| match event {
                Event::TextDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "allow", "text stops exactly before the sequence");
        assert!(
            events
                .iter()
                .any(|event| matches!(event, Event::Usage(usage) if usage.has_token_counts())),
            "usage still reaches settlement after the cut"
        );
        assert!(
            matches!(events.last(), Some(Event::StoppedAtSequence(sequence)) if sequence == "</block>"),
            "the terminal names the matched sequence"
        );
    }

    #[tokio::test]
    async fn a_customer_managed_relay_re_owns_a_declared_credential_failure_after_output() {
        // Text has already streamed (the attempt is committed) when the
        // provider declares a 401 in-stream: the customer still gets their
        // own message, not the house "ask the gateway operator" one.
        let frames = vec![
            Ok::<_, reqwest::Error>(Bytes::from(
                "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            )),
            Ok::<_, reqwest::Error>(Bytes::from(
                "data: {\"error\":{\"code\":401,\"message\":\"Incorrect API key provided\"}}\n\n",
            )),
        ];
        let mut relay = UpstreamRelay::from_stream(
            stream::iter(frames).boxed(),
            Dialect::OpenAiCompatible,
            Instant::now() + Duration::from_secs(5),
        );
        relay.set_customer_managed_provider(Some("openai".to_string()));
        let deadline = Instant::now() + Duration::from_secs(30);
        let per_chunk = Duration::from_secs(5);
        let first = relay
            .next_event(deadline, per_chunk, Instant::now())
            .await
            .expect("yields")
            .expect("event");
        assert!(matches!(&first, Event::TextDelta(text) if text == "hi"));
        let second = relay
            .next_event(deadline, per_chunk, Instant::now())
            .await
            .expect("yields")
            .expect("event");
        match second {
            Event::Failed(failure) => {
                assert_eq!(failure.failure_class, FailureClass::ProviderAuthentication);
                assert!(failure.customer_owned);
                assert!(failure
                    .safe_message
                    .contains("your connected openai credential"));
                assert_eq!(failure.public_error().status_code, 400);
            }
            other => panic!("expected the re-owned failure, got {other:?}"),
        }
    }

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
    async fn a_gemini_partial_then_abnormal_frame_ends_incomplete_not_failed() {
        // A Gemini content frame, its usage, then a structurally malformed frame
        // (a non-string text part). The relay must route the abnormal end
        // through recovery: yield the content, fold the usage, and end on an
        // Incomplete terminal instead of surfacing the malformed failure.
        let frames = vec![
            Ok::<_, reqwest::Error>(Bytes::from(
                "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]}}]}\n\n",
            )),
            Ok::<_, reqwest::Error>(Bytes::from(
                "data: {\"usageMetadata\":{\"promptTokenCount\":9,\"candidatesTokenCount\":3}}\n\n",
            )),
            Ok::<_, reqwest::Error>(Bytes::from(
                "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":5}]}}]}\n\n",
            )),
        ];
        let mut relay = UpstreamRelay::from_stream(
            stream::iter(frames).boxed(),
            Dialect::GeminiGenerateContent,
            Instant::now() + Duration::from_secs(5),
        );
        let deadline = Instant::now() + Duration::from_secs(30);
        let per_chunk = Duration::from_secs(5);
        let mut seen: Vec<Event> = Vec::new();
        loop {
            let event = relay
                .next_event(deadline, per_chunk, Instant::now())
                .await
                .expect("recovery yields events, never the malformed failure");
            match event {
                Some(event) => {
                    let terminal = event.is_terminal();
                    seen.push(event);
                    if terminal {
                        break;
                    }
                }
                None => panic!("the recovered terminal must arrive before EOF"),
            }
        }
        assert!(
            matches!(seen.first(), Some(Event::TextDelta(text)) if text == "partial"),
            "the partial content is delivered"
        );
        assert!(
            seen.iter().any(|event| matches!(event, Event::Usage(_))),
            "last-seen usage is folded so delivered tokens bill"
        );
        assert!(
            matches!(seen.last(), Some(Event::Incomplete)),
            "the turn ends incomplete, not failed"
        );
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

#[cfg(test)]
mod h2_abort_tests {
    use super::*;
    use crate::dialects::Dialect;
    use crate::events::Event;

    fn sse_chunk(text: &str) -> Bytes {
        Bytes::from(format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{text}\"}}}}]}}\n\n"
        ))
    }

    /// Serve one h2c response that streams two deltas, then abort it the
    /// given way after a pacing delay so the client is mid-body when the
    /// abort lands (mirroring a proxy whose upstream dies mid-response).
    async fn relay_over_h2(reset_stream: bool) -> (Vec<Event>, Failure) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            let (socket, _peer) = listener.accept().await.expect("accept");
            let mut connection = h2::server::handshake(socket).await.expect("handshake");
            let accepted = connection.accept().await;
            // The connection must keep being polled for handshake frames and
            // window updates to reach the peer.
            let driver = tokio::spawn(async move {
                let _ = futures_util::future::poll_fn(|cx| connection.poll_closed(cx)).await;
            });
            if let Some(Ok((_request, mut respond))) = accepted {
                let response = http::Response::builder()
                    .status(200)
                    .header("content-type", "text/event-stream")
                    .body(())
                    .expect("response");
                let mut stream = respond.send_response(response, false).expect("headers");
                stream.send_data(sse_chunk("hi"), false).expect("data one");
                stream
                    .send_data(sse_chunk("there"), false)
                    .expect("data two");
                tokio::time::sleep(Duration::from_millis(300)).await;
                if reset_stream {
                    // The exact shape a fronting proxy (Caddy) produces when
                    // its own upstream aborts: the h2 stream resets.
                    stream.send_reset(h2::Reason::INTERNAL_ERROR);
                    let _ = driver.await;
                } else {
                    // Whole-connection abort: every multiplexed stream on the
                    // connection severs at once.
                    driver.abort();
                    drop(stream);
                }
            }
        });
        let client = reqwest::Client::builder()
            .http2_prior_knowledge()
            .build()
            .expect("client");
        let response = client
            .post(format!("http://{address}/v1/chat/completions"))
            .body("{}")
            .send()
            .await
            .expect("send");
        let mut relay = UpstreamRelay::new(
            response,
            Dialect::OpenAiCompatible,
            Instant::now() + Duration::from_secs(5),
        );
        let deadline = Instant::now() + Duration::from_secs(10);
        let per_chunk = Duration::from_secs(5);
        let mut events = Vec::new();
        loop {
            match relay.next_event(deadline, per_chunk, Instant::now()).await {
                Ok(Some(event)) => events.push(event),
                Ok(None) => panic!("an aborted stream must surface a failure, not clean EOF"),
                Err(failure) => return (events, failure),
            }
        }
    }

    #[tokio::test]
    async fn an_h2_stream_reset_mid_stream_classifies_and_never_panics() {
        // Production wire fact (verified live 2026-09-03): the house-lane
        // proxy negotiates ALPN h2, so its "aborting with incomplete
        // response" reaches this relay as RST_STREAM, never h1 truncation.
        let (events, failure) = relay_over_h2(true).await;
        assert!(
            matches!(events.as_slice(), [Event::TextDelta(a), Event::TextDelta(b)] if a == "hi" && b == "there"),
            "delivered deltas precede the abort: {events:?}"
        );
        assert_eq!(failure.failure_class, FailureClass::Transport);
        assert!(failure.failover_eligible, "an aborted rung fails over");
        // The engine's account of the mid-stream break rides to the ledger.
        let detail = failure
            .provider_detail
            .as_deref()
            .expect("transport detail");
        assert!(detail.starts_with("stream "), "{detail}");
    }

    #[tokio::test]
    async fn an_h2_connection_drop_mid_stream_classifies_and_never_panics() {
        let (events, failure) = relay_over_h2(false).await;
        assert_eq!(
            events.len(),
            2,
            "delivered deltas precede the abort: {events:?}"
        );
        assert_eq!(failure.failure_class, FailureClass::Transport);
        assert!(failure.failover_eligible);
    }
}

/// The tokens of two physical dials of one attempt, summed leg by leg; a leg
/// neither reported stays absent.
fn fold_usage(carried: &Usage, current: Usage) -> Usage {
    let add = |a: Option<u64>, b: Option<u64>| match (a, b) {
        (Some(a), Some(b)) => Some(a + b),
        (Some(a), None) | (None, Some(a)) => Some(a),
        (None, None) => None,
    };
    Usage {
        input_tokens: add(carried.input_tokens, current.input_tokens),
        output_tokens: add(carried.output_tokens, current.output_tokens),
        cached_input_tokens: add(carried.cached_input_tokens, current.cached_input_tokens),
        cache_creation_input_tokens: add(
            carried.cache_creation_input_tokens,
            current.cache_creation_input_tokens,
        ),
        reasoning_tokens: add(carried.reasoning_tokens, current.reasoning_tokens),
    }
}
