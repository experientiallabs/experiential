//! Incremental upstream relay: one provider response decoded and normalized
//! into gateway events, plus the shared collection helpers that bound and
//! classify what the relay yields. The waterfall commits a relay to one
//! deployment; the HTTP surfaces then drain it live or to completion.

use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};

use bytes::Bytes;
use futures_util::stream::BoxStream;
use futures_util::StreamExt;

use crate::codex_native_inversion::{NativeToolInverter, NativeToolTranslation};
use crate::dialects::{
    Dialect, FrameDecoder, Normalizer, MAXIMUM_RETAINED_OUTPUT_BYTES, OUTPUT_OVERFLOW_MESSAGE,
};
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{Event, Usage};
use crate::metrics::METRICS;
use crate::stop_sequences::StopSequenceGuard;
use crate::tool_search::{ToolSearchWithholder, WithheldSearchCall};
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
/// first TOKEN (the first semantic event) within the fail-fast first-token
/// bound. Headers, keepalive comments and role-only frames do not count. A
/// stalled lead deployment must not hold the request for its full per-chunk
/// timeout, so this is a transient, capacity-shaped failure that is
/// failover-eligible.
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
    /// Response-side inversion map for Codex native tools translated on a
    /// foreign wire; empty on every native-Responses route (a no-op there).
    native_tool_inverter: NativeToolInverter,
    /// Swallows every call to the gateway's tool-search tool so the caller
    /// never sees it; inert until the tool is named (see `tool_search`).
    tool_search: ToolSearchWithholder,
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
    /// Whether any body byte has arrived: stamps the time-to-first-byte
    /// histogram once. It does NOT satisfy the stall bound below: a provider
    /// can send headers, keepalive comments and role-only frames at once and
    /// still stall for minutes before its first token (2026-09-19, ~2 min
    /// medians on a lane whose first byte was instant).
    first_byte_recorded: bool,
    /// Whether the fail-fast first-token bound still applies. Armed until the
    /// waterfall COMMITS the attempt (`commit`, called at the moment the
    /// first semantic event -- content, reasoning, a tool call, an output
    /// item -- makes this attempt the answer); from then on reads are paced
    /// by the deployment's per-chunk timeout, so a slow reasoning model
    /// streams for as long as it needs once it has started answering. Armed
    /// exactly until commit keeps the stall failover-safe: a refusal delta
    /// the waterfall WITHHOLDS under refusal failover is semantic but not a
    /// commit, and a provider that stalls behind it still trips the bound.
    stall_bound_armed: bool,
    /// Fail-fast bound for the provider's first token, absolute from the dial.
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
            stall_bound_armed: true,
            first_byte_deadline,
            first_token_at: None,
            native_tool_inverter: NativeToolInverter::default(),
            tool_search: ToolSearchWithholder::default(),
            carried_usage: None,
        }
    }

    /// The waterfall committed the attempt on this relay: the first-token
    /// bound is disarmed and every later read is paced by the deployment's
    /// per-chunk timeout. Called at the commit point and nowhere else, so a
    /// semantic event the waterfall withholds (a refusal delta under refusal
    /// failover) leaves the bound armed.
    pub fn commit(&mut self) {
        self.stall_bound_armed = false;
    }

    /// The wall-clock time this relay yielded its first output token, or
    /// `None` if it has not produced one yet. Read at settlement to report
    /// the winning attempt's time-to-first-token.
    pub fn first_token_at(&self) -> Option<SystemTime> {
        self.first_token_at
    }

    /// The upstream an aggregator named as serving this stream (OpenRouter's
    /// per-chunk `provider`), read at commit to settle with the attempt.
    pub fn upstream_provider(&self) -> Option<String> {
        self.normalizer.upstream_provider().map(str::to_string)
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

    /// Carry the Codex native-tool inversion map (see
    /// `codex_native_inversion`); applied to every tool-call event this relay
    /// yields. Empty leaves every event untouched.
    pub fn set_native_tool_translation(&mut self, translation: NativeToolTranslation) {
        self.native_tool_inverter.translation = translation;
    }

    /// Name the gateway's tool-search tool (see `tool_search`): every call to
    /// it is withheld from the yielded events and accumulated for the
    /// waterfall. `None` withholds nothing.
    pub fn set_tool_search_tool_name(&mut self, tool_name: Option<String>) {
        self.tool_search.set_tool_name(tool_name);
    }

    /// How many completed calls to the tool-search tool this relay withheld
    /// and has not handed over yet.
    pub fn withheld_search_call_count(&self) -> usize {
        self.tool_search.withheld_count()
    }

    /// Whether this relay saw any call to the tool-search tool, completed or
    /// still open.
    pub fn withheld_search_call_seen(&self) -> bool {
        self.tool_search.withheld_any()
    }

    /// Hand over the withheld tool-search calls, leaving none behind.
    /// Whether the dial's search calls exceeded the withholder's bounds.
    pub fn withheld_search_overflowed(&self) -> bool {
        self.tool_search.overflowed()
    }

    pub fn take_withheld_search_calls(&mut self) -> Vec<WithheldSearchCall> {
        self.tool_search.take_withheld()
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
        // The gateway's own search tool is withheld first: it is not one of
        // the caller's tools, so it never counts toward one-call-per-turn
        // serialization and never reaches the Codex inversion or the caller.
        let Some(mut event) = self.tool_search.filter(event) else {
            return true;
        };
        if let Some(serializer) = self.tool_serializer.as_mut() {
            let Some(kept) = serializer.filter(event) else {
                return true;
            };
            event = kept;
        }
        for event in self.native_tool_inverter.filter(event) {
            match self.stop_guard.as_mut() {
                Some(guard) => self.ready.extend(guard.filter(event)),
                None => self.ready.push_back(event),
            }
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
            // Until the first semantic event the fail-fast first-token bound
            // applies -- absolute from the dial, so keepalive comments,
            // pings and role-only frames buy the provider nothing; after it,
            // each chunk is paced by the deployment's own per-chunk timeout
            // so long-running generation is never capped.
            let waiting_for_first_token = self.stall_bound_armed;
            let bound = if waiting_for_first_token {
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
                    // A first-token stall while the request deadline still has
                    // budget is the fail-fast case: classify it as a
                    // failover-eligible transient so the next rung is tried at
                    // once. A later chunk stall, or an exhausted request
                    // deadline, keeps the existing transport/deadline mapping.
                    if waiting_for_first_token && !remaining(deadline).is_zero() {
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
mod tests;

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
        cache_creation_1h_input_tokens: match (
            carried.cache_creation_input_tokens.unwrap_or(0),
            carried.cache_creation_1h_input_tokens,
            current.cache_creation_input_tokens.unwrap_or(0),
            current.cache_creation_1h_input_tokens,
        ) {
            (a, None, _, _) if a > 0 => None,
            (_, _, b, None) if b > 0 => None,
            (_, a, _, b) => add(a, b),
        },
        reasoning_tokens: add(carried.reasoning_tokens, current.reasoning_tokens),
    }
}

#[cfg(test)]
mod cache_write_tests {
    use super::{fold_usage, Usage};

    #[test]
    fn redial_preserves_unknown_ttl_until_every_write_leg_is_observed() {
        let known = Usage {
            cache_creation_input_tokens: Some(10),
            cache_creation_1h_input_tokens: Some(4),
            ..Usage::default()
        };
        let unknown = Usage {
            cache_creation_input_tokens: Some(20),
            ..Usage::default()
        };
        let folded = fold_usage(&known, known.clone());
        assert_eq!(folded.cache_creation_input_tokens, Some(20));
        assert_eq!(folded.cache_creation_1h_input_tokens, Some(8));
        assert_eq!(
            fold_usage(&known, unknown.clone()).cache_creation_1h_input_tokens,
            None
        );
        assert_eq!(
            fold_usage(&unknown, known.clone()).cache_creation_1h_input_tokens,
            None
        );
        assert_eq!(
            fold_usage(&Usage::default(), known).cache_creation_1h_input_tokens,
            Some(4)
        );
    }
}

#[cfg(test)]
#[path = "codex_native_stream_tests.rs"]
mod codex_native_stream_tests;
