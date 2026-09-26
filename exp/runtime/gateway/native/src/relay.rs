//! Bounded provider events, committed to one deployment and drained by HTTP surfaces.

mod progress;

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

/// Map collection failures to public errors, honoring aggregate-output bounds.
pub fn collection_public_error(failure: &Failure) -> PublicError {
    if failure.safe_message == OUTPUT_OVERFLOW_MESSAGE {
        return PublicError::provider_output_too_large();
    }
    failure.public_error()
}

/// Approximate retained event bytes. Completed calls charge their full arguments
/// again after streamed deltas, matching the Python bounded aggregator.
pub fn event_retained_bytes(event: &Event) -> usize {
    match event {
        Event::GeminiThoughtPart(part) => crate::dialects::records_retained_bytes(part)
            .unwrap_or(MAXIMUM_RETAINED_OUTPUT_BYTES.saturating_add(1)),
        Event::TextDelta(text) | Event::RefusalDelta(text) | Event::Image(text) => text.len(),
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
        Event::ProviderOutputItemStarted { item_id, .. } => {
            64usize.saturating_add(item_id.as_deref().map_or(0, str::len))
        }
        Event::ChoiceLogprobsDelta(delta) => delta.retained_bytes(),
        Event::ProviderResponsesLogprobs {
            item_id,
            phase,
            records,
            ..
        } => crate::dialects::records_retained_bytes(records)
            .map(|size| {
                size.saturating_add(item_id.len())
                    .saturating_add(phase.len())
            })
            .unwrap_or(MAXIMUM_RETAINED_OUTPUT_BYTES.saturating_add(1)),
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

/// Classify a provider missing its first semantic token within the fail-fast bound.
/// Headers, keepalives and role-only frames do not count. Advance to the next
/// certified deployment without redialing the stalled lane, limiting its cost
/// to one first-token window rather than the full per-chunk timeout.
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
    /// Whether the first-token allowance still applies. Committed genuine
    /// output or a private token begins generation; withheld refusals do not.
    /// Generation uses a progress-idle deadline, independent of whether the
    /// waterfall can still safely fail over.
    stall_bound_armed: bool,
    /// Commitment prevents failover; it does not itself prove generation.
    committed: bool,
    /// Irreversible provider tool work has its own phase: byte-idle and the
    /// hard deadline still apply, but generation may legitimately be silent.
    provider_tools: progress::ProviderTools,
    /// Last genuine normalized generation progress, never the arrival of
    /// transport bytes or protocol scaffolding. The connection timeout bounds
    /// the gap from this instant once generation has begun.
    last_progress_at: Option<Instant>,
    /// Time handed to the consumer is not provider-idle time. Only generation
    /// idle pauses here; first-token and total deadlines remain absolute.
    yielded_at: Option<Instant>,
    /// Fail-fast bound for the provider's first token, absolute from the dial
    /// (`waterfall::first_token_allowance`: the first-token base plus the
    /// input slope; the header phase has its own, shorter first-byte bound).
    first_token_deadline: Instant,
    /// Wall-clock time this relay yielded its first output token (a content,
    /// reasoning, or tool-call delta), or `None` before any token arrives.
    /// Distinct from `first_byte_recorded`: the first byte can be an SSE frame
    /// carrying only role/lifecycle scaffolding, so time-to-first-token is
    /// stamped on the first event that carries visible model output.
    first_token_at: Option<SystemTime>,
    observation: Option<crate::settlement::Observation>,
    capture_reasoning: Option<crate::capture::reasoning::Observer>,
}

impl UpstreamRelay {
    /// Image models put a complete encoded image in one SSE frame.
    pub fn allow_image_output(&mut self) {
        if let FrameDecoder::Sse(decoder) = &mut self.decoder {
            decoder.allow_image_output();
        }
    }

    pub fn new(
        response: reqwest::Response,
        dialect: Dialect,
        first_token_deadline: Instant,
    ) -> Self {
        Self::new_with_reasoning_content_route(response, dialect, first_token_deadline, None)
    }

    pub fn new_with_reasoning_content_route(
        response: reqwest::Response,
        dialect: Dialect,
        first_token_deadline: Instant,
        reasoning_content_route_sha256: Option<String>,
    ) -> Self {
        Self::from_stream_with_reasoning_content_route(
            response.bytes_stream().boxed(),
            dialect,
            first_token_deadline,
            reasoning_content_route_sha256,
        )
    }

    #[cfg(test)]
    fn from_stream(
        stream: BoxStream<'static, reqwest::Result<Bytes>>,
        dialect: Dialect,
        first_token_deadline: Instant,
    ) -> Self {
        Self::from_stream_with_reasoning_content_route(stream, dialect, first_token_deadline, None)
    }

    fn from_stream_with_reasoning_content_route(
        stream: BoxStream<'static, reqwest::Result<Bytes>>,
        dialect: Dialect,
        first_token_deadline: Instant,
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
            committed: false,
            provider_tools: progress::ProviderTools::default(),
            last_progress_at: None,
            yielded_at: None,
            first_token_deadline,
            first_token_at: None,
            native_tool_inverter: NativeToolInverter::default(),
            tool_search: ToolSearchWithholder::default(),
            observation: None,
            capture_reasoning: None,
        }
    }

    pub(crate) fn set_observation(&mut self, observation: crate::settlement::Observation) {
        self.observation = Some(observation);
    }

    pub(crate) fn set_capture_reasoning(&mut self, observer: crate::capture::reasoning::Observer) {
        self.capture_reasoning = Some(observer);
    }

    /// Close the network body before any settlement callback is awaited.
    /// Already normalized usage and terminals remain in the guard snapshot.
    pub(crate) fn close_transport(&mut self) {
        self.stream = futures_util::stream::empty().boxed();
        self.eof = true;
        let events = self.normalizer.finish_metadata_drain();
        self.queue_events(events);
        // Drain only already decoded events through effective stop/tool rules.
        // This never polls the provider and retains a stop-adjusted terminal.
        let mut drain_failure = None;
        loop {
            match self.guard_next_pending() {
                Ok(true) => {}
                Ok(false) => break,
                Err(failure) => {
                    self.pending.clear();
                    drain_failure = Some(failure);
                    break;
                }
            }
        }
        if let Some(observation) = self.observation.take() {
            for event in &self.ready {
                // queue_events already recorded the newest meter, folded
                // across dials. A raw buffered report can be older or partial.
                if !matches!(event, Event::Usage(_)) {
                    observation.record(event);
                }
                observation.record_effective_terminal(event);
            }
        }
        if let Some(failure) = drain_failure {
            self.ready.push_back(Event::Failed(failure));
        }
    }

    fn queue_events(&mut self, events: Vec<Event>) {
        if let Some(observation) = &self.observation {
            // Several dialects retain a parsed meter until terminal encoding.
            // Accounting observes it now, even when this frame yields no event.
            if let Some(usage) = self.normalizer.observed_usage() {
                observation.record(&Event::Usage(usage.clone()));
            }
            for event in &events {
                match event {
                    Event::Usage(_) => observation.record(event),
                    Event::Failed(failure) => {
                        let failure = match self.customer_managed_provider.as_deref() {
                            Some(provider) => crate::stream_errors::customer_credential_failure(
                                failure.clone(),
                                provider,
                            ),
                            None => failure.clone(),
                        };
                        observation.record(&Event::Failed(failure));
                    }
                    _ if event.is_terminal() => observation.record(event),
                    _ => {}
                }
            }
        }
        self.pending.extend(events);
    }

    /// Pin the attempt. Structural output can commit before generation, so
    /// it keeps the full first-token allowance until genuine progress arrives.
    pub fn commit(&mut self) {
        self.committed = true;
        if self.last_progress_at.is_some() {
            self.stall_bound_armed = false;
        }
    }

    /// A private token starts generation without committing the attempt.
    /// Its buffered carrier has not escaped, so a later stall can fail over.
    pub fn private_progress(&mut self) {
        self.stall_bound_armed = false;
    }

    /// Absolute expiry checks are required even when the stream is always
    /// ready: Tokio's timeout polls a ready future before its timer.
    fn read_failure(&self, deadline: Instant, phase_timeout: Duration) -> Option<Failure> {
        if remaining(deadline).is_zero() {
            return Some(stream_timeout_failure(deadline));
        }
        if self.provider_tools.active() {
            return None;
        }
        if self.stall_bound_armed {
            return remaining(self.first_token_deadline)
                .is_zero()
                .then(first_byte_timeout_failure);
        }
        self.last_progress_at
            .filter(|last| last.elapsed() >= phase_timeout)
            .map(|_| {
                Failure::new(
                    FailureClass::Transport,
                    "provider stopped making progress; retry the request",
                )
                .with_retry(false, true)
            })
    }

    /// Prefer the normalizer's latest cumulative report on an abnormal end.
    /// Dialects that yield usage directly keep their last yielded report.
    pub fn usage_before_failure(&self, reported: Option<Usage>) -> Option<Usage> {
        self.normalizer.observed_usage().cloned().or(reported)
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

    /// Enable the probability output requested in the frozen provider payload.
    pub fn set_probability_output(&mut self, chat: bool, payload: &serde_json::Value) {
        self.normalizer.enable_chat_logprobs(
            chat && payload.get("logprobs").and_then(serde_json::Value::as_bool) == Some(true),
        );
        self.normalizer.enable_responses_logprobs(
            payload.get("top_logprobs").is_some_and(|v| !v.is_null())
                || payload
                    .get("include")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|items| items.iter().any(|v| v == "message.output_text.logprobs")),
        );
    }

    /// Enforce stop sequences before yielding events; an empty set is a no-op.
    pub fn set_stop_sequences<I, S>(&mut self, sequences: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.stop_guard = StopSequenceGuard::new(sequences);
    }

    /// Move one normalized event through the stop-sequence guard (if any)
    /// onto the ready queue.
    fn guard_next_pending(&mut self) -> Result<bool, Failure> {
        let Some(mut event) = self.pending.pop_front() else {
            return Ok(false);
        };
        if let (Some(provider), Event::Failed(failure)) =
            (self.customer_managed_provider.as_deref(), &event)
        {
            event = Event::Failed(crate::stream_errors::customer_credential_failure(
                failure.clone(),
                provider,
            ));
        }
        // Progress belongs to the provider, not the outward projection. A
        // stop-sequence match suppresses later text while still draining the
        // provider's genuine generation to its terminal usage report.
        self.provider_tools.observe(&event);
        if event.is_generation_progress() {
            self.last_progress_at = Some(Instant::now());
            if self.committed {
                self.stall_bound_armed = false;
            }
        }
        // The gateway's own search tool is withheld first: it is not one of
        // the caller's tools, so it never counts toward one-call-per-turn
        // serialization and never reaches the Codex inversion or the caller.
        let Some(mut event) = self.tool_search.filter(event) else {
            return Ok(true);
        };
        if let Some(serializer) = self.tool_serializer.as_mut() {
            let Some(kept) = serializer.filter(event) else {
                return Ok(true);
            };
            event = kept;
        }
        for event in self.native_tool_inverter.filter(event)? {
            match self.stop_guard.as_mut() {
                Some(guard) => self.ready.extend(guard.filter(event)),
                None => self.ready.push_back(event),
            }
        }
        Ok(true)
    }

    /// Route an abnormal stream termination through the normalizer's recovery.
    /// The relay is done either way, so mark EOF; when recovery applies (a
    /// Gemini stream that emitted content) the synthesized terminal is buffered
    /// for the caller to drain, otherwise the (possibly reclassified) failure
    /// propagates. See `Normalizer::recover_abnormal_end`.
    fn recover_or_fail(&mut self, failure: Failure) -> Result<(), Failure> {
        self.eof = true;
        self.stream = futures_util::stream::empty().boxed();
        let events = self.normalizer.recover_abnormal_end(failure)?;
        self.queue_events(events);
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
        if let (Some(yielded), Some(last)) =
            (self.yielded_at.take(), self.last_progress_at.as_mut())
        {
            *last += yielded.elapsed();
        }
        loop {
            if let Some(event) = self.ready.pop_front() {
                // Every yielded event exits here, so this is the one place that
                // stamps time-to-first-token: the first event carrying visible
                // model output. Prefix events peeked during commit also passed
                // through here, so the winning attempt's first token is stamped
                // whether it is later replayed from a prefix or drained live.
                if self.first_token_at.is_none() && event.is_output_token() {
                    self.first_token_at = Some(SystemTime::now());
                    if let Some(observation) = &self.observation {
                        observation.record_first_token(self.first_token_at);
                    }
                }
                if let Some(observation) = &self.observation {
                    observation.record(&event);
                    observation.record_effective_terminal(&event);
                }
                self.yielded_at = Some(Instant::now());
                if let Some(observer) = &self.capture_reasoning {
                    observer.observe(&event);
                }
                return Ok(Some(event));
            }
            if self.guard_next_pending()? {
                continue;
            }
            if self.eof {
                return Ok(None);
            }
            // Already decoded events, especially a terminal with usage, are
            // drained first. A slow downstream consumer cannot turn a received
            // terminal into a provider stall. No fresh read may bypass expiry.
            let drain_started = self.normalizer.metadata_drain_started();
            if let Some(started) = drain_started {
                // One existing body-read allowance from declared finish, never
                // renewed by trailers, keepalives or downstream backpressure.
                // EOF ends immediately; the total request deadline still wins.
                if remaining(deadline).is_zero() || started.elapsed() >= phase_timeout {
                    self.close_transport();
                    continue;
                }
            } else if let Some(failure) = self.read_failure(deadline, phase_timeout) {
                return Err(failure);
            }
            // Bytes never renew either bound. Genuine progress renews the
            // generation idle window, while the total request deadline stays
            // fixed across progress and all physical attempts.
            let progress_deadline = if let Some(started) = drain_started {
                started + phase_timeout
            } else if self.provider_tools.active() {
                Instant::now() + phase_timeout
            } else if self.stall_bound_armed {
                self.first_token_deadline
            } else {
                self.last_progress_at.expect("generation has begun") + phase_timeout
            };
            let bound = remaining(deadline).min(remaining(progress_deadline));
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
                            Ok(events) => self.queue_events(events),
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
                        Ok(events) => self.queue_events(events),
                        Err(failure) => {
                            self.recover_or_fail(failure)?;
                        }
                    }
                    continue;
                }
                Err(_) => {
                    if drain_started.is_some() {
                        self.close_transport();
                        continue;
                    }
                    return Err(self
                        .read_failure(deadline, phase_timeout)
                        .unwrap_or_else(|| stream_timeout_failure(deadline)));
                }
            };
            if drain_started.is_some_and(|started| {
                remaining(deadline).is_zero() || started.elapsed() >= phase_timeout
            }) {
                self.close_transport();
                continue;
            }
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
                    Ok(events) => self.queue_events(events),
                    Err(failure) => {
                        self.recover_or_fail(failure)?;
                        break;
                    }
                }
            }
            if self.normalizer.metadata_drain_started().is_some() {
                // An always-ready stream of trailers must not prevent the
                // request owner from cancelling its pending read.
                tokio::task::yield_now().await;
            }
        }
    }
}

impl Drop for UpstreamRelay {
    fn drop(&mut self) {
        self.close_transport();
    }
}

/// Drain one non-streaming attempt, bounding output like the Python aggregation.
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
        let next = committed
            .relay
            .next_event(deadline, phase_timeout, request_started)
            .await;
        if matches!(next, Err(_) | Ok(None) | Ok(Some(Event::Failed(_)))) {
            committed.usage = committed.relay.usage_before_failure(committed.usage.take());
        }
        match next? {
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
mod gemini_usage_tests;
#[cfg(test)]
mod progress_tests;
#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "relay_disconnect_tests.rs"]
mod disconnect_tests;

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

#[cfg(test)]
#[path = "codex_native_stream_tests.rs"]
mod codex_native_stream_tests;
