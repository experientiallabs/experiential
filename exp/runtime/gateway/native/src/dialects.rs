//! Provider wire dialects: SSE normalizers mirroring the event mappers in
//! `exp.runtime.models.providers.streaming`. Upstream payloads are built by
//! the python control plane with the shared `streaming_requests` builders and
//! arrive fully formed in the admission response.
//!
//! This module owns the dialect registry, the dialect-selected frame decoder,
//! and the shared `Normalizer` state machine; each provider's frame mapping
//! lives in its own submodule as `Normalizer` methods.

mod anthropic;
mod bedrock;
mod gemini;
mod openai;

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};

use crate::errors::{Failure, FailureClass};
use crate::events::{
    require_json_object_text, simplified_event, Event, ProviderOutputItemKind, ToolAccumulator,
    Usage,
};
use crate::eventstream::EventStreamDecoder;
use crate::sse::{SseDecoder, SseEvent};

/// The upstream dialects the native engine speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    OpenAiResponses,
    AnthropicMessages,
    OpenAiCompatible,
    GeminiGenerateContent,
    BedrockConverseStream,
}

impl Dialect {
    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "openai_responses" => Some(Dialect::OpenAiResponses),
            "anthropic_messages" => Some(Dialect::AnthropicMessages),
            "openai_compatible" => Some(Dialect::OpenAiCompatible),
            "gemini_generate_content" => Some(Dialect::GeminiGenerateContent),
            "bedrock_converse_stream" => Some(Dialect::BedrockConverseStream),
            _ => None,
        }
    }
}

/// Dialect-selected incremental frame decoder over provider response bytes.
/// SSE dialects reuse the shared SSE decoder; Bedrock decodes the AWS binary
/// event-stream framing into the same frame shape.
pub enum FrameDecoder {
    Sse(SseDecoder),
    EventStream(EventStreamDecoder),
}

impl FrameDecoder {
    pub fn new(dialect: Dialect) -> Self {
        match dialect {
            Dialect::BedrockConverseStream => FrameDecoder::EventStream(EventStreamDecoder::new()),
            Dialect::OpenAiResponses
            | Dialect::AnthropicMessages
            | Dialect::OpenAiCompatible
            | Dialect::GeminiGenerateContent => FrameDecoder::Sse(SseDecoder::new()),
        }
    }

    /// Feed one network chunk, returning every complete frame it closes.
    pub fn feed(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, String> {
        match self {
            FrameDecoder::Sse(decoder) => decoder.feed(chunk),
            FrameDecoder::EventStream(decoder) => decoder.feed(chunk),
        }
    }

    /// Close the stream, recovering or rejecting trailing partial frames.
    pub fn finish(&mut self) -> Result<Option<SseEvent>, String> {
        match self {
            FrameDecoder::Sse(decoder) => decoder.finish(),
            FrameDecoder::EventStream(decoder) => decoder.finish(),
        }
    }
}

/// Aggregate per-request ceiling on retained provider output, mirroring the
/// Python engine's 64 MiB bounded-aggregation limit.
pub const MAXIMUM_RETAINED_OUTPUT_BYTES: usize = 64 * 1024 * 1024;

/// Aggregate ceiling on provider-indexed state retained while normalizing a
/// stream. Byte accounting alone cannot bound empty reasoning fragments or
/// tool starts with many distinct provider-controlled indices.
pub const MAXIMUM_RETAINED_PROVIDER_ENTRIES: usize = 4_096;

/// The sanitized message that marks an aggregate output overflow; the HTTP
/// layer maps it to the shared `provider_output_too_large` public error.
pub const OUTPUT_OVERFLOW_MESSAGE: &str = "provider output exceeded the gateway response limit";

fn malformed(message: &str) -> Failure {
    // A malformed provider response mirrors `ProviderResponseError`: never a
    // same-deployment redial, but a later certified deployment may serve it.
    Failure::new(FailureClass::MalformedResponse, message).with_retry(false, true)
}

fn refusal_failure() -> Failure {
    Failure::new(FailureClass::Refusal, "provider refused the request")
}

fn provider_stream_failed() -> Failure {
    // A provider-declared stream failure mirrors the 5xx classification.
    Failure::new(FailureClass::ProviderInternal, "provider stream failed").with_retry(true, true)
}

fn parse_object(data: &str) -> Result<Map<String, Value>, Failure> {
    match serde_json::from_str::<Value>(data) {
        Ok(Value::Object(object)) => Ok(object),
        Ok(_) => Err(malformed("provider stream event must be a JSON object")),
        Err(_) => Err(malformed("provider stream event is not valid JSON")),
    }
}

fn optional_text(object: &Map<String, Value>, key: &str, label: &str) -> Result<String, Failure> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(text)) => Ok(text.clone()),
        Some(_) => Err(malformed(&format!("{label} must be text"))),
    }
}

/// Complete one streamed tool call, defaulting a zero-argument call to `{}`.
///
/// Providers legally stream no argument fragments (or a single empty one)
/// for a call whose input is empty, so completion seeds the canonical empty
/// object and emits the seeding fragment first, keeping every downstream
/// byte verification consistent with what was streamed.
fn complete_streamed_tool(
    index: u32,
    tool: &mut ToolAccumulator,
    events: &mut Vec<Event>,
) -> Result<(), Failure> {
    tool.completed = true;
    // Only JSON function calls need the empty-object seed; custom (freeform)
    // input is legitimately empty text.
    if tool.raw_arguments.is_empty() && !tool.custom {
        tool.raw_arguments.push_str("{}");
        events.push(if tool.server {
            Event::ServerToolArgumentsDelta {
                index,
                delta: "{}".to_string(),
            }
        } else {
            Event::ToolArgumentsDelta {
                index,
                delta: "{}".to_string(),
            }
        });
    }
    let call = tool.complete().map_err(|message| malformed(&message))?;
    events.push(if tool.server {
        Event::ServerToolUseCompleted { index, call }
    } else {
        Event::ToolCallCompleted { index, call }
    });
    Ok(())
}

fn finish_open_tools(tools: &mut BTreeMap<u32, ToolAccumulator>) -> Result<Vec<Event>, Failure> {
    let mut events = Vec::new();
    for (index, tool) in tools.iter_mut() {
        if !tool.completed {
            complete_streamed_tool(*index, tool, &mut events)?;
        }
    }
    Ok(events)
}

/// Finish open tools on a stream the provider cut off at its output budget.
///
/// A call whose accumulated arguments still parse as a JSON object completes
/// normally; one left mid-fragment by the truncation is DROPPED (marked
/// completed without a `ToolCallCompleted`), because the provider never
/// finished it and the caller's remedy is a larger budget, not a retry of a
/// "malformed" provider. Only the `finish_reason == "length"` terminal may use
/// this; every other terminal keeps the strict object contract.
fn finish_open_tools_truncated(
    tools: &mut BTreeMap<u32, ToolAccumulator>,
) -> Result<Vec<Event>, Failure> {
    let mut events = Vec::new();
    for (index, tool) in tools.iter_mut() {
        if tool.completed {
            continue;
        }
        let parses = tool.custom
            || tool.raw_arguments.is_empty()
            || require_json_object_text(&tool.raw_arguments).is_ok();
        if parses {
            complete_streamed_tool(*index, tool, &mut events)?;
        } else {
            tool.completed = true;
        }
    }
    Ok(events)
}

/// Incremental normalizer of one upstream SSE stream into gateway events.
pub struct Normalizer {
    dialect: Dialect,
    tools: BTreeMap<u32, ToolAccumulator>,
    refusal_seen: bool,
    terminal: bool,
    // Whether any gateway-visible output token (content, reasoning, or a tool
    // call) has been emitted. A clean stream close after content but without a
    // terminal frame can then finish normally instead of failing malformed.
    emitted_output: bool,
    accumulated_tool_bytes: usize,
    accumulated_summary_bytes: usize,
    reasoning_summaries: BTreeMap<(u32, u32), String>,
    openai_output_items: BTreeMap<u32, (ProviderOutputItemKind, Option<String>)>,
    openai_completed_output_items: BTreeSet<u32>,
    // Anthropic accumulation.
    input_tokens: u64,
    output_tokens: u64,
    cache_read: u64,
    cache_write: u64,
    stop_reason: Option<String>,
    // OpenAI-compatible and Gemini accumulation.
    usage: Option<Usage>,
    finish_reason: Option<String>,
    // Gemini accumulation: whole function calls arrive in one part, so the
    // provider supplies no tool index; assignment order mirrors the python
    // mapper's local counter.
    gemini_tool_index: u32,
    // Fireworks-only route identity authorizing reasoning_content capture.
    reasoning_content_route_sha256: Option<String>,
}

impl Normalizer {
    pub fn new(dialect: Dialect) -> Self {
        Self::new_with_reasoning_content_route(dialect, None)
    }

    pub fn new_with_reasoning_content_route(
        dialect: Dialect,
        reasoning_content_route_sha256: Option<String>,
    ) -> Self {
        Self {
            dialect,
            tools: BTreeMap::new(),
            refusal_seen: false,
            terminal: false,
            emitted_output: false,
            accumulated_tool_bytes: 0,
            accumulated_summary_bytes: 0,
            reasoning_summaries: BTreeMap::new(),
            openai_output_items: BTreeMap::new(),
            openai_completed_output_items: BTreeSet::new(),
            input_tokens: 0,
            output_tokens: 0,
            cache_read: 0,
            cache_write: 0,
            stop_reason: None,
            usage: None,
            finish_reason: None,
            gemini_tool_index: 0,
            reasoning_content_route_sha256,
        }
    }

    /// Reserve retained-output budget for accumulated tool-argument text.
    fn reserve_tool_bytes(&mut self, additional: usize) -> Result<(), Failure> {
        self.accumulated_tool_bytes = self.accumulated_tool_bytes.saturating_add(additional);
        if self
            .accumulated_tool_bytes
            .saturating_add(self.accumulated_summary_bytes)
            > MAXIMUM_RETAINED_OUTPUT_BYTES
        {
            return Err(Failure::new(
                FailureClass::ProviderInternal,
                OUTPUT_OVERFLOW_MESSAGE,
            ));
        }
        Ok(())
    }

    /// Reserve retained-output budget for reasoning-summary verification.
    fn reserve_summary_bytes(&mut self, additional: usize) -> Result<(), Failure> {
        self.accumulated_summary_bytes = self.accumulated_summary_bytes.saturating_add(additional);
        if self
            .accumulated_tool_bytes
            .saturating_add(self.accumulated_summary_bytes)
            > MAXIMUM_RETAINED_OUTPUT_BYTES
        {
            return Err(Failure::new(
                FailureClass::ProviderInternal,
                OUTPUT_OVERFLOW_MESSAGE,
            ));
        }
        Ok(())
    }

    /// Reserve one provider-indexed state entry across tools and summaries.
    fn reserve_provider_entry(&self, exists: bool) -> Result<(), Failure> {
        if !exists
            && self
                .tools
                .len()
                .saturating_add(self.reasoning_summaries.len())
                .saturating_add(self.openai_output_items.len())
                >= MAXIMUM_RETAINED_PROVIDER_ENTRIES
        {
            return Err(Failure::new(
                FailureClass::ProviderInternal,
                OUTPUT_OVERFLOW_MESSAGE,
            ));
        }
        Ok(())
    }

    /// Reserve a new tool-call accumulator when this index is not retained.
    fn reserve_tool_entry(&self, index: u32) -> Result<(), Failure> {
        self.reserve_provider_entry(self.tools.contains_key(&index))
    }

    /// Reserve a new reasoning-summary accumulator when this key is not retained.
    fn reserve_summary_entry(&self, key: (u32, u32)) -> Result<(), Failure> {
        self.reserve_provider_entry(self.reasoning_summaries.contains_key(&key))
    }

    /// Bind one OpenAI provider output index to exactly one bounded identity.
    fn bind_openai_output_item(
        &mut self,
        output_index: u32,
        kind: ProviderOutputItemKind,
        item_id: Option<String>,
    ) -> Result<bool, Failure> {
        if let Some(existing) = self.openai_output_items.get(&output_index) {
            return if existing == &(kind, item_id) {
                Ok(false)
            } else {
                Err(malformed(
                    "OpenAI output item changed identity or type during streaming",
                ))
            };
        }
        self.reserve_provider_entry(false)?;
        self.openai_output_items
            .insert(output_index, (kind, item_id));
        Ok(true)
    }

    /// Whether a terminal event already ended the stream.
    pub fn saw_terminal(&self) -> bool {
        self.terminal
    }

    /// Fail if the stream ended without ever producing a terminal event.
    pub fn stream_ended(&self) -> Result<(), Failure> {
        if self.terminal {
            return Ok(());
        }
        Err(Failure::new(
            FailureClass::MalformedResponse,
            "provider stream ended without a terminal event",
        ))
    }

    /// Synthesize the terminal events for a stream that closed cleanly without
    /// an explicit terminal frame.
    ///
    /// Gemini legitimately ends some streams right after its last content frame
    /// without a `finishReason` frame. When content was already emitted, fold
    /// the last-seen usage and complete normally instead of rejecting a real
    /// answer as malformed; a stream that produced no content at all stays
    /// terminal-less so `stream_ended` (or the relay) still fails it closed.
    /// Returns no events when a terminal already ended the stream.
    pub fn on_stream_end(&mut self) -> Vec<Event> {
        if self.terminal || self.dialect != Dialect::GeminiGenerateContent || !self.emitted_output {
            return Vec::new();
        }
        let mut events = Vec::new();
        if let Some(usage) = self.usage.take() {
            events.push(Event::Usage(usage));
        }
        events.push(Event::Completed);
        self.terminal = true;
        events
    }

    /// Recover a Gemini stream that emitted content and then terminated
    /// *abnormally* — a broken transport read, a malformed frame, or a decoder
    /// error — rather than closing cleanly. `on_stream_end` covers the clean
    /// end (last content frame, then EOF, no terminal frame); this covers the
    /// abnormal end, where the underlying failure would otherwise discard a
    /// real partial answer.
    ///
    /// Scoped to Gemini: Gemini uniquely ends legitimate turns without a
    /// terminal frame, so a break after content is far more likely a
    /// truncated-but-usable answer than corruption. When content was already
    /// emitted, synthesize an `Incomplete` terminal (folding last-seen usage)
    /// so the caller receives the partial content with an early-termination
    /// finish reason and a retryable settlement (`incomplete`, not `failed`),
    /// and the delivered tokens still bill. Before any content there is nothing
    /// to preserve, so reclassify the abnormal end as a retryable transport
    /// failure (retry same deployment, then fail over) instead of a hard
    /// malformed reject. Any non-Gemini dialect, or a stream already terminated,
    /// keeps the original failure unchanged.
    ///
    /// A retained-output overflow is never recovered: it is a deliberate gateway
    /// limit (`provider_output_too_large`), not a provider abnormality, so
    /// converting it to `Incomplete` would deliver and bill an over-limit partial
    /// instead of surfacing the overflow — regardless of dialect or content.
    pub fn recover_abnormal_end(&mut self, failure: Failure) -> Result<Vec<Event>, Failure> {
        if failure.safe_message == OUTPUT_OVERFLOW_MESSAGE {
            return Err(failure);
        }
        if self.terminal || self.dialect != Dialect::GeminiGenerateContent {
            return Err(failure);
        }
        if !self.emitted_output {
            return Err(Failure::new(
                FailureClass::Transport,
                "provider transport failed; retry the request",
            )
            .with_retry(true, true));
        }
        let mut events = Vec::new();
        if let Some(usage) = self.usage.take() {
            events.push(Event::Usage(usage));
        }
        events.push(Event::Incomplete);
        self.terminal = true;
        Ok(events)
    }

    /// Feed one decoded SSE frame; a terminal event ends the stream.
    pub fn feed(&mut self, frame: &SseEvent) -> Result<Vec<Event>, Failure> {
        if self.terminal {
            return Ok(Vec::new());
        }
        let events = match self.dialect {
            Dialect::OpenAiResponses => self.feed_openai_responses(frame),
            Dialect::AnthropicMessages => self.feed_anthropic(frame),
            Dialect::OpenAiCompatible => self.feed_openai_compatible(frame),
            Dialect::GeminiGenerateContent => self.feed_gemini(frame),
            Dialect::BedrockConverseStream => self.feed_bedrock(frame),
        }?;
        if events.iter().any(Event::is_output_token) {
            self.emitted_output = true;
        }
        if events.iter().any(Event::is_terminal) {
            self.terminal = true;
        }
        Ok(events)
    }
}

/// Drain one raw provider byte stream through the dialect's frame decoder and
/// normalizer, mirroring the server's collection order, and return simplified
/// canonical events plus the failure that ended the stream (when one did).
/// Shared by the parity-fixture entry point and the golden-fixture tests so
/// exactly one drive loop mirrors the server.
pub fn drain_stream_fixture(dialect: Dialect, chunks: &[Vec<u8>]) -> (Vec<Value>, Option<Failure>) {
    let mut normalizer = Normalizer::new(dialect);
    let mut decoder = FrameDecoder::new(dialect);
    let mut simplified = Vec::new();
    for chunk in chunks {
        let frames = match decoder.feed(chunk) {
            Ok(frames) => frames,
            Err(message) => {
                return recover_or_report(&mut normalizer, simplified, malformed(&message))
            }
        };
        for frame in frames {
            match normalizer.feed(&frame) {
                Ok(events) => simplified.extend(events.iter().map(simplified_event)),
                Err(failure) => return recover_or_report(&mut normalizer, simplified, failure),
            }
            if normalizer.saw_terminal() {
                return (simplified, None);
            }
        }
    }
    match decoder.finish() {
        Ok(Some(frame)) => match normalizer.feed(&frame) {
            Ok(events) => simplified.extend(events.iter().map(simplified_event)),
            Err(failure) => return recover_or_report(&mut normalizer, simplified, failure),
        },
        Ok(None) => {}
        Err(message) => return recover_or_report(&mut normalizer, simplified, malformed(&message)),
    }
    if normalizer.saw_terminal() {
        return (simplified, None);
    }
    // A clean stream close after content, with no terminal frame, completes
    // normally (mirroring the relay's EOF handling) instead of failing closed.
    let synthesized = normalizer.on_stream_end();
    if !synthesized.is_empty() {
        simplified.extend(synthesized.iter().map(simplified_event));
        return (simplified, None);
    }
    match normalizer.stream_ended() {
        Ok(()) => (simplified, None),
        Err(failure) => (simplified, Some(failure)),
    }
}

/// Mirror the relay's abnormal-end handling for the fixture drive loop: route a
/// terminating failure through `recover_abnormal_end`, appending any recovered
/// terminal to the collected events, and report the (possibly reclassified)
/// failure when recovery does not apply. Keeps the fixture drain a faithful
/// mirror of `UpstreamRelay::next_event`.
fn recover_or_report(
    normalizer: &mut Normalizer,
    mut simplified: Vec<Value>,
    failure: Failure,
) -> (Vec<Value>, Option<Failure>) {
    match normalizer.recover_abnormal_end(failure) {
        Ok(events) => {
            simplified.extend(events.iter().map(simplified_event));
            (simplified, None)
        }
        Err(failure) => (simplified, Some(failure)),
    }
}

#[cfg(test)]
mod recover_abnormal_end_tests {
    use super::*;

    fn feed_text(normalizer: &mut Normalizer, text: &str) {
        let frame = SseEvent {
            event: None,
            data: serde_json::json!({
                "candidates": [{"content": {"parts": [{"text": text}]}}]
            })
            .to_string(),
        };
        let events = normalizer.feed(&frame).expect("content frame normalizes");
        assert!(events.iter().any(Event::is_output_token));
    }

    fn incoming() -> Failure {
        Failure::new(FailureClass::MalformedResponse, "boom").with_retry(false, true)
    }

    #[test]
    fn gemini_after_content_recovers_incomplete_and_folds_usage() {
        let mut normalizer = Normalizer::new(Dialect::GeminiGenerateContent);
        feed_text(&mut normalizer, "hi");
        normalizer.usage = Some(Usage {
            input_tokens: Some(5),
            output_tokens: Some(2),
            ..Usage::default()
        });
        let recovered = normalizer
            .recover_abnormal_end(incoming())
            .expect("a partial answer recovers instead of failing");
        assert!(matches!(recovered.first(), Some(Event::Usage(_))));
        assert!(matches!(recovered.last(), Some(Event::Incomplete)));
        assert!(normalizer.saw_terminal());
    }

    #[test]
    fn an_output_overflow_is_never_recovered_even_after_content() {
        // The retained-output ceiling is a deliberate gateway limit: a Gemini
        // stream that emitted content and then overflowed must still surface
        // `provider_output_too_large`, not be delivered and billed as a partial.
        let mut normalizer = Normalizer::new(Dialect::GeminiGenerateContent);
        feed_text(&mut normalizer, "hi");
        let overflow = Failure::new(FailureClass::ProviderInternal, OUTPUT_OVERFLOW_MESSAGE);
        let failure = normalizer
            .recover_abnormal_end(overflow)
            .expect_err("an overflow is not an abnormal end to salvage");
        assert_eq!(failure.safe_message, OUTPUT_OVERFLOW_MESSAGE);
        assert_eq!(failure.failure_class, FailureClass::ProviderInternal);
        assert!(!normalizer.saw_terminal());
    }

    #[test]
    fn gemini_before_content_reclassifies_to_retryable_transport() {
        let mut normalizer = Normalizer::new(Dialect::GeminiGenerateContent);
        let failure = normalizer
            .recover_abnormal_end(incoming())
            .expect_err("nothing to salvage before content");
        assert_eq!(failure.failure_class, FailureClass::Transport);
        assert!(failure.retryable_same_deployment);
        assert!(failure.failover_eligible);
        assert!(!normalizer.saw_terminal());
    }

    #[test]
    fn non_gemini_keeps_the_original_failure_even_after_content() {
        // Recovery is scoped to Gemini; an OpenAI-compatible stream that emitted
        // content and then broke keeps its original malformed classification.
        let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
        let frame = SseEvent {
            event: None,
            data: serde_json::json!({"choices": [{"delta": {"content": "hi"}}]}).to_string(),
        };
        normalizer.feed(&frame).expect("content normalizes");
        let failure = normalizer
            .recover_abnormal_end(incoming())
            .expect_err("non-gemini keeps the original failure");
        assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
        assert!(!normalizer.saw_terminal());
    }

    #[test]
    fn an_already_terminal_stream_keeps_the_original_failure() {
        let mut normalizer = Normalizer::new(Dialect::GeminiGenerateContent);
        feed_text(&mut normalizer, "hi");
        let terminal = SseEvent {
            event: None,
            data: serde_json::json!({"candidates": [{"finishReason": "STOP"}]}).to_string(),
        };
        normalizer.feed(&terminal).expect("terminal normalizes");
        assert!(normalizer.saw_terminal());
        let failure = normalizer
            .recover_abnormal_end(incoming())
            .expect_err("a terminated stream does not re-recover");
        assert_eq!(failure.failure_class, FailureClass::MalformedResponse);
    }
}
