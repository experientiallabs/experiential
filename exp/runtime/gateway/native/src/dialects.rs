//! Wire registry, frame decoding, and normalization of Python-built upstream payloads.
//! Provider frame mappings live in submodules as `Normalizer` methods.

mod anthropic;
mod bedrock;
mod deferred_tools;
mod relay_finish;
mod stream_end;
pub(in crate::dialects) use relay_finish::{
    complete_streamed_tool_or_drop_cut, drop_cut_call, finish_open_tools_relay,
};
mod gemini;
mod openai;
mod responses_logprobs;
pub(crate) use responses_logprobs::records_retained_bytes;

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
    TypesafeSystemone,
}

impl Dialect {
    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "openai_responses" => Some(Dialect::OpenAiResponses),
            "anthropic_messages" => Some(Dialect::AnthropicMessages),
            "openai_compatible" => Some(Dialect::OpenAiCompatible),
            "gemini_generate_content" => Some(Dialect::GeminiGenerateContent),
            "bedrock_converse_stream" => Some(Dialect::BedrockConverseStream),
            "typesafe_systemone" => Some(Dialect::TypesafeSystemone),
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
    Unsupported,
}

impl FrameDecoder {
    pub fn new(dialect: Dialect) -> Self {
        match dialect {
            Dialect::BedrockConverseStream => FrameDecoder::EventStream(EventStreamDecoder::new()),
            Dialect::TypesafeSystemone => FrameDecoder::Unsupported,
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
            FrameDecoder::Unsupported => Err("decision models do not stream".to_string()),
        }
    }

    /// Close the stream, recovering or rejecting trailing partial frames.
    pub fn finish(&mut self) -> Result<Option<SseEvent>, String> {
        match self {
            FrameDecoder::Sse(decoder) => decoder.finish(),
            FrameDecoder::EventStream(decoder) => decoder.finish(),
            FrameDecoder::Unsupported => Err("decision models do not stream".to_string()),
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

/// One provider-supplied discriminator (an item or event type) reduced to a
/// token safe to embed in a malformed-stream reason. The reason crosses into
/// the operator log at the failure boundary, so anything but a short
/// identifier-shaped token is replaced rather than relayed: the next unknown
/// wire shape must be diagnosable from logs without ever logging payload.
fn bounded_wire_token(candidate: &str) -> String {
    let identifier = !candidate.is_empty()
        && candidate.len() <= 64
        && candidate
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'));
    if identifier {
        candidate.to_string()
    } else {
        "non-identifier".to_string()
    }
}

fn refusal_failure() -> Failure {
    Failure::new(FailureClass::Refusal, "provider refused the request")
}

/// Longest provider-declared error detail retained for the ledger, matching
/// the python `GatewayFailure.provider_detail` bound.
const MAXIMUM_STREAM_ERROR_DETAIL_CHARS: usize = 240;

/// Reduce one provider-declared stream error to a bounded single-line detail.
///
/// A provider that opens the stream and then declares its own failure names
/// the mechanism only inside that frame; dropping it left every such kill an
/// undiagnosable "provider stream failed" (2026-09-05: gpt-6-astra streams
/// dying ~120ms post-dispatch with the reason discarded). The code reduces
/// to an identifier-shaped token; the message reduces to one bounded line
/// (cut at the first control character, whitespace collapsed) and is then
/// held to the same identifier screen the caller-facing attribution path
/// uses, so a sentence naming request-specific or infrastructure handles
/// drops while its code token survives. The detail is attached only to
/// failure classes that never relay `provider_detail` to callers (the
/// stream-failure family), so it reaches the ledger and alert samples
/// without widening the caller-facing sanitization boundary.
/// `request_words` are label-shaped values the dispatched payload itself
/// carried (its model id): a provider sentence naming the model unquoted is
/// caller-known, not infrastructure, and must not drop the whole line.
fn provider_error_detail(
    code: Option<&str>,
    message: Option<&str>,
    request_words: &[&str],
) -> Option<String> {
    let code = code
        .filter(|value| !value.is_empty())
        .map(bounded_wire_token);
    let line = message.and_then(|message| {
        let cut: String = message
            .chars()
            .take_while(|character| !character.is_control())
            .collect();
        let collapsed = cut.split_whitespace().collect::<Vec<_>>().join(" ");
        // Provider-side handles are masked, never dropped with the sentence.
        (!collapsed.is_empty())
            .then(|| crate::param_attribution::bounded_masked_line(&collapsed, request_words))
    });
    let detail = match (code, line) {
        (None, None) => return None,
        (Some(code), None) => code,
        (None, Some(line)) => line,
        (Some(code), Some(line)) => format!("{code}: {line}"),
    };
    Some(
        detail
            .chars()
            .take(MAXIMUM_STREAM_ERROR_DETAIL_CHARS)
            .collect(),
    )
}

/// Emit the structured operator line naming one provider-declared failure.
fn log_provider_declared_failure(dialect: &str, detail: &str) {
    let line = serde_json::json!({
        "event": "provider_declared_failure",
        "dialect": dialect,
        "detail": detail,
    });
    eprintln!("exp-gateway-native: {line}");
}

impl Normalizer {
    /// Label-shaped words the dispatched payload itself carried (its model
    /// id), so a provider sentence naming them is not dropped as infrastructure.
    pub fn set_request_words<I, S>(&mut self, words: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.request_words = words.into_iter().map(Into::into).collect();
    }

    /// Classify a provider failure and retain bounded detail; exact relay verdicts
    /// require the raw envelope sentence, never a metadata-derived replacement.
    fn provider_stream_failure(
        &self,
        dialect: &str,
        code: Option<&str>,
        raw_message: Option<&str>,
        relayed_message: Option<&str>,
    ) -> Failure {
        let words: Vec<&str> = self.request_words.iter().map(String::as_str).collect();
        let message = relayed_message.or(raw_message);
        let unwrapped = message.and_then(crate::rejection_shapes::relayed_decode_failure);
        let (code, message): (Option<&str>, Option<&str>) = match &unwrapped {
            Some((upstream_code, upstream_sentence)) => (
                upstream_code.as_deref().or(code),
                Some(upstream_sentence.as_str()),
            ),
            None => (code, message),
        };
        let detail = provider_error_detail(code, message, &words);
        if let Some(detail) = &detail {
            log_provider_declared_failure(dialect, detail);
        }
        let kind = crate::stream_errors::classify_stream_error_with_raw_message(
            code,
            message,
            raw_message,
        );
        // A Responses relay that refuses replayed encrypted reasoning INSIDE
        // the stream (200, then `response.failed`) carries the same repair
        // mark as the pre-stream 4xx, so the waterfall can strip and re-dial.
        let encrypted_reasoning_rejected = dialect == "openai_responses"
            && crate::rejection_shapes::refuses_encrypted_reasoning(code, message);
        crate::stream_errors::stream_failure(kind, detail)
            .with_encrypted_reasoning_rejected(encrypted_reasoning_rejected)
    }
}

fn parse_object(data: &str) -> Result<Map<String, Value>, Failure> {
    match serde_json::from_str::<Value>(data) {
        Ok(Value::Object(object)) => Ok(object),
        Ok(_) => Err(malformed("provider stream event must be a JSON object")),
        Err(error) => {
            // The frame bytes are never logged, hashed, or otherwise derived
            // from (a frame is provider payload, and even an unkeyed digest
            // of low-entropy content invites dictionary guessing): the size
            // and serde's positional description (token category and
            // line/column, never input bytes) name the parse failure and
            // correlate identical malformed frames across requests.
            let line = serde_json::json!({
                "event": "malformed_stream_frame",
                "bytes": data.len(),
                "reason": error.to_string(),
            });
            eprintln!("exp-gateway-native: {line}");
            Err(malformed(&format!(
                "provider stream event is not valid JSON: {error} ({} bytes)",
                data.len()
            )))
        }
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
    if relay_finish::drop_phantom_tool(tool) {
        return Ok(());
    }
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
    let call = tool.complete().map_err(|message| {
        // The offending bytes are never logged, hashed, or otherwise derived
        // from (tool arguments are model output and can carry tenant
        // content, and even an unkeyed digest of low-entropy arguments
        // invites dictionary guessing): the operator line carries only the
        // tool name, the size, and the parse reason, which together
        // correlate identical unparsable shapes across requests.
        let bytes = tool.raw_arguments.len() + tool.withheld_tail.len();
        let line = serde_json::json!({
            "event": "malformed_tool_arguments",
            "name": bounded_wire_token(&tool.name),
            "bytes": bytes,
            "reason": message,
        });
        eprintln!("exp-gateway-native: {line}");
        malformed(&format!("{message} ({bytes} bytes)"))
    })?;
    events.push(if tool.server {
        Event::ServerToolUseCompleted { index, call }
    } else {
        Event::ToolCallCompleted { index, call }
    });
    Ok(())
}

/// Complete one streamed tool call the provider itself marked truncated.
///
/// A call whose accumulated arguments still parse as a JSON object completes
/// normally; one left empty or mid-fragment by the output budget is DROPPED (marked
/// completed without a `ToolCallCompleted`), because the provider never
/// finished it and the caller's remedy is a larger budget, not a retry of a
/// "malformed" provider. Only a provider-declared truncation (a Chat
/// `finish_reason == "length"` terminal, or a Responses function item whose
/// own status is `incomplete`) may use this; every other completion keeps the
/// strict object contract.
fn complete_streamed_tool_truncated(
    index: u32,
    tool: &mut ToolAccumulator,
    events: &mut Vec<Event>,
) -> Result<(), Failure> {
    let parses = tool.custom || require_json_object_text(&tool.raw_arguments).is_ok();
    if parses {
        complete_streamed_tool(index, tool, events)
    } else {
        tool.completed = true;
        Ok(())
    }
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

/// Finish open tools on a stream the provider cut off at its output budget,
/// applying the truncated-call contract of [`complete_streamed_tool_truncated`].
fn finish_open_tools_truncated(
    tools: &mut BTreeMap<u32, ToolAccumulator>,
) -> Result<Vec<Event>, Failure> {
    let mut events = Vec::new();
    for (index, tool) in tools.iter_mut() {
        if !tool.completed {
            complete_streamed_tool_truncated(*index, tool, &mut events)?;
        }
    }
    Ok(events)
}

/// One open OpenAI Responses hosted-tool output item, bound by exact
/// provider identity and carrying its last-seen verbatim JSON so a stream
/// whose terminal arrives before the item's `done` can still close it.
struct OpenAiHostedItem {
    item_type: String,
    item_id: String,
    item: String,
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
    accumulated_image_bytes: usize,
    accumulated_summary_bytes: usize,
    reasoning_summaries: BTreeMap<(u32, u32), String>,
    openai_output_items: BTreeMap<u32, (ProviderOutputItemKind, Option<String>)>,
    openai_hosted_items: BTreeMap<u32, OpenAiHostedItem>,
    openai_completed_output_items: BTreeSet<u32>,
    // Anthropic accumulation.
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_read: u64,
    cache_write: u64,
    cache_write_1h: Option<u64>,
    stop_reason: Option<String>,
    // OpenAI-compatible and Gemini accumulation.
    usage: Option<Usage>,
    openai_usage: crate::events::OpenAiUsageAccumulator,
    finish_reason: Option<String>,
    // Gemini accumulation: whole function calls arrive in one part, so the
    // provider supplies no tool index; assignment order mirrors the python
    // mapper's local counter.
    gemini_tool_index: u32,
    gemini: gemini::StreamState,
    // Fireworks-only route identity authorizing reasoning_content capture.
    reasoning_content_route_sha256: Option<String>,
    // Caller-known label words (the dispatched model id) exempt from the
    // provider-identifier screen on stream-error detail.
    request_words: Vec<String>,
    // A tool call whose arguments failed to parse at its block stop, held
    // until the stop reason arrives (Anthropic `message_delta`, Bedrock
    // `messageStop` both follow the block): a budget truncation drops the
    // call and ends Incomplete; any other ending surfaces this failure.
    deferred_tool_failure: Option<Failure>,
    // Bedrock removes finished blocks from `tools`; only empty stopped blocks
    // stay pending until the final reason authorizes completion or truncation.
    bedrock_empty_stopped_tools: BTreeSet<u32>,
    anthropic_stopped_tools: BTreeSet<u32>,
    // A call the provider cut mid-fragment was dropped under an ending that
    // did not declare truncation; the terminal then settles Incomplete.
    dropped_cut_call: bool,
    // The upstream an aggregator named as serving this stream: OpenRouter
    // stamps `provider` on every Chat Completions chunk once the request
    // opted into its response metadata. First non-empty value wins; a label
    // only (bounded, printable ASCII), never content.
    upstream_provider: Option<String>,
    chat_logprobs: bool,
    responses_logprobs: bool,
}

/// Longest upstream label kept from a stream (mirrors the python settlement bound).
pub const UPSTREAM_PROVIDER_MAX_CHARS: usize = 128;

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
            accumulated_image_bytes: 0,
            accumulated_summary_bytes: 0,
            reasoning_summaries: BTreeMap::new(),
            openai_output_items: BTreeMap::new(),
            openai_hosted_items: BTreeMap::new(),
            openai_completed_output_items: BTreeSet::new(),
            input_tokens: None,
            output_tokens: None,
            cache_read: 0,
            cache_write: 0,
            cache_write_1h: None,
            stop_reason: None,
            usage: None,
            openai_usage: crate::events::OpenAiUsageAccumulator::default(),
            finish_reason: None,
            gemini_tool_index: 0,
            gemini: gemini::StreamState::default(),
            reasoning_content_route_sha256,
            request_words: Vec::new(),
            deferred_tool_failure: None,
            bedrock_empty_stopped_tools: BTreeSet::new(),
            anthropic_stopped_tools: BTreeSet::new(),
            dropped_cut_call: false,
            upstream_provider: None,
            chat_logprobs: false,
            responses_logprobs: false,
        }
    }

    pub fn enable_responses_logprobs(&mut self, enabled: bool) {
        self.responses_logprobs = enabled;
    }

    pub fn enable_chat_logprobs(&mut self, enabled: bool) {
        self.chat_logprobs = enabled;
    }

    /// The upstream an aggregator named as serving this stream, if any chunk said.
    pub fn upstream_provider(&self) -> Option<&str> {
        self.upstream_provider.as_deref()
    }

    /// Latest decoded provider meters, including reports held until terminal.
    /// The relay snapshots these before waiting for another provider frame so
    /// cancellation retains usage that has not yet become an outward event.
    pub(crate) fn observed_usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }

    /// Keep the first upstream label a chunk names; garbage (empty, over-long,
    /// or non-printable-ASCII) is ignored rather than recorded.
    pub(in crate::dialects) fn note_upstream_provider(&mut self, label: &str) {
        if self.upstream_provider.is_some() {
            return;
        }
        let trimmed = label.trim();
        if trimmed.is_empty()
            || trimmed.len() > UPSTREAM_PROVIDER_MAX_CHARS
            || !trimmed.bytes().all(|byte| (0x20..0x7f).contains(&byte))
        {
            return;
        }
        self.upstream_provider = Some(trimmed.to_string());
    }

    /// Bound aggregate image output even when images are delivered incrementally.
    fn reserve_image_bytes(&mut self, additional: usize) -> Result<(), Failure> {
        self.accumulated_image_bytes = self.accumulated_image_bytes.saturating_add(additional);
        self.reserve_tool_bytes(0)
    }

    /// Reserve retained-output budget for accumulated tool-argument text.
    fn reserve_tool_bytes(&mut self, additional: usize) -> Result<(), Failure> {
        self.accumulated_tool_bytes = self.accumulated_tool_bytes.saturating_add(additional);
        if self
            .accumulated_tool_bytes
            .saturating_add(self.accumulated_summary_bytes)
            .saturating_add(self.accumulated_image_bytes)
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
            .saturating_add(self.accumulated_image_bytes)
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
                .saturating_add(self.openai_hosted_items.len())
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
        if self.openai_hosted_items.contains_key(&output_index) {
            return Err(malformed(
                "OpenAI output item changed identity or type during streaming",
            ));
        }
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

    /// Feed one decoded SSE frame; a terminal event ends the stream.
    pub fn feed(&mut self, frame: &SseEvent) -> Result<Vec<Event>, Failure> {
        if self.terminal {
            return Ok(Vec::new());
        }
        let previous_usage = self.usage.clone();
        let result = match self.dialect {
            Dialect::OpenAiResponses => self.feed_openai_responses(frame),
            Dialect::AnthropicMessages => self.feed_anthropic(frame),
            Dialect::OpenAiCompatible => self.feed_openai_compatible(frame),
            Dialect::GeminiGenerateContent => self.feed_gemini(frame),
            Dialect::BedrockConverseStream => self.feed_bedrock(frame),
            Dialect::TypesafeSystemone => {
                Err(malformed("decision models do not serve chat streams"))
            }
        };
        let mut observed = previous_usage;
        for usage in self.usage.iter() {
            observed
                .get_or_insert_with(Usage::default)
                .merge_observed(usage);
        }
        // A later malformed field must not discard meters already decoded in
        // this frame or an earlier one. Preserve them before propagating error.
        self.usage = observed.clone();
        let mut events = result?;
        for event in &mut events {
            if let Event::Usage(usage) = event {
                let merged = observed.get_or_insert_with(Usage::default);
                merged.merge_observed(usage);
                *usage = merged.clone();
            }
        }
        self.usage = observed;
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
    let synthesized = match normalizer.on_stream_end() {
        Ok(events) => events,
        Err(failure) => return recover_or_report(&mut normalizer, simplified, failure),
    };
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
#[path = "dialects/recover_abnormal_end_tests.rs"]
mod recover_abnormal_end_tests;

#[cfg(test)]
mod stream_error_detail_tests {
    use super::*;

    fn frame(payload: serde_json::Value) -> SseEvent {
        SseEvent {
            event: None,
            data: payload.to_string(),
        }
    }

    fn failed_detail(events: &[Event]) -> Option<String> {
        match events.last() {
            Some(Event::Failed(failure)) => failure.provider_detail.clone(),
            other => panic!("expected a failed terminal, got {other:?}"),
        }
    }

    #[test]
    fn secret_shaped_words_are_masked_out_of_the_detail_line() {
        // The identifier screen treats any letter+digit label as a handle,
        // which covers key and token shapes: the word is masked and the
        // sentence around it survives, for every dialect that feeds the
        // shared detail path, Bedrock exception messages included.
        for (message, masked) in [
            (
                "Invalid key sk-abc123def provided.",
                "Invalid key [redacted] provided.",
            ),
            (
                "The access key AKIA9X7EXAMPLE is not authorized for this model.",
                "The access key [redacted] is not authorized for this model.",
            ),
            (
                "Bearer eyJhbGciOi9 was rejected.",
                "Bearer [redacted] was rejected.",
            ),
        ] {
            assert_eq!(
                provider_error_detail(None, Some(message), &[]).as_deref(),
                Some(masked),
                "a credential-shaped word must be masked: {message}"
            );
            assert_eq!(
                provider_error_detail(Some("validation_error"), Some(message), &[]).as_deref(),
                Some(format!("validation_error: {masked}").as_str()),
                "the code rides with the masked sentence: {message}"
            );
        }
    }

    #[test]
    fn provider_error_detail_is_one_bounded_line() {
        assert_eq!(provider_error_detail(None, None, &[]), None);
        assert_eq!(
            provider_error_detail(Some("server_error"), None, &[]).as_deref(),
            Some("server_error")
        );
        assert_eq!(
            provider_error_detail(
                Some("server_error"),
                Some("The model failed  to respond."),
                &[]
            )
            .as_deref(),
            Some("server_error: The model failed to respond.")
        );
        // The line cuts at the first control character: a payload dump never
        // rides past its first row.
        assert_eq!(
            provider_error_detail(None, Some("first line\nsecond line"), &[]).as_deref(),
            Some("first line")
        );
        // A hostile code reduces to the shared identifier token.
        assert_eq!(
            provider_error_detail(Some("weird code!{}"), None, &[]).as_deref(),
            Some("non-identifier")
        );
        // The composed detail never exceeds the python provider_detail bound.
        let long = "x".repeat(400);
        let bounded = provider_error_detail(Some("code"), Some(&long), &[]).expect("bounded");
        assert_eq!(bounded.chars().count(), 240);
    }

    /// Every dialect's provider-declared stream failure carries its bounded
    /// mechanism into `provider_detail` (ledger-only for these classes), so
    /// the next post-open kill is diagnosable from the ledger instead of an
    /// opaque "provider stream failed" (2026-09-05: gpt-6-astra kills ~120ms
    /// post-dispatch across 20 orgs with the reason discarded on the wire).
    #[test]
    fn responses_failed_terminal_carries_its_error_detail() {
        let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
        let events = normalizer
            .feed(&frame(serde_json::json!({
                "type": "response.failed",
                "response": {
                    "status": "failed",
                    "error": {"code": "server_error", "message": "The model failed."},
                },
            })))
            .expect("failed terminal normalizes");
        assert_eq!(
            failed_detail(&events).as_deref(),
            Some("server_error: The model failed.")
        );
    }

    #[test]
    fn responses_error_frame_carries_its_error_detail() {
        let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
        let events = normalizer
            .feed(&frame(serde_json::json!({
                "type": "error",
                "code": "rate_limit_exceeded",
                "message": "Rate limit reached for gpt-6-astra.",
                "param": null,
                "sequence_number": 1,
            })))
            .expect("error frame normalizes");
        // The model id trips the identifier screen (letters+digits label), so
        // it is masked while the code and the sentence around it survive: the
        // mechanism stays named without relaying a label-shaped word.
        assert_eq!(
            failed_detail(&events).as_deref(),
            Some("rate_limit_exceeded: Rate limit reached for [redacted].")
        );
    }

    #[test]
    fn compatible_error_frame_carries_its_error_detail() {
        let mut normalizer = Normalizer::new(Dialect::OpenAiCompatible);
        let events = normalizer
            .feed(&frame(serde_json::json!({
                "error": {"code": 502, "message": "upstream connect error"},
            })))
            .expect("error frame normalizes");
        assert_eq!(
            failed_detail(&events).as_deref(),
            Some("502: upstream connect error")
        );
    }

    #[test]
    fn anthropic_error_frame_carries_its_error_detail() {
        let mut normalizer = Normalizer::new(Dialect::AnthropicMessages);
        let events = normalizer
            .feed(&frame(serde_json::json!({
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"},
            })))
            .expect("error frame normalizes");
        assert_eq!(
            failed_detail(&events).as_deref(),
            Some("overloaded_error: Overloaded")
        );
    }

    #[test]
    fn gemini_error_envelope_carries_its_error_detail() {
        let mut normalizer = Normalizer::new(Dialect::GeminiGenerateContent);
        let events = normalizer
            .feed(&frame(serde_json::json!({
                "error": {"code": 503, "status": "UNAVAILABLE", "message": "Overloaded."},
            })))
            .expect("error envelope normalizes");
        assert_eq!(
            failed_detail(&events).as_deref(),
            Some("UNAVAILABLE: Overloaded.")
        );
    }

    #[test]
    fn unparsable_stream_frames_name_size_and_parse_position() {
        let mut normalizer = Normalizer::new(Dialect::OpenAiResponses);
        let failure = normalizer
            .feed(&SseEvent {
                event: None,
                data: "<html>502 Bad Gateway</html>".to_string(),
            })
            .expect_err("a non-JSON frame is malformed");
        assert!(
            failure.safe_message.contains("not valid JSON")
                && failure.safe_message.contains("(28 bytes)"),
            "reason must carry the parse diagnosis: {}",
            failure.safe_message
        );
    }
}
