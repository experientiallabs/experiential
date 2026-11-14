//! Public Gemini `generateContent` encoding: the streamed
//! `streamGenerateContent?alt=sse` frames, the aggregated non-streaming body,
//! the usage object and Google's error envelope, all rendered from the same
//! provider-neutral events every other surface encodes.

use serde_json::{json, Map, Value};

use crate::encode::{compact_json, ReasoningCarrierCandidate, ReasoningCarrierState};
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::{Event, Usage};
use crate::reasoning_display::{DisplayJoiner, ReasoningOutput};

fn invalid_provider_stream(message: &str) -> PublicError {
    PublicError::new(502, "invalid_provider_stream", message, "api_error")
}

const REFUSAL_MESSAGE: &str = "provider refused the request";

/// The sanitized failure for provider refusals on this surface.
pub fn refusal_failure() -> Failure {
    Failure::new(FailureClass::Refusal, REFUSAL_MESSAGE)
}

/// The domain Google's `ErrorInfo.reason` is scoped to on this gateway.
const ERROR_DOMAIN: &str = "gateway.experientiallabs.ai";

/// Google's symbolic status for one HTTP status (`google.rpc.Code` names).
fn google_status(status: u16) -> &'static str {
    match status {
        400 | 413 | 422 => "INVALID_ARGUMENT",
        401 => "UNAUTHENTICATED",
        403 | 451 => "PERMISSION_DENIED",
        404 => "NOT_FOUND",
        409 => "ABORTED",
        429 => "RESOURCE_EXHAUSTED",
        499 => "CANCELLED",
        500 => "INTERNAL",
        501 => "UNIMPLEMENTED",
        502 | 503 => "UNAVAILABLE",
        504 => "DEADLINE_EXCEEDED",
        _ => "UNKNOWN",
    }
}

/// Render one sanitized public error as Google's error envelope. The
/// gateway's stable machine `code` rides `details[].reason` (Google's
/// `ErrorInfo` shape, where the official SDK and raw callers already look
/// for a reason token), the offending field as `metadata.param`, a refusal's
/// bounded category as `metadata.refusal_reason`, and a known wait as a
/// `RetryInfo` detail.
pub fn google_error_body(error: &PublicError) -> Value {
    let mut metadata = Map::new();
    if let Some(param) = error.param.as_deref().filter(|param| !param.is_empty()) {
        metadata.insert("param".to_string(), Value::String(param.to_string()));
    }
    if let Some(reason) = error.refusal_reason {
        metadata.insert(
            "refusal_reason".to_string(),
            Value::String(reason.as_str().to_string()),
        );
    }
    let mut info = json!({
        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
        "reason": error.code,
        "domain": ERROR_DOMAIN,
    });
    if !metadata.is_empty() {
        info["metadata"] = Value::Object(metadata);
    }
    let mut details = vec![info];
    if let Some(wait) = error.retry_after_seconds {
        details.push(json!({
            "@type": "type.googleapis.com/google.rpc.RetryInfo",
            "retryDelay": format!("{wait}s"),
        }));
    }
    json!({
        "error": {
            "code": error.status_code,
            "message": error.message,
            "status": google_status(error.status_code),
            "details": details,
        }
    })
}

/// Google's `finishReason` for one terminal.
fn finish_reason(terminal: &Event) -> &'static str {
    match terminal {
        Event::Incomplete => "MAX_TOKENS",
        _ => "STOP",
    }
}

/// Google's `usageMetadata` for the normalized usage: the folded ledger input
/// is `promptTokenCount` (cache reads inside it, as Google reports them, with
/// the read leg on `cachedContentTokenCount`), thinking is split out of the
/// output as `thoughtsTokenCount`, and `totalTokenCount` covers everything.
/// Unknown usage renders the three mandatory counters as zero.
pub(crate) fn generate_content_usage(usage: Option<&Usage>) -> Value {
    let usage = match usage {
        Some(usage) if usage.has_token_counts() => usage,
        _ => {
            return json!({
                "promptTokenCount": 0,
                "candidatesTokenCount": 0,
                "totalTokenCount": 0,
            })
        }
    };
    let input = usage.input_tokens.unwrap_or(0);
    let output = usage.output_tokens.unwrap_or(0);
    let thoughts = usage.reasoning_tokens.filter(|count| *count > 0);
    let candidates = output.saturating_sub(thoughts.unwrap_or(0));
    let mut object = json!({
        "promptTokenCount": input,
        "candidatesTokenCount": candidates,
        "totalTokenCount": input.saturating_add(output),
    });
    if let Some(thoughts) = thoughts {
        object["thoughtsTokenCount"] = json!(thoughts);
    }
    if let Some(cached) = usage.cached_input_tokens.filter(|count| *count > 0) {
        object["cachedContentTokenCount"] = json!(cached);
    }
    object
}

/// One `functionCall` part for a completed call; its arguments are the
/// provider's validated JSON object, its `thoughtSignature` rides along so the
/// caller echoes it on the next turn exactly as Gemini's own callers do.
fn function_call_part(call: &crate::events::CompletedToolCall) -> Result<Value, PublicError> {
    let args: Value = serde_json::from_str(&call.raw_arguments).map_err(|_| {
        invalid_provider_stream("generateContent function call arguments were not a JSON object.")
    })?;
    if !args.is_object() {
        return Err(invalid_provider_stream(
            "generateContent function call arguments were not a JSON object.",
        ));
    }
    let mut part = json!({
        "functionCall": {"id": call.call_id, "name": call.name, "args": args},
    });
    if let Some(signature) = call.thought_signature.as_deref() {
        part["thoughtSignature"] = Value::String(signature.to_string());
    }
    Ok(part)
}

/// One `inlineData` part for a generated image (a validated data URL).
fn inline_data_part(url: &str) -> Result<Value, PublicError> {
    let rest = url
        .strip_prefix("data:")
        .ok_or_else(|| invalid_provider_stream("generated image was not a data URL."))?;
    let (media_type, data) = rest
        .split_once(";base64,")
        .ok_or_else(|| invalid_provider_stream("generated image was not base64."))?;
    Ok(json!({"inlineData": {"mimeType": media_type, "data": data}}))
}

/// The parts one event contributes, in provider order, with adjacent text
/// (and adjacent thought text) merged into one part.
#[derive(Default)]
struct PartsBuilder {
    parts: Vec<Value>,
    display: DisplayJoiner,
}

impl PartsBuilder {
    fn text(&mut self, delta: &str, thought: bool) {
        if delta.is_empty() {
            return;
        }
        if let Some(last) = self.parts.last_mut() {
            let same_kind = last.get("text").is_some()
                && last.get("thoughtSignature").is_none()
                && (last.get("thought") == Some(&Value::Bool(true))) == thought;
            if same_kind {
                if let Some(Value::String(text)) = last.get_mut("text") {
                    text.push_str(delta);
                    return;
                }
            }
        }
        let mut part = json!({"text": delta});
        if thought {
            part["thought"] = Value::Bool(true);
        }
        self.parts.push(part);
    }

    fn push(&mut self, part: Value) {
        self.parts.push(part);
    }

    fn take(&mut self) -> Vec<Value> {
        std::mem::take(&mut self.parts)
    }
}

/// Stateful `streamGenerateContent?alt=sse` encoder: one candidate, every
/// frame a whole `GenerateContentResponse`, the terminal frame carrying the
/// `finishReason` and `usageMetadata`.
pub struct GenerateContentEncoder {
    response_id: String,
    model: String,
    ignored_parameters: Vec<String>,
    include_thoughts: bool,
    started: bool,
    terminal: bool,
    saw_tool_use: bool,
    refusal_seen: bool,
    usage: Option<Usage>,
    tool_identities: std::collections::HashMap<u32, (String, String)>,
    tool_arguments: std::collections::HashMap<u32, String>,
    reasoning: ReasoningCarrierState,
    reasoning_content_carrier: Option<String>,
    reasoning_output: ReasoningOutput,
    parts: PartsBuilder,
}

impl GenerateContentEncoder {
    /// Build an encoder disclosing the controls route shaping omitted.
    pub fn new(
        request_id: &str,
        model: &str,
        ignored_parameters: Vec<String>,
        include_thoughts: bool,
    ) -> Self {
        Self {
            response_id: crate::encode::stable_public_id("resp", request_id),
            model: model.to_string(),
            ignored_parameters,
            include_thoughts,
            started: false,
            terminal: false,
            saw_tool_use: false,
            refusal_seen: false,
            usage: None,
            tool_identities: std::collections::HashMap::new(),
            tool_arguments: std::collections::HashMap::new(),
            reasoning: ReasoningCarrierState::default(),
            reasoning_content_carrier: None,
            reasoning_output: ReasoningOutput::default(),
            parts: PartsBuilder::default(),
        }
    }

    /// Attach the authenticated carrier before the terminal is encoded; a
    /// tool turn's hidden reasoning leaves only as this opaque thought part.
    pub fn set_reasoning_content_carrier(&mut self, carrier: String) {
        self.reasoning_content_carrier = Some(carrier);
    }

    /// Whether the rung's readable reasoning may be shown as thought parts.
    pub fn set_reasoning_output(&mut self, output: ReasoningOutput) {
        self.reasoning_output = output;
    }

    /// Return the validated carrier candidate accumulated by a live stream.
    pub fn reasoning_carrier_candidate(
        &self,
    ) -> Result<Option<ReasoningCarrierCandidate>, PublicError> {
        self.reasoning.candidate()
    }

    pub fn saw_terminal(&self) -> bool {
        self.terminal
    }

    /// Whether the provider's output was a refusal: the terminal then renders
    /// as Google's error frame and the attempt settles as the typed refusal.
    pub fn refusal_seen(&self) -> bool {
        self.refusal_seen
    }

    /// Whether any caller-visible part was emitted so far.
    pub fn has_content(&self) -> bool {
        self.started || !self.parts.parts.is_empty()
    }

    fn frame(&mut self, mut body: Value) -> String {
        body["responseId"] = Value::String(self.response_id.clone());
        body["modelVersion"] = Value::String(self.model.clone());
        if !self.started {
            self.started = true;
            if !self.ignored_parameters.is_empty() {
                body["x-experiential-ignored-parameters"] = json!(self.ignored_parameters);
            }
        }
        format!("data: {}\n\n", compact_json(&body))
    }

    /// Flush the pending parts as one content frame, if any.
    fn flush(&mut self) -> Vec<String> {
        let parts = self.parts.take();
        if parts.is_empty() {
            return Vec::new();
        }
        let body = json!({
            "candidates": [{"content": {"role": "model", "parts": parts}, "index": 0}],
        });
        vec![self.frame(body)]
    }

    fn error_frame(&mut self, failure: &Failure) -> String {
        let body = google_error_body(&failure.public_error());
        format!("data: {}\n\n", compact_json(&body))
    }

    /// Encode one ordered normalized provider event into zero or more frames.
    pub fn feed(&mut self, event: &Event) -> Result<Vec<String>, PublicError> {
        if self.terminal {
            return Err(invalid_provider_stream(
                "generateContent stream received an event after its terminal.",
            ));
        }
        self.reasoning.observe(event)?;
        match event {
            Event::ProviderResponsesLogprobs { .. } | Event::ChoiceLogprobsDelta(_) => Err(
                invalid_provider_stream("Token probabilities cannot be projected on this surface."),
            ),
            Event::TextDelta(text) | Event::ProviderTextDelta { delta: text, .. } => {
                self.parts.text(text, false);
                Ok(self.flush())
            }
            Event::Image(url) => {
                let part = inline_data_part(url)?;
                self.parts.push(part);
                Ok(self.flush())
            }
            Event::RefusalDelta(_) | Event::ProviderRefusalDelta { .. } => {
                // Google has no refusal part; the refusal is reported as one
                // sanitized terminal error instead.
                self.refusal_seen = true;
                Ok(Vec::new())
            }
            Event::ReasoningContentDelta { .. }
            | Event::ReasoningTextDelta(_)
            | Event::ReasoningSummaryDelta { .. }
            | Event::ThinkingDelta { .. } => {
                if self.include_thoughts {
                    if let Some(text) = self.display_delta(event) {
                        self.parts.text(&text, true);
                        return Ok(self.flush());
                    }
                }
                Ok(Vec::new())
            }
            Event::ToolCallStarted {
                index,
                call_id,
                name,
                custom,
                ..
            } => {
                if *custom {
                    return Err(invalid_provider_stream(
                        "generateContent cannot represent a custom tool call.",
                    ));
                }
                self.tool_identities
                    .insert(*index, (call_id.clone(), name.clone()));
                self.tool_arguments.insert(*index, String::new());
                Ok(Vec::new())
            }
            Event::ToolArgumentsDelta { index, delta } => {
                // A completed call leaves the maps, so a late delta fails here too.
                let accumulated = self.tool_arguments.get_mut(index).ok_or_else(|| {
                    invalid_provider_stream(
                        "generateContent tool arguments arrived outside an active tool call.",
                    )
                })?;
                accumulated.push_str(delta);
                Ok(Vec::new())
            }
            Event::ToolCallCompleted { index, call } => {
                let identity = self.tool_identities.remove(index).ok_or_else(|| {
                    invalid_provider_stream(
                        "generateContent tool completion omitted its started tool call.",
                    )
                })?;
                let streamed = self.tool_arguments.remove(index).unwrap_or_default();
                if identity != (call.call_id.clone(), call.name.clone())
                    || streamed != call.raw_arguments
                {
                    return Err(invalid_provider_stream(
                        "generateContent tool completion changed streamed identity or bytes.",
                    ));
                }
                self.saw_tool_use = true;
                let part = function_call_part(call)?;
                self.parts.push(part);
                Ok(self.flush())
            }
            Event::ServerToolUseStarted { .. }
            | Event::ServerToolArgumentsDelta { .. }
            | Event::ServerToolUseCompleted { .. }
            | Event::ServerToolResult { .. }
            | Event::HostedToolItemStarted { .. }
            | Event::HostedToolItemProgress { .. }
            | Event::HostedToolItemCompleted { .. } => Err(invalid_provider_stream(
                "generateContent cannot represent a provider-hosted tool item.",
            )),
            Event::ProviderOutputItemStarted { .. }
            | Event::ProviderOutputItemCompleted { .. }
            | Event::GeminiThoughtPart(_)
            | Event::EncryptedReasoning { .. }
            | Event::ThinkingSignature { .. }
            | Event::RedactedThinking { .. }
            | Event::TextBlockStarted { .. }
            | Event::CitationDelta { .. }
            | Event::ProviderTextAnnotation { .. }
            | Event::SafeguardResults(_) => Ok(Vec::new()),
            Event::Usage(usage) => {
                if usage.has_token_counts() {
                    self.usage = Some(usage.clone());
                }
                Ok(Vec::new())
            }
            Event::Completed
            | Event::Incomplete
            | Event::StoppedAtSequence(_)
            | Event::PausedTurn => {
                self.terminal = true;
                if self.refusal_seen {
                    return Ok(vec![self.error_frame(&refusal_failure())]);
                }
                if matches!(event, Event::Completed | Event::StoppedAtSequence(_))
                    && self.reasoning.candidate()?.is_some()
                    && !self.reasoning_output.exposed
                {
                    // The carrier is known only once every tool call has
                    // completed, so it travels as one trailing opaque thought
                    // part the next turn echoes verbatim.
                    let carrier = self.reasoning_content_carrier.clone().ok_or_else(|| {
                        invalid_provider_stream(
                            "generateContent reasoning was not sealed by the gateway authority.",
                        )
                    })?;
                    self.parts.push(json!({"text": carrier, "thought": true}));
                }
                let parts = self.parts.take();
                let mut candidate = json!({"finishReason": finish_reason(event), "index": 0});
                if !parts.is_empty() {
                    candidate["content"] = json!({"role": "model", "parts": parts});
                }
                let body = json!({
                    "candidates": [candidate],
                    "usageMetadata": generate_content_usage(self.usage.as_ref()),
                });
                Ok(vec![self.frame(body)])
            }
            Event::Failed(failure) => {
                self.terminal = true;
                Ok(vec![self.error_frame(failure)])
            }
        }
    }

    fn display_delta(&mut self, event: &Event) -> Option<String> {
        self.parts
            .display
            .delta(event, self.reasoning_output.exposed)
    }
}

/// The terminal outcome aggregated from one generateContent event stream.
pub struct AggregatedGenerateContent {
    pub body: Value,
    pub failure: Option<Failure>,
    pub usage: Option<Usage>,
    pub incomplete: bool,
    pub tool_names: Vec<String>,
    /// Whether the candidate carries no part at all.
    pub empty: bool,
}

/// Build one non-streaming `GenerateContentResponse` from ordered events, the
/// aggregate counterpart of [`GenerateContentEncoder`]: the same part
/// sequence, the same usage, the same trailing carrier on a tool turn.
#[allow(clippy::too_many_arguments)]
pub fn completed_generate_content_body(
    request_id: &str,
    model: &str,
    events: &[Event],
    ignored_parameters: &[String],
    reasoning_content_carrier: Option<&str>,
    reasoning_output: ReasoningOutput,
    include_thoughts: bool,
) -> Result<AggregatedGenerateContent, PublicError> {
    let terminal = events
        .iter()
        .rev()
        .find(|event| event.is_terminal())
        .ok_or_else(|| {
            PublicError::new(
                502,
                "all_routes_failed",
                "Provider stream ended without a terminal result.",
                "api_error",
            )
        })?;
    let usage = events.iter().rev().find_map(|event| match event {
        Event::Usage(candidate) if candidate.has_token_counts() => Some(candidate.clone()),
        _ => None,
    });
    let mut tool_names: Vec<String> = Vec::new();
    for event in events {
        if let Event::ToolCallCompleted { call, .. } = event {
            if !tool_names.contains(&call.name) {
                tool_names.push(call.name.clone());
            }
        }
    }
    if let Event::Failed(failure) = terminal {
        return Ok(AggregatedGenerateContent {
            body: Value::Null,
            failure: Some(failure.clone()),
            usage,
            incomplete: false,
            tool_names,
            empty: true,
        });
    }
    let incomplete = matches!(terminal, Event::Incomplete);
    if events.iter().any(|event| {
        matches!(
            event,
            Event::RefusalDelta(_) | Event::ProviderRefusalDelta { .. }
        )
    }) {
        return Ok(AggregatedGenerateContent {
            body: Value::Null,
            failure: Some(refusal_failure()),
            usage,
            incomplete,
            tool_names,
            empty: true,
        });
    }
    let mut parts = PartsBuilder::default();
    let mut state = ReasoningCarrierState::default();
    for event in events {
        state.observe(event)?;
        match event {
            Event::TextDelta(text) | Event::ProviderTextDelta { delta: text, .. } => {
                parts.text(text, false)
            }
            Event::Image(url) => parts.push(inline_data_part(url)?),
            Event::ToolCallCompleted { call, .. } => {
                if call.custom {
                    return Err(invalid_provider_stream(
                        "generateContent cannot represent a custom tool call.",
                    ));
                }
                parts.push(function_call_part(call)?);
            }
            Event::ReasoningContentDelta { .. }
            | Event::ReasoningTextDelta(_)
            | Event::ReasoningSummaryDelta { .. }
            | Event::ThinkingDelta { .. }
                if include_thoughts =>
            {
                if let Some(text) = parts.display.delta(event, reasoning_output.exposed) {
                    parts.text(&text, true);
                }
            }
            Event::ServerToolUseCompleted { .. }
            | Event::HostedToolItemCompleted { .. }
            | Event::ChoiceLogprobsDelta(_)
            | Event::ProviderResponsesLogprobs { .. } => {
                return Err(invalid_provider_stream(
                    "generateContent cannot represent a provider-hosted tool item or token probabilities.",
                ))
            }
            _ => {}
        }
    }
    let reasoning = if matches!(terminal, Event::Completed | Event::StoppedAtSequence(_)) {
        state.candidate()?
    } else {
        None
    };
    if reasoning.is_some() && !reasoning_output.exposed {
        let carrier = reasoning_content_carrier.ok_or_else(|| {
            invalid_provider_stream(
                "generateContent reasoning was not sealed by the gateway authority.",
            )
        })?;
        parts.push(json!({"text": carrier, "thought": true}));
    }
    let parts = parts.take();
    let empty = parts.is_empty();
    let mut body = json!({
        "candidates": [{
            "content": {"role": "model", "parts": parts},
            "finishReason": finish_reason(terminal),
            "index": 0,
        }],
        "usageMetadata": generate_content_usage(usage.as_ref()),
        "responseId": crate::encode::stable_public_id("resp", request_id),
        "modelVersion": model,
    });
    if !ignored_parameters.is_empty() {
        body["x-experiential-ignored-parameters"] = json!(ignored_parameters);
    }
    Ok(AggregatedGenerateContent {
        body,
        failure: None,
        usage,
        incomplete,
        tool_names,
        empty,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::CompletedToolCall;

    fn usage(input: u64, output: u64, reasoning: Option<u64>, cached: Option<u64>) -> Usage {
        Usage {
            input_tokens: Some(input),
            output_tokens: Some(output),
            cached_input_tokens: cached,
            cache_creation_input_tokens: None,
            cache_creation_1h_input_tokens: None,
            cache_creation_12h_input_tokens: None,
            reasoning_tokens: reasoning,
            image_output_tokens: None,
            billed_units: None,
            cache_writes_read_back: false,
        }
    }

    fn frames_payloads(frames: &[String]) -> Vec<Value> {
        frames
            .iter()
            .map(|frame| {
                let data = frame.strip_prefix("data: ").unwrap().trim_end();
                serde_json::from_str(data).unwrap()
            })
            .collect()
    }

    #[test]
    fn usage_splits_thoughts_out_of_output_and_keeps_cache_reads_inside_input() {
        let value = generate_content_usage(Some(&usage(10, 7, Some(3), Some(4))));
        assert_eq!(
            value,
            json!({
                "promptTokenCount": 10,
                "candidatesTokenCount": 4,
                "totalTokenCount": 17,
                "thoughtsTokenCount": 3,
                "cachedContentTokenCount": 4,
            })
        );
        assert_eq!(
            generate_content_usage(None),
            json!({"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0})
        );
    }

    #[test]
    fn stream_merges_text_and_ends_with_finish_and_usage() {
        let mut encoder =
            GenerateContentEncoder::new("req-1", "coding", vec!["a->b".into()], false);
        let mut frames = Vec::new();
        frames.extend(encoder.feed(&Event::TextDelta("hello ".into())).unwrap());
        frames.extend(encoder.feed(&Event::TextDelta("world".into())).unwrap());
        frames.extend(
            encoder
                .feed(&Event::Usage(usage(2, 2, None, None)))
                .unwrap(),
        );
        frames.extend(encoder.feed(&Event::Completed).unwrap());
        let payloads = frames_payloads(&frames);
        assert_eq!(payloads.len(), 3);
        assert_eq!(
            payloads[0]["candidates"][0]["content"]["parts"][0]["text"],
            "hello "
        );
        assert_eq!(
            payloads[0]["x-experiential-ignored-parameters"],
            json!(["a->b"])
        );
        assert!(payloads[1]
            .get("x-experiential-ignored-parameters")
            .is_none());
        assert_eq!(payloads[2]["candidates"][0]["finishReason"], "STOP");
        assert_eq!(payloads[2]["usageMetadata"]["totalTokenCount"], 4);
        assert!(payloads.iter().all(|p| p["modelVersion"] == "coding"));
        assert!(encoder.saw_terminal());
    }

    #[test]
    fn tool_calls_render_as_function_call_parts_with_their_signature() {
        let call = CompletedToolCall {
            call_id: "call-1".into(),
            name: "search".into(),
            namespace: None,
            caller: None,
            thought_signature: Some("sig".into()),
            provider_item_id: None,
            provider_status: None,
            raw_arguments: r#"{"q":"x"}"#.into(),
            custom: false,
        };
        let events = vec![
            Event::ToolCallStarted {
                custom: false,
                index: 0,
                call_id: "call-1".into(),
                name: "search".into(),
                namespace: None,
                caller: None,
                thought_signature: Some("sig".into()),
            },
            Event::ToolArgumentsDelta {
                index: 0,
                delta: r#"{"q":"x"}"#.into(),
            },
            Event::ToolCallCompleted { index: 0, call },
            Event::Usage(usage(3, 1, None, None)),
            Event::Completed,
        ];
        let aggregated = completed_generate_content_body(
            "req-2",
            "coding",
            &events,
            &[],
            None,
            ReasoningOutput::default(),
            false,
        )
        .unwrap();
        let part = &aggregated.body["candidates"][0]["content"]["parts"][0];
        assert_eq!(part["functionCall"]["name"], "search");
        assert_eq!(part["functionCall"]["args"], json!({"q": "x"}));
        assert_eq!(part["thoughtSignature"], "sig");
        assert_eq!(aggregated.tool_names, vec!["search".to_string()]);
        let mut encoder = GenerateContentEncoder::new("req-2", "coding", Vec::new(), false);
        let mut frames = Vec::new();
        for event in &events {
            frames.extend(encoder.feed(event).unwrap());
        }
        let payloads = frames_payloads(&frames);
        assert_eq!(
            payloads[0]["candidates"][0]["content"]["parts"][0]["functionCall"]["id"],
            "call-1"
        );
    }

    #[test]
    fn tool_argument_deltas_outside_an_active_call_fail_the_stream() {
        let mut encoder = GenerateContentEncoder::new("req-4", "coding", Vec::new(), false);
        let stray = Event::ToolArgumentsDelta {
            index: 0,
            delta: "{".into(),
        };
        assert!(encoder.feed(&stray).is_err());
        let call = CompletedToolCall {
            call_id: "call-1".into(),
            name: "search".into(),
            namespace: None,
            caller: None,
            thought_signature: None,
            provider_item_id: None,
            provider_status: None,
            raw_arguments: "{}".into(),
            custom: false,
        };
        let mut encoder = GenerateContentEncoder::new("req-4", "coding", Vec::new(), false);
        encoder
            .feed(&Event::ToolCallStarted {
                custom: false,
                index: 0,
                call_id: "call-1".into(),
                name: "search".into(),
                namespace: None,
                caller: None,
                thought_signature: None,
            })
            .unwrap();
        encoder
            .feed(&Event::ToolArgumentsDelta {
                index: 0,
                delta: "{}".into(),
            })
            .unwrap();
        encoder
            .feed(&Event::ToolCallCompleted { index: 0, call })
            .unwrap();
        assert!(encoder.feed(&stray).is_err());
    }

    #[test]
    fn refusals_and_failures_are_google_error_frames() {
        let mut encoder = GenerateContentEncoder::new("req-3", "coding", Vec::new(), false);
        encoder.feed(&Event::RefusalDelta("no".into())).unwrap();
        let frames = encoder.feed(&Event::Completed).unwrap();
        let payloads = frames_payloads(&frames);
        assert_eq!(payloads[0]["error"]["code"], 400);
        assert_eq!(payloads[0]["error"]["status"], "INVALID_ARGUMENT");
        assert_eq!(payloads[0]["error"]["details"][0]["reason"], "refusal");
        let mut error =
            PublicError::new(429, "insufficient_quota", "slow down", "rate_limit_error");
        error.retry_after_seconds = Some(5);
        error.param = Some("contents".into());
        let body = google_error_body(&error);
        assert_eq!(body["error"]["status"], "RESOURCE_EXHAUSTED");
        assert_eq!(body["error"]["details"][0]["metadata"]["param"], "contents");
        assert_eq!(body["error"]["details"][1]["retryDelay"], "5s");
    }

    #[test]
    fn thoughts_render_only_when_asked_for() {
        let events = vec![
            Event::ReasoningTextDelta("thinking".into()),
            Event::TextDelta("answer".into()),
            Event::Completed,
        ];
        let shown = completed_generate_content_body(
            "r",
            "m",
            &events,
            &[],
            None,
            ReasoningOutput {
                exposed: false,
                displayed: true,
            },
            true,
        )
        .unwrap();
        let parts = shown.body["candidates"][0]["content"]["parts"]
            .as_array()
            .unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["thought"], true);
        let hidden = completed_generate_content_body(
            "r",
            "m",
            &events,
            &[],
            None,
            ReasoningOutput {
                exposed: false,
                displayed: true,
            },
            false,
        )
        .unwrap();
        assert_eq!(
            hidden.body["candidates"][0]["content"]["parts"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }
}
