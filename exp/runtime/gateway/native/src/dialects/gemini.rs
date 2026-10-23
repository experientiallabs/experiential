//! Gemini `streamGenerateContent` frame mapping plus its golden-fixture tests.

#[path = "gemini_usage.rs"]
mod usage;
pub(super) use usage::StreamState;

use serde_json::Value;

use super::{malformed, parse_object, Normalizer};
use crate::errors::{Failure, FailureClass};
use crate::events::{require_string, Event, ToolAccumulator};

impl Normalizer {
    /// Normalize one Gemini `streamGenerateContent` SSE frame: thought parts
    /// stay capture-only, whole function calls expand to start/arguments/completed,
    /// and the terminal candidate freezes content while metadata trailers can
    /// complete the meter. Its declared outcome is emitted once transport ends.
    /// A prompt-level block (`promptFeedback.blockReason`, delivered with no
    /// candidates at all) is the same
    /// content-free refusal a candidate-level safety finish produces.
    pub(super) fn feed_gemini(
        &mut self,
        frame: &crate::sse::SseEvent,
    ) -> Result<Vec<Event>, Failure> {
        let payload = parse_object(&frame.data)?;
        if self.metadata_drain_started().is_some() {
            if let Some(raw) = payload.get("usageMetadata").filter(|raw| !raw.is_null()) {
                self.observe_gemini_usage(raw)?;
            }
            // The first finish is authoritative. Late content, tools, errors,
            // refusals and further finish reasons cannot reopen the answer.
            return Ok(Vec::new());
        }
        if let Some(error) = payload.get("error").filter(|value| !value.is_null()) {
            // Google's error envelope ({"error":{"code":503,"status":"UNAVAILABLE"}})
            // arrives as a candidate-less frame; without this branch it reads
            // as a usage-only trailer and the stream ends malformed (or, after
            // prior output, as a synthesized completion of a failed answer).
            // It is the provider declaring failure, so it takes the shared
            // retry-then-failover classification the other dialects give it,
            // with its status and message riding as the bounded ledger detail.
            let (code, message) = match error.as_object() {
                Some(error) => (
                    error.get("status").and_then(Value::as_str),
                    error.get("message").and_then(Value::as_str),
                ),
                None => (None, None),
            };
            return Ok(vec![Event::Failed(self.provider_stream_failure(
                "gemini_generate_content",
                code,
                message,
                None,
            ))]);
        }
        if let Some(raw_usage) = payload.get("usageMetadata") {
            if !raw_usage.is_null() {
                self.observe_gemini_usage(raw_usage)?;
            }
        }
        if gemini_prompt_blocked(&payload)? {
            // A candidate-less frame naming `promptFeedback.blockReason` is the
            // provider's verdict on the PROMPT, not a usage-only trailer: no
            // candidate follows and no finishReason will name the outcome.
            // It is the same content-free refusal a candidate-level SAFETY
            // finish is, so it terminates the stream as one (no retry, no
            // failover: every lane blocks the same prompt), after the usage
            // Google reported for the prompt it counted.
            let mut events = Vec::new();
            if let Some(usage) = self.usage.take() {
                events.push(Event::Usage(usage));
            }
            // The block reason names the category the same way a candidate
            // finishReason does. Google's raw token also rides the ledger
            // detail: `OTHER` (and any reason newer than the vocabulary)
            // maps to the unnamed category, and without the token the
            // ledger read a bare "provider refused the request" that no one
            // could tell apart from any other refusal (2026-10-22,
            // gemini-4-argon under Claude Code).
            let reason = gemini_block_reason(&payload).unwrap_or_default();
            events.push(Event::Failed(gemini_refusal("blockReason", &reason)));
            return Ok(events);
        }
        let candidates = match payload.get("candidates") {
            // A usage-only trailer frame legitimately carries no candidates at
            // all; its usageMetadata was already folded above, so continue the
            // stream instead of rejecting the whole answer as malformed. This
            // mirrors the already-handled empty-candidates case below.
            None | Some(Value::Null) => return Ok(Vec::new()),
            Some(value) => value
                .as_array()
                .ok_or_else(|| malformed("Gemini candidates must be an array"))?,
        };
        let mut events = Vec::new();
        if candidates.is_empty() {
            return Ok(events);
        }
        if candidates.len() != 1 {
            return Err(malformed("Gemini stream must contain one candidate"));
        }
        let candidate = candidates[0]
            .as_object()
            .ok_or_else(|| malformed("Gemini candidate must be an object"))?;
        match candidate.get("content") {
            None | Some(Value::Null) => {}
            Some(content) => {
                // A blocked image turn ends with `"content": {}` (no parts at
                // all) beside its IMAGE_RECITATION finish: nothing to emit.
                let parts = match content
                    .as_object()
                    .ok_or_else(|| malformed("Gemini candidate content must be an object"))?
                    .get("parts")
                {
                    None | Some(Value::Null) => &Vec::new(),
                    Some(Value::Array(parts)) => parts,
                    Some(_) => return Err(malformed("Gemini candidate parts must be an array")),
                };
                for raw_part in parts {
                    let part = raw_part
                        .as_object()
                        .ok_or_else(|| malformed("Gemini candidate part must be an object"))?;
                    // Google exposes thought summaries, not full CoT. Preserve the
                    // whole signed part so a signature remains paired with its
                    // text or functionCall; the part itself is capture and replay
                    // evidence, never readable output. An unsigned thought
                    // summary carries only display text, which streams below as
                    // plaintext reasoning instead.
                    let thought = part.get("thought") == Some(&Value::Bool(true));
                    let thought_text = part
                        .get("text")
                        .and_then(Value::as_str)
                        .filter(|text| thought && !text.is_empty());
                    let retained_whole = part.contains_key("thoughtSignature")
                        || (thought && thought_text.is_none());
                    if retained_whole {
                        // A signed generated image keeps its signature and media
                        // type as evidence; its inline bytes are charged to the
                        // image budget and delivered as the image below, so
                        // retaining them here too would hold a multi-MiB image
                        // against the 1 MiB evidence bound and fail the stream.
                        // Its signature alone can pass 1 MB (an image model
                        // signs each image), so it is charged to the stream's
                        // whole retained-output budget, not a per-part bound.
                        let retained = gemini_part_without_inline_data(raw_part);
                        let bytes = crate::dialects::json_retained_bytes(&retained);
                        self.reserve_summary_bytes(bytes.max(64))?;
                        events.push(Event::GeminiThoughtPart(std::sync::Arc::new(retained)));
                    }
                    if thought {
                        // The thought summary text is the model's displayable
                        // reasoning: it renders as plaintext reasoning on rungs
                        // whose reasoning display is on.
                        if let Some(text) = thought_text {
                            // A signed part was charged whole above.
                            if !retained_whole {
                                self.reserve_summary_bytes(text.len())?;
                            }
                            events.push(Event::ReasoningTextDelta(text.to_string()));
                        }
                        continue;
                    }
                    if let Some(call) = part.get("functionCall") {
                        if !call.is_null() {
                            let signature = match part.get("thoughtSignature") {
                                None | Some(Value::Null) => None,
                                Some(Value::String(signature)) if !signature.is_empty() => {
                                    Some(signature.clone())
                                }
                                Some(_) => {
                                    return Err(malformed(
                                        "Gemini thoughtSignature must be nonempty text",
                                    ))
                                }
                            };
                            events.extend(self.gemini_tool_events(call, signature)?);
                            continue;
                        }
                    }
                    if let Some(image) = part.get("inlineData") {
                        let media_type =
                            image
                                .get("mimeType")
                                .and_then(Value::as_str)
                                .ok_or_else(|| {
                                    malformed("Gemini image requires a media type")
                                        .with_retry(false, false)
                                })?;
                        let data = image.get("data").and_then(Value::as_str).ok_or_else(|| {
                            malformed("Gemini image requires base64 data").with_retry(false, false)
                        })?;
                        let url = crate::image_output::inline_image(media_type, data)?;
                        self.reserve_image_bytes(url.len())?;
                        events.push(Event::Image(url));
                        continue;
                    }
                    match part.get("text") {
                        Some(Value::String(text)) => {
                            if !text.is_empty() {
                                events.push(Event::TextDelta(text.clone()));
                            }
                        }
                        // A part with neither visible text nor a function call
                        // (for example a bare thought signature) carries no
                        // gateway-visible output.
                        None | Some(Value::Null) => {}
                        Some(_) => return Err(malformed("Gemini text part must be text")),
                    }
                }
            }
        }
        let finish_reason = match candidate.get("finishReason") {
            None | Some(Value::Null) => return Ok(events),
            Some(Value::String(reason)) => reason.clone(),
            Some(_) => return Err(malformed("Gemini finishReason must be text")),
        };
        let terminal = match finish_reason.as_str() {
            "STOP" | "FINISH_REASON_UNSPECIFIED" => Event::Completed,
            "MAX_TOKENS" => Event::Incomplete,
            // The refusal signal table: safety, copyright,
            // and sensitive-information stops are content-free refusals. The
            // finish token names the category (RECITATION, SPII, SAFETY), so
            // the caller sees which policy declined without any provider prose.
            "SAFETY"
            | "PROHIBITED_CONTENT"
            | "BLOCKLIST"
            | "RECITATION"
            | "SPII"
            | "IMAGE_SAFETY"
            | "IMAGE_PROHIBITED_CONTENT"
            | "IMAGE_RECITATION" => Event::Failed(gemini_refusal("finishReason", &finish_reason)),
            // The model produced an unparsable call or answer. It is a
            // sampling failure, not a verdict on the request: Gemini 3.1 Pro
            // ends about half of fresh tool prompts this way when called
            // directly (live 2026-10-18), and the same request succeeds on a
            // fresh sample. So it re-dials the rung, then fails over.
            "MALFORMED_FUNCTION_CALL" | "MALFORMED_RESPONSE" => Event::Failed(
                Failure::new(
                    FailureClass::MalformedResponse,
                    "provider ended the stream with a malformed answer",
                )
                .with_retry(true, true),
            ),
            _ => Event::Failed(Failure::new(
                FailureClass::ProviderInternal,
                "provider ended the stream unexpectedly",
            )),
        };
        self.gemini.finish = Some(terminal);
        self.gemini.finished_at = Some(std::time::Instant::now());
        Ok(events)
    }

    /// Expand one complete Gemini function call into the canonical tool-call
    /// lifecycle, assigning the deterministic local call-ID fallback and the
    /// canonical compact JSON argument text. The part's `thoughtSignature`
    /// rides the call so the caller can echo it on the next turn; Gemini
    /// signs only the first call of a parallel batch, so the rest carry none.
    fn gemini_tool_events(
        &mut self,
        value: &Value,
        thought_signature: Option<String>,
    ) -> Result<Vec<Event>, Failure> {
        let call = value
            .as_object()
            .ok_or_else(|| malformed("Gemini functionCall must be an object"))?;
        let name = require_string(call, "name", "Gemini functionCall name")
            .map_err(|message| malformed(&message))?;
        let index = self.gemini_tool_index;
        self.gemini_tool_index += 1;
        let call_id = match call.get("id") {
            Some(Value::String(id)) if !id.is_empty() => id.clone(),
            _ => format!("gemini-call-{index}"),
        };
        let raw_arguments = match call.get("args") {
            None => "{}".to_string(),
            Some(arguments @ Value::Object(_)) => serde_json::to_string(arguments)
                .map_err(|_| malformed("Gemini functionCall args must be an object"))?,
            Some(_) => return Err(malformed("Gemini functionCall args must be an object")),
        };
        self.reserve_tool_bytes(raw_arguments.len())?;
        let mut tool = ToolAccumulator::new(call_id.clone(), name.clone());
        tool.raw_arguments = raw_arguments.clone();
        tool.thought_signature = thought_signature.clone();
        let completed = tool.complete().map_err(|message| malformed(&message))?;
        Ok(vec![
            Event::ToolCallStarted {
                custom: false,
                index,
                call_id,
                name,
                namespace: None,
                caller: None,
                thought_signature,
            },
            Event::ToolArgumentsDelta {
                index,
                delta: raw_arguments,
            },
            Event::ToolCallCompleted {
                index,
                call: completed,
            },
        ])
    }
}

/// The retained evidence copy of a signed part: the provider's part with
/// `inlineData.data` (generated image bytes, delivered to the caller as an
/// image event) replaced by its encoded length.
fn gemini_part_without_inline_data(part: &Value) -> Value {
    let mut retained = part.clone();
    if let Some(inline) = retained
        .get_mut("inlineData")
        .and_then(Value::as_object_mut)
    {
        if let Some(Value::String(data)) = inline.remove("data") {
            inline.insert("dataBytes".to_string(), Value::from(data.len()));
        }
    }
    retained
}

/// The content-free refusal for one Gemini verdict token (a candidate
/// `finishReason` or a `promptFeedback.blockReason`). The caller sees only the
/// bounded category; the token itself, screened to an identifier, rides
/// `provider_detail` into the ledger, which refusals never relay to callers.
fn gemini_refusal(field: &str, token: &str) -> Failure {
    Failure::refusal(crate::stream_errors::refusal_reason(Some(token), None)).with_provider_detail(
        Some(format!("{field} {}", super::bounded_wire_token(token))),
    )
}

/// Whether `promptFeedback` reports a prompt-level block: any `blockReason`
/// other than the enum's unspecified default means Google refused the prompt
/// before generating. An absent or null `promptFeedback`, or one that only
/// carries `safetyRatings`, is not a block.
fn gemini_prompt_blocked(payload: &serde_json::Map<String, Value>) -> Result<bool, Failure> {
    let feedback = match payload.get("promptFeedback") {
        None | Some(Value::Null) => return Ok(false),
        Some(value) => value
            .as_object()
            .ok_or_else(|| malformed("Gemini promptFeedback must be an object"))?,
    };
    match feedback.get("blockReason") {
        None | Some(Value::Null) => Ok(false),
        Some(Value::String(reason)) => Ok(reason != "BLOCK_REASON_UNSPECIFIED"),
        Some(_) => Err(malformed("Gemini promptFeedback.blockReason must be text")),
    }
}

/// The `promptFeedback.blockReason` token, if the frame carries one. Only
/// called after `gemini_prompt_blocked` confirmed a present, meaningful
/// reason, so a malformed shape has already been rejected.
fn gemini_block_reason(payload: &serde_json::Map<String, Value>) -> Option<String> {
    payload
        .get("promptFeedback")
        .and_then(Value::as_object)
        .and_then(|feedback| feedback.get("blockReason"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

#[cfg(test)]
#[path = "gemini_usage_tests.rs"]
mod usage_tests;

#[cfg(test)]
#[path = "gemini_tests.rs"]
mod gemini_tests;
