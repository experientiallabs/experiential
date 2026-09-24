//! The OpenAI-compatible Chat Completions frame mapping (split from
//! `openai.rs` for the module line budget): incremental deltas, tool-call
//! fragments, exposure-gated reasoning content, and the `[DONE]` terminal.

use serde_json::Value;

use super::super::{
    finish_open_tools_relay, finish_open_tools_truncated, malformed, parse_object, Normalizer,
};
use crate::errors::Failure;
use crate::events::{require_u64, Event, ToolAccumulator};

/// One optional wire text: absent or null reads as `None`, text as itself,
/// and any other JSON type is the malformed shape it always was.
fn wire_text(value: Option<&Value>, label: &str) -> Result<Option<String>, Failure> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.clone())),
        Some(_) => Err(malformed(&format!("{label} must be text"))),
    }
}

/// Whether a choices-less chunk carries nothing a decoder would need: no
/// top-level key that any OpenAI-family shape uses for content, a finish, a
/// tool call, or an error. Everything else on such a frame is envelope
/// metadata — identity, timing, usage, Azure's prompt-filter report, and
/// provider-specific extras such as Novita's `sla_metrics` (a trailing
/// `choices: null` chunk with no usage; 48 attempts on 2026-09-15 failed a
/// finished stream closed on it under the earlier fixed allowlist). An
/// allowlist cannot keep up with what relays append; the content keys are
/// the closed set worth guarding.
fn is_metadata_only_frame(payload: &serde_json::Map<String, Value>) -> bool {
    const CONTENT_KEYS: [&str; 14] = [
        "delta",
        "message",
        "content",
        "text",
        "tool_calls",
        "function_call",
        "finish_reason",
        "refusal",
        "reasoning_content",
        "completion",
        "output",
        "candidates",
        "error",
        "detail",
    ];
    !payload
        .keys()
        .any(|key| CONTENT_KEYS.contains(&key.as_str()))
}

/// Process-wide mint counter: two streams decoded in the same clock tick
/// (or a clock that cannot be read) still receive distinct ids.
static SYNTHESIZED_CALL_IDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Mint a call id for a relay that streamed none. A client pairs its tool
/// result by this id and nothing else, and providers replay it, so it must
/// never collide: the wall clock separates turns, the process id separates
/// worker processes, the counter separates concurrent streams inside one, and
/// the tool index separates calls within a stream. Short enough for every
/// provider's replay bound.
fn synthesized_call_id(index: u32) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or(0);
    let serial = SYNTHESIZED_CALL_IDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let process = std::process::id();
    format!("call_gw{index}_{nanos:x}_{process:x}_{serial:x}")
}

/// At most this many key names, each cut to this many characters, describe
/// an unrecognized frame shape in the malformed reason.
const FRAME_KEY_NAMES_LIMIT: usize = 8;
const FRAME_KEY_NAME_CHARS: usize = 32;

/// The sorted, bounded key names of one frame: its SHAPE for the ledger, with
/// no value ever read. A key is provider-supplied text, so only an
/// identifier-shaped one (ASCII alphanumerics, `_`, `.`, `-`) is named; any
/// other key (a newline, an ANSI escape, a delimiter) reads as
/// `non-identifier` so the reason stays one clean line.
fn frame_key_names(payload: &serde_json::Map<String, Value>) -> String {
    let mut keys: Vec<String> = payload
        .keys()
        .map(|key| {
            let identifier = !key.is_empty()
                && key
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'));
            if identifier {
                key.chars().take(FRAME_KEY_NAME_CHARS).collect()
            } else {
                "non-identifier".to_string()
            }
        })
        .collect();
    keys.sort();
    let shown = keys.len().min(FRAME_KEY_NAMES_LIMIT);
    let mut line = keys[..shown].join(", ");
    if keys.len() > shown {
        line.push_str(", …");
    }
    if line.is_empty() {
        line.push_str("none");
    }
    line
}

impl Normalizer {
    pub(in crate::dialects) fn feed_openai_compatible(
        &mut self,
        frame: &crate::sse::SseEvent,
    ) -> Result<Vec<Event>, Failure> {
        if frame.data == "[DONE]" {
            return self.finish_openai_compatible_stream();
        }
        let payload = parse_object(&frame.data)?;
        // An OpenAI-compatible relay declaring failure inside the stream (or
        // an error-shaped body answered under HTTP 200 with no `choices`)
        // names the mechanism only here; the bounded detail rides the failure
        // into the ledger. The shared envelope reader covers every spelling
        // the family answers: the documented `error` object, xAI's string
        // `error`, and the flat vLLM / Novita / FastAPI objects.
        // The reader itself decides what is error-shaped: a frame with a
        // non-null `error`, or one carrying no `choices` that the reader
        // recognizes (a flat object needs an error marker, so an ordinary
        // chunk is never read as a failure).
        let document = Value::Object(payload.clone());
        let envelope = if payload.get("error").is_some_and(|value| !value.is_null())
            || !payload.contains_key("choices")
        {
            crate::error_envelope::openai_family_envelope(&document)
        } else {
            None
        };
        let raw_message = envelope.as_ref().and_then(|envelope| envelope.message);
        if envelope.is_some() {
            let (code, message) = match envelope {
                Some(envelope) => {
                    let message = envelope.message.map(|message| {
                        // An aggregator's generic sentence yields to the
                        // upstream provider's own (OpenRouter metadata.raw).
                        envelope
                            .error_object
                            .and_then(|error| {
                                crate::param_attribution::upstream_relayed_message(error, message)
                            })
                            .unwrap_or_else(|| message.to_string())
                    });
                    (envelope.code, message)
                }
                None => (None, None),
            };
            return Ok(vec![Event::Failed(self.provider_stream_failure(
                "openai_compatible",
                code.as_deref(),
                raw_message,
                message.as_deref(),
            ))]);
        }
        let mut events = Vec::new();
        // An aggregator names the upstream that serves the stream on each
        // chunk (OpenRouter `provider`, opted in by its metadata header); the
        // first label is kept for settlement so a zero-data-retention
        // dispatch records which retention-free endpoint answered.
        if let Some(Value::String(provider)) = payload.get("provider") {
            self.note_upstream_provider(provider);
        }
        if let Some(raw_usage) = payload.get("usage") {
            if !raw_usage.is_null() {
                self.usage = Some(
                    self.openai_usage
                        .update_chat(raw_usage)
                        .map_err(|message| malformed(&message))?,
                );
            }
        }
        // A frame without `choices` whose keys are all chunk metadata is a
        // relay's trailing usage-only chunk (Novita, 19 attempts in two days,
        // 2026-09-14..15): its usage was taken above and nothing else is
        // decoded from it. A choices-less frame carrying anything else, and
        // an explicitly non-array `choices`, stay malformed, and the reason
        // names the frame's KEY NAMES (never values) so the next unknown
        // shape a provider sends is diagnosable from the ledger.
        let choices = match payload.get("choices") {
            None | Some(Value::Null) if is_metadata_only_frame(&payload) => return Ok(events),
            Some(Value::Array(choices)) => choices,
            _ => {
                return Err(malformed(&format!(
                    "OpenAI-compatible choices must be an array (frame keys: {})",
                    frame_key_names(&payload)
                )))
            }
        };
        if choices.is_empty() {
            return Ok(events);
        }
        if choices.len() != 1 {
            return Err(malformed(
                "OpenAI-compatible stream must contain one choice",
            ));
        }
        let choice = choices[0]
            .as_object()
            .ok_or_else(|| malformed("OpenAI-compatible choice must be an object"))?;
        let choice_index = choice.get("index").and_then(Value::as_u64).unwrap_or(0);
        if self.chat_logprobs
            && (choice_index != 0 || choice.get("index").is_some_and(|value| !value.is_u64()))
        {
            return Err(malformed("OpenAI-compatible stream choice index must be 0"));
        }
        if self.chat_logprobs {
            if let Some(event) =
                crate::logprobs::parse(choice.get("logprobs"), choice_index as u32)?
            {
                events.push(Event::ChoiceLogprobsDelta(event));
            }
        }
        // Azure asynchronous content-filter annotations carry no delta. Treat
        // their metadata-only choice as an empty delta, then still process the
        // finish reason below: an annotation can terminate with content_filter.
        // An explicitly invalid delta or an unrecognized missing-delta frame
        // remains malformed.
        let is_filter_annotation = ["content_filter_results", "content_filter_offsets"]
            .iter()
            .all(|field| choice.get(*field).is_some_and(Value::is_object));
        let empty_delta = serde_json::Map::new();
        let delta = match choice.get("delta") {
            Some(Value::Object(delta)) => delta,
            None if is_filter_annotation => &empty_delta,
            _ => return Err(malformed("OpenAI-compatible delta must be an object")),
        };
        if let Some(Value::String(content)) = delta.get("content") {
            if !content.is_empty() {
                events.push(Event::TextDelta(content.clone()));
            }
        }
        if let Some(images) = delta.get("images").filter(|value| !value.is_null()) {
            let images = images.as_array().ok_or_else(|| {
                malformed("Chat images must be an array").with_retry(false, false)
            })?;
            for image in images {
                let url = crate::image_output::chat_image(image)?;
                self.reserve_image_bytes(url.len())?;
                events.push(Event::Image(url));
            }
        }
        if let Some(Value::String(refusal)) = delta.get("refusal") {
            self.refusal_seen = true;
            events.push(Event::RefusalDelta(refusal.clone()));
        }
        if let Some(route_sha256) = self.reasoning_content_route_sha256.clone() {
            if let Some(value) = delta
                .get("reasoning_content")
                .filter(|value| !value.is_null())
                .or_else(|| delta.get("reasoning"))
            {
                let reasoning = match value {
                    Value::Null => None,
                    Value::String(text) => Some(text),
                    _ => return Err(malformed("Fireworks reasoning_content delta must be text")),
                };
                if let Some(reasoning) = reasoning.filter(|text| !text.is_empty()) {
                    self.reserve_summary_bytes(reasoning.len())?;
                    events.push(Event::ReasoningContentDelta {
                        route_sha256,
                        delta: reasoning.clone(),
                    });
                }
            }
        }
        if let Some(raw_tools) = delta.get("tool_calls") {
            if !raw_tools.is_null() {
                let items = raw_tools
                    .as_array()
                    .ok_or_else(|| malformed("OpenAI-compatible tool_calls must be an array"))?;
                for value in items {
                    let item = value.as_object().ok_or_else(|| {
                        malformed("OpenAI-compatible tool call must be an object")
                    })?;
                    let index = require_u64(item, "index", "OpenAI-compatible tool index")
                        .map_err(|message| malformed(&message))?
                        as u32;
                    let function =
                        item.get("function")
                            .and_then(Value::as_object)
                            .ok_or_else(|| {
                                malformed("OpenAI-compatible tool function must be an object")
                            })?;
                    let restated_id = wire_text(item.get("id"), "OpenAI-compatible tool ID")?;
                    let restated_name =
                        wire_text(function.get("name"), "OpenAI-compatible tool name")?;
                    if let Some(tool) = self.tools.get_mut(&index) {
                        // An identity is only restated when it is non-empty:
                        // DashScope (Qwen) argument deltas carry `"id": ""`
                        // (documented shape, live 2026-09-03), and an empty
                        // placeholder names nothing, so only a different
                        // NON-EMPTY id or name is a stream that changed
                        // identity. A gateway-minted id yields to nothing:
                        // the caller already holds it.
                        if let Some(repeated_id) = restated_id.filter(|id| !id.is_empty()) {
                            if repeated_id != tool.call_id && !tool.id_synthesized {
                                return Err(malformed(
                                    "OpenAI-compatible stream changed a tool-call ID",
                                ));
                            }
                        }
                        if let Some(repeated_name) = restated_name.filter(|name| !name.is_empty()) {
                            if !tool.started {
                                // The relay opened the entry nameless and names
                                // it now: the call starts here, with whatever
                                // arguments accumulated silently meanwhile.
                                tool.name = repeated_name;
                                tool.started = true;
                                events.push(Event::ToolCallStarted {
                                    custom: false,
                                    index,
                                    call_id: tool.call_id.clone(),
                                    name: tool.name.clone(),
                                    namespace: None,
                                    caller: None,
                                });
                                if !tool.raw_arguments.is_empty() {
                                    events.push(Event::ToolArgumentsDelta {
                                        index,
                                        delta: tool.raw_arguments.clone(),
                                    });
                                }
                            } else if repeated_name != tool.name {
                                return Err(malformed(
                                    "OpenAI-compatible stream changed a tool-call name",
                                ));
                            }
                        }
                    } else {
                        // A null or empty id is a relay that never minted one
                        // (Z.ai GLM via Fireworks/OpenRouter, live 2026-09-05..15):
                        // the gateway mints a stable id so the call is
                        // callable, since the caller pairs its tool result
                        // by this id and nothing else. A null or empty NAME
                        // opens the entry unstarted (no start event yet): a
                        // later frame may name it, and one that never does
                        // and never argues is dropped as a phantom at finish.
                        let (call_id, id_synthesized) = match restated_id {
                            Some(id) if !id.is_empty() => (id, false),
                            _ => (synthesized_call_id(index), true),
                        };
                        let name = restated_name.unwrap_or_default();
                        self.reserve_tool_entry(index)?;
                        let mut tool = ToolAccumulator::new(call_id.clone(), name.clone());
                        tool.id_synthesized = id_synthesized;
                        tool.started = !name.is_empty();
                        if tool.started {
                            events.push(Event::ToolCallStarted {
                                custom: false,
                                index,
                                call_id,
                                name,
                                namespace: None,
                                caller: None,
                            });
                        }
                        self.tools.insert(index, tool);
                    }
                    if let Some(fragment) = function.get("arguments") {
                        if !fragment.is_null() {
                            let raw_fragment = match fragment {
                                Value::String(text) => text.clone(),
                                _ => {
                                    return Err(malformed(
                                        "OpenAI-compatible argument delta must be text",
                                    ))
                                }
                            };
                            self.reserve_tool_bytes(raw_fragment.len())?;
                            let tool = self.tools.get_mut(&index).expect("tool just ensured");
                            // Bytes after the argument object closes are
                            // withheld, never relayed: Azure Foundry's
                            // DeepSeek shim streams `{}` then `""` for a
                            // zero-argument call (live 2026-09-10), and a
                            // client that concatenates deltas must end up
                            // with exactly the completed call's bytes.
                            if let Some(delta) = tool.push_arguments(&raw_fragment) {
                                if tool.started {
                                    events.push(Event::ToolArgumentsDelta { index, delta });
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some(Value::String(finish)) = choice.get("finish_reason") {
            self.finish_reason = Some(finish.clone());
            if matches!(finish.as_str(), "content_filter" | "safety") && !self.refusal_seen {
                self.refusal_seen = true;
                events.push(Event::RefusalDelta(String::new()));
            }
        }
        Ok(events)
    }

    /// End an OpenAI-compatible stream on the finish reason it carried.
    ///
    /// Reached from the `[DONE]` sentinel, or from a clean EOF that arrived
    /// after a `finish_reason` chunk (`openai_compatible_stream_end`): the
    /// finish reason is the provider's declared end of the choice, and both
    /// paths settle it identically.
    fn finish_openai_compatible_stream(&mut self) -> Result<Vec<Event>, Failure> {
        let finish = self.finish_reason.as_deref();
        // A tool call cut off by the output budget (finish_reason=length,
        // arguments still an open JSON fragment) is the provider's honest
        // truncation, not a malformed stream: it surfaces as Incomplete
        // with the truncated call dropped, exactly what the caller must
        // act on (raise max_tokens), never as a 502. Live shape: Tencent
        // TokenHub glm-5.3 at max_tokens=32 streamed `{"` + `city` then
        // finished with length (staging, 2026-09-03). Under any other
        // finish a relay may still close a call mid-fragment
        // (finish_open_tools_relay): that cut call is dropped and the turn
        // settles Incomplete too; a syntax error inside the arguments
        // keeps the strict contract and stays malformed.
        let (mut events, cut_mid_fragment) = if finish == Some("length") {
            (finish_open_tools_truncated(&mut self.tools)?, false)
        } else {
            finish_open_tools_relay(&mut self.tools, finish.unwrap_or("none"))?
        };
        if let Some(usage) = self.usage.take() {
            events.push(Event::Usage(usage));
        }
        if self.refusal_seen || matches!(finish, Some("content_filter" | "safety")) {
            // A `content_filter`/`safety` finish reason names the category;
            // a bare visible-refusal delta names none (Unspecified).
            let reason = match finish {
                Some(code @ ("content_filter" | "safety")) => {
                    crate::stream_errors::refusal_reason(Some(code), None)
                }
                _ => crate::errors::RefusalReason::Unspecified,
            };
            events.push(Event::Failed(Failure::refusal(reason)));
        } else if finish == Some("length") || cut_mid_fragment {
            events.push(Event::Incomplete);
        } else {
            events.push(Event::Completed);
        }
        Ok(events)
    }

    /// Terminal events for an OpenAI-compatible stream that closed cleanly
    /// without the `[DONE]` sentinel but after a `finish_reason` chunk.
    ///
    /// The OpenAI contract ends a choice with its non-null `finish_reason`;
    /// `[DONE]` is the sentinel OpenAI's own server appends afterwards, and
    /// not every compatible server does. Azure AI Foundry's DeepSeek
    /// deployments (DeepSeek-V4-Flash, live 2026-09-15) end a content-filtered
    /// stream with the `finish_reason: "content_filter"` chunk and close the
    /// connection at once, so ~900 refusals a day on that lane were filed as
    /// "provider stream ended without a terminal event" 502s instead of the
    /// refusal the provider declared. A finish reason already seen is a
    /// complete ending: settle it exactly as `[DONE]` would (refusal,
    /// Incomplete, or Completed). Without one, nothing is synthesized and the
    /// stream stays terminal-less for `stream_ended` to fail closed.
    pub(in crate::dialects) fn openai_compatible_stream_end(
        &mut self,
    ) -> Result<Vec<Event>, Failure> {
        if self.finish_reason.is_none() {
            return Ok(Vec::new());
        }
        self.finish_openai_compatible_stream()
    }
}

#[cfg(test)]
#[path = "compatible_tests.rs"]
mod tests;
