//! Output-chain enforcement for identity-scoped guardrails.
//!
//! Rust owns buffering and delivery. A chain whose checks all bind
//! deterministic detectors runs natively through `plan`, with the compiled
//! rules in `detector`. Every other chain crosses the JSON-typed python
//! boundary below, which keeps policy lookup and non-deterministic adapters
//! (`http_json` and future model-based classifiers) in python.
//!
//! This module never logs request text, completions, or replacements.

pub mod detector;
pub mod plan;
mod syntax;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::bridge::Bridge;
use crate::errors::{Failure, FailureClass};
use crate::events::Event;

/// Decision returned by one Python `enforce_output` callback.
#[derive(Debug, Deserialize)]
struct OutputDecision {
    action: String,
    #[serde(default)]
    replacement_text: Option<String>,
    #[serde(default)]
    failure: Option<Failure>,
}

/// Build one fail-closed guardrail failure without request content.
fn closed_failure() -> Failure {
    Failure::new(
        FailureClass::Guardrail,
        "A gateway guardrail could not complete this request.",
    )
}

/// Prefer the classifier-supplied failure when it is already sanitized.
fn decision_failure(decision: &OutputDecision) -> Failure {
    decision.failure.clone().unwrap_or_else(|| {
        let message = if decision.action == "block" {
            "The request was blocked by a gateway guardrail."
        } else {
            "A gateway guardrail could not complete this request."
        };
        Failure::new(FailureClass::Guardrail, message)
    })
}

/// Project collected events into the JSON payload Python inspects once.
pub fn output_argument(request_id: &str, events: &[Event]) -> String {
    let mut text = String::new();
    let mut refusal = false;
    let mut tool_calls: Vec<Value> = Vec::new();
    for event in events {
        match event {
            Event::TextDelta(delta) | Event::ProviderTextDelta { delta, .. } => {
                text.push_str(delta);
            }
            Event::RefusalDelta(_) | Event::ProviderRefusalDelta { .. } => refusal = true,
            Event::ToolCallCompleted { call, .. } => {
                tool_calls.push(json!({
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": call.raw_arguments,
                }));
            }
            _ => {}
        }
    }
    crate::encode::compact_json(&json!({
        "request_id": request_id,
        "text": text,
        "refusal": refusal,
        "tool_calls": tool_calls,
    }))
}

/// Replace text deltas with one rewritten delta. Refusal deltas are dropped,
/// and so is every provider-reasoning event: rewritten output must not leak
/// the redacted content through the model's own reasoning channel.
pub fn apply_text_replacement(events: &[Event], replacement: &str) -> Vec<Event> {
    let mut rewritten = Vec::with_capacity(events.len());
    let mut inserted = false;
    for event in events {
        match event {
            Event::RefusalDelta(_)
            | Event::ProviderRefusalDelta { .. }
            | Event::ProviderOutputItemStarted { .. }
            | Event::ProviderOutputItemCompleted { .. }
            | Event::ReasoningSummaryDelta { .. }
            | Event::ThinkingDelta { .. }
            | Event::ThinkingSignature { .. }
            | Event::RedactedThinking { .. }
            | Event::EncryptedReasoning { .. } => {}
            // Server-tool activity and citations carry the fetched content
            // (queries, result payloads, cited text) that a rewrite must not
            // leak, so they drop with the reasoning channel. Hosted Responses
            // tool items and their annotations are the same class of content.
            Event::TextBlockStarted { .. }
            | Event::CitationDelta { .. }
            | Event::ServerToolUseStarted { .. }
            | Event::ServerToolArgumentsDelta { .. }
            | Event::ServerToolUseCompleted { .. }
            | Event::ServerToolResult { .. }
            | Event::HostedToolItemStarted { .. }
            | Event::HostedToolItemProgress { .. }
            | Event::HostedToolItemCompleted { .. }
            | Event::ProviderTextAnnotation { .. } => {}
            Event::TextDelta(_) | Event::ProviderTextDelta { .. } => {
                if inserted {
                    continue;
                }
                rewritten.push(Event::TextDelta(replacement.to_string()));
                inserted = true;
            }
            Event::Completed
            | Event::Incomplete
            | Event::StoppedAtSequence(_)
            | Event::PausedTurn
            | Event::Failed(_)
                if !inserted =>
            {
                rewritten.push(Event::TextDelta(replacement.to_string()));
                inserted = true;
                rewritten.push(event.clone());
            }
            other => rewritten.push(other.clone()),
        }
    }
    rewritten
}

/// Decision returned by one Python `enforce_output_segment` callback.
#[derive(Debug, Deserialize)]
struct SegmentDecision {
    action: String,
    #[serde(default)]
    release: String,
    #[serde(default)]
    pending: String,
    #[serde(default)]
    failure: Option<Failure>,
}

/// One caller-visible text channel a deterministic redactor can rewrite.
///
/// Channels are independent subjects: a match cannot span two of them, and
/// each provider-owned assistant item is its own channel. Refusal text is a
/// channel too, so a refusal streams redacted instead of being suppressed.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TextChannel {
    Text,
    Refusal,
    ProviderText { output_index: u32, item_id: String },
    ProviderRefusal { output_index: u32, item_id: String },
}

impl TextChannel {
    /// Rebuild this channel's event carrying `delta`.
    fn event(&self, delta: String) -> Event {
        match self {
            Self::Text => Event::TextDelta(delta),
            Self::Refusal => Event::RefusalDelta(delta),
            Self::ProviderText {
                output_index,
                item_id,
            } => Event::ProviderTextDelta {
                output_index: *output_index,
                item_id: item_id.clone(),
                delta,
            },
            Self::ProviderRefusal {
                output_index,
                item_id,
            } => Event::ProviderRefusalDelta {
                output_index: *output_index,
                item_id: item_id.clone(),
                delta,
            },
        }
    }
}

/// Classify one event for the incremental path.
enum StreamAdmission {
    /// Text that the redactor buffers and releases.
    Redactable(TextChannel, String),
    /// Content the caller sees that a deterministic text redactor cannot
    /// rewrite, so the request fails closed rather than leak it.
    Unredactable,
    /// Structure that opens or closes a content item. The buffered tail is
    /// released before it, so an item never closes ahead of its own text.
    Boundary,
    /// Metering or opaque carriers with no caller-readable model content and
    /// no ordering relationship to the open text channel.
    Passthrough,
}

/// Decide how the incremental path may treat one event.
///
/// The buffered path answers this question by dropping every channel a text
/// rewrite could leak through (reasoning, refusals, server tools, citations)
/// after it has seen the whole completion. Incremental release has no such
/// second chance: a byte sent is a byte the caller keeps. The equivalent
/// guarantee is therefore made ahead of the bytes, in two layers. Admission
/// keeps a request that offers tools, asks for thinking, or asks for a
/// reasoning summary on the buffered path, and the route keeps a rung that
/// exposes plaintext reasoning there too. Anything that still reaches this
/// function and carries caller-readable model content that is not plain
/// assistant text fails the stream closed. Only opaque carriers (signatures,
/// encrypted or redacted reasoning blobs) and pure structure pass through,
/// because neither can echo the redacted characters.
///
/// A content-item boundary ends the subject it closes: the text of one
/// provider item or content block is redacted on its own, never joined to
/// the next item's text the way the buffered path concatenates a whole
/// completion.
fn classify(event: &Event) -> StreamAdmission {
    match event {
        Event::TextDelta(delta) => StreamAdmission::Redactable(TextChannel::Text, delta.clone()),
        Event::RefusalDelta(delta) => {
            StreamAdmission::Redactable(TextChannel::Refusal, delta.clone())
        }
        Event::ProviderTextDelta {
            output_index,
            item_id,
            delta,
        } => StreamAdmission::Redactable(
            TextChannel::ProviderText {
                output_index: *output_index,
                item_id: item_id.clone(),
            },
            delta.clone(),
        ),
        Event::ProviderRefusalDelta {
            output_index,
            item_id,
            delta,
        } => StreamAdmission::Redactable(
            TextChannel::ProviderRefusal {
                output_index: *output_index,
                item_id: item_id.clone(),
            },
            delta.clone(),
        ),
        Event::ReasoningSummaryDelta { .. }
        | Event::ThinkingDelta { .. }
        | Event::ReasoningContentDelta { .. }
        | Event::ToolCallStarted { .. }
        | Event::ToolArgumentsDelta { .. }
        | Event::ToolCallCompleted { .. }
        | Event::ServerToolUseStarted { .. }
        | Event::ServerToolArgumentsDelta { .. }
        | Event::ServerToolUseCompleted { .. }
        | Event::ServerToolResult { .. }
        | Event::HostedToolItemStarted { .. }
        | Event::HostedToolItemProgress { .. }
        | Event::HostedToolItemCompleted { .. }
        | Event::CitationDelta { .. }
        | Event::ProviderTextAnnotation { .. } => StreamAdmission::Unredactable,
        Event::ProviderOutputItemStarted { .. }
        | Event::ProviderOutputItemCompleted { .. }
        | Event::TextBlockStarted { .. } => StreamAdmission::Boundary,
        // Keep this match exhaustive: a new event variant must receive an
        // explicit safety classification before guarded streaming compiles.
        Event::ThinkingSignature { .. }
        | Event::RedactedThinking { .. }
        | Event::EncryptedReasoning { .. }
        | Event::Usage(_)
        | Event::Completed
        | Event::Incomplete
        | Event::StoppedAtSequence(_)
        | Event::PausedTurn
        | Event::Failed(_) => StreamAdmission::Passthrough,
    }
}

/// Return what a guarded or unguarded stream may send for one outward event.
///
/// A terminal flushes everything still buffered before it, so no character
/// outlives the stream that carried it. Without a redactor the event passes
/// through, which keeps the streaming routes free of guardrail branching.
pub(crate) async fn released_events(
    redactor: Option<&mut StreamRedactor>,
    bridge: &Bridge,
    outward: Event,
    terminal: bool,
) -> Result<Vec<Event>, Failure> {
    let Some(redactor) = redactor else {
        return Ok(vec![outward]);
    };
    let mut released = if terminal {
        redactor.flush(bridge).await?
    } else {
        Vec::new()
    };
    released.extend(redactor.admit(bridge, outward).await?);
    Ok(released)
}

/// Release a streamed completion incrementally under a deterministic check.
///
/// The data plane owns the buffer and the ordering; Python owns the release
/// boundary and the redaction. Only the trailing window the detector cannot
/// yet decide about is withheld, so the caller's first byte no longer waits
/// for the provider's last one.
pub struct StreamRedactor {
    request_id: String,
    channel: Option<TextChannel>,
    pending: String,
    settled_bytes: u64,
}

impl StreamRedactor {
    /// Open one redactor for the given request.
    pub fn new(request_id: &str) -> Self {
        Self {
            request_id: request_id.to_string(),
            channel: None,
            pending: String::new(),
            settled_bytes: 0,
        }
    }

    /// Feed one outward event and return what the caller may see now.
    ///
    /// Text joins the buffered tail of its channel and comes back redacted
    /// up to the settled boundary. Switching channels flushes the previous
    /// one first, because the new channel's text cannot extend a match in
    /// it. Every other event either passes through untouched or fails the
    /// stream closed.
    pub async fn admit(&mut self, bridge: &Bridge, event: Event) -> Result<Vec<Event>, Failure> {
        match classify(&event) {
            StreamAdmission::Unredactable => Err(closed_failure()),
            StreamAdmission::Passthrough => Ok(vec![event]),
            StreamAdmission::Boundary => {
                let mut released = self.flush(bridge).await?;
                released.push(event);
                Ok(released)
            }
            StreamAdmission::Redactable(channel, delta) => {
                let mut released = Vec::new();
                if self.channel.as_ref().is_some_and(|open| *open != channel) {
                    released.extend(self.flush(bridge).await?);
                }
                self.channel = Some(channel.clone());
                self.pending.push_str(&delta);
                if let Some(text) = self.release(bridge, false).await? {
                    released.push(channel.event(text));
                }
                Ok(released)
            }
        }
    }

    /// Release everything still buffered, because no later text can extend
    /// a match into it.
    pub async fn flush(&mut self, bridge: &Bridge) -> Result<Vec<Event>, Failure> {
        let Some(channel) = self.channel.clone() else {
            return Ok(Vec::new());
        };
        let released = self.release(bridge, true).await?;
        self.channel = None;
        Ok(released
            .map(|text| vec![channel.event(text)])
            .unwrap_or_default())
    }

    /// Ask Python for the settled part of the buffered tail.
    async fn release(
        &mut self,
        bridge: &Bridge,
        final_segment: bool,
    ) -> Result<Option<String>, Failure> {
        if self.pending.is_empty() && !final_segment {
            return Ok(None);
        }
        let argument = crate::encode::compact_json(&json!({
            "request_id": self.request_id,
            "pending": self.pending,
            "final": final_segment,
            "settled_bytes": self.settled_bytes,
        }));
        let sent = self.pending.len();
        let payload = bridge
            .call("enforce_output_segment", argument)
            .await
            .map_err(|_| closed_failure())?;
        let decision: SegmentDecision =
            serde_json::from_str(&payload).map_err(|_| closed_failure())?;
        if decision.action != "allow" {
            return Err(decision.failure.clone().unwrap_or_else(closed_failure));
        }
        self.pending = decision.pending;
        // The completion bound counts the bytes the provider produced, so a
        // replacement shorter than its match must not shrink the running
        // total. What left the buffer is what the tail lost, not what the
        // caller received.
        self.settled_bytes += sent.saturating_sub(self.pending.len()) as u64;
        if decision.release.is_empty() {
            return Ok(None);
        }
        Ok(Some(decision.release))
    }
}

/// Invoke the Python output chain once and return the validated events.
pub async fn enforce_collected_output(
    bridge: &Bridge,
    request_id: &str,
    events: Vec<Event>,
) -> Result<Vec<Event>, Failure> {
    let argument = output_argument(request_id, &events);
    let payload = bridge
        .call("enforce_output", argument)
        .await
        .map_err(|_| closed_failure())?;
    let decision: OutputDecision = serde_json::from_str(&payload).map_err(|_| closed_failure())?;
    match decision.action.as_str() {
        "allow" => Ok(events),
        "modify" => {
            if events
                .iter()
                .any(|event| matches!(event, Event::ToolCallCompleted { .. }))
            {
                return Err(Failure::new(
                    FailureClass::Guardrail,
                    "The request was blocked by a gateway guardrail.",
                ));
            }
            let replacement = decision
                .replacement_text
                .as_deref()
                .ok_or_else(closed_failure)?;
            Ok(apply_text_replacement(&events, replacement))
        }
        "block" | "error" => Err(decision_failure(&decision)),
        _ => Err(closed_failure()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::CompletedToolCall;
    use pyo3::types::PyAnyMethods;

    #[test]
    fn output_argument_is_content_shaped_and_request_keyed() {
        let events = vec![
            Event::TextDelta("hello".to_string()),
            Event::ToolCallCompleted {
                index: 0,
                call: CompletedToolCall {
                    namespace: None,
                    caller: None,
                    call_id: "call-1".to_string(),
                    name: "lookup".to_string(),
                    provider_item_id: None,
                    provider_status: None,
                    raw_arguments: "{\"q\":\"x\"}".to_string(),
                    custom: false,
                },
            },
            Event::Completed,
        ];
        let payload: Value = serde_json::from_str(&output_argument("req-1", &events)).unwrap();
        assert_eq!(payload["request_id"], "req-1");
        assert_eq!(payload["text"], "hello");
        assert_eq!(payload["refusal"], false);
        assert_eq!(payload["tool_calls"][0]["name"], "lookup");
        assert_eq!(payload["tool_calls"][0]["arguments"], "{\"q\":\"x\"}");
    }

    #[test]
    fn text_replacement_drops_every_reasoning_channel() {
        // A rewritten output must not leak the redacted content through the
        // model's own reasoning stream.
        let events = vec![
            Event::ThinkingDelta {
                index: 0,
                delta: "secret plan".to_string(),
            },
            Event::ThinkingSignature {
                index: 0,
                signature: "sig==".to_string(),
            },
            Event::RedactedThinking {
                index: 1,
                data: "opaque==".to_string(),
            },
            Event::EncryptedReasoning {
                output_index: 0,
                item_id: "rs-1".to_string(),
                encrypted_content: "blob==".to_string(),
            },
            Event::TextDelta("disallowed".to_string()),
            Event::Completed,
        ];
        let rewritten = apply_text_replacement(&events, "[redacted]");
        assert!(matches!(
            rewritten.as_slice(),
            [Event::TextDelta(text), Event::Completed] if text == "[redacted]"
        ));
    }

    #[test]
    fn text_replacement_collapses_deltas_and_leaves_tool_calls() {
        let events = vec![
            Event::TextDelta("hel".to_string()),
            Event::TextDelta("lo".to_string()),
            Event::ToolCallCompleted {
                index: 0,
                call: CompletedToolCall {
                    namespace: None,
                    caller: None,
                    call_id: "call-1".to_string(),
                    name: "lookup".to_string(),
                    provider_item_id: None,
                    provider_status: None,
                    raw_arguments: "{}".to_string(),
                    custom: false,
                },
            },
            Event::Completed,
        ];
        let rewritten = apply_text_replacement(&events, "safe");
        let texts: Vec<&str> = rewritten
            .iter()
            .filter_map(|event| match event {
                Event::TextDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, ["safe"]);
        assert!(rewritten
            .iter()
            .any(|event| matches!(event, Event::ToolCallCompleted { .. })));
    }

    #[test]
    fn missing_text_inserts_replacement_before_terminal() {
        let events = vec![Event::Completed];
        let rewritten = apply_text_replacement(&events, "safe");
        assert!(matches!(rewritten[0], Event::TextDelta(ref text) if text == "safe"));
        assert!(matches!(rewritten[1], Event::Completed));
    }

    #[test]
    fn refusal_only_replacement_drops_refusal_deltas() {
        let events = vec![
            Event::RefusalDelta("I cannot".to_string()),
            Event::Completed,
        ];
        let rewritten = apply_text_replacement(&events, "safe");
        assert!(matches!(rewritten[0], Event::TextDelta(ref text) if text == "safe"));
        assert!(matches!(rewritten[1], Event::Completed));
        assert!(!rewritten
            .iter()
            .any(|event| matches!(event, Event::RefusalDelta(_))));
    }

    /// A deterministic control plane: it holds the last four characters and
    /// replaces every occurrence of `secret`, so a match split across two
    /// deltas is only redacted when the tail is buffered correctly.
    const SEGMENT_PLANE: &std::ffi::CStr = cr#"
import json


class Plane:
    """Answer one streaming segment callback deterministically."""

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def enforce_output_segment(self, argument):
        data = json.loads(argument)
        self.calls.append(data)
        if self.fail:
            return json.dumps(
                {
                    "action": "error",
                    "failure": {
                        "failure_class": "guardrail",
                        "safe_message": "A gateway guardrail could not complete this request.",
                    },
                }
            )
        pending = data["pending"]
        boundary = len(pending) if data["final"] else max(0, len(pending) - 4)
        release = pending[:boundary].replace("secret", "[R]")
        return json.dumps(
            {
                "action": "allow",
                "release": release,
                "pending": pending[boundary:],
                "flagged": "[R]" in release,
            }
        )
"#;

    /// Start one bridge over the scripted segment plane.
    fn segment_bridge(fail: bool) -> Bridge {
        pyo3::Python::initialize();
        let object = pyo3::Python::attach(|py| {
            pyo3::types::PyModule::from_code(py, SEGMENT_PLANE, c"segment.py", c"segment")
                .expect("module compiles")
                .getattr("Plane")
                .expect("class exists")
                .call1((fail,))
                .expect("plane instantiates")
                .unbind()
        });
        Bridge::new(object, 1).expect("bridge starts")
    }

    /// Run one future to completion on a fresh runtime.
    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime builds")
            .block_on(future)
    }

    /// Collect the text a caller would have seen from released events.
    fn seen(events: &[Event]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                Event::TextDelta(text) | Event::RefusalDelta(text) => Some(text.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn text_releases_incrementally_and_redacts_across_deltas() {
        let bridge = segment_bridge(false);
        let mut redactor = StreamRedactor::new("req-1");
        let released = block_on(async {
            let mut all = Vec::new();
            for delta in ["hello sec", "ret world", " and more text"] {
                all.extend(
                    redactor
                        .admit(&bridge, Event::TextDelta(delta.to_string()))
                        .await
                        .expect("segment allowed"),
                );
            }
            // Bytes reached the caller before the stream ended.
            assert!(!all.is_empty());
            all.extend(redactor.flush(&bridge).await.expect("flush allowed"));
            all
        });
        let text = seen(&released);
        assert_eq!(text, "hello [R] world and more text");
        assert!(!text.contains("secret"));
    }

    #[test]
    fn settled_bytes_count_consumed_input_not_released_output() {
        let (plane, bridge) = pyo3::Python::attach(|py| {
            let object = pyo3::types::PyModule::from_code(
                py,
                SEGMENT_PLANE,
                c"segment.py",
                c"segment_bytes",
            )
            .expect("module compiles")
            .getattr("Plane")
            .expect("class exists")
            .call1((false,))
            .expect("plane instantiates")
            .unbind();
            let bridge = Bridge::new(object.clone_ref(py), 1).expect("bridge starts");
            (object, bridge)
        });
        let mut redactor = StreamRedactor::new("req-1");
        let released = block_on(async {
            let mut all = Vec::new();
            for delta in ["hello sec", "ret world", " and more text"] {
                all.extend(
                    redactor
                        .admit(&bridge, Event::TextDelta(delta.to_string()))
                        .await
                        .expect("segment allowed"),
                );
            }
            all.extend(redactor.flush(&bridge).await.expect("flush allowed"));
            all
        });
        // The replacement is shorter than its match, so the caller sees fewer
        // bytes than the provider produced. The bound follows the provider.
        assert_eq!(seen(&released).len(), 29);
        let settled = pyo3::Python::attach(|py| {
            use pyo3::prelude::PyAnyMethods;
            let calls = plane.bind(py).getattr("calls").expect("calls recorded");
            let last = calls
                .get_item(calls.len().expect("length") - 1)
                .expect("one call");
            last.get_item("settled_bytes")
                .expect("field present")
                .extract::<u64>()
                .expect("integer")
        });
        assert_eq!(settled, 28);
    }

    #[test]
    fn a_channel_switch_flushes_the_previous_channel() {
        let bridge = segment_bridge(false);
        let mut redactor = StreamRedactor::new("req-1");
        let released = block_on(async {
            let mut all = Vec::new();
            all.extend(
                redactor
                    .admit(&bridge, Event::TextDelta("abcdefgh".to_string()))
                    .await
                    .expect("segment allowed"),
            );
            all.extend(
                redactor
                    .admit(&bridge, Event::RefusalDelta("I cannot".to_string()))
                    .await
                    .expect("segment allowed"),
            );
            all.extend(redactor.flush(&bridge).await.expect("flush allowed"));
            all
        });
        let text: String = released
            .iter()
            .filter_map(|event| match event {
                Event::TextDelta(value) => Some(value.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "abcdefgh");
        assert_eq!(seen(&released), "abcdefghI cannot");
    }

    #[test]
    fn a_content_boundary_flushes_before_it_is_forwarded() {
        let bridge = segment_bridge(false);
        let mut redactor = StreamRedactor::new("req-1");
        let released = block_on(async {
            let mut all = Vec::new();
            all.extend(
                redactor
                    .admit(&bridge, Event::TextDelta("abcdefgh".to_string()))
                    .await
                    .expect("segment allowed"),
            );
            all.extend(
                redactor
                    .admit(&bridge, Event::TextBlockStarted { index: 1 })
                    .await
                    .expect("boundary allowed"),
            );
            all
        });
        assert_eq!(seen(&released), "abcdefgh");
        assert!(matches!(
            released.last(),
            Some(Event::TextBlockStarted { .. })
        ));
    }

    #[test]
    fn reasoning_and_tool_channels_fail_the_stream_closed() {
        let bridge = segment_bridge(false);
        for event in [
            Event::ThinkingDelta {
                index: 0,
                delta: "secret plan".to_string(),
            },
            Event::ReasoningSummaryDelta {
                output_index: 0,
                item_id: "rs-1".to_string(),
                summary_index: 0,
                delta: "summary".to_string(),
            },
            Event::CitationDelta {
                index: 0,
                citation: "{\"url\":\"https://example.test\"}".to_string(),
            },
        ] {
            let mut redactor = StreamRedactor::new("req-1");
            let outcome = block_on(redactor.admit(&bridge, event));
            let failure = outcome.expect_err("unredactable content fails closed");
            assert_eq!(failure.failure_class, FailureClass::Guardrail);
        }
    }

    #[test]
    fn a_failed_segment_releases_nothing() {
        let bridge = segment_bridge(true);
        let mut redactor = StreamRedactor::new("req-1");
        let outcome = block_on(redactor.admit(&bridge, Event::TextDelta("secret".to_string())));
        let failure = outcome.expect_err("a failed segment is terminal");
        assert_eq!(failure.failure_class, FailureClass::Guardrail);
    }

    #[test]
    fn metering_and_opaque_carriers_pass_through() {
        let bridge = segment_bridge(false);
        let mut redactor = StreamRedactor::new("req-1");
        let released = block_on(redactor.admit(
            &bridge,
            Event::EncryptedReasoning {
                output_index: 0,
                item_id: "rs-1".to_string(),
                encrypted_content: "blob==".to_string(),
            },
        ))
        .expect("opaque carriers pass");
        assert!(matches!(
            released.as_slice(),
            [Event::EncryptedReasoning { .. }]
        ));
    }
}
