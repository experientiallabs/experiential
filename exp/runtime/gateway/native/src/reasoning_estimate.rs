//! Estimated reasoning tokens for wires that bill thinking without counting it.
//!
//! Anthropic Messages and Bedrock Converse bill a turn's thinking inside
//! `output_tokens` and publish no separate count, so `Usage::reasoning_tokens`
//! stays `None` (see the usage contract in `events.rs`). Callers that log
//! `completion_tokens_details.reasoning_tokens` (Chat) or
//! `output_tokens_details.reasoning_tokens` (Responses) then read 0 on a turn
//! that thought. OpenRouter reports an estimate on the same lanes: the output
//! total less the visible answer (probed 2026-10-18 on Claude Haiku 4.5 and
//! Sonnet 4.5: 259 output / 240 reasoning and 238 / 215 for a one-line answer).
//!
//! This module reproduces that on the encoded wire only. The estimate is
//! `output_tokens - ceil(visible_bytes / 4)`, floored at zero, where visible
//! bytes are the answer text, refusal text and client tool-call arguments.
//! Bytes over four is the same approximation the data plane already uses for
//! input tokens; the gateway holds no tokenizer for any rung. It applies only
//! when the provider published no count and the stream carried thinking in
//! any form (readable, summarized, plaintext, signed, redacted or encrypted). It never touches `Usage`, so settlement,
//! the ledger and every rate stay exactly as reported: the wire marks the
//! field `estimated_token_details: ["reasoning_tokens"]`, and the upstream
//! usage parser reads a marked count as unreported, so a gateway behind a
//! gateway can never bill an estimate as a reasoning leg.

use serde_json::{json, Value};

use crate::events::{Event, Usage};

/// The usage extension naming the token details that are estimates.
pub const ESTIMATED_TOKEN_DETAILS: &str = "estimated_token_details";

/// Bytes per estimated token, matching the data plane's input approximation.
const BYTES_PER_TOKEN: u64 = 4;

/// Visible output and thinking observed on one attempt's event stream.
#[derive(Debug, Default, Clone)]
pub struct ReasoningEstimate {
    visible_bytes: u64,
    thinking: bool,
    readable_thinking: bool,
}

impl ReasoningEstimate {
    /// Observe every event of one stream in order.
    pub fn from_events(events: &[Event]) -> Self {
        let mut estimate = Self::default();
        for event in events {
            estimate.observe(event);
        }
        estimate
    }

    /// Record one event's visible bytes or thinking evidence.
    pub fn observe(&mut self, event: &Event) {
        match event {
            Event::TextDelta(text) | Event::RefusalDelta(text) => self.add(text),
            Event::ProviderTextDelta { delta, .. }
            | Event::ProviderRefusalDelta { delta, .. }
            | Event::ToolArgumentsDelta { delta, .. } => self.add(delta),
            // Thinking in any provider form: readable, summarized, plaintext,
            // signed, redacted or encrypted. Only the turn's evidence that it
            // reasoned matters; its text is never counted as visible.
            Event::ThinkingDelta { delta, .. }
            | Event::ReasoningSummaryDelta { delta, .. }
            | Event::ReasoningContentDelta { delta, .. }
            | Event::ReasoningTextDelta(delta)
                if !delta.is_empty() =>
            {
                self.thinking = true;
                self.readable_thinking = true;
            }
            Event::ThinkingDelta { .. }
            | Event::ThinkingSignature { .. }
            | Event::RedactedThinking { .. }
            | Event::ReasoningSummaryDelta { .. }
            | Event::ReasoningContentDelta { .. }
            | Event::EncryptedReasoning { .. }
            | Event::ReasoningTextDelta(_)
            | Event::GeminiThoughtPart(_) => self.thinking = true,
            _ => {}
        }
    }

    fn add(&mut self, text: &str) {
        self.visible_bytes = self.visible_bytes.saturating_add(text.len() as u64);
    }

    /// The estimated reasoning count, or `None` when the provider published
    /// one, no thinking ran, or the output total is unknown.
    ///
    /// A published `0` beside readable reasoning text is not a count: the
    /// stream itself shows tokens were spent thinking. OpenAI-compatible
    /// relays of models that bill thinking inside the completion total
    /// (Claude through a chat-completions passthrough) fill the details object
    /// with a zero they never measured, so such a turn is estimated like one
    /// that published nothing. A zero beside signature-only or no thinking
    /// stands.
    pub fn estimate(&self, usage: &Usage) -> Option<u64> {
        let published = match usage.reasoning_tokens {
            Some(0) => !self.readable_thinking,
            Some(_) => true,
            None => false,
        };
        if published || !self.thinking {
            return None;
        }
        let visible = self.visible_bytes.div_ceil(BYTES_PER_TOKEN);
        usage
            .output_tokens
            .map(|output| output.saturating_sub(visible))
    }

    /// Write the estimate into one encoded usage object.
    ///
    /// `details_key` is the wire's output-details object
    /// (`completion_tokens_details` on Chat, `output_tokens_details` on
    /// Responses). The field leaves `unreported_token_details` and joins
    /// `estimated_token_details`, so the count is never mistaken for one the
    /// provider reported.
    pub fn annotate(&self, wire: &mut Value, usage: Option<&Usage>, details_key: &str) {
        let Some(estimate) = usage.and_then(|usage| self.estimate(usage)) else {
            return;
        };
        let Some(object) = wire.as_object_mut() else {
            return;
        };
        let details = object.entry(details_key).or_insert(Value::Null);
        if !details.is_object() {
            *details = json!({});
        }
        details["reasoning_tokens"] = json!(estimate);
        if let Some(Value::Array(unreported)) = object.get_mut("unreported_token_details") {
            unreported.retain(|name| name != "reasoning_tokens");
            if unreported.is_empty() {
                object.remove("unreported_token_details");
            }
        }
        object.insert(ESTIMATED_TOKEN_DETAILS.into(), json!(["reasoning_tokens"]));
    }
}

#[cfg(test)]
#[path = "reasoning_estimate_tests.rs"]
mod tests;
