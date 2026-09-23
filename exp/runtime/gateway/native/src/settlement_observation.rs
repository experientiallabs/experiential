//! Bounded provider facts retained across cancellation of an owning future.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use crate::events::{Event, Usage};
use crate::relay::track_event;

/// Longest generated text retained per leg for a disconnect estimate. A
/// 64k-token answer sits well under this; text past it is counted as
/// overflow characters so the estimate extrapolates instead of forgetting.
const STREAMED_OUTPUT_RETAINED_BYTES: usize = 1 << 20;

/// Generated output observed before any provider terminal, kept verbatim so
/// the control plane's tokenizer can estimate what the provider generated
/// when a cancelled attempt never received the provider's final meter.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct StreamedOutput {
    /// Visible text, refusal text, and tool-call arguments.
    pub text: String,
    /// Reasoning, thinking, and reasoning-summary text.
    pub reasoning: String,
    /// Characters of text past the retained bound.
    pub text_overflow_chars: u64,
    /// Characters of reasoning past the retained bound.
    pub reasoning_overflow_chars: u64,
    /// Generated images, billed per image rather than per token: any at all
    /// leaves the meter unestimable from text.
    pub images: u64,
}

impl StreamedOutput {
    /// Retain as much of `delta` as still fits (cut at a character boundary)
    /// and count the rest as overflow, so a delta crossing the bound neither
    /// vanishes nor skews the density the overflow is extrapolated from.
    fn append(retained: &mut String, overflow: &mut u64, delta: &str) {
        let room = STREAMED_OUTPUT_RETAINED_BYTES.saturating_sub(retained.len());
        if delta.len() <= room {
            retained.push_str(delta);
            return;
        }
        let mut cut = room;
        while cut > 0 && !delta.is_char_boundary(cut) {
            cut -= 1;
        }
        retained.push_str(&delta[..cut]);
        *overflow += delta[cut..].chars().count() as u64;
    }

    /// Retain one normalized event's generated text; anything that is not
    /// generated text (images, signatures, structure, usage) is not output.
    fn record(&mut self, event: &Event) {
        match event {
            Event::TextDelta(delta)
            | Event::RefusalDelta(delta)
            | Event::ProviderTextDelta { delta, .. }
            | Event::ProviderRefusalDelta { delta, .. }
            | Event::ToolArgumentsDelta { delta, .. }
            | Event::ServerToolArgumentsDelta { delta, .. } => {
                Self::append(&mut self.text, &mut self.text_overflow_chars, delta)
            }
            Event::ReasoningSummaryDelta { delta, .. }
            | Event::ThinkingDelta { delta, .. }
            | Event::ReasoningContentDelta { delta, .. } => Self::append(
                &mut self.reasoning,
                &mut self.reasoning_overflow_chars,
                delta,
            ),
            Event::Image(_) => self.images += 1,
            _ => {}
        }
    }
}

/// Shared only by one physical attempt's guard and relay, reset at rebind.
#[derive(Clone, Default)]
pub(crate) struct Observation(Arc<Mutex<Observed>>);

#[derive(Clone)]
pub(crate) struct Observed {
    pub started_at: SystemTime,
    started: Instant,
    pub terminal_at: Option<SystemTime>,
    pub duration: Option<Duration>,
    pub usage: Option<Usage>,
    pub tool_names: Vec<String>,
    pub terminal: Option<Event>,
    pub first_token_at: Option<SystemTime>,
    pub streamed_output: StreamedOutput,
}

impl Default for Observed {
    fn default() -> Self {
        Self {
            started_at: SystemTime::now(),
            started: Instant::now(),
            terminal_at: None,
            duration: None,
            usage: None,
            tool_names: Vec::new(),
            terminal: None,
            first_token_at: None,
            streamed_output: StreamedOutput::default(),
        }
    }
}

impl Observation {
    /// Meter private Gemini thought text without retaining capture parts or emitting events.
    pub(crate) fn record_gemini_reasoning(&self, text: &str) {
        let mut observed = self.0.lock().unwrap_or_else(|error| error.into_inner());
        if observed.terminal.is_none() {
            let output = &mut observed.streamed_output;
            StreamedOutput::append(
                &mut output.reasoning,
                &mut output.reasoning_overflow_chars,
                text,
            );
        }
    }

    /// Remember normalized facts before public delivery can suspend or fail.
    pub(crate) fn record(&self, event: &Event) {
        let mut observed = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // A queued provider terminal fixes the meter/outcome. Earlier tool
        // events may still emerge from normalization before public delivery.
        if observed.terminal.is_some() && (event.is_terminal() || matches!(event, Event::Usage(_)))
        {
            return;
        }
        if let Event::Usage(usage) = event {
            // Partial meters are observations too, never an invented zero.
            if usage.input_tokens.is_some() || usage.output_tokens.is_some() {
                match &mut observed.usage {
                    Some(previous) => previous.merge_observed(usage),
                    None => observed.usage = Some(usage.clone()),
                }
            }
        } else {
            let Observed {
                usage,
                tool_names,
                streamed_output,
                ..
            } = &mut *observed;
            track_event(event, usage, tool_names);
            streamed_output.record(event);
        }
        if event.is_terminal() {
            observed.terminal = Some(event.clone());
            observed.terminal_at = Some(SystemTime::now());
            observed.duration = Some(observed.started.elapsed());
        }
    }

    /// Stamp the effective terminal after normalized output delivery. A queued
    /// terminal still supplies cancellation evidence before this point.
    pub(crate) fn record_effective_terminal(&self, event: &Event) {
        if event.is_terminal() {
            let mut observed = self
                .0
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            observed.terminal_at = Some(SystemTime::now());
            observed.duration = Some(observed.started.elapsed());
            observed.terminal = Some(event.clone());
        }
    }

    pub(crate) fn record_first_token(&self, at: Option<SystemTime>) {
        let mut observed = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if observed.first_token_at.is_none() {
            observed.first_token_at = at;
        }
    }

    pub(crate) fn snapshot(&self) -> Observed {
        self.0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffered_terminal_time_finishes_after_the_visible_first_token() {
        let observation = Observation::default();
        // A complete provider frame can queue its terminal before the relay
        // yields the frame's first visible token.
        observation.record(&Event::Completed);
        std::thread::sleep(Duration::from_millis(1));
        let first_token_at = SystemTime::now();
        observation.record_first_token(Some(first_token_at));
        observation.record_effective_terminal(&Event::Completed);
        let snapshot = observation.snapshot();
        assert!(snapshot.terminal_at.unwrap() >= first_token_at);
        assert!(snapshot.duration.unwrap() >= Duration::from_millis(1));
    }

    #[test]
    fn streamed_output_retains_generated_text_by_leg_and_counts_overflow() {
        let observation = Observation::default();
        observation.record(&Event::TextDelta("Hello".into()));
        observation.record(&Event::ToolArgumentsDelta {
            index: 0,
            delta: "{\"q\":1}".into(),
        });
        observation.record(&Event::ReasoningContentDelta {
            route_sha256: "r".into(),
            delta: "think".into(),
        });
        observation.record(&Event::ThinkingSignature {
            index: 0,
            signature: "sig".into(),
        });
        observation.record(&Event::Image("data:image/png;base64,AAAA".into()));
        let streamed = observation.snapshot().streamed_output;
        assert_eq!(streamed.text, "Hello{\"q\":1}");
        assert_eq!(streamed.reasoning, "think");
        assert_eq!(streamed.text_overflow_chars, 0);
        assert_eq!(streamed.images, 1);

        let mut bounded = StreamedOutput::default();
        bounded.record(&Event::TextDelta(
            "x".repeat(STREAMED_OUTPUT_RETAINED_BYTES),
        ));
        bounded.record(&Event::TextDelta("ééé".into()));
        assert_eq!(bounded.text.len(), STREAMED_OUTPUT_RETAINED_BYTES);
        assert_eq!(bounded.text_overflow_chars, 3);
        assert_eq!(bounded.images, 0);

        // A delta crossing the bound keeps the part that fits, cut at a
        // character boundary, and counts only the remainder as overflow.
        let mut crossing = StreamedOutput::default();
        crossing.record(&Event::TextDelta(
            "x".repeat(STREAMED_OUTPUT_RETAINED_BYTES - 1),
        ));
        crossing.record(&Event::TextDelta("abc".into()));
        assert_eq!(crossing.text.len(), STREAMED_OUTPUT_RETAINED_BYTES);
        assert!(crossing.text.ends_with("xa"));
        assert_eq!(crossing.text_overflow_chars, 2);
        let mut split = StreamedOutput::default();
        split.record(&Event::TextDelta(
            "x".repeat(STREAMED_OUTPUT_RETAINED_BYTES - 1),
        ));
        split.record(&Event::TextDelta("éé".into()));
        assert_eq!(split.text.len(), STREAMED_OUTPUT_RETAINED_BYTES - 1);
        assert_eq!(split.text_overflow_chars, 2);
    }

    #[test]
    fn cancellation_preserves_partial_usage_and_terminal_precedence() {
        let observation = Observation::default();
        observation.record(&Event::Usage(Usage {
            input_tokens: Some(12),
            ..Usage::default()
        }));
        observation.record(&Event::Incomplete);
        observation.record(&Event::Completed);
        let snapshot = observation.snapshot();
        let usage = snapshot.usage.unwrap();
        assert_eq!(usage.input_tokens, Some(12));
        assert_eq!(usage.output_tokens, None);
        assert!(matches!(snapshot.terminal, Some(Event::Incomplete)));
    }
}
