//! Gemini's declared finish freezes output before the transport finishes its meter.

use std::time::Instant;

use serde_json::Value;

use super::super::{malformed, Normalizer};
use crate::errors::Failure;
use crate::events::{bounded_ledger_sum, count_if_present, gemini_usage, Event, Usage};

#[derive(Default)]
pub(in crate::dialects) struct StreamState {
    pub(super) finish: Option<Event>,
    pub(super) finished_at: Option<Instant>,
    candidates: Option<u64>,
    thoughts: Option<u64>,
    pending: Option<Usage>,
}

impl Normalizer {
    /// Parse cumulative legs before folding additive thoughts. Missing fields in
    /// a later snapshot cannot erase an earlier count or its cache evidence.
    pub(super) fn observe_gemini_usage(&mut self, raw: &Value) -> Result<(), Failure> {
        let mut usage = gemini_usage(raw).map_err(|message| malformed(&message))?;
        let object = raw.as_object().expect("gemini_usage validated the object");
        let candidates = self.gemini.candidates.max(
            count_if_present(object, "candidatesTokenCount", "Gemini usageMetadata")
                .map_err(|message| malformed(&message))?,
        );
        let thoughts = self.gemini.thoughts.max(usage.reasoning_tokens);
        usage.output_tokens = Some(
            bounded_ledger_sum(
                &[candidates.unwrap_or(0), thoughts.unwrap_or(0)],
                "Gemini output",
            )
            .map_err(|message| malformed(&message))?,
        );
        usage.reasoning_tokens = thoughts;
        let mut accumulated = self.usage.clone().unwrap_or_default();
        if let Some(pending) = &self.gemini.pending {
            accumulated.merge_observed(pending);
        }
        accumulated.merge_observed(&usage);
        if accumulated.cached_input_tokens > accumulated.input_tokens {
            // Never expose an impossible cache subset or manufacture prompt
            // tokens to make it fit. Without a valid baseline, hold sparse
            // evidence for a later primary report; otherwise reject only this
            // contradictory update and keep the last consistent meter.
            if self.usage.is_none() {
                self.gemini.pending = Some(accumulated);
                self.gemini.candidates = candidates;
                self.gemini.thoughts = thoughts;
            }
            return Ok(());
        }
        self.gemini.pending = None;
        self.gemini.candidates = candidates;
        self.gemini.thoughts = thoughts;
        self.usage = Some(accumulated);
        Ok(())
    }

    /// The absolute start of the metadata-only phase. Transport bytes and
    /// downstream backpressure never renew this window.
    pub(crate) fn metadata_drain_started(&self) -> Option<Instant> {
        self.gemini.finished_at.filter(|_| !self.terminal)
    }

    /// Finish an already declared Gemini outcome once transport closes, times
    /// out, fails, or is cancelled. No further provider read is needed, and the
    /// shared settlement owner receives exactly one terminal after the meter.
    pub(crate) fn finish_metadata_drain(&mut self) -> Vec<Event> {
        let Some(terminal) = self.gemini.finish.take() else {
            return Vec::new();
        };
        self.terminal = true;
        let mut events = Vec::new();
        if let Some(usage) = self.usage.as_ref() {
            events.push(Event::Usage(usage.clone()));
        }
        events.push(terminal);
        events
    }
}
