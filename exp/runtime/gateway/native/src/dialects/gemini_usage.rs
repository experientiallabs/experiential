//! Gemini's declared finish freezes output before the transport finishes its meter.

use std::time::Instant;

use serde_json::Value;

use super::super::{malformed, Normalizer};
use crate::errors::Failure;
use crate::events::{bounded_ledger_sum, count_if_present, gemini_usage, Event};

#[derive(Default)]
pub(in crate::dialects) struct StreamState {
    pub(super) finish: Option<Event>,
    pub(super) finished_at: Option<Instant>,
    candidates: Option<u64>,
    thoughts: Option<u64>,
    pending_cache: Option<u64>,
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
        accumulated.merge_observed(&usage);
        let pending_cache = self
            .gemini
            .pending_cache
            .max(accumulated.cached_input_tokens);
        if accumulated.cached_input_tokens > accumulated.input_tokens {
            // The current report is contradictory, not merely waiting for an
            // older cache subset. Preserve its cache evidence without exposing
            // the invalid meter or accepting its other legs as a new baseline.
            self.gemini.pending_cache = pending_cache;
            return Ok(());
        }
        if pending_cache <= accumulated.input_tokens {
            accumulated.cached_input_tokens = pending_cache;
            self.gemini.pending_cache = None;
        } else {
            // An older unresolved subset must not block newer consistent
            // primary counts. Publish those now and retain only the subset
            // until sufficient input evidence arrives, never fabricating input.
            self.gemini.pending_cache = pending_cache;
            if self.usage.is_none() && accumulated.input_tokens == Some(0) {
                return Ok(());
            }
        }
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
