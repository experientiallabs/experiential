//! Gemini's declared finish freezes output before the transport finishes its meter.

use std::time::Instant;

use serde_json::Value;

use super::super::{malformed, Normalizer};
use crate::errors::Failure;
use crate::events::{bounded_ledger_sum, count_if_present, gemini_usage, Event, Usage};

/// Parsed cumulative evidence is independent of the last publishable meter.
/// Missing protobuf scalar fields are zero defaults, never resets of evidence.
#[derive(Clone, Copy, Default)]
struct MeterFields {
    input: u64,
    candidates: u64,
    thoughts: Option<u64>,
    cache: u64,
}

#[derive(Default)]
pub(in crate::dialects) struct StreamState {
    pub(super) finish: Option<Event>,
    pub(super) finished_at: Option<Instant>,
    fields: MeterFields,
}

impl Normalizer {
    /// Accumulate validated numeric fields before deciding whether their cache
    /// subset is publishable. Withholding a meter never drops its output legs.
    pub(super) fn observe_gemini_usage(&mut self, raw: &Value) -> Result<(), Failure> {
        let parsed = gemini_usage(raw).map_err(|message| malformed(&message))?;
        let object = raw.as_object().expect("gemini_usage validated the object");
        let count = |key| {
            count_if_present(object, key, "Gemini usageMetadata")
                .map_err(|message| malformed(&message))
        };
        let input = count("promptTokenCount")?;
        let cache = count("cachedContentTokenCount")?;
        let candidates = count("candidatesTokenCount")?;
        let previous = self.gemini.fields;
        let cache = previous.cache.max(cache.unwrap_or(0));
        // Two explicit counts in the same report contradict its own subset
        // relation. Retain cache evidence for reconciliation, but do not trust
        // that report's primary/output fields, even after an empty suffix.
        if input.is_some_and(|input| parsed.cached_input_tokens.unwrap_or(0) > input) {
            self.gemini.fields.cache = cache;
            return Ok(());
        }
        let fields = MeterFields {
            input: previous.input.max(input.unwrap_or(0)),
            candidates: previous.candidates.max(candidates.unwrap_or(0)),
            thoughts: previous.thoughts.max(parsed.reasoning_tokens),
            cache,
        };
        let output = bounded_ledger_sum(
            &[fields.candidates, fields.thoughts.unwrap_or(0)],
            "Gemini output",
        )
        .map_err(|message| malformed(&message))?;
        self.gemini.fields = fields;
        // Sparse cache can arrive before prompt counts, including before any
        // valid baseline. Keep all parsed legs pending rather than emitting an
        // impossible zero-input meter or inventing input to cover the subset.
        if fields.cache > fields.input && fields.input == 0 && self.usage.is_none() {
            return Ok(());
        }
        let cache = if fields.cache <= fields.input {
            fields.cache
        } else {
            // Pending cache never blocks newer primary evidence. Retain the
            // last safe subset until enough input arrives to publish it.
            self.usage
                .as_ref()
                .and_then(|usage| usage.cached_input_tokens)
                .unwrap_or(0)
        };
        self.usage = Some(Usage {
            input_tokens: Some(fields.input),
            output_tokens: Some(output),
            cached_input_tokens: Some(cache),
            reasoning_tokens: fields.thoughts,
            ..Usage::default()
        });
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
