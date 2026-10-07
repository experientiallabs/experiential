//! Collector methods for gateway-requested token probabilities (see `super::super::logprobs`).

use super::Collector;

impl Collector {
    /// Whether the host enabled capture-only probabilities at all.
    pub(crate) fn captures_logprobs(&self) -> bool {
        self.config.capture_logprobs
    }

    /// Whether gateway-requested probabilities apply to this admitted request; a
    /// yes reserves the record's (unmarked) sidecar so a later dial's probabilities
    /// always have somewhere to go.
    pub(crate) fn wants_logprobs(&self, request_id: &str) -> bool {
        let Ok(mut pending) = self.pending.lock() else {
            return false;
        };
        let bytes = super::super::logprobs::Captured::default().heap_bytes();
        let room = self.retained_bytes(&pending) + bytes <= self.config.maximum_pending_bytes;
        let Some(entry) = pending.entries.get_mut(request_id) else {
            return false;
        };
        if !self.config.capture_logprobs || !(room || entry.record.provider_logprobs.is_some()) {
            return false;
        }
        if entry.record.provider_logprobs.is_none() {
            entry.record.provider_logprobs = Some(super::super::logprobs::Captured::default());
            entry.bytes += bytes;
            pending.bytes += bytes;
        }
        true
    }

    pub(crate) fn logprobs_injected(&self, request_id: &str) {
        if let Ok(mut pending) = self.pending.lock() {
            if let Some(captured) = pending
                .entries
                .get_mut(request_id)
                .and_then(|entry| entry.record.provider_logprobs.as_mut())
            {
                captured.logprobs_injected = true;
            }
        }
    }

    /// Append capture-only probabilities; an overflow truncates them, never the exchange.
    pub(crate) fn logprobs(
        &self,
        request_id: &str,
        deltas: Vec<crate::logprobs::ChoiceLogprobsDelta>,
        truncated: bool,
    ) {
        if deltas.is_empty() && !truncated {
            return;
        }
        let Ok(mut pending) = self.pending.lock() else {
            return;
        };
        self.expire(&mut pending);
        let retained = self.retained_bytes(&pending);
        let maximum_pending = self.config.maximum_pending_bytes;
        let maximum_response = self.config.maximum_response_bytes;
        let Some(entry) = pending.entries.get_mut(request_id) else {
            return;
        };
        let record_room = super::super::logprobs::record_room(
            self.config.delivery.maximum_record_bytes,
            entry.request_bytes,
            maximum_response,
        );
        let Some(captured) = entry
            .record
            .provider_logprobs
            .as_mut()
            .filter(|captured| captured.logprobs_injected)
        else {
            return;
        };
        let already = captured.bytes;
        let added = captured.extend(deltas, truncated, |extra| {
            let total = already.saturating_add(extra);
            total <= maximum_response
                && total <= record_room
                && retained.saturating_add(extra) <= maximum_pending
        });
        entry.bytes += added;
        pending.bytes += added;
    }
}

#[cfg(test)]
#[path = "collector_logprobs_test.rs"]
mod tests;
