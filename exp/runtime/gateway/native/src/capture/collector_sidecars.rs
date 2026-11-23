//! Bounded provider sidecars, sealed when a request ends without exposed output.

use std::sync::{atomic::Ordering, Arc};

use super::Collector;

impl Collector {
    pub(crate) fn captures_hidden_reasoning(&self) -> bool {
        self.config.capture_hidden_reasoning
    }

    pub(crate) fn reasoning(&self, request_id: &str, delta: &str) {
        if delta.is_empty() {
            return;
        }
        let Ok(mut pending) = self.pending.lock() else {
            return;
        };
        self.expire(&mut pending);
        if pending
            .entries
            .get(request_id)
            .is_some_and(|entry| entry.unexposed)
        {
            return;
        }
        let Some(mut entry) = pending.entries.remove(request_id) else {
            return;
        };
        pending.bytes -= entry.map_bytes();
        let text = entry
            .record
            .provider_reasoning
            .get_or_insert_with(String::new);
        let previous = text.capacity();
        let required = text.len().saturating_add(delta.len());
        if required > self.retained_text_limit()
            || self
                .retained_bytes(&pending)
                .saturating_add(
                    entry
                        .bytes
                        .saturating_sub(entry._admission.retained_bytes.load(Ordering::Acquire)),
                )
                .saturating_add(delta.len())
                > self.config.maximum_pending_bytes
            || text.try_reserve_exact(delta.len()).is_err()
        {
            self.skip();
            return;
        }
        entry.bytes += text.capacity() - previous;
        if self.retained_bytes(&pending).saturating_add(
            entry
                .bytes
                .saturating_sub(entry._admission.retained_bytes.load(Ordering::Acquire)),
        ) > self.config.maximum_pending_bytes
        {
            self.skip();
            return;
        }
        text.push_str(delta);
        pending.bytes += entry.map_bytes();
        pending.entries.insert(request_id.to_owned(), entry);
    }

    /// Share bounded provider parts without JSON encoding or copying their content.
    pub(crate) fn gemini_thought_part(&self, request_id: &str, part: Arc<serde_json::Value>) {
        let Ok(mut pending) = self.pending.lock() else {
            return;
        };
        self.expire(&mut pending);
        if pending
            .entries
            .get(request_id)
            .is_some_and(|entry| entry.unexposed)
        {
            return;
        }
        let Some(mut entry) = pending.entries.remove(request_id) else {
            return;
        };
        pending.bytes -= entry.map_bytes();
        let previous = entry.record.gemini_thought_parts.capacity()
            * std::mem::size_of::<Arc<serde_json::Value>>();
        let heap = super::super::budget::heap_bytes(&part) + 64;
        let part_bytes = super::super::budget::json_bytes(&part);
        if entry.gemini_part_bytes.saturating_add(part_bytes) > self.retained_text_limit()
            || self
                .retained_bytes(&pending)
                .saturating_add(
                    entry
                        .bytes
                        .saturating_sub(entry._admission.retained_bytes.load(Ordering::Acquire)),
                )
                .saturating_add(heap + 64)
                > self.config.maximum_pending_bytes
            || entry
                .record
                .gemini_thought_parts
                .try_reserve_exact(1)
                .is_err()
        {
            self.skip();
            return;
        }
        entry.record.gemini_thought_parts.push(part);
        entry.gemini_part_bytes += part_bytes;
        entry.bytes += heap
            + entry.record.gemini_thought_parts.capacity()
                * std::mem::size_of::<Arc<serde_json::Value>>()
            - previous;
        if self.retained_bytes(&pending).saturating_add(
            entry
                .bytes
                .saturating_sub(entry._admission.retained_bytes.load(Ordering::Acquire)),
        ) > self.config.maximum_pending_bytes
        {
            self.skip();
            return;
        }
        pending.bytes += entry.map_bytes();
        pending.entries.insert(request_id.to_owned(), entry);
    }

    /// Retain provider-order argument text even when a public protocol parses it.
    pub(crate) fn tool_call(&self, request_id: &str, call: &crate::events::CompletedToolCall) {
        let Ok(encoded) = serde_json::to_string(&serde_json::json!({
            "call_id": call.call_id, "name": call.name, "raw_arguments": call.raw_arguments,
            "namespace": call.namespace, "caller": call.caller, "custom": call.custom,
            "provider_item_id": call.provider_item_id,
            "provider_status": call.provider_status.map(|status| status.as_str()),
        })) else {
            return;
        };
        let Ok(mut pending) = self.pending.lock() else {
            return;
        };
        self.expire(&mut pending);
        if pending
            .entries
            .get(request_id)
            .is_some_and(|entry| entry.unexposed)
        {
            return;
        }
        let Some(mut entry) = pending.entries.remove(request_id) else {
            return;
        };
        pending.bytes -= entry.map_bytes();
        let text = entry
            .record
            .provider_tool_calls_json
            .get_or_insert_with(|| "[]".to_owned());
        let previous = text.capacity();
        let extra = encoded.len() + 1;
        if text.len().saturating_add(extra) > self.retained_text_limit()
            || self
                .retained_bytes(&pending)
                .saturating_add(
                    entry
                        .bytes
                        .saturating_sub(entry._admission.retained_bytes.load(Ordering::Acquire)),
                )
                .saturating_add(extra + 2)
                > self.config.maximum_pending_bytes
            || text.try_reserve_exact(extra).is_err()
        {
            self.skip();
            return;
        }
        // Charge the initial [] as well as actual allocator growth.
        entry.bytes += text.capacity() - previous + if text == "[]" { previous } else { 0 };
        if self.retained_bytes(&pending).saturating_add(
            entry
                .bytes
                .saturating_sub(entry._admission.retained_bytes.load(Ordering::Acquire)),
        ) > self.config.maximum_pending_bytes
        {
            self.skip();
            return;
        }
        text.pop();
        if text.len() > 1 {
            text.push(',');
        }
        text.push_str(&encoded);
        text.push(']');
        pending.bytes += entry.map_bytes();
        pending.entries.insert(request_id.to_owned(), entry);
    }
}
