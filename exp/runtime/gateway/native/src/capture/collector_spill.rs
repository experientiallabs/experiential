//! The collector's disk spill: large request messages and retained text bounds.

use std::sync::Arc;

use super::super::record::Request;
use super::Collector;

impl Collector {
    pub(crate) fn spill(&self) -> Option<&Arc<super::super::spill::SpillStore>> {
        self.spill.as_ref()
    }

    /// Hand a large request's messages to a background part writer, so the
    /// capture keeps only the rest of the context (copied here, small) while
    /// the request runs. The writer frees its copy of the messages as soon as
    /// they are on disk, instead of the capture holding them to delivery.
    pub(super) fn spill_messages(
        &self,
        request: &mut Request,
        request_bytes: usize,
    ) -> Option<Arc<super::super::spill::PendingPart>> {
        let store = self.spill.as_ref()?;
        if request_bytes <= self.config.delivery.maximum_record_bytes / 2
            || request.context.get("request")?.get("messages").is_none()
        {
            return None;
        }
        let original = request.context.clone();
        let charge = request_bytes;
        let messages = original.clone();
        let part = store.write_in_background(charge, move |out| {
            let messages = &messages["request"]["messages"];
            serde_json::to_writer(out, messages).map_err(std::io::Error::other)
        })?;
        request.context = Arc::new(super::super::split::context_without_messages(&original));
        Some(part)
    }

    /// In-memory bound for retained provider text (reasoning, tool calls,
    /// thought parts). With spill the record is split instead of trimmed, so
    /// only the pending memory budget applies.
    pub(crate) fn retained_text_limit(&self) -> usize {
        if self.spill.is_some() {
            self.config.maximum_pending_bytes
        } else {
            self.config.maximum_response_bytes
        }
    }
}
