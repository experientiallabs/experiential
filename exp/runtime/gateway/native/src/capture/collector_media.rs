//! The collector's media surfaces: binary parts, buffered answers and provider jobs.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use serde_json::{json, Value};

use super::super::media::{MediaPart, MediaRole, ProviderJob, VectorSummary};
use super::super::record::Response;
use super::{Collector, Entry};

/// Strip the request half from a record the host keeps only the response of.
///
/// The context becomes the empty schema 1 shell, the redacted wire request and
/// spilled messages go, and caller-sent media parts are dropped (their files
/// are removed with their last handle).
pub(super) fn omit_request(entry: &mut Entry) {
    entry.record.request.context = Arc::new(json!({"schema_version": 1, "request": {}}));
    entry.record.messages_part = None;
    entry.record.transport_body = None;
    if let Some(Value::Object(transport)) = entry.record.transport.as_mut() {
        transport.remove("wire_request");
    }
    entry.record.media.omit(MediaRole::Input);
}

impl Collector {
    /// Whether `request_id` has a live capture entry.
    pub(crate) fn registered(&self, request_id: &str) -> bool {
        self.pending
            .lock()
            .is_ok_and(|pending| pending.entries.contains_key(request_id))
    }

    /// Add parts to a live entry, charging their descriptions to the pending budget.
    /// The bytes themselves are on disk (or not kept), never in this budget.
    pub(crate) fn attach_media(&self, request_id: &str, parts: Vec<MediaPart>) -> bool {
        let charge = parts.len() * (std::mem::size_of::<MediaPart>() + 256);
        let Ok(mut pending) = self.pending.lock() else {
            return false;
        };
        self.expire(&mut pending);
        if self.retained_bytes(&pending).saturating_add(charge) > self.config.maximum_pending_bytes
        {
            return false;
        }
        let Some(entry) = pending.entries.get_mut(request_id) else {
            return false;
        };
        entry.record.media.parts.extend(parts);
        entry.bytes += charge;
        pending.bytes += charge;
        true
    }

    /// Finish a buffered media answer: its JSON projection plus its output parts.
    pub(crate) fn finish_media(
        &self,
        request_id: &str,
        response: Response,
        parts: Vec<MediaPart>,
        vector_summary: Option<VectorSummary>,
        deployment_id: Option<String>,
    ) -> bool {
        if !parts.is_empty() && !self.attach_media(request_id, parts) {
            return self.skip();
        }
        if vector_summary.is_some() {
            if let Ok(mut pending) = self.pending.lock() {
                if let Some(entry) = pending.entries.get_mut(request_id) {
                    entry.record.media.vector_summary = vector_summary;
                }
            }
        }
        self.finish_output(request_id, Some(response), None, deployment_id)
    }

    /// Record a provider job's current state as the response of its request.
    ///
    /// The first call (the job's creation) finishes the request's output; a
    /// later call (the job's terminal state) replaces that response while the
    /// entry still waits for settlement. Returns false when no entry remains,
    /// for example after settlement already emitted the record.
    pub(crate) fn finish_job(
        &self,
        request_id: &str,
        body: Value,
        job: ProviderJob,
        deployment_id: Option<String>,
    ) -> bool {
        let mut body = body;
        let source_json = super::super::response::lossless_projection(&mut body);
        let response = Response::Json {
            status: 200,
            body,
            source_json,
        };
        let finished = {
            let Ok(mut pending) = self.pending.lock() else {
                return false;
            };
            self.expire(&mut pending);
            let Some(entry) = pending.entries.get_mut(request_id) else {
                return false;
            };
            entry.record.media.provider_job = Some(job);
            entry.output_finished
        };
        if !finished {
            if self.attach(request_id).is_none() {
                return false;
            }
            return self.finish_output(request_id, Some(response), None, deployment_id);
        }
        let Ok(mut pending) = self.pending.lock() else {
            return false;
        };
        let added = response.heap_bytes();
        if self.retained_bytes(&pending).saturating_add(added) > self.config.maximum_pending_bytes {
            return false;
        }
        let Some(entry) = pending.entries.get_mut(request_id) else {
            return false;
        };
        if entry.response_discarded.load(Ordering::Acquire) {
            return false;
        }
        let removed = entry
            .record
            .response
            .as_ref()
            .map_or(0, Response::heap_bytes);
        entry.record.response = Some(response);
        if deployment_id.is_some() {
            entry.record.deployment_id = deployment_id;
        }
        entry.bytes = entry.bytes.saturating_add(added).saturating_sub(removed);
        pending.bytes = pending.bytes.saturating_add(added).saturating_sub(removed);
        true
    }
}
