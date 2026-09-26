//! Bounded rendezvous of authenticated input, terminal policy and observed output.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use super::budget::string_bytes;
use super::delivery::{Delivery, Limits, Sink};
use super::record::{Record, Request, Response, SCHEMA_VERSION};
use super::response::WireResponse;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Configuration {
    pub delivery: Limits,
    pub maximum_pending_records: usize,
    pub maximum_pending_bytes: usize,
    pub maximum_request_bytes: usize,
    pub maximum_response_bytes: usize,
    pub ttl_seconds: u64,
    pub settlement_required: bool,
    #[serde(default)]
    pub relay_metadata: bool,
    #[serde(default)]
    pub truncate_request: bool,
    #[serde(default)]
    pub asynchronous_delivery: bool,
}

impl Configuration {
    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        self.delivery.validate()?;
        if !(1..=4096).contains(&self.maximum_pending_records)
            || !(1..=8 * 1024 * 1024).contains(&self.maximum_request_bytes)
            || !(1..=4 * 1024 * 1024).contains(&self.maximum_response_bytes)
            || self.maximum_pending_bytes < self.maximum_request_bytes
            || self.maximum_pending_bytes < self.maximum_response_bytes
            || self.maximum_pending_bytes > 256 * 1024 * 1024
            || !(1..=3600).contains(&self.ttl_seconds)
        {
            return Err("invalid capture collector bounds");
        }
        Ok(())
    }
}

/// Stay live through the pending-to-delivery handoff, including capacity waits.
pub(super) struct Admission {
    count: Arc<AtomicUsize>,
    handoff_bytes: Arc<AtomicUsize>,
    retained_bytes: AtomicUsize,
}

impl Admission {
    /// Transfer the map's charge before releasing its lock, without a capacity gap.
    fn handoff(&self, bytes: usize) {
        let previous = self.retained_bytes.fetch_max(bytes, Ordering::AcqRel);
        self.handoff_bytes
            .fetch_add(bytes.saturating_sub(previous), Ordering::AcqRel);
    }
}

impl Drop for Admission {
    fn drop(&mut self) {
        self.handoff_bytes.fetch_sub(
            self.retained_bytes.load(Ordering::Acquire),
            Ordering::AcqRel,
        );
        self.count.fetch_sub(1, Ordering::AcqRel);
    }
}

struct Entry {
    _admission: Arc<Admission>,
    record: Record,
    observation: Option<crate::settlement::Observation>,
    gemini_part_bytes: usize,
    wire: Option<WireResponse>,
    relay: Option<super::relay::Relay>,
    relay_attached: bool,
    relay_required: bool,
    request_bytes: usize,
    expires: Instant,
    bytes: usize,
    attached: bool,
    output_finished: bool,
    checkpointing: bool,
    checkpoint_started: bool,
    unexposed: bool,
    response_allowed: bool,
    response_discarded: Arc<AtomicBool>,
}

impl Entry {
    /// The map owns only bytes not already pinned by this request's shared admission.
    fn map_bytes(&self) -> usize {
        self.bytes
            .saturating_sub(self._admission.retained_bytes.load(Ordering::Acquire))
    }
}

/// Accepted checkpoint ownership survives its request waiter and terminal map removal.
pub(super) struct CheckpointLease {
    admission: Arc<Admission>,
    pending: Arc<Mutex<Pending>>,
    request_id: String,
    ttl: Duration,
}

impl Drop for CheckpointLease {
    fn drop(&mut self) {
        if let Ok(mut pending) = self.pending.lock() {
            if let Some(entry) = pending.entries.get_mut(&self.request_id) {
                if Arc::ptr_eq(&entry._admission, &self.admission) {
                    entry.checkpointing = false;
                    entry.expires = Instant::now() + self.ttl;
                }
            }
        }
    }
}

#[derive(Default)]
struct Pending {
    entries: HashMap<String, Entry>,
    bytes: usize,
    closed: bool,
}

struct MaintainedSink<S> {
    sink: S,
    pending: Arc<Mutex<Pending>>,
    skipped: Arc<AtomicU64>,
}

impl<S: Sink> Sink for MaintainedSink<S> {
    type Prepared = S::Prepared;

    fn preparation_bytes(maximum_record_bytes: usize) -> usize {
        S::preparation_bytes(maximum_record_bytes)
    }

    fn prepare(&self, record: &Record, maximum_bytes: usize) -> Result<Self::Prepared, ()> {
        self.sink.prepare(record, maximum_bytes)
    }

    fn write(&mut self, prepared: &Self::Prepared) -> Result<(), ()> {
        self.sink.write(prepared)
    }

    fn batch_records(&self) -> usize {
        self.sink.batch_records()
    }

    fn batch_bytes(&self) -> usize {
        self.sink.batch_bytes()
    }

    fn batch_delay(&self) -> Duration {
        self.sink.batch_delay()
    }

    fn prepared_bytes(&self, prepared: &Self::Prepared) -> usize {
        self.sink.prepared_bytes(prepared)
    }

    fn write_batch(&mut self, prepared: &[&Self::Prepared]) -> Vec<bool> {
        self.sink.write_batch(prepared)
    }

    fn take_maintenance_failures(&mut self) -> u64 {
        self.sink.take_maintenance_failures()
    }

    #[cfg(test)]
    fn before_receive(&mut self) {
        self.sink.before_receive();
    }

    fn maintain(&mut self) -> Result<(), ()> {
        if let Ok(mut pending) = self.pending.lock() {
            expire_pending(&mut pending, &self.skipped);
        }
        self.sink.maintain()
    }
}

fn expire_pending(pending: &mut Pending, skipped: &AtomicU64) {
    let now = Instant::now();
    pending.entries.retain(|_, entry| {
        if entry.expires <= now && !entry.checkpointing {
            pending.bytes -= entry.map_bytes();
            skipped.fetch_add(1, Ordering::Relaxed);
            false
        } else {
            true
        }
    });
}

pub(crate) struct Collector {
    pub config: Configuration,
    delivery: Delivery,
    pending: Arc<Mutex<Pending>>,
    skipped: Arc<AtomicU64>,
    admissions: Arc<AtomicUsize>,
    handoff_bytes: Arc<AtomicUsize>,
    body_bytes: AtomicUsize,
    body_capacity: Arc<Semaphore>,
}

impl Collector {
    pub(crate) fn new(config: Configuration, sink: impl Sink) -> Result<Self, &'static str> {
        config.validate()?;
        let pending = Arc::new(Mutex::new(Pending::default()));
        let skipped = Arc::new(AtomicU64::new(0));
        let sink = MaintainedSink {
            sink,
            pending: pending.clone(),
            skipped: skipped.clone(),
        };
        Ok(Self {
            delivery: Delivery::new(config.delivery.clone(), sink)?,
            body_capacity: Arc::new(Semaphore::new(config.maximum_pending_bytes)),
            config,
            pending,
            skipped,
            admissions: Arc::new(AtomicUsize::new(0)),
            handoff_bytes: Arc::new(AtomicUsize::new(0)),
            body_bytes: AtomicUsize::new(0),
        })
    }

    /// Admission is the sole authority for input. Duplicate ids never replace a record.
    pub(crate) fn begin(&self, mut request: Request) -> bool {
        let request_bytes = if self.config.truncate_request {
            super::bounds::bound(&mut request, self.config.maximum_request_bytes)
        } else {
            request.json_bytes()
        };
        let record = Record {
            checkpointed: false,
            schema_version: SCHEMA_VERSION,
            request,
            response: None,
            transport: None,
            provider_reasoning: None,
            provider_reasoning_source_json: None,
            provider_tool_calls_json: None,
            deployment_id: None,
            metrics: None,
            gemini_thought_parts: Vec::new(),
            gemini_thought_parts_source_json: None,
            captured_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_secs_f64())
                .unwrap_or(0.0),
        };
        if !record.valid() || request_bytes > self.config.maximum_request_bytes {
            return self.skip();
        }
        // Node and identifier overhead is charged even for an empty context.
        let bytes = record.heap_bytes() + record.request.request_id.capacity() + 512;
        let Ok(mut pending) = self.pending.lock() else {
            return self.skip();
        };
        self.expire(&mut pending);
        if pending.closed
            || self.admissions.load(Ordering::Acquire) >= self.config.maximum_pending_records
            || self.retained_bytes(&pending).saturating_add(bytes)
                > self.config.maximum_pending_bytes
            || pending.entries.contains_key(&record.request.request_id)
        {
            return self.skip();
        }
        pending.bytes += bytes;
        self.admissions.fetch_add(1, Ordering::AcqRel);
        pending.entries.insert(
            record.request.request_id.clone(),
            Entry {
                _admission: Arc::new(Admission {
                    count: self.admissions.clone(),
                    handoff_bytes: self.handoff_bytes.clone(),
                    retained_bytes: AtomicUsize::new(0),
                }),
                record,
                observation: None,
                gemini_part_bytes: 0,
                wire: None,
                relay: None,
                relay_attached: false,
                relay_required: self.config.relay_metadata,
                request_bytes,
                expires: Instant::now() + Duration::from_secs(self.config.ttl_seconds),
                bytes,
                attached: false,
                output_finished: false,
                checkpointing: false,
                checkpoint_started: false,
                unexposed: false,
                response_allowed: !self.config.settlement_required,
                response_discarded: Arc::new(AtomicBool::new(false)),
            },
        );
        true
    }

    /// Freeze resolved provenance once; a pre-dispatch rejection has no selected model.
    pub(crate) fn select_model(&self, request_id: &str, model_id: &str) {
        if model_id.trim().is_empty() || model_id.len() > 512 {
            return;
        }
        if let Ok(mut pending) = self.pending.lock() {
            self.expire(&mut pending);
            let Some(mut entry) = pending.entries.remove(request_id) else {
                return;
            };
            pending.bytes -= entry.map_bytes();
            if entry.record.request.model_id.is_none() && !entry.attached {
                // Replace the original JSON null, including escapes in the selected id.
                entry.request_bytes = entry.request_bytes - 4 + string_bytes(model_id);
                let model = model_id.to_owned();
                entry.bytes += model.capacity();
                entry.record.request.model_id = Some(model);
            }
            if self.retained_bytes(&pending).saturating_add(
                entry
                    .bytes
                    .saturating_sub(entry._admission.retained_bytes.load(Ordering::Acquire)),
            ) > self.config.maximum_pending_bytes
                || entry.request_bytes > self.config.maximum_request_bytes
            {
                self.skip();
                return;
            }
            pending.bytes += entry.map_bytes();
            pending.entries.insert(request_id.to_owned(), entry);
        }
    }

    /// Exactly one original response can attach; keyed replays cannot capture twice.
    pub(crate) fn attach(&self, request_id: &str) -> Option<Arc<AtomicBool>> {
        let Ok(mut pending) = self.pending.lock() else {
            return None;
        };
        self.expire(&mut pending);
        let entry = pending.entries.get_mut(request_id)?;
        if entry.attached {
            return None;
        }
        entry.attached = true;
        Some(entry.response_discarded.clone())
    }

    /// Share only the selected attempt's accounting, never a losing rung's content or meter.
    pub(crate) fn observe_attempt(
        &self,
        request_id: &str,
        observation: crate::settlement::Observation,
    ) {
        if let Ok(mut pending) = self.pending.lock() {
            if let Some(entry) = pending.entries.get_mut(request_id) {
                entry.observation = Some(observation);
            }
        }
    }

    /// Capture a hosted prompt after the winning host-funded lane is frozen.
    /// The destination must recheck consent and merge this idempotent update
    /// without replacing a later response. Keep the shared request tree live
    /// until terminal settlement; no provider response belongs in this write.
    #[cfg(test)]
    pub(crate) fn checkpoint(&self, request_id: &str) -> bool {
        self.checkpoint_receipt(request_id).is_ok_and(|receipt| {
            receipt.is_none_or(|receipt| receipt.blocking_recv().unwrap_or(false))
        })
    }

    pub(crate) fn checkpoint_receipt(
        &self,
        request_id: &str,
    ) -> Result<Option<tokio::sync::oneshot::Receiver<bool>>, ()> {
        if !self.config.settlement_required {
            return Ok(None);
        }
        let mut pending = self.pending.lock().map_err(|_| ())?;
        let Some(entry) = pending.entries.get_mut(request_id) else {
            return Ok(None);
        };
        if entry.checkpoint_started {
            return Err(());
        }
        let record = Record {
            checkpointed: false,
            schema_version: SCHEMA_VERSION,
            request: entry.record.request.clone(),
            response: None,
            transport: None,
            provider_reasoning: None,
            provider_reasoning_source_json: None,
            provider_tool_calls_json: None,
            deployment_id: None,
            metrics: None,
            gemini_thought_parts: Vec::new(),
            gemini_thought_parts_source_json: None,
            captured_at: entry.record.captured_at,
        };
        // The context Arc is shared; charge the copied identifiers and receipt/queue nodes.
        let extra = record.request.request_id.capacity() * 2
            + record.request.scope.organization_id.capacity()
            + record.request.scope.identity_id.capacity()
            + record.request.scope.application_id.capacity()
            + record.request.model_id.as_ref().map_or(0, String::capacity)
            + 2 * std::mem::size_of::<Record>()
            + 2 * std::mem::size_of::<CheckpointLease>()
            + 1024;
        let map_bytes = entry.map_bytes();
        let shared = entry._admission.clone();
        let bytes = entry.bytes.saturating_add(extra);
        if self.retained_bytes(&pending).saturating_add(extra) > self.config.maximum_pending_bytes {
            return Err(());
        }
        let entry = pending.entries.get_mut(request_id).ok_or(())?;
        entry.bytes = bytes;
        shared.handoff(bytes);
        entry.checkpointing = true;
        entry.checkpoint_started = true;
        entry.record.checkpointed = true;
        let lease = CheckpointLease {
            admission: entry._admission.clone(),
            pending: self.pending.clone(),
            request_id: request_id.to_owned(),
            ttl: Duration::from_secs(self.config.ttl_seconds),
        };
        pending.bytes -= map_bytes;
        drop(pending);
        self.delivery
            .checkpoint(record, None, lease, self.config.asynchronous_delivery)
            .map(Some)
    }

    /// Terminal policy controls response retention; the winning lane was frozen
    /// before any hosted prompt checkpoint. The sink rechecks live consent.
    pub(crate) fn settle(&self, request_id: &str, keep_prompt: bool, keep_response: bool) {
        let Ok(mut pending) = self.pending.lock() else {
            return;
        };
        self.expire(&mut pending);
        let Some(mut entry) = pending.entries.remove(request_id) else {
            return;
        };
        pending.bytes -= entry.map_bytes();
        if !keep_prompt {
            entry.response_discarded.store(true, Ordering::Release);
            return;
        }
        if !keep_response {
            entry.response_discarded.store(true, Ordering::Release);
            entry.record.response = None;
            entry.wire = None;
            entry.record.provider_reasoning = None;
            entry.record.provider_tool_calls_json = None;
            entry.record.metrics = None;
            entry.record.gemini_thought_parts.clear();
            entry._admission.handoff(entry.bytes);
            drop(pending);
            self.emit_entry(entry);
            return;
        }
        entry.response_allowed = true;
        if !entry.output_finished || (entry.relay_required && entry.relay.is_none()) {
            // The terminal update supplies output to the earlier prompt checkpoint.
            pending.bytes += entry.map_bytes();
            pending.entries.insert(request_id.to_owned(), entry);
        } else {
            entry._admission.handoff(entry.bytes);
            drop(pending);
            self.emit_entry(entry);
        }
    }

    /// Output may arrive before or after settlement; unpermitted output stays bounded.
    #[cfg(test)]
    pub(crate) fn finish(
        &self,
        request_id: &str,
        response: Option<Response>,
        deployment_id: Option<String>,
    ) -> bool {
        self.finish_output(request_id, response, None, deployment_id)
    }

    /// A request ending before public output has no response tap to finish its capture entry.
    pub(crate) fn finish_unexposed(&self, request_id: &str) {
        if let Ok(mut pending) = self.pending.lock() {
            if let Some(entry) = pending.entries.get_mut(request_id) {
                entry.unexposed = true;
            }
        }
        self.finish_output(request_id, None, None, None);
    }

    /// Transfer bounded wire bytes; only the destination worker builds JSON trees.
    pub(super) fn finish_wire(
        &self,
        request_id: &str,
        wire: WireResponse,
        deployment_id: Option<String>,
    ) -> bool {
        self.finish_output(request_id, None, Some(wire), deployment_id)
    }

    fn finish_output(
        &self,
        request_id: &str,
        response: Option<Response>,
        wire: Option<WireResponse>,
        deployment_id: Option<String>,
    ) -> bool {
        let response_bytes = response.as_ref().map_or(0, Response::json_bytes);
        let response_heap = response.as_ref().map_or(0, Response::heap_bytes)
            + wire.as_ref().map_or(0, WireResponse::heap_bytes)
            + deployment_id.as_ref().map_or(0, String::capacity);
        let Ok(mut pending) = self.pending.lock() else {
            return false;
        };
        self.expire(&mut pending);
        let Some(mut entry) = pending.entries.remove(request_id) else {
            return false;
        };
        pending.bytes -= entry.map_bytes();
        if response_bytes > self.config.maximum_response_bytes {
            return self.skip();
        }
        entry.record.response = response;
        entry.wire = wire;
        if entry.record.response.is_none() && entry.wire.is_none() {
            entry.record.provider_reasoning = None;
            entry.record.provider_tool_calls_json = None;
            entry.record.gemini_thought_parts.clear();
        }
        entry.record.deployment_id = deployment_id;
        entry.record.metrics = if entry.record.response.is_some() || entry.wire.is_some() {
            entry
                .observation
                .as_ref()
                .map(super::metrics::Metrics::observed)
        } else {
            None
        };
        entry.output_finished = true;
        entry.bytes = entry.bytes.saturating_add(response_heap);
        if entry.response_allowed && (!entry.relay_required || entry.relay.is_some()) {
            entry._admission.handoff(entry.bytes);
            drop(pending);
            self.emit_entry(entry)
        } else if self.retained_bytes(&pending).saturating_add(
            entry
                .bytes
                .saturating_sub(entry._admission.retained_bytes.load(Ordering::Acquire)),
        ) <= self.config.maximum_pending_bytes
        {
            pending.bytes += entry.map_bytes();
            pending.entries.insert(request_id.to_owned(), entry);
            true
        } else {
            self.skip()
        }
    }

    /// Only one front relay may supply metadata for the original captured response.
    pub(crate) fn claim_relay(&self, request_id: &str) -> bool {
        if !self.config.relay_metadata {
            return false;
        }
        let Ok(mut pending) = self.pending.lock() else {
            return false;
        };
        let Some(entry) = pending.entries.get_mut(request_id) else {
            return false;
        };
        if entry.relay_attached {
            return false;
        }
        entry.relay_attached = true;
        true
    }

    /// Error responses without public correlation cannot be claimed by a front.
    pub(crate) fn without_relay(&self, request_id: &str) {
        if let Ok(mut pending) = self.pending.lock() {
            if let Some(entry) = pending.entries.get_mut(request_id) {
                entry.relay_required = false;
            }
        }
    }

    pub(super) fn finish_relay(&self, request_id: &str, relay: super::relay::Relay) -> bool {
        let Ok(mut pending) = self.pending.lock() else {
            return false;
        };
        let Some(mut entry) = pending.entries.remove(request_id) else {
            return false;
        };
        pending.bytes -= entry.map_bytes();
        if !entry.relay_attached || entry.relay.is_some() {
            pending.bytes += entry.map_bytes();
            pending.entries.insert(request_id.to_owned(), entry);
            return false;
        }
        entry.bytes += relay.heap_bytes();
        entry.relay = Some(relay);
        if entry.output_finished && entry.response_allowed {
            entry._admission.handoff(entry.bytes);
            drop(pending);
            self.emit_entry(entry)
        } else if self.retained_bytes(&pending).saturating_add(
            entry
                .bytes
                .saturating_sub(entry._admission.retained_bytes.load(Ordering::Acquire)),
        ) <= self.config.maximum_pending_bytes
        {
            pending.bytes += entry.map_bytes();
            pending.entries.insert(request_id.to_owned(), entry);
            true
        } else {
            self.skip()
        }
    }

    fn emit_entry(&self, mut entry: Entry) -> bool {
        if let Some(wire) = entry.wire.as_mut() {
            wire.relay = entry.relay.take();
        }
        // Admission already owns these bytes. Async handoff must not wait for
        // storage capacity; unexposed failures retain the same ownership rule.
        if self.config.asynchronous_delivery || entry.checkpointing || entry.unexposed {
            let lease = CheckpointLease {
                admission: entry._admission,
                pending: self.pending.clone(),
                request_id: entry.record.request.request_id.clone(),
                ttl: Duration::from_secs(self.config.ttl_seconds),
            };
            self.delivery
                .checkpoint(entry.record, entry.wire, lease, true)
                .is_ok()
        } else {
            self.emit(entry.record, entry.wire, Some(entry._admission))
        }
    }

    fn emit(
        &self,
        record: Record,
        wire: Option<WireResponse>,
        admission: Option<Arc<Admission>>,
    ) -> bool {
        let deliver = || self.delivery.submit_wait(record, wire, admission);
        // A blocked destination must not occupy a Tokio executor thread or a
        // collector lock. Python entrypoints already release the interpreter.
        if tokio::runtime::Handle::try_current().is_ok_and(|runtime| {
            runtime.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread
        }) {
            tokio::task::block_in_place(deliver)
        } else {
            deliver()
        }
    }

    /// Reserve one complete bounded response before consuming any provider bytes.
    pub(crate) async fn body_permit(&self) -> Option<OwnedSemaphorePermit> {
        self.body_capacity
            .clone()
            .acquire_many_owned(self.config.maximum_response_bytes as u32)
            .await
            .ok()
    }

    /// Append only authorized winning-rung reasoning, charging allocated capacity.
    /// Overflow excludes the exchange instead of silently publishing partial evidence.
    pub(crate) fn reasoning(&self, request_id: &str, delta: &str) {
        if delta.is_empty() {
            return;
        }
        let Ok(mut pending) = self.pending.lock() else {
            return;
        };
        self.expire(&mut pending);
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
        if required > self.config.maximum_response_bytes
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
        let Some(mut entry) = pending.entries.remove(request_id) else {
            return;
        };
        pending.bytes -= entry.map_bytes();
        let previous = entry.record.gemini_thought_parts.capacity()
            * std::mem::size_of::<Arc<serde_json::Value>>();
        let heap = super::budget::heap_bytes(&part) + 64;
        let part_bytes = super::budget::json_bytes(&part);
        if entry.gemini_part_bytes.saturating_add(part_bytes) > self.config.maximum_response_bytes
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

    fn expire(&self, pending: &mut Pending) {
        expire_pending(pending, &self.skipped);
    }

    /// Include values waiting for destination capacity or acknowledgement.
    fn retained_bytes(&self, pending: &Pending) -> usize {
        pending
            .bytes
            .saturating_add(self.handoff_bytes.load(Ordering::Acquire))
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
        if text.len().saturating_add(extra) > self.config.maximum_response_bytes
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

    pub(crate) fn skip(&self) -> bool {
        self.skipped.fetch_add(1, Ordering::Relaxed);
        false
    }

    /// Reserve actual retained byte capacity across all concurrent response taps.
    pub(crate) fn reserve_body(&self, bytes: usize) -> bool {
        self.body_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |held| {
                held.checked_add(bytes)
                    .filter(|total| *total <= self.config.maximum_pending_bytes)
            })
            .is_ok()
    }

    pub(crate) fn release_body(&self, bytes: usize) {
        self.body_bytes.fetch_sub(bytes, Ordering::AcqRel);
    }

    /// Stop admissions but preserve accepted work until settlement and persistence finish.
    pub(crate) fn close_until(&self, deadline: Instant) -> bool {
        let Ok(mut pending) = self.pending.lock() else {
            return false;
        };
        pending.closed = true;
        drop(pending);
        // An entry owns its admission through emit(), even after removal from
        // the pending map. Closing delivery earlier could reject that handoff.
        while self.admissions.load(Ordering::Acquire) > 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        if self.admissions.load(Ordering::Acquire) > 0 {
            return false;
        }
        self.delivery.close_until(deadline)
    }

    pub(crate) fn counts(&self) -> [u64; 6] {
        let [pending, bytes, persisted, failed, dropped] = self.delivery.counts();
        [
            pending,
            bytes,
            persisted,
            failed,
            dropped,
            self.skipped.load(Ordering::Relaxed),
        ]
    }

    pub(crate) fn maintenance_failures(&self) -> u64 {
        self.delivery.maintenance_failures()
    }
}

#[cfg(test)]
#[path = "collector_test.rs"]
mod tests;
