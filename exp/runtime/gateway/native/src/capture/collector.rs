//! Bounded rendezvous of authenticated input, terminal policy and observed output.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Deserialize;

use super::delivery::{Delivery, Limits, Sink};
use super::record::{Record, Request, Response, SCHEMA_VERSION};

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
}

impl Configuration {
    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        self.delivery.validate()?;
        if !(1..=4096).contains(&self.maximum_pending_records)
            || !(1..=1024 * 1024).contains(&self.maximum_request_bytes)
            || !(1..=4 * 1024 * 1024).contains(&self.maximum_response_bytes)
            || self.maximum_pending_bytes < self.maximum_request_bytes
            || self.maximum_pending_bytes > 256 * 1024 * 1024
            || !(1..=3600).contains(&self.ttl_seconds)
        {
            return Err("invalid capture collector bounds");
        }
        Ok(())
    }
}

struct Entry {
    record: Record,
    expires: Instant,
    bytes: usize,
    attached: bool,
    output_finished: bool,
    response_allowed: bool,
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
    fn write(&mut self, record: &str) -> Result<(), ()> {
        self.sink.write(record)
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
        if entry.expires <= now {
            pending.bytes -= entry.bytes;
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
    body_bytes: AtomicUsize,
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
            config,
            pending,
            skipped,
            body_bytes: AtomicUsize::new(0),
        })
    }

    /// Admission is the sole authority for input. Duplicate ids never replace a record.
    pub(crate) fn begin(&self, request: Request) -> bool {
        let record = Record {
            schema_version: SCHEMA_VERSION,
            request,
            response: None,
            deployment_id: None,
            captured_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_secs_f64())
                .unwrap_or(0.0),
        };
        let Some(encoded) = record.encode(self.config.maximum_request_bytes) else {
            return self.skip();
        };
        // Node and identifier overhead is charged even for an empty context.
        let bytes = encoded.len() + 512;
        let Ok(mut pending) = self.pending.lock() else {
            return self.skip();
        };
        self.expire(&mut pending);
        if pending.closed
            || pending.entries.len() >= self.config.maximum_pending_records
            || pending.bytes.saturating_add(bytes) > self.config.maximum_pending_bytes
            || pending.entries.contains_key(&record.request.request_id)
        {
            return self.skip();
        }
        pending.bytes += bytes;
        pending.entries.insert(
            record.request.request_id.clone(),
            Entry {
                record,
                expires: Instant::now() + Duration::from_secs(self.config.ttl_seconds),
                bytes,
                attached: false,
                output_finished: false,
                response_allowed: !self.config.settlement_required,
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
            pending.bytes -= entry.bytes;
            if entry.record.request.model_id.is_none() && !entry.attached {
                entry.record.request.model_id = Some(model_id.to_owned());
                entry.bytes += model_id.len() + 2;
            }
            if pending.bytes.saturating_add(entry.bytes) > self.config.maximum_pending_bytes
                || entry
                    .record
                    .encode(self.config.maximum_request_bytes)
                    .is_none()
            {
                self.skip();
                return;
            }
            pending.bytes += entry.bytes;
            pending.entries.insert(request_id.to_owned(), entry);
        }
    }

    /// Exactly one original response can attach; keyed replays cannot capture twice.
    pub(crate) fn attach(&self, request_id: &str) -> bool {
        let Ok(mut pending) = self.pending.lock() else {
            return false;
        };
        self.expire(&mut pending);
        let Some(entry) = pending.entries.get_mut(request_id) else {
            return false;
        };
        if entry.attached {
            return false;
        }
        entry.attached = true;
        true
    }

    /// Hosted terminal eligibility precedes durable content, so a dropped BYOK
    /// discard can never leave a previously queued prompt behind.
    pub(crate) fn settle(&self, request_id: &str, keep_prompt: bool, keep_response: bool) {
        let Ok(mut pending) = self.pending.lock() else {
            return;
        };
        self.expire(&mut pending);
        let Some(mut entry) = pending.entries.remove(request_id) else {
            return;
        };
        pending.bytes -= entry.bytes;
        if !keep_prompt {
            return;
        }
        if !keep_response {
            entry.record.response = None;
            self.emit(&entry.record);
            return;
        }
        entry.response_allowed = true;
        self.emit(&entry.record);
        if !entry.output_finished {
            pending.bytes += entry.bytes;
            pending.entries.insert(request_id.to_owned(), entry);
        }
    }

    /// Output may arrive before or after settlement; unpermitted output stays bounded.
    pub(crate) fn finish(
        &self,
        request_id: &str,
        response: Option<Response>,
        deployment_id: Option<String>,
    ) {
        let response_bytes = response
            .as_ref()
            .and_then(|value| serde_json::to_string(value).ok())
            .map_or(0, |encoded| encoded.len());
        let Ok(mut pending) = self.pending.lock() else {
            return;
        };
        self.expire(&mut pending);
        let Some(mut entry) = pending.entries.remove(request_id) else {
            return;
        };
        pending.bytes -= entry.bytes;
        if response_bytes > self.config.maximum_response_bytes {
            self.skip();
            return;
        }
        entry.record.response = response;
        entry.record.deployment_id = deployment_id;
        entry.output_finished = true;
        entry.bytes = entry.bytes.saturating_add(response_bytes);
        if entry.response_allowed {
            self.emit(&entry.record);
        } else if pending.bytes.saturating_add(entry.bytes) <= self.config.maximum_pending_bytes {
            pending.bytes += entry.bytes;
            pending.entries.insert(request_id.to_owned(), entry);
        } else {
            self.skip();
        }
    }

    fn emit(&self, record: &Record) {
        if let Some(encoded) = record.encode(self.config.delivery.maximum_record_bytes) {
            self.delivery.submit(encoded);
        } else {
            self.skip();
        }
    }

    fn expire(&self, pending: &mut Pending) {
        expire_pending(pending, &self.skipped);
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

    pub(crate) fn close_until(&self, deadline: Instant) -> bool {
        if let Ok(mut pending) = self.pending.lock() {
            pending.closed = true;
            self.skipped
                .fetch_add(pending.entries.len() as u64, Ordering::Relaxed);
            pending.entries.clear();
            pending.bytes = 0;
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
}

#[cfg(test)]
#[path = "collector_test.rs"]
mod tests;
