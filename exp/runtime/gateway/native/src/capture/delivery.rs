//! Count- and byte-bounded delivery, isolated from request and bridge executors.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::Deserialize;

use super::collector::Admission;
use super::record::Record;
use super::response::WireResponse;

/// Local SQLite and hosted persistence implement the same off-path destination.
pub(crate) trait Sink: Send + 'static {
    type Prepared;

    /// Maximum retained preparation allocation, reserved before queue admission.
    fn preparation_bytes(maximum_record_bytes: usize) -> usize;

    /// Prepare once, off serving; retrying storage must not re-encode the record.
    fn prepare(&self, record: &Record, maximum_bytes: usize) -> Result<Self::Prepared, ()>;

    /// Acknowledge an idempotent write or intentional policy exclusion. An error
    /// retains the payload for retry; error details must never include content.
    fn write(&mut self, prepared: &Self::Prepared) -> Result<(), ()>;

    /// Destinations opt into batching without changing single-record writers.
    fn batch_records(&self) -> usize {
        1
    }

    /// Stop gathering after this many prepared bytes. One final record may cross it.
    fn batch_bytes(&self) -> usize {
        0
    }

    /// Bound the oldest item's wait for a useful group without delaying serving.
    fn batch_delay(&self) -> Duration {
        Duration::ZERO
    }

    fn prepared_bytes(&self, _prepared: &Self::Prepared) -> usize {
        0
    }

    /// One acknowledgement per record, in input order. False retains ownership.
    fn write_batch(&mut self, prepared: &[&Self::Prepared]) -> Vec<bool> {
        prepared
            .iter()
            .map(|value| self.write(value).is_ok())
            .collect()
    }

    /// Drain cleanup failures discovered after a successful durable write.
    fn take_maintenance_failures(&mut self) -> u64 {
        0
    }

    /// Run retention maintenance without adding storage work to serving.
    fn maintain(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Limits {
    pub maximum_records: usize,
    pub maximum_bytes: usize,
    pub maximum_record_bytes: usize,
}

impl Limits {
    pub(crate) fn validate(&self) -> Result<(), &'static str> {
        if !(1..=4096).contains(&self.maximum_records)
            || !(1..=16 * 1024 * 1024).contains(&self.maximum_record_bytes)
            || self.maximum_bytes < self.maximum_record_bytes
            || self.maximum_bytes > 256 * 1024 * 1024
        {
            return Err("invalid capture delivery bounds");
        }
        Ok(())
    }
}

#[derive(Default)]
struct Counters {
    bytes: AtomicUsize,
    preparation_bytes: AtomicUsize,
    pending: AtomicUsize,
    dropped: AtomicU64,
    persisted: AtomicU64,
    failed: AtomicU64,
    maintenance_failed: AtomicU64,
    capacity: Mutex<()>,
    available: Condvar,
}

struct Pending {
    value: Option<Record>,
    wire: Option<WireResponse>,
    bytes: usize,
    counters: Arc<Counters>,
    completed: Option<mpsc::SyncSender<bool>>,
    // Keep collector admission charged until the destination acknowledges.
    _admission: Option<Admission>,
}

struct Prepared<P> {
    item: Pending,
    value: Option<P>,
}

/// Gather until a count, byte or oldest-item deadline is reached.
/// Failed members keep their slot while acknowledged neighbors release theirs.
fn run_worker<S: Sink>(
    receiver: mpsc::Receiver<Pending>,
    mut sink: S,
    counters: Arc<Counters>,
    maximum_record_bytes: usize,
    preparation_bytes: usize,
) {
    let mut pending: Vec<Prepared<S::Prepared>> = Vec::new();
    let mut maintained = Instant::now();
    let mut delay = Duration::from_millis(25);
    loop {
        if maintained.elapsed() >= Duration::from_secs(1) {
            if sink.maintain().is_err() {
                counters.maintenance_failed.fetch_add(1, Ordering::Relaxed);
            }
            maintained = Instant::now();
        }
        if pending.is_empty() {
            match receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(item) => pending.push(Prepared { item, value: None }),
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        counters
            .preparation_bytes
            .store(preparation_bytes, Ordering::Release);
        let batch_deadline = Instant::now() + sink.batch_delay();
        let mut bytes = pending
            .iter()
            .filter_map(|p| p.value.as_ref())
            .map(|value| sink.prepared_bytes(value))
            .sum::<usize>();
        let mut index = 0;
        'prepare_batch: loop {
            while index < pending.len() {
                let entry = &mut pending[index];
                if entry.value.is_none() && (bytes == 0 || bytes < sink.batch_bytes()) {
                    let record = entry
                        .item
                        .value
                        .as_mut()
                        .expect("unprepared record retained");
                    if let Some(mut wire) = entry.item.wire.take() {
                        record.transport = wire.relay.take().map(super::relay::Relay::decode);
                        record.response = wire.decode();
                        if record.response.is_none() {
                            record.provider_reasoning = None;
                            record.provider_tool_calls_json = None;
                        }
                    }
                    entry.value = sink.prepare(record, maximum_record_bytes).ok();
                    if let Some(value) = &entry.value {
                        bytes += sink.prepared_bytes(value);
                        // The prepared payload owns all retry evidence now. Free
                        // the decoded tree before preparing the next batch member,
                        // but keep its admission charge until durable acknowledgement.
                        entry.item.value = None;
                    } else {
                        counters.failed.fetch_add(1, Ordering::Relaxed);
                        // A failed preparation still owns the one decoded
                        // workspace. Leave later records compact and charged
                        // until it recovers; prepared neighbors can still commit.
                        break 'prepare_batch;
                    }
                }
                index += 1;
            }
            if pending.len() >= sink.batch_records() || bytes >= sink.batch_bytes() {
                break;
            }
            match receiver.recv_timeout(batch_deadline.saturating_duration_since(Instant::now())) {
                Ok(item) => pending.push(Prepared { item, value: None }),
                Err(_) => break,
            }
        }
        let ready: Vec<_> = pending.iter().filter_map(|p| p.value.as_ref()).collect();
        let outcomes = if ready.is_empty() {
            Vec::new()
        } else {
            sink.write_batch(&ready)
        };
        let valid = outcomes.len() == ready.len();
        drop(ready);
        let mut outcome = outcomes.into_iter();
        let mut acknowledged = 0;
        pending.retain(|entry| {
            let persisted = entry.value.is_some() && valid && outcome.next().unwrap_or(false);
            if persisted {
                counters.persisted.fetch_add(1, Ordering::Relaxed);
                acknowledged += 1;
                if let Some(completed) = &entry.item.completed {
                    let _ = completed.send(true);
                }
            } else if entry.value.is_some() {
                counters.failed.fetch_add(1, Ordering::Relaxed);
            }
            !persisted
        });
        counters
            .maintenance_failed
            .fetch_add(sink.take_maintenance_failures(), Ordering::Relaxed);
        if pending.is_empty() {
            counters.preparation_bytes.store(0, Ordering::Release);
            delay = Duration::from_millis(25);
        } else {
            // Prepared strings and admission charges remain live across retries.
            // Keep the preparation charge visible while sleeping too.
            std::thread::sleep(delay);
            delay = if acknowledged > 0 {
                Duration::from_millis(25)
            } else {
                (delay * 2).min(Duration::from_secs(1))
            };
        }
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        let _capacity = self
            .counters
            .capacity
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        self.counters.bytes.fetch_sub(self.bytes, Ordering::AcqRel);
        self.counters.pending.fetch_sub(1, Ordering::AcqRel);
        self.counters.available.notify_all();
    }
}

pub(crate) struct Delivery {
    limits: Limits,
    maximum_queued_bytes: usize,
    sender: Mutex<Option<mpsc::SyncSender<Pending>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
    counters: Arc<Counters>,
}

impl Delivery {
    pub(crate) fn new<S: Sink>(limits: Limits, sink: S) -> Result<Self, &'static str> {
        limits.validate()?;
        let preparation_bytes = S::preparation_bytes(limits.maximum_record_bytes);
        let maximum_queued_bytes = limits
            .maximum_bytes
            .checked_sub(preparation_bytes)
            .filter(|available| *available >= limits.maximum_record_bytes)
            .ok_or("capture byte budget must fit destination preparation and one record")?;
        let (sender, receiver) = mpsc::sync_channel::<Pending>(limits.maximum_records);
        let counters = Arc::new(Counters::default());
        let worker_counters = counters.clone();
        let maximum_record_bytes = limits.maximum_record_bytes;
        let worker = std::thread::Builder::new()
            .name("exp-capture".into())
            .spawn(move || {
                run_worker(
                    receiver,
                    sink,
                    worker_counters,
                    maximum_record_bytes,
                    preparation_bytes,
                )
            })
            .map_err(|_| "cannot start capture delivery worker")?;
        Ok(Self {
            limits,
            maximum_queued_bytes,
            sender: Mutex::new(Some(sender)),
            worker: Mutex::new(Some(worker)),
            counters,
        })
    }

    /// Wait for capacity; accepted records are never discarded to make room.
    #[cfg(test)]
    pub(crate) fn submit(&self, value: Record) -> bool {
        self.enqueue(value, None, None, None)
    }

    /// Transfer ownership to the background writer, not to the database caller.
    pub(super) fn submit_record(
        &self,
        value: Record,
        wire: Option<WireResponse>,
        admission: Option<Admission>,
    ) -> bool {
        self.enqueue(value, wire, admission, None)
    }

    /// Preserve the default collector contract: success means the sink acknowledged.
    pub(super) fn submit_wait(
        &self,
        value: Record,
        wire: Option<WireResponse>,
        admission: Option<Admission>,
    ) -> bool {
        let (completed, outcome) = mpsc::sync_channel(1);
        self.enqueue(value, wire, admission, Some(completed)) && outcome.recv().unwrap_or(false)
    }

    fn enqueue(
        &self,
        value: Record,
        wire: Option<WireResponse>,
        admission: Option<Admission>,
        completed: Option<mpsc::SyncSender<bool>>,
    ) -> bool {
        let bytes = value.heap_bytes() + wire.as_ref().map_or(0, WireResponse::heap_bytes);
        if bytes > self.maximum_queued_bytes {
            return self.dropped();
        }
        // Clone before waiting. Shutdown closes new admissions, while producers
        // already waiting retain their right to deliver and keep the worker alive.
        let Some(sender) = self.sender.lock().ok().and_then(|sender| sender.clone()) else {
            return self.dropped();
        };
        let mut capacity = self
            .counters
            .capacity
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        while self.counters.pending.load(Ordering::Acquire) >= self.limits.maximum_records
            || self
                .counters
                .bytes
                .load(Ordering::Acquire)
                .saturating_add(bytes)
                > self.maximum_queued_bytes
        {
            capacity = self
                .counters
                .available
                .wait(capacity)
                .unwrap_or_else(|e| e.into_inner());
        }
        self.counters.bytes.fetch_add(bytes, Ordering::AcqRel);
        self.counters.pending.fetch_add(1, Ordering::AcqRel);
        drop(capacity);
        let item = Pending {
            value: Some(value),
            wire,
            bytes,
            counters: self.counters.clone(),
            completed,
            _admission: admission,
        };
        if sender.send(item).is_err() {
            return self.dropped();
        }
        true
    }

    fn dropped(&self) -> bool {
        self.counters.dropped.fetch_add(1, Ordering::Relaxed);
        false
    }

    /// Stop new submissions; a timeout reports an unfinished drain without purging it.
    pub(crate) fn close_until(&self, until: Instant) -> bool {
        if let Ok(mut sender) = self.sender.lock() {
            sender.take();
        }
        let Ok(mut worker) = self.worker.lock() else {
            return false;
        };
        let Some(handle) = worker.as_ref() else {
            return self.counters.pending.load(Ordering::Acquire) == 0;
        };
        while !handle.is_finished() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(1));
        }
        if !handle.is_finished() {
            return false;
        }
        worker.take().is_some_and(|handle| handle.join().is_ok())
    }

    pub(crate) fn counts(&self) -> [u64; 5] {
        [
            self.counters.pending.load(Ordering::Acquire) as u64,
            self.counters.bytes.load(Ordering::Acquire) as u64
                + self.counters.preparation_bytes.load(Ordering::Acquire) as u64,
            self.counters.persisted.load(Ordering::Relaxed),
            self.counters.failed.load(Ordering::Relaxed),
            self.counters.dropped.load(Ordering::Relaxed),
        ]
    }

    pub(crate) fn maintenance_failures(&self) -> u64 {
        self.counters.maintenance_failed.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
#[path = "delivery_test.rs"]
mod tests;
