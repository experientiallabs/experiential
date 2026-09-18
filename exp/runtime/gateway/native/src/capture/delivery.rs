//! Count- and byte-bounded delivery, isolated from request and bridge executors.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::Deserialize;

/// Local SQLite and hosted persistence implement the same off-path destination.
pub(crate) trait Sink: Send + 'static {
    /// Persist one versioned record. Errors are deliberately content-free.
    fn write(&mut self, record: &str) -> Result<(), ()>;

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
            || !(1..=8 * 1024 * 1024).contains(&self.maximum_record_bytes)
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
    pending: AtomicUsize,
    dropped: AtomicU64,
    persisted: AtomicU64,
    failed: AtomicU64,
}

struct Pending {
    value: String,
    counters: Arc<Counters>,
}

impl Drop for Pending {
    fn drop(&mut self) {
        self.counters
            .bytes
            .fetch_sub(self.value.capacity(), Ordering::AcqRel);
        self.counters.pending.fetch_sub(1, Ordering::AcqRel);
    }
}

pub(crate) struct Delivery {
    limits: Limits,
    sender: Mutex<Option<mpsc::SyncSender<Pending>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
    deadline: Arc<Mutex<Option<Instant>>>,
    counters: Arc<Counters>,
}

impl Delivery {
    pub(crate) fn new(limits: Limits, mut sink: impl Sink) -> Result<Self, &'static str> {
        limits.validate()?;
        let (sender, receiver) = mpsc::sync_channel::<Pending>(limits.maximum_records);
        let counters = Arc::new(Counters::default());
        let worker_counters = counters.clone();
        let deadline = Arc::new(Mutex::new(None::<Instant>));
        let worker_deadline = deadline.clone();
        let worker = std::thread::Builder::new()
            .name("exp-capture".into())
            .spawn(move || {
                let mut maintained = Instant::now();
                loop {
                    if worker_deadline
                        .lock()
                        .map_or(true, |bound| bound.is_some_and(|at| Instant::now() >= at))
                    {
                        for _ in receiver.try_iter() {
                            worker_counters.dropped.fetch_add(1, Ordering::Relaxed);
                        }
                        break;
                    }
                    if maintained.elapsed() >= Duration::from_secs(1) {
                        if sink.maintain().is_err() {
                            worker_counters.failed.fetch_add(1, Ordering::Relaxed);
                        }
                        maintained = Instant::now();
                    }
                    match receiver.recv_timeout(Duration::from_millis(100)) {
                        Ok(item) => {
                            let counter = if sink.write(&item.value).is_ok() {
                                &worker_counters.persisted
                            } else {
                                &worker_counters.failed
                            };
                            counter.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }
            })
            .map_err(|_| "cannot start capture delivery worker")?;
        Ok(Self {
            limits,
            sender: Mutex::new(Some(sender)),
            worker: Mutex::new(Some(worker)),
            deadline,
            counters,
        })
    }

    /// Drop on saturation. The budget includes the record a slow sink is writing.
    pub(crate) fn submit(&self, value: String) -> bool {
        if value.len() > self.limits.maximum_record_bytes {
            return self.dropped();
        }
        let bytes = value.capacity();
        if self
            .counters
            .bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |held| {
                held.checked_add(bytes)
                    .filter(|total| *total <= self.limits.maximum_bytes)
            })
            .is_err()
        {
            return self.dropped();
        }
        let previous = self.counters.pending.fetch_add(1, Ordering::AcqRel);
        let item = Pending {
            value,
            counters: self.counters.clone(),
        };
        if previous >= self.limits.maximum_records {
            return self.dropped();
        }
        let Ok(sender) = self.sender.lock() else {
            return self.dropped();
        };
        if sender
            .as_ref()
            .is_none_or(|sender| sender.try_send(item).is_err())
        {
            return self.dropped();
        }
        true
    }

    fn dropped(&self) -> bool {
        self.counters.dropped.fetch_add(1, Ordering::Relaxed);
        false
    }

    /// Stop accepting records and drain within a fixed budget, including a stuck sink.
    pub(crate) fn close_until(&self, until: Instant) -> bool {
        if let Ok(mut deadline) = self.deadline.lock() {
            *deadline = Some(deadline.map_or(until, |old| old.min(until)));
        }
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
            self.counters.bytes.load(Ordering::Acquire) as u64,
            self.counters.persisted.load(Ordering::Relaxed),
            self.counters.failed.load(Ordering::Relaxed),
            self.counters.dropped.load(Ordering::Relaxed),
        ]
    }
}

#[cfg(test)]
#[path = "delivery_test.rs"]
mod tests;
