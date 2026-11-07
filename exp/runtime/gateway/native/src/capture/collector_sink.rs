//! Forward destination capabilities while maintaining collector retention.

use super::{expire_pending, Pending, Record, Sink};
use std::sync::{atomic::AtomicU64, Arc, Mutex};
use std::time::Duration;

pub(super) struct MaintainedSink<S> {
    pub(super) sink: S,
    pub(super) pending: Arc<Mutex<Pending>>,
    pub(super) skipped: Arc<AtomicU64>,
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

    fn split_raw_above(&self) -> Option<usize> {
        self.sink.split_raw_above()
    }

    fn independent_retries(&self) -> bool {
        self.sink.independent_retries()
    }

    fn retained_prepared_bytes(&self, prepared: &Self::Prepared, maximum: usize) -> usize {
        self.sink.retained_prepared_bytes(prepared, maximum)
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
