//! Content-free process counters updated beside the native decisions they describe.

use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
pub(super) struct Counters {
    pub replayed: AtomicU64,
    pub abandoned: AtomicU64,
    pub sheds: AtomicU64,
    pub overflows: AtomicU64,
    pub refusals: AtomicU64,
    pub rates: AtomicU64,
    pub fresh: AtomicU64,
    pub surfaced: AtomicU64,
    pub cold: AtomicU64,
    pub backoff: AtomicU64,
    pub forced_backoff: AtomicU64,
}

pub(super) fn increment(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

impl Counters {
    pub fn snapshot(&self) -> Value {
        let n = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        json!({"replayed": n(&self.replayed), "abandoned": n(&self.abandoned),
            "sheds": n(&self.sheds), "overflows": n(&self.overflows), "refusals": n(&self.refusals),
            "rates": n(&self.rates), "fresh": n(&self.fresh), "surfaced": n(&self.surfaced),
            "cold": n(&self.cold), "backoff": n(&self.backoff), "forced_backoff": n(&self.forced_backoff)})
    }
}
