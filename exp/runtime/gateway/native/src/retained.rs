//! One worker-wide budget on provider output the gateway retains across streams.
//!
//! Each stream may retain up to `MAXIMUM_RETAINED_OUTPUT_BYTES` (64 MiB) of
//! provider output: tool arguments and reasoning summaries in the normalizer,
//! Messages blocks held until they can be emitted in order, a non-streaming
//! aggregation, a Responses continuation. Concurrency multiplies that bound
//! into worker memory with nothing to stop it: about 60 such streams exceed a
//! 4 GiB worker, and a caller does not even need a large model to reach it,
//! because a customer-run or BYOK upstream decides how much it streams.
//!
//! Every retaining site therefore also reserves its bytes from this budget
//! through a [`RetainedLease`]. A reservation that would cross the budget is
//! refused and that site takes its existing overflow path (the stream fails
//! with a retryable unavailable error, or a continuation is simply not
//! retained), so retention is never the reason a worker runs out of memory.
//! A lease returns everything it holds when dropped.
//!
//! The budget is OFF by default (unbounded, today's behavior): an embedder
//! opts in by passing `maximum_retained_output_bytes`. Production enables it
//! with a later env flip and a watch, never in the release that ships it.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Default worker-wide retention: unbounded (the budget is off until configured).
pub const DEFAULT_WORKER_RETAINED_OUTPUT_BYTES: usize = usize::MAX;

/// The caller-facing message when the worker-wide budget refuses a stream.
pub const WORKER_RETENTION_EXHAUSTED_MESSAGE: &str =
    "the gateway is retaining too much provider output right now; retry the request";

pub(crate) struct RetainedOutputBudget {
    held: AtomicUsize,
    limit: AtomicUsize,
}

impl RetainedOutputBudget {
    pub(crate) const fn new(limit: usize) -> Self {
        Self {
            held: AtomicUsize::new(0),
            limit: AtomicUsize::new(limit),
        }
    }

    // `fetch_update` is renamed `try_update` in newer toolchains; keep the stable
    // name so every supported toolchain builds with warnings denied.
    #[allow(deprecated)]
    fn reserve(&self, bytes: usize) -> bool {
        let limit = self.limit.load(Ordering::Acquire);
        self.held
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |held| {
                held.checked_add(bytes).filter(|total| *total <= limit)
            })
            .is_ok()
    }

    fn release(&self, bytes: usize) {
        self.held.fetch_sub(bytes, Ordering::AcqRel);
    }

    pub(crate) fn held(&self) -> usize {
        self.held.load(Ordering::Acquire)
    }
}

static WORKER_BUDGET: RetainedOutputBudget =
    RetainedOutputBudget::new(DEFAULT_WORKER_RETAINED_OUTPUT_BYTES);

/// Set the worker-wide budget once at serve start.
pub(crate) fn configure(limit: usize) {
    WORKER_BUDGET.limit.store(limit, Ordering::Release);
}

#[cfg(test)]
thread_local! {
    /// Tests reserve from their own budget so they cannot starve each other.
    static TEST_BUDGET: std::cell::Cell<Option<&'static RetainedOutputBudget>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_test_budget<T>(
    budget: &'static RetainedOutputBudget,
    run: impl FnOnce() -> T,
) -> T {
    TEST_BUDGET.with(|cell| cell.set(Some(budget)));
    let result = run();
    TEST_BUDGET.with(|cell| cell.set(None));
    result
}

fn current_budget() -> &'static RetainedOutputBudget {
    #[cfg(test)]
    if let Some(budget) = TEST_BUDGET.with(std::cell::Cell::get) {
        return budget;
    }
    &WORKER_BUDGET
}

/// One retaining site's share of the worker-wide budget.
#[derive(Debug)]
pub(crate) struct RetainedLease {
    budget: &'static RetainedOutputBudget,
    bytes: usize,
}

impl std::fmt::Debug for RetainedOutputBudget {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RetainedOutputBudget")
            .field("held", &self.held())
            .finish_non_exhaustive()
    }
}

impl Default for RetainedLease {
    fn default() -> Self {
        Self {
            budget: current_budget(),
            bytes: 0,
        }
    }
}

impl RetainedLease {
    /// Reserve `bytes` more; false (and nothing reserved) when the worker is full.
    pub(crate) fn reserve(&mut self, bytes: usize) -> bool {
        if bytes == 0 {
            return true;
        }
        if !self.budget.reserve(bytes) {
            return false;
        }
        self.bytes += bytes;
        true
    }

    /// Bytes this site currently holds.
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    /// Return up to `bytes` this site no longer retains.
    pub(crate) fn release(&mut self, bytes: usize) {
        let returned = bytes.min(self.bytes);
        self.budget.release(returned);
        self.bytes -= returned;
    }

    /// Return everything this site holds.
    pub(crate) fn clear(&mut self) {
        self.release(self.bytes);
    }
}

impl Drop for RetainedLease {
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
#[path = "retained_test.rs"]
mod tests;
