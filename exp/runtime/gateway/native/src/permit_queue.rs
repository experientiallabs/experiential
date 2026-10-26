//! Class-ordered worker admission permits with aging, off unless configured.
//!
//! Every admitted request waits for one of the worker's `max_active_requests`
//! permits. By default that wait is the semaphore's FIFO, exactly as before.
//! With `permit_priority_aging_seconds` set in the serve config, waiters are
//! released by request class instead (`Admission::priority_class`, 0..8,
//! higher first): only the waiter with the highest effective class contends
//! for the semaphore, and every waiter gains one class per aging period so a
//! low class is never starved. Ties go to the earliest arrival. A request with
//! no class waits as class 0, so until a host sends classes the queue is FIFO.
//!
//! The queue only reorders waiting; it never adds permits, never preempts a
//! running request, and a request that finds a free permit with nobody
//! waiting takes it at once.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::{AcquireError, Notify, OwnedSemaphorePermit, Semaphore, TryAcquireError};

/// How often a waiting request re-checks its standing without an event, so
/// aging can promote it while nothing arrives or leaves.
const REEVALUATE: Duration = Duration::from_millis(100);

#[derive(Debug)]
struct Waiter {
    id: u64,
    class: u8,
    since: Instant,
    scope: Option<Arc<Semaphore>>,
}

#[derive(Debug, Default)]
struct Waiters {
    next_id: u64,
    entries: Vec<Waiter>,
    closed: bool,
}

/// The worker's permit queue; FIFO (the plain semaphore) when `aging` is `None`.
#[derive(Debug)]
pub(crate) struct PermitQueue {
    aging: Option<Duration>,
    waiters: Mutex<Waiters>,
    changed: Arc<Notify>,
}

/// Release both capacities before waking the next eligible waiter.
pub(crate) struct ScopedPermit {
    global: Option<OwnedSemaphorePermit>,
    scope: Option<OwnedSemaphorePermit>,
    changed: Arc<Notify>,
}

impl Drop for ScopedPermit {
    fn drop(&mut self) {
        drop(self.global.take());
        drop(self.scope.take());
        self.changed.notify_waiters();
    }
}

/// Removes its waiter on every exit (permit, deadline, disconnect).
struct Registration<'a> {
    queue: &'a PermitQueue,
    id: u64,
}

impl Drop for Registration<'_> {
    fn drop(&mut self) {
        if let Ok(mut waiters) = self.queue.waiters.lock() {
            waiters.entries.retain(|waiter| waiter.id != self.id);
        }
        self.queue.changed.notify_waiters();
    }
}

impl PermitQueue {
    /// A queue ordering by class with one class of aging per `aging_seconds`;
    /// `None`, zero, negative or non-finite keeps the plain FIFO semaphore.
    pub(crate) fn new(aging_seconds: Option<f64>) -> Self {
        let aging = aging_seconds
            .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
            .map(Duration::from_secs_f64);
        Self {
            aging,
            waiters: Mutex::new(Waiters::default()),
            changed: Arc::new(Notify::new()),
        }
    }

    /// Stop waiting admissions before draining already-running streams.
    pub(crate) fn close(&self) {
        if let Ok(mut waiters) = self.waiters.lock() {
            waiters.closed = true;
        }
        self.changed.notify_waiters();
    }

    /// Grant global and supplied capacity together, without holding either while waiting.
    pub(crate) async fn acquire_scoped(
        &self,
        global: &Arc<Semaphore>,
        scope: &Arc<Semaphore>,
        class: Option<u8>,
    ) -> Result<ScopedPermit, TryAcquireError> {
        let id = {
            let mut waiters = self.waiters.lock().map_err(|_| TryAcquireError::Closed)?;
            if waiters.closed || global.is_closed() || scope.is_closed() {
                return Err(TryAcquireError::Closed);
            }
            let id = waiters.next_id;
            waiters.next_id += 1;
            waiters.entries.push(Waiter {
                id,
                class: if self.aging.is_some() {
                    class.unwrap_or(0)
                } else {
                    0
                },
                since: Instant::now(),
                scope: Some(scope.clone()),
            });
            id
        };
        self.changed.notify_waiters();
        let _registration = Registration { queue: self, id };
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let waiters = self.waiters.lock().map_err(|_| TryAcquireError::Closed)?;
                if waiters.closed || global.is_closed() || scope.is_closed() {
                    return Err(TryAcquireError::Closed);
                }
                let aging = self.aging.unwrap_or(Duration::MAX);
                if head(&waiters.entries, aging, Instant::now()) == Some(id) {
                    if let Ok(global_permit) = global.clone().try_acquire_owned() {
                        match scope.clone().try_acquire_owned() {
                            Ok(scope_permit) => {
                                return Ok(ScopedPermit {
                                    global: Some(global_permit),
                                    scope: Some(scope_permit),
                                    changed: self.changed.clone(),
                                })
                            }
                            Err(TryAcquireError::Closed) => return Err(TryAcquireError::Closed),
                            Err(TryAcquireError::NoPermits) => drop(global_permit),
                        }
                    }
                }
            }
            tokio::select! {
                _ = &mut notified => {},
                _ = tokio::time::sleep(REEVALUATE) => {},
            }
        }
    }

    /// Wait for one permit, in class order when the queue is configured.
    pub(crate) async fn acquire(
        &self,
        permits: &Arc<Semaphore>,
        class: Option<u8>,
    ) -> Result<OwnedSemaphorePermit, AcquireError> {
        let Some(aging) = self.aging else {
            return permits.clone().acquire_owned().await;
        };
        let id = {
            let Ok(mut waiters) = self.waiters.lock() else {
                return permits.clone().acquire_owned().await;
            };
            if waiters.entries.is_empty() {
                if let Ok(permit) = permits.clone().try_acquire_owned() {
                    return Ok(permit);
                }
            }
            let id = waiters.next_id;
            waiters.next_id += 1;
            waiters.entries.push(Waiter {
                id,
                class: class.unwrap_or(0),
                since: Instant::now(),
                scope: None,
            });
            id
        };
        // A new arrival may outrank the current head: let every waiter re-check.
        self.changed.notify_waiters();
        let _registration = Registration { queue: self, id };
        loop {
            if permits.is_closed() {
                return permits.clone().acquire_owned().await;
            }
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.is_head(id, aging, Instant::now()) {
                tokio::select! {
                    biased;
                    permit = permits.clone().acquire_owned() => return permit,
                    _ = &mut notified => {}
                    _ = tokio::time::sleep(REEVALUATE) => {}
                }
            } else {
                tokio::select! {
                    _ = &mut notified => {}
                    _ = tokio::time::sleep(REEVALUATE) => {}
                }
            }
        }
    }

    /// Whether `id` has the highest effective class (earliest arrival on ties).
    fn is_head(&self, id: u64, aging: Duration, now: Instant) -> bool {
        let Ok(waiters) = self.waiters.lock() else {
            return true;
        };
        head(&waiters.entries, aging, now) == Some(id)
    }
}

/// The waiter that contends next: highest `class + waited / aging`, then the
/// earliest arrival.
fn head(entries: &[Waiter], aging: Duration, now: Instant) -> Option<u64> {
    let effective = |waiter: &Waiter| {
        f64::from(waiter.class)
            + now.saturating_duration_since(waiter.since).as_secs_f64() / aging.as_secs_f64()
    };
    entries
        .iter()
        .filter(|waiter| {
            waiter
                .scope
                .as_ref()
                .is_none_or(|scope| scope.available_permits() > 0)
        })
        .max_by(|left, right| {
            effective(left)
                .total_cmp(&effective(right))
                .then_with(|| right.id.cmp(&left.id))
        })
        .map(|waiter| waiter.id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn waiter(id: u64, class: u8, since: Instant) -> Waiter {
        Waiter {
            id,
            class,
            since,
            scope: None,
        }
    }

    #[test]
    fn head_is_the_highest_class_then_the_earliest_arrival() {
        let now = Instant::now();
        let aging = Duration::from_secs(10);
        let entries = vec![waiter(0, 2, now), waiter(1, 6, now), waiter(2, 6, now)];
        assert_eq!(head(&entries, aging, now), Some(1));
        assert_eq!(head(&[], aging, now), None);
    }

    #[test]
    fn aging_promotes_a_long_waiting_low_class() {
        let start = Instant::now();
        let aging = Duration::from_secs(1);
        let entries = vec![
            waiter(0, 0, start),
            waiter(1, 5, start + Duration::from_secs(6)),
        ];
        // At 6 s the class-0 waiter has aged to 6.0, past the fresh class 5.
        assert_eq!(
            head(&entries, aging, start + Duration::from_secs(6)),
            Some(0)
        );
        assert_eq!(
            head(&entries, aging, start + Duration::from_secs(4)),
            Some(1)
        );
    }

    #[test]
    fn invalid_aging_keeps_the_fifo_semaphore() {
        for aging in [
            None,
            Some(0.0),
            Some(-1.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
        ] {
            assert!(PermitQueue::new(aging).aging.is_none(), "{aging:?}");
        }
    }

    #[tokio::test]
    async fn unconfigured_queue_is_the_plain_semaphore() {
        let permits = Arc::new(Semaphore::new(1));
        let queue = PermitQueue::new(None);
        let held = queue.acquire(&permits, Some(8)).await.unwrap();
        assert_eq!(permits.available_permits(), 0);
        drop(held);
        assert_eq!(permits.available_permits(), 1);
        assert!(queue.waiters.lock().unwrap().entries.is_empty());
    }

    #[tokio::test]
    async fn waiters_are_released_by_class_not_arrival() {
        let permits = Arc::new(Semaphore::new(1));
        let queue = Arc::new(PermitQueue::new(Some(3_600.0)));
        let held = queue.acquire(&permits, Some(0)).await.unwrap();
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut tasks = Vec::new();
        for (class, delay) in [(1u8, 0u64), (0, 10), (7, 20), (4, 30)] {
            let (queue, permits, order) = (queue.clone(), permits.clone(), order.clone());
            tasks.push(tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(delay)).await;
                let permit = queue.acquire(&permits, Some(class)).await.unwrap();
                order.lock().unwrap().push(class);
                tokio::time::sleep(Duration::from_millis(5)).await;
                drop(permit);
            }));
        }
        tokio::time::sleep(Duration::from_millis(80)).await;
        drop(held);
        for task in tasks {
            task.await.unwrap();
        }
        assert_eq!(*order.lock().unwrap(), vec![7, 4, 1, 0]);
        assert!(queue.waiters.lock().unwrap().entries.is_empty());
    }

    #[tokio::test]
    async fn a_cancelled_waiter_leaves_the_queue() {
        let permits = Arc::new(Semaphore::new(1));
        let queue = PermitQueue::new(Some(10.0));
        let held = queue.acquire(&permits, None).await.unwrap();
        let timed_out =
            tokio::time::timeout(Duration::from_millis(30), queue.acquire(&permits, Some(8))).await;
        assert!(timed_out.is_err());
        assert!(queue.waiters.lock().unwrap().entries.is_empty());
        drop(held);
        assert!(queue.acquire(&permits, Some(0)).await.is_ok());
    }
}
