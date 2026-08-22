//! Idle-time allocator reclamation for the long-lived gateway process.
//!
//! Under high concurrency glibc's per-thread arenas keep freed pages cached
//! indefinitely, so process RSS stays pinned at the traffic peak long after
//! load stops. The reclaim loop watches for the busy-to-idle transition and
//! returns that cached free memory to the operating system exactly once per
//! burst. On non-glibc targets the release call compiles to a no-op.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Semaphore;

/// How often the reclaim loop samples for the busy-to-idle transition.
const SAMPLE_INTERVAL: Duration = Duration::from_secs(5);

#[cfg(all(target_os = "linux", target_env = "gnu"))]
extern "C" {
    /// glibc extension: release free heap memory in every arena to the OS.
    fn malloc_trim(pad: usize) -> std::os::raw::c_int;
}

/// Release allocator-cached free pages back to the operating system.
pub fn release_free_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        malloc_trim(0);
    }
}

/// Whether a trim is due: the plane is idle and traffic landed since the
/// last trim. `available_permits == max_active` means no request holds an
/// active-request permit, zero pending settlements means no terminal
/// accounting write is still in flight, and zero active proxies means no
/// request is relaying through the python fallback engine.
fn trim_due(
    available_permits: usize,
    max_active: usize,
    pending_settlements: usize,
    active_proxies: usize,
    handled_requests: usize,
    trimmed_at: usize,
) -> bool {
    available_permits == max_active
        && pending_settlements == 0
        && active_proxies == 0
        && handled_requests != trimmed_at
}

/// Run forever: after each burst of traffic fully settles, trim once.
pub async fn reclaim_when_idle(
    permits: Arc<Semaphore>,
    max_active: usize,
    handled_requests: Arc<AtomicUsize>,
    pending_settlements: Arc<AtomicUsize>,
    active_proxies: Arc<AtomicUsize>,
) {
    let mut trimmed_at = handled_requests.load(Ordering::SeqCst);
    loop {
        tokio::time::sleep(SAMPLE_INTERVAL).await;
        let handled = handled_requests.load(Ordering::SeqCst);
        if trim_due(
            permits.available_permits(),
            max_active,
            pending_settlements.load(Ordering::SeqCst),
            active_proxies.load(Ordering::SeqCst),
            handled,
            trimmed_at,
        ) {
            // Off the reactor: a large trim walks every arena.
            let _ = tokio::task::spawn_blocking(release_free_memory).await;
            trimmed_at = handled;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_once_per_burst_only_when_idle() {
        assert!(trim_due(64, 64, 0, 0, 10, 0));
        // Busy: a request holds a permit.
        assert!(!trim_due(63, 64, 0, 0, 10, 0));
        // A settlement write is still in flight.
        assert!(!trim_due(64, 64, 1, 0, 10, 0));
        // A proxied request is still relaying through the python engine.
        assert!(!trim_due(64, 64, 0, 1, 10, 0));
        // Nothing handled since the last trim.
        assert!(!trim_due(64, 64, 0, 0, 10, 10));
    }

    #[test]
    fn release_free_memory_is_safe_to_call() {
        release_free_memory();
        release_free_memory();
    }
}
