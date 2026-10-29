//! Byte-weighted admission budget for large decision bodies.
//!
//! A decision body may carry tens of MiB of base64 images, and admission
//! copies it through the Python bridge before the execution permit is taken.
//! Bodies over one MiB first take one unit per started MiB from a per-worker
//! pool, held for the whole request, so concurrent large bodies cannot
//! multiply peak memory without bound. Small bodies never wait on it.

use std::time::Instant;

use tokio::sync::{Semaphore, SemaphorePermit};

use crate::errors::{Failure, FailureClass, PublicError};

/// Bodies at or under this size skip the budget.
pub(crate) const LARGE_BODY_BYTES: usize = 1024 * 1024;
/// Per-worker MiB of large decision bodies admitted at once; above the
/// 64 MiB shared request cap so any single accepted body fits.
const LARGE_BODY_BUDGET_MIB: usize = 128;

static LARGE_BODY_BUDGET: Semaphore = Semaphore::const_new(LARGE_BODY_BUDGET_MIB);

/// The budget units one body of `length` bytes holds (zero for small bodies).
pub(crate) fn body_weight(length: usize) -> u32 {
    if length <= LARGE_BODY_BYTES {
        return 0;
    }
    let units = length.div_ceil(LARGE_BODY_BYTES).min(LARGE_BODY_BUDGET_MIB);
    u32::try_from(units).unwrap_or(u32::MAX)
}

/// Hold this worker's budget for a body of `length` bytes until `deadline`.
pub(crate) async fn reserve_body(
    length: usize,
    deadline: Instant,
) -> Result<Option<SemaphorePermit<'static>>, PublicError> {
    reserve_from(&LARGE_BODY_BUDGET, length, deadline).await
}

async fn reserve_from(
    budget: &Semaphore,
    length: usize,
    deadline: Instant,
) -> Result<Option<SemaphorePermit<'_>>, PublicError> {
    let weight = body_weight(length);
    if weight == 0 {
        return Ok(None);
    }
    match tokio::time::timeout_at(deadline.into(), budget.acquire_many(weight)).await {
        Ok(Ok(permit)) => Ok(Some(permit)),
        Ok(Err(_)) => Err(PublicError::draining()),
        Err(_) => Err(Failure::new(
            FailureClass::Timeout,
            "gateway large-request memory budget deadline exceeded",
        )
        .public_error()),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn small_bodies_are_free_and_large_bodies_weigh_started_mebibytes() {
        assert_eq!(body_weight(0), 0);
        assert_eq!(body_weight(LARGE_BODY_BYTES), 0);
        assert_eq!(body_weight(LARGE_BODY_BYTES + 1), 2);
        assert_eq!(body_weight(64 * LARGE_BODY_BYTES), 64);
        assert_eq!(body_weight(usize::MAX), 128);
    }

    #[tokio::test]
    async fn concurrent_large_bodies_wait_for_budget_and_release_it() {
        let budget = Semaphore::new(64);
        let deadline = Instant::now() + Duration::from_secs(5);
        let small = reserve_from(&budget, 512, deadline).await.unwrap();
        assert!(small.is_none());
        let first = reserve_from(&budget, 48 * LARGE_BODY_BYTES, deadline)
            .await
            .unwrap();
        assert_eq!(budget.available_permits(), 16);
        let short = Instant::now() + Duration::from_millis(20);
        assert!(reserve_from(&budget, 48 * LARGE_BODY_BYTES, short)
            .await
            .is_err());
        drop(first);
        assert_eq!(budget.available_permits(), 64);
        let second = reserve_from(&budget, 48 * LARGE_BODY_BYTES, deadline)
            .await
            .unwrap();
        assert!(second.is_some());
    }

    #[tokio::test]
    async fn a_closed_budget_reports_draining() {
        let budget = Semaphore::new(4);
        budget.close();
        let deadline = Instant::now() + Duration::from_secs(1);
        assert!(reserve_from(&budget, 2 * LARGE_BODY_BYTES, deadline)
            .await
            .is_err());
    }
}
