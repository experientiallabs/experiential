//! The Messages encoder's share of the worker-wide retention budget.

use super::MessagesSseEncoder;
use crate::errors::PublicError;
use crate::retained::RetainedLease;

impl MessagesSseEncoder {
    /// Reserve newly buffered bytes from the worker-wide budget.
    pub(super) fn retain_buffered(&mut self) -> Result<(), PublicError> {
        let additional = self.buffered_bytes.saturating_sub(self.retained.bytes());
        if self.retained.reserve(additional) {
            return Ok(());
        }
        Err(PublicError::new(
            503,
            "gateway_unavailable",
            crate::retained::WORKER_RETENTION_EXHAUSTED_MESSAGE,
            "api_error",
        ))
    }
}

/// Return the budget for bytes no longer buffered (a free function so a
/// caller holding a block borrow can still call it).
pub(super) fn release_buffered(retained: &mut RetainedLease, buffered_bytes: usize) {
    let surplus = retained.bytes().saturating_sub(buffered_bytes);
    retained.release(surplus);
}
