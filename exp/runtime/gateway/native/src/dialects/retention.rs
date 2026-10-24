//! The normalizer's retained-output reservations: per stream and worker-wide.

use crate::errors::{Failure, FailureClass};

use super::{Normalizer, MAXIMUM_RETAINED_OUTPUT_BYTES, OUTPUT_OVERFLOW_MESSAGE};

impl Normalizer {
    /// Bound aggregate image output even when images are delivered incrementally.
    pub(super) fn reserve_image_bytes(&mut self, additional: usize) -> Result<(), Failure> {
        self.accumulated_image_bytes = self.accumulated_image_bytes.saturating_add(additional);
        self.reserve_tool_bytes(0)?;
        self.reserve_worker_bytes(additional)
    }

    /// Draw retained bytes from the worker-wide budget; a full worker refuses
    /// the stream retryably instead of holding output it cannot afford.
    pub(super) fn reserve_worker_bytes(&mut self, additional: usize) -> Result<(), Failure> {
        if self.retained.reserve(additional) {
            Ok(())
        } else {
            Err(Failure::new(
                FailureClass::Unavailable,
                crate::retained::WORKER_RETENTION_EXHAUSTED_MESSAGE,
            ))
        }
    }

    /// Reserve retained-output budget for accumulated tool-argument text.
    pub(super) fn reserve_tool_bytes(&mut self, additional: usize) -> Result<(), Failure> {
        self.accumulated_tool_bytes = self.accumulated_tool_bytes.saturating_add(additional);
        if self
            .accumulated_tool_bytes
            .saturating_add(self.accumulated_summary_bytes)
            .saturating_add(self.accumulated_image_bytes)
            > MAXIMUM_RETAINED_OUTPUT_BYTES
        {
            return Err(Failure::new(
                FailureClass::ProviderInternal,
                OUTPUT_OVERFLOW_MESSAGE,
            ));
        }
        self.reserve_worker_bytes(additional)
    }

    /// Reserve retained-output budget for reasoning-summary verification.
    pub(super) fn reserve_summary_bytes(&mut self, additional: usize) -> Result<(), Failure> {
        self.accumulated_summary_bytes = self.accumulated_summary_bytes.saturating_add(additional);
        if self
            .accumulated_tool_bytes
            .saturating_add(self.accumulated_summary_bytes)
            .saturating_add(self.accumulated_image_bytes)
            > MAXIMUM_RETAINED_OUTPUT_BYTES
        {
            return Err(Failure::new(
                FailureClass::ProviderInternal,
                OUTPUT_OVERFLOW_MESSAGE,
            ));
        }
        self.reserve_worker_bytes(additional)
    }
}
