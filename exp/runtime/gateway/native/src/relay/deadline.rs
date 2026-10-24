//! The messages a caller reads when the request's total deadline runs out.
//! They name what happened (still writing, or nothing yet) and the remedy,
//! never the internal mechanism.

use std::time::Instant;

use super::remaining;
use crate::errors::{Failure, FailureClass};

/// The request deadline ran out while the model was still producing output.
/// A model that streams for the whole deadline is almost always looping, so
/// the message names that and the two remedies a caller controls.
pub const DEADLINE_WHILE_GENERATING_MESSAGE: &str = "the response was cut off because the \
request reached the gateway's time limit while the model was still writing; this usually \
means the model got stuck repeating itself. Retry, or set a lower max_tokens so long \
answers stop on their own";

/// The request deadline ran out before the provider produced any output.
pub const DEADLINE_BEFORE_OUTPUT_MESSAGE: &str = "the request reached the gateway's time \
limit before the provider started answering; retry the request";

/// Classify one mid-stream chunk timeout the way the python transport does:
/// a stalled provider read is a retryable transport failure unless the
/// request's own deadline is exhausted.
pub fn stream_timeout_failure(deadline: Instant) -> Failure {
    if remaining(deadline).is_zero() {
        Failure::new(FailureClass::Timeout, DEADLINE_BEFORE_OUTPUT_MESSAGE)
    } else {
        Failure::new(
            FailureClass::Transport,
            "provider transport failed; retry the request",
        )
        .with_retry(true, true)
    }
}
