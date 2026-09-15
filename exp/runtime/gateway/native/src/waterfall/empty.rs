//! The output-less-terminal rules of the waterfall: when a successful
//! terminal that carried no semantic event is an honest empty answer and
//! when it is a failed attempt (`empty_completion`), and which rungs answer
//! it at once instead of redialing.

use super::DeploymentWire;
use crate::errors::Failure;
use crate::events::{Event, Usage};

/// The empty-completion failure for one rung. On an image-output rung the
/// redial and the ladder are switched off: the chat normalizers carry no
/// image event, so every image generation ends as an empty completion and a
/// redial would bill the house a second image for the same nothing; the
/// exhausted ladder then answers the typed 200 at once. Elsewhere the empty
/// answer is not deterministic and one redial plus the ladder stay on.
pub(crate) fn empty_completion_failure(wire: &DeploymentWire) -> Failure {
    let failure = Failure::empty_completion();
    if wire.image_output {
        failure.with_retry(false, false)
    } else {
        failure
    }
}

/// Whether a successful terminal with no semantic output is a billed empty
/// completion: the turn ended `Completed` (the provider's plain `stop`) while
/// its reported usage counts at least one output or reasoning token. A budget
/// truncation (`Incomplete`), a stop sequence, a paused turn, an unreported
/// usage, or a zero-token stop are honest output-less endings and stay
/// settled as they are; only a paid-for `stop` that delivered nothing fails.
pub(crate) fn billed_empty_completion(terminal: &Event, usage: Option<&Usage>) -> bool {
    if !matches!(terminal, Event::Completed) {
        return false;
    }
    let Some(usage) = usage else {
        return false;
    };
    usage.output_tokens.is_some_and(|tokens| tokens > 0)
        || usage.reasoning_tokens.is_some_and(|tokens| tokens > 0)
}

/// Whether a successful terminal with no semantic output is an unreported
/// empty completion: the turn ended `Completed` while the provider sent no
/// usage report at all. A zero-token stop WITH a report is the provider
/// saying "nothing" and accounting for it; no report and no output means the
/// gateway cannot tell an empty answer from a budget the provider's hidden
/// reasoning exhausted (Meta muse-spark, 2026-09-15: role delta, empty delta
/// `finish_reason: stop`, `[DONE]`, no usage frame, whenever `max_tokens` is
/// below the model's private reasoning), so the caller must not receive a
/// completed empty answer either way.
pub(crate) fn unreported_empty_completion(terminal: &Event, usage: Option<&Usage>) -> bool {
    matches!(terminal, Event::Completed) && usage.is_none()
}
