//! The output-less-terminal rules of the waterfall: when a successful
//! terminal that carried no semantic event is an honest empty answer and
//! when it is a failed attempt (`empty_completion`), and which rungs answer
//! it at once instead of redialing.

use super::{AttemptEnd, DeploymentWire, SettledAttempt, WaterfallContext};
use crate::errors::Failure;
use crate::events::{Event, Usage};
use crate::settlement::AttemptGuard;

/// Do not regenerate an image when a provider reports success without any
/// deliverable output. The first attempt may already have incurred image cost.
/// Ordinary text lanes retain their empty-completion retry policy.
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

/// Settle one output-less terminal (`Completed` or `Incomplete`) reached
/// before any semantic event: retain the output-less continuation while the
/// attempt is still in flight, settle, then answer with the tracked usage
/// ahead of the terminal so the encoders keep the client-visible token
/// accounting.
#[allow(clippy::too_many_arguments)]
pub(super) async fn settle_output_less(
    ctx: &WaterfallContext<'_>,
    guard: &mut AttemptGuard,
    terminal: Event,
    mut events: Vec<Event>,
    usage: Option<Usage>,
    tool_names: Vec<String>,
    depth: usize,
    encrypted_reasoning_stripped: bool,
) -> AttemptEnd {
    let retention_failure = match &ctx.output_less_retention {
        Some(argument) => ctx.bridge.call("remember", argument.clone()).await.err(),
        None => None,
    };
    let outcome = if matches!(terminal, Event::Incomplete | Event::IncompleteToolArguments) {
        "incomplete"
    } else {
        "completed"
    };
    if !guard
        .settle(outcome, usage.as_ref(), &tool_names, None, true)
        .await
    {
        return AttemptEnd::Accounting;
    }
    if let Some(error) = retention_failure {
        // The provider outcome settled above, exactly like a committed
        // attempt's retention failure; only the HTTP result reports it.
        return AttemptEnd::Retention(error);
    }
    if let Some(tracked) = usage {
        events.push(Event::Usage(tracked));
    }
    events.push(terminal);
    AttemptEnd::Settled(SettledAttempt {
        depth,
        events,
        encrypted_reasoning_stripped,
        empty_completion: false,
        tool_search_rounds: Vec::new(),
    })
}
