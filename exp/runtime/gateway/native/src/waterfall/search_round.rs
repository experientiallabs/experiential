//! One gateway-run tool-search round between two dials of the same rung.
//!
//! The dial that ended with only withheld search calls is settled as a
//! completed, billed attempt of its own (non-finalizing: the request stays
//! open for the re-dial); the control plane then runs the search over the
//! deferred catalog, extends the conversation with the call and its result,
//! loads the matched tools, and answers with the rebuilt wire for the same
//! depth. A control plane that cannot answer fails the request closed with
//! the gateway's own error: a half answer is never served.

use super::{WaterfallContext, Won};
use crate::errors::{Failure, FailureClass, PublicError};
use crate::events::Usage;
use crate::settlement::AttemptGuard;
use crate::tool_search::{
    round_argument, ToolSearchRound, ToolSearchRoundReply, WithheldSearchCall,
};

/// The failure of a search call the round budget (or the control plane's
/// withdrawal of the tool) no longer allows: neither retryable nor
/// failover-eligible, so the ladder ends here.
pub(super) fn budget_exhausted() -> Failure {
    Failure::new(
        FailureClass::Internal,
        "gateway tool search round budget exhausted",
    )
}

/// Settle the search-call attempt, ask the control plane for the round, and
/// return its reply; `Err` carries the request's already-finalized outcome.
#[allow(clippy::too_many_arguments)]
pub(super) async fn negotiate(
    ctx: &WaterfallContext<'_>,
    guard: &mut AttemptGuard,
    depth: usize,
    round: u32,
    calls: &[WithheldSearchCall],
    usage: Option<&Usage>,
    tool_names: &[String],
    rounds: &mut Vec<ToolSearchRound>,
) -> Result<ToolSearchRoundReply, Won> {
    // Ask the control plane first, while the search-call attempt is still
    // open: a round that cannot be answered settles THIS attempt as the
    // request's finalizing failure, carrying every round metered so far, so
    // the searches are never lost to an attempt-less abandon.
    let argument = round_argument(ctx.request_id, depth, round, calls, usage);
    let reply: Option<ToolSearchRoundReply> =
        match ctx.bridge.call("tool_search_round", argument).await {
            Ok(text) => serde_json::from_str(&text).ok(),
            Err(_) => None,
        };
    let Some(reply) = reply else {
        guard.record_tool_search_requests(rounds.len() as u32);
        return Err(
            fail_closed(guard, usage, tool_names, "gateway tool search round failed").await,
        );
    };
    rounds.extend(reply.rounds.iter().cloned());
    guard.record_tool_search_requests(rounds.len() as u32);
    // The search-call turn was answered and billed; it closes as completed
    // without finalizing the request, exactly like a throttled rung's
    // non-finalizing settlement leaves the request open for its redial.
    if !guard
        .settle("completed", usage, tool_names, None, false)
        .await
    {
        return Err(Won::Failed(PublicError::internal()));
    }
    Ok(reply)
}

/// Terminalize the request on the still-open search-call attempt with the
/// gateway's own internal error (a finalizing settlement, so the metered
/// rounds ride along).
async fn fail_closed(
    guard: &mut AttemptGuard,
    usage: Option<&Usage>,
    tool_names: &[String],
    message: &str,
) -> Won {
    let failure = Failure::new(FailureClass::Internal, message);
    guard
        .settle("failed", usage, tool_names, Some(&failure), true)
        .await;
    Won::Failed(PublicError::internal())
}
