//! Inspect gateway-generated preambles and pre-commit terminal output while
//! the accounting entry still owns its request-specific host session.

use super::WaterfallContext;
use crate::errors::Failure;
use crate::events::Event;
use crate::guardrails::runtime::RuntimeInspector;
use crate::tool_search::{messages_prelude_events, responses_prelude_events, ToolSearchRound};
use crate::web_search::{web_search_prelude_events, WebSearchAdmission};

/// Only host-inspected admissions populate this context.
pub struct InspectionContext<'a> {
    pub web_search: Option<&'a WebSearchAdmission>,
    pub responses: bool,
}

/// Inspect all text the gateway will synthesize, then any final withheld events.
/// The projection covers Chat citation titles/URLs as well as Messages search
/// blocks; Responses tool-search schemas use their exact hosted-item form.
pub(super) async fn inspect_outward(
    ctx: &WaterfallContext<'_>,
    rounds: &[ToolSearchRound],
    tail: &[Event],
    final_segment: bool,
) -> Result<(), Failure> {
    let Some(inspection) = &ctx.inspection else {
        return Ok(());
    };
    let mut inspector = RuntimeInspector::new(ctx.request_id, ctx.deadline);
    let mut events = inspection.web_search.map_or_else(Vec::new, |search| {
        web_search_prelude_events(search, ctx.request_id)
    });
    events.extend(if inspection.responses {
        responses_prelude_events(rounds, ctx.request_id)
    } else {
        messages_prelude_events(rounds, ctx.request_id)
    });
    events.extend_from_slice(tail);
    if events.is_empty() && !final_segment {
        return Ok(());
    }
    let terminal = events.last().is_some_and(Event::is_terminal);
    for event in events {
        inspector.admit(ctx.bridge, event).await?;
    }
    if !terminal {
        inspector.flush(ctx.bridge, final_segment).await?;
    }
    Ok(())
}
