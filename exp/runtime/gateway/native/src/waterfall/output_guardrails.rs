//! Validate gateway-generated output and terminal responses through the same policy session.

use super::WaterfallContext;
use crate::errors::{Failure, FailureClass};
use crate::events::Event;
use crate::tool_search::{messages_prelude_events, responses_prelude_events, ToolSearchRound};
use crate::web_search::{web_search_prelude_events, WebSearchAdmission};

/// Output policy presence and the protocol projection for gateway-generated content.
pub struct OutputGuardrailContext<'a> {
    pub web_search: Option<&'a WebSearchAdmission>,
    pub responses: bool,
}

impl OutputGuardrailContext<'_> {
    /// Project generated content into the same subject as the provider's answer.
    pub(crate) fn events(&self, request_id: &str, rounds: &[ToolSearchRound]) -> Vec<Event> {
        let mut events = self.web_search.map_or_else(Vec::new, |search| {
            web_search_prelude_events(search, request_id)
        });
        events.extend(if self.responses {
            responses_prelude_events(rounds, request_id)
        } else {
            messages_prelude_events(rounds, request_id)
        });
        events
    }
}

/// Inspect content while accounting still owns its request session.
/// Rewrites of generated metadata cannot be applied by protocol encoders, so fail closed.
pub(super) async fn inspect_outward(
    ctx: &WaterfallContext<'_>,
    rounds: &[ToolSearchRound],
    tail: &[Event],
) -> Result<(), Failure> {
    let Some(output) = &ctx.output_guardrails else {
        return Ok(());
    };
    let mut events = output.events(ctx.request_id, rounds);
    events.extend_from_slice(tail);
    if events.is_empty() {
        return Ok(());
    }
    let checked =
        crate::guardrails::enforce_collected_output(ctx.bridge, ctx.request_id, events.clone())
            .await?;
    if crate::guardrails::output_argument(ctx.request_id, &checked)
        != crate::guardrails::output_argument(ctx.request_id, &events)
    {
        return Err(Failure::new(
            FailureClass::Guardrail,
            "The request was blocked by a gateway guardrail.",
        ));
    }
    Ok(())
}
