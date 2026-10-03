//! Reasoning display: the model's reasoning text returned beside the content.
//!
//! Every rung renders the reasoning text its provider streams unless the rung
//! opts out (`reasoning_output_hidden`) or the request runs an output
//! guardrail. Display is copy only: it never seals a carrier, never changes
//! what replays to a provider, and never replaces the sealed carrier on a tool
//! turn. Each surface renders it in the field its clients already read: Chat
//! `reasoning` (OpenRouter's field, so the `reasoning_content` carrier keeps
//! its meaning), Messages unsigned `thinking` blocks, Responses `summary_text`.

use crate::events::Event;

/// What a winning rung returns of its model's reasoning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReasoningOutput {
    /// The rung replays plaintext `reasoning_content` (Tencent/DeepSeek think
    /// mode) and returns it in that field instead of a sealed carrier.
    pub exposed: bool,
    /// The rung renders its reasoning text as display copy.
    pub displayed: bool,
}

impl From<bool> for ReasoningOutput {
    /// An exposure fact alone, with display off.
    fn from(exposed: bool) -> Self {
        Self {
            exposed,
            displayed: false,
        }
    }
}

/// Which provider unit a display delta belongs to, so consecutive units of
/// one turn (OpenAI summary parts, Anthropic thinking blocks) stay separate
/// paragraphs once flattened into one text field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unit {
    Text,
    Summary(u32, u32),
    Thinking(u32),
    Route,
}

/// The display text one event contributes on a flattened surface (Chat), or
/// `None`. Exposed route reasoning already travels in `reasoning_content`.
fn unit_text(event: &Event, exposed: bool) -> Option<(Unit, &str)> {
    let (unit, delta) = match event {
        Event::ReasoningTextDelta(delta) => (Unit::Text, delta),
        Event::ReasoningSummaryDelta {
            output_index,
            summary_index,
            delta,
            ..
        } => (Unit::Summary(*output_index, *summary_index), delta),
        Event::ThinkingDelta { index, delta } => (Unit::Thinking(*index), delta),
        Event::ReasoningContentDelta { delta, .. } if !exposed => (Unit::Route, delta),
        _ => return None,
    };
    (!delta.is_empty()).then_some((unit, delta.as_str()))
}

/// Flattens a turn's reasoning deltas into one display string, separating
/// provider units with a blank line.
#[derive(Default)]
pub(crate) struct DisplayJoiner {
    last: Option<Unit>,
}

impl DisplayJoiner {
    /// The display delta `event` contributes, prefixed with a paragraph break
    /// when it opens a new provider unit after earlier reasoning.
    pub(crate) fn delta(&mut self, event: &Event, exposed: bool) -> Option<String> {
        let (unit, text) = unit_text(event, exposed)?;
        let separated = self.last.is_some_and(|last| last != unit);
        self.last = Some(unit);
        Some(if separated {
            format!("\n\n{text}")
        } else {
            text.to_string()
        })
    }
}

/// The whole turn's flattened display text, or `None` when it reasoned in no
/// displayable form.
pub(crate) fn flattened(events: &[Event], exposed: bool) -> Option<String> {
    let mut joiner = DisplayJoiner::default();
    let text: String = events
        .iter()
        .filter_map(|event| joiner.delta(event, exposed))
        .collect();
    (!text.is_empty()).then_some(text)
}

/// The plaintext one Messages delta adds to the gateway's unsigned thinking
/// block, or `None`. Anthropic thinking renders as its own signed blocks.
pub(crate) fn unsigned_thinking_text(event: &Event, output: ReasoningOutput) -> Option<&str> {
    let delta = match event {
        Event::ReasoningContentDelta { delta, .. } if output.exposed || output.displayed => delta,
        Event::ReasoningTextDelta(delta) | Event::ReasoningSummaryDelta { delta, .. }
            if output.displayed =>
        {
            delta
        }
        _ => return None,
    };
    (!delta.is_empty()).then_some(delta.as_str())
}

/// The joined text one Messages delta adds to the gateway's unsigned thinking
/// block: the `unsigned_thinking_text` gate, flattened like Chat so
/// consecutive provider units (OpenAI summary parts) stay separate
/// paragraphs. Exposed route reasoning is display text here, since the
/// Messages wire has no `reasoning_content` field.
pub(crate) fn unsigned_thinking_delta(
    joiner: &mut DisplayJoiner,
    event: &Event,
    output: ReasoningOutput,
) -> Option<String> {
    unsigned_thinking_text(event, output)?;
    joiner.delta(event, false)
}

#[cfg(test)]
#[path = "reasoning_display_tests.rs"]
mod tests;
