//! Gateway-emulated `parallel_tool_calls: false`.
//!
//! A caller who sends `parallel_tool_calls: false` asks for at most one tool
//! call per assistant turn. Providers whose wire has no such control (Gemini,
//! Bedrock, and OpenAI-compatible servers that accept but ignore the field)
//! may still emit several calls in one turn. Rather than refuse the request,
//! the route entry marks the rung and this filter serializes the turn: the
//! first tool call streams through untouched and every later call in the same
//! turn is dropped (its start, argument deltas, completion, and its
//! provider-owned Responses item lifecycle). The model sees one result on the
//! next turn and re-issues the remaining calls then, which is exactly the
//! sequential behaviour the caller asked for. Provider-executed server tools
//! and hosted tools are never touched. Nothing here logs request text.
//!
//! Lifetime: one serializer per `UpstreamRelay`, and the waterfall builds a
//! fresh relay for every upstream response, so the kept/dropped state spans
//! exactly one assistant turn. A failover attempt or a Responses continuation
//! is a new relay; state never carries across turns and needs no reset.

use std::collections::BTreeSet;

use crate::events::{Event, ProviderOutputItemKind};

/// Stateful one-call-per-turn filter for one upstream relay.
#[derive(Debug, Default)]
pub struct ToolCallSerializer {
    /// The one tool-call index this turn keeps.
    kept: Option<u32>,
    /// Indexes of calls dropped after the kept one opened.
    dropped: BTreeSet<u32>,
}

impl ToolCallSerializer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the index belongs to the call this turn keeps; a new index
    /// becomes the kept call when none is open yet, otherwise it is dropped.
    fn admit(&mut self, index: u32) -> bool {
        match self.kept {
            None => {
                self.kept = Some(index);
                true
            }
            Some(kept) if kept == index => true,
            Some(_) => {
                self.dropped.insert(index);
                false
            }
        }
    }

    /// Filter one decoded event: `Some(event)` passes, `None` drops it.
    pub fn filter(&mut self, event: Event) -> Option<Event> {
        match &event {
            Event::ToolCallStarted { index, .. } => self.admit(*index).then_some(event),
            Event::ToolArgumentsDelta { index, .. } | Event::ToolCallCompleted { index, .. } => {
                (!self.dropped.contains(index)).then_some(event)
            }
            // A Responses function/custom tool item opens BEFORE its
            // ToolCallStarted and shares its output index, so the item start
            // is where a later call is first seen and dropped.
            Event::ProviderOutputItemStarted {
                output_index, kind, ..
            } if is_tool_item(*kind) => self.admit(*output_index).then_some(event),
            Event::ProviderOutputItemCompleted {
                output_index, kind, ..
            } if is_tool_item(*kind) => (!self.dropped.contains(output_index)).then_some(event),
            _ => Some(event),
        }
    }
}

fn is_tool_item(kind: ProviderOutputItemKind) -> bool {
    matches!(
        kind,
        ProviderOutputItemKind::FunctionCall | ProviderOutputItemKind::CustomToolCall
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::CompletedToolCall;

    fn started(index: u32) -> Event {
        Event::ToolCallStarted {
            custom: false,
            index,
            call_id: format!("call_{index}"),
            name: "lookup".to_string(),
            namespace: None,
            caller: None,
        }
    }

    fn completed(index: u32) -> Event {
        Event::ToolCallCompleted {
            index,
            call: CompletedToolCall {
                call_id: format!("call_{index}"),
                name: "lookup".to_string(),
                namespace: None,
                caller: None,
                provider_item_id: None,
                provider_status: None,
                raw_arguments: "{}".to_string(),
                custom: false,
            },
        }
    }

    fn kinds(events: &[Option<Event>]) -> Vec<String> {
        events
            .iter()
            .map(|event| match event {
                None => "dropped".to_string(),
                Some(Event::ToolCallStarted { index, .. }) => format!("start:{index}"),
                Some(Event::ToolArgumentsDelta { index, .. }) => format!("args:{index}"),
                Some(Event::ToolCallCompleted { index, .. }) => format!("done:{index}"),
                Some(Event::TextDelta(_)) => "text".to_string(),
                Some(Event::Completed) => "completed".to_string(),
                Some(other) => format!("{other:?}")
                    .split(' ')
                    .next()
                    .unwrap_or("")
                    .to_string(),
            })
            .collect()
    }

    #[test]
    fn the_first_call_streams_and_every_later_call_in_the_turn_is_dropped() {
        let mut serializer = ToolCallSerializer::new();
        let out: Vec<Option<Event>> = vec![
            serializer.filter(Event::TextDelta("thinking".into())),
            serializer.filter(started(0)),
            serializer.filter(Event::ToolArgumentsDelta {
                index: 0,
                delta: "{\"city\":\"Paris\"}".into(),
            }),
            serializer.filter(started(1)),
            serializer.filter(Event::ToolArgumentsDelta {
                index: 1,
                delta: "{\"city\":\"Rome\"}".into(),
            }),
            serializer.filter(completed(1)),
            serializer.filter(completed(0)),
            serializer.filter(Event::Completed),
        ];
        assert_eq!(
            kinds(&out),
            vec![
                "text",
                "start:0",
                "args:0",
                "dropped",
                "dropped",
                "dropped",
                "done:0",
                "completed"
            ]
        );
    }

    #[test]
    fn responses_items_are_admitted_at_their_item_start_and_share_the_index_space() {
        let mut serializer = ToolCallSerializer::new();
        let item =
            |output_index: u32, kind: ProviderOutputItemKind| Event::ProviderOutputItemStarted {
                output_index,
                item_id: Some(format!("fc_{output_index}")),
                kind,
                status: None,
                phase: None,
            };
        // A reasoning item never counts as a call.
        assert!(serializer
            .filter(item(0, ProviderOutputItemKind::Reasoning))
            .is_some());
        assert!(serializer
            .filter(item(1, ProviderOutputItemKind::FunctionCall))
            .is_some());
        assert!(serializer.filter(started(1)).is_some());
        // The second function item in the same turn is dropped end to end.
        assert!(serializer
            .filter(item(2, ProviderOutputItemKind::FunctionCall))
            .is_none());
        assert!(serializer.filter(started(2)).is_none());
        assert!(serializer.filter(completed(2)).is_none());
        assert!(serializer
            .filter(Event::ProviderOutputItemCompleted {
                output_index: 2,
                item_id: Some("fc_2".into()),
                kind: ProviderOutputItemKind::FunctionCall,
                status: None,
                phase: None,
            })
            .is_none());
        // The kept item's completion and the message item pass.
        assert!(serializer.filter(completed(1)).is_some());
        assert!(serializer
            .filter(Event::ProviderOutputItemCompleted {
                output_index: 3,
                item_id: None,
                kind: ProviderOutputItemKind::Message,
                status: None,
                phase: None,
            })
            .is_some());
    }

    #[test]
    fn a_single_call_turn_is_untouched() {
        let mut serializer = ToolCallSerializer::new();
        assert!(serializer.filter(started(4)).is_some());
        assert!(serializer
            .filter(Event::ToolArgumentsDelta {
                index: 4,
                delta: "{}".into(),
            })
            .is_some());
        assert!(serializer.filter(completed(4)).is_some());
        assert!(serializer.filter(Event::Completed).is_some());
    }
}
