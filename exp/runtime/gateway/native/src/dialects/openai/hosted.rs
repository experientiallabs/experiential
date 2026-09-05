//! Hosted-tool pass-through for the OpenAI Responses dialect: the
//! provider-executed output items (`web_search_call`, `mcp_call`, ...), their
//! per-type lifecycle and delta frames, and the output-text annotations their
//! answers cite. The provider owns every one of these shapes, so items and
//! frames carry verbatim compact JSON; only exact identity is parsed here.

use serde_json::{Map, Value};

use super::super::{bounded_wire_token, malformed, Normalizer, OpenAiHostedItem};
use super::{openai_identity, openai_index};
use crate::encode::compact_json;
use crate::errors::Failure;
use crate::events::{Event, ProviderOutputItemKind};

/// The documented Responses output-item union beyond the four typed kinds
/// (`message`, `function_call`, `custom_tool_call`, `reasoning`): hosted
/// (provider-executed) tool calls, their caller-facing outputs, and the
/// opaque conversation items. Every one is provider-owned vocabulary this
/// gateway cannot re-model, so the whole item passes through verbatim; an
/// item type outside this closed list stays a malformed stream, with the
/// type named so the next unknown shape is diagnosable from logs.
/// (openai-python 3.x `ResponseOutputItem` union, checked 2026-09-04.)
const OPENAI_HOSTED_OUTPUT_ITEM_TYPES: [&str; 24] = [
    "web_search_call",
    "file_search_call",
    "code_interpreter_call",
    "computer_call",
    "computer_call_output",
    "image_generation_call",
    "local_shell_call",
    "local_shell_call_output",
    "shell_call",
    "shell_call_output",
    "apply_patch_call",
    "apply_patch_call_output",
    "mcp_call",
    "mcp_list_tools",
    "mcp_approval_request",
    "mcp_approval_response",
    "function_call_output",
    "custom_tool_call_output",
    "tool_search_call",
    "tool_search_output",
    "program",
    "program_output",
    "additional_tools",
    "compaction",
];

/// Whether one output-item type belongs to the hosted pass-through union.
pub(super) fn is_openai_hosted_item_type(item_type: &str) -> bool {
    OPENAI_HOSTED_OUTPUT_ITEM_TYPES.contains(&item_type)
}

/// The per-type hosted-tool lifecycle and delta stream events the Responses
/// wire emits between an item's `output_item.added` and `.done`. Each frame
/// passes through verbatim to the caller; the `.done` item remains the
/// authority for the final output array. The `shell_call_command.*` family
/// carries no `item_id` on the wire (openai-python 3.x stream-event union).
pub(super) fn is_openai_hosted_progress_event(event_type: &str) -> bool {
    matches!(
        event_type,
        "response.web_search_call.in_progress"
            | "response.web_search_call.searching"
            | "response.web_search_call.completed"
            | "response.file_search_call.in_progress"
            | "response.file_search_call.searching"
            | "response.file_search_call.completed"
            | "response.code_interpreter_call.in_progress"
            | "response.code_interpreter_call.interpreting"
            | "response.code_interpreter_call.completed"
            | "response.code_interpreter_call_code.delta"
            | "response.code_interpreter_call_code.done"
            | "response.image_generation_call.in_progress"
            | "response.image_generation_call.generating"
            | "response.image_generation_call.completed"
            | "response.image_generation_call.partial_image"
            | "response.mcp_call.in_progress"
            | "response.mcp_call.completed"
            | "response.mcp_call.failed"
            | "response.mcp_call_arguments.delta"
            | "response.mcp_call_arguments.done"
            | "response.mcp_list_tools.in_progress"
            | "response.mcp_list_tools.completed"
            | "response.mcp_list_tools.failed"
            | "response.shell_call_command.added"
            | "response.shell_call_command.delta"
            | "response.shell_call_command.done"
            | "response.shell_call_output_content.delta"
            | "response.shell_call_output_content.done"
    )
}

impl Normalizer {
    /// Open one hosted output item and carry its verbatim JSON.
    ///
    /// Hosted-tool and opaque conversation items pass through verbatim: their
    /// statuses ("searching", "generating", "calling", "failed", ...) and
    /// payload fields are provider-owned vocabulary, so nothing beyond exact
    /// identity is parsed here.
    pub(in crate::dialects) fn openai_hosted_item_added(
        &mut self,
        index: u32,
        item_type: &str,
        item: &Map<String, Value>,
    ) -> Result<Vec<Event>, Failure> {
        let item_id = openai_identity(item, "id", "OpenAI hosted tool item ID")?;
        if self.openai_hosted_items.contains_key(&index)
            || self.openai_output_items.contains_key(&index)
        {
            return Err(malformed("OpenAI stream repeated an output-item start"));
        }
        self.reserve_provider_entry(false)?;
        let serialized = compact_json(&Value::Object(item.clone()));
        self.reserve_tool_bytes(serialized.len())?;
        self.openai_hosted_items.insert(
            index,
            OpenAiHostedItem {
                item_type: item_type.to_string(),
                item_id: item_id.clone(),
                item: serialized.clone(),
            },
        );
        Ok(vec![Event::HostedToolItemStarted {
            output_index: index,
            item_id,
            item_type: item_type.to_string(),
            item: serialized,
        }])
    }

    /// Complete one hosted output item with its final verbatim JSON.
    pub(in crate::dialects) fn openai_hosted_item_done(
        &mut self,
        index: u32,
        done_type: &str,
        item: &Map<String, Value>,
    ) -> Result<Vec<Event>, Failure> {
        let item_id = openai_identity(item, "id", "OpenAI hosted tool item ID")?;
        let state = self
            .openai_hosted_items
            .get_mut(&index)
            .ok_or_else(|| malformed("OpenAI hosted tool item completed before its start"))?;
        if state.item_type != done_type || state.item_id != item_id {
            return Err(malformed(
                "OpenAI hosted tool item changed identity at completion",
            ));
        }
        let serialized = compact_json(&Value::Object(item.clone()));
        let item_type = state.item_type.clone();
        state.item = serialized.clone();
        self.reserve_tool_bytes(serialized.len())?;
        Ok(vec![Event::HostedToolItemCompleted {
            output_index: index,
            item_id,
            item_type,
            item: serialized,
        }])
    }

    /// Pass one per-type hosted lifecycle or delta frame through verbatim.
    pub(in crate::dialects) fn openai_hosted_progress(
        &mut self,
        event_type: &str,
        payload: &Map<String, Value>,
    ) -> Result<Vec<Event>, Failure> {
        let index = openai_index(payload, "output_index", "OpenAI output_index")?;
        let state = self.openai_hosted_items.get(&index).ok_or_else(|| {
            malformed(&format!(
                "OpenAI hosted tool event arrived before its output item ({})",
                bounded_wire_token(event_type),
            ))
        })?;
        if let Some(Value::String(item_id)) = payload.get("item_id") {
            if item_id != &state.item_id {
                return Err(malformed(&format!(
                    "OpenAI hosted tool event changed item identity ({})",
                    bounded_wire_token(event_type),
                )));
            }
        }
        let serialized = compact_json(&Value::Object(payload.clone()));
        let item_id = state.item_id.clone();
        self.reserve_tool_bytes(serialized.len())?;
        Ok(vec![Event::HostedToolItemProgress {
            output_index: index,
            item_id,
            event_type: event_type.to_string(),
            payload: serialized,
        }])
    }

    /// Close every hosted item the terminal reached before its `done` frame.
    ///
    /// The last-seen verbatim JSON is re-served: the provider owns each item
    /// type's status vocabulary, so nothing inside is rewritten.
    pub(in crate::dialects) fn openai_sweep_hosted_items(&mut self) -> Vec<Event> {
        let unfinished: Vec<_> = self
            .openai_hosted_items
            .iter()
            .filter(|(index, _)| !self.openai_completed_output_items.contains(index))
            .map(|(index, state)| {
                (
                    *index,
                    state.item_id.clone(),
                    state.item_type.clone(),
                    state.item.clone(),
                )
            })
            .collect();
        let mut events = Vec::with_capacity(unfinished.len());
        for (output_index, item_id, item_type, item) in unfinished {
            self.openai_completed_output_items.insert(output_index);
            events.push(Event::HostedToolItemCompleted {
                output_index,
                item_id,
                item_type,
                item,
            });
        }
        events
    }

    /// Attach one whole verbatim output-text annotation (URL citations from
    /// hosted web search) to the open assistant message item.
    pub(in crate::dialects) fn openai_text_annotation(
        &mut self,
        payload: &Map<String, Value>,
    ) -> Result<Vec<Event>, Failure> {
        let output_index = openai_index(payload, "output_index", "OpenAI output_index")?;
        let item_id = openai_identity(payload, "item_id", "OpenAI message item ID")?;
        let mut events = Vec::new();
        if self.bind_openai_output_item(
            output_index,
            ProviderOutputItemKind::Message,
            Some(item_id.clone()),
        )? {
            events.push(Event::ProviderOutputItemStarted {
                output_index,
                item_id: Some(item_id.clone()),
                kind: ProviderOutputItemKind::Message,
                status: None,
                phase: None,
            });
        }
        let annotation = payload
            .get("annotation")
            .and_then(Value::as_object)
            .ok_or_else(|| malformed("OpenAI text annotation must be an object"))?;
        let serialized = compact_json(&Value::Object(annotation.clone()));
        self.reserve_tool_bytes(serialized.len())?;
        events.push(Event::ProviderTextAnnotation {
            output_index,
            item_id,
            annotation: serialized,
        });
        Ok(events)
    }
}
