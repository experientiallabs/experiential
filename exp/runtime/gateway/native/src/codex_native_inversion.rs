//! Response-side inversion of the Codex native-tool translation.
//!
//! When a Codex CLI Responses request is served on a NON-native provider wire
//! (Chat Completions / Anthropic Messages), the Python request shaping in
//! `exp/runtime/models/providers/codex_tools.py` rewrites the caller's native
//! `namespace` / `custom` tool declarations into ordinary function tools and
//! carries an inverse map (`GatewayRequest.native_tool_translation`, keyed by
//! the provider-facing mangled name) so the response can be reshaped into the
//! items the Codex caller declared.
//!
//! The provider therefore returns plain `function_call`s whose names are the
//! mangled provider names and whose arguments are JSON. This module inverts
//! each tool-call event at the single relay yield point, BEFORE the encoder,
//! so the started and completed events stay byte-consistent (the encoder's
//! `tool_completed` verifies they match) and the existing encoder emits the
//! native shape:
//!
//! * a namespaced function tool -> its original name with `namespace` restored;
//! * a `custom` (freeform-grammar) tool -> `custom = true` with its freeform
//!   `input` unwrapped from the JSON `{"input": "..."}` the function shape
//!   forced, so the encoder renders a `custom_tool_call`.
//!
//! An ordinary function tool (no map entry) is left untouched, and the map is
//! empty on every native-Responses route, so that path is byte-for-byte
//! unchanged.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::events::Event;

/// Provider-facing mangled name -> (origin name, origin namespace, is custom).
///
/// This is the exact JSON shape produced by
/// `NativeToolMapping.as_dict()` on the Python side (a tuple serializes to a
/// three-element array; `serde` reads it back into this tuple).
pub type NativeToolTranslation = HashMap<String, (String, Option<String>, bool)>;

/// Unwrap the freeform input a translated `custom` tool carries.
///
/// The tool was presented to the model as a function with a single required
/// `input` string, so a well-formed call is `{"input": "<freeform text>"}`.
/// The freeform text is returned verbatim. Anything else (malformed JSON, a
/// missing or non-string `input`) falls back to the raw argument text so a
/// misbehaving model degrades to passing its bytes through rather than losing
/// the call.
fn unwrap_custom_input(raw_arguments: &str) -> String {
    match serde_json::from_str::<Value>(raw_arguments) {
        Ok(Value::Object(map)) => match map.get("input") {
            Some(Value::String(input)) => input.clone(),
            _ => raw_arguments.to_string(),
        },
        _ => raw_arguments.to_string(),
    }
}

/// Rewrite one tool-call event in place using the translation map.
///
/// A no-op for every event that is not a tool call, and for any tool call
/// whose (mangled) name is absent from the map. Both the started and completed
/// events are rewritten identically so downstream consistency checks hold.
pub fn invert_tool_event(event: &mut Event, translation: &NativeToolTranslation) {
    if translation.is_empty() {
        return;
    }
    match event {
        Event::ToolCallStarted {
            name,
            namespace,
            custom,
            ..
        } => {
            if let Some((origin_name, origin_namespace, is_custom)) = translation.get(name) {
                *name = origin_name.clone();
                *namespace = origin_namespace.clone();
                *custom = *is_custom;
            }
        }
        Event::ToolCallCompleted { call, .. } => {
            if let Some((origin_name, origin_namespace, is_custom)) = translation.get(&call.name) {
                call.name = origin_name.clone();
                call.namespace = origin_namespace.clone();
                if *is_custom {
                    call.custom = true;
                    call.raw_arguments = unwrap_custom_input(&call.raw_arguments);
                }
            }
        }
        _ => {}
    }
}

/// Per-response translation state. Wrapped custom arguments cannot be emitted
/// before the JSON string is complete, including escape sequences. The normalizer
/// already retains the full call; hold only its index here, then emit the unwrapped
/// bytes once before completion. Ordinary tools retain their incremental deltas.
#[derive(Default)]
pub struct NativeToolInverter {
    pub translation: NativeToolTranslation,
    custom_calls: HashSet<u32>,
}

impl NativeToolInverter {
    pub fn filter(&mut self, mut event: Event) -> Vec<Event> {
        if let Event::ToolCallStarted { index, name, .. } = &event {
            if self.translation.get(name).is_some_and(|entry| entry.2) {
                self.custom_calls.insert(*index);
            }
        }
        if let Event::ToolArgumentsDelta { index, .. } = &event {
            if self.custom_calls.contains(index) {
                return Vec::new();
            }
        }
        invert_tool_event(&mut event, &self.translation);
        if let Event::ToolCallCompleted { index, call } = &event {
            if self.custom_calls.remove(index) {
                return vec![
                    Event::ToolArgumentsDelta {
                        index: *index,
                        delta: call.raw_arguments.clone(),
                    },
                    event,
                ];
            }
        }
        vec![event]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{CompletedToolCall, Event};

    fn translation() -> NativeToolTranslation {
        let mut map = NativeToolTranslation::new();
        // A namespaced function tool: hoisted to `multi_agent_v1__close_agent`.
        map.insert(
            "multi_agent_v1__close_agent".to_string(),
            (
                "close_agent".to_string(),
                Some("multi_agent_v1".to_string()),
                false,
            ),
        );
        // A freeform custom tool (apply_patch): presented as a function.
        map.insert(
            "apply_patch".to_string(),
            ("apply_patch".to_string(), None, true),
        );
        map
    }

    fn completed(name: &str, raw_arguments: &str) -> Event {
        Event::ToolCallCompleted {
            index: 0,
            call: CompletedToolCall {
                call_id: "call-1".to_string(),
                name: name.to_string(),
                namespace: None,
                caller: None,
                provider_item_id: None,
                provider_status: None,
                raw_arguments: raw_arguments.to_string(),
                custom: false,
            },
        }
    }

    #[test]
    fn namespaced_started_regains_its_namespace_and_name() {
        let map = translation();
        let mut event = Event::ToolCallStarted {
            custom: false,
            index: 0,
            call_id: "call-1".to_string(),
            name: "multi_agent_v1__close_agent".to_string(),
            namespace: None,
            caller: None,
        };
        invert_tool_event(&mut event, &map);
        match event {
            Event::ToolCallStarted {
                name, namespace, ..
            } => {
                assert_eq!(name, "close_agent");
                assert_eq!(namespace.as_deref(), Some("multi_agent_v1"));
            }
            _ => panic!("expected ToolCallStarted"),
        }
    }

    #[test]
    fn namespaced_completed_regains_name_and_namespace_and_stays_a_function() {
        let map = translation();
        let mut event = completed("multi_agent_v1__close_agent", "{\"id\":\"a\"}");
        invert_tool_event(&mut event, &map);
        match event {
            Event::ToolCallCompleted { call, .. } => {
                assert_eq!(call.name, "close_agent");
                assert_eq!(call.namespace.as_deref(), Some("multi_agent_v1"));
                assert!(!call.custom);
                assert_eq!(call.raw_arguments, "{\"id\":\"a\"}");
            }
            _ => panic!("expected ToolCallCompleted"),
        }
    }

    #[test]
    fn custom_completed_unwraps_input_and_becomes_custom() {
        let map = translation();
        let mut event = completed(
            "apply_patch",
            "{\"input\": \"*** Begin Patch\\n*** End Patch\"}",
        );
        invert_tool_event(&mut event, &map);
        match event {
            Event::ToolCallCompleted { call, .. } => {
                assert_eq!(call.name, "apply_patch");
                assert!(call.custom);
                assert_eq!(call.raw_arguments, "*** Begin Patch\n*** End Patch");
            }
            _ => panic!("expected ToolCallCompleted"),
        }
    }

    #[test]
    fn custom_completed_with_malformed_arguments_passes_bytes_through() {
        let map = translation();
        let mut event = completed("apply_patch", "not json at all");
        invert_tool_event(&mut event, &map);
        match event {
            Event::ToolCallCompleted { call, .. } => {
                assert!(call.custom);
                assert_eq!(call.raw_arguments, "not json at all");
            }
            _ => panic!("expected ToolCallCompleted"),
        }
    }

    #[test]
    fn unmapped_tool_call_is_untouched() {
        let map = translation();
        let mut event = completed("exec_command", "{\"cmd\":[\"ls\"]}");
        invert_tool_event(&mut event, &map);
        match event {
            Event::ToolCallCompleted { call, .. } => {
                assert_eq!(call.name, "exec_command");
                assert!(!call.custom);
                assert_eq!(call.raw_arguments, "{\"cmd\":[\"ls\"]}");
            }
            _ => panic!("expected ToolCallCompleted"),
        }
    }

    #[test]
    fn empty_translation_is_a_no_op() {
        let map = NativeToolTranslation::new();
        let mut event = completed("apply_patch", "{\"input\":\"x\"}");
        invert_tool_event(&mut event, &map);
        match event {
            Event::ToolCallCompleted { call, .. } => {
                assert_eq!(call.name, "apply_patch");
                assert!(!call.custom);
                assert_eq!(call.raw_arguments, "{\"input\":\"x\"}");
            }
            _ => panic!("expected ToolCallCompleted"),
        }
    }
}
