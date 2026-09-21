//! Tool blocks whose stop reason arrives later (Anthropic `message_delta`,
//! Bedrock `messageStop`). Missing arguments stay pending until a normal final
//! reason authorizes the zero-argument seed. Complete argument objects still
//! finish immediately. A mid-fragment call is dropped as incomplete; a syntax
//! error is held until the stop reason, forgiven for a declared budget cut and
//! otherwise surfaced as malformed.

use super::{complete_streamed_tool, drop_cut_call, Normalizer};
use crate::errors::Failure;
use crate::events::{Event, ToolAccumulator};

impl Normalizer {
    /// Complete a nonempty tool at its block stop before the final reason.
    /// Missing arguments remain open; a parse failure is held rather than
    /// raised because the final reason may declare budget truncation.
    pub(super) fn complete_tool_deferring_failure(
        &mut self,
        index: u32,
        tool: &mut ToolAccumulator,
        events: &mut Vec<Event>,
    ) {
        // Missing argument bytes can mean a legitimate zero-argument call or
        // a budget cut before any input arrived. Only the final reason can
        // authorize the empty-object seed; keep nonempty calls streaming.
        if !tool.custom && tool.raw_arguments.is_empty() {
            return;
        }
        if drop_cut_call(tool, "block_stop") {
            self.dropped_cut_call = true;
            return;
        }
        let mut tool_events = Vec::new();
        match complete_streamed_tool(index, tool, &mut tool_events) {
            Ok(()) => events.extend(tool_events),
            Err(failure) => {
                if self.deferred_tool_failure.is_none() {
                    self.deferred_tool_failure = Some(failure);
                }
            }
        }
    }

    /// Resolve a held tool-argument failure at the terminal: a provider-declared
    /// budget truncation forgives it (the unfinished call was dropped), anything
    /// else surfaces it now.
    pub(super) fn resolve_deferred_tool_failure(&mut self, truncated: bool) -> Result<(), Failure> {
        match self.deferred_tool_failure.take() {
            Some(failure) if !truncated => Err(failure),
            _ => Ok(()),
        }
    }
}
