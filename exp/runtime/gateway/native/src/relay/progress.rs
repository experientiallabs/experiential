//! Provider-executed tools are irreversible work, not stalled token generation.
//! A client tool call whose arguments a provider buffers is the same kind of
//! legitimate silence, bounded more generously but never removed.

use std::collections::HashSet;
use std::time::Duration;

use crate::events::{hosted_item_type_is_invocation, Event};

/// Track the provider's active tool calls using its exact result identities.
/// These sets inherit the normalizer's bounded provider-entry ceiling.
#[derive(Default)]
pub(super) struct ProviderTools {
    server_calls: HashSet<String>,
    hosted_items: HashSet<String>,
}

impl ProviderTools {
    pub(super) fn active(&self) -> bool {
        !self.server_calls.is_empty() || !self.hosted_items.is_empty()
    }

    pub(super) fn observe(&mut self, event: &Event) {
        match event {
            Event::ServerToolUseStarted { call_id, .. } => {
                self.server_calls.insert(call_id.clone());
            }
            // Closing the argument block is NOT completion of server work.
            // The result arrives later under another content-block index.
            Event::ServerToolResult { block, .. } => {
                if let Ok(serde_json::Value::Object(result)) = serde_json::from_str(block) {
                    if let Some(id) = result.get("tool_use_id").and_then(|id| id.as_str()) {
                        self.server_calls.remove(id);
                    }
                }
            }
            Event::HostedToolItemStarted {
                item_id, item_type, ..
            } if hosted_item_type_is_invocation(item_type) || item_type == "mcp_list_tools" => {
                // Remote discovery performs provider work without being a
                // billable invocation in the ledger's tool-name vocabulary.
                self.hosted_items.insert(item_id.clone());
            }
            Event::HostedToolItemCompleted { item_id, .. } => {
                self.hosted_items.remove(item_id);
            }
            _ => {}
        }
    }
}

/// How many generation-idle windows an open client tool call's arguments may
/// stay silent. Providers buffer long string arguments on every wire:
/// Anthropic emits a whole file for Claude Code's Write tool only once it is
/// generated, with no pings meanwhile (a large write died at the same byte on
/// every retry, 2026-10-05), and vLLM's DeepSeek tool parser on the OpenAI
/// chat wire holds 800 to 2,800 characters of arguments at a time around
/// escaped quotes (7 to 14 s silences on an idle server; 400 streams across
/// 70 orgs killed in 12 hours, 2026-10-20). The attempt is committed once a
/// call starts, so an earlier verdict could not fail over anyway. Ten windows
/// is 600 s, Claude Code's own client timeout.
const BUFFERED_TOOL_ARGUMENT_IDLE_WINDOWS: u32 = 10;

/// Track the client tool calls whose arguments are still being generated.
#[derive(Default)]
pub(super) struct BufferedToolArguments {
    open: HashSet<u32>,
}

impl BufferedToolArguments {
    pub(super) fn observe(&mut self, event: &Event) {
        match event {
            Event::ToolCallStarted { index, .. } => {
                self.open.insert(*index);
            }
            Event::ToolCallCompleted { index, .. } => {
                self.open.remove(index);
            }
            _ => {}
        }
    }

    /// The generation-idle bound: widened only while a tool call's arguments are open.
    pub(super) fn idle_bound(&self, phase_timeout: Duration) -> Duration {
        if self.open.is_empty() {
            phase_timeout
        } else {
            phase_timeout.saturating_mul(BUFFERED_TOOL_ARGUMENT_IDLE_WINDOWS)
        }
    }
}
