//! Provider-executed tools are irreversible work, not stalled token generation.

use std::collections::HashSet;

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
impl super::UpstreamRelay {
    /// Refresh only generation idle when validated private reasoning arrives.
    pub(super) fn observe_unexposed_reasoning_progress(&mut self) {
        if self.normalizer.take_unexposed_reasoning_progress() {
            self.last_progress_at = Some(std::time::Instant::now());
            self.private_progress();
        }
    }
}
