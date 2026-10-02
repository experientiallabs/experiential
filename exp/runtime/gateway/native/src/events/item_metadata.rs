//! Provider output-item metadata encoding.

use super::{ProviderAssistantMessagePhase, ProviderOutputItemStatus};
use serde_json::Value;

/// Count hosted invocations only, not results, approvals, listings or opaque items.
pub fn hosted_item_type_is_invocation(item_type: &str) -> bool {
    item_type.ends_with("_call")
}

pub(super) fn add_provider_item_metadata(
    payload: &mut Value,
    item_id: &Option<String>,
    status: Option<ProviderOutputItemStatus>,
    phase: Option<ProviderAssistantMessagePhase>,
) {
    if let Some(item_id) = item_id {
        payload["item_id"] = Value::String(item_id.clone());
    }
    if let Some(status) = status {
        payload["status"] = Value::String(status.as_str().to_string());
    }
    if let Some(phase) = phase {
        payload["phase"] = Value::String(phase.as_str().to_string());
    }
}
