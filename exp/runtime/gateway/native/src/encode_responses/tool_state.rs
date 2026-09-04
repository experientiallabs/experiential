//! One accumulated public tool-call output item (function or freeform
//! custom), split from `encode_responses.rs` for the module line budget.

use serde_json::{json, Value};

use crate::events::ProviderOutputItemStatus;

pub(super) struct ToolState {
    pub(super) item_id: Option<String>,
    pub(super) output_index: usize,
    pub(super) call_id: String,
    pub(super) name: String,
    /// Nested tool tree (Responses `namespace`) that declared this call;
    /// re-emitted verbatim so the caller can round-trip the item (the
    /// provider rejects a namespaced call replayed without it).
    pub(super) namespace: Option<String>,
    pub(super) arguments: String,
    pub(super) status: Option<ProviderOutputItemStatus>,
    pub(super) done: bool,
    /// Freeform custom tool call: renders the `custom_tool_call` item type
    /// with opaque `input` text and the custom input event names.
    pub(super) custom: bool,
}

impl ToolState {
    /// The current official Responses tool-call item.
    pub(super) fn item(&self, fallback_status: ProviderOutputItemStatus) -> Value {
        let (item_type, payload_key) = if self.custom {
            ("custom_tool_call", "input")
        } else {
            ("function_call", "arguments")
        };
        // Field order matches the provider envelope (id first when present);
        // the committed goldens pin it byte-for-byte.
        let mut item = serde_json::Map::new();
        if let Some(item_id) = &self.item_id {
            item.insert("id".to_string(), json!(item_id));
        }
        item.insert("type".to_string(), json!(item_type));
        item.insert("call_id".to_string(), json!(self.call_id));
        item.insert("name".to_string(), json!(self.name));
        if let Some(namespace) = &self.namespace {
            item.insert("namespace".to_string(), json!(namespace));
        }
        item.insert(payload_key.to_string(), json!(self.arguments));
        item.insert(
            "status".to_string(),
            json!(self.status.unwrap_or(fallback_status).as_str()),
        );
        Value::Object(item)
    }
}
