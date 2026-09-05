//! Accumulated per-item output state for the public Responses encoder:
//! reasoning, assistant message, provider-owned start, and hosted-tool item
//! records, plus the ordered public output-slot vocabulary.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use crate::events::{
    ProviderAssistantMessagePhase, ProviderOutputItemKind, ProviderOutputItemStatus,
};

/// One accumulated reasoning item with provider-indexed summary parts and
/// an optional opaque encrypted payload the caller replays verbatim.
pub(crate) struct ReasoningState {
    pub(crate) item_id: String,
    pub(crate) output_index: usize,
    pub(crate) parts: BTreeMap<u32, String>,
    pub(crate) encrypted_content: Option<String>,
    pub(crate) status: Option<ProviderOutputItemStatus>,
    pub(crate) done: bool,
}

impl ReasoningState {
    pub(crate) fn item(
        &self,
        include_content: bool,
        fallback_status: ProviderOutputItemStatus,
        include_encrypted_content: bool,
    ) -> Value {
        let summary: Vec<Value> = if include_content {
            self.parts
                .values()
                .map(|text| json!({"type": "summary_text", "text": text}))
                .collect()
        } else {
            Vec::new()
        };
        let mut item = json!({
            "id": self.item_id,
            "type": "reasoning",
            "summary": summary,
            "status": self.status.unwrap_or(fallback_status).as_str(),
        });
        if include_encrypted_content {
            if let Some(encrypted) = &self.encrypted_content {
                item.as_object_mut()
                    .expect("reasoning item is an object")
                    .insert("encrypted_content".to_string(), json!(encrypted));
            }
        }
        item
    }
}

/// Provider-owned output item reserved before its content-bearing event.
pub(crate) struct ProviderOutputStart {
    pub(crate) item_id: Option<String>,
    pub(crate) kind: ProviderOutputItemKind,
    pub(crate) output_index: usize,
    pub(crate) status: Option<ProviderOutputItemStatus>,
    pub(crate) phase: Option<ProviderAssistantMessagePhase>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MessageKey {
    Synthetic,
    Provider(u32),
}

/// One independently addressable assistant message output item.
pub(crate) struct MessageState {
    pub(crate) item_id: String,
    pub(crate) output_index: usize,
    pub(crate) status: Option<ProviderOutputItemStatus>,
    pub(crate) phase: Option<ProviderAssistantMessagePhase>,
    pub(crate) text: String,
    pub(crate) refusal: String,
    /// Verbatim provider annotations (URL citations from hosted web search)
    /// attached to this message's text part, in arrival order.
    pub(crate) annotations: Vec<Value>,
    pub(crate) text_started: bool,
    pub(crate) refusal_started: bool,
    pub(crate) done: bool,
}

impl MessageState {
    pub(crate) fn item(
        &self,
        include_content: bool,
        fallback_status: ProviderOutputItemStatus,
    ) -> Value {
        let mut content = Vec::new();
        if include_content && self.text_started {
            content.push(json!({
                "type": "output_text",
                "text": self.text,
                "annotations": self.annotations,
            }));
        }
        if include_content && self.refusal_started {
            content.push(json!({"type": "refusal", "refusal": self.refusal}));
        }
        let mut item = json!({
            "id": self.item_id,
            "type": "message",
            "role": "assistant",
            "status": self.status.unwrap_or(fallback_status).as_str(),
            "content": content,
        });
        if let Some(phase) = self.phase {
            item["phase"] = json!(phase.as_str());
        }
        item
    }
}

#[derive(Clone, Copy)]
pub(crate) enum OutputSlot {
    Message(MessageKey),
    Tool(u32),
    Reasoning(u32),
    HostedTool(u32),
    FireworksReasoning,
}

/// One provider-hosted tool output item carried verbatim: the provider owns
/// its shape, so the gateway re-serves the last-seen item JSON untouched.
pub(crate) struct HostedToolState {
    pub(crate) item_id: String,
    pub(crate) output_index: usize,
    pub(crate) item: Value,
    pub(crate) done: bool,
}
