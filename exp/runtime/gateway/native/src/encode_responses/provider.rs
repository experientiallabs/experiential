//! Provider-owned Responses output identity and ordering helpers.

use super::*;

impl ResponsesSseEncoder {
    /// Create one stable reasoning output item on first use.
    pub(super) fn ensure_reasoning(
        &mut self,
        provider_output_index: u32,
        item_id: &str,
        frames: &mut Vec<String>,
    ) -> Result<(), PublicError> {
        if let Some(existing) = self.reasoning.get(&provider_output_index) {
            return if existing.item_id == item_id {
                Ok(())
            } else {
                Err(invalid_provider_stream(
                    "Responses reasoning item changed provider identity.",
                ))
            };
        }
        let state = ReasoningState {
            item_id: item_id.to_string(),
            output_index: self.output_order.len(),
            parts: BTreeMap::new(),
            encrypted_content: None,
        };
        let frame = self.event(
            "response.output_item.added",
            json!({
                "output_index": state.output_index,
                "item": state.item(false, false),
            }),
        );
        self.reasoning.insert(provider_output_index, state);
        self.output_order
            .push(OutputSlot::Reasoning(provider_output_index));
        frames.push(frame);
        Ok(())
    }

    /// Reserve one provider-owned output item in start order.
    pub(super) fn provider_output_item_started(
        &mut self,
        provider_output_index: u32,
        item_id: &str,
        kind: ProviderOutputItemKind,
    ) -> Result<Vec<String>, PublicError> {
        if self
            .provider_output_starts
            .contains_key(&provider_output_index)
        {
            return Err(invalid_provider_stream(
                "Responses provider output-item index was started twice.",
            ));
        }
        let output_index = self.output_order.len();
        self.provider_output_starts.insert(
            provider_output_index,
            ProviderOutputStart {
                item_id: item_id.to_string(),
                kind,
                output_index,
            },
        );
        match kind {
            ProviderOutputItemKind::Reasoning => {
                let state = ReasoningState {
                    item_id: item_id.to_string(),
                    output_index,
                    parts: BTreeMap::new(),
                    encrypted_content: None,
                };
                let frame = self.event(
                    "response.output_item.added",
                    json!({
                        "output_index": output_index,
                        "item": state.item(false, false),
                    }),
                );
                self.reasoning.insert(provider_output_index, state);
                self.output_order
                    .push(OutputSlot::Reasoning(provider_output_index));
                Ok(vec![frame])
            }
            ProviderOutputItemKind::FunctionCall => {
                self.output_order
                    .push(OutputSlot::Tool(provider_output_index));
                Ok(Vec::new())
            }
            ProviderOutputItemKind::Message => {
                if self.message_output_index.is_some() {
                    return Err(invalid_provider_stream(
                        "Responses provider emitted more than one assistant message item.",
                    ));
                }
                self.message_id = item_id.to_string();
                self.message_output_index = Some(output_index);
                self.output_order.push(OutputSlot::Message);
                Ok(vec![self.event(
                    "response.output_item.added",
                    json!({
                        "output_index": output_index,
                        "item": self.message_item(false),
                    }),
                )])
            }
        }
    }

    /// Retain one opaque encrypted reasoning payload on its output item.
    pub(super) fn encrypted_reasoning(
        &mut self,
        provider_output_index: u32,
        item_id: &str,
        encrypted_content: &str,
    ) -> Result<Vec<String>, PublicError> {
        let mut frames = Vec::new();
        self.ensure_reasoning(provider_output_index, item_id, &mut frames)?;
        let state = self
            .reasoning
            .get_mut(&provider_output_index)
            .expect("reasoning state just ensured");
        state.encrypted_content = Some(encrypted_content.to_string());
        Ok(frames)
    }
}
