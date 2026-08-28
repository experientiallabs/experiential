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
            status: None,
            done: false,
        };
        let frame = self.event(
            "response.output_item.added",
            json!({
                "output_index": state.output_index,
                "item": state.item(
                    false,
                    ProviderOutputItemStatus::InProgress,
                    false,
                ),
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
        item_id: Option<&str>,
        kind: ProviderOutputItemKind,
        status: Option<ProviderOutputItemStatus>,
        phase: Option<ProviderAssistantMessagePhase>,
    ) -> Result<Vec<String>, PublicError> {
        if self
            .provider_output_starts
            .contains_key(&provider_output_index)
        {
            return Err(invalid_provider_stream(
                "Responses provider output-item index was started twice.",
            ));
        }
        if kind != ProviderOutputItemKind::Message && phase.is_some() {
            return Err(invalid_provider_stream(
                "Responses provider attached a message phase to a non-message item.",
            ));
        }
        if kind != ProviderOutputItemKind::FunctionCall && item_id.is_none() {
            return Err(invalid_provider_stream(
                "Responses provider output item omitted its required item ID.",
            ));
        }
        let output_index = self.output_order.len();
        self.provider_output_starts.insert(
            provider_output_index,
            ProviderOutputStart {
                item_id: item_id.map(str::to_string),
                kind,
                output_index,
                status,
                phase,
            },
        );
        match kind {
            ProviderOutputItemKind::Reasoning => {
                let state = ReasoningState {
                    item_id: item_id.expect("reasoning ID checked").to_string(),
                    output_index,
                    parts: BTreeMap::new(),
                    encrypted_content: None,
                    status,
                    done: false,
                };
                let frame = self.event(
                    "response.output_item.added",
                    json!({
                        "output_index": output_index,
                        "item": state.item(
                            false,
                            ProviderOutputItemStatus::InProgress,
                            false,
                        ),
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
                let key = MessageKey::Provider(provider_output_index);
                let state = MessageState {
                    item_id: item_id.expect("message ID checked").to_string(),
                    output_index,
                    status,
                    phase,
                    text: String::new(),
                    refusal: String::new(),
                    text_started: false,
                    refusal_started: false,
                    done: false,
                };
                let item = state.item(false, ProviderOutputItemStatus::InProgress);
                self.messages.insert(key, state);
                self.output_order.push(OutputSlot::Message(key));
                Ok(vec![self.event(
                    "response.output_item.added",
                    json!({"output_index": output_index, "item": item}),
                )])
            }
        }
    }

    /// Close one provider-owned output item with exact status and phase.
    pub(super) fn provider_output_item_completed(
        &mut self,
        provider_output_index: u32,
        item_id: Option<&str>,
        kind: ProviderOutputItemKind,
        status: Option<ProviderOutputItemStatus>,
        phase: Option<ProviderAssistantMessagePhase>,
    ) -> Result<Vec<String>, PublicError> {
        let start = self
            .provider_output_starts
            .get(&provider_output_index)
            .ok_or_else(|| {
                invalid_provider_stream("Responses output item completed before its start.")
            })?;
        if start.kind != kind || start.item_id.as_deref() != item_id {
            return Err(invalid_provider_stream(
                "Responses output item changed provider identity at completion.",
            ));
        }
        if kind != ProviderOutputItemKind::Message && phase.is_some() {
            return Err(invalid_provider_stream(
                "Responses provider attached a message phase to a non-message item.",
            ));
        }
        if start.phase.is_some() && phase.is_some() && start.phase != phase {
            return Err(invalid_provider_stream(
                "Responses assistant message changed phase at completion.",
            ));
        }
        match kind {
            ProviderOutputItemKind::Message => {
                let key = MessageKey::Provider(provider_output_index);
                let state = self.messages.get_mut(&key).ok_or_else(|| {
                    invalid_provider_stream("Responses message completed before its start.")
                })?;
                if state.done {
                    return Err(invalid_provider_stream(
                        "Responses message item completed twice.",
                    ));
                }
                state.status = status.or(state.status);
                state.phase = phase.or(state.phase);
                Ok(self.close_message(key, status.unwrap_or(ProviderOutputItemStatus::Completed)))
            }
            ProviderOutputItemKind::Reasoning => {
                let state = self
                    .reasoning
                    .get_mut(&provider_output_index)
                    .ok_or_else(|| {
                        invalid_provider_stream("Responses reasoning completed before its start.")
                    })?;
                if state.done {
                    return Err(invalid_provider_stream(
                        "Responses reasoning item completed twice.",
                    ));
                }
                state.status = status.or(state.status);
                Ok(self.close_reasoning(
                    provider_output_index,
                    status.unwrap_or(ProviderOutputItemStatus::Completed),
                ))
            }
            ProviderOutputItemKind::FunctionCall => {
                if let Some(state) = self.tools.get_mut(&provider_output_index) {
                    state.status = status.or(state.status);
                }
                Ok(Vec::new())
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
        if state.done {
            return Err(invalid_provider_stream(
                "Responses encrypted reasoning arrived after item completion.",
            ));
        }
        state.encrypted_content = Some(encrypted_content.to_string());
        Ok(frames)
    }
}
