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
        if !matches!(
            kind,
            ProviderOutputItemKind::FunctionCall | ProviderOutputItemKind::CustomToolCall
        ) && item_id.is_none()
        {
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
            ProviderOutputItemKind::FunctionCall | ProviderOutputItemKind::CustomToolCall => {
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
                    annotations: Vec::new(),
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
            ProviderOutputItemKind::FunctionCall | ProviderOutputItemKind::CustomToolCall => {
                if let Some(state) = self.tools.get_mut(&provider_output_index) {
                    state.status = status.or(state.status);
                }
                Ok(Vec::new())
            }
        }
    }

    /// Open one verbatim hosted tool output item at the next public slot.
    pub(super) fn hosted_started(
        &mut self,
        provider_output_index: u32,
        item: &str,
    ) -> Result<Vec<String>, PublicError> {
        if self.hosted.contains_key(&provider_output_index) {
            return Err(invalid_provider_stream(
                "Responses hosted tool item was started twice.",
            ));
        }
        let parsed: Value = serde_json::from_str(item).map_err(|_| {
            invalid_provider_stream("Responses hosted tool item is not valid JSON.")
        })?;
        let item_id = parsed
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                invalid_provider_stream("Responses hosted tool item omitted its item ID.")
            })?
            .to_string();
        let output_index = self.output_order.len();
        self.hosted.insert(
            provider_output_index,
            HostedToolState {
                item_id,
                output_index,
                item: parsed.clone(),
                done: false,
            },
        );
        self.output_order
            .push(OutputSlot::HostedTool(provider_output_index));
        Ok(vec![self.event(
            "response.output_item.added",
            json!({"output_index": output_index, "item": parsed}),
        )])
    }

    /// Re-emit one verbatim hosted tool progress frame at the public index.
    ///
    /// Only the public output index and sequence number are re-stamped; every
    /// provider payload field passes through untouched.
    pub(super) fn hosted_progress(
        &mut self,
        provider_output_index: u32,
        event_type: &str,
        payload: &str,
    ) -> Result<Vec<String>, PublicError> {
        let output_index = {
            let state = self.hosted.get(&provider_output_index).ok_or_else(|| {
                invalid_provider_stream("Responses hosted tool event arrived before its item.")
            })?;
            if state.done {
                return Err(invalid_provider_stream(
                    "Responses hosted tool event arrived after item completion.",
                ));
            }
            state.output_index
        };
        let mut fields = match serde_json::from_str::<Value>(payload) {
            Ok(Value::Object(fields)) => fields,
            _ => {
                return Err(invalid_provider_stream(
                    "Responses hosted tool progress payload is not a JSON object.",
                ))
            }
        };
        // `event` stamps the gateway's own type and monotonic sequence.
        fields.remove("type");
        fields.remove("sequence_number");
        fields.insert("output_index".to_string(), json!(output_index));
        Ok(vec![self.event(event_type, Value::Object(fields))])
    }

    /// Complete one hosted tool item with its final verbatim JSON.
    pub(super) fn hosted_completed(
        &mut self,
        provider_output_index: u32,
        item: &str,
    ) -> Result<Vec<String>, PublicError> {
        let parsed: Value = serde_json::from_str(item).map_err(|_| {
            invalid_provider_stream("Responses hosted tool item is not valid JSON.")
        })?;
        let output_index = {
            let state = self.hosted.get_mut(&provider_output_index).ok_or_else(|| {
                invalid_provider_stream("Responses hosted tool item completed before its start.")
            })?;
            if state.done {
                return Err(invalid_provider_stream(
                    "Responses hosted tool item completed twice.",
                ));
            }
            if parsed.get("id").and_then(Value::as_str) != Some(state.item_id.as_str()) {
                return Err(invalid_provider_stream(
                    "Responses hosted tool item changed provider identity.",
                ));
            }
            state.item = parsed.clone();
            state.done = true;
            state.output_index
        };
        Ok(vec![self.event(
            "response.output_item.done",
            json!({"output_index": output_index, "item": parsed}),
        )])
    }

    /// Close one still-open hosted tool item with its last-seen verbatim JSON.
    pub(super) fn close_hosted(&mut self, provider_output_index: u32) -> Vec<String> {
        let (output_index, item) = {
            let state = match self.hosted.get_mut(&provider_output_index) {
                Some(state) => state,
                None => return Vec::new(),
            };
            if state.done {
                return Vec::new();
            }
            state.done = true;
            (state.output_index, state.item.clone())
        };
        vec![self.event(
            "response.output_item.done",
            json!({"output_index": output_index, "item": item}),
        )]
    }

    /// Attach one verbatim provider annotation to the open message item.
    pub(super) fn text_annotation(
        &mut self,
        provider_output_index: u32,
        item_id: &str,
        annotation: &str,
    ) -> Result<Vec<String>, PublicError> {
        let key = MessageKey::Provider(provider_output_index);
        let mut frames = Vec::new();
        self.ensure_message(key, Some(item_id), &mut frames)?;
        let parsed: Value = serde_json::from_str(annotation)
            .map_err(|_| invalid_provider_stream("Responses text annotation is not valid JSON."))?;
        let (item_id, output_index, annotation_index, start_part) = {
            let state = self.messages.get_mut(&key).expect("message just ensured");
            if state.done {
                return Err(invalid_provider_stream(
                    "Responses annotation arrived after message completion.",
                ));
            }
            if state.refusal_started {
                return Err(invalid_provider_stream(
                    "Responses annotation cannot attach to a refusal part.",
                ));
            }
            let start_part = !state.text_started;
            state.text_started = true;
            state.annotations.push(parsed.clone());
            (
                state.item_id.clone(),
                state.output_index,
                state.annotations.len() - 1,
                start_part,
            )
        };
        if start_part {
            frames.push(self.event(
                "response.content_part.added",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                }),
            ));
        }
        frames.push(self.event(
            "response.output_text.annotation.added",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "annotation_index": annotation_index,
                "annotation": parsed,
            }),
        ));
        Ok(frames)
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
