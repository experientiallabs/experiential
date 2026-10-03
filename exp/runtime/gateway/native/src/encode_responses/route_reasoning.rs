//! The route-bound reasoning item of the public Responses encoder: the one
//! `reasoning` output item a reasoning-content route (Fireworks, Hunyuan, or a
//! declared `reasoning_content_native` origin) opens, split from
//! `encode_responses` so the implementation stays within the repository line
//! budget.
//!
//! The item always carries the sealed tool-turn carrier as `encrypted_content`.
//! On a rung marked `reasoning_output_exposed` it also streams the model's
//! plaintext reasoning as one `summary_text` part, the Responses twin of the
//! Chat wire's `reasoning_content` and the Messages thinking block; elsewhere
//! the plaintext never leaves the gateway.

use super::*;

/// The one summary part an exposed route reasoning item streams into.
const EXPOSED_SUMMARY_INDEX: u32 = 0;

impl ResponsesSseEncoder {
    /// Open the route reasoning item on its first delta and, on an exposed
    /// rung, stream the delta as summary text.
    pub(super) fn route_reasoning(
        &mut self,
        route_sha256: &str,
        delta: &str,
    ) -> Result<Vec<String>, PublicError> {
        if let Some(existing) = &self.fireworks_reasoning_route_sha256 {
            if existing != route_sha256 {
                return Err(invalid_provider_stream(
                    "Responses Fireworks reasoning changed provider route.",
                ));
            }
        } else {
            self.fireworks_reasoning_route_sha256 = Some(route_sha256.to_string());
        }
        let mut frames = Vec::new();
        if self.fireworks_reasoning.is_none() {
            let state = ReasoningState {
                item_id: stable_public_id("rs", &format!("{}:fireworks", self.response_id)),
                output_index: self.output_order.len(),
                parts: BTreeMap::new(),
                encrypted_content: None,
                status: Some(ProviderOutputItemStatus::InProgress),
                done: false,
            };
            frames.push(self.event(
                "response.output_item.added",
                json!({
                    "output_index": state.output_index,
                    "item": state.item(
                        false,
                        ProviderOutputItemStatus::InProgress,
                        false,
                    ),
                }),
            ));
            self.fireworks_reasoning = Some(state);
            self.output_order.push(OutputSlot::FireworksReasoning);
        }
        if !self.envelope.reasoning_output_exposed || delta.is_empty() {
            return Ok(frames);
        }
        let (item_id, output_index, new_part) = {
            let state = self
                .fireworks_reasoning
                .as_mut()
                .expect("route reasoning state just opened");
            let new_part = !state.parts.contains_key(&EXPOSED_SUMMARY_INDEX);
            state
                .parts
                .entry(EXPOSED_SUMMARY_INDEX)
                .or_default()
                .push_str(delta);
            (state.item_id.clone(), state.output_index, new_part)
        };
        if new_part {
            frames.push(self.event(
                "response.reasoning_summary_part.added",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": EXPOSED_SUMMARY_INDEX,
                    "part": {"type": "summary_text", "text": ""},
                }),
            ));
        }
        frames.push(self.event(
            "response.reasoning_summary_text.delta",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "summary_index": EXPOSED_SUMMARY_INDEX,
                "delta": delta,
            }),
        ));
        Ok(frames)
    }

    /// Complete the route reasoning item, closing its exposed summary part.
    pub(super) fn close_route_reasoning(
        &mut self,
        fallback_status: ProviderOutputItemStatus,
    ) -> Vec<String> {
        let Some(state) = self.fireworks_reasoning.as_mut() else {
            return Vec::new();
        };
        if state.done {
            return Vec::new();
        }
        state.done = true;
        state.status = Some(fallback_status);
        let item_id = state.item_id.clone();
        let output_index = state.output_index;
        let parts = state.parts.clone();
        let item = state.item(
            true,
            fallback_status,
            self.envelope.include_encrypted_reasoning,
        );
        let mut frames = Vec::new();
        for (summary_index, text) in parts {
            frames.push(self.event(
                "response.reasoning_summary_text.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": summary_index,
                    "text": text,
                }),
            ));
            frames.push(self.event(
                "response.reasoning_summary_part.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": summary_index,
                    "part": {"type": "summary_text", "text": text},
                }),
            ));
        }
        frames.push(self.event(
            "response.output_item.done",
            json!({"output_index": output_index, "item": item}),
        ));
        frames
    }
}
