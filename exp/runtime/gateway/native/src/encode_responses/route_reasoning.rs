//! The route-bound reasoning item of the public Responses encoder: the one
//! `reasoning` output item a reasoning-content route (Fireworks, Hunyuan, or a
//! declared `reasoning_content_native` origin) opens, split from
//! `encode_responses` so the implementation stays within the repository line
//! budget.
//!
//! The item always carries the sealed tool-turn carrier as `encrypted_content`.
//! On a rung marked `reasoning_output_exposed`, or one whose reasoning display
//! is on, it also streams the model's plaintext reasoning as one
//! `summary_text` part, the Responses twin of the Chat wire's reasoning fields
//! and the Messages thinking block; on an opted-out rung the plaintext never
//! leaves the gateway. Display-only plaintext from an origin with no replay
//! route streams into its own carrier-free reasoning item.

use super::*;

/// The one summary part an exposed route reasoning item streams into.
const EXPOSED_SUMMARY_INDEX: u32 = 0;

/// Reasoning-map key of the first item that carries display-only plaintext
/// reasoning (OpenAI-compatible origins without a replay route, Gemini thought
/// summaries); each resumed segment counts down from it. Provider output
/// indices and Anthropic block indices never reach this range.
const DISPLAYED_REASONING_OUTPUT_INDEX: u32 = u32::MAX - 1;

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
        let shown = self.envelope.reasoning_output_exposed || self.envelope.reasoning_displayed;
        if !shown || delta.is_empty() {
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

    /// Stream display-only plaintext reasoning as summary text of one
    /// reasoning item, on a rung whose reasoning display is on. The item has
    /// no encrypted content, so a replayed copy carries nothing to the
    /// provider.
    pub(super) fn displayed_reasoning(&mut self, delta: &str) -> Result<Vec<String>, PublicError> {
        if !self.envelope.reasoning_displayed || delta.is_empty() {
            return Ok(Vec::new());
        }
        // An `item_` id marks the item as the gateway's own: a caller that
        // echoes it back has it dropped instead of sent to a provider that
        // never issued it.
        // Reasoning that resumes after another output item opens a fresh item,
        // so the response keeps provider order.
        let key = match self.displayed_reasoning_key {
            Some(key) if matches!(self.output_order.last(), Some(OutputSlot::Reasoning(last)) if *last == key) => {
                key
            }
            Some(key) => key - 1,
            None => DISPLAYED_REASONING_OUTPUT_INDEX,
        };
        self.displayed_reasoning_key = Some(key);
        let item_id = stable_public_id("item", &format!("{}:reasoning:{key}", self.response_id));
        self.reasoning_summary_delta(key, 0, &item_id, delta)
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
