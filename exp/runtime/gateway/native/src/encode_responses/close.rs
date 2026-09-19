//! Closing the assistant message items of the public Responses encoder,
//! split from `encode_responses` so the implementation stays within the
//! repository line budget.

use super::*;

impl ResponsesSseEncoder {
    /// Emit content and output completion for one assistant message.
    pub(super) fn close_message(
        &mut self,
        key: MessageKey,
        fallback_status: ProviderOutputItemStatus,
    ) -> Vec<String> {
        let (
            item_id,
            output_index,
            text,
            refusal,
            annotations,
            first_search_citation,
            text_started,
            refusal_started,
            item,
        ) = {
            let state = match self.messages.get_mut(&key) {
                Some(state) => state,
                None => return Vec::new(),
            };
            if state.done {
                return Vec::new();
            }
            state.done = true;
            if matches!(
                state.status,
                None | Some(ProviderOutputItemStatus::InProgress)
            ) {
                state.status = Some(fallback_status);
            }
            // The gateway's own search cites the synthetic (foreign-rung)
            // message it answered from; provider-keyed messages carry only
            // the provider's own annotations.
            let first_search_citation = state.annotations.len();
            if let (MessageKey::Synthetic, Some(web_search)) = (key, self.web_search.as_ref()) {
                state.annotations.extend(url_citations(
                    &state.text,
                    &web_search.results,
                    CitationShape::Responses,
                ));
            }
            (
                state.item_id.clone(),
                state.output_index,
                state.text.clone(),
                state.refusal.clone(),
                state.annotations.clone(),
                first_search_citation,
                state.text_started,
                state.refusal_started,
                state.item(true, fallback_status),
            )
        };
        let mut frames: Vec<String> = Vec::new();
        let mut content_index = 0;
        if text_started {
            for (offset, annotation) in annotations[first_search_citation..].iter().enumerate() {
                frames.push(self.event(
                    "response.output_text.annotation.added",
                    json!({
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": content_index,
                        "annotation_index": first_search_citation + offset,
                        "annotation": annotation,
                    }),
                ));
            }
            frames.push(self.event(
                "response.output_text.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "text": text,
                    "logprobs": [],
                }),
            ));
            let part = json!({"type": "output_text", "text": text, "annotations": annotations});
            frames.push(self.event(
                "response.content_part.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": part,
                }),
            ));
            content_index += 1;
        }
        if refusal_started {
            frames.push(self.event(
                "response.refusal.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "refusal": refusal,
                }),
            ));
            let part = json!({"type": "refusal", "refusal": refusal});
            frames.push(self.event(
                "response.content_part.done",
                json!({
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": part,
                }),
            ));
        }
        frames.push(self.event(
            "response.output_item.done",
            json!({
                "output_index": output_index,
                "item": item,
            }),
        ));
        frames
    }
}
