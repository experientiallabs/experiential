//! Display reasoning on the Messages surface: readable non-Anthropic reasoning
//! streamed into the gateway's unsigned thinking block, split from
//! `encode_messages` so the implementation stays within the repository line
//! budget.

use super::*;
use crate::reasoning_display::unsigned_thinking_delta;

impl MessagesSseEncoder {
    /// Stream one display-reasoning event into the unsigned thinking block.
    ///
    /// A block that already closed (later output intervened) is never
    /// reopened: a fresh block starts, and its text starts without the
    /// paragraph break that only separates units inside one block.
    pub(super) fn displayed_thinking(&mut self, event: &Event) -> Result<Vec<String>, PublicError> {
        let open = self
            .blocks
            .iter()
            .rposition(|block| block.kind == BlockKind::Thinking(EXPOSED_REASONING_BLOCK_INDEX))
            .is_some_and(|position| {
                self.blocks[position].anthropic_index.is_none()
                    || self.open_position == Some(position)
            });
        if !open {
            self.reasoning_display = DisplayJoiner::default();
        }
        match unsigned_thinking_delta(&mut self.reasoning_display, event, self.reasoning_output) {
            Some(delta) => self.thinking_delta(EXPOSED_REASONING_BLOCK_INDEX, &delta),
            None => Ok(Vec::new()),
        }
    }
}
