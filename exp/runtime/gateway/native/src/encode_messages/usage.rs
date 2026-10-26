//! The Anthropic usage object rendered on every public Messages frame,
//! split from `encode_messages` so the implementation stays within the
//! repository line budget.

use serde_json::{json, Value};

use super::MessagesSseEncoder;
use crate::events::Usage;
use crate::tool_search::annotate_messages_tool_search_usage;
use crate::web_search::annotate_messages_usage;

impl MessagesSseEncoder {
    /// The `message_start` meters: the upstream's own start usage when known,
    /// else the pre-dispatch estimate in Anthropic's start-frame shape (the
    /// counted prompt as `input_tokens`, both cache legs `0` because nothing
    /// is cached before dispatch, and Anthropic's `output_tokens: 1`
    /// placeholder), else the zero placeholder.
    pub(super) fn start_usage(&self) -> Value {
        let usage = match (self.usage.as_ref(), self.pre_dispatch_input_estimate) {
            (Some(usage), _) => messages_usage(Some(usage)),
            (None, Some(estimate)) => usage_object(estimate, 0, 0, 1),
            (None, None) => messages_usage(None),
        };
        self.metered(usage)
    }

    /// Both gateway meters on one usage object: `server_tool_use` gains
    /// `web_search_requests` and/or `tool_search_requests`, neither when the
    /// gateway ran nothing.
    pub(super) fn metered(&self, usage: Value) -> Value {
        annotate_messages_tool_search_usage(
            annotate_messages_usage(usage, self.web_search_requests()),
            self.tool_search.as_ref().map(|search| search.requests),
        )
    }

    fn web_search_requests(&self) -> Option<u32> {
        self.web_search.as_ref().map(|search| search.requests)
    }
}

/// Anthropic's usage object for every Messages frame that carries one:
/// `message_start.message.usage`, `message_delta.usage`, and the
/// non-streamed body's `usage`. All four token legs are always present,
/// `0` when the provider reported none, because Anthropic clients (the
/// official SDK accumulators, Claude Code's context meter) read the cache
/// legs by key and treat an absent key as "not Anthropic's shape".
///
/// Mapping from the normalized `Usage` (whose `input_tokens` is the FOLDED
/// total the ledger bills: uncached + cache reads + cache writes):
///
/// | Anthropic field                | source                                              |
/// |--------------------------------|-----------------------------------------------------|
/// | `input_tokens`                 | `input_tokens - cached_input_tokens - cache_creation_input_tokens` (uncached input, saturating) |
/// | `cache_creation_input_tokens`  | `cache_creation_input_tokens`, else `0` (Anthropic or Bedrock rungs) |
/// | `cache_read_input_tokens`      | `cached_input_tokens`, else `0` (Anthropic `cache_read_input_tokens`, OpenAI-wire `prompt_tokens_details.cached_tokens` / `input_tokens_details.cached_tokens`, Gemini `cachedContentTokenCount`, Bedrock `cacheReadInputTokens`) |
/// | `output_tokens`                | `output_tokens` (reasoning folded in where the provider bills it additively) |
///
/// When the provider reported the write lifetimes, `cache_creation` carries
/// Anthropic's own `ephemeral_5m_input_tokens` / `ephemeral_1h_input_tokens`
/// split, so a caller pricing writes from this object charges each lifetime
/// at its own rate. The split is omitted when it was not observed (it is
/// never inferred) and when a 12-hour subset exists, which Anthropic's two
/// lifetimes cannot express.
///
/// Unknown usage (no provider report) renders every leg as `0`.
pub(crate) fn messages_usage(usage: Option<&Usage>) -> Value {
    let usage = match usage {
        Some(usage) if usage.has_token_counts() => usage,
        _ => return usage_object(0, 0, 0, 0),
    };
    let cached = usage.cached_input_tokens.unwrap_or(0);
    let creation = usage.cache_creation_input_tokens.unwrap_or(0);
    // Both cache legs come back out of the folded ledger total so callers
    // see the provider's own shape: input_tokens excludes cached reads and
    // cache writes, each reported on its own leg.
    let mut object = usage_object(
        usage
            .input_tokens
            .unwrap_or(0)
            .saturating_sub(cached)
            .saturating_sub(creation),
        creation,
        cached,
        usage.output_tokens.unwrap_or(0),
    );
    if let Some(split) = write_lifetimes(usage, creation) {
        object["cache_creation"] = split;
    }
    object
}

/// Anthropic's `cache_creation` lifetime object, present only when the
/// one-hour subset was observed and no 12-hour subset exists.
fn write_lifetimes(usage: &Usage, creation: u64) -> Option<Value> {
    if usage.cache_creation_12h_input_tokens.is_some() {
        return None;
    }
    let hour = usage.cache_creation_1h_input_tokens?.min(creation);
    Some(json!({
        "ephemeral_5m_input_tokens": creation - hour,
        "ephemeral_1h_input_tokens": hour,
    }))
}

/// The four-leg Anthropic usage object in Anthropic's own field order.
pub(super) fn usage_object(input: u64, cache_creation: u64, cache_read: u64, output: u64) -> Value {
    json!({
        "input_tokens": input,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
        "output_tokens": output,
    })
}
