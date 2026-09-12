//! The Anthropic usage object rendered on every public Messages frame,
//! split from `encode_messages` so the implementation stays within the
//! repository line budget.

use serde_json::{json, Value};

use crate::events::Usage;

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
/// | `cache_creation_input_tokens`  | `cache_creation_input_tokens`, else `0` (Anthropic-wire rungs only) |
/// | `cache_read_input_tokens`      | `cached_input_tokens`, else `0` (Anthropic `cache_read_input_tokens`, OpenAI-wire `prompt_tokens_details.cached_tokens` / `input_tokens_details.cached_tokens`, Gemini `cachedContentTokenCount`, Bedrock `cacheReadInputTokens`) |
/// | `output_tokens`                | `output_tokens` (reasoning folded in where the provider bills it additively) |
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
    usage_object(
        usage
            .input_tokens
            .unwrap_or(0)
            .saturating_sub(cached)
            .saturating_sub(creation),
        creation,
        cached,
        usage.output_tokens.unwrap_or(0),
    )
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
