//! OpenAI's cache-write lifetime split on an OpenAI-shaped input total.
//!
//! A Responses usage of a model that takes `prompt_cache_options` reports
//! `input_tokens_details.cache_write_tokens_details` with `ttl_30m_tokens`
//! and `ttl_12h_tokens`, which never overlap and sum to `cache_write_tokens`
//! (https://developers.openai.com/api/docs/guides/prompt-caching). Only the
//! 12-hour count is kept: it is the subset priced apart from ordinary writes.

use serde_json::{Map, Value};

use super::MAXIMUM_LEDGER_COUNT;

/// Read `ttl_12h_tokens` from one usage object's input details. An absent
/// details object or split stays unknown; a malformed one fails the stream.
pub(super) fn twelve_hour_writes(
    object: &Map<String, Value>,
    detail_key: &str,
) -> Result<Option<u64>, String> {
    let Some(details) = object.get(detail_key).and_then(Value::as_object) else {
        return Ok(None);
    };
    let split = match details.get("cache_write_tokens_details") {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Object(split)) => split,
        Some(_) => return Err("cache_write_tokens_details must be an object".into()),
    };
    match split.get("ttl_12h_tokens") {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .filter(|count| *count <= MAXIMUM_LEDGER_COUNT)
            .map(Some)
            .ok_or_else(|| "ttl_12h_tokens must be a non-negative integer".to_string()),
    }
}

/// Place the 12-hour subset of `writes`. A rung asked for 12-hour writes whose
/// provider reported writes without the split bills every write at 12 hours:
/// the lifetime it asked for, never the cheaper default.
pub(super) fn resolve_twelve_hour_writes(
    reported: Option<u64>,
    writes: Option<u64>,
    requested: bool,
) -> Option<u64> {
    reported.or(writes.filter(|_| requested))
}

#[cfg(test)]
#[path = "cache_ttl_tests.rs"]
mod tests;
