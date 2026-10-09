//! Cache read and write subsets of an OpenAI-shaped input total.

use serde_json::{Map, Value};

use super::super::Usage;
use super::{bounded_ledger_sum, optional_usage_detail};

pub(super) fn validate_cache_subsets(usage: &Usage) -> Result<(), String> {
    if let (Some(hour), Some(written)) = (
        usage.cache_creation_1h_input_tokens,
        usage.cache_creation_input_tokens,
    ) {
        if hour > written {
            return Err("one-hour cache writes exceed total cache writes".into());
        }
    }
    let subsets = bounded_ledger_sum(
        &[
            usage.cached_input_tokens.unwrap_or(0),
            usage.cache_creation_input_tokens.unwrap_or(0),
        ],
        "cache subsets",
    )?;
    if usage.input_tokens.is_some_and(|input| subsets > input) {
        return Err("cache read and write tokens exceed total input tokens".into());
    }
    Ok(())
}

/// Read the cache subsets of OpenAI-shaped total input. Disjoint reads and
/// writes must fit input together; writes reported within reads must fit the
/// reads, which must fit input.
pub(super) fn cache_subsets(
    object: &Map<String, Value>,
    detail_key: &str,
    input_tokens: Option<u64>,
    writes_within_reads: bool,
) -> Result<(Option<u64>, Option<u64>), String> {
    let reads = optional_usage_detail(object, detail_key, "cached_tokens", "cached_tokens")?;
    let writes = optional_usage_detail(
        object,
        detail_key,
        "cache_write_tokens",
        "cache_write_tokens",
    )?;
    // A write not covered by its read count is reported the ordinary disjoint
    // way; it must then fit input like any other rung. Usage arrives after the
    // content already streamed, so a placeable report never fails the stream.
    if writes_within_reads && writes.unwrap_or(0) <= reads.unwrap_or(0) {
        if input_tokens.is_some_and(|input| reads.unwrap_or(0) > input) {
            return Err("cache read tokens exceed total input tokens".to_string());
        }
        return Ok((reads, writes));
    }
    let subsets = bounded_ledger_sum(&[reads.unwrap_or(0), writes.unwrap_or(0)], "cache subsets")?;
    if input_tokens.is_some_and(|input| subsets > input) {
        return Err("cache read and write tokens exceed total input tokens".to_string());
    }
    Ok((reads, writes))
}

/// Move tokens written and read back in one call out of the read leg, so the
/// coalesced counts satisfy the disjoint contract every settlement prices.
/// A write without a covering read count cannot be placed and is malformed.
pub(super) fn separate_written_reads(usage: &mut Usage) -> Result<(), String> {
    let writes = usage.cache_creation_input_tokens.unwrap_or(0);
    if writes == 0 {
        return Ok(());
    }
    // Uncovered writes are the disjoint shape: leave them for the ordinary check.
    let Some(reads) = usage.cached_input_tokens.filter(|reads| *reads >= writes) else {
        return Ok(());
    };
    usage.cached_input_tokens = Some(reads - writes);
    usage.cache_writes_read_back = true;
    Ok(())
}
