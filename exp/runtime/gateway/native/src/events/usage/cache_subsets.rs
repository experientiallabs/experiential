//! Cache read and write subsets of an OpenAI-shaped input total.

use serde_json::{Map, Value};

use super::super::Usage;
use super::{bounded_ledger_sum, optional_usage_detail};

/// Check the cache-write TTL subsets and fit the input total to the cache
/// subsets.
///
/// Reads and writes are disjoint subsets of input, so a report whose reads
/// plus writes exceed its input total is internally inconsistent. Moonshot's
/// Kimi wire relayed by Novita sends such reports (2026-10-21..22: 37 of
/// ~2,800 kimi-k3 attempts in three hours, the same prompt failing on every
/// retry), and the provider has already generated and billed the answer by the
/// time usage arrives. Failing the attempt threw that answer away and settled
/// nothing. Instead every token the provider placed in a cache leg keeps its
/// leg, and the input total is raised to cover them (no fresh remainder): no
/// reported read or write is dropped or repriced lower, so the attempt is never
/// under-billed against the provider's own legs, and an overlap is priced once
/// in each leg it was reported in. A provider that reads its writes back in
/// the same call should still be configured `writes_within_reads`, which
/// prices that overlap once at its combined rate. The adjustment is logged
/// with the counts so the shape stays visible.
pub(super) fn fit_cache_subsets(usage: &mut Usage) -> Result<(), String> {
    if let (Some(hour), Some(written)) = (
        usage.cache_creation_1h_input_tokens,
        usage.cache_creation_input_tokens,
    ) {
        if hour > written {
            return Err("one-hour cache writes exceed total cache writes".into());
        }
    }
    if let Some(half_day) = usage.cache_creation_12h_input_tokens {
        let timed = half_day.saturating_add(usage.cache_creation_1h_input_tokens.unwrap_or(0));
        if usage
            .cache_creation_input_tokens
            .is_none_or(|written| timed > written)
        {
            return Err("12-hour cache writes exceed total cache writes".into());
        }
    }
    let subsets = bounded_ledger_sum(
        &[
            usage.cached_input_tokens.unwrap_or(0),
            usage.cache_creation_input_tokens.unwrap_or(0),
        ],
        "cache subsets",
    )?;
    if let Some(input) = usage.input_tokens.filter(|input| subsets > *input) {
        let line = serde_json::json!({
            "event": "cache_subsets_exceed_input",
            "input_tokens": input,
            "cached_input_tokens": usage.cached_input_tokens,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens,
            "settled_input_tokens": subsets,
        });
        eprintln!("exp-gateway-native: {line}");
        usage.input_tokens = Some(subsets);
    }
    Ok(())
}

/// Read the cache subsets of OpenAI-shaped total input. Their fit against
/// input is decided once over the coalesced report (`fit_cache_subsets`),
/// after a `writes_within_reads` rung's written reads are separated; only an
/// unpersistable sum fails here.
pub(super) fn cache_subsets(
    object: &Map<String, Value>,
    detail_key: &str,
) -> Result<(Option<u64>, Option<u64>), String> {
    let reads = optional_usage_detail(object, detail_key, "cached_tokens", "cached_tokens")?;
    let writes = optional_usage_detail(
        object,
        detail_key,
        "cache_write_tokens",
        "cache_write_tokens",
    )?;
    bounded_ledger_sum(&[reads.unwrap_or(0), writes.unwrap_or(0)], "cache subsets")?;
    Ok((reads, writes))
}

/// Move tokens written and read back in one call out of the read leg, so the
/// coalesced counts satisfy the disjoint contract every settlement prices.
/// A write without a covering read count is left as a disjoint write.
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
