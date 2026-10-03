//! Provider usage parsing shared by every dialect normalizer: bounded
//! ledger counts, the OpenAI/Gemini/Bedrock usage mappers, and the reasoning
//! subset-folding rules documented on `crate::events`.

use serde_json::{Map, Value};

use super::Usage;

impl Usage {
    /// Mark unknown subsets that an OpenAI-compatible shape represents as zero.
    pub(crate) fn unreported_token_details(&self) -> Vec<&'static str> {
        [
            ("cached_tokens", self.cached_input_tokens),
            ("cache_write_tokens", self.cache_creation_input_tokens),
            ("reasoning_tokens", self.reasoning_tokens),
        ]
        .into_iter()
        .filter_map(|(name, count)| count.is_none().then_some(name))
        .collect()
    }

    pub fn has_token_counts(&self) -> bool {
        self.input_tokens.is_some() && self.output_tokens.is_some()
    }

    /// Coalesce cumulative reports from one attempt, never adding snapshots or
    /// replacing an observed count with an absent leg. A lower stale snapshot
    /// cannot reduce an already witnessed count; explicit zero stays known.
    pub(crate) fn merge_observed(&mut self, newer: &Usage) {
        let writes = self.cache_creation_input_tokens;
        let next_writes = newer.cache_creation_input_tokens;
        let next_hour = newer
            .cache_creation_1h_input_tokens
            .filter(|hour| next_writes.or(writes).is_some_and(|total| *hour <= total));
        if next_writes.is_some() && next_writes > writes {
            // Only growth invalidates the earlier allocation. An equal total
            // with no breakdown adds no evidence, and a lower total is stale.
            self.cache_creation_1h_input_tokens = next_hour;
        } else if next_writes.is_none() || next_writes == writes {
            self.cache_creation_1h_input_tokens =
                self.cache_creation_1h_input_tokens.max(next_hour);
        }
        self.input_tokens = self.input_tokens.max(newer.input_tokens);
        self.output_tokens = self.output_tokens.max(newer.output_tokens);
        self.cached_input_tokens = self.cached_input_tokens.max(newer.cached_input_tokens);
        self.cache_creation_input_tokens = self
            .cache_creation_input_tokens
            .max(newer.cache_creation_input_tokens);
        self.reasoning_tokens = self.reasoning_tokens.max(newer.reasoning_tokens);
    }
}

/// Largest count the durable ledger can persist: usage lands in signed
/// 64-bit SQLite INTEGER columns, so anything above `i64::MAX` could never
/// settle and is treated as a provider contract violation at the parser.
pub const MAXIMUM_LEDGER_COUNT: u64 = i64::MAX as u64;

/// Read an optional non-negative count, mirroring `require_integer`: absent
/// or null counts as zero because providers omit zero-valued usage fields,
/// while a present non-integer (or unpersistably large) value is a provider
/// contract violation.
pub fn count_or_zero(object: &Map<String, Value>, key: &str, label: &str) -> Result<u64, String> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(0),
        Some(value) => value
            .as_u64()
            .filter(|count| *count <= MAXIMUM_LEDGER_COUNT)
            .ok_or_else(|| format!("{label} must be a non-negative integer")),
    }
}

/// Read one count only when its key is present and non-null: an absent key
/// yields `None` so a partial usage report never overwrites an earlier leg
/// with an invented zero.
pub fn count_if_present(
    object: &Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<Option<u64>, String> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .filter(|count| *count <= MAXIMUM_LEDGER_COUNT)
            .map(Some)
            .ok_or_else(|| format!("{label}.{key} must be a non-negative integer")),
    }
}

/// Read one optional token subset, mirroring `_optional_usage_detail`: an
/// absent detail object stays unknown instead of zero.
fn optional_usage_detail(
    object: &Map<String, Value>,
    detail_key: &str,
    field_name: &str,
    label: &str,
) -> Result<Option<u64>, String> {
    let details = match object.get(detail_key) {
        None | Some(Value::Null) => return Ok(None),
        Some(value) => value
            .as_object()
            .ok_or_else(|| format!("{label} details must be an object"))?,
    };
    match details.get(field_name) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .filter(|count| *count <= MAXIMUM_LEDGER_COUNT)
            .map(Some)
            .ok_or_else(|| format!("{label} must be a non-negative integer")),
    }
}

/// Sum persistable legs into one ledger count. Individually persistable legs
/// whose total is not are a provider contract violation, never a clamped or
/// wrapped total.
pub fn bounded_ledger_sum(legs: &[u64], label: &str) -> Result<u64, String> {
    legs.iter()
        .try_fold(0u64, |total, leg| total.checked_add(*leg))
        .filter(|total| *total <= MAXIMUM_LEDGER_COUNT)
        .ok_or_else(|| format!("{label} token total overflows a persistable count"))
}

/// Accounting established by a provider total within one completion attempt.
#[derive(Clone, Copy)]
enum ReasoningAccounting {
    Subset,
    Additive,
}

/// A positive reasoning count distinguishes the two possible total shapes.
fn openai_reasoning_accounting(
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    reasoning_tokens: Option<u64>,
    total_tokens: Option<u64>,
) -> Option<ReasoningAccounting> {
    let reasoning = reasoning_tokens.filter(|reasoning| *reasoning > 0)?;
    let subset_total = input_tokens?.checked_add(output_tokens?)?;
    match total_tokens? {
        total if total == subset_total => Some(ReasoningAccounting::Subset),
        total if Some(total) == subset_total.checked_add(reasoning) => {
            Some(ReasoningAccounting::Additive)
        }
        _ => None,
    }
}

/// Fold additive reasoning once, leaving undecided reports on the count heuristic.
fn fold_openai_shaped_reasoning(
    output_tokens: u64,
    reasoning_tokens: Option<u64>,
    accounting: Option<ReasoningAccounting>,
    label: &str,
) -> Result<u64, String> {
    let Some(reasoning) = reasoning_tokens.filter(|reasoning| *reasoning > 0) else {
        return Ok(output_tokens);
    };
    let additive = match accounting {
        Some(ReasoningAccounting::Subset) => false,
        Some(ReasoningAccounting::Additive) => true,
        None => reasoning > output_tokens,
    };
    if additive {
        bounded_ledger_sum(&[output_tokens, reasoning], label)
    } else {
        Ok(output_tokens)
    }
}

/// Parse an OpenAI-shaped usage object from a terminal Responses payload: an
/// omitted object is unknown usage, while a malformed one fails the stream.
/// `output_tokens_details.reasoning_tokens` folds into `output_tokens` when
/// the provider's `total_tokens` shows it was reported additively.
#[cfg(test)]
pub fn openai_usage(value: Option<&Value>) -> Result<Option<Usage>, String> {
    let value = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(value) => value,
    };
    let object = value
        .as_object()
        .ok_or_else(|| "OpenAI usage must be an object".to_string())?;
    OpenAiUsageAccumulator::default()
        .update(object, false)
        .map(Some)
}

/// Raw per-dial counters retained before additive reasoning normalization.
/// Sparse reports must be combined before deciding whether reasoning is extra.
///
/// `writes_within_reads` selects the rung's cache-write accounting. By default
/// reads and writes are disjoint slices of input. A rung whose provider creates
/// the cache and then reads the written tokens back in the same call (Google
/// explicit caching relayed by OpenRouter reports `cached_tokens` and
/// `cache_write_tokens` over the same prefix) reports writes as a subset of
/// reads; its counts are normalized to the disjoint contract, so written
/// tokens leave the read leg and are priced once at the cache-write rate.
/// The provider bills both legs for those tokens, so that rung's authored
/// cache-write rate must be its write rate plus its read rate.
#[derive(Clone, Default)]
pub(crate) struct OpenAiUsageAccumulator {
    reported: Usage,
    reasoning_accounting: Option<ReasoningAccounting>,
    reasoning_accounting_confirmed: bool,
    reasoning_counters_changed: bool,
    input_tokens_changed: bool,
    initial_total_tokens: Option<u64>,
    writes_within_reads: bool,
    // Sparse TTL evidence stays private until a write total can cover it.
    pending_cache_creation_1h_input_tokens: Option<u64>,
}

impl OpenAiUsageAccumulator {
    /// Select the rung's cache-write accounting before any usage arrives.
    pub(crate) fn set_writes_within_reads(&mut self, writes_within_reads: bool) {
        self.writes_within_reads = writes_within_reads;
    }

    pub(crate) fn update_chat(&mut self, value: &Value) -> Result<Usage, String> {
        let object = value
            .as_object()
            .ok_or("OpenAI-compatible usage must be an object")?;
        self.update(object, true)
    }

    pub(crate) fn update_responses(
        &mut self,
        value: Option<&Value>,
    ) -> Result<Option<Usage>, String> {
        match value {
            None | Some(Value::Null) => Ok(None),
            Some(value) => {
                let object = value.as_object().ok_or("OpenAI usage must be an object")?;
                self.update(object, false).map(Some)
            }
        }
    }

    fn update(&mut self, object: &Map<String, Value>, chat: bool) -> Result<Usage, String> {
        let (input_key, output_key, input_details, output_details) = if chat {
            (
                "prompt_tokens",
                "completion_tokens",
                "prompt_tokens_details",
                "completion_tokens_details",
            )
        } else {
            (
                "input_tokens",
                "output_tokens",
                "input_tokens_details",
                "output_tokens_details",
            )
        };
        let input_tokens = count_if_present(object, input_key, "OpenAI usage")?;
        let output_tokens = count_if_present(object, output_key, "OpenAI usage")?;
        let unreported = unreported_token_details(object, input_details, output_details)?;
        let reasoning_tokens = optional_usage_detail(
            object,
            output_details,
            "reasoning_tokens",
            "OpenAI reasoning_tokens",
        )?
        .filter(|_| !unreported[3]);
        let total_tokens = count_if_present(object, "total_tokens", "OpenAI usage")?;
        let (cached_input_tokens, cache_creation_input_tokens) = cache_subsets(
            object,
            input_details,
            input_tokens,
            self.writes_within_reads,
        )?;
        let cached_input_tokens = cached_input_tokens.filter(|_| !unreported[0]);
        let cache_creation_input_tokens = cache_creation_input_tokens.filter(|_| !unreported[1]);
        let cache_creation_1h_input_tokens = optional_usage_detail(
            object,
            input_details,
            "cache_write_1h_tokens",
            "cache_write_1h_tokens",
        )?
        .filter(|_| !unreported[2])
        .max(self.pending_cache_creation_1h_input_tokens);
        let covering_writes =
            cache_creation_input_tokens.or(self.reported.cache_creation_input_tokens);
        if let (Some(hour), Some(written)) = (cache_creation_1h_input_tokens, covering_writes) {
            if hour > written {
                return Err("one-hour cache writes exceed total cache writes".into());
            }
        }
        let mut candidate = self.clone();
        let input_changed = self
            .reported
            .input_tokens
            .zip(input_tokens)
            .is_some_and(|(old, new)| old != new);
        candidate.input_tokens_changed |= input_changed;
        candidate.reasoning_counters_changed |= input_changed
            || [
                (self.reported.output_tokens, output_tokens),
                (self.reported.reasoning_tokens, reasoning_tokens),
                (self.initial_total_tokens, total_tokens),
            ]
            .into_iter()
            .any(|(old, new)| old.zip(new).is_some_and(|(old, new)| old != new));
        candidate.initial_total_tokens = (!candidate.reasoning_counters_changed)
            .then_some(self.initial_total_tokens.or(total_tokens))
            .flatten();
        candidate.pending_cache_creation_1h_input_tokens = covering_writes
            .is_none()
            .then_some(cache_creation_1h_input_tokens)
            .flatten();
        candidate.reported.merge_observed(&Usage {
            input_tokens,
            output_tokens,
            cached_input_tokens,
            cache_creation_input_tokens,
            cache_creation_1h_input_tokens: covering_writes.and(cache_creation_1h_input_tokens),
            reasoning_tokens,
        });
        // Initial split fragments support a provisional mode until counters
        // change. A coherent raw output/reasoning/total sample may correct it;
        // an established coherent mode then survives sparse and stale updates.
        // Only an unchanged input count may be reused by a coherent sample.
        if !candidate.reasoning_accounting_confirmed {
            let coherent = openai_reasoning_accounting(
                input_tokens.or_else(|| {
                    (!candidate.input_tokens_changed)
                        .then_some(self.reported.input_tokens)
                        .flatten()
                }),
                output_tokens,
                reasoning_tokens,
                total_tokens,
            );
            if coherent.is_some() {
                candidate.reasoning_accounting = coherent;
                candidate.reasoning_accounting_confirmed = true;
            } else if !candidate.reasoning_counters_changed {
                candidate.reasoning_accounting = openai_reasoning_accounting(
                    candidate.reported.input_tokens,
                    candidate.reported.output_tokens,
                    candidate.reported.reasoning_tokens,
                    candidate.initial_total_tokens,
                );
            }
        }
        let mut normalized = candidate.reported.clone();
        if self.writes_within_reads {
            separate_written_reads(&mut normalized)?;
        }
        validate_cache_subsets(&normalized)?;
        normalized.output_tokens = normalized
            .output_tokens
            .map(|output| {
                fold_openai_shaped_reasoning(
                    output,
                    normalized.reasoning_tokens,
                    candidate.reasoning_accounting,
                    "OpenAI output",
                )
            })
            .transpose()?;
        *self = candidate;
        Ok(normalized)
    }
}

/// Restore missing meter evidence without rejecting required integer wire fields.
fn unreported_token_details(
    object: &Map<String, Value>,
    input_details: &str,
    output_details: &str,
) -> Result<[bool; 4], String> {
    let mut unreported = [false; 4];
    let Some(raw) = object.get("unreported_token_details") else {
        return Ok(unreported);
    };
    let fields = raw
        .as_array()
        .ok_or("unreported_token_details must be an array")?;
    for field in fields {
        let name = field
            .as_str()
            .ok_or("unreported_token_details entries must be strings")?;
        let index = match name {
            "cached_tokens" => 0,
            "cache_write_tokens" => 1,
            "cache_write_1h_tokens" => 2,
            "reasoning_tokens" => 3,
            _ => return Err("unreported_token_details contains an unknown field".into()),
        };
        if unreported[index] {
            return Err("unreported_token_details contains a duplicate field".into());
        }
        let group = if index == 3 {
            output_details
        } else {
            input_details
        };
        if optional_usage_detail(object, group, name, name)?.is_some_and(|value| value != 0) {
            return Err("an unreported token detail cannot contain a positive count".into());
        }
        unreported[index] = true;
    }
    Ok(unreported)
}

fn validate_cache_subsets(usage: &Usage) -> Result<(), String> {
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

/// Parse a Chat Completions usage object: a malformed object fails the stream
/// instead of silently dropping token accounting.
/// `completion_tokens_details.reasoning_tokens` folds into `output_tokens`
/// when the provider's `total_tokens` shows it was reported additively.
#[cfg(test)]
pub fn openai_compatible_usage(value: &Value) -> Result<Usage, String> {
    OpenAiUsageAccumulator::default().update_chat(value)
}

/// Read the cache subsets of OpenAI-shaped total input. Disjoint reads and
/// writes must fit input together; writes reported within reads must fit the
/// reads, which must fit input.
fn cache_subsets(
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
fn separate_written_reads(usage: &mut Usage) -> Result<(), String> {
    let writes = usage.cache_creation_input_tokens.unwrap_or(0);
    if writes == 0 {
        return Ok(());
    }
    // Uncovered writes are the disjoint shape: leave them for the ordinary check.
    let Some(reads) = usage.cached_input_tokens.filter(|reads| *reads >= writes) else {
        return Ok(());
    };
    usage.cached_input_tokens = Some(reads - writes);
    Ok(())
}

/// Parse a present Gemini `usageMetadata`: its non-optional proto3 int32
/// fields have implicit presence, so omitted scalar counts mean zero.
/// See google/ai/generativelanguage/v1beta/generative_service.proto in
/// https://github.com/googleapis/googleapis and ProtoJSON default-value rules:
/// https://protobuf.dev/programming-guides/json/#presence-and-default-values
/// The dialect keeps an absent usage object unknown. This provider-specific
/// scalar rule also covers an omitted thinking count, without asserting that
/// any reasoning tokens were generated.
///
/// Google defines thinking tokens as ADDITIVE to `candidatesTokenCount`
/// (`totalTokenCount` = prompt + candidates + thoughts, and response pricing
/// is the sum of output and thinking tokens), so a reported
/// `thoughtsTokenCount` is folded into `output_tokens`; `reasoning_tokens`
/// names the subset the ledger prices at the reasoning rate.
pub fn gemini_usage(value: &Value) -> Result<Usage, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "Gemini usageMetadata must be an object".to_string())?;
    let reasoning_tokens =
        count_or_zero(object, "thoughtsTokenCount", "Gemini thoughtsTokenCount")?;
    let candidates_tokens = count_or_zero(
        object,
        "candidatesTokenCount",
        "Gemini candidatesTokenCount",
    )?;
    let output_tokens =
        bounded_ledger_sum(&[candidates_tokens, reasoning_tokens], "Gemini output")?;
    Ok(Usage {
        input_tokens: Some(count_or_zero(
            object,
            "promptTokenCount",
            "Gemini promptTokenCount",
        )?),
        output_tokens: Some(output_tokens),
        cached_input_tokens: Some(count_or_zero(
            object,
            "cachedContentTokenCount",
            "Gemini cachedContentTokenCount",
        )?),
        cache_creation_input_tokens: None,
        cache_creation_1h_input_tokens: None,
        reasoning_tokens: Some(reasoning_tokens),
    })
}

/// Parse Bedrock `metadata.usage`: cache read and write legs fold into total
/// input, cached input reports the read leg, and omitted cache legs mean
/// zero. Primary omissions stay unknown. Legs and the folded total beyond the
/// persistable ledger range are provider contract violations and fail the
/// stream rather than reaching settlement as a value the ledger could never
/// write. Converse bills a reasoning model's thinking inside `outputTokens`
/// and publishes no separate count, so `reasoning_tokens` stays unknown.
pub fn bedrock_usage(value: Option<&Value>) -> Result<Usage, String> {
    let usage = value
        .and_then(Value::as_object)
        .ok_or_else(|| "Bedrock metadata.usage must be an object".to_string())?;
    let fresh = count_if_present(usage, "inputTokens", "Bedrock usage")?;
    let cache_read = count_or_zero(
        usage,
        "cacheReadInputTokens",
        "Bedrock cacheReadInputTokens",
    )?;
    let cache_write = count_or_zero(
        usage,
        "cacheWriteInputTokens",
        "Bedrock cacheWriteInputTokens",
    )?;
    let input_tokens = fresh
        .map(|fresh| bounded_ledger_sum(&[fresh, cache_read, cache_write], "Bedrock input"))
        .transpose()?;
    Ok(Usage {
        input_tokens,
        output_tokens: count_if_present(usage, "outputTokens", "Bedrock usage")?,
        cached_input_tokens: Some(cache_read),
        cache_creation_input_tokens: count_if_present(
            usage,
            "cacheWriteInputTokens",
            "Bedrock usage",
        )?,
        cache_creation_1h_input_tokens: None,
        reasoning_tokens: None,
    })
}

/// Fetch a required string field from a provider JSON object.
pub fn require_string(
    object: &Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<String, String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("{label} must be text"))
}

/// Fetch a required provider identity with the public contract's character bound.
pub fn require_bounded_string(
    object: &Map<String, Value>,
    key: &str,
    label: &str,
    maximum_chars: usize,
) -> Result<String, String> {
    let value = require_string(object, key, label)?;
    let length = value.chars().count();
    if length == 0 || length > maximum_chars {
        return Err(format!(
            "{label} must contain between 1 and {maximum_chars} characters"
        ));
    }
    Ok(value)
}

/// Fetch a required non-negative integer field from a provider JSON object,
/// bounded like every parsed count so no downstream consumer can receive a
/// value outside the persistable signed 64-bit range.
pub fn require_u64(object: &Map<String, Value>, key: &str, label: &str) -> Result<u64, String> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .filter(|count| *count <= MAXIMUM_LEDGER_COUNT)
        .ok_or_else(|| format!("{label} must be a non-negative integer"))
}

#[cfg(test)]
mod sparse_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn established_reasoning_accounting_survives_sparse_growth_and_stale_totals() {
        for chat in [false, true] {
            for additive in [false, true] {
                let (input, output, details) = if chat {
                    (
                        "prompt_tokens",
                        "completion_tokens",
                        "completion_tokens_details",
                    )
                } else {
                    ("input_tokens", "output_tokens", "output_tokens_details")
                };
                let first = json!({
                    input: 100, output: 10, details: {"reasoning_tokens": 5},
                    "total_tokens": if additive { 115 } else { 110 }
                });
                let mut accumulator = OpenAiUsageAccumulator::default();
                for (report, expected_output, expected_reasoning) in [
                    (first.clone(), if additive { 15 } else { 10 }, 5),
                    // For additive accounting the retained old total now matches
                    // input + raw output, but it must not change the dialect.
                    (json!({output: 15}), if additive { 20 } else { 15 }, 5),
                    (json!({output: 20}), if additive { 25 } else { 20 }, 5),
                    (
                        json!({details: {"reasoning_tokens": 8}}),
                        if additive { 28 } else { 20 },
                        8,
                    ),
                    (first, if additive { 28 } else { 20 }, 8),
                    (json!({}), if additive { 28 } else { 20 }, 8),
                    (
                        json!({"total_tokens": if additive { 128 } else { 120 }}),
                        if additive { 28 } else { 20 },
                        8,
                    ),
                ] {
                    let observed = accumulator
                        .update(report.as_object().unwrap(), chat)
                        .unwrap();
                    assert_eq!(observed.input_tokens, Some(100));
                    assert_eq!(observed.output_tokens, Some(expected_output), "{report}");
                    assert_eq!(observed.reasoning_tokens, Some(expected_reasoning));
                }
            }
        }
    }

    #[test]
    fn additive_reasoning_growth_rejects_overflow_without_advancing_raw_state() {
        let mut accumulator = OpenAiUsageAccumulator::default();
        let first = json!({
            "input_tokens": 100, "output_tokens": 10,
            "output_tokens_details": {"reasoning_tokens": 5}, "total_tokens": 115
        });
        accumulator.update_responses(Some(&first)).unwrap();
        assert!(accumulator
            .update_responses(Some(&json!({"output_tokens": MAXIMUM_LEDGER_COUNT})))
            .is_err());
        let observed = accumulator
            .update_responses(Some(&json!({"output_tokens": 20})))
            .unwrap()
            .unwrap();
        assert_eq!(observed.output_tokens, Some(25));
        assert_eq!(observed.reasoning_tokens, Some(5));
    }

    #[test]
    fn undecided_reasoning_waits_for_a_positive_decisive_total() {
        for first in [
            json!({"input_tokens": 100, "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 5}}),
            json!({"input_tokens": 100, "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 102}),
        ] {
            let mut accumulator = OpenAiUsageAccumulator::default();
            accumulator.update_responses(Some(&first)).unwrap();
            let observed = accumulator
                .update_responses(Some(&json!({
                    "output_tokens": 10, "output_tokens_details": {"reasoning_tokens": 5},
                    "total_tokens": 110
                })))
                .unwrap()
                .unwrap();
            assert_eq!(observed.output_tokens, Some(10));
            assert!(matches!(
                accumulator.reasoning_accounting,
                Some(ReasoningAccounting::Subset)
            ));
        }
    }

    #[test]
    fn missing_reasoning_does_not_let_a_stale_total_establish_subset_accounting() {
        let mut accumulator = OpenAiUsageAccumulator::default();
        for report in [
            json!({"input_tokens": 100, "output_tokens": 10, "total_tokens": 115}),
            json!({"output_tokens": 15, "output_tokens_details": {"reasoning_tokens": 5}}),
        ] {
            accumulator.update_responses(Some(&report)).unwrap();
        }
        let observed = accumulator
            .update_responses(Some(&json!({
                "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 5},
                "total_tokens": 125
            })))
            .unwrap()
            .unwrap();
        assert_eq!(observed.output_tokens, Some(25));
    }

    #[test]
    fn late_totals_cannot_establish_accounting_after_raw_growth() {
        for (later_output, late_total) in [(15, 115), (20, 120)] {
            let mut accumulator = OpenAiUsageAccumulator::default();
            for report in [
                json!({"input_tokens": 100, "output_tokens": 10, "total_tokens": 115}),
                json!({"output_tokens": 15, "output_tokens_details": {"reasoning_tokens": 5}}),
                json!({"output_tokens": later_output}),
                json!({"total_tokens": late_total}),
            ] {
                accumulator.update_responses(Some(&report)).unwrap();
            }
            assert!(accumulator.reasoning_accounting.is_none());
            for additive in [false, true] {
                let mut decided = accumulator.clone();
                let observed = decided
                    .update_responses(Some(&json!({
                        "output_tokens": 25, "output_tokens_details": {"reasoning_tokens": 5},
                        "total_tokens": if additive { 130 } else { 125 }
                    })))
                    .unwrap()
                    .unwrap();
                assert_eq!(observed.output_tokens, Some(if additive { 30 } else { 25 }));
            }
        }
    }

    #[test]
    fn coherent_old_snapshot_establishes_mode_from_its_own_raw_counts() {
        let mut accumulator = OpenAiUsageAccumulator::default();
        accumulator
            .update_responses(Some(&json!({"input_tokens": 100, "output_tokens": 20})))
            .unwrap();
        let observed = accumulator
            .update_responses(Some(&json!({
                "input_tokens": 100, "output_tokens": 10,
                "output_tokens_details": {"reasoning_tokens": 5}, "total_tokens": 115
            })))
            .unwrap()
            .unwrap();
        assert_eq!(observed.output_tokens, Some(25));
    }

    #[test]
    fn changed_input_cannot_be_reused_in_a_delayed_counter_sample() {
        let mut accumulator = OpenAiUsageAccumulator::default();
        for report in [
            json!({"input_tokens": 100, "output_tokens": 10, "total_tokens": 115}),
            json!({"input_tokens": 105}),
            json!({"output_tokens": 10, "output_tokens_details": {"reasoning_tokens": 5},
                "total_tokens": 115}),
        ] {
            accumulator.update_responses(Some(&report)).unwrap();
        }
        assert!(accumulator.reasoning_accounting.is_none());
        let observed = accumulator
            .update_responses(Some(&json!({
                "input_tokens": 105, "output_tokens": 15,
                "output_tokens_details": {"reasoning_tokens": 5}, "total_tokens": 125
            })))
            .unwrap()
            .unwrap();
        assert_eq!(observed.output_tokens, Some(20));
    }

    #[test]
    fn unchanged_initial_fragments_preserve_total_first_accounting() {
        for additive in [false, true] {
            let pieces = [
                json!({"input_tokens": 100}),
                json!({"output_tokens": 10}),
                json!({"output_tokens_details": {"reasoning_tokens": 5}}),
                json!({"total_tokens": if additive { 115 } else { 110 }}),
            ];
            for order in [[0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 1, 0], [1, 0, 3, 2]] {
                let mut accumulator = OpenAiUsageAccumulator::default();
                let mut observed = None;
                for index in order {
                    observed = accumulator.update_responses(Some(&pieces[index])).unwrap();
                }
                assert_eq!(
                    observed.unwrap().output_tokens,
                    Some(if additive { 15 } else { 10 })
                );
            }
        }
    }

    #[test]
    fn unreported_meter_marker_rejects_malformed_or_contradictory_evidence() {
        for marker in [
            json!(null),
            json!({}),
            json!("cached_tokens"),
            json!([1]),
            json!(["other"]),
            json!(["cached_tokens", "cached_tokens"]),
        ] {
            assert!(OpenAiUsageAccumulator::default()
                .update_chat(&json!({
                    "prompt_tokens": 10, "completion_tokens": 10,
                    "unreported_token_details": marker
                }))
                .is_err());
        }
        for field in [
            "cached_tokens",
            "cache_write_tokens",
            "cache_write_1h_tokens",
            "reasoning_tokens",
        ] {
            for count in [json!(1), json!(-1), json!(true), json!(0.5), json!("0")] {
                let group = if field == "reasoning_tokens" {
                    "completion_tokens_details"
                } else {
                    "prompt_tokens_details"
                };
                let mut report = json!({
                    "prompt_tokens": 10, "completion_tokens": 10,
                    "unreported_token_details": [field]
                });
                report[group] = json!({field: count});
                assert!(OpenAiUsageAccumulator::default()
                    .update_chat(&report)
                    .is_err());
            }
        }
    }

    #[test]
    fn sparse_cache_ttl_is_retained_checked_and_invalidated_on_write_growth() {
        let mut accumulator = OpenAiUsageAccumulator::default();
        let partial = accumulator
            .update_chat(&json!({
                "prompt_tokens_details": {"cache_write_1h_tokens": 3}
            }))
            .unwrap();
        assert_eq!(partial.cache_creation_1h_input_tokens, None);
        assert!(accumulator
            .update_chat(&json!({
                "prompt_tokens": 20,
                "prompt_tokens_details": {"cache_write_tokens": 2}
            }))
            .is_err());
        let complete = accumulator
            .update_chat(&json!({
                "prompt_tokens": 20, "completion_tokens": 2,
                "prompt_tokens_details": {"cache_write_tokens": 5}
            }))
            .unwrap();
        assert_eq!(complete.cache_creation_1h_input_tokens, Some(3));
        let grown = accumulator
            .update_chat(&json!({
                "prompt_tokens_details": {"cache_write_tokens": 10}
            }))
            .unwrap();
        assert_eq!(grown.cache_creation_1h_input_tokens, None);
        let zero = accumulator
            .update_chat(&json!({
                "prompt_tokens_details": {"cache_write_1h_tokens": 0}
            }))
            .unwrap();
        assert_eq!(zero.cache_creation_1h_input_tokens, Some(0));
    }

    #[test]
    fn cache_ttl_refuses_malformed_counts_and_non_subsets() {
        for bad in [
            json!(-1),
            json!(true),
            json!(1.5),
            json!("2"),
            json!(101),
            json!(MAXIMUM_LEDGER_COUNT + 1),
        ] {
            assert!(OpenAiUsageAccumulator::default().update_chat(&json!({
                "prompt_tokens": 120, "completion_tokens": 10,
                "prompt_tokens_details": {"cache_write_tokens": 100, "cache_write_1h_tokens": bad}
            })).is_err());
            assert!(OpenAiUsageAccumulator::default().update_responses(Some(&json!({
                "input_tokens": 120, "output_tokens": 10,
                "input_tokens_details": {"cache_write_tokens": 100, "cache_write_1h_tokens": bad}
            }))).is_err());
        }
    }

    #[test]
    fn responses_sparse_raw_counters_match_whole_report_and_never_fold_twice() {
        let full = json!({"input_tokens":100,"output_tokens":10,"output_tokens_details":{"reasoning_tokens":5},"total_tokens":115});
        let expected = openai_usage(Some(&full)).unwrap().unwrap();
        let mut accumulator = OpenAiUsageAccumulator::default();
        for report in [
            json!({"input_tokens":100}),
            json!({"output_tokens":10}),
            json!({"output_tokens_details":{"reasoning_tokens":5},"total_tokens":115}),
            full.clone(),
            full,
        ] {
            let result = accumulator
                .update_responses(Some(&report))
                .unwrap()
                .unwrap();
            if report.get("total_tokens").is_some() {
                assert_eq!(result.input_tokens, expected.input_tokens);
                assert_eq!(result.output_tokens, expected.output_tokens);
                assert_eq!(result.reasoning_tokens, expected.reasoning_tokens);
            }
        }
        assert_eq!(expected.output_tokens, Some(15));
    }

    #[test]
    fn sparse_response_cache_is_checked_when_input_arrives_later() {
        let mut accumulator = OpenAiUsageAccumulator::default();
        accumulator
            .update_responses(Some(&json!({"input_tokens_details":{"cached_tokens":200}})))
            .unwrap();
        assert!(accumulator
            .update_responses(Some(&json!({"input_tokens":100,"output_tokens":1})))
            .is_err());
        let valid = accumulator
            .update_responses(Some(&json!({"input_tokens":250,"output_tokens":1})))
            .unwrap()
            .unwrap();
        assert_eq!(valid.input_tokens, Some(250));
        assert_eq!(valid.cached_input_tokens, Some(200));
    }
}
