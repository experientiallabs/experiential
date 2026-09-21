//! Provider usage parsing shared by every dialect normalizer: bounded
//! ledger counts, the OpenAI/Gemini/Bedrock usage mappers, and the reasoning
//! subset-folding rules documented on `crate::events`.

use serde_json::{Map, Value};

use super::Usage;

impl Usage {
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

/// Resolve the output total of an OpenAI-shaped usage object so that
/// `reasoning_tokens` names a subset of it (see the module documentation).
///
/// The provider's own `total_tokens` is authoritative when it matches either
/// accounting: `input + output` is the documented subset shape and the output
/// total is forwarded as reported; `input + output + reasoning` is the
/// additive shape (xAI, natively or relayed by Azure Foundry) and reasoning is
/// folded in. Without a decisive total, a reasoning count above the output
/// total cannot occur under subset semantics and is folded.
fn fold_openai_shaped_reasoning(
    input_tokens: Option<u64>,
    output_tokens: u64,
    reasoning_tokens: Option<u64>,
    total_tokens: Option<u64>,
    label: &str,
) -> Result<u64, String> {
    let Some(reasoning) = reasoning_tokens.filter(|reasoning| *reasoning > 0) else {
        return Ok(output_tokens);
    };
    let subset_total = input_tokens.and_then(|input| input.checked_add(output_tokens));
    let additive_total = subset_total.and_then(|total| total.checked_add(reasoning));
    let additive = match total_tokens {
        Some(total) if Some(total) == subset_total => false,
        Some(total) if Some(total) == additive_total => true,
        _ => reasoning > output_tokens,
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
#[derive(Clone, Default)]
pub(crate) struct OpenAiUsageAccumulator {
    reported: Usage,
    total_tokens: Option<u64>,
}

impl OpenAiUsageAccumulator {
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
        let reasoning_tokens = optional_usage_detail(
            object,
            output_details,
            "reasoning_tokens",
            "OpenAI reasoning_tokens",
        )?;
        let total_tokens = count_if_present(object, "total_tokens", "OpenAI usage")?;
        let (cached_input_tokens, cache_creation_input_tokens) =
            cache_subsets(object, input_details, input_tokens)?;
        let mut candidate = self.clone();
        candidate.reported.merge_observed(&Usage {
            input_tokens,
            output_tokens,
            cached_input_tokens,
            cache_creation_input_tokens,
            cache_creation_1h_input_tokens: None,
            reasoning_tokens,
        });
        candidate.total_tokens = candidate.total_tokens.max(total_tokens);
        let mut normalized = candidate.reported.clone();
        validate_cache_subsets(&normalized)?;
        normalized.output_tokens = normalized
            .output_tokens
            .map(|output| {
                fold_openai_shaped_reasoning(
                    normalized.input_tokens,
                    output,
                    normalized.reasoning_tokens,
                    candidate.total_tokens,
                    "OpenAI output",
                )
            })
            .transpose()?;
        *self = candidate;
        Ok(normalized)
    }
}

#[cfg(test)]
mod sparse_tests {
    use super::*;
    use serde_json::json;

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

fn validate_cache_subsets(usage: &Usage) -> Result<(), String> {
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

/// Cache reads and writes are disjoint subsets of OpenAI-shaped total input.
fn cache_subsets(
    object: &Map<String, Value>,
    detail_key: &str,
    input_tokens: Option<u64>,
) -> Result<(Option<u64>, Option<u64>), String> {
    let reads = optional_usage_detail(object, detail_key, "cached_tokens", "cached_tokens")?;
    let writes = optional_usage_detail(
        object,
        detail_key,
        "cache_write_tokens",
        "cache_write_tokens",
    )?;
    let subsets = bounded_ledger_sum(&[reads.unwrap_or(0), writes.unwrap_or(0)], "cache subsets")?;
    if input_tokens.is_some_and(|input| subsets > input) {
        return Err("cache read and write tokens exceed total input tokens".to_string());
    }
    Ok((reads, writes))
}

/// Parse a present Gemini `usageMetadata`: its non-optional proto3 int32
/// fields have implicit presence, so omitted scalar counts mean zero.
/// See google/ai/generativelanguage/v1beta/generative_service.proto in
/// https://github.com/googleapis/googleapis and ProtoJSON default-value rules:
/// https://protobuf.dev/programming-guides/json/#presence-and-default-values
/// The dialect keeps an absent usage object unknown. An omitted thinking
/// subset stays unspecified rather than asserting a model has reasoning.
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
    let reasoning_tokens = match object.get("thoughtsTokenCount") {
        None | Some(Value::Null) => None,
        Some(_) => Some(count_or_zero(
            object,
            "thoughtsTokenCount",
            "Gemini thoughtsTokenCount",
        )?),
    };
    let candidates_tokens = count_or_zero(
        object,
        "candidatesTokenCount",
        "Gemini candidatesTokenCount",
    )?;
    let output_tokens = bounded_ledger_sum(
        &[candidates_tokens, reasoning_tokens.unwrap_or(0)],
        "Gemini output",
    )?;
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
        reasoning_tokens,
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
