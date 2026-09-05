//! Provider usage parsing shared by every dialect normalizer: bounded
//! ledger counts, the OpenAI/Gemini/Bedrock usage mappers, and the reasoning
//! subset-folding rules documented on `crate::events`.

use serde_json::{Map, Value};

use super::Usage;

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
    input_tokens: u64,
    output_tokens: u64,
    reasoning_tokens: Option<u64>,
    total_tokens: Option<u64>,
    label: &str,
) -> Result<u64, String> {
    let Some(reasoning) = reasoning_tokens.filter(|reasoning| *reasoning > 0) else {
        return Ok(output_tokens);
    };
    let subset_total = input_tokens.checked_add(output_tokens);
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
pub fn openai_usage(value: Option<&Value>) -> Result<Option<Usage>, String> {
    let value = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(value) => value,
    };
    let object = value
        .as_object()
        .ok_or_else(|| "OpenAI usage must be an object".to_string())?;
    let input_tokens = count_or_zero(object, "input_tokens", "OpenAI input_tokens")?;
    let reported_output = count_or_zero(object, "output_tokens", "OpenAI output_tokens")?;
    let reasoning_tokens = optional_usage_detail(
        object,
        "output_tokens_details",
        "reasoning_tokens",
        "OpenAI reasoning_tokens",
    )?;
    let total_tokens = count_if_present(object, "total_tokens", "OpenAI usage")?;
    let output_tokens = fold_openai_shaped_reasoning(
        input_tokens,
        reported_output,
        reasoning_tokens,
        total_tokens,
        "OpenAI output",
    )?;
    Ok(Some(Usage {
        input_tokens: Some(input_tokens),
        output_tokens: Some(output_tokens),
        cached_input_tokens: optional_usage_detail(
            object,
            "input_tokens_details",
            "cached_tokens",
            "OpenAI cached_tokens",
        )?,
        cache_creation_input_tokens: None,
        reasoning_tokens,
    }))
}

/// Parse a Chat Completions usage object: a malformed object fails the stream
/// instead of silently dropping token accounting.
/// `completion_tokens_details.reasoning_tokens` folds into `output_tokens`
/// when the provider's `total_tokens` shows it was reported additively.
pub fn openai_compatible_usage(value: &Value) -> Result<Usage, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "OpenAI-compatible usage must be an object".to_string())?;
    let input_tokens = count_or_zero(object, "prompt_tokens", "prompt_tokens")?;
    let completion_tokens = count_or_zero(object, "completion_tokens", "completion_tokens")?;
    let reasoning_tokens = optional_usage_detail(
        object,
        "completion_tokens_details",
        "reasoning_tokens",
        "reasoning_tokens",
    )?;
    let total_tokens = count_if_present(object, "total_tokens", "OpenAI-compatible usage")?;
    let output_tokens = fold_openai_shaped_reasoning(
        input_tokens,
        completion_tokens,
        reasoning_tokens,
        total_tokens,
        "OpenAI-compatible output",
    )?;
    Ok(Usage {
        input_tokens: Some(input_tokens),
        output_tokens: Some(output_tokens),
        cached_input_tokens: optional_usage_detail(
            object,
            "prompt_tokens_details",
            "cached_tokens",
            "cached_tokens",
        )?,
        cache_creation_input_tokens: None,
        reasoning_tokens,
    })
}

/// Parse Gemini `usageMetadata`: cached tokens are an input subset, absent
/// counts are zero (`require_integer` parity), and `thoughtsTokenCount` stays
/// unknown when omitted.
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
    let output_tokens = match reasoning_tokens {
        Some(reasoning) => bounded_ledger_sum(&[candidates_tokens, reasoning], "Gemini output")?,
        None => candidates_tokens,
    };
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
        reasoning_tokens,
    })
}

/// Parse Bedrock `metadata.usage`: cache read and write legs fold into total
/// input, cached input reports the read leg, and absent counts are zero
/// (`require_integer` parity). Legs and the folded total beyond the
/// persistable ledger range are provider contract violations and fail the
/// stream rather than reaching settlement as a value the ledger could never
/// write. Converse bills a reasoning model's thinking inside `outputTokens`
/// and publishes no separate count, so `reasoning_tokens` stays unknown.
pub fn bedrock_usage(value: Option<&Value>) -> Result<Usage, String> {
    let usage = value
        .and_then(Value::as_object)
        .ok_or_else(|| "Bedrock metadata.usage must be an object".to_string())?;
    let fresh = count_or_zero(usage, "inputTokens", "Bedrock inputTokens")?;
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
    let input_tokens = bounded_ledger_sum(&[fresh, cache_read, cache_write], "Bedrock input")?;
    Ok(Usage {
        input_tokens: Some(input_tokens),
        output_tokens: Some(count_or_zero(
            usage,
            "outputTokens",
            "Bedrock outputTokens",
        )?),
        cached_input_tokens: Some(cache_read),
        cache_creation_input_tokens: None,
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
