//! Provider-neutral stream events, the Rust mirror of `GatewayEvent`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::errors::Failure;

/// Normalized token usage mirroring `GatewayUsage` semantics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
}

impl Usage {
    pub fn has_token_counts(&self) -> bool {
        self.input_tokens.is_some() && self.output_tokens.is_some()
    }
}

/// One completed tool call with provider-order raw argument text.
#[derive(Debug, Clone)]
pub struct CompletedToolCall {
    pub call_id: String,
    pub name: String,
    /// Raw provider-order JSON argument text, already validated as one object.
    pub raw_arguments: String,
}

/// One ordered provider-neutral stream event.
#[derive(Debug, Clone)]
pub enum Event {
    TextDelta(String),
    RefusalDelta(String),
    ToolCallStarted {
        index: u32,
        call_id: String,
        name: String,
    },
    ToolArgumentsDelta {
        index: u32,
        delta: String,
    },
    ToolCallCompleted {
        index: u32,
        call: CompletedToolCall,
    },
    Usage(Usage),
    Completed,
    Incomplete,
    Failed(Failure),
}

impl Event {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Event::Completed | Event::Incomplete | Event::Failed(_)
        )
    }
}

/// Render one event as the content-bearing JSON object used by dialect parity
/// fixtures: the same field vocabulary the fixture-event parser accepts, plus
/// the failure class and safe message for terminal failures.
pub fn simplified_event(event: &Event) -> Value {
    match event {
        Event::TextDelta(text) => serde_json::json!({"kind": "text_delta", "text": text}),
        Event::RefusalDelta(text) => serde_json::json!({"kind": "refusal_delta", "text": text}),
        Event::ToolCallStarted {
            index,
            call_id,
            name,
        } => serde_json::json!({
            "kind": "tool_call_started",
            "index": index,
            "call_id": call_id,
            "name": name,
        }),
        Event::ToolArgumentsDelta { index, delta } => serde_json::json!({
            "kind": "tool_arguments_delta",
            "index": index,
            "text": delta,
        }),
        Event::ToolCallCompleted { index, call } => serde_json::json!({
            "kind": "tool_call_completed",
            "index": index,
            "call_id": call.call_id,
            "name": call.name,
            "raw_arguments": call.raw_arguments,
        }),
        Event::Usage(usage) => serde_json::json!({
            "kind": "usage",
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
        }),
        Event::Completed => serde_json::json!({"kind": "completed"}),
        Event::Incomplete => serde_json::json!({"kind": "incomplete"}),
        Event::Failed(failure) => serde_json::json!({
            "kind": "failed",
            "failure_class": failure.failure_class.as_str(),
            "safe_message": failure.safe_message,
        }),
    }
}

/// Validate one raw tool-argument accumulation as a single JSON object.
pub fn require_json_object_text(raw: &str) -> Result<(), String> {
    match serde_json::from_str::<Value>(raw) {
        Ok(Value::Object(_)) => Ok(()),
        Ok(_) => Err("streamed tool arguments must decode to an object".to_string()),
        Err(_) => Err("streamed tool arguments are not valid JSON".to_string()),
    }
}

/// Accumulated per-stream state for one incrementally emitted function call.
#[derive(Debug, Clone)]
pub struct ToolAccumulator {
    pub call_id: String,
    pub name: String,
    pub raw_arguments: String,
    pub completed: bool,
}

impl ToolAccumulator {
    pub fn new(call_id: String, name: String) -> Self {
        Self {
            call_id,
            name,
            raw_arguments: String::new(),
            completed: false,
        }
    }

    pub fn complete(&self) -> Result<CompletedToolCall, String> {
        require_json_object_text(&self.raw_arguments)?;
        // Mirror the python ToolCall model constraints so both engines accept
        // exactly the same provider tool-call streams (a call the python
        // engine rejects must not become client-visible history here).
        if self.call_id.is_empty()
            || self.call_id.chars().count() > 256
            || self.name.is_empty()
            || self.name.chars().count() > 256
            || self.raw_arguments.chars().count() > 4_000_000
        {
            return Err("streamed tool call is incomplete".to_string());
        }
        Ok(CompletedToolCall {
            call_id: self.call_id.clone(),
            name: self.name.clone(),
            raw_arguments: self.raw_arguments.clone(),
        })
    }
}

/// Read an optional non-negative count, mirroring `require_integer`: absent
/// or null counts as zero because providers omit zero-valued usage fields,
/// while a present non-integer value is a provider contract violation.
pub fn count_or_zero(object: &Map<String, Value>, key: &str, label: &str) -> Result<u64, String> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(0),
        Some(value) => value
            .as_u64()
            .ok_or_else(|| format!("{label} must be a non-negative integer")),
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
            .map(Some)
            .ok_or_else(|| format!("{label} must be a non-negative integer")),
    }
}

/// Parse an OpenAI-shaped usage object from a terminal Responses payload,
/// mirroring `streaming_usage.openai_usage`: an omitted object is unknown
/// usage, while a malformed one fails the stream.
pub fn openai_usage(value: Option<&Value>) -> Result<Option<Usage>, String> {
    let value = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(value) => value,
    };
    let object = value
        .as_object()
        .ok_or_else(|| "OpenAI usage must be an object".to_string())?;
    Ok(Some(Usage {
        input_tokens: Some(count_or_zero(
            object,
            "input_tokens",
            "OpenAI input_tokens",
        )?),
        output_tokens: Some(count_or_zero(
            object,
            "output_tokens",
            "OpenAI output_tokens",
        )?),
        cached_input_tokens: optional_usage_detail(
            object,
            "input_tokens_details",
            "cached_tokens",
            "OpenAI cached_tokens",
        )?,
        reasoning_tokens: optional_usage_detail(
            object,
            "output_tokens_details",
            "reasoning_tokens",
            "OpenAI reasoning_tokens",
        )?,
    }))
}

/// Parse a Chat Completions usage object, mirroring
/// `streaming_usage.openai_compatible_usage`: a malformed object fails the
/// stream instead of silently dropping token accounting.
pub fn openai_compatible_usage(value: &Value) -> Result<Usage, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "OpenAI-compatible usage must be an object".to_string())?;
    Ok(Usage {
        input_tokens: Some(count_or_zero(object, "prompt_tokens", "prompt_tokens")?),
        output_tokens: Some(count_or_zero(
            object,
            "completion_tokens",
            "completion_tokens",
        )?),
        cached_input_tokens: optional_usage_detail(
            object,
            "prompt_tokens_details",
            "cached_tokens",
            "cached_tokens",
        )?,
        reasoning_tokens: optional_usage_detail(
            object,
            "completion_tokens_details",
            "reasoning_tokens",
            "reasoning_tokens",
        )?,
    })
}

/// Parse Gemini `usageMetadata`, mirroring the python `_usage` normalizer:
/// cached tokens are an input subset, absent counts are zero (`require_integer`
/// parity), and `thoughtsTokenCount` stays unknown when omitted.
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
    Ok(Usage {
        input_tokens: Some(count_or_zero(
            object,
            "promptTokenCount",
            "Gemini promptTokenCount",
        )?),
        output_tokens: Some(count_or_zero(
            object,
            "candidatesTokenCount",
            "Gemini candidatesTokenCount",
        )?),
        cached_input_tokens: Some(count_or_zero(
            object,
            "cachedContentTokenCount",
            "Gemini cachedContentTokenCount",
        )?),
        reasoning_tokens,
    })
}

/// Parse Bedrock `metadata.usage`, mirroring the python `_usage` normalizer:
/// cache read and write legs fold into total input, cached input reports the
/// read leg, and absent counts are zero (`require_integer` parity). A total
/// beyond the representable count range is a provider contract violation and
/// fails the stream rather than persisting a clamped or wrapped number.
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
    let input_tokens = fresh
        .checked_add(cache_read)
        .and_then(|total| total.checked_add(cache_write))
        .ok_or_else(|| "Bedrock input token total overflows a 64-bit count".to_string())?;
    Ok(Usage {
        input_tokens: Some(input_tokens),
        output_tokens: Some(count_or_zero(
            usage,
            "outputTokens",
            "Bedrock outputTokens",
        )?),
        cached_input_tokens: Some(cache_read),
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

/// Fetch a required non-negative integer field from a provider JSON object.
pub fn require_u64(object: &Map<String, Value>, key: &str, label: &str) -> Result<u64, String> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label} must be a non-negative integer"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn openai_compatible_usage_counts_absent_fields_as_zero() {
        let usage = openai_compatible_usage(&json!({"prompt_tokens": 7})).expect("valid usage");
        assert_eq!(usage.input_tokens, Some(7));
        assert_eq!(usage.output_tokens, Some(0));
        assert_eq!(usage.cached_input_tokens, None);
        assert_eq!(usage.reasoning_tokens, None);
    }

    #[test]
    fn openai_compatible_usage_rejects_malformed_counts() {
        assert!(openai_compatible_usage(&json!({"prompt_tokens": "7"})).is_err());
        assert!(openai_compatible_usage(&json!([1])).is_err());
        assert!(openai_compatible_usage(
            &json!({"prompt_tokens": 1, "completion_tokens": 1, "prompt_tokens_details": 3})
        )
        .is_err());
    }

    #[test]
    fn bedrock_usage_folds_cache_legs_and_rejects_unrepresentable_totals() {
        let usage = bedrock_usage(Some(&json!({
            "inputTokens": 9,
            "outputTokens": 4,
            "cacheReadInputTokens": 2,
            "cacheWriteInputTokens": 1,
        })))
        .expect("valid usage");
        assert_eq!(usage.input_tokens, Some(12));
        assert_eq!(usage.cached_input_tokens, Some(2));
        // Individually valid legs whose sum is unrepresentable are a
        // provider contract violation, never a clamped or wrapped total.
        assert!(bedrock_usage(Some(&json!({
            "inputTokens": u64::MAX,
            "outputTokens": 1,
            "cacheReadInputTokens": 1,
        })))
        .is_err());
        assert!(bedrock_usage(Some(&json!(null))).is_err());
        assert!(bedrock_usage(None).is_err());
    }

    #[test]
    fn openai_usage_treats_absent_and_null_objects_as_unknown() {
        assert!(openai_usage(None).expect("absent is unknown").is_none());
        assert!(openai_usage(Some(&serde_json::Value::Null))
            .expect("null is unknown")
            .is_none());
        let usage = openai_usage(Some(&json!({
            "input_tokens": 2,
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 1},
        })))
        .expect("valid usage")
        .expect("usage present");
        assert_eq!(usage.reasoning_tokens, Some(1));
        assert_eq!(usage.cached_input_tokens, None);
    }
}
