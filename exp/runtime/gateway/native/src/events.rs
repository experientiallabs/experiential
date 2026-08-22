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
fn count_or_zero(object: &Map<String, Value>, key: &str, label: &str) -> Result<u64, String> {
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
