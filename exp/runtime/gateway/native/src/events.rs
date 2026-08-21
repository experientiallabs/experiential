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
        matches!(self, Event::Completed | Event::Incomplete | Event::Failed(_))
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
    pub index: u32,
    pub call_id: String,
    pub name: String,
    pub raw_arguments: String,
    pub completed: bool,
}

impl ToolAccumulator {
    pub fn new(index: u32, call_id: String, name: String) -> Self {
        Self {
            index,
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

/// Parse an OpenAI-shaped usage object from a terminal Responses payload,
/// mirroring `streaming_usage.openai_usage`.
pub fn openai_usage(value: Option<&Value>) -> Option<Usage> {
    let object = value?.as_object()?;
    let input = object.get("input_tokens")?.as_u64()?;
    let output = object.get("output_tokens")?.as_u64()?;
    let cached = object
        .get("input_tokens_details")
        .and_then(Value::as_object)
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64);
    let reasoning = object
        .get("output_tokens_details")
        .and_then(Value::as_object)
        .and_then(|details| details.get("reasoning_tokens"))
        .and_then(Value::as_u64);
    Some(Usage {
        input_tokens: Some(input),
        output_tokens: Some(output),
        cached_input_tokens: cached,
        reasoning_tokens: reasoning,
    })
}

/// Parse a Chat Completions usage object, mirroring
/// `streaming_usage.openai_compatible_usage`.
pub fn openai_compatible_usage(value: &Value) -> Option<Usage> {
    let object = value.as_object()?;
    let prompt = object.get("prompt_tokens")?.as_u64()?;
    let completion = object.get("completion_tokens")?.as_u64()?;
    let cached = object
        .get("prompt_tokens_details")
        .and_then(Value::as_object)
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64);
    let reasoning = object
        .get("completion_tokens_details")
        .and_then(Value::as_object)
        .and_then(|details| details.get("reasoning_tokens"))
        .and_then(Value::as_u64);
    Some(Usage {
        input_tokens: Some(prompt),
        output_tokens: Some(completion),
        cached_input_tokens: cached,
        reasoning_tokens: reasoning,
    })
}

/// Fetch a required string field from a provider JSON object.
pub fn require_string(object: &Map<String, Value>, key: &str, label: &str) -> Result<String, String> {
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
