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
    pub provider_item_id: Option<String>,
    pub provider_status: Option<ProviderOutputItemStatus>,
    /// Raw provider-order argument text: a validated JSON object for
    /// function calls, freeform text for custom (freeform) tool calls.
    pub raw_arguments: String,
    /// Whether this is a freeform custom tool call (Responses-only).
    pub custom: bool,
}

/// Provider-owned Responses output-item kind whose identity must remain exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderOutputItemKind {
    Reasoning,
    FunctionCall,
    CustomToolCall,
    Message,
}

/// Provider-owned Responses item lifecycle status preserved byte-for-byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderOutputItemStatus {
    InProgress,
    Completed,
    Incomplete,
}

impl ProviderOutputItemStatus {
    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "in_progress" => Some(Self::InProgress),
            "completed" => Some(Self::Completed),
            "incomplete" => Some(Self::Incomplete),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Incomplete => "incomplete",
        }
    }
}

/// Optional phase attached to an OpenAI Responses assistant message item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderAssistantMessagePhase {
    Commentary,
    FinalAnswer,
}

impl ProviderAssistantMessagePhase {
    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "commentary" => Some(Self::Commentary),
            "final_answer" => Some(Self::FinalAnswer),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Commentary => "commentary",
            Self::FinalAnswer => "final_answer",
        }
    }
}

/// One ordered provider-neutral stream event.
#[derive(Debug, Clone)]
pub enum Event {
    TextDelta(String),
    RefusalDelta(String),
    /// One text delta for a specific provider-owned assistant message item.
    ProviderTextDelta {
        output_index: u32,
        item_id: String,
        delta: String,
    },
    /// One refusal delta for a specific provider-owned assistant message item.
    ProviderRefusalDelta {
        output_index: u32,
        item_id: String,
        delta: String,
    },
    /// Reserve a public Responses slot at the provider's item-start boundary.
    ProviderOutputItemStarted {
        output_index: u32,
        item_id: Option<String>,
        kind: ProviderOutputItemKind,
        status: Option<ProviderOutputItemStatus>,
        phase: Option<ProviderAssistantMessagePhase>,
    },
    /// Close one provider-owned output item with its exact lifecycle metadata.
    ProviderOutputItemCompleted {
        output_index: u32,
        item_id: Option<String>,
        kind: ProviderOutputItemKind,
        status: Option<ProviderOutputItemStatus>,
        phase: Option<ProviderAssistantMessagePhase>,
    },
    ReasoningSummaryDelta {
        output_index: u32,
        summary_index: u32,
        item_id: String,
        delta: String,
    },
    /// Verbatim Anthropic extended-thinking text for one provider block.
    ThinkingDelta {
        index: u32,
        delta: String,
    },
    /// Opaque cryptographic signature closing one Anthropic thinking block;
    /// it must round-trip byte-exact or the provider rejects the replay.
    ThinkingSignature {
        index: u32,
        signature: String,
    },
    /// One complete opaque Anthropic redacted-thinking block.
    RedactedThinking {
        index: u32,
        data: String,
    },
    /// One opaque OpenAI Responses encrypted reasoning payload, keyed by its
    /// provider output-item index.
    EncryptedReasoning {
        output_index: u32,
        item_id: String,
        encrypted_content: String,
    },
    /// Opaque Fireworks Chat reasoning, bound to the exact issuing route.
    ReasoningContentDelta {
        route_sha256: String,
        delta: String,
    },
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
    /// One complete Anthropic server-tool content block carried verbatim.
    ///
    /// `server_tool_use` blocks fold their streamed `input_json_delta`
    /// fragments into the final `input`; result blocks arrive whole in their
    /// start frame. Server tools are admitted only onto Anthropic-only
    /// Messages routes, so exactly the Messages encoder re-emits the block;
    /// other surfaces can never legally receive it.
    ServerToolBlock {
        index: u32,
        block: Value,
    },
    Usage(Usage),
    Completed,
    Incomplete,
    /// The provider paused a long-running server-tool turn (`pause_turn`):
    /// the caller resends the conversation as-is to continue. Terminal for
    /// this attempt and billed like a completed turn.
    Paused,
    Failed(Failure),
}

impl Event {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Event::Completed | Event::Incomplete | Event::Paused | Event::Failed(_)
        )
    }

    /// Whether this event carries the first visible model output, used to
    /// stamp time-to-first-token. A content, refusal, reasoning, or tool-argument
    /// delta counts only when it carries at least one character: an empty delta
    /// (a role-establishing or empty refusal frame) is not a visible token and
    /// must not stamp TTFT early. A tool-call start is itself the first token of
    /// a tool-only turn, so it counts even before any arguments stream. Purely
    /// structural frames are excluded so TTFT is not stamped early: the Responses
    /// `ProviderOutputItemStarted` reserves a slot at the item-start boundary
    /// *before* the first delta arrives, and the opaque reasoning-carrier frames
    /// (`ThinkingSignature`, `RedactedThinking`, `EncryptedReasoning`) never lead
    /// a turn on their own. Usage, item-close, and lifecycle/terminal frames are
    /// not output tokens either.
    pub fn is_output_token(&self) -> bool {
        match self {
            Event::TextDelta(text) | Event::RefusalDelta(text) => !text.is_empty(),
            Event::ProviderTextDelta { delta, .. }
            | Event::ProviderRefusalDelta { delta, .. }
            | Event::ReasoningSummaryDelta { delta, .. }
            | Event::ThinkingDelta { delta, .. }
            | Event::ReasoningContentDelta { delta, .. }
            | Event::ToolArgumentsDelta { delta, .. } => !delta.is_empty(),
            Event::ToolCallStarted { .. } => true,
            // A server-tool block is visible model output and may lead a
            // search-only turn, so it stamps TTFT like a tool-call start.
            Event::ServerToolBlock { .. } => true,
            _ => false,
        }
    }
}

/// Render one event as the content-bearing JSON object used by dialect parity
/// fixtures: the same field vocabulary the fixture-event parser accepts, plus
/// the failure class and safe message for terminal failures.
pub fn simplified_event(event: &Event) -> Value {
    match event {
        Event::TextDelta(text) => serde_json::json!({"kind": "text_delta", "text": text}),
        Event::RefusalDelta(text) => serde_json::json!({"kind": "refusal_delta", "text": text}),
        Event::ProviderTextDelta {
            output_index,
            item_id,
            delta,
        } => serde_json::json!({
            "kind": "provider_text_delta",
            "output_index": output_index,
            "item_id": item_id,
            "text": delta,
        }),
        Event::ProviderRefusalDelta {
            output_index,
            item_id,
            delta,
        } => serde_json::json!({
            "kind": "provider_refusal_delta",
            "output_index": output_index,
            "item_id": item_id,
            "text": delta,
        }),
        Event::ProviderOutputItemStarted {
            output_index,
            item_id,
            kind,
            status,
            phase,
        } => {
            let mut payload = serde_json::json!({
                "kind": "provider_output_item_started",
                "output_index": output_index,
                "item_type": match kind {
                    ProviderOutputItemKind::Reasoning => "reasoning",
                    ProviderOutputItemKind::FunctionCall => "function_call",
                    ProviderOutputItemKind::CustomToolCall => "custom_tool_call",
                    ProviderOutputItemKind::Message => "message",
                },
            });
            add_provider_item_metadata(&mut payload, item_id, *status, *phase);
            payload
        }
        Event::ProviderOutputItemCompleted {
            output_index,
            item_id,
            kind,
            status,
            phase,
        } => {
            let mut payload = serde_json::json!({
                "kind": "provider_output_item_completed",
                "output_index": output_index,
                "item_type": match kind {
                ProviderOutputItemKind::Reasoning => "reasoning",
                ProviderOutputItemKind::FunctionCall => "function_call",
                ProviderOutputItemKind::CustomToolCall => "custom_tool_call",
                ProviderOutputItemKind::Message => "message",
                },
            });
            add_provider_item_metadata(&mut payload, item_id, *status, *phase);
            payload
        }
        Event::ReasoningSummaryDelta {
            output_index,
            summary_index,
            item_id,
            delta,
        } => serde_json::json!({
            "kind": "reasoning_summary_delta",
            "output_index": output_index,
            "summary_index": summary_index,
            "item_id": item_id,
            "text": delta,
        }),
        Event::ThinkingDelta { index, delta } => serde_json::json!({
            "kind": "thinking_delta",
            "index": index,
            "text": delta,
        }),
        Event::ThinkingSignature { index, signature } => serde_json::json!({
            "kind": "thinking_signature",
            "index": index,
            "signature": signature,
        }),
        Event::RedactedThinking { index, data } => serde_json::json!({
            "kind": "redacted_thinking",
            "index": index,
            "data": data,
        }),
        Event::EncryptedReasoning {
            output_index,
            item_id,
            encrypted_content,
        } => serde_json::json!({
            "kind": "encrypted_reasoning",
            "output_index": output_index,
            "item_id": item_id,
            "encrypted_content": encrypted_content,
        }),
        Event::ReasoningContentDelta {
            route_sha256,
            delta,
        } => serde_json::json!({
            "kind": "reasoning_content_delta",
            "route_sha256": route_sha256,
            "text": delta,
        }),
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
        Event::ToolCallCompleted { index, call } => {
            let mut payload = serde_json::json!({
                "kind": "tool_call_completed",
                "index": index,
                "call_id": call.call_id,
                "name": call.name,
                "raw_arguments": call.raw_arguments,
            });
            if let Some(item_id) = &call.provider_item_id {
                payload["item_id"] = Value::String(item_id.clone());
            }
            if let Some(status) = call.provider_status {
                payload["status"] = Value::String(status.as_str().to_string());
            }
            payload
        }
        Event::Usage(usage) => serde_json::json!({
            "kind": "usage",
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
        }),
        Event::ServerToolBlock { index, block } => serde_json::json!({
            "kind": "server_tool_block",
            "index": index,
            "block": block,
        }),
        Event::Completed => serde_json::json!({"kind": "completed"}),
        Event::Incomplete => serde_json::json!({"kind": "incomplete"}),
        Event::Paused => serde_json::json!({"kind": "paused"}),
        Event::Failed(failure) => serde_json::json!({
            "kind": "failed",
            "failure_class": failure.failure_class.as_str(),
            "safe_message": failure.safe_message,
        }),
    }
}

fn add_provider_item_metadata(
    payload: &mut Value,
    item_id: &Option<String>,
    status: Option<ProviderOutputItemStatus>,
    phase: Option<ProviderAssistantMessagePhase>,
) {
    if let Some(item_id) = item_id {
        payload["item_id"] = Value::String(item_id.clone());
    }
    if let Some(status) = status {
        payload["status"] = Value::String(status.as_str().to_string());
    }
    if let Some(phase) = phase {
        payload["phase"] = Value::String(phase.as_str().to_string());
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
    pub provider_item_id: Option<String>,
    pub provider_status: Option<ProviderOutputItemStatus>,
    pub raw_arguments: String,
    pub completed: bool,
    pub custom: bool,
}

impl ToolAccumulator {
    pub fn new(call_id: String, name: String) -> Self {
        Self {
            call_id,
            name,
            provider_item_id: None,
            provider_status: None,
            raw_arguments: String::new(),
            completed: false,
            custom: false,
        }
    }

    pub fn complete(&self) -> Result<CompletedToolCall, String> {
        if !self.custom {
            // Custom (freeform) tool input is opaque text by contract; only
            // function arguments must parse as one JSON object.
            require_json_object_text(&self.raw_arguments)?;
        }
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
            provider_item_id: self.provider_item_id.clone(),
            provider_status: self.provider_status,
            raw_arguments: self.raw_arguments.clone(),
            custom: self.custom,
        })
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
/// read leg, and absent counts are zero (`require_integer` parity). Legs and
/// the folded total beyond the persistable ledger range are provider
/// contract violations and fail the stream rather than reaching settlement
/// as a value the ledger could never write.
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
        .filter(|total| *total <= MAXIMUM_LEDGER_COUNT)
        .ok_or_else(|| "Bedrock input token total overflows a persistable count".to_string())?;
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
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn output_tokens_lead_a_turn_but_control_frames_do_not() {
        // Content, reasoning, and tool-call deltas are the first visible output.
        assert!(Event::TextDelta("hi".to_string()).is_output_token());
        assert!(Event::RefusalDelta("no".to_string()).is_output_token());
        assert!(Event::ProviderTextDelta {
            output_index: 0,
            item_id: "msg_1".to_string(),
            delta: "hi".to_string(),
        }
        .is_output_token());
        assert!(Event::ThinkingDelta {
            index: 0,
            delta: "hmm".to_string(),
        }
        .is_output_token());
        // A tool-only turn's first token is the tool call itself.
        assert!(Event::ToolCallStarted {
            index: 0,
            call_id: "call_1".to_string(),
            name: "get".to_string(),
        }
        .is_output_token());
        // Usage, terminals, and opaque reasoning-carrier frames never lead.
        assert!(!Event::Usage(Usage::default()).is_output_token());
        assert!(!Event::Completed.is_output_token());
        assert!(!Event::Incomplete.is_output_token());
        assert!(!Event::ThinkingSignature {
            index: 0,
            signature: "sig".to_string(),
        }
        .is_output_token());
        // A Responses item-start reserves a slot before the first delta; it
        // must not stamp TTFT early -- the following delta is the real token.
        assert!(!Event::ProviderOutputItemStarted {
            output_index: 0,
            item_id: Some("msg_1".to_string()),
            kind: ProviderOutputItemKind::Message,
            status: None,
            phase: None,
        }
        .is_output_token());
        // An empty delta (role-establishing or empty refusal frame) carries no
        // visible token, so it must not stamp TTFT.
        assert!(!Event::TextDelta(String::new()).is_output_token());
        assert!(!Event::RefusalDelta(String::new()).is_output_token());
        assert!(!Event::ProviderTextDelta {
            output_index: 0,
            item_id: "msg_1".to_string(),
            delta: String::new(),
        }
        .is_output_token());
    }

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
        assert!(
            openai_compatible_usage(&json!({"prompt_tokens": MAXIMUM_LEDGER_COUNT + 1})).is_err()
        );
        assert!(openai_compatible_usage(&json!([1])).is_err());
        assert!(openai_compatible_usage(
            &json!({"prompt_tokens": 1, "completion_tokens": 1, "prompt_tokens_details": 3})
        )
        .is_err());
    }

    #[test]
    fn parsed_counts_are_bounded_to_the_persistable_ledger_range() {
        let at_bound = json!({"count": MAXIMUM_LEDGER_COUNT});
        let over_bound = json!({"count": MAXIMUM_LEDGER_COUNT + 1});
        let at_object = at_bound.as_object().expect("object");
        let over_object = over_bound.as_object().expect("object");
        // Exactly i64::MAX is persistable and accepted; one past it is a
        // provider contract violation everywhere counts are parsed.
        assert_eq!(
            count_or_zero(at_object, "count", "count"),
            Ok(MAXIMUM_LEDGER_COUNT)
        );
        assert!(count_or_zero(over_object, "count", "count").is_err());
        assert_eq!(
            require_u64(at_object, "count", "count"),
            Ok(MAXIMUM_LEDGER_COUNT)
        );
        assert!(require_u64(over_object, "count", "count").is_err());
        assert!(openai_usage(Some(&json!({
            "input_tokens": 1,
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": MAXIMUM_LEDGER_COUNT + 1},
        })))
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
        // A leg beyond the persistable ledger range fails at the parser.
        assert!(bedrock_usage(Some(&json!({
            "inputTokens": MAXIMUM_LEDGER_COUNT + 1,
            "outputTokens": 1,
        })))
        .is_err());
        // Individually persistable legs whose folded total is not are a
        // provider contract violation, never a clamped or wrapped total.
        assert!(bedrock_usage(Some(&json!({
            "inputTokens": MAXIMUM_LEDGER_COUNT,
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
