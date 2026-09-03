//! Provider-neutral stream events, the Rust mirror of `GatewayEvent`.
//!
//! # Usage contract
//!
//! Every usage mapper in this module emits OpenAI subset semantics:
//! `reasoning_tokens` (when known) counts a SUBSET of `output_tokens`, and
//! `cached_input_tokens` a subset of `input_tokens`. Settlement prices the
//! reasoning subset at the reasoning rate and the remainder of `output_tokens`
//! at the output rate, so a wire that reports reasoning OUTSIDE its output
//! total would bill every reasoning token at zero unless the mapper folds it
//! back in. Per wire:
//!
//! - OpenAI Responses (`openai_usage`): `output_tokens_details.reasoning_tokens`
//!   is a documented subset of `output_tokens`; forwarded as reported.
//! - Chat Completions (`openai_compatible_usage`): OpenAI, OpenRouter, DeepSeek,
//!   Fireworks, and Azure-relayed Fireworks fold reasoning into
//!   `completion_tokens`; xAI (native and relayed by Azure Foundry) reports it
//!   outside. A deployment declaring `reasoning_tokens_additive` folds every
//!   reported count; without the declaration a reasoning count above
//!   `completion_tokens`, impossible under subset semantics, is taken as proof
//!   of an additive provider and folded. A folded provider is never touched.
//! - Gemini (`gemini_usage`): `thoughtsTokenCount` is additive by Google's
//!   definition (`totalTokenCount` = prompt + candidates + thoughts), so it is
//!   folded into `output_tokens` unconditionally.
//! - Anthropic Messages and Bedrock Converse: thinking is billed inside the
//!   provider's `output_tokens` and no separate count is published, so
//!   `reasoning_tokens` stays `None` and the total is forwarded as reported.
//!
//! A fold whose total leaves the persistable ledger range is a provider
//! contract violation and fails the stream; totals are never clamped.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::errors::Failure;

/// Normalized token usage mirroring `GatewayUsage` semantics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
    /// Cache-write tokens inside the input total, present only when the
    /// provider reported a nonzero count (Anthropic-only today). The ledger
    /// keeps billing the folded input total; this leg exists so callers see
    /// their prompt being cached (Claude Code displays it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
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
    /// One provider-executed Anthropic server tool invocation opening
    /// (`server_tool_use`); the provider runs the tool itself, so these
    /// never become client tool calls or affect the tool-use stop reason.
    ServerToolUseStarted {
        index: u32,
        call_id: String,
        name: String,
    },
    /// Raw provider-order input fragment for one open server tool use.
    ServerToolArgumentsDelta {
        index: u32,
        delta: String,
    },
    /// One completed server tool invocation with its validated input text.
    ServerToolUseCompleted {
        index: u32,
        call: CompletedToolCall,
    },
    /// One whole verbatim Anthropic server-tool result content block
    /// (`web_search_tool_result`), carried as compact JSON text: the result
    /// arrives complete in its start frame and must reach the caller intact.
    ServerToolResult {
        index: u32,
        block: String,
    },
    /// Provider text content-block boundary on the Anthropic wire. Emitted
    /// before that block's first text delta so the Messages encoder can
    /// mirror the provider's block structure (citations attach per block);
    /// encoders without a block concept ignore it.
    TextBlockStarted {
        index: u32,
    },
    /// One whole verbatim citation object attached to the open Anthropic
    /// text block (`citations_delta`), carried as compact JSON text.
    CitationDelta {
        index: u32,
        citation: String,
    },
    Usage(Usage),
    Completed,
    Incomplete,
    /// Anthropic `pause_turn` terminal: the provider paused a long-running
    /// server-tool turn and expects the caller to resend the conversation to
    /// continue it. Settlement treats it like a completed turn; the Messages
    /// encoder must preserve the stop reason or the caller never resumes.
    PausedTurn,
    Failed(Failure),
}

impl Event {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Event::Completed | Event::Incomplete | Event::PausedTurn | Event::Failed(_)
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
            | Event::ToolArgumentsDelta { delta, .. }
            | Event::ServerToolArgumentsDelta { delta, .. } => !delta.is_empty(),
            Event::ToolCallStarted { .. } | Event::ServerToolUseStarted { .. } => true,
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
        Event::ServerToolUseStarted {
            index,
            call_id,
            name,
        } => serde_json::json!({
            "kind": "server_tool_use_started",
            "index": index,
            "call_id": call_id,
            "name": name,
        }),
        Event::ServerToolArgumentsDelta { index, delta } => serde_json::json!({
            "kind": "server_tool_arguments_delta",
            "index": index,
            "text": delta,
        }),
        Event::ServerToolUseCompleted { index, call } => serde_json::json!({
            "kind": "server_tool_use_completed",
            "index": index,
            "call_id": call.call_id,
            "name": call.name,
            "raw_arguments": call.raw_arguments,
        }),
        Event::ServerToolResult { index, block } => serde_json::json!({
            "kind": "server_tool_result",
            "index": index,
            "block": block,
        }),
        Event::TextBlockStarted { index } => serde_json::json!({
            "kind": "text_block_started",
            "index": index,
        }),
        Event::CitationDelta { index, citation } => serde_json::json!({
            "kind": "citation_delta",
            "index": index,
            "citation": citation,
        }),
        Event::Usage(usage) => {
            let mut payload = serde_json::json!({
                "kind": "usage",
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
                "reasoning_tokens": usage.reasoning_tokens,
            });
            if let Some(creation) = usage.cache_creation_input_tokens {
                payload["cache_creation_input_tokens"] = serde_json::json!(creation);
            }
            payload
        }
        Event::Completed => serde_json::json!({"kind": "completed"}),
        Event::Incomplete => serde_json::json!({"kind": "incomplete"}),
        Event::PausedTurn => serde_json::json!({"kind": "paused_turn"}),
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
    /// Whether this is a provider-executed Anthropic server tool
    /// (`server_tool_use`), whose lifecycle events stay on the dedicated
    /// server-tool variants and never count toward the tool-use stop reason.
    pub server: bool,
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
            server: false,
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
        cache_creation_input_tokens: None,
        reasoning_tokens: optional_usage_detail(
            object,
            "output_tokens_details",
            "reasoning_tokens",
            "OpenAI reasoning_tokens",
        )?,
    }))
}

/// Fold a reasoning count the provider reports OUTSIDE its output total back
/// into that total, so the emitted usage satisfies the reasoning-subset
/// contract (see the module documentation). A folded total beyond the
/// persistable ledger range is a provider contract violation, never a clamped
/// or wrapped total.
fn fold_additive_reasoning(
    output_tokens: u64,
    reasoning_tokens: u64,
    label: &str,
) -> Result<u64, String> {
    output_tokens
        .checked_add(reasoning_tokens)
        .filter(|total| *total <= MAXIMUM_LEDGER_COUNT)
        .ok_or_else(|| format!("{label} output token total overflows a persistable count"))
}

/// Parse a Chat Completions usage object: a malformed object fails the stream
/// instead of silently dropping token accounting.
///
/// `completion_tokens_details.reasoning_tokens` is a subset of
/// `completion_tokens` on OpenAI and every provider that mirrors it, and is
/// forwarded as reported. A provider that reports reasoning additively (xAI,
/// natively or relayed by Azure Foundry) has its count folded into
/// `output_tokens` while `reasoning_tokens` keeps naming the subset. The fold
/// applies when the deployment declares `declared_additive`
/// (`GatewayDeploymentCapabilities.reasoning_tokens_additive`) or when the
/// reported reasoning count is ABOVE `completion_tokens`, which cannot occur
/// under subset semantics and so identifies an additive provider on its own.
/// An undeclared additive provider is missed only when its reasoning happens
/// to be no larger than its visible answer.
pub fn openai_compatible_usage(value: &Value, declared_additive: bool) -> Result<Usage, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "OpenAI-compatible usage must be an object".to_string())?;
    let completion_tokens = count_or_zero(object, "completion_tokens", "completion_tokens")?;
    let reasoning_tokens = optional_usage_detail(
        object,
        "completion_tokens_details",
        "reasoning_tokens",
        "reasoning_tokens",
    )?;
    let output_tokens = match reasoning_tokens {
        Some(reasoning) if declared_additive || reasoning > completion_tokens => {
            fold_additive_reasoning(completion_tokens, reasoning, "OpenAI-compatible")?
        }
        _ => completion_tokens,
    };
    Ok(Usage {
        input_tokens: Some(count_or_zero(object, "prompt_tokens", "prompt_tokens")?),
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
        Some(reasoning) => fold_additive_reasoning(candidates_tokens, reasoning, "Gemini")?,
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

#[cfg(test)]
#[path = "events/tests.rs"]
mod tests;
