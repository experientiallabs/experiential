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
//! - OpenAI-shaped wires (Responses via `openai_usage`, Chat Completions via
//!   `openai_compatible_usage`): OpenAI, OpenRouter, DeepSeek, and Fireworks
//!   report reasoning inside the output total; xAI (native and relayed by
//!   Azure Foundry) reports it outside on both wires. The provider's own
//!   `total_tokens` decides: `input + output` is the subset shape and is
//!   forwarded as reported, `input + output + reasoning` is the additive shape
//!   and folds (`fold_openai_shaped_reasoning`). Without a decisive total, a
//!   reasoning count above the output total is impossible under subset
//!   semantics and folds.
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
use serde_json::Value;

use crate::errors::Failure;

/// Normalized token usage mirroring `GatewayUsage` semantics.
///
/// `input_tokens` is EVERY prompt token the provider bills at some rate:
/// wires that report their cache legs beside a fresh-input count (Anthropic
/// Messages `input_tokens` + `cache_read_input_tokens` +
/// `cache_creation_input_tokens`, Bedrock Converse `inputTokens` +
/// `cacheReadInputTokens` + `cacheWriteInputTokens`) are folded here, and
/// wires whose total already includes them (OpenAI `prompt_tokens` /
/// `input_tokens`, Gemini `promptTokenCount`) pass through. Both
/// `cached_input_tokens` (read leg) and `cache_creation_input_tokens` (write
/// leg) are therefore disjoint SUBSETS of `input_tokens` on every wire, and a
/// consumer prices `input - cached - creation` as fresh input, the read leg at
/// the cached rate and the write leg at the cache-write rate.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
    /// Cache-write tokens inside the input total, exactly as the wire reported
    /// them, zero included: Anthropic `cache_creation_input_tokens` (the sum of
    /// its 5-minute and 1-hour `cache_creation` legs), Bedrock
    /// `cacheWriteInputTokens`, OpenAI `prompt_tokens_details.cache_write_tokens`
    /// (Chat) / `input_tokens_details.cache_write_tokens` (Responses). `None`
    /// means the wire gave NO usable write count (Gemini, OpenAI models and
    /// compatible relays that omit the detail) — never "reported zero" — so a
    /// consumer may approximate only a `None` leg and must price a reported
    /// `0` as zero writes. Rides the settle callback
    /// (`cache_creation_input_tokens`) and the client usage blocks (Claude
    /// Code displays it).
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
    /// Nested tool tree (Responses `namespace`) that declared this call,
    /// preserved verbatim through retention and the client stream because
    /// the provider rejects a namespaced call replayed without it.
    pub namespace: Option<String>,
    /// Opaque SDK 3.0 `caller` attribution (for example
    /// `{"type": "program", "id": ...}`) naming the program that invoked
    /// this call; carried verbatim like `namespace` so the item
    /// round-trips exactly as the provider emitted it.
    pub caller: Option<Value>,
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
        /// Nested tool tree (Responses `namespace`) that declared this call;
        /// present only on native Responses streams and preserved verbatim
        /// because the provider rejects a namespaced call replayed without it.
        namespace: Option<String>,
        /// Opaque SDK 3.0 `caller` attribution carried verbatim like
        /// `namespace`.
        caller: Option<Value>,
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
    /// One OpenAI Responses hosted-tool output item opening
    /// (`web_search_call`, `mcp_call`, `code_interpreter_call`, ...). The
    /// provider executes the tool itself and owns the item's shape, so the
    /// whole item is carried verbatim as compact JSON: the caller (and its
    /// next-turn echo) must see exactly what the provider produced.
    HostedToolItemStarted {
        output_index: u32,
        item_id: String,
        item_type: String,
        item: String,
    },
    /// One verbatim per-type lifecycle or delta frame for an open hosted
    /// tool item (`response.web_search_call.searching`,
    /// `response.mcp_call_arguments.delta`, ...), carried as compact JSON.
    /// The Responses encoder re-stamps only the public output index and
    /// sequence number; every other payload field passes through untouched.
    HostedToolItemProgress {
        output_index: u32,
        item_id: String,
        event_type: String,
        payload: String,
    },
    /// One completed hosted tool item with its final verbatim JSON, from the
    /// provider's `response.output_item.done` (or the last-seen item when the
    /// terminal response arrived first).
    HostedToolItemCompleted {
        output_index: u32,
        item_id: String,
        item_type: String,
        item: String,
    },
    /// One whole verbatim OpenAI Responses output-text annotation
    /// (`response.output_text.annotation.added`: URL citations from hosted
    /// web search), carried as compact JSON and attached to the open
    /// provider-owned assistant message item.
    ProviderTextAnnotation {
        output_index: u32,
        item_id: String,
        annotation: String,
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
    /// The gateway cut the stream at one of the caller's stop sequences on a
    /// wire that has no stop field (OpenAI Responses). Settles as completed;
    /// the Messages encoder reports `stop_sequence` with this exact value.
    StoppedAtSequence(String),
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
            Event::Completed
                | Event::Incomplete
                | Event::StoppedAtSequence(_)
                | Event::PausedTurn
                | Event::Failed(_)
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
            Event::ToolCallStarted { .. }
            | Event::ServerToolUseStarted { .. }
            // A hosted tool item start is the provider beginning visible
            // work, the same first-token signal as a tool-call start.
            | Event::HostedToolItemStarted { .. } => true,
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
            namespace,
            caller,
        } => {
            let mut payload = serde_json::json!({
                "kind": "tool_call_started",
                "index": index,
                "call_id": call_id,
                "name": name,
            });
            if let Some(namespace) = namespace {
                payload["namespace"] = Value::String(namespace.clone());
            }
            if let Some(caller) = caller {
                payload["caller"] = caller.clone();
            }
            payload
        }
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
            if let Some(namespace) = &call.namespace {
                payload["namespace"] = Value::String(namespace.clone());
            }
            if let Some(caller) = &call.caller {
                payload["caller"] = caller.clone();
            }
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
        Event::HostedToolItemStarted {
            output_index,
            item_id,
            item_type,
            item,
        } => serde_json::json!({
            "kind": "hosted_tool_item_started",
            "output_index": output_index,
            "item_id": item_id,
            "item_type": item_type,
            "item": item,
        }),
        Event::HostedToolItemProgress {
            output_index,
            item_id,
            event_type,
            payload,
        } => serde_json::json!({
            "kind": "hosted_tool_item_progress",
            "output_index": output_index,
            "item_id": item_id,
            "event_type": event_type,
            "payload": payload,
        }),
        Event::HostedToolItemCompleted {
            output_index,
            item_id,
            item_type,
            item,
        } => serde_json::json!({
            "kind": "hosted_tool_item_completed",
            "output_index": output_index,
            "item_id": item_id,
            "item_type": item_type,
            "item": item,
        }),
        Event::ProviderTextAnnotation {
            output_index,
            item_id,
            annotation,
        } => serde_json::json!({
            "kind": "provider_text_annotation",
            "output_index": output_index,
            "item_id": item_id,
            "annotation": annotation,
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
        // A stop-sequence cut is a completed turn to every python consumer
        // (guardrails, retention); only the public encoders name the sequence.
        Event::Completed | Event::StoppedAtSequence(_) => serde_json::json!({"kind": "completed"}),
        Event::Incomplete => serde_json::json!({"kind": "incomplete"}),
        Event::PausedTurn => serde_json::json!({"kind": "paused_turn"}),
        Event::Failed(failure) => {
            let mut value = serde_json::json!({
                "kind": "failed",
                "failure_class": failure.failure_class.as_str(),
                "safe_message": failure.safe_message,
            });
            if let Some(reason) = failure.refusal_reason {
                value["refusal_reason"] = serde_json::json!(reason.as_str());
            }
            value
        }
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

/// Whether one hosted Responses item type names a tool INVOCATION.
///
/// Only invocations join the ledger's tool names: the hosted union also
/// carries results (`*_call_output`), approvals, listings, and opaque
/// conversation items (`additional_tools`, `compaction`), and recording one
/// of those would report a tool call that never occurred.
pub fn hosted_item_type_is_invocation(item_type: &str) -> bool {
    item_type.ends_with("_call")
}

/// Validate one raw tool-argument accumulation as a single JSON object.
///
/// The parse-failure reason carries serde's positional description (token
/// category and line/column, never input bytes), so an unparsable shape is
/// diagnosable from the boundary log without ever logging payload.
pub fn require_json_object_text(raw: &str) -> Result<(), String> {
    match serde_json::from_str::<Value>(raw) {
        Ok(Value::Object(_)) => Ok(()),
        Ok(_) => Err("streamed tool arguments must decode to an object".to_string()),
        Err(error) => Err(format!(
            "streamed tool arguments are not valid JSON: {error}"
        )),
    }
}

/// Incremental scan of one JSON-argument accumulation that knows the byte at
/// which the top-level value closed.
///
/// A tool call's arguments are one JSON object, so nothing a provider streams
/// after the byte that closes it can be argument content: it is either an
/// extra delta the shim never should have sent or noise. The scan is a
/// constant-time-per-byte bracket/string tracker (never a re-parse of the
/// whole accumulation), exact for any well-formed prefix; a malformed prefix
/// simply never closes and keeps the strict parse at completion.
#[derive(Debug, Clone, Default)]
pub struct JsonValueScan {
    depth: u32,
    in_string: bool,
    escaped: bool,
    /// Whether the top-level value has closed (depth returned to zero after
    /// a container opened).
    pub closed: bool,
}

impl JsonValueScan {
    /// Feed one fragment; returns the byte offset within it at which the
    /// top-level value closed (exclusive, i.e. the first byte of the tail),
    /// or `None` when the fragment left the value open. Only ASCII
    /// structural bytes advance the scan, so the offset is always a char
    /// boundary.
    pub fn feed(&mut self, fragment: &str) -> Option<usize> {
        if self.closed {
            return Some(0);
        }
        for (offset, byte) in fragment.bytes().enumerate() {
            if self.in_string {
                if self.escaped {
                    self.escaped = false;
                } else if byte == b'\\' {
                    self.escaped = true;
                } else if byte == b'"' {
                    self.in_string = false;
                }
                continue;
            }
            match byte {
                b'"' => self.in_string = true,
                b'{' | b'[' => self.depth += 1,
                b'}' | b']' if self.depth > 0 => {
                    self.depth -= 1;
                    if self.depth == 0 {
                        self.closed = true;
                        return Some(offset + 1);
                    }
                }
                _ => {}
            }
        }
        None
    }
}

/// Why a tail streamed after a complete argument object was dropped rather
/// than failing the call. Each shape reproduces exactly one parse, so no
/// argument content is ever invented or chosen between alternatives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedundantTail {
    /// Only whitespace (a pretty-printed buffer's trailing newline).
    Whitespace,
    /// One or more exact repetitions of the complete value (`{}{}`, a
    /// duplicated whole-object delta).
    DuplicateValue,
    /// Empty JSON literals after a zero-argument call (`{}""`: Azure
    /// Foundry's DeepSeek shim, live 2026-09-10, 222 attempts in one day).
    EmptyLiterals,
}

impl RedundantTail {
    fn as_str(self) -> &'static str {
        match self {
            RedundantTail::Whitespace => "whitespace",
            RedundantTail::DuplicateValue => "duplicate_value",
            RedundantTail::EmptyLiterals => "empty_literals",
        }
    }
}

/// Classify the bytes a provider streamed AFTER its argument object closed.
///
/// Accepts only tails whose removal is unambiguous: whitespace; exact
/// repetitions of the whole value; and, after a zero-argument `{}` only,
/// empty literals (`""`, `{}`, `[]`) that carry no argument content. A tail
/// that is merely a suffix of the value (`{"a":1}}`) is NOT accepted: the same
/// bytes arise from a dropped inner delta (`{"a":1,"b":{` lost from
/// `{"a":1,"b":{"c":2}}`), so the parse would be a guess.
pub fn redundant_tail(value: &str, tail: &str) -> Option<RedundantTail> {
    let value = value.trim();
    let tail = tail.trim();
    if tail.is_empty() {
        return Some(RedundantTail::Whitespace);
    }
    let mut rest = tail;
    let mut duplicates = true;
    while !rest.is_empty() {
        match rest.strip_prefix(value) {
            Some(after) if !value.is_empty() => rest = after.trim_start(),
            _ => {
                duplicates = false;
                break;
            }
        }
    }
    if duplicates {
        return Some(RedundantTail::DuplicateValue);
    }
    if value != "{}" {
        return None;
    }
    let mut rest = tail;
    while !rest.is_empty() {
        rest = rest
            .strip_prefix("\"\"")
            .or_else(|| rest.strip_prefix("{}"))
            .or_else(|| rest.strip_prefix("[]"))?
            .trim_start();
    }
    Some(RedundantTail::EmptyLiterals)
}

/// Accumulated per-stream state for one incrementally emitted function call.
#[derive(Debug, Clone)]
pub struct ToolAccumulator {
    pub call_id: String,
    pub name: String,
    /// Nested tool tree (Responses `namespace`) that declared this call.
    pub namespace: Option<String>,
    /// Opaque SDK 3.0 `caller` attribution carried verbatim.
    pub caller: Option<Value>,
    pub provider_item_id: Option<String>,
    pub provider_status: Option<ProviderOutputItemStatus>,
    pub raw_arguments: String,
    pub completed: bool,
    pub custom: bool,
    /// Whether this is a provider-executed Anthropic server tool
    /// (`server_tool_use`), whose lifecycle events stay on the dedicated
    /// server-tool variants and never count toward the tool-use stop reason.
    pub server: bool,
    /// Scan of `raw_arguments` for dialects that accumulate through
    /// [`ToolAccumulator::push_arguments`]; dialects that append directly
    /// leave it untouched and never withhold anything.
    scan: JsonValueScan,
    /// Bytes streamed after the argument object closed, never emitted to the
    /// caller; reconciled by [`ToolAccumulator::complete`].
    pub withheld_tail: String,
}

/// Opaque tool IDs share the Python model bound, including signature carriers.
const MAXIMUM_TOOL_CALL_ID_CHARACTERS: usize = 65_536;

impl ToolAccumulator {
    pub fn new(call_id: String, name: String) -> Self {
        Self {
            call_id,
            name,
            namespace: None,
            caller: None,
            provider_item_id: None,
            provider_status: None,
            raw_arguments: String::new(),
            completed: false,
            custom: false,
            server: false,
            scan: JsonValueScan::default(),
            withheld_tail: String::new(),
        }
    }

    /// Append one streamed argument fragment, returning the part the caller
    /// may see: everything up to and including the byte that closes the
    /// argument object. Whatever follows that byte is withheld (never
    /// emitted) and judged at completion, so the deltas a client receives
    /// always concatenate to the completed call's bytes. Custom (freeform)
    /// input is opaque text and passes through whole.
    pub fn push_arguments(&mut self, fragment: &str) -> Option<String> {
        if self.custom {
            self.raw_arguments.push_str(fragment);
            return Some(fragment.to_string());
        }
        match self.scan.feed(fragment) {
            None => {
                self.raw_arguments.push_str(fragment);
                Some(fragment.to_string())
            }
            Some(closed_at) => {
                let (value, tail) = fragment.split_at(closed_at);
                self.raw_arguments.push_str(value);
                self.withheld_tail.push_str(tail);
                (!value.is_empty()).then(|| value.to_string())
            }
        }
    }

    pub fn complete(&self) -> Result<CompletedToolCall, String> {
        if !self.custom {
            if !self.withheld_tail.is_empty() {
                // A provider streamed bytes after its argument object closed.
                // Only a content-free tail is dropped (its shape reaches the
                // operator log; never its bytes); anything else is validated
                // as the concatenation the provider actually sent, so the
                // failure names the same parse position it always did.
                match redundant_tail(&self.raw_arguments, &self.withheld_tail) {
                    Some(RedundantTail::Whitespace) => {}
                    Some(shape) => {
                        let line = serde_json::json!({
                            "event": "tool_arguments_tail_dropped",
                            "name": self.name,
                            "shape": shape.as_str(),
                            "tail_bytes": self.withheld_tail.len(),
                        });
                        eprintln!("exp-gateway-native: {line}");
                    }
                    None => {
                        let mut streamed = self.raw_arguments.clone();
                        streamed.push_str(&self.withheld_tail);
                        require_json_object_text(&streamed)?;
                    }
                }
            }
            // Custom (freeform) tool input is opaque text by contract; only
            // function arguments must parse as one JSON object.
            require_json_object_text(&self.raw_arguments)?;
        }
        // Mirror the python ToolCall model constraints so both engines accept
        // exactly the same provider tool-call streams (a call the python
        // engine rejects must not become client-visible history here).
        if self.call_id.is_empty()
            || self.call_id.chars().count() > MAXIMUM_TOOL_CALL_ID_CHARACTERS
            || self.name.is_empty()
            || self.name.chars().count() > 256
            || self
                .namespace
                .as_ref()
                .is_some_and(|namespace| namespace.is_empty() || namespace.chars().count() > 256)
            || self.raw_arguments.chars().count() > 4_000_000
        {
            return Err("streamed tool call is incomplete".to_string());
        }
        Ok(CompletedToolCall {
            call_id: self.call_id.clone(),
            name: self.name.clone(),
            namespace: self.namespace.clone(),
            caller: self.caller.clone(),
            provider_item_id: self.provider_item_id.clone(),
            provider_status: self.provider_status,
            raw_arguments: self.raw_arguments.clone(),
            custom: self.custom,
        })
    }
}

mod usage;
pub use usage::*;

#[cfg(test)]
mod tests;
