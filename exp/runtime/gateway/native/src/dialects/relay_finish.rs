//! Finishing tool calls a provider cut short: a call whose arguments end
//! mid-token is the provider's truncation misreported (whatever its finish
//! reason, item status, or stop reason says), not corruption, so the call is
//! DROPPED and the turn settles `Incomplete`; a syntax error inside the
//! arguments keeps the strict malformed contract. Every dialect completes
//! its tool calls through these helpers so the verdict is one rule. Child of
//! `dialects` so the parent stays under the hand-authored line budget.

use std::collections::BTreeMap;

use serde_json::Value;

use super::{bounded_wire_token, complete_streamed_tool};
use crate::errors::Failure;
use crate::events::{Event, ToolAccumulator};

/// Whether accumulated tool arguments end mid-token: valid JSON so far that
/// simply stops (serde's end-of-input error), as opposed to a syntax error
/// inside the text. A model never emits a well-formed answer that ends
/// mid-string, so this shape is the provider cutting the stream whatever its
/// finish reason says.
pub(in crate::dialects) fn arguments_end_mid_fragment(raw: &str) -> bool {
    // Only an OBJECT prefix can be a cut tool call: a fragment that opens as an
    // array or scalar could never have become the required arguments object,
    // so its early end is corruption and keeps the strict contract.
    raw.trim_start().starts_with('{')
        && matches!(serde_json::from_str::<Value>(raw), Err(error) if error.is_eof())
}

/// Drop one open call whose arguments end mid-fragment, returning whether it
/// was dropped. `declared` names what the provider claimed about the ending
/// (`tool_calls`, `completed`, `block_stop`, `stream_end`) so the operator
/// line keeps the misreport visible. Argument bytes are never logged.
pub(in crate::dialects) fn drop_cut_call(tool: &mut ToolAccumulator, declared: &str) -> bool {
    if tool.completed
        || tool.custom
        || tool.raw_arguments.is_empty()
        || !arguments_end_mid_fragment(&tool.raw_arguments)
    {
        return false;
    }
    let line = serde_json::json!({
        "event": "tool_arguments_cut_mid_fragment",
        "name": bounded_wire_token(&tool.name),
        "bytes": tool.raw_arguments.len(),
        "declared": declared,
    });
    eprintln!("exp-gateway-native: {line}");
    tool.completed = true;
    true
}

/// Complete one streamed call, or drop it when the provider cut it
/// mid-fragment. Returns whether the call was dropped (the terminal must then
/// settle `Incomplete`); a syntax error inside the arguments propagates as the
/// malformed failure it always was.
pub(in crate::dialects) fn complete_streamed_tool_or_drop_cut(
    index: u32,
    tool: &mut ToolAccumulator,
    events: &mut Vec<Event>,
    declared: &str,
) -> Result<bool, Failure> {
    if drop_cut_call(tool, declared) {
        return Ok(true);
    }
    complete_streamed_tool(index, tool, events)?;
    Ok(false)
}

/// Drop a tool call the provider never named and never argued: an empty
/// placeholder entry (`{"id": "", "function": {"name": "", "arguments": ""}}`,
/// OpenRouter's GLM and Hunyuan relays, live 2026-09-12..15) is not a call, so
/// it is dropped instead of failing the stream. A NAMELESS call that did carry
/// arguments is left for the strict completion contract (a name cannot be
/// invented). Returns whether the entry was dropped.
pub(in crate::dialects) fn drop_phantom_tool(tool: &mut ToolAccumulator) -> bool {
    if tool.completed || tool.started || !tool.raw_arguments.is_empty() {
        return false;
    }
    let line = serde_json::json!({
        "event": "tool_call_phantom_dropped",
        "call_id_synthesized": tool.id_synthesized,
    });
    eprintln!("exp-gateway-native: {line}");
    tool.completed = true;
    true
}

/// Finish open tools on a stream that declared a normal finish (`stop` /
/// `tool_calls`, a `completed` Responses terminal, a clean stream end).
/// Aggregator relays (OpenRouter, Tencent TokenHub, the house vLLM lanes) have
/// been observed closing a stream with `finish_reason` `stop`/`tool_calls`
/// while a call's arguments end mid-fragment (ledger 2026-09-07: 57 attempts
/// in 12h, fragments from 1 KB to 87 KB, every one an end-of-input parse
/// error), and OpenAI's own Responses wire marks such a function_call item
/// `completed` (gpt-5.6-luna, 2026-09-10..14; exp#896). That is the
/// provider's truncation misreported, not corruption in a served answer: the
/// cut call is DROPPED and the turn settles `Incomplete` (the caller's remedy
/// is a larger budget), while a genuine syntax error inside the arguments
/// keeps the strict malformed contract. Returns whether any call was dropped.
pub(in crate::dialects) fn finish_open_tools_relay(
    tools: &mut BTreeMap<u32, ToolAccumulator>,
    declared: &str,
) -> Result<(Vec<Event>, bool), Failure> {
    let mut events = Vec::new();
    let mut truncated = false;
    for (index, tool) in tools.iter_mut() {
        if tool.completed {
            continue;
        }
        if complete_streamed_tool_or_drop_cut(*index, tool, &mut events, declared)? {
            truncated = true;
        }
    }
    Ok((events, truncated))
}
