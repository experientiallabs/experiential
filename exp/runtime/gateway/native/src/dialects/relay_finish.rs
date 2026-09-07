//! Finishing open tool calls on an OpenAI-compatible RELAY stream that
//! declared a normal finish (`stop` / `tool_calls`): a call whose arguments
//! end mid-token is the provider's cut misreported, not corruption, and the
//! turn settles `Incomplete`. Child of `dialects` so the parent stays under
//! the hand-authored line budget.

use std::collections::BTreeMap;

use serde_json::Value;

use super::complete_streamed_tool;
use crate::errors::Failure;
use crate::events::{Event, ToolAccumulator};

/// Whether accumulated tool arguments end mid-token: valid JSON so far that
/// simply stops (serde's end-of-input error), as opposed to a syntax error
/// inside the text. A model never emits a well-formed answer that ends
/// mid-string, so this shape is the provider cutting the stream whatever its
/// finish reason says.
fn arguments_end_mid_fragment(raw: &str) -> bool {
    // Only an OBJECT prefix can be a cut tool call: a fragment that opens as an
    // array or scalar could never have become the required arguments object,
    // so its early end is corruption and keeps the strict contract.
    raw.trim_start().starts_with('{')
        && matches!(serde_json::from_str::<Value>(raw), Err(error) if error.is_eof())
}

/// Finish open tools on an OpenAI-compatible RELAY stream that declared a
/// normal finish. Aggregator relays (OpenRouter, Tencent TokenHub, the house
/// vLLM lanes) have been observed closing a stream with `finish_reason`
/// `stop`/`tool_calls` while a call's arguments end mid-fragment (ledger
/// 2026-09-07: 57 attempts in 12h, fragments from 1 KB to 87 KB, every one
/// an end-of-input parse error). That is the provider's truncation misreported,
/// not corruption in a served answer: the cut call is DROPPED and the turn
/// settles `Incomplete` (the caller's remedy is a larger budget), while a
/// genuine syntax error inside the arguments keeps the strict malformed
/// contract. Returns whether any call was dropped this way. OpenAI's own
/// Responses wire is NOT routed here: its item status is authoritative, so a
/// completed item with cut arguments stays fail-closed.
pub(in crate::dialects) fn finish_open_tools_relay(
    tools: &mut BTreeMap<u32, ToolAccumulator>,
) -> Result<(Vec<Event>, bool), Failure> {
    let mut events = Vec::new();
    let mut truncated = false;
    for (index, tool) in tools.iter_mut() {
        if tool.completed {
            continue;
        }
        if !tool.custom
            && !tool.raw_arguments.is_empty()
            && arguments_end_mid_fragment(&tool.raw_arguments)
        {
            let line = serde_json::json!({
                "event": "tool_arguments_cut_mid_fragment",
                "name": tool.name,
                "bytes": tool.raw_arguments.len(),
            });
            eprintln!("exp-gateway-native: {line}");
            tool.completed = true;
            truncated = true;
            continue;
        }
        complete_streamed_tool(*index, tool, &mut events)?;
    }
    Ok((events, truncated))
}
