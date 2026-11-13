"""Freeze reservation components for the admitted tool-search conversation."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.attempt_tokens import (
    INPUT_TOKEN_HEADROOM_PERCENT,
    MESSAGE_FRAMING_TOKENS,
    OPAQUE_BYTES_PER_TOKEN,
    TOOLS_PRESENT_TOKENS,
    _count_completion_messages,
    _count_completion_suffix,
    _count_tool,
    _PromptCounter,
)
from exp.runtime.gateway.contracts import GatewayRequest, GatewayToolDefinition
from exp.runtime.gateway.native_search_wires import tool_registration
from exp.runtime.gateway.replay_identity import provider_replay_authority


def prompt_registration(
    request: GatewayRequest, tools: Sequence[GatewayToolDefinition]
) -> JsonObject:
    """Export the exact shared counter program without counting any prompt twice."""
    messages = _PromptCounter()
    _count_completion_messages(request, messages)
    suffix = _PromptCounter()
    _count_completion_suffix(request, suffix)
    definitions: JsonObject = {}
    for tool in tools:
        counter = _PromptCounter()
        _count_tool(tool, counter)
        definitions[str(tool_registration(tool)["key"])] = counter.configuration()
    return {
        "messages": messages.configuration(),
        "suffix": suffix.configuration(),
        "tools": definitions,
        "replay": provider_replay_authority(request),
        "message_count": len(request.messages),
        "message_framing": MESSAGE_FRAMING_TOKENS,
        "tools_present": TOOLS_PRESENT_TOKENS,
        "opaque_bytes_per_token": OPAQUE_BYTES_PER_TOKEN,
        "headroom_percent": INPUT_TOKEN_HEADROOM_PERCENT,
        "reset_letters": reset_letters(),
    }


@cache
def reset_letters() -> str:
    """Freeze this runtime's exact isalpha boundary for bounded BPE counting."""
    ranges: list[str] = []
    start: int | None = None
    for code in range(0x110001):
        letter = code < 0x110000 and chr(code).isalpha()
        if letter and start is None:
            start = code
        elif not letter and start is not None:
            ranges.append(f"\\x{{{start:x}}}-\\x{{{code - 1:x}}}")
            start = None
    return "\\A[" + "".join(ranges) + "]\\z"
