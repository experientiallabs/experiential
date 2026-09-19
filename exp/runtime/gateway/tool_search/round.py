"""One gateway tool-search round: search, extend the conversation, rebuild the rung.

The data plane withheld the model's ``tool_search`` call(s) and asks the
control plane for the next dispatch of the same rung. This module answers:
it runs the search over the deferred corpus, appends the assistant call and
a tool result naming the matched tools, moves those tools into the loaded
set, and returns the rebuilt wire entry for the depth plus the facts the
response renders (query, pattern, matched names).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from exp.common.core.artifacts import JsonObject
from exp.common.models.model import ToolCall
from exp.runtime.gateway.contracts import GatewayMessage, GatewayRequest, GatewayToolDefinition
from exp.runtime.gateway.tool_search.contracts import MAXIMUM_TOOL_SEARCH_QUERY_CHARACTERS
from exp.runtime.gateway.tool_search.plan import ToolSearchState
from exp.runtime.gateway.tool_search.search import (
    ToolSearchPatternError,
    bm25_search,
    clamp_limit,
    regex_search,
)


@dataclass(frozen=True)
class WithheldSearchCall:
    """One ``tool_search`` call the data plane withheld from the caller."""

    call_id: str
    name: str
    raw_arguments: str


@dataclass(frozen=True)
class RoundOutcome:
    """The conversation extension and rendering facts of one round."""

    request: GatewayRequest
    rounds: list[JsonObject]
    exhausted: bool


def parse_calls(raw: object) -> list[WithheldSearchCall]:
    """Decode the data plane's ``calls`` array.

    Args:
        raw: The JSON array from the ``tool_search_round`` argument.

    Returns:
        The withheld calls in arrival order (malformed entries skipped).
    """
    calls: list[WithheldSearchCall] = []
    if not isinstance(raw, list):
        return calls
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        call_id = entry.get("call_id")
        name = entry.get("name")
        arguments = entry.get("arguments")
        if isinstance(call_id, str) and call_id and isinstance(name, str):
            calls.append(
                WithheldSearchCall(
                    call_id=call_id,
                    name=name,
                    raw_arguments=arguments if isinstance(arguments, str) else "{}",
                )
            )
    return calls


def _arguments(call: WithheldSearchCall) -> JsonObject:
    try:
        decoded = json.loads(call.raw_arguments or "{}")
    except ValueError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def search_once(
    state: ToolSearchState, arguments: JsonObject
) -> tuple[list[GatewayToolDefinition], str | None, str | None, str | None]:
    """Run one search over the deferred corpus.

    Args:
        state: Per-request search state.
        arguments: The model's decoded call arguments.

    Returns:
        ``(matched, query, pattern, error)``; ``error`` names an unusable request.
    """
    limit = clamp_limit(arguments.get("limit"), default=state.search.default_limit)
    query = arguments.get("query")
    pattern = arguments.get("pattern")
    query = query[:MAXIMUM_TOOL_SEARCH_QUERY_CHARACTERS] if isinstance(query, str) else None
    pattern = pattern if isinstance(pattern, str) else None
    if pattern and state.search.mode in {"regex", "any"}:
        try:
            return regex_search(pattern, state.deferred, limit=limit), None, pattern, None
        except ToolSearchPatternError as exc:
            return [], None, pattern, str(exc)
    if query and state.search.mode in {"bm25", "any"}:
        return bm25_search(query, state.deferred, limit=limit), query, None, None
    return (
        [],
        query,
        pattern,
        "provide a natural-language `query` or a regular-expression `pattern`",
    )


def _result_content(matched: Sequence[GatewayToolDefinition], error: str | None) -> str:
    if error is not None:
        return json.dumps({"error": error, "matched": []}, separators=(",", ":"))
    return json.dumps(
        {
            "matched": [
                {"name": tool.name, "description": tool.description or ""} for tool in matched
            ],
            "loaded": bool(matched),
            "note": (
                "The matched tools are now loaded and can be called directly."
                if matched
                else "No deferred tool matched; answer with the tools you have or search again."
            ),
        },
        separators=(",", ":"),
    )


def perform_round(
    request: GatewayRequest,
    state: ToolSearchState,
    calls: Sequence[WithheldSearchCall],
) -> RoundOutcome:
    """Extend the conversation with the searches and their results.

    Args:
        request: The provider request the previous dial was built from.
        state: Per-request search state (mutated: rounds, loaded/deferred).
        calls: The withheld search calls of the finished dial.

    Returns:
        The extended request (with the current dispatch tools) and the round facts.
    """
    tool_calls: list[ToolCall] = []
    results: list[GatewayMessage] = []
    rounds: list[JsonObject] = []
    for call in calls:
        arguments = _arguments(call)
        matched, query, pattern, error = search_once(state, arguments)
        for tool in matched:
            if tool in state.deferred:
                state.deferred.remove(tool)
                state.loaded.append(tool)
        tool_calls.append(
            ToolCall(
                call_id=call.call_id,
                name=state.tool_name,
                arguments=arguments,
                raw_arguments=call.raw_arguments or "{}",
            )
        )
        results.append(
            GatewayMessage(
                role="tool",
                tool_call_id=call.call_id,
                content=_result_content(matched, error),
            )
        )
        rounds.append(
            {
                "call_id": call.call_id,
                "query": query,
                "pattern": pattern,
                "declared_name": state.search.tool_name or state.search.tool_type,
                "matched": [tool.name for tool in matched],
                "matched_tools": [
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                    for tool in matched
                ],
            }
        )
    state.rounds_done += 1
    state.rounds.extend(rounds)
    messages = (
        *request.messages,
        GatewayMessage(role="assistant", tool_calls=tuple(tool_calls)),
        *results,
    )
    extended = request.model_copy(
        update={"messages": messages, "tools": state.dispatch_tools(), "tool_choice": None}
    )
    return RoundOutcome(request=extended, rounds=rounds, exhausted=state.exhausted)


RebuildDispatch = Callable[[GatewayRequest, int], JsonObject]
"""Rebuilds one depth's wire entry from an extended provider request."""
