"""Compare whole deferred-tool rounds against the existing policy and result encoder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.gateway.tool_search.contracts import GatewayToolSearch
from exp.runtime.gateway.tool_search.plan import ToolSearchState
from exp.runtime.gateway.tool_search.round import parse_calls, perform_round


def cases() -> list[JsonObject]:
    """Cover evolving corpora, grouped calls, declaration modes, and exact UTF-8 results."""
    result: list[JsonObject] = []
    modes: tuple[Literal["any", "bm25", "regex"], ...] = ("any", "bm25", "regex")
    for i in range(210):
        search = GatewayToolSearch(
            declared_as="openrouter_tool",
            tool_type="openrouter:tool_search",
            tool_name="search_declared" if i % 2 else None,
            mode=modes[i % 3],
            max_rounds=1 + i % 8,
            default_limit=1 + i % 10,
        )
        tools = [
            GatewayToolDefinition(name=name, description=description, parameters={"type": "object"})
            for name, description in (
                ("currentWeather", "Weather 世界 🌞"),
                ("forecast", "weather forecast"),
                ("flightWeather", None),
                ("get_flights", "flight schedule"),
                ("zebra", "animals"),
            )
        ]
        state = ToolSearchState(search, "tool_search", [], tools.copy())
        config: JsonObject = {
            "mode": search.mode,
            "tool_name": state.tool_name,
            "declared_name": search.tool_name or search.tool_type,
            "max_rounds": search.max_rounds,
            "default_limit": search.default_limit,
            "loaded": [],
            "deferred": [t.model_dump(mode="json") for t in tools],
            "gateway_tool": state.gateway_tool().model_dump(mode="json"),
        }
        request = GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(GatewayMessage(role="user", content="weather"),),
            tools=state.dispatch_tools(),
        )
        steps: list[JsonObject] = []
        arguments = [
            '{"query":"weather","limit":1}',
            '{"query":"flight"}',
            '{"pattern":"["}',
            '{"query":"weather","pattern":"^get","limit":2}',
            "{}",
            "not-json",
            "[1,2]",
            '{"query":"zebra","limit":999999999999999999999999999999}',
            '{"query":"weather","limit":-99999999999999999999999999999}',
            '{"pattern":"w.*","limit":false}',
            '{"pattern":"(?x)animal"}',
            "",
        ]
        for j in range(search.max_rounds):
            raw: list[JsonObject] = [
                {
                    "call_id": f"call-{j}-{k}",
                    "name": "tool_search",
                    "arguments": arguments[(i + j + k) % len(arguments)],
                }
                for k in range(1 + i % 3)
            ]
            if i % 11 == 0:
                raw += [{"name": "missing-id"}, {"call_id": "missing-name"}]
            parsed = parse_calls(raw)
            try:
                outcome = perform_round(request, state, parsed)
            except ValueError:
                steps.append({"calls": raw, "error": True})
                break
            new_messages = outcome.request.messages[len(request.messages) :]
            frame: JsonObject = {
                "calls": [
                    {
                        "call_id": c.call_id,
                        "name": c.name,
                        "arguments": c.arguments,
                        "raw_arguments": c.raw_arguments,
                    }
                    for c in new_messages[0].tool_calls
                ],
                "results": [m.content for m in new_messages[1:]],
            }
            steps.append(
                {
                    "calls": raw,
                    "rounds": outcome.rounds,
                    "frame": frame,
                    "exhausted": outcome.exhausted,
                    "tools": [t.model_dump(mode="json") for t in outcome.request.tools],
                }
            )
            request = outcome.request
        result.append({"configuration": config, "steps": steps})
    return result


if __name__ == "__main__":
    Path(__file__).with_name("generated").joinpath("search_rounds.jsonl").write_text(
        "".join(
            json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n" for case in cases()
        )
    )
