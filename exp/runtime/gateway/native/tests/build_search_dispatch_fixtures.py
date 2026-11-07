"""Freeze complete tool-search handoffs through the existing per-rung dispatch builder."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.attempt_tokens import counted_input_tokens, worst_case_input_tokens
from exp.runtime.gateway.contracts import GatewayToolDefinition
from exp.runtime.gateway.native_execution import InflightRequest
from exp.runtime.gateway.native_rungs_test import (
    _deployment,
    _NoSigningClient,
    _profile,
    _request,
    _route,
)
from exp.runtime.gateway.native_search_registration import search_registration
from exp.runtime.gateway.native_tool_search_test import _Plane
from exp.runtime.gateway.tool_search.contracts import GatewayToolSearch
from exp.runtime.gateway.tool_search.plan import ToolSearchState


def cases() -> list[JsonObject]:
    """Exercise real capability shaping, rung headers, and wire metadata during rebuilds."""
    cases: list[JsonObject] = []
    for dialect, provider in [
        ("openai_compatible", "openai-compatible"),
        ("openai_responses", "openai"),
        ("anthropic_messages", "anthropic"),
        ("gemini_generate_content", "gemini"),
    ]:
        declaration = GatewayToolSearch(
            declared_as="openrouter_tool", tool_type="openrouter:tool_search", max_rounds=3
        )
        tools = [
            GatewayToolDefinition(
                name=name, description=f"weather {name}", parameters={"type": "object"}
            )
            for name in ("weather", "forecast", "history")
        ]
        search = ToolSearchState(declaration, "tool_search", [], tools)
        request = _request().model_copy(
            update={"tools": search.dispatch_tools(), "maximum_output_tokens": 512}
        )
        deployment = _deployment("lane", provider)
        assert deployment.capabilities is not None
        deployment = deployment.model_copy(
            update={
                "capabilities": deployment.capabilities.model_copy(update={"supports_tools": True}),
                "gateway": deployment.gateway.model_copy(
                    update={
                        "capabilities": deployment.gateway.capabilities.model_copy(
                            update={"supports_streaming_tool_arguments": True}
                        )
                    }
                ),
            }
        )
        route = _route((deployment,))
        entry = InflightRequest(
            authorization=route.snapshot.authorization,
            route=route,
            request=request,
            public_request=request,
            deadline_monotonic=time.monotonic() + 60,
            resolved_wires=((_profile(dialect), _NoSigningClient()),),
            tool_search=search,
            reserved_output_tokens_by_depth=(512,),
            signers=(None,),
            dispatch_bindings=(None,),
            reasoning_carrier_authorities=(None,),
        )
        config = search_registration(entry)
        assert config is not None
        plane = _Plane(entry)
        steps: list[JsonObject] = []
        for i in range(3):
            calls = [
                {
                    "call_id": f"search-{i}",
                    "name": "tool_search",
                    "arguments": json.dumps({"query": "weather", "limit": 1}),
                }
            ]
            response = json.loads(
                plane.tool_search_round(
                    json.dumps({"request_id": "req", "route_depth": 0, "calls": calls})
                )
            )
            steps.append(
                {
                    "calls": calls,
                    "expected": response["wire"],
                    "counted_input": counted_input_tokens(entry.request),
                    "reserved_input": worst_case_input_tokens(entry.request),
                }
            )
        cases.append({"config": config, "steps": steps})
    return cases


if __name__ == "__main__":
    with patch("exp.runtime.gateway.native_search_wires.uuid4", return_value=UUID(int=1071)):
        Path(__file__).with_name("generated").joinpath("search_dispatches.jsonl").write_text(
            "".join(
                json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n"
                for case in cases()
            )
        )
