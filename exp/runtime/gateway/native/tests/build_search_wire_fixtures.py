"""Freeze search continuations through all five existing provider encoders."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.attempt_tokens import counted_input_tokens, worst_case_input_tokens
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.gateway.guardrails.classification_window import (
    ClassificationWindow,
    classification_window,
)
from exp.runtime.gateway.guardrails.contracts import ClassifierCoverageError
from exp.runtime.gateway.guardrails.native_owner import text_subjects
from exp.runtime.gateway.guardrails.subjects import observation_subject_bytes
from exp.runtime.gateway.native_recovery import recovery_prefix_digest
from exp.runtime.gateway.native_search_counter import prompt_registration
from exp.runtime.gateway.native_search_subject import subject_registration
from exp.runtime.gateway.native_search_wires import Builder, compile_wire, tool_registration
from exp.runtime.gateway.tool_contracts import GatewayProviderNativeTool
from exp.runtime.gateway.tool_search.contracts import GatewayToolSearch
from exp.runtime.gateway.tool_search.plan import ToolSearchState
from exp.runtime.gateway.tool_search.round import parse_calls, perform_round
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.dialect_dispatch import dialect_stream_payload


def builder(dialect: str) -> Builder:
    """Use the actual encoder; no template implementation participates in the reference."""
    profile = GatewayWireProfile(
        dialect=dialect,
        url="https://provider.test/api",
        model_id="claude-sonnet-4-6" if dialect == "anthropic_messages" else "model",
        forwards_cache_control=True,
    )

    def build(request: GatewayRequest) -> JsonObject:
        payload = dialect_stream_payload(profile, request)
        signed = dialect == "bedrock_converse_stream"
        return {
            "dialect": dialect,
            "url": profile.url,
            "headers": {},
            "upstream_payload": None if signed else payload,
            "upstream_body": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if signed
            else None,
        }

    return build


def cases() -> list[JsonObject]:
    """Cover parallel calls, exhaustion, hidden tool fields, role merging, and cache markers."""
    result: list[JsonObject] = []
    for dialect in (
        "openai_compatible",
        "openai_responses",
        "anthropic_messages",
        "gemini_generate_content",
        "bedrock_converse_stream",
    ):
        build = builder(dialect)
        for i in range(18):
            search = GatewayToolSearch(
                declared_as="openrouter_tool",
                tool_type="openrouter:tool_search",
                max_rounds=3,
                default_limit=1,
            )
            tools = [
                GatewayToolDefinition(
                    name=name,
                    description=f"Weather {name} 世界",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string", "description": "世界"}},
                    },
                    cache_control={"type": "ephemeral"} if i % 2 else None,
                    input_examples=({"city": "Bern"},) if i % 5 == 0 else None,
                )
                for name in ("currentWeather", "weatherForecast", "flightWeather")
            ]
            loaded = (
                []
                if i % 3 == 0
                else [GatewayToolDefinition(name="sendEmail", parameters={"type": "object"})]
            )
            state = ToolSearchState(search, "tool_search", loaded, tools.copy())
            messages = [
                GatewayMessage(role="system", content="Be precise."),
                GatewayMessage(role="user", content="Weather?"),
            ]
            if i % 3 == 1:
                messages.append(GatewayMessage(role="assistant", content="I will look it up."))
            surface = (
                GatewayApiSurface.RESPONSES
                if dialect == "openai_responses" and i % 2
                else GatewayApiSurface.MESSAGES
            )
            request = GatewayRequest(
                surface=surface,
                messages=tuple(messages),
                tools=state.dispatch_tools(),
                maximum_output_tokens=512,
                stream=True,
                parallel_tool_calls=bool(i % 2),
                provider_cache_control={"type": "ephemeral"} if i % 4 == 0 else None,
                provider_native_tools=(
                    GatewayProviderNativeTool(index=0, tool={"type": "web_search"}),
                )
                if dialect == "openai_responses" and i % 2
                else (),
                provider_server_tools=({"type": "web_search_20250305", "name": "web_search"},)
                if dialect == "anthropic_messages" and i % 2
                else (),
            )
            config: JsonObject = {
                "mode": search.mode,
                "tool_name": state.tool_name,
                "declared_name": search.tool_type,
                "max_rounds": search.max_rounds,
                "default_limit": search.default_limit,
                "loaded": [tool_registration(t) for t in loaded],
                "deferred": [tool_registration(t) for t in tools],
                "gateway_tool": tool_registration(state.gateway_tool()),
            }
            program = compile_wire(
                request,
                state,
                build,
                profile=GatewayWireProfile(
                    dialect=dialect,
                    url="https://provider.test/api",
                    model_id="claude-sonnet-4-6" if dialect == "anthropic_messages" else "model",
                    forwards_cache_control=True,
                ),
            )
            counter = prompt_registration(
                request, (*state.loaded, *state.deferred, state.gateway_tool())
            )
            subject = subject_registration(
                request, (*state.loaded, *state.deferred, state.gateway_tool())
            )
            window = ClassificationWindow(
                system_chars=7, latest_user_chars=6, prior_messages=i % 3, prior_chars=5
            )
            steps: list[JsonObject] = []
            for j in range(3):
                calls: list[JsonObject] = [
                    {
                        "call_id": f"call-{j}-{k}",
                        "name": "tool_search",
                        "arguments": json.dumps({"query": "weather", "limit": 1}),
                    }
                    for k in range(1 + i % 2)
                ]
                outcome = perform_round(request, state, parse_calls(calls))
                request = outcome.request
                if i >= 15:
                    rewritten = (
                        *request.messages[:-1],
                        request.messages[-1].model_copy(
                            update={"content": f"Inspected result {j}"}
                        ),
                        GatewayMessage(role="user", content=f"Policy annotation {j}"),
                        GatewayMessage(
                            role="developer", content=f"Follow the inspected result {j}"
                        ),
                    )
                    request = request.model_copy(update={"messages": rewritten})
                try:
                    subjects = text_subjects(request)
                except ClassifierCoverageError:
                    subjects = None
                steps.append(
                    {
                        "calls": calls,
                        "replacement": json.loads(observation_subject_bytes(request))["messages"]
                        if i >= 15
                        else None,
                        "expected": build(request),
                        "counted_input": counted_input_tokens(request),
                        "reserved_input": worst_case_input_tokens(request),
                        "subject": json.loads(observation_subject_bytes(request)),
                        "visible": request.model_dump(mode="json", by_alias=True),
                        "recovery_prefix": recovery_prefix_digest(request),
                        "subject_size": len(observation_subject_bytes(request)),
                        "text_subjects": subjects,
                        "window_text": classification_window(request, window),
                    }
                )
            result.append(
                {
                    "configuration": config,
                    "plan": program,
                    "counter": counter,
                    "steps": steps,
                    "subject": subject,
                    "window": window.model_dump(mode="json"),
                    "rewrites": i >= 15,
                }
            )
    return result


if __name__ == "__main__":
    with patch("exp.runtime.gateway.native_search_wires.uuid4", return_value=UUID(int=1071)):
        Path(__file__).with_name("generated").joinpath("search_wires.jsonl").write_text(
            "".join(
                json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n"
                for case in cases()
            )
        )
