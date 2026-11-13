"""Freeze provider request controls and tool definitions while admission owns Python."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast
from uuid import uuid4

from exp.common.core.artifacts import JsonObject, JsonValue, sha256_json
from exp.common.models.model import ToolCall
from exp.runtime.gateway.contracts import GatewayMessage, GatewayRequest, GatewayToolDefinition
from exp.runtime.gateway.json_object import JSON_OBJECT_SYSTEM_INSTRUCTION
from exp.runtime.gateway.routing import GatewayRoutingError
from exp.runtime.gateway.tool_search.plan import ToolSearchState
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.history_tool_arguments import opaque_tool_arguments_wire
from exp.runtime.models.providers.image_detail import _downgrades_original

Builder = Callable[[GatewayRequest], JsonObject]


def tool_registration(tool: GatewayToolDefinition) -> JsonObject:
    """Freeze a tool's definition and provider fields with deferred loading cleared."""
    value = tool.model_dump(mode="json")
    value.update(
        cache_control=tool.cache_control,
        eager_input_streaming=tool.eager_input_streaming,
        allowed_callers=None if tool.allowed_callers is None else list(tool.allowed_callers),
        input_examples=None if tool.input_examples is None else list(tool.input_examples),
    )
    return {**value, "key": sha256_json(value)}


def encoder_options(profile: GatewayWireProfile, *, strip_reasoning: bool = False) -> JsonObject:
    """Freeze only the non-secret controls needed to rebuild an inspected conversation."""
    return {
        "model_id": profile.model_id,
        "reasoning_route": profile.fireworks_reasoning_route_sha256
        or profile.hunyuan_reasoning_route_sha256,
        "fireworks": profile.fireworks_reasoning_route_sha256 is not None,
        "exposed": profile.reasoning_output_exposed,
        "deepseek": profile.deepseek_reasoning_history,
        "leading_only": profile.system_messages_leading_only,
        "cache_control": profile.forwards_cache_control,
        "cache_options": profile.forwards_prompt_cache_options,
        "strip_reasoning": strip_reasoning,
        "json_instruction": JSON_OBJECT_SYSTEM_INSTRUCTION,
    }


def compile_wire(
    request: GatewayRequest,
    state: ToolSearchState,
    build: Builder,
    *,
    profile: GatewayWireProfile,
    strip_reasoning: bool = False,
    requirements: JsonObject | None = None,
) -> JsonObject:
    """Freeze one rung's controls and definitions using its actual encoder and policy."""
    marker = uuid4().hex
    arguments: JsonObject = {"native_search_argument": marker}
    raw = json.dumps(arguments, separators=(",", ":"))
    assistant = GatewayMessage(
        role="assistant",
        tool_calls=(
            ToolCall(
                call_id=f"search_{marker}",
                name=state.tool_name,
                arguments=arguments,
                raw_arguments=raw,
            ),
        ),
    )
    result = GatewayMessage(
        role="tool", tool_call_id=f"search_{marker}", content=f"result_{marker}"
    )
    pilot = request.model_copy(
        update={
            "messages": (*request.messages, assistant, result),
            "tool_choice": None,
        }
    )
    wire = build(pilot)
    signed = wire["upstream_body"] is not None
    payload = _payload(wire)
    dialect = str(wire["dialect"])
    shaping: JsonObject = {
        "opaque_arguments": opaque_tool_arguments_wire(profile),
        "downgrade_original": _downgrades_original(profile),
        "vertex_anthropic": profile.anthropic_vertex_version is not None,
        "requirements": requirements,
        "thinking": None,
    }
    if dialect == "openai_compatible":
        closed = _payload(
            build(
                pilot.model_copy(
                    update={"messages": (GatewayMessage(role="user", content="probe"),)}
                )
            )
        )
        if any(closed.get(key) != payload.get(key) for key in ("thinking", "output_config")):
            shaping["thinking"] = {
                "closed": {key: closed.get(key) for key in ("thinking", "output_config")},
                "open": {key: payload.get(key) for key in ("thinking", "output_config")},
            }
    _, static = _tools(payload, dialect)
    definitions: JsonObject = {}
    for tool in (*state.loaded, *state.deferred, state.gateway_tool()):
        registration = tool_registration(tool)
        key = str(registration["key"])
        if key in definitions:
            continue
        # Deferred capabilities are checked only when that tool is loaded. A
        # declined definition stays a failed template until a round requests it.
        candidate = pilot.model_copy(
            update={
                "messages": (assistant, result),
                "tools": (tool.model_copy(update={"defer_loading": None}),),
                "provider_native_tools": (),
                "provider_server_tools": (),
            }
        )
        try:
            encoded, _ = _tools(_payload(build(candidate)), dialect)
            definitions[key] = list[JsonValue](encoded[tool.name])
        except (ProviderCapabilityError, GatewayRoutingError, ValueError):
            definitions[key] = None
    for key in (
        "messages",
        "input",
        "contents",
        "system",
        "systemInstruction",
        "instructions",
        "reasoning_history",
    ):
        payload.pop(key, None)
    wire["upstream_payload"] = None
    wire["upstream_body"] = None
    return {
        "wire": wire,
        "payload": payload,
        "signed": signed,
        "dialect": dialect,
        "definitions": definitions,
        "static_tools": list[JsonValue](static),
        "encoder": {
            **encoder_options(profile, strip_reasoning=strip_reasoning),
            "shaping": shaping,
        },
    }


def _payload(wire: JsonObject) -> JsonObject:
    body = wire.get("upstream_body")
    value = json.loads(body) if isinstance(body, str) else wire.get("upstream_payload")
    if not isinstance(value, dict):
        message = "A search rung must carry an object payload"
        raise ValueError(message)
    return cast("JsonObject", value)


def _objects(value: object) -> list[JsonObject]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        message = "A search wire template has an invalid object array"
        raise ValueError(message)
    return cast("list[JsonObject]", value)


def _tools(
    payload: JsonObject, dialect: str
) -> tuple[dict[str, list[JsonObject]], list[JsonObject]]:
    if dialect == "bedrock_converse_stream":
        config = cast("JsonObject", payload["toolConfig"])
        entries = _objects(config.pop("tools"))
    elif dialect == "gemini_generate_content":
        entries = _objects(_objects(payload.pop("tools"))[0]["functionDeclarations"])
    else:
        entries = _objects(payload.pop("tools"))
    definitions: dict[str, list[JsonObject]] = {}
    static: list[JsonObject] = []
    last: str | None = None
    for index, entry in enumerate(entries):
        if dialect == "bedrock_converse_stream" and "cachePoint" in entry and last is not None:
            definitions[last].append(entry)
            continue
        definition = (
            entry.get("function")
            if dialect == "openai_compatible"
            else entry.get("toolSpec")
            if dialect == "bedrock_converse_stream"
            else entry
        )
        name = definition.get("name") if isinstance(definition, dict) else None
        function = (
            entry.get("type") == "function"
            if dialect in {"openai_compatible", "openai_responses"}
            else "input_schema" in entry and "type" not in entry
            if dialect == "anthropic_messages"
            else True
        )
        if isinstance(name, str) and function:
            definitions[name] = [entry]
            last = name
        else:
            static.append({"index": index, "tool": entry})
            last = None
    return definitions, static
