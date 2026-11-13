"""Freeze complete input subjects and adapter authority for native tool-search rounds."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from exp.common.core.artifacts import JsonObject
from exp.common.models.model import ToolCall
from exp.runtime.gateway.contracts import GatewayMessage, GatewayRequest, GatewayToolDefinition
from exp.runtime.gateway.guardrails.contracts import GuardrailCheckStage
from exp.runtime.gateway.guardrails.subjects import _subject_value
from exp.runtime.gateway.native_search_wires import tool_registration
from exp.runtime.gateway.replay_identity import provider_replay_authority

if TYPE_CHECKING:
    from exp.runtime.gateway.guardrails.session import GuardrailSession


def subject_registration(
    request: GatewayRequest, tools: Sequence[GatewayToolDefinition]
) -> JsonObject:
    """Keep full observation fields and their separate hosted-classifier projection."""
    defaults = GatewayRequest(
        surface=request.surface,
        messages=(
            GatewayMessage(role="assistant", tool_calls=(ToolCall(call_id="id", name="fn"),)),
        ),
    )
    message = cast("JsonObject", _subject_value(defaults.messages[0]))
    call = _subject_value(defaults.messages[0].tool_calls[0])
    message["tool_calls"] = []
    return {
        "request": _subject_value(request),
        "visible": request.model_dump(mode="json", by_alias=True),
        "message_default": message,
        "call_default": call,
        "message_keys": list(defaults.messages[0].model_dump(mode="json", by_alias=True)),
        "call_keys": list(
            defaults.messages[0].tool_calls[0].model_dump(mode="json", by_alias=True)
        ),
        "tool_keys": list(
            GatewayToolDefinition(name="fn", parameters={}).model_dump(mode="json", by_alias=True)
        ),
        "definitions": {
            str(tool_registration(tool)["key"]): _subject_value(
                tool.model_copy(update={"defer_loading": None})
            )
            for tool in tools
        },
        "recovery_identity": {
            "envelope": provider_replay_authority(
                request.model_copy(update={"messages": (), "tools": ()})
            ),
            "gateway": None if request.gateway is None else request.gateway.replay_identity(),
        },
    }


def input_templates(
    session: GuardrailSession | None, *, observing: bool = False
) -> list[JsonObject]:
    """Retain each execution mode independently from the first input gate."""
    if session is None:
        return []
    return [
        session.engine.native_input_template(policy=policy, check=check)
        for policy in session.policies
        if (policy.mode == "observe") is observing
        for check in policy.checks
        if check.stage is GuardrailCheckStage.INPUT
    ]
