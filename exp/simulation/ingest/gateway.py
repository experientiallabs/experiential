"""Normalize identity-scoped native gateway exchanges into build evidence."""

from __future__ import annotations

from pathlib import Path

from pydantic import JsonValue

from exp.common.claas import ClaasScope, Experience
from exp.common.core.artifacts import JsonObject, SourceIdentity, sha256_json
from exp.runtime.claas.store import ExperienceStore
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.local_capture import GATEWAY_CAPTURE_APPLICATION
from exp.runtime.openai_protocol.requests import decode_responses
from exp.simulation.ingest.chat_json import CHAT_JSON_SOURCE
from exp.simulation.ingest.otlp import TraceNormalizationIssue, TraceNormalizationResult


def load_gateway_capture(
    path: Path, *, identity_id: str, source_id: str | None = None, limit: int = 1000
) -> TraceNormalizationResult:
    """Read a bounded identity-only corpus without any hosted service or provider call.

    A gateway exchange is not a complete agent episode or proof of task success.
    Explicit response links remain provenance; unrelated chats are never joined by
    matching their text. Missing context is an exclusion, not fabricated evidence.
    """
    scope = ClaasScope(user_id=identity_id, application_id=GATEWAY_CAPTURE_APPLICATION)
    rows = ExperienceStore(path, scope).read_after(limit=limit)
    documents: list[JsonValue] = []
    issues: list[TraceNormalizationIssue] = []
    for row in rows:
        try:
            documents.append(_conversation(row.experience))
        except ValueError:
            issues.append(
                TraceNormalizationIssue(
                    row.experience.experience_id,
                    "Capture lacks supported effective context or a complete response; "
                    "collect fresh traffic with capture enabled.",
                )
            )
    return CHAT_JSON_SOURCE.normalize(
        documents,
        source=SourceIdentity(
            kind="production",
            source_id=source_id or f"gateway:{identity_id}",
            sha256=sha256_json(documents),
        ),
        initial_issues=issues,
    )


def _conversation(experience: Experience) -> JsonObject:
    """Retain the full source exchange and expose observed messages and tool schemas."""
    context = experience.request.get("exp_context")
    if not isinstance(context, dict) or context.get("schema_version") != 1:
        raise ValueError("effective capture context is required")
    raw_request = context.get("request")
    if not isinstance(raw_request, dict):
        raise ValueError("effective request is required")
    request = GatewayRequest.model_validate(raw_request)
    messages: list[JsonObject] = [
        message.model_dump(mode="json", exclude_none=True) for message in request.messages
    ]
    messages.extend(_output_messages(experience))
    tool_names: dict[str, str] = {}
    for message in messages:
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function", call)
                call_id = call.get("id", call.get("call_id"))
                if isinstance(function, dict) and isinstance(call_id, str):
                    name = function.get("name")
                    if isinstance(name, str):
                        tool_names[call_id] = name
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in tool_names:
                raise ValueError("tool result has no observed matching call")
            message["name"] = tool_names[call_id]
    tools: list[JsonValue] = [
        {
            "name": tool.name,
            "description": tool.description or "No description supplied by the caller.",
            "input_schema": tool.parameters,
        }
        for tool in request.tools
    ]
    return {
        "id": experience.experience_id,
        "messages": list(messages),
        "exp.request.tools": tools,
        "exp.request.context": {
            "gateway_request": context,
            "gateway_response": experience.response,
            "identity_id": experience.scope.user_id,
            "captured_at": experience.captured_at.isoformat(),
            "response_id": experience.response_id,
            "parent_response_id": experience.parent_response_id,
            "deployment_id": experience.provenance.deployment_id,
            "model_id": experience.provenance.model_id,
        },
    }


def _output_messages(experience: Experience) -> list[JsonObject]:
    """Normalize public completed output, without asserting agent task success."""
    if experience.protocol == "responses":
        output = experience.response.get("output")
        if not isinstance(output, list) or not output:
            raise ValueError("response has no observed output")
        decoded = decode_responses({"model": "captured", "input": output})
        return [
            message.model_dump(mode="json", exclude_none=True)
            for message in decoded.request.messages
        ]
    choices = experience.response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("capture needs exactly one observed choice")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ValueError("capture has no complete assistant message")
    return [choice["message"]]
