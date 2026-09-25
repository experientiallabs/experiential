"""Normalize identity-scoped native gateway exchanges into build evidence."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject, SourceIdentity, sha256_json
from exp.common.traces.capture import CaptureMetrics
from exp.common.traces.ingest.chat_json import CHAT_JSON_SOURCE
from exp.common.traces.ingest.otlp import TraceNormalizationIssue, TraceNormalizationResult
from exp.common.traces.ingest.sources import TraceSourceError
from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.gateway.capture_context import restore_capture_context
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.ingest.metrics import apply_gateway_metrics
from exp.runtime.gateway.local_capture import GATEWAY_CAPTURE_APPLICATION
from exp.runtime.gateway.local_capture_contracts import CapturedExchange, LocalCaptureScope
from exp.runtime.gateway.local_capture_store import LocalCaptureStore
from exp.runtime.gateway.replay_identity import provider_replay_authority
from exp.runtime.openai_protocol.requests import decode_responses


def load_gateway_capture(
    path: Path, *, identity_id: str, source_id: str | None = None
) -> TraceNormalizationResult:
    """Read the complete retained identity snapshot without a hosted service or provider call.

    A gateway exchange is not a complete agent episode or proof of task success.
    Explicit response links remain provenance; unrelated chats are never joined by
    matching their text. Missing context is an exclusion, not fabricated evidence.
    """
    scope = LocalCaptureScope(user_id=identity_id, application_id=GATEWAY_CAPTURE_APPLICATION)
    try:
        rows = LocalCaptureStore(path, scope).read_snapshot()
    except (sqlite3.Error, ValueError) as exc:
        raise TraceSourceError(
            "Cannot read this identity's gateway capture. Collect fresh traffic "
            "with capture enabled and select its traffic.db file."
        ) from exc
    by_response_id = {row.experience.response_id: row.experience for row in rows}
    documents: list[JsonValue] = []
    issues: list[TraceNormalizationIssue] = []
    for row in rows:
        try:
            documents.append(_conversation(row.experience, by_response_id))
        except ValueError:
            issues.append(
                TraceNormalizationIssue(
                    row.experience.experience_id,
                    "Capture lacks supported effective context or a complete response; "
                    "collect fresh traffic with capture enabled.",
                )
            )
    normalized = CHAT_JSON_SOURCE.normalize(
        documents,
        source=SourceIdentity(
            kind="production",
            source_id=source_id or f"gateway:{identity_id}",
            sha256=sha256_json([row.experience.model_dump(mode="json") for row in rows]),
        ),
        initial_issues=issues,
    )
    return apply_gateway_metrics(normalized)


def _conversation(
    experience: CapturedExchange, by_response_id: Mapping[str, CapturedExchange]
) -> JsonObject:
    """Retain the full source exchange and expose observed messages and tool schemas."""
    output = experience.request.get("exp_capture_output")
    if isinstance(output, dict) and output.get("metrics") is not None:
        CaptureMetrics.model_validate(output["metrics"])
    context = experience.request.get("exp_context")
    if not isinstance(context, dict) or context.get("schema_version") != 1:
        raise ValueError("effective capture context is required")
    context = restore_capture_context(context)
    raw_request = context.get("request")
    if not isinstance(raw_request, dict):
        raise ValueError("effective request is required")
    request = GatewayRequest.model_validate(raw_request)
    messages = _request_messages(request, context.get("provider_context"))
    lineage, missing_parent = _restore_linked_reasoning(messages, experience, by_response_id)
    output_message_start = len(messages)
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
    document: JsonObject = {
        "id": experience.experience_id,
        "messages": list(messages),
        "exp.request.tools": tools,
        "exp.request.context": {
            "gateway_request": context,
            "gateway_response": experience.response,
            "capture_output": experience.request.get("exp_capture_output"),
            "output_message_start": output_message_start,
            "identity_id": experience.scope.user_id,
            "captured_at": experience.captured_at.isoformat(),
            "response_id": experience.response_id,
            "parent_response_id": experience.parent_response_id,
            "deployment_id": experience.provenance.deployment_id,
            "model_id": experience.provenance.model_id,
            "linked_response_ids": lineage,
            "missing_parent_response_id": missing_parent,
        },
    }
    if experience.episode_id:
        document["exp.conversation.id"] = experience.episode_id
    elif experience.protocol == "responses":
        document["exp.conversation.id"] = (
            f"response:{missing_parent or (lineage[-1] if lineage else experience.response_id)}"
        )
    return document


def _restore_linked_reasoning(
    messages: list[JsonObject],
    experience: CapturedExchange,
    by_response_id: Mapping[str, CapturedExchange],
) -> tuple[list[JsonValue], str | None]:
    """Recover observed reasoning only through explicit, identity-scoped response links.

    Expanded history fixes each parent's output position. A changed post-guardrail
    visible turn is never overwritten. Missing links remain explicit evidence gaps.
    """
    seen = {experience.response_id}
    lineage: list[JsonValue] = []
    parent_id = experience.parent_response_id
    while parent_id:
        if parent_id in seen:
            raise ValueError("cyclic response lineage")
        seen.add(parent_id)
        parent = by_response_id.get(parent_id)
        if parent is None:
            return lineage, parent_id
        if parent.scope != experience.scope or parent.protocol != "responses":
            raise ValueError("response lineage scope mismatch")
        lineage.append(parent_id)
        context = parent.request.get("exp_context")
        if not isinstance(context, dict):
            return lineage, parent_id
        request = GatewayRequest.model_validate(restore_capture_context(context).get("request"))
        outputs = _linked_output_turns(parent)
        for position, output in enumerate(outputs, start=len(request.messages)):
            if position >= len(messages):
                raise ValueError("response lineage exceeds expanded history")
            target = messages[position]
            if _visible_turn(target) == _visible_turn(output):
                reasoning = output.get("reasoning_content")
                if isinstance(reasoning, str):
                    target["reasoning_content"] = reasoning
        parent_id = parent.parent_response_id
    return lineage, None


def _linked_output_turns(experience: CapturedExchange) -> list[JsonObject]:
    """Match public output items to the gateway's retained assistant segments.

    A retained segment combines one text item with adjacent function calls.
    A second text item or a native item starts another turn. Only this comparison
    projection is grouped; the exact public response and raw arguments stay intact.
    """
    turns: list[JsonObject] = []
    current: JsonObject | None = None
    has_message = False
    for output in _output_messages(experience):
        if _empty_reasoning_item(output):
            continue
        if output.get("role") != "assistant" or output.get("provider_native_item") is not None:
            turns.append(output)
            current = None
            has_message = False
            continue
        is_message = isinstance(output.get("content"), str) or isinstance(
            output.get("provider_item_id"), str
        )
        if current is None or (is_message and has_message):
            current = {"role": "assistant", "content": None, "tool_calls": []}
            turns.append(current)
            has_message = False
        if is_message:
            current["content"] = output.get("content") or None
            has_message = True
        calls = output.get("tool_calls")
        if isinstance(calls, list):
            retained_calls = current["tool_calls"]
            assert isinstance(retained_calls, list)
            retained_calls.extend(calls)
        reasoning = output.get("reasoning_content")
        if isinstance(reasoning, str):
            current["reasoning_content"] = reasoning
    return turns


def _empty_reasoning_item(message: JsonObject) -> bool:
    """Empty public reasoning lifecycle items are not retained conversation turns."""
    item = message.get("provider_native_item")
    return (
        isinstance(item, dict)
        and item.get("type") == "reasoning"
        and item.get("summary") == []
        and not item.get("encrypted_content")
        and not message.get("content")
        and not message.get("tool_calls")
    )


def _visible_turn(message: JsonObject) -> JsonObject:
    """Compare linked output without letting serialization-only carriers change identity."""
    calls = message.get("tool_calls")
    return {
        "role": message.get("role"),
        "content": message.get("content"),
        "tool_calls": [
            {key: call.get(key) for key in ("call_id", "name", "arguments")}
            for call in calls
            if isinstance(call, dict)
        ]
        if isinstance(calls, list)
        else [],
    }


def _output_messages(experience: CapturedExchange) -> list[JsonObject]:
    """Normalize public completed output, without asserting agent task success."""
    if experience.protocol == "responses":
        output = experience.response.get("output")
        if not isinstance(output, list) or not output:
            raise ValueError("response has no observed output")
        decoded = decode_responses({"model": "captured", "input": output})
        messages = _request_messages(decoded.request, provider_replay_authority(decoded.request))
    elif experience.protocol == "messages":
        content = experience.response.get("content")
        if not isinstance(content, list) or not content:
            raise ValueError("message has no observed output")
        decoded = decode_messages(
            {
                "model": "captured",
                "max_tokens": 1,
                "messages": [{"role": "assistant", "content": content}],
            }
        )
        messages = _request_messages(decoded.request, provider_replay_authority(decoded.request))
    else:
        messages = _chat_output(experience)
    output = experience.request.get("exp_capture_output")
    if isinstance(output, dict):
        _restore_output_tools(messages, output)
        reasoning = output.get("provider_reasoning")
        source = output.get("provider_reasoning_source_json")
        if isinstance(source, str):
            reasoning = json.loads(source)
        if isinstance(reasoning, str):
            assistant = next(
                (message for message in reversed(messages) if message.get("role") == "assistant"),
                None,
            )
            if assistant is None:
                raise ValueError("reasoning has no assistant output")
            assistant["reasoning_content"] = reasoning
    return messages


def _restore_output_tools(messages: list[JsonObject], output: JsonObject) -> None:
    """Restore exact provider argument text by call identity, not by tool position."""
    source = output.get("provider_tool_calls_json")
    if source is None:
        return
    if not isinstance(source, str):
        raise ValueError("invalid captured provider tools")
    calls = json.loads(source)
    if not isinstance(calls, list):
        raise ValueError("invalid captured provider tools")
    for captured in calls:
        if not isinstance(captured, dict) or not isinstance(captured.get("raw_arguments"), str):
            raise ValueError("invalid captured provider tool")
        for message in messages:
            tools = message.get("tool_calls")
            if not isinstance(tools, list):
                continue
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                function = tool.get("function", tool)
                if (
                    isinstance(function, dict)
                    and tool.get("id", tool.get("call_id")) == captured.get("call_id")
                    and function.get("name") == captured.get("name")
                ):
                    function["arguments"] = captured["raw_arguments"]


def _chat_output(experience: CapturedExchange) -> list[JsonObject]:
    """Read exactly one complete public Chat assistant turn."""
    choices = experience.response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("capture needs exactly one observed choice")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ValueError("capture has no complete assistant message")
    return [choice["message"]]


def _request_messages(request: GatewayRequest, provider: JsonValue) -> list[JsonObject]:
    """Restore serialization-excluded reasoning, tool failures, and raw arguments."""
    messages: list[JsonObject] = [
        message.model_dump(mode="json", exclude_none=True) for message in request.messages
    ]
    replay = provider.get("provider_replay") if isinstance(provider, dict) else None
    if isinstance(replay, list):
        for entry in replay:
            if not isinstance(entry, dict):
                raise ValueError("invalid provider capture context")
            index = entry.get("message_index")
            if not isinstance(index, int) or not 0 <= index < len(messages):
                raise ValueError("invalid provider message index")
            message = messages[index]
            for key, value in entry.items():
                if key not in {"message_index", "tool_calls"}:
                    message[key] = value
            calls = message.get("tool_calls")
            retained = entry.get("tool_calls")
            if isinstance(calls, list) and isinstance(retained, list):
                for call in retained:
                    if not isinstance(call, dict):
                        raise ValueError("invalid retained tool call")
                    position = call.get("tool_call_index")
                    if not isinstance(position, int) or not 0 <= position < len(calls):
                        raise ValueError("invalid retained tool index")
                    target = calls[position]
                    if not isinstance(target, dict) or target.get("call_id") != call.get("call_id"):
                        raise ValueError("retained tool identity mismatch")
                    raw = call.get("raw_arguments")
                    if isinstance(raw, str):
                        target["arguments"] = raw
                    target["provider_context"] = call
    for message in messages:
        blocks = message.get("provider_reasoning")
        if isinstance(blocks, list):
            text = "".join(
                block["content"]
                for block in blocks
                if isinstance(block, dict)
                and block.get("kind") == "exposed_reasoning_content"
                and isinstance(block.get("content"), str)
            )
            if text:
                message["reasoning_content"] = text
    return messages
