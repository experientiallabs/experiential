"""Freeze the current continuation's content and replay-size authority at admission."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayMessage
from exp.runtime.gateway.native_responses import ContinuationContext


def continuation_registration(context: ContinuationContext | None) -> JsonObject | None:
    """Transfer private retention context only when the caller permits storage."""
    if context is None or not context.retain:
        return None
    return {
        "namespace": context.namespace.model_dump(mode="json"),
        "episode_key": context.episode_key,
        "response_id": context.response_id,
        "messages": [message.model_dump(mode="json") for message in context.messages],
        "replay": [_replay(message, index) for index, message in enumerate(context.messages)],
        "message_template": GatewayMessage.model_construct(role="assistant").model_dump(
            mode="json"
        ),
        "route_bindings": [binding.model_dump(mode="json") for binding in context.route_bindings],
    }


def _replay(message: GatewayMessage, index: int) -> JsonObject:
    """Keep the same hidden fields counted by ContinuationState.size_bytes."""
    authority: JsonObject = {"message_index": index}
    if message.provider_item_id is not None:
        authority.update(
            provider_item_id=message.provider_item_id,
            provider_output_index=message.provider_output_index,
            provider_status=message.provider_status,
            provider_phase=message.provider_phase,
        )
    if message.provider_reasoning:
        blocks: list[JsonObject] = []
        for block in message.provider_reasoning:
            serialized = block.model_dump(mode="json")
            if block.kind == "encrypted_reasoning":
                serialized.update(output_index=block.output_index, status=block.status)
            blocks.append(serialized)
        authority["provider_reasoning"] = blocks
    calls: list[JsonObject] = []
    for call in message.tool_calls:
        if all(
            getattr(call, name) is None
            for name in (
                "raw_arguments",
                "provider_item_id",
                "provider_output_index",
                "provider_status",
                "provider_namespace",
                "provider_caller",
                "provider_thought_signature",
            )
        ):
            continue
        retained: JsonObject = {
            "call_id": call.call_id,
            "raw_arguments": call.raw_arguments,
            "provider_item_id": call.provider_item_id,
            "provider_output_index": call.provider_output_index,
            "provider_status": call.provider_status,
        }
        for name in ("provider_namespace", "provider_caller", "provider_thought_signature"):
            value = getattr(call, name)
            if value is not None:
                retained[name] = value
        calls.append(retained)
    if calls:
        authority["tool_calls"] = calls
    if message.tool_is_error:
        authority["tool_is_error"] = True
    for name in ("provider_tool_name", "provider_tool_namespace", "provider_tool_caller"):
        value = getattr(message, name)
        if value is not None:
            authority[name] = value
    return authority
