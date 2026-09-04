"""Reconstruct ordered canonical history from replayed Responses output items."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
from exp.runtime.gateway.contracts import (
    EncryptedReasoningBlock,
    GatewayMessage,
    SealedReasoningContentBlock,
)
from exp.runtime.openai_protocol.errors import invalid_field


@dataclass(frozen=True)
class ReplayedReasoning:
    """One validated provider reasoning item and its public input index."""

    index: int
    block: EncryptedReasoningBlock | SealedReasoningContentBlock


@dataclass(frozen=True)
class ReplayedMessage:
    """One validated canonical Responses message item."""

    index: int
    message: GatewayMessage


@dataclass(frozen=True)
class ReplayedFunctionCall:
    """One validated assistant function call."""

    index: int
    call: ToolCall


@dataclass(frozen=True)
class ReplayedNativeItem:
    """One Codex-native input item carried byte-for-byte (tool namespaces,
    freeform tool calls and their outputs)."""

    index: int
    role: Literal["developer", "assistant", "tool"]
    item: JsonObject


@dataclass(frozen=True)
class ReplayedFunctionOutput:
    """One validated function result.

    ``name`` and ``namespace`` are the optional tool attribution Codex
    serializes on outputs of namespaced calls, re-emitted verbatim.
    """

    index: int
    call_id: str
    output: str
    name: str | None = None
    namespace: str | None = None


ReplayedInput = (
    ReplayedReasoning
    | ReplayedMessage
    | ReplayedFunctionCall
    | ReplayedFunctionOutput
    | ReplayedNativeItem
)


def responses_input_messages(value: str | tuple[ReplayedInput, ...]) -> tuple[GatewayMessage, ...]:
    """Convert replay items while keeping one Fireworks assistant segment indivisible.

    Native Responses output indexes reflect the first observed provider delta, so a
    Fireworks text item may appear before its reasoning carrier. The carrier authenticates
    the complete assistant text and calls, therefore all contiguous assistant output items
    are collected before reconstructing that turn instead of assuming reasoning is first.
    """
    if isinstance(value, str):
        return (GatewayMessage(role="user", content=value),)
    messages: list[GatewayMessage] = []
    segment: list[ReplayedReasoning | ReplayedMessage | ReplayedFunctionCall] = []

    def flush_segment() -> None:
        """Emit one contiguous assistant segment without regrouping ordinary OpenAI items."""
        if not segment:
            return
        reasoning = [item.block for item in segment if isinstance(item, ReplayedReasoning)]
        if any(block.kind == "sealed_reasoning_content" for block in reasoning):
            assistant_items = [item for item in segment if isinstance(item, ReplayedMessage)]
            if len(assistant_items) > 1:
                raise invalid_field(
                    f"input.{assistant_items[1].index}",
                    "A Fireworks continuation may contain only one assistant message item.",
                )
            content = assistant_items[0].message.content if assistant_items else None
            calls = tuple(item.call for item in segment if isinstance(item, ReplayedFunctionCall))
            messages.append(
                GatewayMessage(
                    role="assistant",
                    content=content,
                    tool_calls=calls,
                    provider_reasoning=tuple(reasoning),
                )
            )
            segment.clear()
            return

        pending: list[EncryptedReasoningBlock | SealedReasoningContentBlock] = []
        pending_calls: list[ToolCall] = []

        def flush_calls() -> None:
            """Emit contiguous provider calls as one assistant turn."""
            if not pending_calls:
                return
            messages.append(
                GatewayMessage(
                    role="assistant",
                    tool_calls=tuple(pending_calls),
                    provider_reasoning=tuple(pending),
                )
            )
            pending.clear()
            pending_calls.clear()

        for item in segment:
            if isinstance(item, ReplayedReasoning):
                flush_calls()
                pending.append(item.block)
            elif isinstance(item, ReplayedMessage):
                flush_calls()
                messages.append(
                    item.message.model_copy(update={"provider_reasoning": tuple(pending)})
                )
                pending.clear()
            else:
                pending_calls.append(item.call)
        flush_calls()
        if pending:
            messages.append(GatewayMessage(role="assistant", provider_reasoning=tuple(pending)))
        segment.clear()

    for item in value:
        if isinstance(item, ReplayedNativeItem):
            # Native items break the assistant segment and keep their exact
            # position; the payload builder re-emits them verbatim.
            flush_segment()
            messages.append(GatewayMessage(role=item.role, provider_native_item=item.item))
        elif isinstance(item, ReplayedMessage) and item.message.role != "assistant":
            flush_segment()
            messages.append(item.message)
        elif isinstance(item, ReplayedFunctionOutput):
            flush_segment()
            messages.append(
                GatewayMessage(
                    role="tool",
                    content=item.output,
                    tool_call_id=item.call_id,
                    provider_tool_name=item.name,
                    provider_tool_namespace=item.namespace,
                )
            )
        else:
            segment.append(item)
    flush_segment()
    return tuple(messages)
