"""Reconstruct ordered canonical history from replayed Responses output items."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeGuard

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
from exp.common.models.content import MessageContentPart
from exp.runtime.gateway.contracts import (
    EncryptedReasoningBlock,
    ExposedReasoningContentBlock,
    GatewayMessage,
    SealedReasoningContentBlock,
)
from exp.runtime.openai_protocol.errors import invalid_field


@dataclass(frozen=True)
class ReplayedReasoning:
    """One validated provider reasoning item and its public input index.

    Attributes:
        index: The item's position in the caller's ``input`` array.
        block: The provider-opaque encrypted reasoning or the gateway-sealed carrier.
        visible: The caller-visible summary parts replayed beside a gateway
            carrier, one block per part. Trace capture keeps them; provider
            replay never sees them, since the carrier holds the authenticated text.
    """

    index: int
    block: EncryptedReasoningBlock | SealedReasoningContentBlock
    visible: tuple[ExposedReasoningContentBlock, ...] = ()


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
    serializes on outputs of namespaced calls, re-emitted verbatim; ``caller``
    is SDK 3.0's opaque programmatic tool-calling attribution, re-emitted the
    same way.
    """

    index: int
    call_id: str
    output: str
    name: str | None = None
    namespace: str | None = None
    caller: JsonObject | None = None
    content_parts: tuple[MessageContentPart, ...] = ()
    """Ordered canonical parts when the SDK list form carried an image
    beside text; empty for the plain-text form, whose message shape is
    unchanged."""


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
                    "A gateway reasoning-carrier continuation may contain only one "
                    "assistant message item.",
                )
            content = assistant_items[0].message.content if assistant_items else None
            calls = tuple(item.call for item in segment if isinstance(item, ReplayedFunctionCall))
            messages.append(
                GatewayMessage(
                    role="assistant",
                    content=content,
                    tool_calls=calls,
                    provider_reasoning=tuple(reasoning),
                    capture_only_reasoning=tuple(
                        visible
                        for item in segment
                        if isinstance(item, ReplayedReasoning)
                        for visible in item.visible
                    ),
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

    # Assistant output items of the current run, with any developer or system
    # instructions a client recorded between them (see ``flush_run``).
    run: list[ReplayedReasoning | ReplayedMessage | ReplayedFunctionCall] = []

    def flush_run() -> None:
        """Emit one assistant output run, keeping a carrier with its own tool turn.

        Codex records items as they stream and can append a developer reminder
        after a function call it is already executing but before that turn's
        reasoning item lands, so a gateway carrier arrives separated from its
        calls by an instruction. A response that issued a function call cannot be
        followed by another response before the call's output, so in a run that
        holds a carrier, an instruction recorded after a function call belongs
        inside that call's turn: the turn is rebuilt whole and the instruction
        follows it, ahead of the tool outputs. An instruction before any call
        still delimits two responses, and a run without a carrier keeps every
        item in its original position.
        """
        carried = any(
            isinstance(item, ReplayedReasoning) and item.block.kind == "sealed_reasoning_content"
            for item in run
        )
        trailing: list[GatewayMessage] = []
        for item in run:
            if not _is_instruction(item):
                segment.append(item)
            elif carried and any(isinstance(entry, ReplayedFunctionCall) for entry in segment):
                trailing.append(item.message)
            else:
                flush_segment()
                messages.append(item.message)
        flush_segment()
        messages.extend(trailing)
        run.clear()

    for item in value:
        if isinstance(item, ReplayedNativeItem):
            # Native items break the assistant segment and keep their exact
            # position; the payload builder re-emits them verbatim.
            flush_run()
            messages.append(GatewayMessage(role=item.role, provider_native_item=item.item))
        elif isinstance(item, ReplayedMessage) and _is_instruction(item) and run:
            run.append(item)
        elif isinstance(item, ReplayedMessage) and item.message.role != "assistant":
            flush_run()
            messages.append(item.message)
        elif isinstance(item, ReplayedFunctionOutput):
            flush_run()
            messages.append(
                GatewayMessage(
                    role="tool",
                    content=item.output,
                    content_parts=item.content_parts,
                    tool_call_id=item.call_id,
                    provider_tool_name=item.name,
                    provider_tool_namespace=item.namespace,
                    provider_tool_caller=item.caller,
                )
            )
        else:
            run.append(item)
    flush_run()
    return tuple(messages)


def _is_instruction(
    item: ReplayedReasoning | ReplayedMessage | ReplayedFunctionCall,
) -> TypeGuard[ReplayedMessage]:
    """Return whether a replayed item is a developer or system instruction message."""
    return isinstance(item, ReplayedMessage) and item.message.role in {"developer", "system"}
