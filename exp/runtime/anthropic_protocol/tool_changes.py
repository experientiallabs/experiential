"""Mid-conversation tool changes (``tool_addition`` / ``tool_removal``) on every wire.

Anthropic lets a ``role: "system"`` message inside ``messages`` change which
tools the model is offered from that point on, without editing ``tools`` (the
cached prefix stays intact):

* ``{"type": "tool_addition", "tool": {"type": "tool_reference", "name": X}}``
  offers a tool declared in ``tools`` (usually ``defer_loading: true``), or
  re-offers one an earlier removal withdrew;
* ``{"type": "tool_addition", "tool": {"type": "tool_definition",
  "definition": {...}}}`` defines a tool by value (beta
  ``inline-tools-2026-09-15``); a later definition under the same name
  replaces the earlier one;
* ``{"type": "tool_removal", "tool": {"type": "tool_reference", "name": X}}``
  withdraws a tool from that point on.

Claude Code puts these blocks after its ``text`` reminder in a mid-conversation
system turn whenever MCP tools arrive late or deferred tools load
(https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages).

No dispatch wire carries them as sent: the gateway re-emits every
mid-conversation system turn as instruction text (user text on Anthropic
rungs), and no other dialect has a positional tool list. So every rung,
Anthropic's included, receives the same explicit translation:

* the change is applied to the request's ``tools`` (:func:`apply_tool_changes`):
  a referenced tool loses ``defer_loading`` so it is offered immediately, and
  a defined tool joins ``tools`` (replacing a same-named declaration in
  place), disclosed as :data:`TOOL_ADDITION_DISCLOSURE`. The tool is offered
  from the start of the request rather than only from the system turn's
  position, a superset that keeps every tool the caller made available
  callable;
* a removal keeps the tool declared (earlier history may call it) and is
  stated as an instruction, disclosed as :data:`TOOL_REMOVAL_DISCLOSURE`;
* each block also becomes one line of text in its system turn
  (:func:`tool_change_text`), so the model learns of the change where it
  happened. A cache marker on the block (or inside its definition) stays on
  that line, the same prompt boundary.

A reference to a name that neither ``tools`` nor an earlier definition
declares is a named 400, as on Anthropic (``tool_reference_unresolved``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import Field

from exp.common.core.artifacts import JsonObject
from exp.runtime.anthropic_protocol.media_blocks import AnthropicWireModel, CacheControl
from exp.runtime.anthropic_protocol.server_tools import ServerTool
from exp.runtime.gateway.contracts import GatewayToolDefinition
from exp.runtime.openai_protocol.errors import invalid_field

TOOL_ADDITION_DISCLOSURE: Final = "messages.content.tool_addition->tools"
"""Disclosed when a ``tool_addition`` block is applied to the request's ``tools``."""

TOOL_REMOVAL_DISCLOSURE: Final = "messages.content.tool_removal->text"
"""Disclosed when a ``tool_removal`` block is stated as an instruction, the tool kept."""


class CustomTool(AnthropicWireModel):
    """One caller-defined custom tool with its JSON Schema declaration.

    Declared in ``tools`` or defined by value in a ``tool_addition`` block.
    Descriptions are preserved without a per-field size limit. The gateway's
    total request-body cap and provider context limits still apply.

    ``eager_input_streaming``, ``defer_loading``, ``allowed_callers``, and
    ``input_examples`` are provider-native tool annotations the live API
    accepts bare (each verified 2026-08-30; Claude Code sends
    ``eager_input_streaming`` conditionally). They forward verbatim on
    Anthropic rungs, which own their validity rules, and drop with
    disclosure elsewhere.

    Attributes:
        name: Tool name, unique among the request's tools.
        description: Optional verbatim tool instructions; bounded by the aggregate request limit.
        input_schema: The caller JSON Schema, unchanged.
        cache_control: Optional caller cache marker.
        type: Optional ``custom`` discriminator.
        strict: Whether the provider must enforce the schema.
        eager_input_streaming: Native Anthropic streaming selector.
        defer_loading: Native deferred-loading selector.
        allowed_callers: Native programmatic-tool caller allowlist.
        input_examples: Provider-visible example inputs.
    """

    name: str = Field(min_length=1, max_length=256)
    description: str | None = None
    input_schema: JsonObject
    cache_control: CacheControl | None = None
    type: Literal["custom"] | None = None
    strict: bool = False
    eager_input_streaming: bool | None = None
    defer_loading: bool | None = None
    allowed_callers: tuple[str, ...] | None = None
    input_examples: tuple[JsonObject, ...] | None = None


class ToolChangeReference(AnthropicWireModel):
    """A tool a mid-conversation tool change names, declared in ``tools``.

    Attributes:
        type: Always ``tool_reference``.
        name: The tool's declared (or earlier-defined) name.
    """

    type: Literal["tool_reference"]
    name: str = Field(min_length=1, max_length=256)


class ToolChangeDefinition(AnthropicWireModel):
    """A custom tool a ``tool_addition`` block defines by value.

    Anthropic also accepts server tools and ``mcp_toolset`` here; the gateway
    serves custom definitions only, and anything else is a named 400
    (``wire_validation.rejected_block_hint``).

    Attributes:
        type: Always ``tool_definition``.
        definition: The defined custom tool.
    """

    type: Literal["tool_definition"]
    definition: CustomTool


class ToolAdditionBlock(AnthropicWireModel):
    """Offer a tool from this point of the conversation.

    Attributes:
        type: Always ``tool_addition``.
        tool: The tool, by reference or by value.
        cache_control: Optional caller cache marker.
    """

    type: Literal["tool_addition"]
    tool: ToolChangeReference | ToolChangeDefinition
    cache_control: CacheControl | None = None


class ToolRemovalBlock(AnthropicWireModel):
    """Withdraw a tool from this point of the conversation.

    Attributes:
        type: Always ``tool_removal``.
        tool: The withdrawn tool, by reference.
        cache_control: Optional caller cache marker.
    """

    type: Literal["tool_removal"]
    tool: ToolChangeReference
    cache_control: CacheControl | None = None


@dataclass(frozen=True, slots=True)
class ToolChange:
    """One decoded mid-conversation tool change, in conversation order.

    Attributes:
        kind: ``addition`` or ``removal``.
        name: The tool the block names (a reference) or defines.
        definition: The canonical tool a by-value addition defines, else ``None``.
        param: Dotted public path of the block, for error messages.
    """

    kind: Literal["addition", "removal"]
    name: str
    definition: GatewayToolDefinition | None
    param: str


def tool_change_text(change: ToolChange) -> str:
    """Return the instruction line that states one tool change in its system turn.

    Args:
        change: The decoded change.

    Returns:
        One sentence naming the tool and what changed.
    """
    if change.kind == "removal":
        return f"Tool removed: {change.name}. It is no longer available; do not call it."
    if change.definition is not None and change.definition.defer_loading:
        return f"Tool declared: {change.name}. It is deferred until a tool search loads it."
    return f"Tool added: {change.name}. Its full definition is now in your tools; you can call it."


def apply_tool_changes(
    tools: tuple[GatewayToolDefinition, ...],
    server_tool_names: frozenset[str],
    changes: Sequence[ToolChange],
) -> tuple[tuple[GatewayToolDefinition, ...], tuple[str, ...]]:
    """Apply a conversation's tool changes to the request's declared tools.

    Args:
        tools: The custom tools declared in ``tools``, in caller order.
        server_tool_names: Names of the server tools declared in ``tools``.
        changes: Every tool change in the conversation, in order.

    Returns:
        ``(tools, disclosures)``: the tools every rung dispatches (a
        referenced tool undeferred, a defined tool added or replaced in
        place) and the disclosures the changes owe, in first-seen order.

    Raises:
        OpenAIProtocolError: A reference names no declared or earlier-defined
            tool, or a definition reuses a server tool's name.
    """
    resolved = list(tools)
    disclosures: list[str] = []

    def disclose(disclosure: str) -> None:
        """Record one disclosure once."""
        if disclosure not in disclosures:
            disclosures.append(disclosure)

    for change in changes:
        position = next(
            (index for index, tool in enumerate(resolved) if tool.name == change.name), None
        )
        if change.definition is not None:
            if change.name in server_tool_names:
                raise invalid_field(
                    f"{change.param}.tool.definition.name",
                    f"tool_addition defines '{change.name}', which is already used by a "
                    "server tool in tools. Give the tool a different name.",
                )
            if position is None:
                resolved.append(change.definition)
            else:
                resolved[position] = change.definition
            disclose(TOOL_ADDITION_DISCLOSURE)
            continue
        if position is None and change.name not in server_tool_names:
            block = "tool_addition" if change.kind == "addition" else "tool_removal"
            raise invalid_field(
                f"{change.param}.tool.name",
                f"{block} references '{change.name}', which is not declared in tools or "
                "defined by an earlier tool_addition. Declare the tool in tools (with "
                "defer_loading: true to withhold it until it is added).",
            )
        if change.kind == "removal":
            disclose(TOOL_REMOVAL_DISCLOSURE)
            continue
        if position is not None and resolved[position].defer_loading:
            resolved[position] = resolved[position].model_copy(update={"defer_loading": None})
        disclose(TOOL_ADDITION_DISCLOSURE)
    return tuple(resolved), tuple(disclosures)


def gateway_tool(tool: CustomTool) -> GatewayToolDefinition:
    """Convert one Anthropic custom tool to the canonical tool definition.

    Args:
        tool: The validated custom tool.

    Returns:
        The canonical tool definition.
    """
    return GatewayToolDefinition(
        name=tool.name,
        description=tool.description,
        parameters=tool.input_schema,
        strict=tool.strict,
        cache_control=(
            tool.cache_control.model_dump(mode="json", exclude_none=True)
            if tool.cache_control is not None
            else None
        ),
        eager_input_streaming=tool.eager_input_streaming,
        defer_loading=tool.defer_loading,
        allowed_callers=tool.allowed_callers,
        input_examples=tool.input_examples,
    )


def decode_tool_change(block: ToolAdditionBlock | ToolRemovalBlock, param: str) -> ToolChange:
    """Decode one mid-conversation tool change block.

    Args:
        block: The validated block.
        param: Dotted public path of the block.

    Returns:
        The change, a by-value definition already canonical. The
        definition's cache marker is left off the tool: it bounds the same
        prefix as the block's own, so it rides the block's instruction line
        (:func:`tool_change_marker`) and the tools array gains no breakpoint
        the caller did not put there.
    """
    tool = block.tool
    if isinstance(tool, ToolChangeDefinition):
        definition = gateway_tool(tool.definition)
        return ToolChange(
            kind="addition",
            name=definition.name,
            definition=definition.model_copy(update={"cache_control": None}),
            param=param,
        )
    kind: Literal["addition", "removal"] = (
        "addition" if isinstance(block, ToolAdditionBlock) else "removal"
    )
    return ToolChange(kind=kind, name=tool.name, definition=None, param=param)


def tool_change_marker(
    block: ToolAdditionBlock | ToolRemovalBlock, param: str
) -> CacheControl | None:
    """Return the cache marker a tool change carries, on the block or in its definition.

    Args:
        block: The validated block.
        param: Dotted public path of the block.

    Returns:
        The marker, or ``None`` when neither carries one.

    Raises:
        OpenAIProtocolError: Both carry one (Anthropic's own rule).
    """
    defined = (
        block.tool.definition.cache_control
        if isinstance(block.tool, ToolChangeDefinition)
        else None
    )
    if block.cache_control is not None and defined is not None:
        raise invalid_field(
            f"{param}.cache_control",
            "cache_control goes on the tool_addition block or in its definition, not both.",
        )
    return block.cache_control or defined


def tools_after_changes(
    declared: Sequence[CustomTool | ServerTool],
    contents: Sequence[str | Sequence[object]],
) -> tuple[tuple[GatewayToolDefinition, ...], tuple[str, ...]]:
    """Return the custom tools every rung dispatches, with tool changes applied.

    Args:
        declared: The request's validated ``tools`` entries.
        contents: Every message's validated ``content``, in conversation order.

    Returns:
        The canonical custom tools and the tool-change disclosures (``()``
        when the conversation carries no tool change).

    Raises:
        OpenAIProtocolError: A change cannot be applied (see :func:`apply_tool_changes`).
    """
    tools = tuple(gateway_tool(tool) for tool in declared if isinstance(tool, CustomTool))
    changes = [
        decode_tool_change(block, f"messages.{message_index}.content.{block_index}")
        for message_index, content in enumerate(contents)
        if not isinstance(content, str)
        for block_index, block in enumerate(content)
        if isinstance(block, (ToolAdditionBlock, ToolRemovalBlock))
    ]
    if not changes:
        return tools, ()
    server_tool_names = frozenset(tool.name for tool in declared if isinstance(tool, ServerTool))
    return apply_tool_changes(tools, server_tool_names, changes)
