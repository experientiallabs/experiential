"""Mid-conversation tool changes applied to the request's tools."""

from __future__ import annotations

import pytest

from exp.runtime.anthropic_protocol.tool_changes import (
    TOOL_ADDITION_DISCLOSURE,
    TOOL_REMOVAL_DISCLOSURE,
    ToolChange,
    apply_tool_changes,
    tool_change_text,
)
from exp.runtime.gateway.contracts import GatewayToolDefinition
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


def _tool(
    name: str, *, deferred: bool = False, description: str | None = None
) -> GatewayToolDefinition:
    """One canonical tool, optionally deferred."""
    return GatewayToolDefinition(
        name=name,
        description=description,
        parameters={"type": "object"},
        defer_loading=True if deferred else None,
    )


def _reference(kind: str, name: str) -> ToolChange:
    """One change that names a tool by reference."""
    return ToolChange(
        kind="addition" if kind == "addition" else "removal",
        name=name,
        definition=None,
        param="messages.1.content.0",
    )


def test_an_addition_by_reference_undefers_the_named_tool_only() -> None:
    """The referenced deferred tool is offered immediately; its siblings stay deferred."""
    tools, disclosures = apply_tool_changes(
        (_tool("Bash"), _tool("a", deferred=True), _tool("b", deferred=True)),
        frozenset(),
        [_reference("addition", "a")],
    )
    assert [(tool.name, tool.defer_loading) for tool in tools] == [
        ("Bash", None),
        ("a", None),
        ("b", True),
    ]
    assert disclosures == (TOOL_ADDITION_DISCLOSURE,)


def test_a_definition_appends_or_replaces_in_place() -> None:
    """A new name joins the end; a declared name is replaced at its position."""
    new = ToolChange(
        kind="addition", name="late", definition=_tool("late"), param="messages.1.content.0"
    )
    replacement = ToolChange(
        kind="addition",
        name="Bash",
        definition=_tool("Bash", description="v2"),
        param="messages.3.content.0",
    )
    tools, disclosures = apply_tool_changes(
        (_tool("Bash"), _tool("Read")), frozenset(), [new, replacement]
    )
    assert [tool.name for tool in tools] == ["Bash", "Read", "late"]
    assert tools[0].description == "v2"
    assert disclosures == (TOOL_ADDITION_DISCLOSURE,)


def test_a_removal_keeps_the_tool_declared_and_is_disclosed() -> None:
    """History may still call a removed tool, so its definition stays."""
    declared = (_tool("Bash"), _tool("a", deferred=True))
    tools, disclosures = apply_tool_changes(declared, frozenset(), [_reference("removal", "a")])
    assert tools == declared
    assert disclosures == (TOOL_REMOVAL_DISCLOSURE,)


def test_a_server_tool_reference_resolves_and_changes_no_custom_tool() -> None:
    """A reference may name a declared server tool."""
    declared = (_tool("Bash"),)
    tools, disclosures = apply_tool_changes(
        declared,
        frozenset({"web_search"}),
        [_reference("removal", "web_search"), _reference("addition", "web_search")],
    )
    assert tools == declared
    assert disclosures == (TOOL_REMOVAL_DISCLOSURE, TOOL_ADDITION_DISCLOSURE)


@pytest.mark.parametrize("kind", ["addition", "removal"])
def test_an_unresolved_reference_is_a_named_400(kind: str) -> None:
    """A name neither tools nor an earlier definition declares is refused."""
    with pytest.raises(OpenAIProtocolError) as excinfo:
        apply_tool_changes((_tool("Bash"),), frozenset(), [_reference(kind, "ghost")])
    assert excinfo.value.detail.param == "messages.1.content.0.tool.name"
    assert f"tool_{kind} references 'ghost'" in excinfo.value.detail.message


def test_a_definition_may_not_take_a_server_tool_name() -> None:
    """A by-value definition reusing a server tool's name is a conflict."""
    change = ToolChange(
        kind="addition",
        name="web_search",
        definition=_tool("web_search"),
        param="messages.1.content.0",
    )
    with pytest.raises(OpenAIProtocolError) as excinfo:
        apply_tool_changes((), frozenset({"web_search"}), [change])
    assert excinfo.value.detail.param == "messages.1.content.0.tool.definition.name"


def test_each_change_states_itself_in_one_line() -> None:
    """Additions, deferred definitions, and removals each read as one sentence."""
    assert tool_change_text(_reference("addition", "a")) == (
        "Tool added: a. Its full definition is now in your tools; you can call it."
    )
    deferred = ToolChange(
        kind="addition",
        name="d",
        definition=_tool("d", deferred=True),
        param="messages.1.content.0",
    )
    assert tool_change_text(deferred) == (
        "Tool declared: d. It is deferred until a tool search loads it."
    )
    assert tool_change_text(_reference("removal", "a")) == (
        "Tool removed: a. It is no longer available; do not call it."
    )
