"""Tests for whether declared model capabilities can serve one request."""

from __future__ import annotations

import pytest

from exp.common.models import ModelCapabilities, ModelRequest
from exp.common.models.capability_fit import capabilities_support, requires_tool_protocol
from exp.common.models.model import ModelMessage
from exp.common.tasks import ToolSchema

_TOOL = ToolSchema(name="run_tests", description="Run the tests.", input_schema={"type": "object"})


def _request(*, tools: bool = False, maximum_output_tokens: int | None = None) -> ModelRequest:
    """Return one user request, optionally offering a tool or capping its output."""
    return ModelRequest(
        messages=(ModelMessage(role="user", content="hi"),),
        tools=(_TOOL,) if tools else (),
        maximum_output_tokens=maximum_output_tokens,
    )


@pytest.mark.parametrize(
    ("supports_tools", "tools", "expected"),
    [(False, True, False), (None, True, True), (True, True, True), (False, False, True)],
)
def test_only_an_explicit_no_tools_declaration_excludes_a_tool_request(
    supports_tools: bool | None, tools: bool, expected: bool
) -> None:
    """Unknown tool support is permissive; a declared ``False`` rules a tool request out."""
    capabilities = ModelCapabilities(supports_tools=supports_tools)
    assert capabilities_support(capabilities, _request(tools=tools)) is expected


@pytest.mark.parametrize(
    ("declared", "requested", "expected"),
    [(1_000, 2_000, False), (2_000, 2_000, True), (None, 2_000, True), (1_000, None, True)],
)
def test_a_declared_output_limit_below_the_request_excludes_the_model(
    declared: int | None, requested: int | None, expected: bool
) -> None:
    """Only a declared limit smaller than the requested output rules a model out."""
    capabilities = ModelCapabilities(maximum_output_tokens=declared)
    request = _request(maximum_output_tokens=requested)
    assert capabilities_support(capabilities, request) is expected


def test_a_tool_result_in_history_needs_tool_support() -> None:
    """A conversation that already used tools keeps needing them."""
    history = ModelRequest(
        messages=(
            ModelMessage(role="user", content="hi"),
            ModelMessage(role="tool", content="passed", tool_call_id="call-1"),
        )
    )
    assert requires_tool_protocol(history)
    assert not requires_tool_protocol(_request())
