"""Tests for the closed ``tool_use`` caller model."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from exp.runtime.anthropic_protocol.tool_use_caller import ToolUseCaller

_ADAPTER: TypeAdapter[object] = TypeAdapter(ToolUseCaller)


@pytest.mark.parametrize(
    "caller",
    (
        {"type": "direct"},
        {"type": "code_execution_20250825", "tool_id": "srvtoolu_a"},
        {"type": "code_execution_20260120", "tool_id": "srvtoolu_a"},
    ),
)
def test_official_caller_shapes_round_trip(caller: dict[str, str]) -> None:
    """Every SDK request shape validates and dumps back byte-for-byte."""
    validated = _ADAPTER.validate_python(caller)
    assert _ADAPTER.dump_python(validated, mode="json") == caller


@pytest.mark.parametrize(
    "caller",
    (
        {"type": "program", "id": "x"},
        {"type": "code_execution_20260120"},
        {"type": "code_execution_20260120", "tool_id": ""},
        {"type": "direct", "tool_id": "x"},
        "direct",
        None,
    ),
)
def test_other_shapes_are_refused(caller: object) -> None:
    """Unknown types, missing or empty ids, extra keys, and non-objects fail."""
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python(caller)
