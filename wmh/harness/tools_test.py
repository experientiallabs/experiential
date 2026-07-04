"""Tests for the rollout agent's tool surface: parsing, rendering, and action mapping."""

from __future__ import annotations

import pytest

from wmh.core.types import ActionKind
from wmh.harness.tools import DEFAULT_TOOLS, parse_tool_call, render_tools, resolve_tools, to_action


def test_parse_tool_call_extracts_json_from_prose() -> None:
    call = parse_tool_call('sure!\n```json\n{"tool": "bash", "arguments": {"command": "ls"}}\n```')
    assert call is not None
    assert call.tool == "bash"
    assert call.arguments == {"command": "ls"}


def test_parse_tool_call_rejects_non_call_json_and_prose() -> None:
    assert parse_tool_call('{"foo": 1}') is None  # no `tool` field
    assert parse_tool_call("no json here") is None


def test_to_action_is_a_tool_call() -> None:
    call = parse_tool_call('{"tool": "write_file", "arguments": {"path": "/a", "content": "x"}}')
    assert call is not None
    action = to_action(call)
    assert action.kind == ActionKind.TOOL_CALL
    assert action.name == "write_file"


def test_resolve_tools_rejects_unknown_and_requires_submit() -> None:
    with pytest.raises(ValueError, match="not_a_tool"):
        resolve_tools(["bash", "submit", "not_a_tool"])
    with pytest.raises(ValueError, match="submit"):
        resolve_tools(["bash", "read_file"])  # no submit -> a run could never end


def test_render_tools_lists_names_and_args() -> None:
    rendered = render_tools(resolve_tools(list(DEFAULT_TOOLS)))
    assert "submit" in rendered
    assert "bash" in rendered
    assert "arguments:" in rendered
