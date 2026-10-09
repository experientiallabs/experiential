"""Tests for the strict OpenAI custom tool wire schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exp.runtime.openai_protocol.custom_tool_wire import ChatCustomTool, ChatCustomToolCall


def test_a_grammar_format_requires_its_grammar_body() -> None:
    """``type: grammar`` needs the nested grammar; ``type: text`` must not carry one."""
    with pytest.raises(ValidationError):
        ChatCustomTool.model_validate({"name": "x", "format": {"type": "grammar"}})
    with pytest.raises(ValidationError):
        ChatCustomTool.model_validate(
            {
                "name": "x",
                "format": {"type": "text", "grammar": {"syntax": "lark", "definition": "a"}},
            }
        )
    tool = ChatCustomTool.model_validate(
        {
            "name": "x",
            "format": {"type": "grammar", "grammar": {"syntax": "regex", "definition": "a"}},
        }
    )
    assert tool.format is not None and tool.format.grammar is not None


def test_a_custom_call_is_strict_about_its_shape() -> None:
    """A custom call needs its type and body and refuses a function body beside them."""
    call = ChatCustomToolCall.model_validate(
        {"id": "c", "type": "custom", "custom": {"name": "x", "input": "raw"}, "index": 0}
    )
    assert call.custom.input == "raw"
    with pytest.raises(ValidationError):
        ChatCustomToolCall.model_validate(
            {
                "id": "c",
                "type": "custom",
                "custom": {"name": "x", "input": "raw"},
                "function": {"name": "x", "arguments": "{}"},
            }
        )
