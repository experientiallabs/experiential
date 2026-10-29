"""Field-specific public errors from the Messages wire validation."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, Field

from exp.common.core.artifacts import JsonObject
from exp.runtime.anthropic_protocol.wire_validation import rejected_block_hint, validate_wire
from exp.runtime.openai_protocol.errors import OpenAIProtocolError


class _Wire(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    max_tokens: int | None = None


def test_validate_wire_returns_the_validated_model_or_names_the_field() -> None:
    """The generic validator returns the model it was handed and names the offending field."""
    assert validate_wire({"model": "coding"}, _Wire).model == "coding"
    with pytest.raises(OpenAIProtocolError) as unknown:
        validate_wire({"model": "coding", "stream": True}, _Wire)
    assert unknown.value.detail.param == "stream"
    with pytest.raises(OpenAIProtocolError) as missing:
        validate_wire({}, _Wire)
    assert missing.value.detail.param == "model"


def test_rejected_block_hint_names_the_offending_block() -> None:
    """A known-but-unsupported block is named by its exact path, tool_result sub-blocks included."""
    assert rejected_block_hint({"messages": "not a list"}) is None
    video: JsonObject = {
        "messages": [{"role": "user", "content": [{"type": "video", "source": {}}]}]
    }
    hint = rejected_block_hint(video)
    assert hint is not None
    assert hint[0] == "messages.0.content.0"
    nested: JsonObject = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": "ok"}, {"type": "video"}],
                    }
                ],
            }
        ]
    }
    nested_hint = rejected_block_hint(nested)
    assert nested_hint is not None
    assert nested_hint[0] == "messages.0.content.0.content.1"
    # A document is a served tool_result sub-block (Claude Code's Read on a
    # PDF), so it is never the block a hint blames.
    document_only: JsonObject = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": "ok"}, {"type": "document"}],
                    }
                ],
            }
        ]
    }
    assert rejected_block_hint(document_only) is None
    # A tool_reference is a served tool_result sub-block too (Claude Code's
    # client-side ToolSearch), so it never takes the blame either.
    reference_only: JsonObject = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "tool_reference", "tool_name": "Lookup"}],
                    }
                ],
            }
        ]
    }
    assert rejected_block_hint(reference_only) is None


def test_tool_change_blocks_are_known_block_types() -> None:
    """A served tool change never takes the unknown-block blame; MCP references are named."""
    served: JsonObject = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "tools changed"},
                    {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "a"}},
                    {"type": "tool_removal", "tool": {"type": "tool_reference", "name": "b"}},
                ],
            }
        ]
    }
    assert rejected_block_hint(served) is None
    mcp: JsonObject = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "tool_removal",
                        "tool": {"type": "mcp_tool_reference", "server_name": "s", "name": "t"},
                    }
                ],
            }
        ]
    }
    hint = rejected_block_hint(mcp)
    assert hint is not None
    assert hint[0] == "messages.0.content.0.tool.type"
    assert "mcp_tool_reference is not supported" in hint[1]
