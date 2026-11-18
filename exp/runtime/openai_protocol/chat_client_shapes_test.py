"""Tests for the disclosed rewrites of other clients' Chat request shapes."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.openai_protocol.chat_client_shapes import (
    EXTRA_BODY_FLATTENED,
    RESPONSES_FUNCTION_TOOLS_RESHAPED,
    RESPONSES_TOOL_CHOICE_RESHAPED,
    TEMPLATE_EFFORT_TRANSLATED,
    TEMPLATE_THINKING_TRANSLATED,
    flatten_extra_body,
    reshape_responses_function_tools,
    translate_template_reasoning_kwargs,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import decode_chat

_BASE: JsonObject = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
_SCHEMA: JsonObject = {"type": "object", "properties": {"path": {"type": "string"}}}


def test_a_responses_shaped_function_tool_decodes_as_a_chat_tool() -> None:
    """Cursor's flat tools and tool_choice decode exactly like the Chat spelling."""
    flat = decode_chat(
        {
            **_BASE,
            "tools": [
                {
                    "type": "function",
                    "name": "read",
                    "description": "Read a file.",
                    "parameters": _SCHEMA,
                    "strict": False,
                }
            ],
            "tool_choice": {"type": "function", "name": "read"},
        }
    ).request
    nested = decode_chat(
        {
            **_BASE,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file.",
                        "parameters": _SCHEMA,
                        "strict": False,
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "read"}},
        }
    ).request
    assert flat.tools == nested.tools
    assert flat.tool_choice == nested.tool_choice
    assert RESPONSES_FUNCTION_TOOLS_RESHAPED in flat.ignored_parameters
    assert RESPONSES_TOOL_CHOICE_RESHAPED in flat.ignored_parameters
    assert RESPONSES_FUNCTION_TOOLS_RESHAPED not in nested.ignored_parameters


def test_a_tool_carrying_both_shapes_keeps_its_named_refusal() -> None:
    """A stray flat name beside a Chat function object is the caller's to fix."""
    tool: JsonObject = {"type": "function", "name": "x", "function": {"name": "read"}}
    payload, disclosures = reshape_responses_function_tools({"tools": [tool]})
    assert payload == {"tools": [tool]} and disclosures == ()
    with pytest.raises(OpenAIProtocolError) as refused:
        decode_chat({**_BASE, "tools": [tool]})
    assert refused.value.detail.param == "tools.0.name"


def test_a_literal_extra_body_is_merged_like_the_sdk_merges_it() -> None:
    """WorkBuddy's literal extra_body reaches the same fields the SDK would send."""
    request = decode_chat(
        {**_BASE, "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}}
    ).request
    assert request.top_k == 20
    assert request.reasoning_effort == "none"
    assert EXTRA_BODY_FLATTENED in request.ignored_parameters
    assert flatten_extra_body({**_BASE, "extra_body": None}) == (_BASE, ())
    agreeing, _ = flatten_extra_body({**_BASE, "top_k": 3, "extra_body": {"top_k": 3}})
    assert agreeing["top_k"] == 3
    with pytest.raises(OpenAIProtocolError) as conflict:
        flatten_extra_body({**_BASE, "top_k": 3, "extra_body": {"top_k": 4}})
    assert conflict.value.detail.param == "extra_body.top_k"
    with pytest.raises(OpenAIProtocolError) as shape:
        flatten_extra_body({**_BASE, "extra_body": "x"})
    assert shape.value.detail.param == "extra_body"
    assert "top level" in shape.value.detail.message


def test_template_reasoning_switches_become_the_canonical_controls() -> None:
    """opencode's template effort and the DeepSeek/GLM thinking switch are translated."""
    effort = decode_chat({**_BASE, "chat_template_kwargs": {"reasoning_effort": "high"}}).request
    assert effort.reasoning_effort == "high"
    assert TEMPLATE_EFFORT_TRANSLATED in effort.ignored_parameters
    thinking = decode_chat({**_BASE, "chat_template_kwargs": {"thinking": True}}).request
    assert thinking.thinking_default_enable
    assert TEMPLATE_THINKING_TRANSLATED in thinking.ignored_parameters
    off = decode_chat({**_BASE, "chat_template_kwargs": {"thinking": False}}).request
    assert off.reasoning_effort == "none"
    same, disclosures = translate_template_reasoning_kwargs(
        {"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "low"}}
    )
    assert same == {"reasoning_effort": "low"}
    assert disclosures == (TEMPLATE_EFFORT_TRANSLATED,)
    with pytest.raises(OpenAIProtocolError) as conflict:
        translate_template_reasoning_kwargs(
            {"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "high"}}
        )
    assert conflict.value.detail.param == "chat_template_kwargs.reasoning_effort"
    with pytest.raises(OpenAIProtocolError) as switch:
        translate_template_reasoning_kwargs(
            {"chat_template_kwargs": {"thinking": True, "enable_thinking": False}}
        )
    assert switch.value.detail.param == "chat_template_kwargs.thinking"
    # An unknown template effort tier is still named by the effort validator.
    with pytest.raises(OpenAIProtocolError):
        decode_chat({**_BASE, "chat_template_kwargs": {"reasoning_effort": "turbo"}})


def test_an_empty_model_is_named_as_empty() -> None:
    """GitHub Copilot's empty model reads as empty, not as a contradiction."""
    with pytest.raises(OpenAIProtocolError) as refused:
        decode_chat({**_BASE, "model": ""})
    assert refused.value.detail.param == "model"
    assert refused.value.detail.message == (
        "Invalid value for 'model': expected a non-empty string, but got an empty string instead."
    )
