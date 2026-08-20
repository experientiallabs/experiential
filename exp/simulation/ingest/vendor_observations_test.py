"""Tests for shared vendor observation readers."""

from __future__ import annotations

import pytest

from exp.simulation.ingest.vendor_observations import (
    VendorModelIdentity,
    VendorTokenUsage,
    VendorToolCall,
    declared_completion_text,
    declared_error_message,
    declared_model_identity,
    declared_tool_calls,
    declared_usage,
)
from exp.simulation.ingest.vendor_records import VendorTraceFormatError


def test_declared_tool_calls_reads_openai_and_nested_shapes() -> None:
    """Tool calls are read from message tool calls and from nested output envelopes."""
    output = {
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ]
    }

    assert declared_tool_calls(output) == (
        VendorToolCall(name="get_weather", arguments='{"city": "Paris"}', call_id="call-1"),
    )
    assert declared_tool_calls({"text": "no tools here"}) == ()
    assert declared_tool_calls(None) == ()


def test_declared_completion_text_prefers_visible_assistant_text() -> None:
    """Visible text wins over structured output, and structured output is retained compactly."""
    assert declared_completion_text("It is 18C.") == "It is 18C."
    assert declared_completion_text({"content": [{"type": "text", "text": "It is 18C."}]}) == (
        "It is 18C."
    )
    assert declared_completion_text({"score": 3}) == '{"score":3}'


def test_declared_usage_requires_complete_accounting() -> None:
    """Usage is retained only with both token counts, and bad counts fail loudly."""
    assert declared_usage({"promptTokens": 11, "completionTokens": 5}) == VendorTokenUsage(
        input_tokens=11, output_tokens=5
    )
    assert declared_usage({"input_tokens": 11}) is None
    assert declared_usage(None) is None
    with pytest.raises(VendorTraceFormatError, match="non-negative integer"):
        declared_usage({"input_tokens": -1, "output_tokens": 5})


def test_declared_model_identity_never_infers_a_provider() -> None:
    """Identity needs both fields, and a lone model name is returned as evidence."""
    identity, model_id = declared_model_identity(
        {"model": "gpt-4o", "provider": "openai"},
        model_keys=("model",),
        provider_keys=("provider",),
    )

    assert identity == VendorModelIdentity(provider="openai", model_id="gpt-4o")
    assert model_id == "gpt-4o"

    partial, partial_model = declared_model_identity(
        {"model": "gpt-4o"}, model_keys=("model",), provider_keys=("provider",)
    )

    assert partial is None
    assert partial_model == "gpt-4o"


def test_declared_error_message_reads_text_and_structured_errors() -> None:
    """Error text is read verbatim and structured errors are retained as compact JSON."""
    assert declared_error_message({"error": "boom"}, keys=("error",), label="span") == "boom"
    assert declared_error_message({"error": {"code": 500}}, keys=("error",), label="span") == (
        '{"code":500}'
    )
    assert declared_error_message({"error": {}}, keys=("error",), label="span") is None
    assert declared_error_message({}, keys=("error",), label="span") is None
