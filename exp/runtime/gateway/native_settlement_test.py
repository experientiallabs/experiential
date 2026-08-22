"""Tests for native gateway control plane settlement wire parsing."""

from __future__ import annotations

from exp.runtime.gateway.native_settlement import usage_from_payload


def test_usage_from_payload_handles_tokens_and_tool_names() -> None:
    """Settlement usage covers token totals, tool-only, and absent cases."""
    assert usage_from_payload(None, []) is None
    tools_only = usage_from_payload(None, ["search"])
    assert tools_only is not None and tools_only.tool_names == ("search",)
    complete = usage_from_payload(
        {"input_tokens": 10, "output_tokens": 3, "cached_input_tokens": 2},
        [],
    )
    assert complete is not None
    assert complete.input_tokens == 10
    assert complete.output_tokens == 3
    assert complete.cached_input_tokens == 2
