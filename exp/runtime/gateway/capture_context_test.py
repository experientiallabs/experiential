"""Effective context capture preserves tools and never mutates serving input."""

import json

from exp.runtime.gateway.capture_context import capture_request_context
from exp.runtime.openai_protocol.requests import decode_chat


def test_capture_context_preserves_tools_and_generation_settings() -> None:
    """The saved context includes definitions, not only observed tool calls."""
    request = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "lookup"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look up a record.",
                        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
                    },
                }
            ],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "temperature": 0.2,
            "max_tokens": 128,
        }
    ).request
    before = request.model_dump_json()
    context = capture_request_context(request)
    assert context is not None
    assert context["request"] == request.model_dump(mode="json", exclude_none=True)
    assert request.tools[0].parameters["type"] == "object"
    assert request.model_dump_json() == before
    assert capture_request_context(request, maximum_bytes=1) is None


def test_excluded_provider_carriers_are_retained_separately() -> None:
    """A provider's native tool declaration survives the capture-only projection."""
    request = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
        }
    ).request.model_copy(
        update={"provider_thinking_config": {"type": "enabled", "budget_tokens": 32}}
    )
    context = capture_request_context(request)
    assert context is not None
    provider = context["provider_context"]
    assert isinstance(provider, dict)
    assert provider["provider_thinking_config"] == {"type": "enabled", "budget_tokens": 32}


def test_capture_context_is_storable_and_omits_transport_replay_key() -> None:
    """Normalization touches the stored copy, not the served prompt or opaque key."""
    request = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": "a\x00b\ud800"}]}
    ).request.model_copy(update={"idempotency_key": "private-header"})
    before = request.model_dump()
    context = capture_request_context(request)
    assert context is not None
    serialized = json.dumps(context)
    assert "\\u0000" not in serialized
    assert "\\ud800" not in serialized
    assert "private-header" not in serialized
    assert request.model_dump() == before
