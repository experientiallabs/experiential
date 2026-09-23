"""Effective context capture preserves tools and never mutates serving input."""

import json
from unittest.mock import patch

import pytest

from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.gateway.capture_context import (
    capture_context_document,
    capture_request_context,
    restore_capture_context,
)
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.replay_identity import canonical_request_sha256, provider_replay_authority
from exp.runtime.openai_protocol.requests import decode_chat


@pytest.mark.parametrize("session_id", [None, "harness-session-123"])
def test_session_correlation_is_capture_only_and_reasoning_is_optional(
    session_id: str | None,
) -> None:
    """Capture accepts absent reasoning and never changes the provider request or replay key."""
    request = decode_chat(
        {"model": "open-model", "messages": [{"role": "user", "content": "hello"}]}
    ).request
    before = request.model_dump_json()
    digest = canonical_request_sha256(request)
    context = capture_request_context(request, session_id=session_id)
    assert context is not None
    assert context.get("session_id") == session_id
    assert context["request"] == request.model_dump(mode="json", exclude_none=True)
    assert request.model_dump_json() == before
    assert canonical_request_sha256(request) == digest


@pytest.mark.parametrize("session_id", ["", " ", "bad\nvalue", "bad\x00value", "雪", "x" * 513])
def test_invalid_optional_session_never_rejects_capture(session_id: str) -> None:
    """Malformed correlation is not required evidence and cannot break serving."""
    request = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": "hello"}]}
    ).request
    context = capture_request_context(request, session_id=session_id)
    assert context is not None and "session_id" not in context


@pytest.mark.parametrize("visible", ["Compare α and 雪.\n", "line one\x00line two\t"])
def test_sealed_messages_history_retains_visible_reasoning_only_for_capture(visible: str) -> None:
    """Visible history survives capture without joining authenticated provider replay."""
    carrier = "x-experiential-hunyuan-reasoning-v1:ZGVwbG95bWVudC0x:c2VhbGVkLWVudmVsb3Bl"
    request = decode_messages(
        {
            "model": "coding",
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": "Inspect the environment."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": visible, "signature": ""},
                        {"type": "tool_use", "id": "call-1", "name": "lookup", "input": {}},
                        {"type": "redacted_thinking", "data": carrier},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call-1", "content": "done"}
                    ],
                },
            ],
        }
    ).request
    before = request.model_dump_json()
    authority = provider_replay_authority(request)
    digest = canonical_request_sha256(request)
    context = capture_request_context(request)
    assert context is not None
    restored = restore_capture_context(context)
    provider = restored["provider_context"]
    assert isinstance(provider, dict)
    replay = provider["provider_replay"]
    assert isinstance(replay, list)
    assistant = next(
        entry for entry in replay if isinstance(entry, dict) and entry["message_index"] == 1
    )
    assert isinstance(assistant, dict)
    assert assistant["provider_reasoning"] == [
        {"kind": "sealed_reasoning_content", "carrier": carrier, "deployment_hint": "deployment-1"},
        {"kind": "exposed_reasoning_content", "content": visible},
    ]
    assert request.model_dump_json() == before
    assert provider_replay_authority(request) == authority
    assert canonical_request_sha256(request) == digest
    without_visible_copy = request.model_copy(
        update={
            "messages": tuple(
                message.model_copy(update={"capture_only_reasoning": ()})
                for message in request.messages
            )
        }
    )
    assert canonical_request_sha256(without_visible_copy) == digest
    assert provider_replay_authority(without_visible_copy) == authority
    assert [block.kind for block in request.messages[1].provider_reasoning] == [
        "sealed_reasoning_content"
    ]
    assert capture_request_context(request, maximum_bytes=1) is None


def test_storable_context_is_encoded_once_without_a_normalization_copy() -> None:
    """Ordinary traffic never pays for exceptional-text projection or a second size pass."""
    request = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": "café 雪"}]}
    ).request
    with (
        patch("exp.runtime.gateway.capture_context.json.dumps", wraps=json.dumps) as dumps,
        patch("exp.runtime.gateway.capture_context.normalize_durable_object") as normalize,
    ):
        assert capture_request_context(request) is not None
    assert dumps.call_count == 1
    normalize.assert_not_called()


def test_native_context_projection_never_serializes_ordinary_input() -> None:
    """The native envelope owns sizing; no intermediate JSON copy precedes it."""
    request = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": "x" * 1_100_000 + "雪"}]}
    ).request
    with patch("exp.runtime.gateway.capture_context.json.dumps") as dumps:
        context = capture_context_document(request)
    dumps.assert_not_called()
    assert context["request"] == request.model_dump(mode="json", exclude_none=True)


def test_valid_surrogate_pair_and_literal_escape_do_not_need_a_second_encoding() -> None:
    """A fast-path candidate that normalizes unchanged retains the original size check."""
    request = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": "😀 literal \\u0000"}]}
    ).request
    with patch("exp.runtime.gateway.capture_context.json.dumps", wraps=json.dumps) as dumps:
        context = capture_request_context(request)
    assert context is not None and "source_json" not in context
    assert dumps.call_count == 1


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


def test_capture_retains_hosted_effective_settings_without_changing_serving_input() -> None:
    """Capture keeps the old hosted envelope's excluded, nonempty request settings."""
    request = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": "hi"}]}
    ).request.model_copy(
        update={
            "provider_thinking_config": {"type": "enabled", "budget_tokens": 32},
            "diagnostics": {"trace": True},
            "speed": "fast",
            "provider_beta_tokens": ("interleaved-thinking",),
            "ignored_parameters": ("seed",),
            "idempotency_key": "not-captured",
        }
    )
    before = request.model_dump_json()
    context = capture_context_document(request)
    assert context["provider_internal"] == {
        "provider_thinking_config": {"type": "enabled", "budget_tokens": 32},
        "diagnostics": {"trace": True},
        "speed": "fast",
        "provider_beta_tokens": ["interleaved-thinking"],
        "ignored_parameters": ["seed"],
    }
    assert "not-captured" not in json.dumps(context)
    assert request.model_dump_json() == before


def test_capture_context_is_storable_and_omits_transport_replay_key() -> None:
    """Normalization touches the stored copy, not the served prompt or opaque key."""
    request = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": "a\x00b\ud800"}]}
    ).request.model_copy(update={"idempotency_key": "private-header"})
    before = request.model_dump()
    context = capture_request_context(request)
    assert context is not None
    serialized = json.dumps(context)
    assert "\x00" not in serialized
    restored = restore_capture_context(context)
    assert GatewayRequest.model_validate(restored["request"]).messages[0].content == "a\x00b\ud800"
    # Escapes inside source_json are ordinary text after JSONB decodes the envelope.
    source = context["source_json"]
    assert isinstance(source, str) and "\x00" not in source
    assert "private-header" not in serialized
    assert request.model_dump() == before


def test_lossless_context_retains_colliding_keys_and_enforces_total_budget() -> None:
    """A lossless sidecar never bypasses the admission memory ceiling."""
    request = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": "a\0b"}]}
    ).request
    context = capture_request_context(request)
    assert context is not None
    size = len(json.dumps(context, ensure_ascii=True, separators=(",", ":")))
    assert capture_request_context(request, maximum_bytes=size - 1) is None
    restored = GatewayRequest.model_validate(restore_capture_context(context)["request"])
    assert restored.messages[0].content == "a\0b"
