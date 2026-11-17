"""The strict thinking-signature opt-in through the real native Messages engine.

Each request crosses the Rust data plane, the admission callback, and an
Anthropic rung backed by a loopback upstream, so the header's transport, the
pre-dispatch refusal, and the unchanged default are all observed on the wire.
"""

from __future__ import annotations

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.tests.native_messages_test import (
    _REQUEST_TIMEOUT_SECONDS,
    _engine,  # noqa: F401 - module fixture `engine`.
    _ServingEngine,
    _SseUpstream,
)

pytest.importorskip("exp_gateway_native")

_HEADER = "x-experiential-strict-thinking-signatures"
_ANTHROPIC_ENGINE = pytest.mark.parametrize(
    "engine", (pytest.param("anthropic:claude-sonnet-5", id="anthropic"),), indirect=True
)


def _replay(engine: _ServingEngine, thinking: JsonObject, **headers: str) -> httpx.Response:
    """POST one conversation whose assistant turn replays ``thinking`` before its text."""
    with _SseUpstream.payloads_lock:
        _SseUpstream.payloads.clear()
    return httpx.post(
        f"{engine.base}/v1/messages",
        headers={"x-api-key": engine.raw_key, "anthropic-version": "2023-06-01", **headers},
        json={
            "model": "coding",
            "max_tokens": 128,
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": [thinking, {"type": "text", "text": "ok"}]},
                {"role": "user", "content": "again"},
            ],
        },
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )


def _upstream_payloads() -> list[JsonObject]:
    """Return every body the loopback Anthropic upstream received."""
    with _SseUpstream.payloads_lock:
        return list(_SseUpstream.payloads)


@_ANTHROPIC_ENGINE
@pytest.mark.parametrize(
    "thinking",
    (
        pytest.param({"type": "thinking", "thinking": "plan"}, id="missing"),
        pytest.param({"type": "thinking", "thinking": "plan", "signature": None}, id="null"),
        pytest.param({"type": "thinking", "thinking": "plan", "signature": ""}, id="empty"),
    ),
)
def test_strict_unsigned_thinking_is_an_anthropic_400_before_dispatch(
    engine: _ServingEngine, thinking: JsonObject
) -> None:
    """The opt-in refuses a stripped signature in Anthropic's envelope; nothing dispatches."""
    response = _replay(engine, thinking, **{_HEADER: "true"})
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "signature is required" in body["error"]["message"]
    assert body["error"]["message"].endswith("(param: messages.1.content.0.signature)")
    assert _upstream_payloads() == []


@_ANTHROPIC_ENGINE
def test_strict_signed_thinking_dispatches_byte_exact(engine: _ServingEngine) -> None:
    """A signed block reaches the Anthropic wire unchanged under the opt-in."""
    signed: JsonObject = {"type": "thinking", "thinking": "plan", "signature": "sig=="}
    response = _replay(engine, signed, **{_HEADER: "true"})
    assert response.status_code == 200, response.text
    captured = _upstream_payloads()
    assert len(captured) == 1
    messages = captured[0]["messages"]
    assert isinstance(messages, list)
    assistant = messages[1]
    assert isinstance(assistant, dict)
    content = assistant["content"]
    assert isinstance(content, list)
    assert content[0] == signed


@_ANTHROPIC_ENGINE
@pytest.mark.parametrize("headers", ({}, {_HEADER: "false"}), ids=("absent", "false"))
def test_without_the_opt_in_unsigned_thinking_is_dropped_on_the_anthropic_wire(
    engine: _ServingEngine, headers: dict[str, str]
) -> None:
    """The default is unchanged: the unsigned block is gateway-issued and never forwarded."""
    response = _replay(engine, {"type": "thinking", "thinking": "plan"}, **headers)
    assert response.status_code == 200, response.text
    captured = _upstream_payloads()
    assert len(captured) == 1
    assert "thinking" not in str(captured[0]["messages"])
