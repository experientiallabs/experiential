"""OpenRouter's ``usage: {"include": true}`` traverses the served native gateway."""

from __future__ import annotations

import json

import httpx
import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.gateway.tests.native_messages_test import (
    _engine,  # noqa: F401 - module fixture `engine`.
    _ServingEngine,
    _SseUpstream,
)

pytest.importorskip("exp_gateway_native")

_DISCLOSURES = "x-experiential-ignored-parameters"
_MESSAGES = [{"role": "user", "content": "hello"}]


def _post(engine: _ServingEngine, body: JsonObject) -> httpx.Response:
    """POST one Chat Completions body with the seeded key."""
    return httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json=body,
        timeout=30.0,
    )


def _last_upstream_payload() -> JsonObject:
    """Return the most recent body the loopback upstream received."""
    with _SseUpstream.payloads_lock:
        return _SseUpstream.payloads[-1]


def _sse_payloads(raw: str) -> list[JsonObject]:
    """Parse every JSON data frame of one SSE body."""
    return [
        json.loads(line.removeprefix("data: "))
        for line in raw.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


@pytest.mark.parametrize("include", (True, False))
def test_chat_usage_accounting_is_served_and_never_forwarded(
    engine: _ServingEngine, include: bool
) -> None:
    """A buffered call carrying ``usage`` serves with usage and no provider sees the field.

    The decoder used to refuse it with 400 ``unsupported_parameter`` ("The
    parameter 'usage' is not supported by this gateway profile.").
    """
    response = _post(
        engine, {"model": "coding", "messages": _MESSAGES, "usage": {"include": include}}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["usage"]["prompt_tokens"] == 9
    assert body["usage"]["completion_tokens"] == 4
    assert body.get(_DISCLOSURES, []) == (
        [] if include else ["usage.include->ignored(always_included)"]
    )
    sent = _last_upstream_payload()
    assert "usage" not in sent
    assert '"content": "hello"' in json.dumps(sent)


def test_chat_usage_accounting_on_a_stream_emits_the_final_usage_chunk(
    engine: _ServingEngine,
) -> None:
    """``usage.include: true`` on a stream ends with the usage chunk, as ``include_usage`` would."""
    with httpx.stream(
        "POST",
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={"model": "coding", "messages": _MESSAGES, "stream": True, "usage": {"include": True}},
        timeout=30.0,
    ) as response:
        assert response.status_code == 200
        raw = b"".join(response.iter_bytes()).decode()
    payloads = _sse_payloads(raw)
    assert payloads
    assert payloads[0][_DISCLOSURES] == ["usage.include->translated(stream_options.include_usage)"]
    usage_chunks = [
        usage for payload in payloads if isinstance(usage := payload.get("usage"), dict)
    ]
    assert usage_chunks, "stream never emitted the usage chunk"
    assert usage_chunks[-1]["prompt_tokens"] == 9
    assert usage_chunks[-1]["completion_tokens"] == 4
    sent = _last_upstream_payload()
    assert "usage" not in sent


@pytest.mark.parametrize(
    ("usage", "param"),
    (
        (True, "usage"),
        ({"include": "yes"}, "usage.include"),
        ({"include": True, "cost": True}, "usage.cost"),
    ),
)
def test_chat_usage_accounting_malformed_is_a_typed_400_before_dispatch(
    engine: _ServingEngine, usage: JsonValue, param: str
) -> None:
    with _SseUpstream.payloads_lock:
        before = len(_SseUpstream.payloads)
    response = _post(engine, {"model": "coding", "messages": _MESSAGES, "usage": usage})
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == "invalid_parameter"
    assert error["param"] == param
    with _SseUpstream.payloads_lock:
        assert len(_SseUpstream.payloads) == before
