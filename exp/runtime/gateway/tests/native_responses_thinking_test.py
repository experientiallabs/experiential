"""DeepSeek's ``thinking`` switch on Responses traverses the served native gateway."""

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


def _post(engine: _ServingEngine, body: JsonObject) -> httpx.Response:
    """POST one Responses body with the seeded key."""
    return httpx.post(
        f"{engine.base}/v1/responses",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json=body,
        timeout=30.0,
    )


def _last_upstream_payload() -> JsonObject:
    """Return the most recent body the loopback upstream received."""
    with _SseUpstream.payloads_lock:
        return _SseUpstream.payloads[-1]


@pytest.mark.parametrize(
    ("kind", "route_disclosure"),
    (
        # The seeded route has no reasoning rung: the level-less enable drops
        # with disclosure, and the off switch's ``none`` effort is unsent.
        ("enabled", "enable_thinking->dropped(no_reasoning_rung)"),
        ("disabled", "reasoning_effort"),
    ),
)
def test_responses_thinking_is_served_and_never_forwarded(
    engine: _ServingEngine, kind: str, route_disclosure: str
) -> None:
    """A Responses call carrying DeepSeek's ``thinking`` serves; no provider sees the field."""
    response = _post(engine, {"model": "coding", "input": "hello", "thinking": {"type": kind}})
    assert response.status_code == 200, response.text
    assert response.json()["x-experiential-ignored-parameters"] == [
        "thinking->translated(reasoning.effort)",
        route_disclosure,
    ]
    sent = _last_upstream_payload()
    assert "thinking" not in sent
    assert '"hello"' in json.dumps(sent)


@pytest.mark.parametrize(
    ("thinking", "param"),
    (
        ("enabled", "thinking"),
        ({"type": "on"}, "thinking.type"),
    ),
)
def test_responses_thinking_malformed_is_a_typed_400_before_dispatch(
    engine: _ServingEngine, thinking: JsonValue, param: str
) -> None:
    """A malformed ``thinking`` is a typed 400 and never reaches the upstream."""
    with _SseUpstream.payloads_lock:
        before = len(_SseUpstream.payloads)
    response = _post(engine, {"model": "coding", "input": "hello", "thinking": thinking})
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["code"] == "invalid_parameter"
    assert error["param"] == param
    with _SseUpstream.payloads_lock:
        assert len(_SseUpstream.payloads) == before
