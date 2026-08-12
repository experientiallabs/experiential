"""HTTP routing endpoint transcript, retry, and caller-validation tests."""

from __future__ import annotations

import hashlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from wmo.runtime.router import create_router_endpoint
from wmo.runtime.router.runtime_test import _runtime


def test_endpoint_requires_episode_rejects_stream_and_preserves_tool_transcript() -> None:
    """HTTP is non-streaming and preserves tools while keeping later turns sticky."""
    runtime, client = _runtime()
    app = FastAPI()
    app.include_router(create_router_endpoint({"router-a": runtime}))
    http = TestClient(app)
    payload = {
        "model": "router-a",
        "messages": [
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-in",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path":"a"}'},
                    }
                ],
            },
            {"role": "tool", "content": "result", "tool_call_id": "call-in"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "read a file",
                    "parameters": {"type": "object"},
                },
            }
        ],
    }
    assert http.post("/v1/chat/completions", json=payload).status_code == 400
    assert (
        http.post(
            "/v1/chat/completions",
            json={**payload, "stream": True},
            headers={"X-WMO-Episode-ID": "episode-a"},
        ).status_code
        == 400
    )
    response = http.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-WMO-Episode-ID": "episode-a"},
    )

    assert response.status_code == 200
    assert response.headers["X-WMO-Episode-ID-SHA256"] == hashlib.sha256(b"episode-a").hexdigest()
    assert "episode-a" not in response.text
    assert response.json()["choices"][0]["message"]["tool_calls"][0]["id"] == "call-out"
    captured = client.requests[-1]
    assert [message.role for message in captured.messages] == ["user", "assistant", "tool"]
    assert captured.messages[1].assistant_action is not None
    assert captured.messages[1].assistant_action.tool_calls[0].call_id == "call-in"
    assert captured.messages[2].tool_call_id == "call-in"
    next_turn = http.post(
        "/v1/chat/completions",
        json={
            **payload,
            "messages": [*payload["messages"], {"role": "user", "content": "next turn"}],
        },
        headers={"X-WMO-Episode-ID": "episode-a"},
    )
    assert next_turn.status_code == 200
    assert next_turn.headers["X-WMO-Routed-Model"] == response.headers["X-WMO-Routed-Model"]
    assert (
        next_turn.json()["routing_decision"]["decision_id"]
        == response.json()["routing_decision"]["decision_id"]
    )
    assert client.embed_calls == 1
    assert len(client.requests[-1].messages) == 4
    assert "episode-a" not in next_turn.text


def test_endpoint_provider_retry_reuses_exact_cached_request_decision() -> None:
    """A provider failure can retry the same turn without reselection or decision drift."""
    runtime, client = _runtime()
    client.completion_error = RuntimeError("provider unavailable")
    app = FastAPI()
    app.include_router(create_router_endpoint({"router-a": runtime}))
    http = TestClient(app)
    payload = {"model": "router-a", "messages": [{"role": "user", "content": "retry me"}]}
    headers = {"X-WMO-Episode-ID": "episode-a"}

    assert http.post("/v1/chat/completions", json=payload, headers=headers).status_code == 502
    cached = next(iter(runtime._request_decisions.values()))  # noqa: SLF001 - retry identity probe
    response = http.post("/v1/chat/completions", json=payload, headers=headers)

    assert response.status_code == 200
    assert response.json()["routing_decision"]["decision_id"] == cached.decision_id
    assert client.embed_calls == 1
    assert client.complete_calls == 2


@pytest.mark.parametrize(
    "request_update",
    [
        {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "x",
                            "function": {"name": "read", "arguments": "[1]"},
                        }
                    ],
                }
            ]
        },
        {
            "tools": [
                {"function": {"name": "read", "parameters": {}}},
                {"function": {"name": "read", "parameters": {}}},
            ]
        },
        {"tool_choice": {"function": {"name": "missing"}}},
        {"max_completion_tokens": 0},
    ],
)
def test_invalid_http_request_never_reaches_selection_or_provider(
    request_update: dict[str, object],
) -> None:
    """Caller message, tool, choice, and token validation failures remain actionable 4xx."""
    runtime, client = _runtime()
    app = FastAPI()
    app.include_router(create_router_endpoint({"router-a": runtime}))
    http = TestClient(app)
    payload: dict[str, object] = {
        "model": "router-a",
        "messages": [{"role": "user", "content": "validate me"}],
        **request_update,
    }

    response = http.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-WMO-Episode-ID": "episode-a"},
    )

    assert response.status_code in {400, 422}
    assert client.embed_calls == 0
    assert client.complete_calls == 0
