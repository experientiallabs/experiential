"""End-to-end parity tests for hosted and local gateway composition."""

from __future__ import annotations

import json
import re
import threading
from http.server import ThreadingHTTPServer
from inspect import Parameter, signature
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient
from pydantic import JsonValue

from wmo.runtime.gateway.composition import GatewayRuntimeConfig, create_gateway_runtime
from wmo.runtime.gateway.ledger import SQLiteAttemptLedger
from wmo.runtime.gateway.lifecycle import load_local_gateway
from wmo.runtime.gateway.tests.launch_test import _configure_gateway, _LoopbackProvider
from wmo.runtime.gateway.usage import read_usage_report

_DYNAMIC_FIELDS = frozenset({"completed_at", "created", "created_at", "id", "request_id"})


def test_public_factory_is_injected_and_worker_owned() -> None:
    """The public call site contains no local storage, lock, server, or environment input."""
    parameters = signature(create_gateway_runtime).parameters

    assert tuple(parameters) == (
        "config",
        "authority",
        "ledger",
        "routes",
        "executor",
        "clock",
        "readiness",
        "usage",
        "replay",
        "continuations",
        "wall_clock",
        "terminal_flusher",
    )
    assert all(item.kind is Parameter.KEYWORD_ONLY for item in parameters.values())
    with pytest.raises(ValueError, match="finite and positive"):
        GatewayRuntimeConfig(graceful_timeout_seconds=float("inf"))


def test_injected_runtime_matches_local_http_surface_end_to_end(tmp_path: Path) -> None:
    """Injected worker dependencies preserve every managed local gateway route shape."""
    _LoopbackProvider.calls = 0
    provider = ThreadingHTTPServer(("127.0.0.1", 0), _LoopbackProvider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    manager, raw_key = _configure_gateway(
        tmp_path,
        base_url=f"http://127.0.0.1:{provider.server_port}/v1",
    )
    local = load_local_gateway(
        tmp_path,
        graceful_timeout_seconds=1,
        environment={"LOOPBACK_PROVIDER_KEY": "provider-secret-canary"},
    )
    service = local.service
    ledger = cast(SQLiteAttemptLedger, service._ledger)  # noqa: SLF001 - parity seam evidence
    injected = create_gateway_runtime(
        config=GatewayRuntimeConfig(
            graceful_timeout_seconds=1,
            title="WMO local gateway",
        ),
        authority=service._control,  # noqa: SLF001 - parity seam evidence
        ledger=ledger,
        routes=service._routes,  # noqa: SLF001 - parity seam evidence
        executor=service._executor,  # noqa: SLF001 - parity seam evidence
        clock=service._clock,  # noqa: SLF001 - parity seam evidence
        readiness=service._readiness_probe,  # noqa: SLF001 - parity seam evidence
        usage=lambda: read_usage_report(ledger, organization_id=manager.organization_id),
    )

    try:
        with TestClient(local.app) as local_client, TestClient(injected.app) as injected_client:
            local_results = _exercise_surface(local_client, raw_key=raw_key)
            injected_results = _exercise_surface(injected_client, raw_key=raw_key)
        assert local.state.ready is False
        assert injected.state.ready is False
    finally:
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=2)

    assert local_results == injected_results
    assert _LoopbackProvider.calls == 8
    assert not provider_thread.is_alive()


def _exercise_surface(client: TestClient, *, raw_key: str) -> dict[str, JsonValue]:
    """Exercise every managed route with completed, streaming, and error traffic."""
    authorization = {"Authorization": f"Bearer {raw_key}"}
    live = client.get("/health/live")
    ready = client.get("/health/ready")
    models = client.get("/v1/models", headers=authorization)
    malformed = client.post(
        "/v1/chat/completions",
        headers={**authorization, "Content-Type": "application/json"},
        content="{",
    )
    chat = client.post(
        "/v1/chat/completions",
        headers=authorization,
        json={
            "model": "coding",
            "messages": [{"role": "user", "content": "chat-completed-canary"}],
        },
    )
    responses = client.post(
        "/v1/responses",
        headers=authorization,
        json={"model": "coding", "input": "responses-completed-canary"},
    )
    chat_stream = client.post(
        "/v1/chat/completions",
        headers=authorization,
        json={
            "model": "coding",
            "messages": [{"role": "user", "content": "chat-stream-canary"}],
            "stream": True,
        },
    )
    responses_stream = client.post(
        "/v1/responses",
        headers=authorization,
        json={"model": "coding", "input": "responses-stream-canary", "stream": True},
    )
    usage_json = client.get("/usage.json")
    usage_page = client.get("/usage")

    assert all(
        response.status_code == 200
        for response in (
            live,
            ready,
            usage_json,
            usage_page,
            models,
            chat,
            responses,
            chat_stream,
            responses_stream,
        )
    )
    assert malformed.status_code == 400
    return {
        "health_live": live.json(),
        "health_ready": ready.json(),
        "usage_json": _without_usage_counts(usage_json.json()),
        "usage_page_shape": _usage_page_shape(usage_page.text),
        "models": models.json(),
        "malformed": malformed.json(),
        "chat": _stable_json(chat.json()),
        "responses": _stable_json(responses.json()),
        "chat_stream": _stable_sse(chat_stream.text),
        "responses_stream": _stable_sse(responses_stream.text),
    }


def _stable_json(value: JsonValue) -> JsonValue:
    """Replace public request timestamps and opaque IDs while preserving response shape."""
    if isinstance(value, dict):
        return {
            key: (
                "<dynamic>" if key in _DYNAMIC_FIELDS or key.endswith("_id") else _stable_json(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_stable_json(item) for item in value]
    return value


def _stable_sse(body: str) -> list[JsonValue]:
    """Decode public SSE data frames and normalize only opaque identity fields."""
    frames: list[JsonValue] = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ")
        if payload == "[DONE]":
            frames.append(payload)
        else:
            frames.append(_stable_json(cast(JsonValue, json.loads(payload))))
    return frames


def _without_usage_counts(value: JsonValue) -> JsonValue:
    """Keep the usage schema while ignoring counts changed by the first client run."""
    if isinstance(value, dict):
        return {
            key: (
                0
                if isinstance(item, (int, float))
                else _without_usage_counts(cast(JsonValue, item))
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_without_usage_counts(item) for item in value]
    return value


def _usage_page_shape(body: str) -> str:
    """Return stable HTML structure without request-dependent numeric text."""
    return re.sub(r"\d+(?:\.\d+)?", "0", body)
