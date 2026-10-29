"""Request-owned attempt budgets cross official SDKs and the real native server."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import anthropic
import httpx
import openai
import pytest
from websockets.sync.client import connect

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.health import DeploymentHealthRegistry
from exp.runtime.gateway.native_bridge_test import _configured_pool_gateway
from exp.runtime.gateway.tests.launch_test import _ServedGateway, _unused_port
from exp.runtime.gateway.tests.native_tool_search_test import _TOOLS_CHAT, _configure, _Provider


class ScriptedProvider:
    """Count actual HTTP dispatches, not SDK calls or attempt reservations.

    Attributes:
        server: Loopback server receiving model generation requests.
        calls: Model ids observed in physical requests.
        bodies: Fully decoded provider bodies proving no gateway leakage.
        times: Monotonic start time for every physical request.
        thread: Owned provider serving thread.
    """

    def __init__(self, status: int, failures: int) -> None:
        """Fail the primary's first requests, then return a finite completion."""
        self.calls: list[str] = []
        self.bodies: list[JsonObject] = []
        self.times: list[float] = []
        provider = self

        class Handler(BaseHTTPRequestHandler):
            """Expose the scripted chat-compatible origin."""

            def do_POST(self) -> None:  # noqa: N802
                """Record a physical call and answer it deterministically."""
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                model = body["model"]
                provider.calls.append(model)
                provider.bodies.append(body)
                provider.times.append(time.monotonic())
                fails = model == "alpha-model-exact" and provider.calls.count(model) <= failures
                if fails:
                    payload = b'{"error":{"message":"temporary provider failure"}}'
                    self.send_response(status)
                    self.send_header("content-type", "application/json")
                    if status == 429:
                        self.send_header("retry-after", "1")
                else:
                    payload = (
                        b'data: {"choices":[{"index":0,"delta":{"content":"answer"},'
                        b'"finish_reason":"stop"}]}\n\n'
                        b'data: {"choices":[],"usage":{"prompt_tokens":12,'
                        b'"completion_tokens":3}}\n\n'
                        b"data: [DONE]\n\n"
                    )
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                """Keep test server diagnostics out of process output."""
                del format, args

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        """Release the server and its thread."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@contextmanager
def serving(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: int = 500,
    failures: int = 100,
    health_threshold: int = 2,
) -> Iterator[tuple[_ServedGateway, str, ScriptedProvider, Path]]:
    """Serve one isolated two-rung native gateway with a counted upstream."""
    provider = ScriptedProvider(status, failures)
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-provider-key")
    origin = f"http://127.0.0.1:{provider.server.server_port}/v1"
    manager, key = _configured_pool_gateway(root, base_urls=(origin, origin))
    gateway = _ServedGateway(root, _unused_port())
    gateway.control_plane._accounting._health = DeploymentHealthRegistry(
        failure_threshold=health_threshold
    )  # noqa: SLF001
    try:
        gateway.start()
        yield gateway, key, provider, manager.database_path
    finally:
        gateway.stop()
        provider.close()


def sdk_call(base: str, key: str, surface: str, stream: bool, policy: JsonObject) -> None:
    """Consume one SDK operation with SDK retries explicitly disabled."""
    if surface == "messages":
        with anthropic.Anthropic(base_url=base, api_key=key, max_retries=0) as client:
            if stream:
                with client.messages.stream(
                    model="coding",
                    max_tokens=32,
                    messages=[{"role": "user", "content": "hi"}],
                    extra_body={"gateway": policy},
                ) as response:
                    response.get_final_message()
            else:
                client.messages.create(
                    model="coding",
                    max_tokens=32,
                    messages=[{"role": "user", "content": "hi"}],
                    extra_body={"gateway": policy},
                )
        return
    with openai.OpenAI(base_url=base + "/v1", api_key=key, max_retries=0) as client:
        if surface == "chat":
            response = client.chat.completions.create(
                model="coding",
                messages=[{"role": "user", "content": "hi"}],
                stream=stream,
                extra_body={"gateway": policy},
            )
        else:
            response = client.responses.create(
                model="coding", input="hi", stream=stream, extra_body={"gateway": policy}
            )
        if isinstance(response, openai.Stream):
            with response:
                list(response)


@pytest.mark.parametrize("surface", ["chat", "responses", "messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status", [500, 408, 429])
def test_single_attempt_is_one_physical_http_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str, stream: bool, status: int
) -> None:
    """Every protocol honors total1 even for SDK-retryable provider failures."""
    with serving(tmp_path, monkeypatch, status=status) as (gateway, key, provider, database):
        with pytest.raises((openai.APIError, anthropic.APIError)):
            sdk_call(
                f"http://127.0.0.1:{gateway.port}",
                key,
                surface,
                stream,
                {"retry": {"max_total_attempts": 1}},
            )
        assert provider.calls == ["alpha-model-exact"]
        assert "gateway" not in provider.bodies[0]
        with sqlite3.connect(database) as connection:
            rows = connection.execute("select state from gateway_attempts").fetchall()
        assert rows == [("failed",)]


@pytest.mark.parametrize("cap,expected", [(1, 1), (2, 2), (4, 4)])
def test_per_route_cap_includes_initial_call_and_disables_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap: int, expected: int
) -> None:
    """A caller can raise ordinary retries to four but never cross its route."""
    with serving(tmp_path, monkeypatch, health_threshold=10) as (gateway, key, provider, _database):
        with pytest.raises(openai.APIError):
            sdk_call(
                f"http://127.0.0.1:{gateway.port}",
                key,
                "chat",
                False,
                {"retry": {"max_attempts_per_route": cap}, "routing": {"allow_fallbacks": False}},
            )
        assert provider.calls == ["alpha-model-exact"] * expected


def test_backoff_waits_and_fallback_remains_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exponential ordinary retry waits, then a failed lead may fall back."""
    with serving(tmp_path, monkeypatch) as (gateway, key, provider, _database):
        sdk_call(
            f"http://127.0.0.1:{gateway.port}",
            key,
            "responses",
            False,
            {
                "retry": {
                    "max_attempts_per_route": 2,
                    "backoff": {"type": "exponential", "base_delay_ms": 200, "max_delay_ms": 200},
                }
            },
        )
        assert provider.calls == ["alpha-model-exact", "alpha-model-exact", "beta-model-exact"]
        assert provider.times[1] - provider.times[0] >= 0.09


@pytest.mark.parametrize("cap_key", ["max_total_attempts", "max_attempts_per_route"])
def test_tool_search_cannot_dispatch_a_second_model_turn_past_explicit_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap_key: str
) -> None:
    """The search-only first turn is billed but never misreported as a final answer."""
    provider = _Provider()
    monkeypatch.setenv("TEST_PROVIDER_KEY", "synthetic-provider-key")
    key = _configure(tmp_path, f"http://127.0.0.1:{provider.server.server_port}/v1")
    gateway = _ServedGateway(tmp_path, _unused_port())
    try:
        gateway.start()
        response = httpx.post(
            f"http://127.0.0.1:{gateway.port}/v1/chat/completions",
            headers={"authorization": f"Bearer {key}"},
            json={
                "model": "coding",
                "messages": [{"role": "user", "content": "weather"}],
                "tools": _TOOLS_CHAT,
                "gateway": {"retry": {cap_key: 1}},
            },
            timeout=10,
        )
        assert response.status_code == 400
        assert "gateway.retry" in response.text
        assert len(provider.requests) == 1
    finally:
        gateway.stop()
        provider.close()


def test_websocket_close_during_backoff_cancels_before_another_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A closed socket must not spend its remaining retry budget in the background."""
    with serving(tmp_path, monkeypatch, health_threshold=10) as (gateway, key, provider, database):
        with connect(
            f"ws://127.0.0.1:{gateway.port}/v1/responses",
            additional_headers={"authorization": f"Bearer {key}"},
        ) as socket:
            socket.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "model": "coding",
                        "input": "hi",
                        "gateway": {
                            "retry": {
                                "backoff": {
                                    "type": "exponential",
                                    "base_delay_ms": 2000,
                                    "max_delay_ms": 2000,
                                }
                            }
                        },
                    }
                )
            )
            deadline = time.monotonic() + 5
            while not provider.calls:
                assert time.monotonic() < deadline
                time.sleep(0.01)
        time.sleep(2.1)
        assert provider.calls == ["alpha-model-exact"]
        with sqlite3.connect(database) as connection:
            assert connection.execute("select count(*) from gateway_attempts").fetchone() == (1,)


@pytest.mark.parametrize("status", [500, 429])
def test_per_route_one_preserves_eligible_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    """A route cap limits redials, not the independent authorized fallback."""
    with serving(tmp_path, monkeypatch, status=status) as (gateway, key, provider, _database):
        sdk_call(
            f"http://127.0.0.1:{gateway.port}",
            key,
            "chat",
            False,
            {"retry": {"max_attempts_per_route": 1}},
        )
        assert provider.calls == ["alpha-model-exact", "beta-model-exact"]


def test_keyed_replay_refuses_changed_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing controls under the same key never replays another policy's response."""
    with serving(tmp_path, monkeypatch, failures=0) as (gateway, key, provider, _database):
        with openai.OpenAI(
            base_url=f"http://127.0.0.1:{gateway.port}/v1", api_key=key, max_retries=0
        ) as client:
            client.chat.completions.create(
                model="coding",
                messages=[{"role": "user", "content": "hi"}],
                extra_headers={"Idempotency-Key": "policy-key"},
                extra_body={"gateway": {"retry": {"max_total_attempts": 1}}},
            )
            with pytest.raises(openai.ConflictError):
                client.chat.completions.create(
                    model="coding",
                    messages=[{"role": "user", "content": "hi"}],
                    extra_headers={"Idempotency-Key": "policy-key"},
                    extra_body={"gateway": {"retry": {"max_total_attempts": 2}}},
                )
        assert len(provider.calls) == 1


def test_websocket_generation_honors_cap_and_prewarm_rejects_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generating WS requests share policy admission; no-generation frames reject it."""
    with serving(tmp_path, monkeypatch) as (gateway, key, provider, _database):
        with connect(
            f"ws://127.0.0.1:{gateway.port}/v1/responses",
            additional_headers={"authorization": f"Bearer {key}"},
        ) as socket:
            socket.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "generate": False,
                        "model": "coding",
                        "input": "hi",
                        "gateway": {},
                    }
                )
            )
            error = json.loads(socket.recv(timeout=5))
            assert error["error"]["param"] == "gateway"
            assert not provider.calls
            socket.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "model": "coding",
                        "input": "hi",
                        "gateway": {"retry": {"max_total_attempts": 1}},
                    }
                )
            )
            error = json.loads(socket.recv(timeout=5))
            assert error["type"] == "error"
            assert provider.calls == ["alpha-model-exact"]


def test_explicit_four_does_not_override_the_health_breaker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default circuit can terminate before a caller's larger retry ceiling."""
    with serving(tmp_path, monkeypatch) as (gateway, key, provider, _database):
        with pytest.raises(openai.APIError):
            sdk_call(
                f"http://127.0.0.1:{gateway.port}",
                key,
                "chat",
                False,
                {"retry": {"max_attempts_per_route": 4}, "routing": {"allow_fallbacks": False}},
            )
        assert len(provider.calls) == 2
