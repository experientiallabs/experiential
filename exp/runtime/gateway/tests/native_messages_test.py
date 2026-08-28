"""End-to-end Anthropic Messages tests against the served native engine.

One shared native serving subprocess (the same driver pattern as
``native_engine_disconnect_test``) serves a seeded root whose ``coding``
alias points at a local OpenAI-compatible SSE mock upstream. The tests drive
``POST /v1/messages`` with Anthropic-shaped requests through the real Rust
data plane and shared python control plane.

The Anthropic passthrough upstream dialect is deliberately not driven here:
``anthropic`` is a fixed-origin provider whose connection config rejects a
custom ``base_url``, so it cannot be pointed at a loopback mock without
weakening that production invariant.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelCapabilities
from exp.runtime.gateway.lifecycle_test import _configured_gateway

pytest.importorskip("exp_gateway_native")

_HOST = "127.0.0.1"
_REQUEST_TIMEOUT_SECONDS = 30.0

_DRIVER_SOURCE = textwrap.dedent(
    '''
    """Serve the native gateway engine over one seeded root until SIGTERM."""

    import json
    import os
    import socket
    import sys
    from pathlib import Path

    from exp.runtime.gateway.lifecycle import load_gateway_components
    from exp.runtime.gateway.native_bridge import NativeControlPlane

    import exp_gateway_native


    def main() -> None:
        """Compose the control plane, announce the public port, and serve."""
        config = json.loads(sys.argv[1])
        if "openai_base_url" in config:
            import exp.runtime.models.registry as model_registry

            model_registry.OPENAI_BASE_URL = config["openai_base_url"]
        environment = {"TEST_PROVIDER_KEY": os.environ["TEST_PROVIDER_KEY"]}
        if "OPENAI_API_KEY" in os.environ:
            environment["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
        components = load_gateway_components(
            Path(config["root"]),
            environment=environment,
        )
        control_plane = NativeControlPlane(
            components,
            request_timeout_seconds=config["request_timeout_seconds"],
        )
        last_error = None
        for _attempt in range(5):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            sys.stdout.write(json.dumps({"port": port}) + "\\n")
            sys.stdout.flush()
            try:
                exp_gateway_native.serve(
                    control_plane,
                    json.dumps(
                        {
                            "host": "127.0.0.1",
                            "port": port,
                            "max_active_requests": 8,
                            "request_timeout_seconds": config["request_timeout_seconds"],
                            "graceful_timeout_seconds": 2.0,
                        }
                    ),
                )
                return
            except RuntimeError as error:
                if "failed to bind" not in str(error):
                    raise
                last_error = error
        raise SystemExit(f"no loopback port could be bound: {last_error}")


    if __name__ == "__main__":
        main()
    '''
).strip()


def _sse_frame(payload: object) -> bytes:
    """Encode one provider SSE data frame."""
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


def _content_chunk(text: str) -> bytes:
    """Encode one OpenAI-compatible streamed content delta."""
    return _sse_frame(
        {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
    )


def _terminal_frames(finish_reason: str) -> bytes:
    """Encode the finishing chunk, usage chunk, and done sentinel."""
    return b"".join(
        (
            _sse_frame({"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}),
            _sse_frame(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 9,
                        "completion_tokens": 4,
                        "prompt_tokens_details": {"cached_tokens": 2},
                    },
                }
            ),
            b"data: [DONE]\n\n",
        )
    )


def _zero_output_terminal_frames(finish_reason: str) -> bytes:
    """Encode a terminal with no content deltas: finish, real usage, done sentinel."""
    return b"".join(
        (
            _sse_frame({"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}),
            _sse_frame(
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 9,
                        "completion_tokens": 0,
                        "prompt_tokens_details": {"cached_tokens": 2},
                    },
                }
            ),
            b"data: [DONE]\n\n",
        )
    )


class _SseUpstream(BaseHTTPRequestHandler):
    """OpenAI-compatible SSE mock whose shape is selected by the prompt."""

    payloads: list[JsonObject] = []
    payloads_lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Stream one canned SSE response selected by the request prompt."""
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        with self.payloads_lock:
            self.payloads.append(payload)
        prompt = payload["messages"][-1]["content"]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        try:
            if prompt == "tool-token":
                self.wfile.write(_content_chunk("calling "))
                self.wfile.write(
                    _sse_frame(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "call-1",
                                                "type": "function",
                                                "function": {"name": "search", "arguments": ""},
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ]
                        }
                    )
                )
                self.wfile.write(
                    _sse_frame(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "function": {"arguments": '{"q":"x"}'},
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ]
                        }
                    )
                )
                self.wfile.write(_terminal_frames("tool_calls"))
            elif prompt == "empty-token":
                self.wfile.write(_zero_output_terminal_frames("stop"))
            elif prompt == "truncated-token":
                self.wfile.write(_zero_output_terminal_frames("length"))
            else:
                self.wfile.write(_content_chunk("hello "))
                self.wfile.write(_content_chunk("world"))
                self.wfile.write(_terminal_frames("stop"))
            self.wfile.flush()
        except OSError:
            return

    def log_message(self, format: str, *args: object) -> None:
        """Suppress request logs so test output cannot retain payload context."""
        del format, args


class _ResponsesUpstream(BaseHTTPRequestHandler):
    """Native Responses SSE mock that issues one opaque tool continuation."""

    payloads: list[JsonObject] = []
    payloads_lock = threading.Lock()
    raw_arguments = '{ "query" : "λ" }'
    encrypted_content = "provider-opaque-state"

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Return a tool turn first and visible text after its function output."""
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        with self.payloads_lock:
            self.payloads.append(payload)
        input_items = payload.get("input", [])
        continued = any(
            isinstance(item, dict) and item.get("type") == "function_call_output"
            for item in input_items
        )
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        try:
            if continued:
                self.wfile.write(
                    _sse_frame(
                        {
                            "type": "response.output_text.delta",
                            "output_index": 0,
                            "item_id": "msg_provider_continued",
                            "content_index": 0,
                            "delta": "continued-ok",
                        }
                    )
                )
            else:
                self.wfile.write(
                    _sse_frame(
                        {
                            "type": "response.output_item.done",
                            "output_index": 0,
                            "item": {
                                "id": "rs_provider",
                                "type": "reasoning",
                                "summary": [],
                                "encrypted_content": self.encrypted_content,
                                "status": "completed",
                            },
                        }
                    )
                )
                tool = {
                    "id": "fc_provider",
                    "type": "function_call",
                    "call_id": "call-one",
                    "name": "lookup",
                    "arguments": self.raw_arguments,
                }
                self.wfile.write(
                    _sse_frame(
                        {
                            "type": "response.output_item.added",
                            "output_index": 1,
                            "item": {**tool, "status": "in_progress"},
                        }
                    )
                )
                self.wfile.write(
                    _sse_frame(
                        {
                            "type": "response.output_item.done",
                            "output_index": 1,
                            "item": {**tool, "status": "completed"},
                        }
                    )
                )
                self.wfile.write(
                    _sse_frame(
                        {
                            "type": "response.output_item.done",
                            "output_index": 2,
                            "item": {
                                "id": "rs_provider_2",
                                "type": "reasoning",
                                "summary": [],
                                "encrypted_content": "provider-opaque-state-2",
                                "status": "completed",
                            },
                        }
                    )
                )
            self.wfile.write(
                _sse_frame(
                    {
                        "type": "response.completed",
                        "response": {
                            "status": "completed",
                            "usage": {
                                "input_tokens": 9,
                                "output_tokens": 4,
                                "input_tokens_details": {"cached_tokens": 0},
                                "output_tokens_details": {"reasoning_tokens": 2},
                            },
                        },
                    }
                )
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError:
            return

    def log_message(self, format: str, *args: object) -> None:
        """Suppress request logs so test output cannot retain payload context."""
        del format, args


@dataclass(frozen=True)
class _ServingEngine:
    """One live native serving subprocess and its access facts."""

    port: int
    raw_key: str
    root: Path

    @property
    def base(self) -> str:
        """Return the public gateway origin."""
        return f"http://{_HOST}:{self.port}"


def _messages_body(prompt: str, *, stream: bool = False, tools: bool = False) -> JsonObject:
    """Return one Anthropic Messages body targeting the seeded alias."""
    payload: JsonObject = {
        "model": "coding",
        "max_tokens": 64,
        "system": "be terse",
        "messages": [{"role": "user", "content": prompt}],
    }
    if stream:
        payload["stream"] = True
    if tools:
        payload["tools"] = [
            {"name": "search", "description": "look up", "input_schema": {"type": "object"}}
        ]
    return payload


@pytest.fixture(scope="module", name="engine")
def _engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_ServingEngine]:
    """Serve one shared native engine subprocess over a seeded root.

    Yields:
        The live serving facts as a :class:`_ServingEngine`.
    """
    root = tmp_path_factory.mktemp("native-messages-root")
    with _SseUpstream.payloads_lock:
        _SseUpstream.payloads.clear()
    upstream = ThreadingHTTPServer((_HOST, 0), _SseUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    _manager, raw_key = _configured_gateway(
        root,
        base_url=f"http://{_HOST}:{upstream.server_address[1]}/v1",
        capabilities=ModelCapabilities(
            chat_max_tokens_field="max_completion_tokens",
            maximum_output_tokens=128,
            maximum_temperature=1.0,
        ),
    )
    driver = root / "native_messages_driver.py"
    driver.write_text(_DRIVER_SOURCE + "\n")
    config = json.dumps(
        {
            "root": str(root),
            "request_timeout_seconds": _REQUEST_TIMEOUT_SECONDS,
        }
    )
    stderr_log = root / "driver-stderr.log"
    environment = dict(os.environ)
    environment["TEST_PROVIDER_KEY"] = "provider-secret-canary"
    stderr_sink = stderr_log.open("wb")
    process = subprocess.Popen(  # noqa: S603 - the interpreter runs our generated driver.
        [sys.executable, str(driver), config],
        stdout=subprocess.PIPE,
        stderr=stderr_sink,
        env=environment,
        text=True,
    )
    try:
        announced_ports: list[int] = []

        def _collect_announcements() -> None:
            """Record every port announcement the driver prints on stdout."""
            assert process.stdout is not None
            for line in process.stdout:
                announced_ports.append(int(json.loads(line)["port"]))

        reader = threading.Thread(target=_collect_announcements, daemon=True)
        reader.start()
        live_deadline = time.monotonic() + 30
        port = 0
        while True:
            if announced_ports:
                port = announced_ports[-1]
                try:
                    live = httpx.get(f"http://{_HOST}:{port}/health/live", timeout=1.0)
                    if live.status_code == 200 and live.json() == {"status": "live"}:
                        models = httpx.get(
                            f"http://{_HOST}:{port}/v1/models",
                            headers={"authorization": f"Bearer {raw_key}"},
                            timeout=2.0,
                        )
                        if models.status_code == 200 and [
                            item["id"] for item in models.json()["data"]
                        ] == ["coding"]:
                            break
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    pass
            assert process.poll() is None, f"driver died: {stderr_log.read_text()}"
            assert time.monotonic() < live_deadline, "native engine never became live"
            time.sleep(0.05)
        yield _ServingEngine(port=port, raw_key=raw_key, root=root)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        exit_code = process.wait(timeout=20)
        stderr_sink.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)
        assert exit_code == 0, f"driver exited {exit_code}: {stderr_log.read_text()}"


@pytest.fixture(scope="module", name="responses_engine")
def _responses_engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_ServingEngine]:
    """Serve a native OpenAI Responses route against a deterministic loopback provider."""
    from exp.common.models import GatewayDeploymentCapabilities, GatewayTokenPrices
    from exp.runtime.gateway.catalog_authority import (
        ConnectionConfig,
        upsert_connection,
        upsert_singleton_deployment,
    )

    root = tmp_path_factory.mktemp("native-responses-root")
    with _ResponsesUpstream.payloads_lock:
        _ResponsesUpstream.payloads.clear()
    upstream = ThreadingHTTPServer((_HOST, 0), _ResponsesUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    manager, raw_key = _configured_gateway(
        root,
        base_url=f"http://{_HOST}:{upstream.server_address[1]}/compatible/v1",
    )
    upsert_connection(
        root,
        name="openai-responses-test",
        connection=ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY"),
        replace=False,
    )
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias="responses",
        connection_name="openai-responses-test",
        provider_model="gpt-5.6-sol",
        exact_model_id="responses-test-revision",
        revision=None,
        capabilities=ModelCapabilities(
            supports_reasoning=True,
            supports_tools=True,
            supports_temperature=False,
        ),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="responses",
        alias_name="responses",
        revision_id="revision-responses",
        pool_id="responses",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="responses")
    driver = root / "native_responses_driver.py"
    driver.write_text(_DRIVER_SOURCE + "\n")
    config = json.dumps(
        {
            "root": str(root),
            "request_timeout_seconds": _REQUEST_TIMEOUT_SECONDS,
            "openai_base_url": f"http://{_HOST}:{upstream.server_address[1]}/v1",
        }
    )
    stderr_log = root / "driver-stderr.log"
    environment = dict(os.environ)
    environment.update(
        {
            "TEST_PROVIDER_KEY": "provider-secret-canary",
            "OPENAI_API_KEY": "openai-test-key",
        }
    )
    stderr_sink = stderr_log.open("wb")
    process = subprocess.Popen(  # noqa: S603 - the interpreter runs our generated driver.
        [sys.executable, str(driver), config],
        stdout=subprocess.PIPE,
        stderr=stderr_sink,
        env=environment,
        text=True,
    )
    try:
        announced_ports: list[int] = []

        def _collect_announcements() -> None:
            """Record every port announcement the driver prints on stdout."""
            assert process.stdout is not None
            for line in process.stdout:
                announced_ports.append(int(json.loads(line)["port"]))

        reader = threading.Thread(target=_collect_announcements, daemon=True)
        reader.start()
        live_deadline = time.monotonic() + 30
        port = 0
        while True:
            if announced_ports:
                port = announced_ports[-1]
                try:
                    live = httpx.get(f"http://{_HOST}:{port}/health/live", timeout=1.0)
                    if live.status_code == 200:
                        models = httpx.get(
                            f"http://{_HOST}:{port}/v1/models",
                            headers={"authorization": f"Bearer {raw_key}"},
                            timeout=2.0,
                        )
                        if models.status_code == 200 and {
                            item["id"] for item in models.json()["data"]
                        } == {"coding", "responses"}:
                            break
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    pass
            assert process.poll() is None, f"driver died: {stderr_log.read_text()}"
            assert time.monotonic() < live_deadline, "native Responses engine never became live"
            time.sleep(0.05)
        yield _ServingEngine(port=port, raw_key=raw_key, root=root)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        exit_code = process.wait(timeout=20)
        stderr_sink.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=5)
        assert exit_code == 0, f"driver exited {exit_code}: {stderr_log.read_text()}"


def _normalized(body: JsonObject) -> JsonObject:
    """Return one Anthropic message with its request-derived identity removed."""
    normalized = dict(body)
    identity = normalized.pop("id")
    assert isinstance(identity, str) and identity.startswith("msg_")
    return normalized


def _completed_attempts(base: str) -> int:
    """Read the completed-attempt total from the live usage report."""
    report = httpx.get(f"{base}/usage.json", timeout=5.0).json()
    for count in report["totals"]["terminal_counts"]:
        if count["state"] == "completed":
            return int(count["attempts"])
    return 0


def test_non_streaming_message_answers_the_anthropic_shape_and_accounts(
    engine: _ServingEngine,
) -> None:
    """A non-streaming request returns one Anthropic message and settles."""
    completed_before = _completed_attempts(engine.base)
    response = httpx.post(
        f"{engine.base}/v1/messages",
        headers={"x-api-key": engine.raw_key, "anthropic-version": "2023-06-01"},
        json=_messages_body("fast-token"),
        timeout=30.0,
    )
    assert response.status_code == 200
    assert response.headers["x-request-id"]
    assert response.headers["x-gateway-alias"] == "coding"
    assert _normalized(response.json()) == {
        "type": "message",
        "role": "assistant",
        "model": "coding",
        "content": [{"type": "text", "text": "hello world"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 7, "output_tokens": 4, "cache_read_input_tokens": 2},
    }
    assert _completed_attempts(engine.base) == completed_before + 1


def test_native_admission_uses_the_catalog_token_field_on_the_provider_wire(
    engine: _ServingEngine,
) -> None:
    """Rust dispatches the Python-frozen payload with the exact model token alias."""
    payload = _messages_body("token-field")
    payload["max_tokens"] = 63
    with _SseUpstream.payloads_lock:
        before = len(_SseUpstream.payloads)

    response = httpx.post(
        f"{engine.base}/v1/messages",
        headers={"x-api-key": engine.raw_key},
        json=payload,
        timeout=30.0,
    )

    assert response.status_code == 200
    with _SseUpstream.payloads_lock:
        dispatched = _SseUpstream.payloads[before:]
    assert len(dispatched) == 1
    assert dispatched[0]["max_completion_tokens"] == 63
    assert "max_tokens" not in dispatched[0]


def test_streaming_message_emits_the_full_anthropic_lifecycle(
    engine: _ServingEngine,
) -> None:
    """A streaming request emits the ordered Anthropic SSE lifecycle."""
    with httpx.stream(
        "POST",
        f"{engine.base}/v1/messages",
        headers={"x-api-key": engine.raw_key},
        json=_messages_body("fast-token", stream=True),
        timeout=30.0,
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        raw = b"".join(response.iter_bytes()).decode()
    names = [
        line.removeprefix("event: ") for line in raw.splitlines() if line.startswith("event: ")
    ]
    assert names == [
        "message_start",
        "ping",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in raw.splitlines()
        if line.startswith("data: ")
    ]
    text = "".join(
        payload["delta"]["text"] for payload in payloads if payload["type"] == "content_block_delta"
    )
    assert text == "hello world"
    message_delta = next(payload for payload in payloads if payload["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "end_turn"
    assert message_delta["usage"] == {
        "input_tokens": 7,
        "output_tokens": 4,
        "cache_read_input_tokens": 2,
    }


def test_tool_calls_translate_to_tool_use_blocks(engine: _ServingEngine) -> None:
    """Upstream tool calls become Anthropic tool_use blocks and stop_reason."""
    response = httpx.post(
        f"{engine.base}/v1/messages",
        headers={"x-api-key": engine.raw_key},
        json=_messages_body("tool-token", tools=True),
        timeout=30.0,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"][0] == {"type": "text", "text": "calling "}
    assert body["content"][1] == {
        "type": "tool_use",
        "id": "call-1",
        "name": "search",
        "input": {"q": "x"},
    }


def test_protocol_and_key_failures_are_anthropic_shaped(engine: _ServingEngine) -> None:
    """Bad keys, unknown fields, and count_tokens answer Anthropic envelopes."""
    bad_key = httpx.post(
        f"{engine.base}/v1/messages",
        headers={"x-api-key": "exp_vk_invalid"},
        json=_messages_body("fast-token"),
        timeout=10.0,
    )
    assert bad_key.status_code == 401
    assert bad_key.json()["type"] == "error"
    assert bad_key.json()["error"]["type"] == "authentication_error"

    missing_key = httpx.post(
        f"{engine.base}/v1/messages", json=_messages_body("fast-token"), timeout=10.0
    )
    assert missing_key.status_code == 401
    assert "x-api-key" in missing_key.json()["error"]["message"]

    unknown_field = httpx.post(
        f"{engine.base}/v1/messages",
        headers={"x-api-key": engine.raw_key},
        json={**_messages_body("fast-token"), "unknown_field": 3},
        timeout=10.0,
    )
    assert unknown_field.status_code == 400
    assert unknown_field.json()["error"]["type"] == "invalid_request_error"
    assert "unknown_field" in unknown_field.json()["error"]["message"]

    count_tokens = httpx.post(
        f"{engine.base}/v1/messages/count_tokens",
        headers={"x-api-key": engine.raw_key},
        json={},
        timeout=10.0,
    )
    assert count_tokens.status_code == 404
    assert count_tokens.json()["error"]["type"] == "not_found_error"


def test_native_rejects_unsupported_reasoning_effort_with_the_public_field_error(
    engine: _ServingEngine,
) -> None:
    """The native plane returns the local field error before any provider dispatch."""
    payload = {
        "model": "coding",
        "input": "hello",
        "reasoning": {"effort": "high"},
    }
    headers = {"authorization": f"Bearer {engine.raw_key}"}
    native = httpx.post(
        f"{engine.base}/v1/responses",
        headers=headers,
        json=payload,
        timeout=10.0,
    )

    expected = {
        "error": {
            "message": (
                "The parameter 'reasoning.effort' is not supported by this model route. "
                "Remove the field or choose a different model."
            ),
            "type": "invalid_request_error",
            "param": "reasoning.effort",
            "code": "unsupported_parameter",
        }
    }
    assert native.status_code == 400
    assert native.json() == expected


def test_native_rejects_unsupported_sampling_with_the_public_field_error(
    engine: _ServingEngine,
) -> None:
    """A route without top-k support fails the request locally with the field name."""
    payload = {**_messages_body("fast-token"), "top_k": 3}
    headers = {"x-api-key": engine.raw_key}
    native = httpx.post(
        f"{engine.base}/v1/messages",
        headers=headers,
        json=payload,
        timeout=10.0,
    )

    assert native.status_code == 400
    assert "top_k" in native.json()["error"]["message"]


@pytest.mark.parametrize(
    ("field", "value"),
    (("temperature", 1.1), ("max_tokens", 129)),
)
def test_native_enforces_catalog_generation_limits_before_dispatch(
    engine: _ServingEngine,
    field: str,
    value: float | int,
) -> None:
    """Model-specific sampling and output ceilings fail locally, with no upstream dispatch."""
    payload = _messages_body("must-not-dispatch")
    payload[field] = value
    headers = {"x-api-key": engine.raw_key}
    with _SseUpstream.payloads_lock:
        dispatched_before = len(_SseUpstream.payloads)

    native = httpx.post(
        f"{engine.base}/v1/messages",
        headers=headers,
        json=payload,
        timeout=10.0,
    )

    assert native.status_code == 400
    assert field in native.json()["error"]["message"]
    with _SseUpstream.payloads_lock:
        assert len(_SseUpstream.payloads) == dispatched_before


_ZERO_OUTPUT_CHAT_CASES = (
    pytest.param("empty-token", "stop", id="empty-completion"),
    pytest.param("truncated-token", "length", id="max-tokens-truncation"),
)
_ZERO_OUTPUT_RESPONSES_CASES = (
    pytest.param("empty-token", "completed", id="empty-completion"),
    pytest.param("truncated-token", "incomplete", id="max-tokens-truncation"),
)
_ZERO_OUTPUT_MESSAGES_CASES = (
    pytest.param("empty-token", "end_turn", id="empty-completion"),
    pytest.param("truncated-token", "max_tokens", id="max-tokens-truncation"),
)


@pytest.mark.parametrize(("prompt", "finish_reason"), _ZERO_OUTPUT_CHAT_CASES)
def test_chat_non_stream_zero_output_keeps_client_visible_usage(
    engine: _ServingEngine,
    prompt: str,
    finish_reason: str,
) -> None:
    """A successful zero-output completion still returns the real token usage."""
    response = httpx.post(
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={"model": "coding", "messages": [{"role": "user", "content": prompt}]},
        timeout=30.0,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["finish_reason"] == finish_reason
    usage = body["usage"]
    assert usage is not None, "zero-output completion dropped client-visible usage"
    assert usage["prompt_tokens"] == 9
    assert usage["completion_tokens"] == 0
    assert usage["total_tokens"] == 9
    assert usage["prompt_tokens_details"]["cached_tokens"] == 2


@pytest.mark.parametrize(("prompt", "finish_reason"), _ZERO_OUTPUT_CHAT_CASES)
def test_chat_stream_zero_output_emits_the_include_usage_chunk(
    engine: _ServingEngine,
    prompt: str,
    finish_reason: str,
) -> None:
    """A zero-output stream still ends with the requested usage chunk."""
    with httpx.stream(
        "POST",
        f"{engine.base}/v1/chat/completions",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={
            "model": "coding",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        timeout=30.0,
    ) as response:
        assert response.status_code == 200
        raw = b"".join(response.iter_bytes()).decode()
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in raw.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    finish_reasons = [
        choice["finish_reason"]
        for payload in payloads
        for choice in payload.get("choices", ())
        if choice.get("finish_reason")
    ]
    assert finish_reasons == [finish_reason]
    usage_chunks = [payload["usage"] for payload in payloads if payload.get("usage")]
    assert usage_chunks, "zero-output stream never emitted the include_usage chunk"
    assert usage_chunks[-1]["prompt_tokens"] == 9
    assert usage_chunks[-1]["completion_tokens"] == 0


@pytest.mark.parametrize(("prompt", "status"), _ZERO_OUTPUT_RESPONSES_CASES)
def test_responses_non_stream_zero_output_keeps_client_visible_usage(
    engine: _ServingEngine,
    prompt: str,
    status: str,
) -> None:
    """A zero-output Responses result still carries the real token usage."""
    response = httpx.post(
        f"{engine.base}/v1/responses",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={"model": "coding", "input": prompt},
        timeout=30.0,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == status
    usage = body["usage"]
    assert usage is not None, "zero-output response dropped client-visible usage"
    assert usage["input_tokens"] == 9
    assert usage["output_tokens"] == 0
    assert usage["input_tokens_details"]["cached_tokens"] == 2


@pytest.mark.parametrize(("prompt", "status"), _ZERO_OUTPUT_RESPONSES_CASES)
def test_responses_stream_zero_output_keeps_terminal_usage(
    engine: _ServingEngine,
    prompt: str,
    status: str,
) -> None:
    """A zero-output Responses stream still reports usage on its terminal event."""
    with httpx.stream(
        "POST",
        f"{engine.base}/v1/responses",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={"model": "coding", "input": prompt, "stream": True},
        timeout=30.0,
    ) as response:
        assert response.status_code == 200
        raw = b"".join(response.iter_bytes()).decode()
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in raw.splitlines()
        if line.startswith("data: ")
    ]
    terminal = next(payload for payload in payloads if payload["type"] == f"response.{status}")
    usage = terminal["response"]["usage"]
    assert usage is not None, "zero-output stream terminal dropped client-visible usage"
    assert usage["input_tokens"] == 9
    assert usage["output_tokens"] == 0


@pytest.mark.parametrize(("prompt", "stop_reason"), _ZERO_OUTPUT_MESSAGES_CASES)
def test_messages_non_stream_zero_output_keeps_real_input_tokens(
    engine: _ServingEngine,
    prompt: str,
    stop_reason: str,
) -> None:
    """A zero-output Messages result reports the real input count, never zero."""
    response = httpx.post(
        f"{engine.base}/v1/messages",
        headers={"x-api-key": engine.raw_key},
        json=_messages_body(prompt),
        timeout=30.0,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == stop_reason
    assert body["usage"] == {
        "input_tokens": 7,
        "output_tokens": 0,
        "cache_read_input_tokens": 2,
    }


@pytest.mark.parametrize(("prompt", "stop_reason"), _ZERO_OUTPUT_MESSAGES_CASES)
def test_messages_stream_zero_output_keeps_real_input_tokens(
    engine: _ServingEngine,
    prompt: str,
    stop_reason: str,
) -> None:
    """A zero-output Messages stream reports the real input count on its final delta."""
    with httpx.stream(
        "POST",
        f"{engine.base}/v1/messages",
        headers={"x-api-key": engine.raw_key},
        json=_messages_body(prompt, stream=True),
        timeout=30.0,
    ) as response:
        assert response.status_code == 200
        raw = b"".join(response.iter_bytes()).decode()
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in raw.splitlines()
        if line.startswith("data: ")
    ]
    message_delta = next(payload for payload in payloads if payload["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == stop_reason
    assert message_delta["usage"] == {
        "input_tokens": 7,
        "output_tokens": 0,
        "cache_read_input_tokens": 2,
    }


def test_thinking_carriers_reject_non_anthropic_routes_before_dispatch(
    engine: _ServingEngine,
) -> None:
    """Thinking config and history need an Anthropic-only route; the seeded
    OpenAI-compatible route rejects both in the Anthropic envelope with no
    upstream dispatch."""
    with _SseUpstream.payloads_lock:
        dispatched_before = len(_SseUpstream.payloads)

    config = httpx.post(
        f"{engine.base}/v1/messages",
        headers={"x-api-key": engine.raw_key},
        json={
            **_messages_body("must-not-dispatch"),
            "thinking": {"type": "enabled", "budget_tokens": 2048},
        },
        timeout=10.0,
    )
    assert config.status_code == 400
    assert config.json()["error"]["type"] == "invalid_request_error"
    assert "thinking" in config.json()["error"]["message"]

    history = httpx.post(
        f"{engine.base}/v1/messages",
        headers={"x-api-key": engine.raw_key},
        json={
            **_messages_body("must-not-dispatch"),
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "private", "signature": "sig=="},
                        {"type": "text", "text": "done"},
                    ],
                },
                {"role": "user", "content": "continue"},
            ],
        },
        timeout=10.0,
    )
    assert history.status_code == 400
    assert "thinking" in history.json()["error"]["message"]

    with _SseUpstream.payloads_lock:
        assert len(_SseUpstream.payloads) == dispatched_before


def test_encrypted_reasoning_include_rejects_non_responses_routes(
    engine: _ServingEngine,
) -> None:
    """The encrypted reasoning include selector needs a native Responses route."""
    response = httpx.post(
        f"{engine.base}/v1/responses",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={
            "model": "coding",
            "input": "fast-token",
            "include": ["reasoning.encrypted_content"],
        },
        timeout=10.0,
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["param"] == "include"
    assert body["error"]["code"] == "unsupported_parameter"


def _responses_result(
    response: httpx.Response, *, stream: bool
) -> tuple[JsonObject, list[JsonObject]]:
    """Return the terminal response and every SSE payload from one public response."""
    assert response.status_code == 200, response.text
    if not stream:
        return response.json(), []
    payloads = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    terminal = next(payload for payload in payloads if payload["type"] == "response.completed")
    return terminal["response"], payloads


@pytest.mark.parametrize("stream", (False, True))
def test_native_openai_responses_retains_hidden_reasoning_for_tool_continuation(
    responses_engine: _ServingEngine,
    stream: bool,
) -> None:
    """Buffered and streaming routes replay every private tool-turn identity and byte."""
    with _ResponsesUpstream.payloads_lock:
        _ResponsesUpstream.payloads.clear()
    headers = {"authorization": f"Bearer {responses_engine.raw_key}"}
    first = httpx.post(
        f"{responses_engine.base}/v1/responses",
        headers=headers,
        json={
            "model": "responses",
            "input": "use the lookup tool",
            "stream": stream,
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {"type": "object"},
                }
            ],
        },
        timeout=30.0,
    )
    first_body, first_events = _responses_result(first, stream=stream)
    first_output = cast(list[JsonObject], first_body["output"])
    public_items = list(first_output)
    if stream:
        public_items.extend(
            cast(JsonObject, payload["item"])
            for payload in first_events
            if payload["type"] == "response.output_item.done"
        )
    reasoning_items = [item for item in public_items if item["type"] == "reasoning"]
    assert reasoning_items
    assert all("encrypted_content" not in item for item in reasoning_items)
    call = next(item for item in first_output if item["type"] == "function_call")
    assert cast(str, call["arguments"]).encode() == _ResponsesUpstream.raw_arguments.encode()

    second = httpx.post(
        f"{responses_engine.base}/v1/responses",
        headers=headers,
        json={
            "model": "responses",
            "previous_response_id": first_body["id"],
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call-one",
                    "output": "tool-result",
                }
            ],
            "stream": stream,
        },
        timeout=30.0,
    )
    second_body, _second_events = _responses_result(second, stream=stream)
    assert second_body["status"] == "completed"
    second_output = cast(list[JsonObject], second_body["output"])
    assert any(
        content.get("text") == "continued-ok"
        for item in second_output
        if item["type"] == "message"
        for content in cast(list[JsonObject], item["content"])
    )

    with _ResponsesUpstream.payloads_lock:
        upstream = tuple(_ResponsesUpstream.payloads)
    assert len(upstream) == 2
    assert upstream[0]["include"] == ["reasoning.encrypted_content"]
    replay = cast(list[JsonObject], upstream[1]["input"])
    assert replay[-4:] == [
        {
            "type": "reasoning",
            "id": "rs_provider",
            "summary": [],
            "encrypted_content": _ResponsesUpstream.encrypted_content,
        },
        {
            "type": "function_call",
            "id": "fc_provider",
            "call_id": "call-one",
            "name": "lookup",
            "arguments": _ResponsesUpstream.raw_arguments,
        },
        {
            "type": "reasoning",
            "id": "rs_provider_2",
            "summary": [],
            "encrypted_content": "provider-opaque-state-2",
        },
        {
            "type": "function_call_output",
            "call_id": "call-one",
            "output": "tool-result",
        },
    ]


def test_store_false_responses_cannot_be_continued(engine: _ServingEngine) -> None:
    """store:false answers normally but its response ID is never retained."""
    first = httpx.post(
        f"{engine.base}/v1/responses",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={"model": "coding", "input": "fast-token", "store": False},
        timeout=30.0,
    )
    assert first.status_code == 200
    body = first.json()
    assert body["status"] == "completed"

    continued = httpx.post(
        f"{engine.base}/v1/responses",
        headers={"authorization": f"Bearer {engine.raw_key}"},
        json={
            "model": "coding",
            "input": "fast-token",
            "previous_response_id": body["id"],
        },
        timeout=30.0,
    )
    assert continued.status_code == 400
    assert continued.json()["error"]["code"] == "continuation_unavailable"
