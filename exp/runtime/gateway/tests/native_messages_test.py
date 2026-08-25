"""End-to-end Anthropic Messages tests against the served native engine.

One shared native serving subprocess (the same driver pattern as
``native_engine_disconnect_test``) serves a seeded root whose ``coding``
alias points at a local OpenAI-compatible SSE mock upstream. The tests drive
``POST /v1/messages`` with Anthropic-shaped requests through the real Rust
data plane and shared python control plane, then prove the deprecated
python engine (scheduled for removal with the python data plane) serves the
identical surface over the same root.

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

import httpx
import pytest
from fastapi.testclient import TestClient

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.lifecycle import load_local_gateway
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
        components = load_gateway_components(
            Path(config["root"]),
            environment={"TEST_PROVIDER_KEY": os.environ["TEST_PROVIDER_KEY"]},
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


class _SseUpstream(BaseHTTPRequestHandler):
    """OpenAI-compatible SSE mock whose shape is selected by the prompt."""

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract.
        """Stream one canned SSE response selected by the request prompt."""
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
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
    upstream = ThreadingHTTPServer((_HOST, 0), _SseUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    _manager, raw_key = _configured_gateway(
        root,
        base_url=f"http://{_HOST}:{upstream.server_address[1]}/v1",
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
        assert response.headers["content-type"] == "text/event-stream"
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
        json={**_messages_body("fast-token"), "top_k": 3},
        timeout=10.0,
    )
    assert unknown_field.status_code == 400
    assert unknown_field.json()["error"]["type"] == "invalid_request_error"
    assert "top_k" in unknown_field.json()["error"]["message"]

    count_tokens = httpx.post(
        f"{engine.base}/v1/messages/count_tokens",
        headers={"x-api-key": engine.raw_key},
        json={},
        timeout=10.0,
    )
    assert count_tokens.status_code == 404
    assert count_tokens.json()["error"]["type"] == "not_found_error"


def test_python_engine_serves_the_identical_surface(engine: _ServingEngine) -> None:
    """The deprecated python engine answers the same bytes modulo request identity."""
    native = httpx.post(
        f"{engine.base}/v1/messages",
        headers={"x-api-key": engine.raw_key},
        json=_messages_body("fast-token"),
        timeout=30.0,
    )
    runtime = load_local_gateway(
        engine.root,
        graceful_timeout_seconds=1,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )
    with TestClient(runtime.app) as client:
        python_engine = client.post(
            "/v1/messages",
            headers={"x-api-key": engine.raw_key},
            json=_messages_body("fast-token"),
        )
        python_bad_key = client.post(
            "/v1/messages",
            headers={"x-api-key": "exp_vk_invalid"},
            json=_messages_body("fast-token"),
        )
    assert python_engine.status_code == 200
    assert _normalized(python_engine.json()) == _normalized(native.json())
    assert python_bad_key.status_code == 401
    assert python_bad_key.json()["error"]["type"] == "authentication_error"
